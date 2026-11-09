"""
AGENTE 1 - CLASSIFICADOR DE IMPACTO (mBERT)
==============================================
Secao 3.4.1 do TCC. Mantem o pipeline descrito no texto - tokenizacao,
embedding contextual do token [CLS], camada densa fine-tuned, softmax
sobre {Alto, Medio, Baixo} - e adiciona tres camadas que o escopo do
projeto exige e a versao anterior nao tinha:

  (a) VALIDACAO CRUZADA ESTRATIFICADA, no lugar de um holdout unico;
  (b) CALIBRACAO DE PROBABILIDADE por temperature scaling;
  (c) ENRIQUECIMENTO SEMANTICO: extracao de entidades de mercado e
      inferencia da DIRECAO do impacto (positivo/negativo).

--------------------------------------------------------------------
POR QUE (a), (b) E (c) - justificativa para a metodologia do TCC
--------------------------------------------------------------------

(a) VALIDACAO CRUZADA. Um holdout unico de 20% sobre um corpus de
    poucas centenas de noticias produz metrica com variancia alta: a
    troca da semente aleatoria pode mover o F1 em varios pontos. Uma
    banca pode questionar legitimamente se o resultado reportado nao
    foi apenas uma particao favoravel. K-fold estratificado (KOHAVI,
    1995) resolve isso: cada documento e avaliado exatamente uma vez,
    fora da amostra de treino, e o resultado vira media +/- desvio.
    Contra: custo computacional k vezes maior.

(b) CALIBRACAO. Redes neurais profundas modernas sao sistematicamente
    superconfiantes: o softmax de um BERT fine-tuned costuma marcar
    0,97 em predicoes que acertam ~0,80 das vezes (GUO et al., 2017).
    Isso importa aqui porque o Agente 2 usa o score como insumo de
    decisao e o painel final o mostra ao gestor. Temperature scaling
    corrige a escala com UM unico parametro aprendido, sem alterar a
    ordenacao das predicoes - ou seja, nao muda a acuracia, so torna
    a probabilidade interpretavel como confianca real.

(c) ENRIQUECIMENTO SEMANTICO. O objetivo do sistema e cruzar noticias
    com performance de vendas. Um rotulo "Alto" isolado nao possui
    chave de juncao com a base IQVIA: nao diz QUAL produto, QUAL
    laboratorio nem em QUE direcao o faturamento e afetado. A extracao
    de entidades fornece a chave de juncao, e a direcao distingue
    RISCO de OPORTUNIDADE - separacao que o Agente 4 precisa fazer e
    que, sem esta camada, ele nao teria como sustentar.
    Esta camada e opcional (parametro `enriquecer=False`) para quem
    quiser reproduzir estritamente o texto original da secao 3.4.1.

--------------------------------------------------------------------
RELATORIO DE AVALIACAO (orientacao do professor, out/2026)
--------------------------------------------------------------------
Ao fim de cada treino o agente passa a gerar, sozinho, em
`outputs/relatorio_agente1/`: acuracia e F1 por ciclo - tanto por FOLD
quanto por EPOCA -, avaliacao por CLASSE e matriz de confusao, com
tabelas (.csv) e figuras (.png) prontas para o texto. O conteudo e o
racional estao em `_relatorio_agente1.py`.

REGRA QUE NAO PODE SER QUEBRADA: toda avaliacao reportada sai das
predicoes OUT-OF-FOLD, em que cada documento foi classificado pelo
modelo que NAO o viu em treino. Avaliar o modelo final sobre o proprio
corpus de treino inflou o F1-macro deste projeto de 0,66 para 0,79 -
treze pontos de ilusao. Por isso o modo `treinar` da linha de comando
exporta para o Agente 2 a planilha out-of-fold, e nao a classificacao
do modelo final sobre o corpus ja visto.

CUSTO: R$ 0,00. mBERT (bert-base-multilingual-cased) e aberto; o
gazetteer e deterministico. Nenhuma chamada de API paga.

Requisitos:
    pip install torch transformers scikit-learn pandas openpyxl --break-system-packages
    (matplotlib e opcional: sem ele saem as tabelas, sem as figuras)

USO BASICO:
    from _schema import preparar_corpus
    from agente1_classificador_mbert import AgenteClassificadorImpacto

    df, _ = preparar_corpus(pd.read_excel("corpus.xlsx"))
    agente = AgenteClassificadorImpacto()
    relatorio = agente.treinar(df)            # k-fold + relatorio + modelo final
    agente.salvar("modelos/agente1_mbert")

    df_saida = agente.classificar_dataframe(df_novas)
"""

from __future__ import annotations

import inspect
import json
import math
import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from _schema import COL_ID, COL_LABEL, COL_TEXTO, ROTULOS_CANONICOS
from dominio_farma import extrair_entidades, inferir_direcao

MODELO_BASE = "bert-base-multilingual-cased"  # mBERT (DEVLIN et al., 2019)
SEED_PADRAO = 42
DIR_RELATORIO_PADRAO = "outputs/relatorio_agente1"


# =====================================================================
# Reprodutibilidade
# =====================================================================


def fixar_seed(seed: int = SEED_PADRAO) -> None:
    """
    Fixa todas as fontes de aleatoriedade do pipeline.

    Sem isso, duas execucoes do mesmo codigo produzem metricas
    diferentes, e o numero publicado no TCC deixa de ser verificavel
    por terceiros. `torch.manual_seed` sozinho nao basta: o shuffle do
    DataLoader e a inicializacao da camada de classificacao usam os
    geradores de `random` e `numpy`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# =====================================================================
# Estruturas de resultado
# =====================================================================


@dataclass
class ResultadoClassificacao:
    """Saida do agente para um documento."""

    texto: str
    classe_predita: str
    score: float                       # probabilidade calibrada da classe vencedora
    probabilidades: dict[str, float]
    entidades: dict = field(default_factory=dict)
    direcao: dict = field(default_factory=dict)


class _DatasetNoticias(Dataset):
    """Adaptador entre encodings do tokenizer e o Trainer do HuggingFace."""

    def __init__(self, encodings: dict, labels: list[int] | None = None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


class _TrainerPonderado(Trainer):
    """
    Trainer com entropia cruzada ponderada por frequencia inversa.

    PROBLEMA QUE RESOLVE: corpus de noticias farmaceuticas coletado por
    API e naturalmente desbalanceado - "Baixo" costuma responder por
    60-75% das linhas. Com a loss padrao, o otimo local mais facil e
    prever sempre a classe majoritaria: a acuracia parece boa (~70%)
    e o recall de "Alto" fica proximo de zero. Como "Alto" e exatamente
    a classe que o sistema existe para detectar, esse modelo seria
    inutil na pratica apesar da metrica agregada aceitavel.

    A ponderacao por frequencia inversa (KING; ZENG, 2001) faz o erro
    na classe rara custar mais, forcando o modelo a aprende-la.
    Contra: pode elevar levemente o numero de falsos positivos em
    "Alto" - trade-off aceitavel aqui, porque o Agente 2 faz uma
    segunda filtragem e o custo de perder um risco real e maior que o
    de investigar um alarme falso.
    """

    def __init__(self, *args, pesos_classe: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesos_classe = pesos_classe

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        peso = self.pesos_classe.to(logits.device) if self.pesos_classe is not None else None
        loss = F.cross_entropy(logits, labels, weight=peso)
        inputs["labels"] = labels  # restaura para nao quebrar callbacks
        return (loss, outputs) if return_outputs else loss


def _training_args_compat(**kwargs) -> TrainingArguments:
    """
    Constroi TrainingArguments tolerante a versao da biblioteca.

    A API do `transformers` renomeou `evaluation_strategy` para
    `eval_strategy` na v4.41 e removeu/adicionou parametros ao longo
    das versoes. Filtrar pelos parametros realmente aceitos evita que o
    script quebre quando voce rodar no Colab, cuja versao instalada
    muda sem aviso - problema classico de reproducao de TCC meses
    depois da escrita.
    """
    aceitos = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "eval_strategy" in kwargs and "eval_strategy" not in aceitos:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in aceitos})


def verificar_ambiente() -> None:
    """
    Confere as dependencias de treino ANTES de comecar, e falha com uma
    mensagem acionavel.

    Sem esta checagem, a ausencia do `accelerate` so aparece la dentro
    da construcao do TrainingArguments, como um ImportError enterrado a
    oito quadros de profundidade na pilha - depois de o usuario ja ter
    esperado o carregamento do modelo e a tokenizacao. A mensagem
    original nem sequer diz que o pacote faltou no requirements: sugere
    reinstalar o transformers inteiro.

    Falhar cedo e explicito custa tres linhas e economiza uma sessao de
    depuracao.
    """
    faltando: list[str] = []
    try:
        import accelerate  # noqa: F401
    except ImportError:
        faltando.append('"accelerate>=1.1.0"')

    if faltando:
        raise ImportError(
            "\n"
            + "=" * 62
            + f"\n[Agente 1] Dependencia de treino ausente: {', '.join(faltando)}\n"
            + "=" * 62
            + "\n\nRode:\n"
            + f"    pip install {' '.join(faltando)}\n\n"
            + "Ou instale tudo de uma vez, a partir da raiz do projeto:\n"
            + "    pip install -r requirements.txt\n"
        )


def _metricas(pred) -> dict:
    """Metricas por epoca. F1 macro e o criterio de selecao do melhor checkpoint."""
    y_true = pred.label_ids
    y_pred = np.argmax(pred.predictions, axis=1)
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "acuracia": float((y_true == y_pred).mean()),
    }


# =====================================================================
# Agente 1
# =====================================================================


class AgenteClassificadorImpacto:
    """
    Agente 1 - fine-tuning de mBERT para classificar o impacto de
    noticias sobre o faturamento de medicamentos para diabetes.
    """

    def __init__(
        self,
        modelo_base: str = MODELO_BASE,
        max_length: int = 256,
        seed: int = SEED_PADRAO,
        enriquecer: bool = True,
    ):
        self.modelo_base = modelo_base
        self.max_length = max_length
        self.seed = seed
        self.enriquecer = enriquecer
        self.tokenizer: AutoTokenizer | None = None
        self.model: AutoModelForSequenceClassification | None = None
        self.classes_: list[str] = []
        self.temperatura: float = 1.0
        self.relatorio_treino: dict = {}
        # Planilha de predicoes out-of-fold - a entrada honesta do Agente 2.
        self.predicoes_oof_: pd.DataFrame | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        fixar_seed(seed)

    # -----------------------------------------------------------------
    # Preparo interno
    # -----------------------------------------------------------------
    def _tokenizar(self, textos: list[str]) -> dict:
        """
        Tokeniza SEM padding fixo.

        A versao anterior usava `padding="max_length"`, o que forcava
        todo lote a 256 tokens mesmo quando a maior sequencia tinha 60.
        Combinado com o DataCollatorWithPadding (que ja faz padding
        dinamico por lote), isso era trabalho redundante: ate ~4x mais
        FLOPs por passo em corpus com textos curtos. Delegar o padding
        ao collator e a pratica recomendada e nao altera o resultado.
        """
        return self.tokenizer(textos, truncation=True, max_length=self.max_length)

    def _preparar(self, df: pd.DataFrame, coluna_texto: str, coluna_label: str):
        """Valida entrada e devolve (df_filtrado, textos, labels_numericos)."""
        for col in (coluna_texto, coluna_label):
            if col not in df.columns:
                raise KeyError(
                    f"Coluna '{col}' ausente. Rode _schema.preparar_corpus() antes "
                    f"de treinar. Colunas disponiveis: {list(df.columns)}"
                )

        antes = len(df)
        df = df.dropna(subset=[coluna_texto, coluna_label]).copy()
        df = df[df[coluna_texto].astype(str).str.strip() != ""]
        if len(df) < antes:
            print(f"[Agente 1] {antes - len(df)} linha(s) descartada(s) por texto ou rotulo ausente.")

        # Ordem canonica fixa (Alto, Medio, Baixo) em vez da ordem
        # alfabetica do LabelEncoder: garante que o indice de cada
        # classe seja o mesmo entre execucoes e entre corpus diferentes,
        # o que evita um modelo salvo ser recarregado com as classes
        # trocadas - bug silencioso e dificil de diagnosticar.
        presentes = [c for c in ROTULOS_CANONICOS if c in set(df[coluna_label])]
        extras = sorted(set(df[coluna_label]) - set(ROTULOS_CANONICOS))
        self.classes_ = presentes + extras
        if extras:
            print(f"[Agente 1] AVISO: rotulos fora do padrao encontrados: {extras}")

        idx = {c: i for i, c in enumerate(self.classes_)}
        y = df[coluna_label].map(idx).to_numpy()
        return df, df[coluna_texto].astype(str).tolist(), y

    def _pesos_classe(self, y: np.ndarray) -> torch.Tensor:
        """
        Peso inversamente proporcional a frequencia da classe.

        Usa a mesma formula de `class_weight='balanced'` do
        scikit-learn: w_c = N / (K * n_c). A propriedade util dela e que
        a soma dos pesos sobre todas as amostras continua igual a N, ou
        seja, a ponderacao redistribui importancia entre as classes sem
        alterar a escala global da loss - o learning rate efetivo
        permanece o mesmo, e o efeito medido e so o do rebalanceamento.

        O clamp em 1.0 para classes com contagem zero evita divisao por
        zero quando um fold nao sorteia nenhum exemplo de uma classe
        muito rara.
        """
        contagem = np.bincount(y, minlength=len(self.classes_)).astype(float)
        contagem[contagem == 0] = 1.0
        pesos = contagem.sum() / (len(contagem) * contagem)
        return torch.tensor(pesos, dtype=torch.float)

    def _novo_modelo(self) -> AutoModelForSequenceClassification:
        """
        Instancia um modelo novo com cabeca de classificacao do tamanho
        correto.

        `ignore_mismatched_sizes=True` cobre o caso em que `modelo_base`
        aponta para um checkpoint que JA foi fine-tuned com um numero
        diferente de classes - situacao real quando se retreina sobre um
        corpus ampliado. Sem o parametro, a biblioteca aborta com erro
        de shape na camada final. Com ele, o corpo pre-treinado e
        aproveitado e apenas a cabeca de classificacao e reinicializada,
        que e exatamente o comportamento desejado.

        `id2label`/`label2id` sao gravados na config do modelo: assim o
        mapeamento indice->rotulo viaja junto com o checkpoint e nao
        depende de um arquivo auxiliar.
        """
        return AutoModelForSequenceClassification.from_pretrained(
            self.modelo_base,
            num_labels=len(self.classes_),
            id2label={i: c for i, c in enumerate(self.classes_)},
            label2id={c: i for i, c in enumerate(self.classes_)},
            ignore_mismatched_sizes=True,
        )

    def _treinar_um_modelo(
        self,
        X_tr: list[str], y_tr: np.ndarray,
        X_va: list[str], y_va: np.ndarray,
        epocas: int, batch_size: int, lr: float,
        output_dir: str, paciencia: int, verboso: bool = True,
    ) -> tuple[AutoModelForSequenceClassification, np.ndarray, list[dict]]:
        """
        Treina um modelo e devolve (modelo, logits_de_validacao, historico).

        O HISTORICO e o `log_history` do Trainer: a perda de treino a
        cada `logging_steps` e as metricas de validacao ao fim de cada
        epoca. E a materia-prima da curva de treino do relatorio - o
        grafico que mostra em que epoca o modelo comecou a decorar em
        vez de aprender. Sem devolve-lo aqui, essa informacao morre
        junto com o Trainer ao fim do fold.
        """
        modelo = self._novo_modelo()
        ds_tr = _DatasetNoticias(self._tokenizar(X_tr), list(map(int, y_tr)))
        ds_va = _DatasetNoticias(self._tokenizar(X_va), list(map(int, y_va)))

        # Warmup em NUMERO DE PASSOS, nao em proporcao.
        # `warmup_ratio` esta depreciado e sera removido no transformers
        # v5.2 - passar a proporcao faria o script quebrar numa
        # atualizacao futura da biblioteca, possivelmente meses depois,
        # quando ninguem lembra o que mudou. Calcular os passos aqui
        # produz exatamente o mesmo cronograma de aquecimento (10% do
        # treino) e nao depende de API depreciada.
        passos_por_epoca = max(1, math.ceil(len(X_tr) / batch_size))
        total_passos = passos_por_epoca * max(1, epocas)
        warmup = max(1, int(0.1 * total_passos))

        args = _training_args_compat(
            output_dir=output_dir,
            num_train_epochs=epocas,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=lr,
            warmup_steps=warmup,       # estabiliza as primeiras atualizacoes
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            logging_steps=25,
            report_to=[],
            fp16=torch.cuda.is_available(),
            # pin_memory so faz sentido quando ha GPU para onde
            # transferir. Em CPU o PyTorch emite um UserWarning a cada
            # DataLoader criado - com k folds x epocas, isso enche a
            # saida de ruido e esconde os avisos que importam.
            dataloader_pin_memory=torch.cuda.is_available(),
            seed=self.seed,
            disable_tqdm=not verboso,
        )

        trainer = _TrainerPonderado(
            model=modelo,
            args=args,
            train_dataset=ds_tr,
            eval_dataset=ds_va,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=_metricas,
            pesos_classe=self._pesos_classe(y_tr),
            # Early stopping evita que o modelo continue treinando
            # depois do ponto de sobreajuste. Em corpus pequeno, mBERT
            # decora o conjunto de treino em 2-3 epocas.
            callbacks=[EarlyStoppingCallback(early_stopping_patience=paciencia)],
        )
        trainer.train()
        logits = trainer.predict(ds_va).predictions
        historico = [dict(registro) for registro in trainer.state.log_history]
        return trainer.model, logits, historico

    # -----------------------------------------------------------------
    # Calibracao
    # -----------------------------------------------------------------
    @staticmethod
    def _ajustar_temperatura(logits: np.ndarray, y: np.ndarray, max_iter: int = 200) -> float:
        """
        Temperature scaling (GUO et al., 2017): aprende um escalar T > 0
        que minimiza a log-verossimilhanca negativa sobre dados FORA da
        amostra de treino, e divide os logits por T antes do softmax.

        Propriedade importante: como T e um escalar positivo unico, a
        ordem relativa dos logits nao muda - acuracia, precisao, recall
        e F1 permanecem identicos. Muda apenas a magnitude da
        probabilidade, que passa a refletir a frequencia empirica de
        acerto. E o que torna legitimo o Agente 2 tratar o score como
        confianca e o painel exibi-lo ao gestor.
        """
        lg = torch.tensor(logits, dtype=torch.float)
        yy = torch.tensor(y, dtype=torch.long)
        log_t = torch.zeros(1, requires_grad=True)  # T = exp(log_t) garante T > 0
        otimizador = torch.optim.LBFGS([log_t], lr=0.05, max_iter=max_iter)

        def passo():
            otimizador.zero_grad()
            perda = F.cross_entropy(lg / torch.exp(log_t), yy)
            perda.backward()
            return perda

        otimizador.step(passo)
        return float(torch.exp(log_t).item())

    @staticmethod
    def _erro_calibracao(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
        """
        Expected Calibration Error - quanto a confianca media difere da
        acuracia real, por faixa de confianca. Vale a pena reportar no
        TCC o ECE antes e depois da calibracao: e uma evidencia
        quantitativa direta de que a etapa fez efeito.
        """
        conf = probs.max(axis=1)
        acertos = (probs.argmax(axis=1) == y).astype(float)
        limites = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(limites[:-1], limites[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum():
                ece += (m.mean()) * abs(acertos[m].mean() - conf[m].mean())
        return float(ece)

    # -----------------------------------------------------------------
    # Treino
    # -----------------------------------------------------------------
    def treinar(
        self,
        df: pd.DataFrame,
        coluna_texto: str = COL_TEXTO,
        coluna_label: str = COL_LABEL,
        n_folds: int = 5,
        epocas: int = 5,
        batch_size: int = 16,
        lr: float = 2e-5,
        paciencia: int = 2,
        treinar_modelo_final: bool = True,
        output_dir: str = "outputs/agente1_treino",
        salvar_relatorio: str | None = "outputs/agente1_metricas.json",
        dir_relatorio: str | None = DIR_RELATORIO_PADRAO,
        verboso: bool = True,
    ) -> dict:
        """
        Executa a validacao cruzada estratificada, gera o relatorio de
        avaliacao e, ao final, treina o modelo de producao sobre 100% do
        corpus.

        Por que treinar de novo no corpus inteiro depois do k-fold: os
        k modelos da validacao existem para ESTIMAR o desempenho, nao
        para ser usados em producao - cada um viu apenas (k-1)/k dos
        dados. A pratica padrao e reportar a metrica do k-fold e
        entregar um modelo treinado com tudo (HASTIE et al., 2009).

        Parametros relevantes:
            n_folds: 1 desativa a validacao cruzada e usa holdout 80/20.
            treinar_modelo_final: False economiza um treino quando voce
                so quer a metrica para o texto do TCC.
            dir_relatorio: pasta das tabelas e figuras de avaliacao.
                None desliga a geracao (apenas o JSON e mantido).
            verboso: False silencia as barras de progresso e os avisos
                internos do transformers. Util em teste automatizado,
                onde a saida relevante fica soterrada sob centenas de
                linhas de log de biblioteca.

        Retorna um dicionario com metricas por fold, agregado
        media +/- desvio, matriz de confusao e ECE antes/depois da
        calibracao - pronto para virar tabela na secao de resultados.
        """
        verificar_ambiente()
        fixar_seed(self.seed)

        if not verboso:
            from transformers import logging as _hf_logging

            _hf_logging.set_verbosity_error()      # relatorios de carregamento
            _hf_logging.disable_progress_bar()     # barras de salvar/carregar pesos

        df, textos, y = self._preparar(df, coluna_texto, coluna_label)
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelo_base)

        n = len(textos)
        dist = {c: int((y == i).sum()) for i, c in enumerate(self.classes_)}
        print(f"[Agente 1] Corpus: {n} documentos | classes: {dist} | device: {self.device}")

        minimo = min(dist.values())
        if n_folds > 1 and minimo < n_folds:
            n_folds = max(2, minimo)
            print(f"[Agente 1] AVISO: classe minoritaria tem {minimo} exemplos. n_folds reduzido para {n_folds}.")
        if minimo < 10:
            print(
                f"[Agente 1] AVISO METODOLOGICO: a classe menos frequente tem apenas {minimo} "
                "exemplos. As metricas dessa classe terao intervalo de confianca muito largo; "
                "declare isso como limitacao no TCC ou amplie a anotacao."
            )

        textos_arr = np.array(textos, dtype=object)
        oof_logits = np.zeros((n, len(self.classes_)), dtype=np.float32)
        # Em que fold cada documento foi AVALIADO. Guardar isso e o que
        # permite auditar depois que nenhuma linha foi avaliada por um
        # modelo que a tinha visto, e sustenta a tabela por ciclo.
        fold_da_linha = np.zeros(n, dtype=int)
        metricas_folds: list[dict] = []
        historicos: list[tuple[int, list[dict]]] = []

        if n_folds > 1:
            splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
            particoes = list(splitter.split(textos_arr, y))
        else:
            i_tr, i_va = train_test_split(
                np.arange(n), test_size=0.2, random_state=self.seed, stratify=y
            )
            particoes = [(i_tr, i_va)]

        for k, (i_tr, i_va) in enumerate(particoes, start=1):
            print(f"\n{'=' * 62}\n[Agente 1] Fold {k}/{len(particoes)} "
                  f"(treino={len(i_tr)}, validacao={len(i_va)})\n{'=' * 62}")
            modelo, logits, historico = self._treinar_um_modelo(
                textos_arr[i_tr].tolist(), y[i_tr],
                textos_arr[i_va].tolist(), y[i_va],
                epocas=epocas, batch_size=batch_size, lr=lr,
                output_dir=f"{output_dir}/fold{k}", paciencia=paciencia, verboso=verboso,
            )
            oof_logits[i_va] = logits
            fold_da_linha[i_va] = k
            historicos.append((k, historico))

            y_pred = logits.argmax(axis=1)
            f1_por_classe = f1_score(
                y[i_va], y_pred, average=None, labels=np.arange(len(self.classes_)), zero_division=0
            )
            metricas_folds.append({
                "fold": k,
                "n_treino": int(len(i_tr)),
                "n_validacao": int(len(i_va)),
                "acuracia": float((y[i_va] == y_pred).mean()),
                "f1_macro": f1_score(y[i_va], y_pred, average="macro", zero_division=0),
                "f1_weighted": f1_score(y[i_va], y_pred, average="weighted", zero_division=0),
                **{f"f1_{c.lower()}": float(f1_por_classe[i]) for i, c in enumerate(self.classes_)},
            })
            print(f"[Agente 1] Fold {k}: acuracia = {metricas_folds[-1]['acuracia']:.4f} | "
                  f"F1-macro = {metricas_folds[-1]['f1_macro']:.4f}")

            # Libera VRAM antes do proximo fold: sem isso, 5 modelos
            # mBERT acumulados estouram a memoria da GPU gratuita do Colab.
            del modelo
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Libera DISCO: o checkpoint deste fold ja cumpriu sua funcao.
            # O modelo de cada fold existe apenas para produzir os logits
            # out-of-fold - ele nunca vai para producao, quem vai e o
            # modelo final treinado sobre o corpus inteiro. Manter os
            # checkpoints custa caro: um bert-base-multilingual salvo com
            # o estado do otimizador passa de 2 GB, e 5 folds acumulam
            # mais de 10 GB. Rodando no Colab com a pasta montada no
            # Google Drive, isso estoura a cota gratuita de 15 GB e ainda
            # transfere tudo pela rede, o que e lentissimo.
            shutil.rmtree(f"{output_dir}/fold{k}", ignore_errors=True)

        # --- Avaliacao consolidada sobre as predicoes out-of-fold -----
        y_pred_oof = oof_logits.argmax(axis=1)
        relatorio = classification_report(
            y, y_pred_oof, target_names=self.classes_, output_dict=True, zero_division=0
        )
        print("\n" + "=" * 62)
        print("[Agente 1] Desempenho consolidado (predicoes out-of-fold)")
        print("=" * 62)
        print(classification_report(y, y_pred_oof, target_names=self.classes_, zero_division=0))

        probs_antes = torch.softmax(torch.tensor(oof_logits), dim=1).numpy()
        ece_antes = self._erro_calibracao(probs_antes, y)
        self.temperatura = self._ajustar_temperatura(oof_logits, y)
        probs_depois = torch.softmax(torch.tensor(oof_logits) / self.temperatura, dim=1).numpy()
        ece_depois = self._erro_calibracao(probs_depois, y)
        print(f"[Agente 1] Calibracao: T = {self.temperatura:.4f} | "
              f"ECE {ece_antes:.4f} -> {ece_depois:.4f}")

        # Planilha out-of-fold: e ela que vai para o Agente 2 e para
        # qualquer analise de erro, justamente por nao conter predicao
        # de modelo que ja tinha visto a linha.
        self.predicoes_oof_ = self._montar_predicoes_oof(
            df, probs_depois, fold_da_linha, coluna_texto, coluna_label
        )

        f1s = [m["f1_macro"] for m in metricas_folds]
        acuracias = [m["acuracia"] for m in metricas_folds]
        self.relatorio_treino = {
            "modelo_base": self.modelo_base,
            "n_documentos": n,
            "distribuicao_classes": dist,
            "n_folds": len(particoes),
            "hiperparametros": {
                "epocas": epocas, "batch_size": batch_size, "learning_rate": lr,
                "max_length": self.max_length, "seed": self.seed,
                "loss": "cross_entropy ponderada por frequencia inversa",
                "early_stopping_patience": paciencia,
            },
            "metricas_por_fold": metricas_folds,
            "f1_macro_media": float(np.mean(f1s)),
            "f1_macro_desvio": float(np.std(f1s)),
            "acuracia_media": float(np.mean(acuracias)),
            "acuracia_desvio": float(np.std(acuracias)),
            "relatorio_out_of_fold": relatorio,
            "matriz_confusao": confusion_matrix(y, y_pred_oof).tolist(),
            "matriz_confusao_labels": self.classes_,
            "calibracao": {
                "temperatura": self.temperatura,
                "ece_antes": ece_antes,
                "ece_depois": ece_depois,
            },
        }
        print(f"[Agente 1] F1-macro = {np.mean(f1s):.4f} +/- {np.std(f1s):.4f} | "
              f"acuracia = {np.mean(acuracias):.4f} +/- {np.std(acuracias):.4f} "
              f"({len(particoes)} folds)")

        # --- Relatorio de avaliacao (tabelas e figuras) ---------------
        if dir_relatorio:
            # Import tardio: matplotlib e opcional e o relatorio nao
            # pode ser pre-requisito para treinar.
            from _relatorio_agente1 import gerar as gerar_relatorio

            avaliacao = gerar_relatorio(
                destino=dir_relatorio,
                y=y,
                y_pred=y_pred_oof,
                classes=self.classes_,
                metricas_folds=metricas_folds,
                historicos=historicos,
            )
            self.relatorio_treino["relatorio_avaliacao"] = avaliacao["caminhos"]

        # --- Modelo de producao --------------------------------------
        if treinar_modelo_final:
            print(f"\n{'=' * 62}\n[Agente 1] Treinando modelo final sobre 100% do corpus\n{'=' * 62}")
            i_tr, i_va = train_test_split(
                np.arange(n), test_size=0.1, random_state=self.seed, stratify=y
            )
            self.model, _, _ = self._treinar_um_modelo(
                textos_arr[i_tr].tolist(), y[i_tr],
                textos_arr[i_va].tolist(), y[i_va],
                epocas=epocas, batch_size=batch_size, lr=lr,
                output_dir=f"{output_dir}/final", paciencia=paciencia, verboso=verboso,
            )
            self.model.to(self.device)
            # O modelo ja esta em memoria e sera gravado por salvar() no
            # destino definitivo; o checkpoint intermediario nao serve
            # mais para nada.
            shutil.rmtree(f"{output_dir}/final", ignore_errors=True)

        if salvar_relatorio:
            Path(salvar_relatorio).parent.mkdir(parents=True, exist_ok=True)
            with open(salvar_relatorio, "w", encoding="utf-8") as f:
                json.dump(self.relatorio_treino, f, ensure_ascii=False, indent=2)
            print(f"[Agente 1] Metricas salvas em: {salvar_relatorio}")

        return self.relatorio_treino

    def _montar_predicoes_oof(
        self,
        df: pd.DataFrame,
        probabilidades: np.ndarray,
        fold_da_linha: np.ndarray,
        coluna_texto: str,
        coluna_label: str,
    ) -> pd.DataFrame:
        """
        Monta a planilha de predicoes out-of-fold, ja com a temperatura
        aplicada e com a coluna `fold` que identifica em que ciclo cada
        documento foi avaliado.

        Esta e a planilha que deve alimentar o Agente 2 quando se esta
        avaliando o sistema sobre o corpus anotado. A alternativa -
        classificar o mesmo corpus com o modelo final - entrega ao
        Agente 2 probabilidades que o modelo produziu sobre textos que
        ele ja tinha visto, e qualquer metrica calculada dali para
        frente herda essa contaminacao.
        """
        out = df.copy()
        idx = probabilidades.argmax(axis=1)
        out["classe_predita"] = [self.classes_[i] for i in idx]
        out["score"] = probabilidades.max(axis=1)
        for j, classe in enumerate(self.classes_):
            out[f"prob_{classe.lower()}"] = probabilidades[:, j]
        out["fold"] = fold_da_linha
        out["origem_predicao"] = "out_of_fold"
        if coluna_label in out.columns:
            out["acertou"] = out["classe_predita"].values == out[coluna_label].values
        return self._anexar_enriquecimento(out, out[coluna_texto].astype(str).tolist())

    def _anexar_enriquecimento(self, out: pd.DataFrame, textos: list[str]) -> pd.DataFrame:
        """
        Anexa entidades de mercado e direcao do impacto.

        Deterministico e independente do modelo neural: depende so do
        gazetteer, entao pode ser aplicado tanto as predicoes
        out-of-fold quanto as de inferencia, sem risco de vazamento.
        """
        if not self.enriquecer or not textos:
            return out
        entidades = [extrair_entidades(t) for t in textos]
        direcoes = [inferir_direcao(e) for e in entidades]
        for campo in entidades[0].to_dict():
            out[campo] = [e.to_dict()[campo] for e in entidades]
        for campo in direcoes[0].to_dict():
            out[campo] = [d.to_dict()[campo] for d in direcoes]
        return out

    # -----------------------------------------------------------------
    # Inferencia
    # -----------------------------------------------------------------
    def classificar(self, textos: list[str], batch_size: int = 32) -> list[ResultadoClassificacao]:
        """
        Classifica textos novos, aplicando a temperatura aprendida e,
        se `enriquecer=True`, anexando entidades e direcao do impacto.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Agente 1 nao esta pronto: chame treinar() ou carregar() antes de classificar()."
            )
        if not textos:
            return []

        self.model.eval()
        resultados: list[ResultadoClassificacao] = []
        with torch.no_grad():
            for i in range(0, len(textos), batch_size):
                lote = [str(t) for t in textos[i : i + batch_size]]
                enc = self.tokenizer(
                    lote, truncation=True, padding=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits / self.temperatura
                probs = torch.softmax(logits, dim=1).cpu().numpy()

                for texto, vetor in zip(lote, probs):
                    idx = int(np.argmax(vetor))
                    entidades_d, direcao_d = {}, {}
                    if self.enriquecer:
                        ents = extrair_entidades(texto)
                        entidades_d = ents.to_dict()
                        direcao_d = inferir_direcao(ents).to_dict()
                    resultados.append(
                        ResultadoClassificacao(
                            texto=texto,
                            classe_predita=self.classes_[idx],
                            score=float(vetor[idx]),
                            probabilidades={c: float(p) for c, p in zip(self.classes_, vetor)},
                            entidades=entidades_d,
                            direcao=direcao_d,
                        )
                    )
        return resultados

    def classificar_dataframe(
        self,
        df: pd.DataFrame,
        coluna_texto: str = COL_TEXTO,
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """
        Versao orientada a DataFrame - e a que o Orquestrador deve usar
        para noticias NOVAS, ainda nao vistas pelo modelo.

        Devolve o DataFrame original acrescido de: classe_predita, score,
        prob_<classe> por classe, colunas de entidade e de direcao.
        Preserva o indice e o doc_id, o que permite ao Agente 2 detectar
        auto-referencia e ao Agente 4 fazer a juncao com a base de vendas.

        A coluna `origem_predicao` marca a procedencia das predicoes:
        aqui elas vem do modelo final. Se o DataFrame for o proprio
        corpus de treino, as metricas calculadas sobre esta saida serao
        otimistas - use `predicoes_oof_` nesse caso.
        """
        if coluna_texto not in df.columns:
            raise KeyError(f"Coluna '{coluna_texto}' ausente. Rode _schema.preparar_corpus() antes.")

        resultados = self.classificar(df[coluna_texto].astype(str).tolist(), batch_size=batch_size)
        out = df.copy()
        out["classe_predita"] = [r.classe_predita for r in resultados]
        out["score"] = [r.score for r in resultados]
        for classe in self.classes_:
            out[f"prob_{classe.lower()}"] = [r.probabilidades[classe] for r in resultados]
        out["origem_predicao"] = "modelo_final"

        if self.enriquecer:
            for campo in resultados[0].entidades:
                out[campo] = [r.entidades[campo] for r in resultados]
            for campo in resultados[0].direcao:
                out[campo] = [r.direcao[campo] for r in resultados]

        if COL_ID not in out.columns:
            out[COL_ID] = [f"doc{i:06d}" for i in range(len(out))]
        return out

    # -----------------------------------------------------------------
    # Persistencia
    # -----------------------------------------------------------------
    def salvar(self, caminho: str) -> None:
        """
        Salva modelo, tokenizer e METADADOS.

        A versao anterior salvava apenas modelo + tokenizer + classes.
        Faltava a temperatura de calibracao: ao recarregar, o agente
        voltava a produzir scores nao calibrados, silenciosamente
        divergentes dos reportados no TCC. Persistir o estado completo
        e o que torna o experimento reproduzivel.
        """
        if self.model is None:
            raise RuntimeError("Nao ha modelo treinado para salvar.")
        os.makedirs(caminho, exist_ok=True)
        self.model.save_pretrained(caminho)
        self.tokenizer.save_pretrained(caminho)
        meta = {
            "classes": self.classes_,
            "temperatura": self.temperatura,
            "modelo_base": self.modelo_base,
            "max_length": self.max_length,
            "seed": self.seed,
            "enriquecer": self.enriquecer,
            "relatorio_treino": self.relatorio_treino,
        }
        with open(os.path.join(caminho, "agente1_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[Agente 1] Modelo e metadados salvos em: {caminho}")

    @classmethod
    def carregar(cls, caminho: str) -> "AgenteClassificadorImpacto":
        """Restaura o agente com o mesmo estado com que foi salvo."""
        meta_path = os.path.join(caminho, "agente1_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"'{meta_path}' nao encontrado. O diretorio foi salvo por uma versao "
                "antiga do agente? Retreine e salve novamente."
            )
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        agente = cls(
            modelo_base=meta.get("modelo_base", MODELO_BASE),
            max_length=meta.get("max_length", 256),
            seed=meta.get("seed", SEED_PADRAO),
            enriquecer=meta.get("enriquecer", True),
        )
        agente.tokenizer = AutoTokenizer.from_pretrained(caminho)
        agente.model = AutoModelForSequenceClassification.from_pretrained(caminho).to(agente.device)
        agente.classes_ = meta["classes"]
        agente.temperatura = float(meta.get("temperatura", 1.0))
        agente.relatorio_treino = meta.get("relatorio_treino", {})
        return agente


# =====================================================================
# CLI
# =====================================================================

def _ler_planilha(caminho: str) -> pd.DataFrame:
    return pd.read_csv(caminho) if caminho.lower().endswith(".csv") else pd.read_excel(caminho)


def _cli() -> None:
    """
    Dois modos explicitos, porque sao dois momentos diferentes do
    projeto e confundi-los e o erro mais comum:

      treinar    - roda uma vez (ou a cada mudanca no corpus), gasta GPU,
                   produz o modelo, o relatorio de avaliacao E a planilha
                   out-of-fold que o Agente 2 consome
      classificar- roda quantas vezes quiser sobre noticias NOVAS,
                   carregando o modelo ja treinado (segundos, sem treino)

    A planilha exportada pelo modo `treinar` e a OUT-OF-FOLD. A diferenca
    nao e detalhe: classificar o corpus de treino com o modelo final
    produziria, neste projeto, F1-macro 0,79 contra os 0,66 reais.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Agente 1 - Classificador de impacto (mBERT).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemplos:\n"
            "  # treina, gera o relatorio e deixa a planilha pronta para o Agente 2\n"
            "  python agents/agente1_classificador_mbert.py treinar corpus.xlsx\n\n"
            "  # classifica noticias novas com o modelo salvo\n"
            "  python agents/agente1_classificador_mbert.py classificar novas.xlsx\n"
        ),
    )
    sub = parser.add_subparsers(dest="modo", required=True)

    # ---- treinar ----------------------------------------------------
    p_tr = sub.add_parser("treinar", help="Fine-tuning com validacao cruzada")
    p_tr.add_argument("planilha", help="Caminho do .xlsx/.csv anotado")
    p_tr.add_argument("--coluna-texto", default=COL_TEXTO)
    p_tr.add_argument("--coluna-label", default=COL_LABEL)
    p_tr.add_argument("--folds", type=int, default=5, help="1 desativa o k-fold (holdout 80/20)")
    p_tr.add_argument("--epocas", type=int, default=5)
    p_tr.add_argument("--batch-size", type=int, default=16)
    p_tr.add_argument("--lr", type=float, default=2e-5,
                      help="Taxa de aprendizado (padrao 2e-5, faixa usual de fine-tuning "
                           "de BERT: 2e-5 a 5e-5)")
    p_tr.add_argument("--silencioso", action="store_true",
                      help="Suprime barras de progresso e os relatorios de carregamento do "
                           "transformers, deixando visivel so o log dos agentes")
    p_tr.add_argument("--modelo-base", default=MODELO_BASE,
                      help=f"Checkpoint HuggingFace ou pasta local (default: {MODELO_BASE})")
    p_tr.add_argument("--max-length", type=int, default=256)
    p_tr.add_argument("--sem-enriquecimento", action="store_true",
                      help="Desliga entidades e direcao (fiel a secao 3.4.1 original)")
    p_tr.add_argument("--saida", default="modelos/agente1_mbert",
                      help="Pasta onde o modelo treinado sera salvo")
    p_tr.add_argument("--output-dir", default="outputs/agente1_treino",
                      help="Pasta de trabalho do treino (checkpoints temporarios). "
                           "No Colab com o projeto no Drive, use /content/treino: "
                           "escrever no Drive e lento e consome cota.")
    p_tr.add_argument("--dir-relatorio", default=DIR_RELATORIO_PADRAO,
                      help="Pasta das tabelas e figuras de avaliacao (vazio desliga)")
    p_tr.add_argument("--saida-classificado", default="outputs/agente1_classificado.xlsx",
                      help="Planilha out-of-fold, entrada do Agente 2")

    # ---- classificar ------------------------------------------------
    p_cl = sub.add_parser("classificar", help="Classifica noticias novas com um modelo ja treinado")
    p_cl.add_argument("planilha", help="Planilha de noticias a classificar")
    p_cl.add_argument("--modelo", default="modelos/agente1_mbert",
                      help="Pasta do modelo salvo pelo modo 'treinar'")
    p_cl.add_argument("--coluna-texto", default=COL_TEXTO)
    p_cl.add_argument("--saida", default="outputs/agente1_classificado_novas.xlsx")

    args = parser.parse_args()

    from _schema import preparar_corpus

    df, _ = preparar_corpus(_ler_planilha(args.planilha))

    if args.modo == "treinar":
        agente = AgenteClassificadorImpacto(
            modelo_base=args.modelo_base,
            max_length=args.max_length,
            enriquecer=not args.sem_enriquecimento,
        )
        print(f"[Agente 1] Dispositivo: {agente.device} "
              f"(GPU disponivel: {torch.cuda.is_available()})")
        agente.treinar(
            df,
            coluna_texto=args.coluna_texto,
            coluna_label=args.coluna_label,
            n_folds=args.folds,
            epocas=args.epocas,
            batch_size=args.batch_size,
            lr=args.lr,
            output_dir=args.output_dir,
            dir_relatorio=args.dir_relatorio or None,
            verboso=not args.silencioso,
        )
        agente.salvar(args.saida)

        classificado = agente.predicoes_oof_
        destino = args.saida_classificado
        Path(destino).parent.mkdir(parents=True, exist_ok=True)
        classificado.to_excel(destino, index=False)
        print(f"\n[Agente 1] Planilha OUT-OF-FOLD -> {destino}")
        print(f"[Agente 1] {len(classificado)} linhas, cada uma classificada pelo modelo do fold "
              "em que ela estava escondida.")
        print("[Agente 1] Use ESTA planilha para avaliar o sistema. Classificar o corpus de "
              "treino com o modelo final infla a metrica.")
    else:
        agente = AgenteClassificadorImpacto.carregar(args.modelo)
        print(f"[Agente 1] Modelo carregado de '{args.modelo}' | classes: {agente.classes_} | "
              f"T = {agente.temperatura:.4f}")
        classificado = agente.classificar_dataframe(df, coluna_texto=args.coluna_texto)
        destino = args.saida
        Path(destino).parent.mkdir(parents=True, exist_ok=True)
        classificado.to_excel(destino, index=False)
        print(f"[Agente 1] Planilha classificada -> {destino}")

    print(f"[Agente 1] Distribuicao predita: "
          f"{classificado['classe_predita'].value_counts().to_dict()}")
    print("[Agente 1] Proximo passo: rodar o Agente 2 sobre esse arquivo.")


if __name__ == "__main__":
    _cli()
