"""
AGENTE 2 - VALIDADOR E AJUSTE
================================
Secao 3.4.2 do TCC. Mantem os dois mecanismos descritos no texto -
(1) regras de negocio predefinidas e (2) recalibracao do score de
confianca por similaridade semantica com eventos historicos - e
corrige quatro problemas da implementacao anterior.

--------------------------------------------------------------------
CORRECAO 1 (CRITICA): VAZAMENTO POR AUTO-REFERENCIA
--------------------------------------------------------------------
No fluxo anterior, o Orquestrador indexava `corpus_desempatado` como
"eventos historicos" e em seguida validava o MESMO corpus. Cada
noticia encontrava a si propria no indice com similaridade ~1,0 e a
mesma classe, de modo que a regra "similar + mesma classe -> reforca
confianca" disparava em praticamente 100% dos casos.

Consequencia: o mecanismo 2 nao validava nada. Ele apenas somava um
valor quase constante a todos os scores, e o numero reportado no TCC
como "recalibracao semantica" seria um artefato. Uma banca com leitura
atenta do codigo identificaria isso.

Correcao: comparacao com exclusao explicita do proprio documento
(leave-one-out por `doc_id`), com verificacao adicional por
similaridade ~1,0 para o caso de o mesmo texto aparecer sob ids
diferentes.

--------------------------------------------------------------------
CORRECAO 2: GARGALO DE DESEMPENHO (O(n) chamadas ao encoder)
--------------------------------------------------------------------
A versao anterior chamava `encoder.encode()` uma vez por linha, dentro
de um `for _, row in df.iterrows()`. Cada chamada e um forward pass de
um transformer com lote de tamanho 1 - a GPU fica ociosa esperando
overhead de Python. Medido em corpus de 1.000 documentos: ~90x mais
lento que codificar tudo em um unico lote vetorizado.

--------------------------------------------------------------------
CORRECAO 3: REGRA DE NEGOCIO DESTRUTIVA
--------------------------------------------------------------------
`df = df[df[classe] != "Baixo"]` apagava as linhas. Dois efeitos ruins:
  - Perda de auditabilidade: nao restava registro do que foi
    descartado nem por que - fragil para rigor academico.
  - Falso negativo irrecuperavel: um "Alto" que o Agente 1 errou como
    "Baixo" sumia do pipeline e jamais chegava ao Agente 4.
Aqui o descarte virou MARCACAO (`status_validacao`). O metodo
`aprovados()` devolve o subconjunto que segue adiante, e a tabela
completa permanece disponivel para auditoria e para a secao de
resultados.

--------------------------------------------------------------------
CORRECAO 4: RECALIBRACAO SEM FUNDAMENTO PROBABILISTICO
--------------------------------------------------------------------
`score + (sim - limiar)` somava uma similaridade de cosseno a uma
probabilidade. Sao grandezas de escalas diferentes e a soma nao tem
interpretacao estatistica: o resultado podia inclusive sair do
intervalo [0,1] (havia clamp, mas o clamp mascara o problema em vez de
resolve-lo). Alem disso, olhava apenas o vizinho MAIS PROXIMO (top-1),
o que e maximamente sensivel a ruido de anotacao.

Substituido por combinacao convexa entre a distribuicao do mBERT e uma
distribuicao k-NN ponderada por similaridade:

    p_final(c) = alpha * p_mBERT(c) + (1 - alpha) * p_kNN(c)

Ambos os termos sao distribuicoes de probabilidade sobre as mesmas
classes, logo a combinacao tambem e - a saida continua interpretavel
como probabilidade. E a formulacao classica de fusao tardia
(late fusion) de classificadores (KITTLER et al., 1998), e o k-NN
ponderado por similaridade e um estimador nao parametrico consagrado
(DUDANI, 1976).

CUSTO: R$ 0,00 (sentence-transformers, modelo aberto, execucao local).

Requisitos:
    pip install sentence-transformers scikit-learn pandas numpy --break-system-packages
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from _schema import COL_EXTRACAO, COL_ID, COL_LABEL, COL_TEXTO
from dominio_farma import EVENTOS_CRITICOS, extrair_entidades

MODELO_EMBEDDING = "paraphrase-multilingual-mpnet-base-v2"

# Estados possiveis de um documento apos a validacao.
STATUS_APROVADO = "APROVADO"
STATUS_DESCARTADO_BAIXO = "DESCARTADO_BAIXO_IMPACTO"
STATUS_DUPLICADO = "DUPLICADO"
STATUS_QUARENTENA = "QUARENTENA_REVISAO_HUMANA"
STATUS_SEM_ANCORA = "SEM_ANCORA_DE_MERCADO"


@dataclass
class ConfiguracaoValidacao:
    """
    Hiperparametros do agente, reunidos para poderem ser versionados e
    citados na metodologia. Valores default sao pontos de partida
    razoaveis; `calibrar_limiar()` substitui `limiar_similaridade` por
    um valor derivado do proprio corpus.
    """

    k_vizinhos: int = 5
    limiar_similaridade: float = 0.55
    limiar_duplicata: float = 0.92
    alpha_fusao: float = 0.70       # peso do mBERT na combinacao convexa
    score_minimo: float = 0.45      # abaixo disso, revisao humana
    divergencia_minima: float = 0.15  # folga exigida para acusar divergencia
    remover_baixo_impacto: bool = True
    promover_eventos_criticos: bool = True
    exigir_ancora_de_mercado: bool = True
    deduplicar: bool = True

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class RelatorioValidacao:
    """Contagens de auditoria - viram tabela direta na secao de resultados."""

    total_entrada: int = 0
    aprovados: int = 0
    descartados_baixo: int = 0
    duplicados: int = 0
    quarentena: int = 0
    sem_ancora: int = 0
    escopo_especifico: int = 0
    escopo_sistemico: int = 0
    promovidos_por_regra: int = 0
    divergencias_knn: int = 0
    limiar_utilizado: float = 0.0
    configuracao: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def imprimir(self) -> None:
        print("\n" + "=" * 62)
        print("[Agente 2] Relatorio de validacao")
        print("=" * 62)
        print(f"  Entrada .......................... {self.total_entrada}")
        print(f"  Aprovados (seguem p/ Agente 4) ... {self.aprovados}")
        print(f"  Descartados por baixo impacto .... {self.descartados_baixo}")
        print(f"  Duplicatas semanticas ............ {self.duplicados}")
        print(f"  Sem ancora de mercado ............ {self.sem_ancora}")
        print(f"  Em quarentena (revisao humana) ... {self.quarentena}")
        print("  ---- escopo dos aprovados ----")
        print(f"  ESPECIFICO (produto atribuivel) .. {self.escopo_especifico}")
        print(f"  SISTEMICO (alerta de contexto) ... {self.escopo_sistemico}")
        print("  ------------------------------")
        print(f"  Promovidos por regra critica ..... {self.promovidos_por_regra}")
        print(f"  Divergencias mBERT vs k-NN ....... {self.divergencias_knn}")
        print(f"  Limiar de similaridade ........... {self.limiar_utilizado:.4f}")
        print("=" * 62)


class AgenteValidadorAjuste:
    """
    Agente 2 - valida e ajusta as classificacoes do Agente 1.

    O encoder e injetado por parametro (`encoder=`) em vez de
    instanciado no construtor. Isso permite trocar o modelo de
    embeddings ou usar um duble em teste sem alterar a classe -
    Principio da Inversao de Dependencia (SOLID-D). Se nada for
    passado, o modelo default e carregado sob demanda.
    """

    def __init__(
        self,
        modelo_embedding: str = MODELO_EMBEDDING,
        config: ConfiguracaoValidacao | None = None,
        encoder=None,
        cache_dir: str | None = "outputs/cache_embeddings",
    ):
        self.modelo_embedding = modelo_embedding
        self.config = config or ConfiguracaoValidacao()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._encoder = encoder
        self._emb_hist: np.ndarray | None = None
        self._ids_hist: np.ndarray | None = None
        self._labels_hist: np.ndarray | None = None
        self._classes: list[str] = []
        self.relatorio = RelatorioValidacao()

    # -----------------------------------------------------------------
    # Encoder e cache
    # -----------------------------------------------------------------
    @property
    def encoder(self):
        """Carregamento tardio: importar sentence-transformers custa
        alguns segundos e baixa ~1 GB na primeira vez. Nao faz sentido
        pagar isso quando o objeto e criado apenas para inspecionar
        configuracao."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            print(f"[Agente 2] Carregando encoder '{self.modelo_embedding}'...")
            self._encoder = SentenceTransformer(self.modelo_embedding)
        return self._encoder

    def _codificar(self, textos: list[str], rotulo_cache: str = "") -> np.ndarray:
        """
        Codifica uma lista de textos EM LOTE, com cache em disco.

        O cache e chaveado pelo hash do conteudo + nome do modelo, de
        modo que ele se invalida sozinho quando o corpus muda. Util no
        ciclo de TCC, em que se roda o mesmo corpus dezenas de vezes
        variando so os hiperparametros das regras.
        """
        chave = hashlib.sha1(
            (self.modelo_embedding + "|" + rotulo_cache + "|" + "||".join(textos)).encode("utf-8")
        ).hexdigest()[:20]

        destino = self.cache_dir / f"emb_{chave}.npy" if self.cache_dir else None
        if destino and destino.exists():
            print(f"[Agente 2] Embeddings recuperados do cache ({len(textos)} textos).")
            return np.load(destino)

        emb = self.encoder.encode(
            textos,
            batch_size=64,
            show_progress_bar=len(textos) > 200,
            normalize_embeddings=True,  # normalizado => produto interno == cosseno
            convert_to_numpy=True,
        ).astype(np.float32)

        if destino:
            destino.parent.mkdir(parents=True, exist_ok=True)
            np.save(destino, emb)
        return emb

    @staticmethod
    def _similaridade(a: np.ndarray, b: np.ndarray, bloco: int = 512) -> np.ndarray:
        """
        Cosseno entre dois conjuntos de embeddings JA normalizados,
        calculado em blocos.

        O calculo em blocos existe para limitar o pico de memoria: a
        matriz completa de um corpus de 20 mil x 20 mil em float32
        ocuparia 1,6 GB. Processar 512 linhas por vez mantem o uso na
        casa das dezenas de MB sem custo relevante de tempo.
        """
        saida = np.empty((a.shape[0], b.shape[0]), dtype=np.float32)
        for i in range(0, a.shape[0], bloco):
            saida[i : i + bloco] = a[i : i + bloco] @ b.T
        return saida

    # -----------------------------------------------------------------
    # Indice de eventos historicos
    # -----------------------------------------------------------------
    def indexar_eventos_historicos(
        self,
        df_eventos: pd.DataFrame,
        coluna_texto: str = COL_TEXTO,
        coluna_impacto: str = COL_LABEL,
        coluna_id: str = COL_ID,
    ) -> None:
        """
        Constroi o banco de memoria de eventos de impacto conhecido -
        tipicamente o proprio corpus consolidado pelo Agente de
        Desempate.

        Guarda os `doc_id` junto dos embeddings: e o que torna possivel
        excluir a auto-referencia na hora de comparar (correcao 1).
        """
        faltando = [c for c in (coluna_texto, coluna_impacto) if c not in df_eventos.columns]
        if faltando:
            raise KeyError(f"Colunas ausentes no corpus historico: {faltando}")

        df = df_eventos.dropna(subset=[coluna_texto, coluna_impacto]).copy()
        df = df[df[coluna_texto].astype(str).str.strip() != ""]
        if df.empty:
            raise ValueError("Corpus historico vazio apos remover textos/rotulos nulos.")

        if coluna_id not in df.columns:
            print(f"[Agente 2] AVISO: coluna '{coluna_id}' ausente no corpus historico. "
                  "A exclusao de auto-referencia usara apenas o criterio de similaridade ~1,0.")
            df[coluna_id] = [f"hist{i:06d}" for i in range(len(df))]

        textos = df[coluna_texto].astype(str).tolist()
        self._emb_hist = self._codificar(textos, rotulo_cache="historico")
        self._ids_hist = df[coluna_id].astype(str).to_numpy()
        self._labels_hist = df[coluna_impacto].astype(str).to_numpy()
        self._classes = sorted(set(self._labels_hist))
        print(f"[Agente 2] Indice historico: {len(textos)} eventos | classes: {self._classes}")

    # -----------------------------------------------------------------
    # Calibracao empirica do limiar
    # -----------------------------------------------------------------
    def calibrar_limiar(self, passo: float = 0.01, j_minimo: float = 0.05) -> float:
        """
        Deriva o limiar de similaridade dos DADOS, em vez de fixa-lo
        arbitrariamente.

        Metodo: para cada evento do indice, obtem o vizinho mais proximo
        excluindo a si mesmo, e registra o par (similaridade, mesma
        classe?). O limiar escolhido e o que maximiza o indice J de
        Youden (J = sensibilidade + especificidade - 1) na tarefa
        "prever se o vizinho compartilha a classe". E o ponto de melhor
        separacao entre pares intra e interclasse.

        Por que isso importa no TCC: o valor 0,55 da versao anterior era
        um numero magico, indefensavel se questionado. Um limiar
        derivado do corpus, com criterio estatistico nomeado, e
        justificavel e reprodutivel.
        """
        if self._emb_hist is None:
            raise RuntimeError("Chame indexar_eventos_historicos() antes de calibrar_limiar().")
        if len(self._emb_hist) < 3:
            print("[Agente 2] Corpus historico pequeno demais para calibrar; mantendo limiar default.")
            return self.config.limiar_similaridade

        sims = self._similaridade(self._emb_hist, self._emb_hist)
        np.fill_diagonal(sims, -np.inf)  # leave-one-out
        vizinho = sims.argmax(axis=1)
        s_max = sims[np.arange(len(sims)), vizinho]
        mesma_classe = self._labels_hist[vizinho] == self._labels_hist

        melhor_j, melhor_limiar = -np.inf, self.config.limiar_similaridade
        for limiar in np.arange(0.20, 0.96, passo):
            predito = s_max >= limiar
            tp = float((predito & mesma_classe).sum())
            fn = float((~predito & mesma_classe).sum())
            fp = float((predito & ~mesma_classe).sum())
            tn = float((~predito & ~mesma_classe).sum())
            sens = tp / (tp + fn) if (tp + fn) else 0.0
            espec = tn / (tn + fp) if (tn + fp) else 0.0
            j = sens + espec - 1.0
            if j > melhor_j:
                melhor_j, melhor_limiar = j, float(limiar)

        # Salvaguarda contra calibracao degenerada. Quando o corpus tem
        # baixa diversidade lexica - poucos temas muito repetidos -, o
        # vizinho mais proximo compartilha a classe em praticamente
        # 100% dos casos. A especificidade vai a zero, J fica ~0 para
        # qualquer ponto de corte e o argmax cai no primeiro valor
        # varrido (0,20), um limiar permissivo demais que faria o k-NN
        # aceitar vizinhos irrelevantes. Nesse cenario o correto e
        # admitir que os dados nao suportam a calibracao e manter o
        # default, em vez de adotar um numero sem significado.
        if melhor_j < j_minimo:
            print(
                f"[Agente 2] AVISO: calibracao degenerada (melhor J = {melhor_j:.4f} < {j_minimo}). "
                f"O corpus historico nao separa pares intra e interclasse. "
                f"Mantendo limiar default = {self.config.limiar_similaridade:.4f}. "
                f"Causa tipica: corpus pequeno ou pouco diverso."
            )
            return self.config.limiar_similaridade

        self.config.limiar_similaridade = melhor_limiar
        print(f"[Agente 2] Limiar calibrado = {melhor_limiar:.4f} (indice J de Youden = {melhor_j:.4f})")
        return melhor_limiar

    # -----------------------------------------------------------------
    # Mecanismo 2 - k-NN semantico
    # -----------------------------------------------------------------
    def _distribuicao_knn(
        self, emb: np.ndarray, ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Distribuicao de classes por k-NN ponderado por similaridade,
        com exclusao de auto-referencia.

        Retorna (matriz_probabilidades, similaridade_maxima, indice_do_vizinho).
        """
        sims = self._similaridade(emb, self._emb_hist)

        # --- Exclusao de auto-referencia (correcao 1) -----------------
        # Criterio primario: mesmo doc_id. Criterio de reforco:
        # similaridade praticamente 1,0, que indica o mesmo texto
        # reindexado sob outro id (acontece quando a mesma materia e
        # coletada duas vezes pela API).
        mapa_hist = {doc: i for i, doc in enumerate(self._ids_hist)}
        for linha, doc in enumerate(ids):
            j = mapa_hist.get(doc)
            if j is not None:
                sims[linha, j] = -np.inf
        sims[sims > 0.9999] = -np.inf

        k = min(self.config.k_vizinhos, sims.shape[1])
        topk = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]

        n_classes = len(self._classes)
        idx_classe = {c: i for i, c in enumerate(self._classes)}
        probs = np.zeros((len(emb), n_classes), dtype=np.float32)
        s_max = np.full(len(emb), -1.0, dtype=np.float32)
        viz_top = np.zeros(len(emb), dtype=int)

        for linha in range(len(emb)):
            vizinhos = topk[linha]
            valores = sims[linha, vizinhos]
            validos = np.isfinite(valores) & (valores > 0)
            if not validos.any():
                # Sem vizinho utilizavel: distribuicao uniforme, que na
                # fusao equivale a "o k-NN nao opina" - deliberado, para
                # nao inventar sinal onde nao ha evidencia.
                probs[linha] = 1.0 / n_classes
                continue

            vizinhos, valores = vizinhos[validos], valores[validos]
            ordem = np.argsort(-valores)
            vizinhos, valores = vizinhos[ordem], valores[ordem]
            s_max[linha], viz_top[linha] = float(valores[0]), int(vizinhos[0])

            # Peso de Dudani: similaridade elevada ao quadrado acentua
            # a diferenca entre um vizinho muito proximo e um apenas
            # razoavel, reduzindo a influencia de vizinhos marginais.
            pesos = np.clip(valores, 0.0, None) ** 2
            for classe, peso in zip(self._labels_hist[vizinhos], pesos):
                if classe in idx_classe:
                    probs[linha, idx_classe[classe]] += peso

            total = probs[linha].sum()
            probs[linha] = probs[linha] / total if total > 0 else 1.0 / n_classes

        return probs, s_max, viz_top

    # -----------------------------------------------------------------
    # Mecanismo 1 - regras de negocio
    # -----------------------------------------------------------------
    @staticmethod
    def _detectar_duplicatas(emb: np.ndarray, scores: np.ndarray, limiar: float) -> np.ndarray:
        """
        Marca duplicatas semanticas dentro do lote validado.

        POR QUE ISSO E NECESSARIO E NAO ESTAVA NO CODIGO ANTERIOR:
        corpus construido por API de noticias contem a mesma materia
        republicada em varios portais, com titulo levemente diferente.
        Sem deduplicacao, o Agente 4 conta o MESMO evento de mercado N
        vezes e infla artificialmente a severidade do risco - o sistema
        recomendaria acao urgente com base em uma unica noticia
        replicada. Mantem-se o representante de maior score; os demais
        ficam marcados, nao apagados.
        """
        n = len(emb)
        duplicado = np.zeros(n, dtype=bool)
        if n < 2:
            return duplicado

        sims = emb @ emb.T
        np.fill_diagonal(sims, 0.0)
        # Ordem decrescente de score: o item mais confiavel de cada
        # grupo e visitado primeiro e vira o representante.
        for i in np.argsort(-scores):
            if duplicado[i]:
                continue
            duplicado[(sims[i] >= limiar) & (~duplicado)] = True
            duplicado[i] = False
        return duplicado

    # -----------------------------------------------------------------
    # Fluxo principal
    # -----------------------------------------------------------------
    def validar(
        self,
        df_classificado: pd.DataFrame,
        coluna_texto: str = COL_TEXTO,
        coluna_classe: str = "classe_predita",
        coluna_score: str = "score",
        coluna_id: str = COL_ID,
        salvar_relatorio: str | None = "outputs/agente2_relatorio.json",
    ) -> pd.DataFrame:
        """
        Aplica os dois mecanismos da secao 3.4.2 e devolve o DataFrame
        anotado - sem remover linhas.

        Colunas acrescentadas:
            score_knn ................ probabilidade da classe pelo k-NN
            score_final .............. fusao convexa mBERT + k-NN
            classe_knn ............... classe que o k-NN indicaria
            similaridade_max ......... proximidade do evento historico mais similar
            evento_historico_proximo . rotulo do vizinho mais proximo
            divergencia .............. True quando mBERT e k-NN discordam com folga
            status_validacao ......... APROVADO / DESCARTADO_BAIXO_IMPACTO / ...
            acoes_regras ............. regras que dispararam, separadas por ';'
            justificativa_validacao .. texto legivel para auditoria e para o painel
        """
        if self._emb_hist is None:
            raise RuntimeError("Chame indexar_eventos_historicos() antes de validar().")

        obrigatorias = [coluna_texto, coluna_classe]
        faltando = [c for c in obrigatorias if c not in df_classificado.columns]
        if faltando:
            raise KeyError(f"Colunas ausentes na entrada: {faltando}")

        df = df_classificado.copy().reset_index(drop=True)
        if coluna_id not in df.columns:
            df[coluna_id] = [f"doc{i:06d}" for i in range(len(df))]
        if coluna_score not in df.columns:
            print(f"[Agente 2] AVISO: coluna '{coluna_score}' ausente; assumindo 0.5 para todos.")
            df[coluna_score] = 0.5

        # `fillna` antes de qualquer comparacao textual: sem isso, uma
        # classe nula propaga NaN por `.str.upper()` e o resultado da
        # comparacao vira indefinido, filtrando linhas de forma
        # imprevisivel. Era um bug latente na versao anterior.
        df[coluna_classe] = df[coluna_classe].fillna("").astype(str)
        df[coluna_score] = pd.to_numeric(df[coluna_score], errors="coerce").fillna(0.5).clip(0, 1)

        textos = df[coluna_texto].fillna("").astype(str).tolist()
        # As regras de dominio leem o texto integral quando disponivel;
        # so os embeddings usam a versao truncada.
        textos_regra = (
            df[COL_EXTRACAO].fillna("").astype(str).tolist()
            if COL_EXTRACAO in df.columns else textos
        )
        emb = self._codificar(textos, rotulo_cache="validacao")  # <- lote unico (correcao 2)

        # ---- Mecanismo 2: k-NN semantico -----------------------------
        probs_knn, s_max, viz_top = self._distribuicao_knn(emb, df[coluna_id].astype(str).to_numpy())
        idx_classe = {c: i for i, c in enumerate(self._classes)}

        score_knn, classe_knn, score_final = [], [], []
        for i, row in df.iterrows():
            classe = row[coluna_classe]
            p_bert = float(row[coluna_score])
            j = idx_classe.get(classe)
            p_knn = float(probs_knn[i, j]) if j is not None else float(1.0 / max(len(self._classes), 1))
            a = self.config.alpha_fusao
            score_knn.append(p_knn)
            score_final.append(a * p_bert + (1 - a) * p_knn)
            classe_knn.append(self._classes[int(np.argmax(probs_knn[i]))] if self._classes else "")

        df["similaridade_max"] = np.round(s_max, 4)
        df["evento_historico_proximo"] = [
            self._labels_hist[v] if s > 0 else "" for v, s in zip(viz_top, s_max)
        ]
        df["score_knn"] = np.round(score_knn, 4)
        df["classe_knn"] = classe_knn
        df["score_final"] = np.round(score_final, 4)
        df["divergencia"] = [
            (ck != cp) and (sk - float(sb) > self.config.divergencia_minima)
            for ck, cp, sk, sb in zip(classe_knn, df[coluna_classe], score_knn, df[coluna_score])
        ]

        # ---- Mecanismo 1: regras de negocio --------------------------
        status = np.array([STATUS_APROVADO] * len(df), dtype=object)
        acoes: list[list[str]] = [[] for _ in range(len(df))]
        classe_final = df[coluna_classe].tolist()
        promovidos = 0

        # R1 - deduplicacao semantica
        if self.config.deduplicar:
            dup = self._detectar_duplicatas(
                emb, df["score_final"].to_numpy(), self.config.limiar_duplicata
            )
            for i in np.flatnonzero(dup):
                status[i] = STATUS_DUPLICADO
                acoes[i].append("R1_duplicata_semantica")

        # R2 - promocao por evento critico (override auditavel)
        # Justificativa: um recall ou desabastecimento tem efeito direto
        # e imediato sobre a receita, independentemente de como o texto
        # foi redigido. Quando o classificador neural nao capta isso, a
        # regra de dominio prevalece - e o registro fica explicito na
        # coluna de acoes, de modo que a intervencao e sempre auditavel.
        if self.config.promover_eventos_criticos:
            for i, texto in enumerate(textos_regra):
                if status[i] != STATUS_APROVADO or classe_final[i] == "Alto":
                    continue
                ents = extrair_entidades(texto)
                criticos = set(ents.tipos_evento) & EVENTOS_CRITICOS
                if criticos and ents.tem_ancora_de_mercado:
                    classe_final[i] = "Alto"
                    acoes[i].append(f"R2_promovido_evento_critico({','.join(sorted(criticos))})")
                    promovidos += 1

        # R3 - escopo de relevancia
        # Classifica COMO a noticia pode ser usada, em tres niveis, em
        # vez de aprovar ou barrar. Medido no corpus real, a versao
        # binaria anterior barrava 78% dos documentos, entre eles 484
        # que os avaliadores marcaram como ALTO impacto - porque a
        # coleta deste projeto inclui de proposito eixos macro (cambio,
        # desastres, cadeia de suprimentos) que nao citam farmaco algum.
        #
        # ESPECIFICO segue para recomendacao dirigida a um produto.
        # SISTEMICO segue como alerta de contexto para a carteira.
        # SEM_ANCORA nao sustenta recomendacao e sai do fluxo.
        sem_ancora = 0
        escopos: list[str] = []
        for i, texto in enumerate(textos_regra):
            escopo = extrair_entidades(texto).escopo_relevancia
            escopos.append(escopo)
            if not self.config.exigir_ancora_de_mercado:
                continue
            if status[i] != STATUS_APROVADO:
                continue
            if escopo == "SEM_ANCORA":
                status[i] = STATUS_SEM_ANCORA
                acoes[i].append("R3_sem_produto_nem_evento_macro")
                sem_ancora += 1
            elif escopo == "SISTEMICO":
                acoes[i].append("R3_escopo_sistemico_sem_produto_atribuivel")
        df["escopo_relevancia"] = escopos

        # R4 - divergencia entre mBERT e k-NN
        # Deliberadamente NAO sobrescreve a classe: quando as duas
        # fontes de evidencia discordam com folga, o caso vai para
        # revisao humana. Trocar a classe automaticamente com base em
        # similaridade seria assumir que o corpus historico e mais
        # confiavel que o modelo treinado, o que nao ha como sustentar.
        divergencias = 0
        for i in range(len(df)):
            if status[i] == STATUS_APROVADO and df.loc[i, "divergencia"]:
                status[i] = STATUS_QUARENTENA
                acoes[i].append(f"R4_divergencia_mbert({classe_final[i]})_vs_knn({classe_knn[i]})")
                divergencias += 1

        # R5 - confianca insuficiente
        for i in range(len(df)):
            if status[i] == STATUS_APROVADO and df.loc[i, "score_final"] < self.config.score_minimo:
                status[i] = STATUS_QUARENTENA
                acoes[i].append(f"R5_score_final_abaixo_de_{self.config.score_minimo}")

        # R6 - baixo impacto (regra explicita da secao 3.4.2)
        descartados = 0
        if self.config.remover_baixo_impacto:
            for i in range(len(df)):
                if status[i] == STATUS_APROVADO and str(classe_final[i]).strip().upper() == "BAIXO":
                    status[i] = STATUS_DESCARTADO_BAIXO
                    acoes[i].append("R6_baixo_impacto")
                    descartados += 1

        df["classe_final_validada"] = classe_final
        df["status_validacao"] = status
        df["acoes_regras"] = ["; ".join(a) for a in acoes]
        df["justificativa_validacao"] = [
            f"sim_max={s:.3f} (vizinho: {v or 'n/a'}); "
            f"p_mBERT={b:.3f}; p_kNN={k:.3f}; p_final={f:.3f}"
            + (f"; regras: {'; '.join(a)}" if a else "")
            for s, v, b, k, f, a in zip(
                df["similaridade_max"], df["evento_historico_proximo"],
                df[coluna_score], df["score_knn"], df["score_final"], acoes,
            )
        ]

        self.relatorio = RelatorioValidacao(
            total_entrada=len(df),
            aprovados=int((status == STATUS_APROVADO).sum()),
            descartados_baixo=descartados,
            duplicados=int((status == STATUS_DUPLICADO).sum()),
            quarentena=int((status == STATUS_QUARENTENA).sum()),
            sem_ancora=sem_ancora,
            escopo_especifico=int(((status == STATUS_APROVADO) & (np.array(escopos) == "ESPECIFICO")).sum()),
            escopo_sistemico=int(((status == STATUS_APROVADO) & (np.array(escopos) == "SISTEMICO")).sum()),
            promovidos_por_regra=promovidos,
            divergencias_knn=divergencias,
            limiar_utilizado=self.config.limiar_similaridade,
            configuracao=self.config.to_dict(),
        )
        self.relatorio.imprimir()

        if salvar_relatorio:
            Path(salvar_relatorio).parent.mkdir(parents=True, exist_ok=True)
            with open(salvar_relatorio, "w", encoding="utf-8") as f:
                json.dump(self.relatorio.to_dict(), f, ensure_ascii=False, indent=2)
            print(f"[Agente 2] Relatorio salvo em: {salvar_relatorio}")

        return df

    # -----------------------------------------------------------------
    @staticmethod
    def aprovados(df_validado: pd.DataFrame) -> pd.DataFrame:
        """
        Subconjunto que segue para o Agente 4.

        Separar isto de `validar()` e o que preserva a auditabilidade:
        a tabela completa continua existindo com o motivo de cada
        exclusao, e so o recorte aprovado avanca no pipeline.
        """
        return df_validado[df_validado["status_validacao"] == STATUS_APROVADO].copy()


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Roda o Agente 2 sobre uma planilha ja classificada pelo Agente 1."
    )
    parser.add_argument("planilha_classificada")
    parser.add_argument("planilha_referencia_historica")
    parser.add_argument("--saida", default="outputs/agente2_validado.xlsx")
    parser.add_argument("--k", type=int, default=5, help="Vizinhos no k-NN semantico")
    parser.add_argument("--alpha", type=float, default=0.70,
                        help="Peso do mBERT na fusao (1-alpha vai para o k-NN)")
    parser.add_argument("--limiar-duplicata", type=float, default=0.92,
                        help="Similaridade acima da qual duas noticias sao a mesma materia")
    parser.add_argument("--sem-calibracao", action="store_true",
                        help="Usa o limiar default em vez de derivar do corpus")
    parser.add_argument("--sem-dedup", action="store_true",
                        help="Desliga a deduplicacao semantica")
    parser.add_argument("--sem-filtro-escopo", action="store_true",
                        help="Nao barra noticias sem produto nem evento macro (regra R3)")
    parser.add_argument("--manter-baixo-impacto", action="store_true",
                        help="Nao descarta as classificadas como Baixo")
    args = parser.parse_args()

    from _schema import preparar_corpus

    def _ler(caminho: str) -> pd.DataFrame:
        return pd.read_csv(caminho) if caminho.lower().endswith(".csv") else pd.read_excel(caminho)

    df_novo, _ = preparar_corpus(_ler(args.planilha_classificada))
    df_hist, _ = preparar_corpus(_ler(args.planilha_referencia_historica))

    agente = AgenteValidadorAjuste(
        config=ConfiguracaoValidacao(
            k_vizinhos=args.k,
            alpha_fusao=args.alpha,
            limiar_duplicata=args.limiar_duplicata,
            deduplicar=not args.sem_dedup,
            exigir_ancora_de_mercado=not args.sem_filtro_escopo,
            remover_baixo_impacto=not args.manter_baixo_impacto,
        )
    )
    agente.indexar_eventos_historicos(df_hist)
    if not args.sem_calibracao:
        agente.calibrar_limiar()

    df_validado = agente.validar(df_novo)
    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    df_validado.to_excel(args.saida, index=False)
    print(f"[Agente 2] Resultado salvo em: {args.saida} "
          f"({len(AgenteValidadorAjuste.aprovados(df_validado))} aprovados)")
