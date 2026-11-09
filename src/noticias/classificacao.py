"""***************************************************************************************
SCRIPT    : classificacao.py
OBJETIVO  : Agentes 1 e 2 aplicados as noticias novas.

            Para cada noticia define:
              nivel_impacto  alto | medio | baixo
              sentimento     positivo | neutro | negativo   (+ score -1..1)
              escopo         proprio | concorrente | mercado | nenhum
              grupos_alvo    grupos de produto proprio citados (ex.: Grupo A)
              categoria      tema da noticia

            O IMPACTO vem da melhor fonte disponivel, nesta ordem:
              1. corpus anotado (data/estatistica_kappa/noticias_classificadas_ampliado.xlsx)
                 -> voto majoritario dos 3 avaliadores; empate total = "medio"
                 (o caso que o Agente de Desempate resolveria)
              2. LLM (OPENAI_API_KEY + usar_llm=True), com o mesmo criterio do
                 Agente de Desempate
              3. Naive Bayes treinado no proprio corpus anotado (acuracia
                 de ~68% em validacao, contra 49% da classe majoritaria)
              4. regras por palavra-chave (so se o corpus nao existir)

            A coluna `origem_classificacao` registra qual das tres foi usada,
            para nao misturar rotulo humano com rotulo heuristico na analise.

            Agente 2 (Validador): mantem so ALTO e MEDIO e descarta noticias
            sem relacao com o mercado (escopo = nenhum).

            O mapa de entidades (que palavra aponta para produto proprio,
            concorrente ou mercado) fica em data/config/mapa_entidades_noticias.json
            e pode ser editado sem mexer no codigo.

AUTOR     : Marcos Arambasic
DATA      : OUT/2026
****************************************************************************************/
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
CAMINHO_MAPA = RAIZ / "data" / "config" / "mapa_entidades_noticias.json"
CAMINHO_CORPUS = RAIZ / "data" / "estatistica_kappa" / "noticias_classificadas_ampliado.xlsx"

# ----------------------------------------------------------------------------
# Mapa de entidades padrao (gravado em data/config na primeira execucao).
# Os produtos sinteticos nao tem nome real; o mapa diz a qual GRUPO proprio
# cada molecula/marca corresponde. Ajuste conforme o portfolio da empresa.
# ----------------------------------------------------------------------------
MAPA_PADRAO = {
    "_comentario": "Edite livremente. Termos em minusculas e sem acento. "
                   "proprio: grupo de produto -> termos; concorrente/mercado: lista de termos.",
    "id_mercado": 1,
    "proprio": {
        "Grupo A": ["empagliflozina", "empagliflozin", "jardiance"],
        "Grupo B": ["linagliptina", "linagliptin", "trajenta"],
    },
    "concorrente": [
        "dapagliflozina", "dapagliflozin", "forxiga", "canagliflozina", "canagliflozin",
        "semaglutida", "semaglutide", "ozempic", "wegovy", "rybelsus", "tirzepatida",
        "tirzepatide", "mounjaro", "liraglutida", "liraglutide", "glp-1", "glp1",
        "insulina", "insulin", "degludeca", "degludec", "metformina", "metformin",
        "sitagliptina", "sitagliptin", "novo nordisk", "eli lilly", "astrazeneca",
    ],
    "mercado": [
        "sglt2", "sglt-2", "isglt2", "diabetes", "diabetico", "diabetic", "dm2",
        "hemoglobina glicada", "glycated", "hba1c", "metabolic", "metabolica",
        "obesidade", "obesity", "anvisa", "cmed", "conitec", "pbm", "farmacia popular",
        "sus", "ifa", "insumo farmaceutico", "cambio", "dolar", "exchange rate",
        "dollar", "reajuste", "drug price", "preco de medicamento", "mace",
        "cardiovascular", "doenca renal", "kidney", "farmaceutic", "pharmaceutical",
        "pharmacy", "medicamento", "medicine",
    ],
}

# ----------------------------------------------------------------------------
# Regras de impacto (fallback). Mesmo criterio do prompt do Agente de Desempate.
# ----------------------------------------------------------------------------
TERMOS_ALTO = [
    "aprova", "aprovad", "approv", "registro", "recall", "recolhimento", "suspen",
    "proib", "banned", "desabastec", "falta de", "shortage", "escassez",
    "ruptura", "reajuste", "price cap", "teto de preco", "patente", "patent",
    "generico", "generic", "biossimilar", "incorpora", "incorporacao", "conitec",
    "reembolso", "reimbursement", "tarifa", "tariff", "lancamento", "launch",
    "resultado de fase 3", "phase 3", "phase iii",
]
TERMOS_MEDIO = [
    "estudo", "study", "trial", "pesquisa", "research", "parceria", "partnership",
    "investimento", "investment", "aquisicao", "acquisition", "mercado", "market",
    "vendas", "sales", "cambio", "dolar", "exchange rate", "dollar", "inflacao",
    "inflation", "guideline", "diretriz", "consulta publica", "demanda", "demand",
]

LEXICO_POSITIVO = [
    "aprova", "approv", "benefic", "benefit", "eficaz", "effective", "reduz o risco",
    "reduces risk", "lower risk", "melhora", "improv", "crescimento", "growth",
    "amplia", "expand", "acesso", "access", "incorpora", "desconto", "queda do dolar",
    "recorde", "record", "sucesso", "success", "protege", "protect", "avanco",
    "breakthrough",
]
LEXICO_NEGATIVO = [
    "recall", "recolhimento", "suspen", "proib", "banned", "falta", "shortage", "escassez",
    "desabastec", "risco", "risk", "efeito adverso", "adverse", "side effect", "morte",
    "death", "queda", "decline", "fall", "crise", "crisis", "multa", "fine", "processo",
    "lawsuit", "alta do dolar", "aumento de preco", "price increase", "fraude", "fraud",
    "investigac", "investigation", "atraso", "delay", "fracass", "fail", "corte", "cut",
    "warn", "alerta", "alert", "severe", "grave", "concern", "preocupa", "hair loss",
    "perda", "loss", "danger", "perigo", "toxic",
]

CATEGORIAS = {
    "regulatorio": ["anvisa", "conitec", "cmed", "regulat", "fda", "ema ", "aprova", "registro", "approv"],
    "preco_e_acesso": ["preco", "price", "reajuste", "reembolso", "pbm", "sus", "farmacia popular", "desconto"],
    "abastecimento": ["falta", "shortage", "escassez", "desabastec", "ruptura", "ifa", "insumo", "supply"],
    "concorrencia": ["semaglut", "ozempic", "wegovy", "mounjaro", "tirzepat", "dapagliflo", "glp-1", "lancamento"],
    "evidencia_clinica": ["estudo", "study", "trial", "pesquisa", "research", "eficacia", "mace"],
    "macroeconomia": ["dolar", "dollar", "cambio", "exchange rate", "inflacao", "inflation", "juros"],
}

PROMPT_LLM = """Voce e um especialista em analise de impacto de noticias sobre o \
mercado farmaceutico de medicamentos para diabetes (iSGLT2, DPP-4, GLP-1, insulinas).

Classifique a noticia quanto a:
- impacto no faturamento de laboratorios do segmento: ALTO, MEDIO ou BAIXO
  ALTO: pode alterar vendas no curto prazo (aprovacao regulatoria, reembolso,
  recall, ruptura de estoque, resultado decisivo de estudo, mudanca de preco).
  MEDIO: relevante ao setor sem efeito direto e imediato (pesquisa, parcerias, tendencias).
  BAIXO: mencao tangencial ou sem relacao com o faturamento.
- sentimento para o mercado/produto citado: positivo, neutro ou negativo

Responda APENAS em JSON:
{"impacto": "ALTO|MEDIO|BAIXO", "sentimento": "positivo|neutro|negativo", "justificativa": "..."}
"""


# ============================================================================
# Utilitarios
# ============================================================================
def normalizar(texto) -> str:
    if texto is None or (isinstance(texto, float) and pd.isna(texto)):
        return ""
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", t.lower()).strip()


def id_noticia(url: str) -> str:
    return hashlib.sha1(str(url).encode("utf-8")).hexdigest()[:12]


def carregar_mapa(caminho: Path = CAMINHO_MAPA) -> dict:
    if not caminho.exists():
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_text(json.dumps(MAPA_PADRAO, ensure_ascii=False, indent=2), encoding="utf-8")
        return MAPA_PADRAO
    return json.loads(caminho.read_text(encoding="utf-8"))


_CACHE_RE: dict[str, re.Pattern] = {}


def _padrao(termo: str) -> re.Pattern:
    """
    Termo curto (<= 4 letras) casa como palavra inteira ("sus" nao casa com
    "suspensao"); termo longo casa como inicio de palavra ("aprova" casa com
    "aprovacao"), evitando falsos positivos no meio de outras palavras.
    """
    if termo not in _CACHE_RE:
        t = re.escape(termo.strip())
        _CACHE_RE[termo] = re.compile(rf"\b{t}\b" if len(termo.strip()) <= 4 else rf"\b{t}")
    return _CACHE_RE[termo]


def _contem(texto: str, termos) -> list[str]:
    return [t for t in termos if _padrao(t).search(texto)]


# ============================================================================
# Rotulos do corpus anotado
# ============================================================================
_NORMALIZA_ROTULO = {"ALTO": "alto", "MEDIO": "medio", "MÉDIO": "medio", "BAIXO": "baixo"}


def rotulos_do_corpus(caminho: Path = CAMINHO_CORPUS) -> dict[str, str]:
    """url -> impacto consolidado (voto majoritario dos avaliadores)."""
    if not caminho.exists():
        return {}
    df = pd.read_excel(caminho)
    cols = [c for c in df.columns if str(c).lower().startswith("classificacao")
            and "kappa" not in str(c).lower()]
    if not cols or "url" not in df.columns:
        return {}
    saida = {}
    for r in df[["url", *cols] + (["classificacao kappa"] if "classificacao kappa" in df.columns else [])].itertuples(index=False):
        url = r[0]
        kappa = r[-1] if "classificacao kappa" in df.columns else None
        if isinstance(kappa, str) and kappa.strip():
            saida[url] = _NORMALIZA_ROTULO.get(kappa.strip().upper(), "medio")
            continue
        votos = [_NORMALIZA_ROTULO.get(str(v).strip().upper()) for v in r[1:1 + len(cols)]]
        votos = [v for v in votos if v]
        if not votos:
            continue
        cont = pd.Series(votos).value_counts()
        saida[url] = cont.index[0] if cont.iloc[0] > 1 or len(votos) == 1 else "medio"
    return saida


# ============================================================================
# Classificadores
# ============================================================================
class NaiveBayesImpacto:
    """
    Naive Bayes multinomial (presenca de termos, suavizacao de Laplace)
    treinado no corpus anotado pelos 3 avaliadores.

    E o substituto leve do Agente 1 (mBERT) enquanto o fine-tuning nao esta
    disponivel. Validacao: 5 particoes 80/20 sobre o corpus sem duplicatas
    (1.104 noticias) -> acuracia media de 67,7%, contra 48,9% da classe
    majoritaria e 46,3% das regras por palavra-chave.
    """

    CLASSES = ("alto", "medio", "baixo")

    def __init__(self):
        self.ok = False

    @staticmethod
    def _tokens(texto: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]{3,}", texto))

    def treinar(self, textos: list[str], rotulos: list[str]) -> "NaiveBayesImpacto":
        import math
        from collections import Counter
        cont = {c: Counter() for c in self.CLASSES}
        n = Counter(rotulos)
        for t, y in zip(textos, rotulos):
            cont[y].update(self._tokens(t))
        vocab = set().union(*[set(c) for c in cont.values()])
        self.v = len(vocab)
        self.tot = {c: sum(cont[c].values()) for c in self.CLASSES}
        self.cont = cont
        self.prior = {c: math.log(max(n[c], 1) / len(rotulos)) for c in self.CLASSES}
        self.ok = len(rotulos) > 50
        return self

    def prever(self, texto: str) -> str:
        import math
        toks = self._tokens(texto)
        s = {c: self.prior[c] + sum(math.log((self.cont[c][w] + 1) / (self.tot[c] + self.v))
                                    for w in toks) for c in self.CLASSES}
        return max(s, key=s.get)


def treinar_nb_do_corpus(caminho: Path = CAMINHO_CORPUS) -> NaiveBayesImpacto:
    nb = NaiveBayesImpacto()
    if not caminho.exists():
        return nb
    df = pd.read_excel(caminho)
    rot = rotulos_do_corpus(caminho)
    df["y"] = df["url"].map(rot)
    df = df.dropna(subset=["y"])
    df["t"] = [normalizar(f"{a} {b}") for a, b in zip(df["titulo"], df["descricao"])]
    df = df.drop_duplicates("t")
    return nb.treinar(df["t"].tolist(), df["y"].tolist())


def impacto_heuristico(texto: str) -> str:
    if _contem(texto, TERMOS_ALTO):
        return "alto"
    if _contem(texto, TERMOS_MEDIO):
        return "medio"
    return "baixo"


def sentimento_heuristico(texto: str) -> tuple[str, float]:
    pos = len(_contem(texto, LEXICO_POSITIVO))
    neg = len(_contem(texto, LEXICO_NEGATIVO))
    if pos + neg == 0:
        return "neutro", 0.0
    score = (pos - neg) / (pos + neg)
    if score > 0.2:
        return "positivo", round(score, 4)
    if score < -0.2:
        return "negativo", round(score, 4)
    return "neutro", round(score, 4)


def vincular_entidades(texto: str, mapa: dict) -> tuple[str, list[str], list[str]]:
    """Retorna (escopo, grupos_proprios_citados, termos_encontrados)."""
    grupos, termos = [], []
    for grupo, lista in mapa.get("proprio", {}).items():
        achou = _contem(texto, lista)
        if achou:
            grupos.append(grupo)
            termos += achou
    conc = _contem(texto, mapa.get("concorrente", []))
    merc = _contem(texto, mapa.get("mercado", []))
    termos += conc + merc
    if grupos:
        return "proprio", grupos, termos
    if conc:
        return "concorrente", [], termos
    if merc:
        return "mercado", [], termos
    return "nenhum", [], termos


def categoria(texto: str) -> str:
    melhor, n = "geral", 0
    for cat, termos in CATEGORIAS.items():
        k = len(_contem(texto, termos))
        if k > n:
            melhor, n = cat, k
    return melhor


class ClassificadorLLM:
    """Classificacao via OpenAI. So e usado com usar_llm=True e chave definida."""

    def __init__(self, modelo: str | None = None):
        from openai import OpenAI          # import tardio: dependencia opcional
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.modelo = modelo or os.environ.get("OPENAI_MODELO", "gpt-4o-mini")

    def classificar(self, titulo: str, descricao: str) -> dict:
        r = self.client.chat.completions.create(
            model=self.modelo, temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": PROMPT_LLM},
                      {"role": "user", "content": f"Titulo: {titulo}\nDescricao: {descricao}"}],
        )
        d = json.loads(r.choices[0].message.content)
        return {
            "impacto": _NORMALIZA_ROTULO.get(str(d.get("impacto", "")).upper(), "baixo"),
            "sentimento": str(d.get("sentimento", "neutro")).lower(),
            "justificativa": d.get("justificativa", ""),
        }


# ============================================================================
# Pipeline de classificacao
# ============================================================================
def classificar_noticias(df: pd.DataFrame, usar_llm: bool = False,
                         mapa: dict | None = None, log=print) -> pd.DataFrame:
    """
    df precisa ter: titulo, descricao, url. Opcionais: conteudo, data_publicacao,
    fonte_nome, palavra_chave.
    """
    mapa = mapa or carregar_mapa()
    corpus = rotulos_do_corpus()
    nb = treinar_nb_do_corpus()
    llm = None
    if usar_llm:
        if os.environ.get("OPENAI_API_KEY"):
            try:
                llm = ClassificadorLLM()
            except Exception as e:      # noqa: BLE001
                log(f"  LLM indisponivel ({e}); usando regras")
        else:
            log("  OPENAI_API_KEY nao definida; usando regras")

    linhas = []
    for r in df.itertuples(index=False):
        titulo = getattr(r, "titulo", "") or ""
        descricao = getattr(r, "descricao", "") or ""
        conteudo = getattr(r, "conteudo", "") or ""
        texto = normalizar(f"{titulo} {descricao} {conteudo}")
        escopo, grupos, termos = vincular_entidades(texto, mapa)

        sentimento, score = sentimento_heuristico(normalizar(f"{titulo} {descricao}"))
        justificativa = ""
        if r.url in corpus:
            impacto, origem = corpus[r.url], "corpus_anotado"
        elif llm is not None and escopo != "nenhum":
            try:
                c = llm.classificar(titulo, descricao)
                impacto, origem, justificativa = c["impacto"], "llm", c["justificativa"]
                if c["sentimento"] in ("positivo", "neutro", "negativo"):
                    sentimento = c["sentimento"]
            except Exception as e:      # noqa: BLE001
                log(f"  falha no LLM ({e}); usando o classificador do corpus")
                impacto, origem = nb.prever(normalizar(f"{titulo} {descricao}")), "naive_bayes_corpus"
        elif nb.ok:
            impacto, origem = nb.prever(normalizar(f"{titulo} {descricao}")), "naive_bayes_corpus"
        else:
            impacto, origem = impacto_heuristico(texto), "regras"

        if escopo == "nenhum" and origem != "corpus_anotado":
            impacto = "baixo"   # sem vinculo com o mercado nao ha impacto a cruzar

        linhas.append({
            "id_noticia": id_noticia(r.url),
            "data_publicacao": getattr(r, "data_publicacao", None),
            "fonte": getattr(r, "fonte_nome", "") or "",
            "titulo": titulo,
            "descricao": descricao,
            "url": r.url,
            "palavra_chave": getattr(r, "palavra_chave", "") or "",
            "categoria": categoria(texto),
            "nivel_impacto": impacto,
            "sentimento": sentimento,
            "score_sentimento": score,
            "escopo": escopo,
            "grupos_alvo": "|".join(grupos),
            "termos_encontrados": "|".join(sorted(set(termos)))[:300],
            "id_mercado": mapa.get("id_mercado", 1) if escopo != "nenhum" else None,
            "origem_classificacao": origem,
            "justificativa_llm": justificativa,
        })
    return pd.DataFrame(linhas)


def validar_noticias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtro de entrada do Agente 4: so ALTO/MEDIO, com vinculo com o mercado e,
    quando a base veio do Agente 2 completo (orquestrador), so as APROVADAS
    (descartadas, duplicadas, em quarentena e sem ancora ficam de fora).
    """
    ok = df["nivel_impacto"].isin(["alto", "medio"]) & (df["escopo"] != "nenhum")
    if "status_validacao" in df.columns:
        st = df["status_validacao"]
        ok &= st.isna() | (st == "APROVADO")
    return df[ok].copy()
