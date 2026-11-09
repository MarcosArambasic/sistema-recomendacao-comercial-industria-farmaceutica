"""***************************************************************************************
SCRIPT    : substitutos.py
OBJETIVO  : Motores SUBSTITUTOS dos Agentes 1 e 2, usados pelo Orquestrador
            quando as dependencias pesadas nao estao instaladas.

    Agente 1 (mBERT)  -> ClassificadorNBImpacto
        Naive Bayes multinomial treinado no corpus desempatado, com a MESMA
        interface publica do AgenteClassificadorImpacto: treinar() com
        validacao cruzada estratificada e predicoes out-of-fold,
        classificar_dataframe(), predicoes_oof_ e relatorio_treino. Tambem
        anexa entidades e direcao do impacto pelo dominio_farma, igual ao
        mBERT, para que o Agente 2 receba as mesmas colunas.

    Agente 2 (encoder sentence-transformers) -> EncoderLexical
        Embeddings TF-IDF com hashing (unigramas + bigramas), normalizados
        em L2. Tem a assinatura de SentenceTransformer.encode() e e
        injetado no AgenteValidadorAjuste pelo parametro `encoder=`, que o
        proprio agente oferece (inversao de dependencia). Todas as regras
        R1-R6 e o k-NN rodam normalmente; muda apenas a NATUREZA da
        similaridade (lexical em vez de semantica).

Por que existem: o painel e a rotina semanal precisam rodar em qualquer
maquina, inclusive sem GPU e sem baixar ~2 GB de modelos. O Orquestrador
registra em cada execucao qual motor foi usado ("motor_agente1",
"motor_agente2"), de modo que um resultado do substituto nunca seja
confundido com o do mBERT na analise.
****************************************************************************************/
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from _schema import COL_ID, COL_LABEL, COL_TEXTO, ROTULOS_CANONICOS
from dominio_farma import extrair_entidades, inferir_direcao, normalizar


# ============================================================================
# Encoder lexical (substituto do sentence-transformers)
# ============================================================================
class EncoderLexical:
    """TF-IDF com hashing em `dim` dimensoes; vetores L2-normalizados."""

    def __init__(self, dim: int = 4096, idf: dict | None = None):
        self.dim = dim
        self.idf = idf or {}

    @staticmethod
    def _termos(texto: str) -> list[str]:
        toks = re.findall(r"[a-z0-9]{3,}", normalizar(texto))
        return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]

    def _indice(self, termo: str) -> int:
        return int(hashlib.md5(termo.encode()).hexdigest()[:8], 16) % self.dim

    def ajustar_idf(self, textos: list[str]) -> "EncoderLexical":
        df_ = Counter()
        for t in textos:
            df_.update(set(self._termos(t)))
        n = max(len(textos), 1)
        self.idf = {k: math.log((1 + n) / (1 + v)) + 1 for k, v in df_.items()}
        return self

    def encode(self, textos, batch_size=64, show_progress_bar=False,
               normalize_embeddings=True, convert_to_numpy=True):
        m = np.zeros((len(textos), self.dim), dtype=np.float32)
        for i, t in enumerate(textos):
            for termo, n in Counter(self._termos(str(t))).items():
                m[i, self._indice(termo)] += (1 + math.log(n)) * self.idf.get(termo, 1.0)
        if normalize_embeddings:
            norma = np.linalg.norm(m, axis=1, keepdims=True)
            m = np.divide(m, norma, out=np.zeros_like(m), where=norma > 0)
        return m


# ============================================================================
# Classificador Naive Bayes (substituto do mBERT)
# ============================================================================
class ClassificadorNBImpacto:
    """Naive Bayes multinomial com suavizacao de Laplace e a interface do Agente 1."""

    motor = "naive_bayes"

    def __init__(self, seed: int = 42, enriquecer: bool = True):
        self.seed = seed
        self.enriquecer = enriquecer
        self.classes_: list[str] = []
        self.predicoes_oof_: pd.DataFrame | None = None
        self.relatorio_treino: dict = {}
        self._modelo = None

    # ---------------- nucleo ----------------
    @staticmethod
    def _tokens(texto: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]{3,}", normalizar(texto)))

    def _ajustar(self, textos, y):
        cont = {c: Counter() for c in self.classes_}
        n = Counter(y)
        for t, c in zip(textos, y):
            cont[c].update(self._tokens(t))
        vocab = set().union(*[set(c) for c in cont.values()])
        return {
            "cont": cont, "v": len(vocab),
            "tot": {c: sum(cont[c].values()) for c in self.classes_},
            "prior": {c: math.log(max(n[c], 1) / len(y)) for c in self.classes_},
        }

    def _probs(self, modelo, texto: str) -> np.ndarray:
        toks = self._tokens(texto)
        log = np.array([
            modelo["prior"][c] + sum(math.log((modelo["cont"][c][w] + 1) / (modelo["tot"][c] + modelo["v"]))
                                     for w in toks)
            for c in self.classes_])
        p = np.exp(log - log.max())
        return p / p.sum()

    # ---------------- interface do Agente 1 ----------------
    def treinar(self, df: pd.DataFrame, coluna_texto: str = COL_TEXTO,
                coluna_label: str = COL_LABEL, n_folds: int = 5, **_ignorado) -> dict:
        """
        Validacao cruzada estratificada (k folds) com predicoes OUT-OF-FOLD e,
        ao final, modelo de producao sobre 100% do corpus.

        Remove textos duplicados antes de particionar: o corpus tem a mesma
        materia coletada mais de uma vez, e com duplicatas o mesmo texto
        cairia no treino e na validacao, inflando a metrica.
        """
        d = df.dropna(subset=[coluna_texto, coluna_label]).copy()
        d = d[d[coluna_label].isin(ROTULOS_CANONICOS)]
        antes = len(d)
        d = d.drop_duplicates(subset=coluna_texto).reset_index(drop=True)
        self.classes_ = [c for c in ROTULOS_CANONICOS if c in set(d[coluna_label])]
        textos = d[coluna_texto].astype(str).tolist()
        y = d[coluna_label].tolist()

        rng = np.random.default_rng(self.seed)
        fold = np.zeros(len(d), dtype=int)
        for c in self.classes_:                          # estratificacao manual
            idx = np.flatnonzero(np.array(y) == c)
            rng.shuffle(idx)
            fold[idx] = np.arange(len(idx)) % n_folds + 1

        probs = np.zeros((len(d), len(self.classes_)))
        metricas_folds = []
        for k in range(1, n_folds + 1):
            tr, va = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            m = self._ajustar([textos[i] for i in tr], [y[i] for i in tr])
            for i in va:
                probs[i] = self._probs(m, textos[i])
            pred = [self.classes_[j] for j in probs[va].argmax(axis=1)]
            real = [y[i] for i in va]
            metricas_folds.append({"fold": k, "n_validacao": int(len(va)),
                                   **metricas_classificacao(real, pred, self.classes_)["resumo"]})

        pred_oof = [self.classes_[j] for j in probs.argmax(axis=1)]
        aval = metricas_classificacao(y, pred_oof, self.classes_)
        f1s = [m["f1_macro"] for m in metricas_folds]
        self.relatorio_treino = {
            "motor": self.motor, "n_documentos": len(d), "duplicatas_removidas": antes - len(d),
            "n_folds": n_folds, "metricas_por_fold": metricas_folds,
            "f1_macro_media": float(np.mean(f1s)), "f1_macro_desvio": float(np.std(f1s)),
            "acuracia_media": float(np.mean([m["acuracia"] for m in metricas_folds])),
            "out_of_fold": aval,
        }

        out = d.copy()
        out["classe_predita"] = pred_oof
        out["score"] = probs.max(axis=1)
        for j, c in enumerate(self.classes_):
            out[f"prob_{c.lower()}"] = probs[:, j]
        out["fold"] = fold
        out["origem_predicao"] = "out_of_fold"
        out["acertou"] = out["classe_predita"] == out[coluna_label]
        self.predicoes_oof_ = self._enriquecer(out, coluna_texto)

        self._modelo = self._ajustar(textos, y)          # producao
        return self.relatorio_treino

    def classificar_dataframe(self, df: pd.DataFrame, coluna_texto: str = COL_TEXTO, **_ignorado) -> pd.DataFrame:
        if self._modelo is None:
            raise RuntimeError("treine o classificador antes de classificar")
        out = df.copy()
        probs = np.array([self._probs(self._modelo, str(t)) for t in out[coluna_texto]]) \
            if len(out) else np.zeros((0, len(self.classes_)))
        out["classe_predita"] = [self.classes_[j] for j in probs.argmax(axis=1)] if len(out) else []
        out["score"] = probs.max(axis=1) if len(out) else []
        for j, c in enumerate(self.classes_):
            out[f"prob_{c.lower()}"] = probs[:, j] if len(out) else []
        out["origem_predicao"] = "modelo_final"
        if COL_ID not in out.columns:
            out[COL_ID] = [f"doc{i:06d}" for i in range(len(out))]
        return self._enriquecer(out, coluna_texto)

    def _enriquecer(self, out: pd.DataFrame, coluna_texto: str) -> pd.DataFrame:
        if not self.enriquecer or out.empty:
            return out
        col = "texto_extracao" if "texto_extracao" in out.columns else coluna_texto
        ents = [extrair_entidades(str(t)) for t in out[col]]
        dirs = [inferir_direcao(e) for e in ents]
        for campo in ents[0].to_dict():
            out[campo] = [e.to_dict()[campo] for e in ents]
        for campo in dirs[0].to_dict():
            out[campo] = [d.to_dict()[campo] for d in dirs]
        return out


# ============================================================================
# Metricas de classificacao (sem scikit-learn)
# ============================================================================
def metricas_classificacao(y, p, classes) -> dict:
    y, p = np.asarray(y, dtype=object), np.asarray(p, dtype=object)
    por_classe = {}
    for c in classes:
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        por_classe[c] = {"precisao": round(pr, 4), "recall": round(rc, 4),
                         "f1": round(2 * pr * rc / (pr + rc) if pr + rc else 0.0, 4),
                         "suporte": int((y == c).sum())}
    acc = float((y == p).mean()) if len(y) else 0.0
    pe = sum(float((y == c).mean()) * float((p == c).mean()) for c in classes) if len(y) else 0.0
    f1_macro = float(np.mean([v["f1"] for v in por_classe.values()])) if por_classe else 0.0
    return {
        "resumo": {"acuracia": round(acc, 4), "f1_macro": round(f1_macro, 4),
                   "kappa_cohen": round((acc - pe) / (1 - pe), 4) if pe < 1 else 0.0},
        "por_classe": por_classe,
        "matriz_confusao": {r: {c: int(((y == r) & (p == c)).sum()) for c in classes} for r in classes},
    }
