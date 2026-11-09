"""***************************************************************************************
SCRIPT    : agente3_oportunidades_mercado.py
OBJETIVO  : Agente 3 - Analisador de Oportunidades de Mercado.

            Treinado sobre a fato_analise_kpi_final, gera RECOMENDACOES DE
            MERCADO para cada setor x alocacao x produto proprio, a partir
            dos indicadores de mercado (market share, CAGR, CV e indice
            sazonal). A saida e um JSON que o Agente 4 usa para cruzar as
            noticias de impacto ALTO e MEDIO com as oportunidades e riscos
            que cada setor tem em cada alocacao.

O QUE O MODELO PREVE (aprendizado supervisionado):

    A variacao do market share do produto na alocacao no MES SEGUINTE:

        alvo = mrkt_share(t+1) - mrkt_share(t)

    Prever a variacao (e nao o nivel) faz a comparacao com a persistencia
    ("o share do mes que vem e igual ao deste") ser direta: o modelo so
    agrega valor se errar menos do que prever variacao zero.

DA PREVISAO A RECOMENDACAO (camada prescritiva, deterministica e auditavel):

    share_previsto   = share(t) + variacao prevista
    mercado_previsto = mercado do mesmo mes do ano anterior x crescimento
                       interanual recente (sazonal ingenuo com tendencia)
    benchmark_uf     = percentil 75 do share do produto entre as alocacoes
                       da mesma UF no mes (benchmarking interno)

    oportunidade_un  = mercado_previsto x max(0, benchmark_uf - share_previsto)
    risco_un         = mercado_previsto x max(0, share_medio_3m - share_previsto)

    Regras de tipo (limiares parametrizaveis em LIMIARES). As comparacoes
    sao RELATIVAS AOS PARES (mesmo produto/UF): a erosao estrutural, comum a
    toda a UF, nao diferencia uma alocacao de outra.
      queda prevista 10 p.p. pior que a dos pares -> DEFENDER_SHARE (share alto)
                                                    RECUPERAR_SHARE (share baixo)
      gap p/ benchmark e mercado crescendo acima
      da mediana da UF                            -> CAPTURAR_CRESCIMENTO
      gap para o benchmark                        -> AMPLIAR_PENETRACAO
      proximo mes e pico sazonal                  -> ANTECIPAR_PICO_SAZONAL
      nenhum dos anteriores                       -> MANTER_POSICAO

VAZAMENTO DE INFORMACAO: o indice_sazonal da fato_analise_kpi e calculado
sobre a media do ANO inteiro, que no historico inclui o mes t+1. Por isso o
indice do mes corrente NAO entra como variavel; entra o indice do mesmo mes
do ano anterior ao mes previsto (t-11), que so usa dados ja encerrados.

REGULARIZACAO E INTERPRETABILIDADE: XGBoost com L1/L2 (reg_alpha/reg_lambda)
e contribuicoes SHAP exatas por arvore (TreeSHAP, via pred_contribs do
proprio XGBoost - nao depende da biblioteca shap).

AUTOR     : Marcos Arambasic
DATA      : OUT/2026
****************************************************************************************/
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    XGBOOST_DISPONIVEL = True
except ImportError:          # pragma: no cover - depende do ambiente
    XGBOOST_DISPONIVEL = False


AGENTE = "agente3_oportunidades_mercado"
VERSAO_SCHEMA = "2.0"

# ============================================================================
# Configuracao
# ============================================================================
FEATURES: dict[str, str] = {
    "mrkt_share": "market share do produto na alocacao no mes",
    "cd_mrkt_share": "faixa do market share (dim_classifica_kpis)",
    "share_media_3m": "market share medio dos ultimos 3 meses",
    "share_media_12m": "market share medio dos ultimos 12 meses",
    "delta_share_1m": "variacao do share contra o mes anterior",
    "delta_share_3m": "variacao do share contra 3 meses atras",
    "share_benchmark_uf": "percentil 75 do share do produto na UF",
    "gap_benchmark": "distancia do share recente ao benchmark da UF",
    "share_proprios_alocacao": "share somado dos produtos proprios na alocacao",
    "cagr_medio": "CAGR do produto na UF",
    "cd_cagr_medio": "faixa do CAGR",
    "cv": "coeficiente de variacao (12 meses) do produto na UF",
    "cd_cv": "faixa do CV",
    "indice_sazonal_previsto": "indice sazonal do mes previsto no ano anterior",
    "cd_indice_sazonal_previsto": "faixa do indice sazonal do mes previsto",
    "mercado_log": "tamanho do mercado na alocacao (log)",
    "mercado_var_12m": "crescimento interanual do mercado (media 3m)",
    "mes_alvo": "mes do calendario previsto",
    "id_produto": "produto",
    "id_uf": "UF",
}
ALVO = "delta_share_prox"

HIPERPARAMETROS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.5,      # L1
    "reg_lambda": 1.0,     # L2
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}

LIMIARES = {
    "erosao_relativa": 0.10,        # queda prevista 10 p.p. pior que a mediana dos pares (produto x UF)
    "unidades_minimas": 5.0,        # abaixo disso, oportunidade/risco e ruido
    "crescimento_mercado": 0.05,    # mercado +5% a/a -> mercado "crescendo"
    "percentil_benchmark": 0.75,
    "fracao_alta": 0.10,            # 10% maiores valores em jogo -> prioridade alta
    "fracao_media": 0.30,           # proximos 20% -> media
}

TIPOS = {
    "CAPTURAR_CRESCIMENTO": "oportunidade",
    "AMPLIAR_PENETRACAO": "oportunidade",
    "ANTECIPAR_PICO_SAZONAL": "oportunidade",
    "DEFENDER_SHARE": "risco",
    "RECUPERAR_SHARE": "risco",
    "MANTER_POSICAO": "neutro",
}
PRIORIDADES = ("alta", "media", "baixa")

# Como o Agente 4 deve ler uma noticia contra cada tipo de recomendacao
REGRAS_SENSIBILIDADE = {
    "CAPTURAR_CRESCIMENTO": {
        "noticia_positiva_produto_ou_mercado": "amplifica a oportunidade",
        "noticia_negativa_produto": "atenua - rever antes de investir esforco",
        "noticia_positiva_concorrente": "aumenta a urgencia (janela pode fechar)",
    },
    "AMPLIAR_PENETRACAO": {
        "noticia_positiva_produto_ou_mercado": "amplifica a oportunidade",
        "noticia_negativa_produto": "atenua",
        "noticia_negativa_concorrente": "amplifica (espaco para ganhar share)",
    },
    "ANTECIPAR_PICO_SAZONAL": {
        "noticia_ruptura_ou_abastecimento": "eleva para prioridade alta",
        "noticia_positiva_mercado": "amplifica",
    },
    "DEFENDER_SHARE": {
        "noticia_positiva_concorrente": "confirma o risco - prioridade alta",
        "noticia_negativa_produto": "agrava o risco",
        "noticia_negativa_concorrente": "atenua o risco",
    },
    "RECUPERAR_SHARE": {
        "noticia_negativa_concorrente": "abre janela de recuperacao",
        "noticia_positiva_concorrente": "agrava o risco",
    },
    "MANTER_POSICAO": {
        "qualquer_noticia_alto_impacto": "reavaliar - pode mudar o tipo da recomendacao",
    },
}

CHAVES_ID = ["id_produto", "id_alocacao"]


# ============================================================================
# Utilitarios de matriz (painel completo produto x alocacao x mes)
# ============================================================================
def _lag(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype="float64")
    if k > 0:
        out[:, k:] = a[:, :-k]
    else:
        out[:, :k] = a[:, -k:]
    return out


def _media_movel(a: np.ndarray, janela: int, minimo: int) -> np.ndarray:
    """Media movel ate t (inclusive), ignorando NaN, com minimo de observacoes."""
    v = np.nan_to_num(a, nan=0.0)
    n = (~np.isnan(a)).astype("float64")
    cs_v = np.cumsum(v, axis=1)
    cs_n = np.cumsum(n, axis=1)
    soma = cs_v - np.nan_to_num(_lag(cs_v, janela), nan=0.0)
    cont = cs_n - np.nan_to_num(_lag(cs_n, janela), nan=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = soma / cont
    m[cont < minimo] = np.nan
    return m


# ============================================================================
# Preparacao do dataset
# ============================================================================
@dataclass
class Painel:
    dados: pd.DataFrame            # longo, uma linha por produto x alocacao x mes
    meses: list
    concorrentes: pd.DataFrame     # share dos concorrentes por alocacao x mes


def montar_dataset(df_kpi_final: pd.DataFrame, df_produto: pd.DataFrame) -> Painel:
    """
    Monta o painel completo (produto proprio x alocacao x mes) com variaveis
    e alvo. Meses em que o produto nao vendeu na alocacao entram com share 0:
    ausencia de venda e informacao (e onde costuma estar a oportunidade).
    """
    obrig = {"periodo_demanda", "id_produto", "id_setor", "id_alocacao", "id_outlet",
             "id_uf", "id_mercado", "grupo", "demanda_produto", "demanda_mercado",
             "mrkt_share", "cagr_medio", "cv", "indice_sazonal",
             "cd_mrkt_share", "cd_cagr_medio", "cd_cv", "cd_indice_sazonal"}
    faltando = obrig - set(df_kpi_final.columns)
    if faltando:
        raise ValueError(f"fato_analise_kpi_final sem as colunas: {sorted(faltando)}")

    d = df_kpi_final.copy()
    d["periodo_demanda"] = pd.to_datetime(d["periodo_demanda"])
    proprios = set(df_produto.loc[df_produto["tipo_produto"].str.upper() == "PROPRIO",
                                  "id_produto"])
    if not proprios:
        raise ValueError("dim_produto sem produtos PROPRIO")

    meses = sorted(d["periodo_demanda"].unique())
    idx_mes = {m: i for i, m in enumerate(meses)}
    M = len(meses)

    p = d[d["id_produto"].isin(proprios)]
    conc = d[~d["id_produto"].isin(proprios)][
        ["periodo_demanda", "id_alocacao", "id_produto", "mrkt_share"]]

    pares = (p.groupby(CHAVES_ID, as_index=False)
              .agg(id_setor=("id_setor", "first"), id_outlet=("id_outlet", "first"),
                   id_uf=("id_uf", "first"), id_mercado=("id_mercado", "first"),
                   grupo=("grupo", "first"))
              .sort_values(CHAVES_ID).reset_index(drop=True))
    P = len(pares)
    pos_par = pd.Series(np.arange(P),
                        index=pd.MultiIndex.from_frame(pares[CHAVES_ID]))

    ip = pos_par.reindex(pd.MultiIndex.from_frame(p[CHAVES_ID])).to_numpy()
    im = p["periodo_demanda"].map(idx_mes).to_numpy()

    def matriz(col: str, vazio: float) -> np.ndarray:
        a = np.full((P, M), vazio, dtype="float64")
        a[ip, im] = p[col].astype("float64").to_numpy()
        return a

    S = matriz("mrkt_share", 0.0)
    cd_S = matriz("cd_mrkt_share", np.nan)
    IS = matriz("indice_sazonal", np.nan)
    cd_IS = matriz("cd_indice_sazonal", np.nan)

    # cd do share em meses sem venda: menor faixa observada
    cd_S[np.isnan(cd_S)] = np.nanmin(cd_S) if np.isfinite(np.nanmin(cd_S)) else 1

    # mercado: nivel da alocacao (igual para todos os produtos)
    merc = d.groupby(["id_alocacao", "periodo_demanda"])["demanda_mercado"].max()
    chave_merc = pd.MultiIndex.from_arrays([
        np.repeat(pares["id_alocacao"].to_numpy(), M),
        np.tile(np.array(meses), P),
    ])
    MK = merc.reindex(chave_merc).to_numpy(dtype="float64").reshape(P, M)
    MK = np.nan_to_num(MK, nan=0.0)

    # CAGR e CV vivem em produto x UF: preenche meses sem venda pela UF
    uf_tab = (p.groupby(["id_produto", "id_uf", "periodo_demanda"])
               [["cagr_medio", "cd_cagr_medio", "cv", "cd_cv"]].first())
    chave_uf = pd.MultiIndex.from_arrays([
        np.repeat(pares["id_produto"].to_numpy(), M),
        np.repeat(pares["id_uf"].to_numpy(), M),
        np.tile(np.array(meses), P),
    ])
    uf_m = uf_tab.reindex(chave_uf)
    CAGR = uf_m["cagr_medio"].to_numpy("float64").reshape(P, M)
    cd_CAGR = uf_m["cd_cagr_medio"].to_numpy("float64").reshape(P, M)
    CV = uf_m["cv"].to_numpy("float64").reshape(P, M)
    cd_CV = uf_m["cd_cv"].to_numpy("float64").reshape(P, M)

    # ---- variaveis de historico (so passado) ------------------------------
    S3 = _media_movel(S, 3, 1)
    S12 = _media_movel(S, 12, 3)
    MK3 = _media_movel(MK, 3, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        var12 = MK3 / _lag(MK3, 12) - 1.0
    var12[~np.isfinite(var12)] = np.nan

    # mercado previsto t+1: mesmo mes do ano anterior x crescimento a/a recente
    base_sazonal = _lag(MK, 11)
    mk_prev = base_sazonal * (1.0 + np.nan_to_num(var12, nan=0.0))
    mk_prev = np.where(np.isnan(mk_prev), MK3, mk_prev)

    mes_num = np.array([pd.Timestamp(m).month for m in meses])
    mes_alvo = np.tile(mes_num % 12 + 1, (P, 1)).astype("float64")

    longo = pd.DataFrame({
        "periodo_demanda": np.tile(np.array(meses), P),
        "id_produto": np.repeat(pares["id_produto"].to_numpy(), M),
        "id_alocacao": np.repeat(pares["id_alocacao"].to_numpy(), M),
        "id_setor": np.repeat(pares["id_setor"].to_numpy(), M),
        "id_outlet": np.repeat(pares["id_outlet"].to_numpy(), M),
        "id_uf": np.repeat(pares["id_uf"].to_numpy(), M),
        "id_mercado": np.repeat(pares["id_mercado"].to_numpy(), M),
        "grupo": np.repeat(pares["grupo"].to_numpy(), M),
        "mrkt_share": S.ravel(),
        "cd_mrkt_share": cd_S.ravel(),
        "share_media_3m": S3.ravel(),
        "share_media_12m": S12.ravel(),
        "delta_share_1m": (S - _lag(S, 1)).ravel(),
        "delta_share_3m": (S - _lag(S, 3)).ravel(),
        "cagr_medio": CAGR.ravel(),
        "cd_cagr_medio": cd_CAGR.ravel(),
        "cv": CV.ravel(),
        "cd_cv": cd_CV.ravel(),
        "indice_sazonal_previsto": _lag(IS, 11).ravel(),
        "cd_indice_sazonal_previsto": _lag(cd_IS, 11).ravel(),
        "demanda_mercado": MK.ravel(),
        "mercado_log": np.log1p(MK).ravel(),
        "mercado_var_12m": var12.ravel(),
        "mercado_previsto": mk_prev.ravel(),
        "mercado_real_prox": _lag(MK, -1).ravel(),
        "mes_alvo": mes_alvo.ravel(),
        "share_prox": _lag(S, -1).ravel(),
    })
    longo[ALVO] = longo["share_prox"] - longo["mrkt_share"]

    # benchmark interno: P75 do share do produto entre alocacoes da UF
    q = LIMIARES["percentil_benchmark"]
    longo["share_benchmark_uf"] = (
        longo.groupby(["periodo_demanda", "id_produto", "id_uf"])["mrkt_share"]
        .transform(lambda x: x.quantile(q))
    )
    longo["gap_benchmark"] = longo["share_benchmark_uf"] - longo["share_media_3m"]
    longo["share_proprios_alocacao"] = (
        longo.groupby(["periodo_demanda", "id_alocacao"])["mrkt_share"].transform("sum")
    )
    return Painel(dados=longo, meses=[pd.Timestamp(m) for m in meses], concorrentes=conc)


def dividir_treino_teste(painel: Painel, meses_teste: int = 3) -> tuple:
    """
    Split TEMPORAL: os ultimos `meses_teste` meses que tem alvo conhecido vao
    para teste. O ultimo mes da base nao tem t+1 e fica para a inferencia.
    """
    rot = painel.dados.dropna(subset=[ALVO])
    meses = sorted(rot["periodo_demanda"].unique())
    if len(meses) <= meses_teste + 12:
        raise ValueError(f"historico insuficiente: {len(meses)} meses rotulados")
    corte = meses[-meses_teste]
    treino = rot[rot["periodo_demanda"] < corte]
    teste = rot[rot["periodo_demanda"] >= corte]
    return treino, teste, pd.Timestamp(corte)


# ============================================================================
# Modelo
# ============================================================================
class _RidgeFallback:
    """
    Regressao ridge em numpy, usada só quando o xgboost nao esta instalado,
    para o contrato de saida poder ser exercitado. 
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.mediana_ = np.nanmedian(X, axis=0)
        self.mediana_ = np.where(np.isnan(self.mediana_), 0.0, self.mediana_)
        Xi = self._imputar(X)
        self.media_ = Xi.mean(axis=0)
        self.escala_ = Xi.std(axis=0)
        self.escala_[self.escala_ == 0] = 1.0
        Xn = (Xi - self.media_) / self.escala_
        A = Xn.T @ Xn + self.alpha * np.eye(Xn.shape[1])
        self.coef_ = np.linalg.solve(A, Xn.T @ (y - y.mean()))
        self.intercepto_ = float(y.mean())
        return self

    def _imputar(self, X):
        return np.where(np.isnan(X), self.mediana_, X)

    def predict(self, X):
        return ((self._imputar(X) - self.media_) / self.escala_) @ self.coef_ + self.intercepto_

    def contribuicoes(self, X):
        return ((self._imputar(X) - self.media_) / self.escala_) * self.coef_


def _metricas(y: np.ndarray, p: np.ndarray) -> dict:
    e = p - y
    ss_res, ss_tot = float(np.sum(e ** 2)), float(np.sum((y - y.mean()) ** 2))
    return {
        "mae": round(float(np.mean(np.abs(e))), 6),
        "rmse": round(float(np.sqrt(np.mean(e ** 2))), 6),
        "r2": round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else 0.0,
    }


def _wape(real: np.ndarray, prev: np.ndarray) -> float:
    ok = ~np.isnan(real) & ~np.isnan(prev)
    den = np.sum(np.abs(real[ok]))
    return round(float(np.sum(np.abs(real[ok] - prev[ok])) / den), 4) if den > 0 else float("nan")


@dataclass
class ResultadoTreino:
    algoritmo: str
    hiperparametros: dict
    n_treino: int
    n_teste: int
    periodo_treino: tuple
    periodo_teste: tuple
    metricas: dict
    baselines: dict
    ganho_sobre_baseline_pct: float
    supera_baseline: bool
    previsao_mercado: dict
    importancia: list = field(default_factory=list)


class AgenteOportunidadesMercado:
    """Agente 3 - recomendacoes de mercado por setor x alocacao x produto."""

    def __init__(self, hiperparametros: dict | None = None, forcar_fallback: bool = False):
        self.hiper = {**HIPERPARAMETROS, **(hiperparametros or {})}
        self.usar_xgb = XGBOOST_DISPONIVEL and not forcar_fallback
        self.modelo = None
        self.resultado: ResultadoTreino | None = None
        self.painel: Painel | None = None

    # ------------------------------------------------------------------
    def _novo_modelo(self):
        if self.usar_xgb:
            return xgb.XGBRegressor(objective="reg:squarederror", **self.hiper)
        return _RidgeFallback()

    @staticmethod
    def _X(df: pd.DataFrame) -> np.ndarray:
        return df[list(FEATURES)].to_numpy(dtype="float64")

    def _contribuicoes(self, X: np.ndarray) -> np.ndarray:
        """Contribuicao de cada variavel na previsao (sem a coluna de vies)."""
        if self.usar_xgb:
            c = self.modelo.get_booster().predict(
                xgb.DMatrix(X, missing=np.nan), pred_contribs=True)
            return c[:, :-1]
        return self.modelo.contribuicoes(X)

    # ------------------------------------------------------------------
    def treinar(self, painel: Painel, meses_teste: int = 3,
                reajustar_com_tudo: bool = True) -> ResultadoTreino:
        self.painel = painel
        treino, teste, corte = dividir_treino_teste(painel, meses_teste)

        # exige 12 meses de historico para a linha entrar no treino
        treino = treino[treino["share_media_12m"].notna()]

        modelo = self._novo_modelo()
        modelo.fit(self._X(treino), treino[ALVO].to_numpy())
        self.modelo = modelo

        y = teste[ALVO].to_numpy()
        prev = modelo.predict(self._X(teste))
        share_real = teste["share_prox"].to_numpy()
        share_t = teste["mrkt_share"].to_numpy()

        m_modelo = _metricas(share_real, np.clip(share_t + prev, 0, 1))
        m_persist = _metricas(share_real, share_t)
        m_media3 = _metricas(share_real, teste["share_media_3m"].to_numpy())
        melhor_base = min(m_persist["mae"], m_media3["mae"])
        ganho = round((1 - m_modelo["mae"] / melhor_base) * 100, 2) if melhor_base > 0 else 0.0

        # direcao: acerto do sinal da variacao quando ela e relevante
        relevante = np.abs(y) > 0.005
        acerto_dir = float(np.mean(np.sign(prev[relevante]) == np.sign(y[relevante]))) \
            if relevante.any() else float("nan")
        m_modelo["acerto_direcao_variacoes_relevantes"] = round(acerto_dir, 4)

        prev_merc = {
            "metodo": "sazonal ingenuo com crescimento interanual (t-11 x var. 12m)",
            "wape": _wape(teste["mercado_real_prox"].to_numpy(),
                          teste["mercado_previsto"].to_numpy()),
            "wape_ingenuo_mes_atual": _wape(teste["mercado_real_prox"].to_numpy(),
                                            teste["demanda_mercado"].to_numpy()),
        }

        contrib = self._contribuicoes(self._X(teste))
        imp = np.mean(np.abs(contrib), axis=0)
        importancia = sorted(
            [{"variavel": f, "descricao": FEATURES[f], "shap_medio_abs": round(float(v), 8)}
             for f, v in zip(FEATURES, imp)],
            key=lambda r: -r["shap_medio_abs"],
        )

        self.resultado = ResultadoTreino(
            algoritmo="xgboost" if self.usar_xgb else "ridge_fallback",
            hiperparametros=self.hiper if self.usar_xgb else {"alpha": 1.0},
            n_treino=int(len(treino)), n_teste=int(len(teste)),
            periodo_treino=(f"{treino.periodo_demanda.min():%Y-%m}",
                            f"{treino.periodo_demanda.max():%Y-%m}"),
            periodo_teste=(f"{corte:%Y-%m}", f"{teste.periodo_demanda.max():%Y-%m}"),
            metricas=m_modelo,
            baselines={"persistencia": m_persist, "media_3m": m_media3},
            ganho_sobre_baseline_pct=ganho,
            supera_baseline=bool(ganho > 0),
            previsao_mercado=prev_merc,
            importancia=importancia,
        )

        if reajustar_com_tudo:
            # producao: o modelo final usa tudo o que tem rotulo
            tudo = painel.dados.dropna(subset=[ALVO])
            tudo = tudo[tudo["share_media_12m"].notna()]
            self.modelo = self._novo_modelo().fit(self._X(tudo), tudo[ALVO].to_numpy())
        return self.resultado

    # ------------------------------------------------------------------
    def recomendar(self, periodo: str | None = None) -> pd.DataFrame:
        """
        Gera as recomendacoes para o mes seguinte ao `periodo` (AAAA-MM).
        Default: ultimo mes da base.
        """
        if self.modelo is None:
            raise RuntimeError("treine o agente antes de recomendar")
        d = self.painel.dados
        ref = pd.Timestamp(periodo + "-01") if periodo else max(self.painel.meses)
        a = d[d["periodo_demanda"] == ref].copy()
        if a.empty:
            raise ValueError(f"periodo {ref:%Y-%m} nao existe na fato_analise_kpi_final")

        X = self._X(a)
        a["delta_previsto"] = self.modelo.predict(X)
        contrib = self._contribuicoes(X)

        L = LIMIARES
        a["share_previsto"] = np.clip(a["mrkt_share"] + a["delta_previsto"], 0, 1)
        mk = a["mercado_previsto"].fillna(a["demanda_mercado"]).clip(lower=0)
        a["mercado_previsto"] = mk
        a["oportunidade_un"] = mk * (a["share_benchmark_uf"] - a["share_previsto"]).clip(lower=0)
        a["risco_un"] = mk * (a["share_media_3m"] - a["share_previsto"]).clip(lower=0)

        # Erosao ANORMAL: variacao prevista pior que a dos pares (mesmo produto,
        # mesma UF). A erosao estrutural, comum a toda a UF, nao diferencia uma
        # alocacao de outra - ela aparece no resumo do setor, nao como acao local.
        with np.errstate(divide="ignore", invalid="ignore"):
            a["variacao_relativa"] = np.where(
                a["share_media_3m"] > 0, a["share_previsto"] / a["share_media_3m"] - 1, 0.0)
        a["variacao_relativa_pares"] = (
            a.groupby(["id_produto", "id_uf"])["variacao_relativa"].transform("median"))
        erosao = ((a["variacao_relativa"] < a["variacao_relativa_pares"] - L["erosao_relativa"])
                  & (a["risco_un"] >= L["unidades_minimas"]))
        cd_max_share = np.nanmax(a["cd_mrkt_share"])
        cd_max_saz = np.nanmax(a["cd_indice_sazonal_previsto"]) \
            if a["cd_indice_sazonal_previsto"].notna().any() else np.inf
        gap = a["oportunidade_un"] >= L["unidades_minimas"]
        # mercado da alocacao crescendo (>5% a/a) E acima da mediana da UF:
        # e onde o crescimento do mercado esta concentrado
        merc_uf = a.groupby("id_uf")["mercado_var_12m"].transform("median")
        crescendo = (a["mercado_var_12m"] > L["crescimento_mercado"]) & (a["mercado_var_12m"] > merc_uf)
        pico = (a["cd_indice_sazonal_previsto"] >= cd_max_saz) & (
            a["share_previsto"] * a["mercado_previsto"] * (a["indice_sazonal_previsto"] - 1)
            >= L["unidades_minimas"])

        a["tipo_recomendacao"] = np.select(
            [erosao & (a["cd_mrkt_share"] >= cd_max_share - 1), erosao,
             gap & crescendo, gap, pico],
            ["DEFENDER_SHARE", "RECUPERAR_SHARE",
             "CAPTURAR_CRESCIMENTO", "AMPLIAR_PENETRACAO", "ANTECIPAR_PICO_SAZONAL"],
            default="MANTER_POSICAO",
        )
        a["natureza"] = a["tipo_recomendacao"].map(TIPOS)
        a["alerta_volatilidade"] = a["cd_cv"] >= np.nanmax(a["cd_cv"])

        # pico sazonal: o valor em jogo e o volume extra esperado
        extra_saz = mk * a["share_previsto"] * (a["indice_sazonal_previsto"].fillna(1) - 1).clip(lower=0)
        a["valor_em_jogo_un"] = np.where(a["tipo_recomendacao"] == "ANTECIPAR_PICO_SAZONAL",
                                         extra_saz, a["oportunidade_un"] + a["risco_un"])
        a.loc[a["natureza"] == "neutro", "valor_em_jogo_un"] = 0.0

        ativo = a["natureza"] != "neutro"
        a["score_prioridade"] = 0.0
        a.loc[ativo, "score_prioridade"] = a.loc[ativo, "valor_em_jogo_un"].rank(pct=True)
        a["prioridade"] = np.select(
            [a["score_prioridade"] > 1 - L["fracao_alta"],
             a["score_prioridade"] > 1 - L["fracao_media"]],
            ["alta", "media"], default="baixa")

        # fatores explicativos (top 3 por |SHAP|)
        nomes = list(FEATURES)
        ordem = np.argsort(-np.abs(contrib), axis=1)[:, :3]
        a["fatores"] = [
            [{"variavel": nomes[j], "descricao": FEATURES[nomes[j]],
              "contribuicao_pp": round(float(contrib[i, j]) * 100, 4),
              "efeito": "eleva o share previsto" if contrib[i, j] > 0 else "reduz o share previsto"}
             for j in ordem[i]]
            for i in range(len(a))
        ]

        # concorrentes mais fortes na alocacao (para cruzar noticias de concorrentes)
        c = self.painel.concorrentes
        c = c[c["periodo_demanda"] == ref].sort_values("mrkt_share", ascending=False)
        top = c.groupby("id_alocacao").head(3).groupby("id_alocacao").apply(
            lambda g: [{"id_produto": int(r.id_produto), "mrkt_share": round(float(r.mrkt_share), 4)}
                       for r in g.itertuples()], include_groups=False)
        a["concorrentes"] = a["id_alocacao"].map(top).apply(lambda v: v if isinstance(v, list) else [])

        a["acao_sugerida"] = [_texto_acao(r) for r in a.itertuples()]
        a["periodo_referencia"] = f"{ref:%Y-%m}"
        a["periodo_previsto"] = f"{ref + pd.DateOffset(months=1):%Y-%m}"
        return a.reset_index(drop=True)

    # ------------------------------------------------------------------
    def exportar_json(self, recs: pd.DataFrame, top_por_setor: int = 20,
                      incluir_neutros: bool = False) -> dict:
        r = self.resultado
        itens_df = recs if incluir_neutros else recs[recs["natureza"] != "neutro"]
        avisos = []
        if r.algoritmo != "xgboost":
            avisos.append("xgboost nao instalado - resultados do ridge_fallback NAO "
                          "valem como resultado do TCC (pip install xgboost)")
        if not r.supera_baseline:
            avisos.append("o modelo nao superou a melhor baseline no teste temporal")

        setores = []
        for id_setor, g in itens_df.groupby("id_setor"):
            g = g.sort_values("valor_em_jogo_un", ascending=False)
            setores.append({
                "id_setor": int(id_setor),
                "id_uf": int(g["id_uf"].iloc[0]),
                "oportunidade_total_un": round(float(g["oportunidade_un"].sum()), 2),
                "risco_total_un": round(float(g["risco_un"].sum()), 2),
                "n_alocacoes_com_acao": int(g["id_alocacao"].nunique()),
                "n_itens": int(len(g)),
                "itens": [_item(x) for x in g.head(top_por_setor).itertuples()],
            })
        setores.sort(key=lambda s: -(s["oportunidade_total_un"] + s["risco_total_un"]))

        return {
            "schema_version": VERSAO_SCHEMA,
            "agente": AGENTE,
            "gerado_em": datetime.now().isoformat(timespec="seconds"),
            "status": "ok" if not avisos else "alerta",
            "periodo_referencia": recs["periodo_referencia"].iloc[0],
            "periodo_previsto": recs["periodo_previsto"].iloc[0],
            "fonte_treino": {
                "tabela": "fato_analise_kpi_final",
                "linhas_painel": int(len(self.painel.dados)),
                "periodo": [f"{min(self.painel.meses):%Y-%m}", f"{max(self.painel.meses):%Y-%m}"],
            },
            "modelo": {
                "algoritmo": r.algoritmo,
                "alvo": "variacao do market share no mes seguinte (produto x alocacao)",
                "hiperparametros": r.hiperparametros,
                "n_treino": r.n_treino, "n_teste": r.n_teste,
                "periodo_treino": list(r.periodo_treino),
                "periodo_teste": list(r.periodo_teste),
                "metricas_teste": r.metricas,
                "baselines": r.baselines,
                "ganho_sobre_baseline_pct": r.ganho_sobre_baseline_pct,
                "supera_baseline": r.supera_baseline,
                "previsao_mercado": r.previsao_mercado,
                "importancia_global": r.importancia,
            },
            "instrucoes_agente4": {
                "filtro_noticias": {"nivel_impacto": ["alto", "medio"]},
                "chaves_de_cruzamento": [
                    "noticia.id_produto == item.chaves_noticia.id_produto (produto proprio)",
                    "noticia.id_produto in item.chaves_noticia.ids_concorrentes (concorrente forte na alocacao)",
                    "noticia.id_mercado == item.chaves_noticia.id_mercado (noticia de mercado/classe)",
                ],
                "regras_de_sensibilidade": REGRAS_SENSIBILIDADE,
                "limiares": LIMIARES,
            },
            "resumo": {
                "n_itens": int(len(itens_df)),
                "n_itens_no_json": int(sum(len(s["itens"]) for s in setores)),
                "por_tipo": {k: int(v) for k, v in itens_df["tipo_recomendacao"].value_counts().items()},
                "por_prioridade": {k: int(v) for k, v in itens_df["prioridade"].value_counts().items()},
                "oportunidade_total_un": round(float(itens_df["oportunidade_un"].sum()), 2),
                "risco_total_un": round(float(itens_df["risco_un"].sum()), 2),
            },
            "setores": setores,
            "avisos": avisos,
        }


# ============================================================================
# Texto e serializacao
# ============================================================================
def _pct(x) -> str:
    return f"{float(x) * 100:.1f}%"


def _texto_acao(r) -> str:
    t = r.tipo_recomendacao
    un_op = f"{r.oportunidade_un:,.0f}".replace(",", ".")
    un_ri = f"{r.risco_un:,.0f}".replace(",", ".")
    if t == "CAPTURAR_CRESCIMENTO":
        return (f"Mercado em expansao e share de {_pct(r.share_previsto)} abaixo do benchmark da UF "
                f"({_pct(r.share_benchmark_uf)}): intensificar visitacao e presenca no ponto de venda "
                f"para capturar ~{un_op} unidades no proximo mes.")
    if t == "AMPLIAR_PENETRACAO":
        return (f"Share de {_pct(r.share_previsto)} abaixo do benchmark da UF ({_pct(r.share_benchmark_uf)}) "
                f"em mercado estavel: ampliar cobertura de prescritores (~{un_op} unidades de potencial).")
    if t == "DEFENDER_SHARE":
        return (f"Posicao forte com erosao prevista para {_pct(r.share_previsto)} "
                f"(media 3m {_pct(r.share_media_3m)}): reforcar relacionamento com prescritores e "
                f"monitorar concorrentes (~{un_ri} unidades em risco).")
    if t == "RECUPERAR_SHARE":
        return (f"Share baixo e em queda (previsto {_pct(r.share_previsto)}): rever abordagem na "
                f"alocacao antes de ampliar esforco (~{un_ri} unidades em risco).")
    if t == "ANTECIPAR_PICO_SAZONAL":
        return (f"Mes previsto historicamente acima da media (indice {r.indice_sazonal_previsto:.2f}): "
                f"garantir abastecimento e antecipar visitas.")
    return "Sem oportunidade ou risco relevante previstos: manter a rotina de visitacao."


def _f(x, casas=6):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), casas)


def _i(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else int(x)


def _item(r) -> dict:
    ids_conc = [c["id_produto"] for c in r.concorrentes]
    grupo = None if pd.isna(r.grupo) else str(r.grupo)
    return {
        "id_item": f"{r.periodo_previsto}|S{int(r.id_setor)}|A{int(r.id_alocacao)}|P{int(r.id_produto)}",
        "id_setor": int(r.id_setor), "id_alocacao": int(r.id_alocacao),
        "id_outlet": int(r.id_outlet), "id_uf": int(r.id_uf),
        "id_produto": int(r.id_produto), "id_mercado": int(r.id_mercado), "grupo": grupo,
        "tipo_recomendacao": r.tipo_recomendacao, "natureza": r.natureza,
        "prioridade": r.prioridade, "score_prioridade": _f(r.score_prioridade, 4),
        "alerta_volatilidade": bool(r.alerta_volatilidade),
        "acao_sugerida": r.acao_sugerida,
        "indicadores": {
            "mrkt_share": _f(r.mrkt_share), "cd_mrkt_share": _i(r.cd_mrkt_share),
            "cagr_medio": _f(r.cagr_medio), "cd_cagr_medio": _i(r.cd_cagr_medio),
            "cv": _f(r.cv), "cd_cv": _i(r.cd_cv),
            "indice_sazonal_previsto": _f(r.indice_sazonal_previsto),
            "cd_indice_sazonal_previsto": _i(r.cd_indice_sazonal_previsto),
            "share_benchmark_uf": _f(r.share_benchmark_uf),
        },
        "previsao": {
            "share_atual": _f(r.mrkt_share), "share_media_3m": _f(r.share_media_3m),
            "share_previsto": _f(r.share_previsto),
            "variacao_pp": _f((r.share_previsto - r.share_media_3m) * 100, 4),
            "mercado_previsto_un": _f(r.mercado_previsto, 2),
            "oportunidade_un": _f(r.oportunidade_un, 2), "risco_un": _f(r.risco_un, 2),
            "valor_em_jogo_un": _f(r.valor_em_jogo_un, 2),
        },
        "fatores_explicativos": r.fatores,
        "concorrentes_relevantes": r.concorrentes,
        "chaves_noticia": {
            "id_produto": int(r.id_produto), "id_mercado": int(r.id_mercado),
            "grupo": grupo, "id_uf": int(r.id_uf), "ids_concorrentes": ids_conc,
        },
    }


# ============================================================================
# Validacao do contrato (para o Orquestrador / Agente 4)
# ============================================================================
def validar_saida(saida: dict) -> list[str]:
    """Retorna a lista de erros do contrato; lista vazia = valido."""
    erros: list[str] = []
    obrig = ["schema_version", "agente", "gerado_em", "status", "periodo_referencia",
             "periodo_previsto", "fonte_treino", "modelo", "instrucoes_agente4",
             "resumo", "setores", "avisos"]
    for k in obrig:
        if k not in saida:
            erros.append(f"campo obrigatorio ausente: {k}")
    if erros:
        return erros
    if saida["schema_version"] != VERSAO_SCHEMA:
        erros.append(f"schema_version {saida['schema_version']} != {VERSAO_SCHEMA}")
    if saida["agente"] != AGENTE:
        erros.append("agente invalido")
    if saida["status"] not in ("ok", "alerta", "erro"):
        erros.append("status invalido")
    for k in ("periodo_referencia", "periodo_previsto"):
        try:
            datetime.strptime(saida[k], "%Y-%m")
        except (TypeError, ValueError):
            erros.append(f"{k} fora do formato AAAA-MM")
    if not isinstance(saida["setores"], list):
        return erros + ["setores deve ser lista"]

    campos_item = ["id_item", "id_setor", "id_alocacao", "id_produto", "id_mercado",
                   "tipo_recomendacao", "natureza", "prioridade", "acao_sugerida",
                   "indicadores", "previsao", "fatores_explicativos", "chaves_noticia"]
    n = 0
    for s in saida["setores"]:
        for k in ("id_setor", "itens", "oportunidade_total_un", "risco_total_un"):
            if k not in s:
                erros.append(f"setor sem {k}")
        for it in s.get("itens", []):
            n += 1
            for k in campos_item:
                if k not in it:
                    erros.append(f"{it.get('id_item', '?')}: sem {k}")
            if it.get("tipo_recomendacao") not in TIPOS:
                erros.append(f"{it.get('id_item')}: tipo invalido")
            elif it.get("natureza") != TIPOS[it["tipo_recomendacao"]]:
                erros.append(f"{it.get('id_item')}: natureza incoerente com o tipo")
            if it.get("prioridade") not in PRIORIDADES:
                erros.append(f"{it.get('id_item')}: prioridade invalida")
            if it.get("id_setor") != s.get("id_setor"):
                erros.append(f"{it.get('id_item')}: item fora do setor")
            pv = it.get("previsao", {})
            sp = pv.get("share_previsto")
            if sp is not None and not 0 <= sp <= 1:
                erros.append(f"{it.get('id_item')}: share_previsto fora de [0,1]")
            for k in ("oportunidade_un", "risco_un"):
                if pv.get(k) is not None and pv[k] < 0:
                    erros.append(f"{it.get('id_item')}: {k} negativo")
    if n != saida["resumo"].get("n_itens_no_json"):
        erros.append("resumo.n_itens_no_json nao bate com os itens")
    return erros[:50]


def salvar_json(saida: dict, caminho: str | Path) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    return caminho


# ============================================================================
# Execucao pela linha de comando
# ============================================================================
def _ler_kpi_final(caminho: Path) -> pd.DataFrame:
    """
    Le a fato_analise_kpi_final. Se `caminho` for a pasta do historico, junta
    tambem os meses gerados pela execucao semanal (data/semanal/*/), para que
    cada mes novo entre no treino sem regenerar o historico.
    """
    if caminho.is_dir():
        arqs = sorted(caminho.glob("fato_analise_kpi_final*.csv*"))
        semanal = caminho.parent / "semanal"
        if semanal.exists():
            arqs += sorted(semanal.glob("*/fato_analise_kpi_final.csv"))
    else:
        arqs = [caminho]
    if not arqs or not arqs[0].exists():
        raise SystemExit(f"fato_analise_kpi_final nao encontrada em {caminho}.\n"
                         "Rode antes: python scripts/gerar_kpi_final.py")
    df = pd.concat((pd.read_csv(a, dtype={"grupo": "string"}, low_memory=False)
                    for a in arqs), ignore_index=True)
    # o historico vem primeiro na lista e tem precedencia; a pasta semanal so
    # acrescenta meses que ele nao cobre
    return df.drop_duplicates(["periodo_demanda", "id_produto", "id_alocacao"], keep="first")


def main(argv=None) -> int:
    raiz = Path(__file__).resolve().parent.parent.parent
    ap = argparse.ArgumentParser(description="Agente 3 - oportunidades de mercado")
    ap.add_argument("--kpi-final", type=Path, default=raiz / "data" / "sintetico")
    ap.add_argument("--produtos", type=Path, default=raiz / "data" / "sintetico" / "dim_produto.csv")
    ap.add_argument("--periodo", help="mes de referencia AAAA-MM (default: ultimo da base)")
    ap.add_argument("--meses-teste", type=int, default=3)
    ap.add_argument("--top", type=int, default=20, help="itens por setor no JSON")
    ap.add_argument("--saida", type=Path, default=raiz / "data" / "agentes" / "agente3_saida.json")
    ap.add_argument("--fallback", action="store_true", help="forca o ridge (teste sem xgboost)")
    args = ap.parse_args(argv)

    painel = montar_dataset(_ler_kpi_final(args.kpi_final), pd.read_csv(args.produtos))
    ag = AgenteOportunidadesMercado(forcar_fallback=args.fallback)
    r = ag.treinar(painel, meses_teste=args.meses_teste)
    recs = ag.recomendar(args.periodo)
    saida = ag.exportar_json(recs, top_por_setor=args.top)

    erros = validar_saida(saida)
    if erros:
        print("CONTRATO INVALIDO:\n  " + "\n  ".join(erros))
        return 1
    salvar_json(saida, args.saida)
    csv = args.saida.with_name("agente3_recomendacoes_completas.csv")
    recs.drop(columns=["fatores", "concorrentes"]).to_csv(csv, index=False)

    print(f"\nAgente 3 ({r.algoritmo}) - teste {r.periodo_teste[0]} a {r.periodo_teste[1]}")
    print(f"  MAE share  modelo {r.metricas['mae']:.5f} | persistencia "
          f"{r.baselines['persistencia']['mae']:.5f} | media 3m {r.baselines['media_3m']['mae']:.5f}"
          f"  -> ganho {r.ganho_sobre_baseline_pct:+.2f}%")
    print(f"  previsao de mercado WAPE {r.previsao_mercado['wape']:.4f} "
          f"(ingenuo {r.previsao_mercado['wape_ingenuo_mes_atual']:.4f})")
    print(f"  {saida['periodo_referencia']} -> {saida['periodo_previsto']}: "
          f"{saida['resumo']['n_itens']:,} recomendacoes; {saida['resumo']['por_tipo']}")
    print(f"  JSON: {args.saida}\n  CSV completo: {csv}")
    for a in saida["avisos"]:
        print(f"  AVISO: {a}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
