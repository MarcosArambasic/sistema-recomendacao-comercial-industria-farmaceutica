"""***************************************************************************************
SCRIPT    : kpi_final.py
OBJETIVO  : Monta a fato_analise_kpi_final - a base de TREINO do Agente 3.

            E a fato_analise_kpi mensal com duas coisas a mais:

            1) o SETOR e a ALOCACAO de cada outlet (via fato_alocacao), porque
               o Agente 4 cruza noticias com oportunidades "por setor e por
               alocacao" - grao que a fato_analise_kpi nao tem;
            2) os quatro indicadores de mercado CLASSIFICADOS em grupos
               (cd_mrkt_share, cd_cagr_medio, cd_cv, cd_indice_sazonal) pelas
               faixas da dim_classifica_kpis - as mesmas usadas na
               fato_analise_semanal_final, para os dois lados falarem a
               mesma lingua.

            Os valores numericos sao mantidos ao lado dos codigos: o modelo
            usa os dois (o codigo e a leitura de negocio, o valor guarda a
            informacao que a faixa descarta) e o Agente 4 precisa dos valores
            para justificar a recomendacao.

GRAO      : periodo_demanda (mes) x id_produto x id_alocacao
            (cada outlet pertence a exatamente uma alocacao/setor)

AUTOR     : Marcos Arambasic
DATA      : OUT/2026
****************************************************************************************/
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# indicador de mercado -> coluna de codigo na tabela final
MAPA_KPI_MERCADO: dict[str, str] = {
    "mrkt_share": "cd_mrkt_share",
    "cagr_medio": "cd_cagr_medio",
    "cv": "cd_cv",
    "indice_sazonal": "cd_indice_sazonal",
}

COLUNAS_KPI_FINAL = [
    "periodo_demanda", "nr_ano", "nr_mes",
    "id_produto", "grupo", "id_mercado", "id_setor", "id_alocacao", "id_outlet", "id_uf",
    "demanda_produto", "demanda_mercado",
    "mrkt_share", "cagr_medio", "cv", "indice_sazonal",
    "cd_mrkt_share", "cd_cagr_medio", "cd_cv", "cd_indice_sazonal",
]


def classificar_coluna(valores: pd.Series, faixas: pd.DataFrame) -> pd.Series:
    """Aplica as faixas de UM kpi (valor_inicial < x <= valor_final)."""
    faixas = faixas.sort_values("valor_inicial")
    cortes = [-np.inf, *faixas["valor_final"].tolist()[:-1], np.inf]
    return pd.cut(
        valores, bins=cortes, labels=faixas["cd_grupo_padrao"].tolist(),
        include_lowest=True, right=True,
    ).astype("Int64")


def montar_kpi_final(df_kpi: pd.DataFrame, df_alocacao: pd.DataFrame,
                     dim_classifica_kpis: pd.DataFrame) -> pd.DataFrame:
    """
    Parametros
    ----------
    df_kpi : fato_analise_kpi (mensal, produto x outlet)
    df_alocacao : fato_alocacao [id_alocacao, id_outlet, id_uf, id_setor]
    dim_classifica_kpis : [desc_kpi, valor_inicial, valor_final, cd_grupo_padrao]
    """
    obrig = {"desc_kpi", "valor_inicial", "valor_final", "cd_grupo_padrao"}
    if obrig - set(dim_classifica_kpis.columns):
        raise ValueError(f"dim_classifica_kpis sem {sorted(obrig - set(dim_classifica_kpis.columns))}")

    if df_alocacao["id_outlet"].duplicated().any():
        raise ValueError("fato_alocacao tem outlet em mais de uma alocacao - "
                         "o grao da kpi_final ficaria ambiguo")

    out = df_kpi.copy()
    out["periodo_demanda"] = pd.to_datetime(out["periodo_demanda"])
    out = out.merge(df_alocacao[["id_outlet", "id_alocacao", "id_setor"]],
                    on="id_outlet", how="inner")

    for kpi, col in MAPA_KPI_MERCADO.items():
        faixas = dim_classifica_kpis[dim_classifica_kpis["desc_kpi"] == kpi]
        if faixas.empty:
            raise ValueError(f"dim_classifica_kpis nao tem faixas para '{kpi}'")
        out[col] = classificar_coluna(out[kpi], faixas)

    out["nr_ano"] = out["periodo_demanda"].dt.year
    out["nr_mes"] = out["periodo_demanda"].dt.month
    return out[COLUNAS_KPI_FINAL].reset_index(drop=True)
