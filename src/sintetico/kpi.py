"""***************************************************************************************
SCRIPT    : kpi.py
OBJETIVO  : Calculo do fato_analise_kpi mensal (ETAPA 4 / secao 15 do prompt
            mestre). Grao de saida conforme o DDL: periodo x produto x outlet.

OS QUATRO KPIs:
    market share  = demanda_produto / demanda_mercado
    CAGR medio    = (valor_final / valor_inicial)^(1/n_anos) - 1
    CV            = desvio_padrao / media
    indice sazonal= demanda_do_mes / media_mensal_do_ano

DECISAO DE GRAO (importante, decidida com o grupo):
    market share e indice sazonal sao calculados NO OUTLET.
    CAGR e CV sao calculados em PRODUTO x UF e propagados para os outlets
    daquela UF.

    Motivo: CAGR anual e CV exigem serie temporal estavel. No grao de outlet
    a serie tem valores baixos, muitos zeros (a esparsidade faz o produto
    existir so em parte dos outlets) e meses faltando - o CAGR estoura para
    +/- infinito e o CV vira ruido pu ro. Em produto x UF a serie e densa e o
    indicador fica estavel. A coluna do DDL continua preenchida do mesmo
    jeito; muda so a granularidade do calculo, o que fica documentado.

DENOMINADOR DO MARKET SHARE: a demanda de TODOS os 30 produtos (proprios +
concorrentes) no mesmo outlet e mes. E o mercado de fato, nao so a carteira
propria.

AUTOR     : Marcos Arambasic
DATA      : SET/2026
****************************************************************************************/
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from . import config as cfg


def _fim_do_mes(ts: pd.Timestamp) -> pd.Timestamp:
    return ts + pd.offsets.MonthEnd(0)


def _cagr_por_serie(s: pd.Series, min_meses: int = 12) -> float:
    """
    CAGR anual sobre uma serie mensal, comparando a media do primeiro ano
    disponivel com a do ultimo (mesma convencao documentada no Excel).

    Retorna 0.0 quando nao ha janela suficiente ou quando o valor inicial e
    zero - preferivel a propagar inf/NaN para o banco.
    """
    s = s.dropna()
    if len(s) < min_meses:
        return 0.0
    por_ano = s.groupby(s.index.year).mean()
    if len(por_ano) < 2:
        return 0.0
    anos = sorted(por_ano.index)
    ini, fim = por_ano[anos[0]], por_ano[anos[-1]]
    n = anos[-1] - anos[0]
    if ini <= 0 or fim <= 0 or n <= 0:
        return 0.0
    return float((fim / ini) ** (1.0 / n) - 1.0)


def _indicadores_por_produto_uf(df: pd.DataFrame,
                                janela_cv_meses: int = 12) -> pd.DataFrame:
    """
    CAGR e CV em produto x UF, usando a serie mensal completa ate cada
    periodo (janela expansivel para o CAGR; janela movel para o CV).

    O CV usa janela movel de 12 meses porque e um indicador de
    variabilidade RECENTE - com janela expansivel ele ficaria cada vez mais
    "suavizado" e perderia poder de discriminacao ao longo do tempo.
    """
    serie = (
        df.groupby(["id_produto", "id_uf", "periodo_demanda"], as_index=False)["demanda_unidades"]
        .sum()
        .sort_values(["id_produto", "id_uf", "periodo_demanda"])
    )

    # CV movel de 12 meses (inclui o mes corrente)
    g = serie.groupby(["id_produto", "id_uf"], sort=False)["demanda_unidades"]
    media_movel = g.transform(lambda x: x.rolling(janela_cv_meses, min_periods=3).mean())
    dp_movel = g.transform(lambda x: x.rolling(janela_cv_meses, min_periods=3).std(ddof=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        serie["cv"] = np.where(media_movel > 0, dp_movel / media_movel, 0.0)
    serie["cv"] = serie["cv"].fillna(0.0)

    # ---- CAGR expansivel, vetorizado ------------------------------------
    # Definicao (mesma convencao do Excel): compara a media do PRIMEIRO ano
    # disponivel com a media do ano CORRENTE ate o mes em questao.
    #
    #   CAGR(t) = (media_ano_corrente_ate_t / media_primeiro_ano)^(1/n_anos) - 1
    #
    # Escrito de forma vetorizada em vez de laco por grupo: a versao ingenua
    # fazia 35.640 recortes de serie e levava ~50s, esta roda em fracao de
    # segundo e produz exatamente o mesmo resultado.
    serie["_ano"] = serie["periodo_demanda"].dt.year
    chave = ["id_produto", "id_uf"]

    primeiro_ano = serie.groupby(chave, sort=False)["_ano"].transform("min")

    # media do primeiro ano (completo) de cada serie
    media_1o = (
        serie[serie["_ano"] == primeiro_ano]
        .groupby(chave, sort=False)["demanda_unidades"].mean()
        .rename("media_primeiro_ano")
    )
    serie = serie.merge(media_1o, on=chave, how="left")

    # media do ano corrente acumulada ate o mes (expanding dentro do ano)
    serie["media_ano_corrente"] = (
        serie.groupby(chave + ["_ano"], sort=False)["demanda_unidades"]
        .transform(lambda x: x.expanding().mean())
    )

    n_anos = serie["_ano"] - primeiro_ano
    razao = serie["media_ano_corrente"] / serie["media_primeiro_ano"]

    valido = (
        (n_anos > 0)
        & (serie["media_primeiro_ano"] > 0)
        & (serie["media_ano_corrente"] > 0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        cagr = np.where(valido, razao ** (1.0 / n_anos.where(n_anos > 0, 1)) - 1.0, 0.0)
    serie["cagr_medio"] = np.nan_to_num(cagr, nan=0.0, posinf=0.0, neginf=0.0)

    return serie[["id_produto", "id_uf", "periodo_demanda", "cv", "cagr_medio"]]


def calcular_analise_kpi(
    df_demanda: pd.DataFrame,
    dims,
    janela_cv_meses: int = 12,
) -> pd.DataFrame:
    """
    Monta o fato_analise_kpi no grao periodo x produto x outlet.

    Parametros
    ----------
    df_demanda : fato_demanda_aberta_mensal
        ['periodo_demanda','id_produto','id_outlet','id_uf','demanda_unidades']
    dims : Dimensoes (do gerador)
    """
    df = df_demanda.copy()
    df["periodo_demanda"] = pd.to_datetime(df["periodo_demanda"])

    prod = dims.produtos.set_index("id_produto")
    df["id_mercado"] = df["id_produto"].map(prod["id_mercado"])
    df["grupo"] = df["id_produto"].map(prod["grupo_produto"])

    # ---- demanda do mercado no outlet/mes (todos os 30 produtos) ----------
    mercado_outlet = (
        df.groupby(["periodo_demanda", "id_outlet"], as_index=False)["demanda_unidades"]
        .sum()
        .rename(columns={"demanda_unidades": "demanda_mercado"})
    )
    out = df.rename(columns={"demanda_unidades": "demanda_produto"}).merge(
        mercado_outlet, on=["periodo_demanda", "id_outlet"], how="left"
    )

    # ---- totais nacionais por produto e do mercado, no mes ---------------
    total_produto = (
        df.groupby(["periodo_demanda", "id_produto"], as_index=False)["demanda_unidades"]
        .sum().rename(columns={"demanda_unidades": "demanda_produto_total"})
    )
    total_mercado = (
        df.groupby("periodo_demanda", as_index=False)["demanda_unidades"]
        .sum().rename(columns={"demanda_unidades": "demanda_mercado_total"})
    )
    out = out.merge(total_produto, on=["periodo_demanda", "id_produto"], how="left")
    out = out.merge(total_mercado, on="periodo_demanda", how="left")

    # ---- medias moveis (12 meses) do proprio outlet ----------------------
    out = out.sort_values(["id_produto", "id_outlet", "periodo_demanda"])
    gp = out.groupby(["id_produto", "id_outlet"], sort=False)
    out["demanda_media_produto"] = gp["demanda_produto"].transform(
        lambda x: x.rolling(janela_cv_meses, min_periods=1).mean()
    )
    gm = out.groupby("id_outlet", sort=False)
    out["demanda_media_mercado"] = gm["demanda_mercado"].transform(
        lambda x: x.rolling(janela_cv_meses, min_periods=1).mean()
    )

    # ---- market share ----------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        out["mrkt_share"] = np.where(
            out["demanda_mercado"] > 0,
            out["demanda_produto"] / out["demanda_mercado"],
            0.0,
        )
    out["mrkt_share"] = out["mrkt_share"].clip(0.0, 1.0).fillna(0.0)

    # ---- indice sazonal: demanda do mes / media mensal do ano ------------
    out["_ano"] = out["periodo_demanda"].dt.year
    media_ano = (
        out.groupby(["id_produto", "id_outlet", "_ano"])["demanda_produto"]
        .transform("mean")
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        out["indice_sazonal"] = np.where(
            media_ano > 0, out["demanda_produto"] / media_ano, 0.0
        )
    out["indice_sazonal"] = out["indice_sazonal"].fillna(0.0)

    # ---- CAGR e CV calculados em produto x UF e propagados ---------------
    ind = _indicadores_por_produto_uf(df, janela_cv_meses)
    out = out.merge(ind, on=["id_produto", "id_uf", "periodo_demanda"], how="left")
    out["cv"] = out["cv"].fillna(0.0)
    out["cagr_medio"] = out["cagr_medio"].fillna(0.0)

    # ---- metadados -------------------------------------------------------
    out["data_atualizacao_base"] = out["periodo_demanda"].apply(
        lambda p: _fim_do_mes(p) + timedelta(days=cfg.LATENCIA_DIAS)
    )

    colunas = [
        "data_atualizacao_base", "periodo_demanda",
        "id_produto", "grupo", "id_mercado", "id_outlet", "id_uf",
        "demanda_produto", "demanda_mercado",
        "demanda_media_produto", "demanda_media_mercado",
        "demanda_produto_total", "demanda_mercado_total",
        "mrkt_share", "cagr_medio", "cv", "indice_sazonal",
    ]
    return out[colunas].reset_index(drop=True)


KPIS_CLASSIFICAVEIS = ["mrkt_share", "cagr_medio", "cv", "indice_sazonal",
                       "cobertura_cota_projetada"]

# KPIs em que zero significa AUSENCIA (produto nao vendido / sem cota), nao
# um nivel baixo - incluir esses zeros deslocaria todos os quantis para baixo
KPIS_IGNORAR_ZERO = {"mrkt_share", "indice_sazonal", "cobertura_cota_projetada"}


def sugerir_faixas_kpi(df_kpi: pd.DataFrame,
                       quantis: tuple[float, ...] = (0.25, 0.50, 0.75),
                       kpis: list[str] | None = None) -> pd.DataFrame:
    """
    Sugere cortes para a dim_classifica_kpis a partir dos quantis reais da
    base gerada.

    Resolve a pendencia do Excel, que dizia que as faixas de CV seriam
    "a determinar depois da base criada" - agora a base existe. A saida ja
    vem no formato das colunas de dim_classifica_kpis, para copiar e ajustar.

    Processa apenas os KPIs presentes no DataFrame, entao da para chamar com
    uma tabela que tenha so parte das colunas (ex.: a cobertura projetada,
    que so existe no grao semanal).
    """
    rotulos = ["baixo", "medio", "alto", "muito_alto"]
    linhas = []

    candidatos = kpis or KPIS_CLASSIFICAVEIS
    presentes = [k for k in candidatos if k in df_kpi.columns]
    if not presentes:
        raise ValueError(
            f"nenhum KPI classificavel encontrado. Colunas recebidas: "
            f"{list(df_kpi.columns)[:10]}"
        )

    for kpi in presentes:
        s = df_kpi[kpi].replace([np.inf, -np.inf], np.nan).dropna()
        if kpi in KPIS_IGNORAR_ZERO:
            s = s[s > 0]
        if s.empty:
            continue

        cortes, rot = _cortes_do_kpi(kpi, s, quantis, rotulos)
        limites = [-np.inf, *cortes, np.inf]

        for i, rotulo in enumerate(rot):
            linhas.append({
                "desc_kpi": kpi,
                "valor_inicial": round(limites[i], 6) if np.isfinite(limites[i]) else -999999.0,
                "valor_final": round(limites[i + 1], 6) if np.isfinite(limites[i + 1]) else 999999.0,
                "desc_grupo": rotulo,
                "cd_grupo_padrao": i + 1,
            })
    return pd.DataFrame(linhas)


def _cortes_do_kpi(kpi: str, s: pd.Series, quantis: tuple[float, ...],
                   rotulos: list[str]) -> tuple[list[float], list[str]]:
    """
    Escolhe os cortes de cada KPI. Quantil NAO serve para todos:

    * cobertura_cota_projetada -> usa os THRESHOLDS DE NEGOCIO do Excel
      (<=65% critico, <=75% atencao, <80% proximo, >=80% bateu plano).
      Sao a regra operacional que a empresa ja usa; trocar por quantis
      inventaria uma regra nova e desconectada de como o time e cobrado.

    * cagr_medio -> usa cortes SEMANTICOS. O CAGR tem uma massa de ~27% de
      valores exatamente zero (meses sem 2 anos de historico), entao o 25o e
      o 50o percentil caem os dois em 0,0 e a faixa "medio" nasce vazia. Alem
      disso, o sinal do CAGR tem significado proprio: crescer 1% e cair 1%
      nao sao "a mesma faixa" so porque estao proximos na distribuicao.

    * demais KPIs -> quantis da propria base, que e o que o Excel pedia
      ("a determinar depois da base criada").
    """
    if kpi == "cobertura_cota_projetada":
        t = cfg.THRESHOLDS_COBERTURA
        return (
            [t["critico"], t["atencao"], t["proximo"]],
            ["critico", "atencao", "proximo", "bateu_plano"],
        )

    if kpi == "cagr_medio":
        return (
            [-0.02, 0.02, 0.50],
            ["queda", "estavel_ou_sem_historico", "crescimento", "crescimento_forte"],
        )

    return [float(s.quantile(q)) for q in quantis], rotulos


if __name__ == "__main__":
    from .cota import calcular_cota_mensal
    from .gerador import agregar_por_setor, gerar_demanda_mensal

    df, dims, diag = gerar_demanda_mensal()
    kpi = calcular_analise_kpi(df, dims)

    print(f"fato_analise_kpi: {len(kpi):,} linhas")
    print("\nAmostra:")
    print(kpi.head(3).to_string(index=False))

    print("\nDistribuicao dos KPIs (linhas com demanda > 0):")
    ativo = kpi[kpi.demanda_produto > 0]
    print(ativo[["mrkt_share", "cagr_medio", "cv", "indice_sazonal"]]
          .describe(percentiles=[0.25, 0.5, 0.75, 0.95]).to_string())

    print("\nFaixas sugeridas para dim_classifica_kpis:")
    print(sugerir_faixas_kpi(ativo).to_string(index=False))
