"""***************************************************************************************
SCRIPT    : pipeline_semanal.py
OBJETIVO  : Pipeline semanal completo (ETAPAS 1, 3, 5, 6 e 7 do prompt mestre),
            com suporte a BACKFILL historico sobre as 229 semanas do periodo.

            Roda tanto para uma semana isolada (operacao normal) quanto para
            todo o historico (gerar dados de treino do XGBoost).

ETAPAS:
  1. desagregar_semanal   mensal -> fato_demanda_aberta_semanal
  3. alocar_semanal       -> fato_demanda_alocada_semanal (acumulado no mes)
  5. calcular_desempenho  -> fato_desempenho_semanal (cobertura e projetada)
  6. montar_analise       -> fato_analise_semanal (KPI mensal + desempenho)
  7. classificar          -> fato_analise_semanal_final (via dim_classifica_kpis
                            e dim_regras_diagnostico)

DECISOES DE MODELAGEM:

  * RECONCILIACAO EXATA. A soma das semanas de um mes e identica a demanda
    mensal daquele mes. O rateio usa dias da semana / dias do mes, com ruido,
    e depois renormaliza. O arredondamento para inteiro usa o metodo do maior
    resto, entao a soma continua batendo apos arredondar.

  * ACUMULADO REINICIA TODO MES. Conforme a secao 14 do prompt: semana 1 = 100,
    semana 2 = 220, semana 3 = 350... e no mes seguinte volta do zero.

  * RATEIO DA COTA PARA O OUTLET. A cota existe em setor x produto, mas o
    desempenho e medido em alocacao (outlet). O rateio e proporcional a
    participacao historica do outlet no setor (media movel de 3 meses
    ANTERIORES) - usar o mes corrente vazaria informacao do futuro para
    dentro da meta.

  * SO PRODUTOS PROPRIOS. Nao existe cota para produto de concorrente, entao
    desempenho e analise semanal cobrem apenas os 10 proprios.

AUTOR     : Marcos Arambasic 
DATA      : SET/2026
****************************************************************************************/
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .calendario import calendario_dataframe


# ============================================================================
# Utilidades
# ============================================================================
def _arredondar_preservando_soma(valores: np.ndarray, total: np.ndarray) -> np.ndarray:
    """
    Arredonda para inteiro preservando a soma exata por grupo (metodo do
    maior resto). Sem isso, arredondar semana a semana faria a soma das
    semanas divergir da demanda mensal e a reconciliacao quebraria.

    `valores` e `total` tem a mesma forma (linhas = grupo, colunas = semanas).
    """
    piso = np.floor(valores)
    resto = valores - piso
    faltando = np.rint(total - piso.sum(axis=1)).astype(int)

    saida = piso.copy()
    for i in range(saida.shape[0]):
        k = faltando[i]
        if k <= 0:
            continue
        ordem = np.argsort(-resto[i])[:k]
        saida[i, ordem] += 1
    return saida


# ============================================================================
# ETAPA 1 - desagregacao mensal -> semanal
# ============================================================================
def desagregar_semanal(
    df_mensal: pd.DataFrame,
    ruido_semanal: float = 0.12,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Distribui a demanda mensal entre as semanas comerciais do mes.

    Base do rateio: proporcao de dias da semana no mes. O ruido introduz
    variacao semanal realista (nem toda semana vende o proporcional exato),
    e a renormalizacao garante que a soma continue igual ao mes.
    """
    rng = np.random.default_rng(cfg.RANDOM_SEED if seed is None else seed)

    df = df_mensal.copy()
    df["periodo_demanda"] = pd.to_datetime(df["periodo_demanda"])

    # calendario dos meses presentes no proprio df (e nao do PERIODO_FIM do
    # config): a execucao semanal pode gerar meses alem do horizonte da base
    if df.empty:
        return df.assign(nr_semana=pd.Series(dtype=int))
    _ini, _fim = df["periodo_demanda"].min(), df["periodo_demanda"].max()
    cal = calendario_dataframe(f"{_ini:%Y-%m}", f"{_fim:%Y-%m}")
    cal["periodo_mensal"] = pd.to_datetime(cal["periodo_mensal"])
    cal["periodo_demanda"] = pd.to_datetime(cal["periodo_demanda"])

    # O calendario vira MATRIZES (linhas = mes, colunas = nr_semana). Indexar
    # matriz com numpy e ordens de magnitude mais rapido do que fazer .loc com
    # milhoes de tuplas (mes, semana), que era o gargalo desta etapa.
    max_sem = int(cal["nr_semana"].max())

    def _matriz(coluna: str, fill=0.0):
        return (
            cal.pivot_table(index="periodo_mensal", columns="nr_semana",
                            values=coluna, fill_value=fill, aggfunc="first")
            .reindex(columns=range(1, max_sem + 1), fill_value=fill)
            .sort_index()
        )

    fracoes = _matriz("fracao_mes")

    # As DATAS nao passam por matriz. A versao anterior convertia
    # datetime -> int64 (nanossegundos) -> datetime para caber no pivot_table,
    # e esse round-trip quebra no pandas 3.x: as datas voltavam como
    # 1970-01-01 (epoch). Agora as datas entram por MERGE com o calendario no
    # final, que e rapido e independe de versao do pandas/numpy.
    meses = fracoes.index
    idx_mes = pd.Series(range(len(meses)), index=meses)

    df = df[df["periodo_demanda"].isin(meses)].copy()
    linha_mes = df["periodo_demanda"].map(idx_mes).to_numpy()

    base = fracoes.to_numpy()[linha_mes]                    # (n_linhas, max_sem)
    ativo = base > 0

    ruido = rng.lognormal(0.0, ruido_semanal, size=base.shape)
    peso = np.where(ativo, base * ruido, 0.0)
    soma = peso.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        peso = np.where(soma > 0, peso / soma, 0.0)

    total = df["demanda_unidades"].to_numpy(dtype=float)
    valores = peso * total[:, None]
    inteiros = _arredondar_preservando_soma(valores, total)

    # formato longo, descartando semanas inexistentes no mes
    idx_linha, idx_sem = np.nonzero(ativo)
    linha_cal = linha_mes[idx_linha]          # posicao do mes na matriz

    saida = pd.DataFrame({
        "periodo_mensal": meses.to_numpy()[linha_cal],
        "nr_semana": (idx_sem + 1).astype("int16"),
        "id_produto": df["id_produto"].to_numpy()[idx_linha],
        "id_outlet": df["id_outlet"].to_numpy()[idx_linha],
        "id_uf": df["id_uf"].to_numpy()[idx_linha],
        "demanda_unidades": inteiros[idx_linha, idx_sem],
    })

    # datas e contagens de dias vem do calendario por merge - sem conversao
    # manual de tipo, entao funciona igual em pandas 2.x e 3.x
    ref = cal[["periodo_mensal", "nr_semana", "periodo_demanda",
               "dias_corridos_ate_fim_semana", "dias_no_mes"]].rename(
        columns={"dias_corridos_ate_fim_semana": "dias_corridos"}
    )
    saida["nr_semana"] = saida["nr_semana"].astype("int64")
    ref = ref.astype({"nr_semana": "int64"})

    saida = saida.merge(ref, on=["periodo_mensal", "nr_semana"], how="left")

    if saida["periodo_demanda"].isna().any():
        n = int(saida["periodo_demanda"].isna().sum())
        raise ValueError(
            f"{n} linhas sem data no calendario apos o merge - "
            "combinacao (periodo_mensal, nr_semana) inexistente."
        )

    saida["data_atualizacao_base"] = saida["periodo_demanda"] + pd.Timedelta(days=cfg.LATENCIA_DIAS)

    colunas = ["periodo_demanda", "periodo_mensal", "nr_semana", "id_produto",
               "id_outlet", "id_uf", "demanda_unidades", "dias_corridos",
               "dias_no_mes", "data_atualizacao_base"]
    return saida[colunas]


# ============================================================================
# ETAPA 3 - alocacao semanal com acumulado
# ============================================================================
def alocar_semanal(df_semanal: pd.DataFrame, dims) -> pd.DataFrame:
    """
    Mapeia a demanda semanal para a alocacao (outlet -> setor) e acumula
    dentro do mes. O acumulado REINICIA a cada mes novo.
    """
    mapa = dims.alocacao[["id_outlet", "id_alocacao", "id_setor"]]
    df = df_semanal.merge(mapa, on="id_outlet", how="left")

    df = df.sort_values(["id_produto", "id_alocacao", "periodo_mensal", "nr_semana"])
    df["demanda_unidades_acumulada"] = (
        df.groupby(["id_produto", "id_alocacao", "periodo_mensal"], sort=False)["demanda_unidades"]
        .cumsum()
    )
    return df


# ============================================================================
# ETAPA 5 - desempenho semanal
# ============================================================================
def _ratear_cota_por_outlet(df_cota: pd.DataFrame, df_mensal: pd.DataFrame,
                            dims, janela: int = 3) -> pd.DataFrame:
    """
    Distribui a cota de (setor, produto) entre os outlets do setor,
    proporcionalmente a participacao historica do outlet.

    A participacao usa a media dos `janela` meses ANTERIORES. Usar o mes
    corrente faria a meta "saber" o resultado antes dele acontecer, o que
    inflaria artificialmente a cobertura e contaminaria o treino do modelo.
    """
    mapa = dims.alocacao[["id_outlet", "id_alocacao", "id_setor"]]
    d = df_mensal.merge(mapa, on="id_outlet", how="left")
    d["periodo_demanda"] = pd.to_datetime(d["periodo_demanda"])

    d = d.sort_values(["id_produto", "id_alocacao", "periodo_demanda"])
    d["media_hist"] = (
        d.groupby(["id_produto", "id_alocacao"], sort=False)["demanda_unidades"]
        .transform(lambda s: s.shift(1).rolling(janela, min_periods=1).mean())
    )
    # no primeiro mes nao ha historico: cai para rateio igualitario
    d["media_hist"] = d["media_hist"].fillna(1.0)

    total_setor = (
        d.groupby(["id_produto", "id_setor", "periodo_demanda"])["media_hist"]
        .transform("sum")
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        d["peso_outlet"] = np.where(total_setor > 0, d["media_hist"] / total_setor, 0.0)

    c = df_cota.copy()
    c["periodo_demanda"] = pd.to_datetime(c["periodo_demanda"])
    out = d.merge(
        c[["periodo_demanda", "id_setor", "id_produto", "cota_unidades"]],
        on=["periodo_demanda", "id_setor", "id_produto"], how="inner",
    )
    out["cota_outlet"] = out["cota_unidades"] * out["peso_outlet"]
    return out[["periodo_demanda", "id_produto", "id_alocacao", "id_setor", "cota_outlet"]]


def calcular_desempenho(df_alocado: pd.DataFrame, df_cota: pd.DataFrame,
                        df_mensal: pd.DataFrame, dims) -> pd.DataFrame:
    """
    fato_desempenho_semanal: cobertura da cota e cobertura projetada.

        cobertura           = acumulado / cota
        cobertura_projetada = (acumulado / dias_corridos x dias_do_mes) / cota

    Cobre apenas produtos proprios (sao os unicos com cota).
    """
    cota_outlet = _ratear_cota_por_outlet(df_cota, df_mensal, dims)

    df = df_alocado.merge(
        cota_outlet,
        left_on=["periodo_mensal", "id_produto", "id_alocacao", "id_setor"],
        right_on=["periodo_demanda", "id_produto", "id_alocacao", "id_setor"],
        how="inner", suffixes=("", "_mensal"),
    )

    ritmo = df["demanda_unidades_acumulada"] / df["dias_corridos"].clip(lower=1)
    df["demanda_unidades_projetada"] = ritmo * df["dias_no_mes"]

    cota = df["cota_outlet"].replace(0, np.nan)
    df["cobertura_cota"] = (df["demanda_unidades_acumulada"] / cota).fillna(0.0)
    df["cobertura_cota_projetada"] = (df["demanda_unidades_projetada"] / cota).fillna(0.0)
    df["cota_mensal"] = df["cota_outlet"]

    return df[[
        "periodo_demanda", "periodo_mensal", "nr_semana",
        "id_produto", "id_setor", "id_alocacao", "id_uf", "id_outlet",
        "demanda_unidades_acumulada", "demanda_unidades_projetada", "cota_mensal",
        "cobertura_cota", "cobertura_cota_projetada",
    ]]


# ============================================================================
# ETAPA 6 - analise semanal (KPI mensal + desempenho semanal)
# ============================================================================
def montar_analise_semanal(df_desempenho: pd.DataFrame, df_kpi: pd.DataFrame,
                           dims, defasagem_kpi: int = 1) -> pd.DataFrame:
    """
    Cruza o desempenho da semana com os KPIs do mes.

    defasagem_kpi=1 usa os KPIs do MES ANTERIOR - que e como o sistema opera
    de verdade: na 1a semana de M os KPIs disponiveis sao os de M-1, porque o
    mes corrente ainda nao fechou. Isso tambem evita vazamento temporal no
    treino do XGBoost.
    """
    kpi = df_kpi.copy()
    kpi["periodo_demanda"] = pd.to_datetime(kpi["periodo_demanda"])
    kpi = kpi[[
        "periodo_demanda", "id_produto", "id_outlet", "id_mercado",
        "mrkt_share", "cagr_medio", "cv", "indice_sazonal",
    ]].rename(columns={"periodo_demanda": "periodo_kpi"})

    df = df_desempenho.copy()
    df["periodo_kpi"] = df["periodo_mensal"] - pd.DateOffset(months=defasagem_kpi)

    out = df.merge(kpi, on=["periodo_kpi", "id_produto", "id_outlet"], how="left")

    # meses sem KPI anterior (inicio da serie) ficam com indicador neutro
    for c in ["mrkt_share", "cagr_medio", "cv", "indice_sazonal"]:
        out[c] = out[c].fillna(0.0)
    out["id_mercado"] = out["id_mercado"].fillna(1).astype("int64")

    out["data_atualizacao_base"] = out["periodo_demanda"] + pd.Timedelta(days=cfg.LATENCIA_DIAS)

    return out[[
        "data_atualizacao_base", "periodo_demanda", "periodo_mensal", "nr_semana",
        "id_produto", "id_mercado", "id_setor", "id_alocacao", "id_uf",
        "cobertura_cota_projetada", "mrkt_share", "cagr_medio", "cv", "indice_sazonal",
    ]]


# ============================================================================
# ETAPA 7 - classificacao
# ============================================================================
MAPA_KPI_COLUNA = {
    "cobertura_cota_projetada": "cd_cobertura_cota_projetada",
    "mrkt_share": "cd_mrkt_share",
    "cagr_medio": "cd_cagr_medio",
    "cv": "cd_cv",
    "indice_sazonal": "cd_indice_sazonal",
}


def classificar(df_analise: pd.DataFrame, dim_classifica_kpis: pd.DataFrame) -> pd.DataFrame:
    """
    Converte os indicadores numericos em codigos de grupo, usando as faixas
    de `dim_classifica_kpis`.

    As faixas vem do BANCO (ou de um CSV com as mesmas colunas), nunca
    hard-coded aqui - assim voces podem mudar os cortes sem tocar no Python,
    como pede a secao 18 do prompt.

    dim_classifica_kpis precisa ter: desc_kpi, valor_inicial, valor_final,
    desc_grupo, cd_grupo_padrao.
    """
    obrigatorias = {"desc_kpi", "valor_inicial", "valor_final", "cd_grupo_padrao"}
    faltando = obrigatorias - set(dim_classifica_kpis.columns)
    if faltando:
        raise ValueError(f"dim_classifica_kpis sem as colunas: {sorted(faltando)}")

    out = df_analise.copy()

    for kpi, col_cd in MAPA_KPI_COLUNA.items():
        faixas = dim_classifica_kpis[dim_classifica_kpis["desc_kpi"] == kpi]
        if faixas.empty:
            raise ValueError(
                f"dim_classifica_kpis nao tem faixas para '{kpi}'. "
                f"KPIs presentes: {sorted(dim_classifica_kpis['desc_kpi'].unique())}"
            )
        faixas = faixas.sort_values("valor_inicial")
        cortes = [-np.inf, *faixas["valor_final"].tolist()[:-1], np.inf]
        codigos = faixas["cd_grupo_padrao"].tolist()

        out[col_cd] = pd.cut(
            out[kpi], bins=cortes, labels=codigos, include_lowest=True, right=True
        ).astype("Int64")

    out["nr_ano"] = out["periodo_demanda"].dt.year
    out["nr_mes"] = out["periodo_demanda"].dt.month

    return out[[
        "nr_ano", "nr_mes", "nr_semana",
        "id_produto", "id_mercado", "id_setor", "id_alocacao", "id_uf",
        *MAPA_KPI_COLUNA.values(),
    ]]


def aplicar_diagnostico(df_final: pd.DataFrame,
                        dim_regras_diagnostico: pd.DataFrame) -> pd.DataFrame:
    """
    Junta o diagnostico textual a partir da combinacao de codigos, usando
    `dim_regras_diagnostico`.

    Nao inventa nenhum diagnostico: se a combinacao nao existir na tabela de
    regras, o resultado vem nulo e e reportado na contagem de cobertura.
    """
    # Atencao ao nome: o DDL grafa "mkt" (sem o "r") em
    # cd_mkt_indicator_padrao, enquanto a coluna do KPI e "mrkt_share".
    # A inconsistencia vem do schema original; aqui seguimos o DDL.
    chaves = [
        ("cd_mkt_indicator_padrao", "cd_mrkt_share"),
        ("cd_cagr_indicator_padrao", "cd_cagr_medio"),
        ("cd_cv_indicator_padrao", "cd_cv"),
        ("cd_sazonal_indicator_padrao", "cd_indice_sazonal"),
        ("cd_cob_proj_indicator_padrao", "cd_cobertura_cota_projetada"),
    ]
    faltando = [c for c, _ in chaves if c not in dim_regras_diagnostico.columns]
    if faltando:
        raise ValueError(f"dim_regras_diagnostico sem as colunas: {faltando}")

    regras = dim_regras_diagnostico[[c for c, _ in chaves] + ["diagnostico"]].copy()
    regras.columns = [d for _, d in chaves] + ["diagnostico"]

    out = df_final.merge(regras, on=[d for _, d in chaves], how="left")
    cobertura = out["diagnostico"].notna().mean()
    if cobertura < 1.0:
        print(f"[aviso] {1 - cobertura:.1%} das linhas ficaram sem diagnostico - "
              "faltam combinacoes em dim_regras_diagnostico")
    return out


# ============================================================================
# Orquestracao do backfill
# ============================================================================
def executar_backfill(df_mensal: pd.DataFrame, df_cota: pd.DataFrame,
                      df_kpi: pd.DataFrame, dims,
                      seed: int | None = None,
                      ao_concluir=None,
                      manter_em_memoria: bool = True) -> dict[str, pd.DataFrame]:
    """
    Roda o pipeline semanal sobre todo o historico.

    Parametros de memoria (importam: as tabelas somam ~1,5 GB, e os merges
    chegam a dobrar isso durante a execucao):

      ao_concluir : callback(nome_tabela, df) chamado assim que cada tabela
          fica pronta. Serve para exportar na hora e liberar em seguida.
      manter_em_memoria : se False, cada tabela e descartada apos o callback
          e o retorno traz so as contagens. Use quando estiver exportando
          para CSV e nao precisar dos DataFrames de volta.
    """
    import gc

    proprios = dims.produtos.loc[dims.produtos.tipo_produto == "PROPRIO", "id_produto"]
    resultado: dict[str, pd.DataFrame] = {}
    contagens: dict[str, int] = {}

    def _entregar(nome: str, df: pd.DataFrame) -> None:
        contagens[nome] = len(df)
        if ao_concluir is not None:
            ao_concluir(nome, df)
        if manter_em_memoria:
            resultado[nome] = df

    # 1) desagregacao mensal -> semanal
    semanal = desagregar_semanal(df_mensal, seed=seed)
    _entregar("fato_demanda_aberta_semanal", semanal)

    # 3) alocacao (so proprios - sao os unicos com cota)
    alocado = alocar_semanal(semanal[semanal.id_produto.isin(proprios)], dims)
    del semanal
    gc.collect()
    _entregar("fato_demanda_alocada_semanal", alocado)

    # 5) desempenho
    desempenho = calcular_desempenho(alocado, df_cota, df_mensal, dims)
    del alocado
    gc.collect()
    _entregar("fato_desempenho_semanal", desempenho)

    # 6) analise semanal
    analise = montar_analise_semanal(desempenho, df_kpi, dims)
    del desempenho
    gc.collect()
    _entregar("fato_analise_semanal", analise)

    if not manter_em_memoria:
        del analise
        gc.collect()
        return {k: v for k, v in contagens.items()}   # so as contagens

    return resultado


if __name__ == "__main__":
    import time

    from .cota import calcular_cota_mensal
    from .gerador import agregar_por_setor, gerar_demanda_mensal
    from .kpi import calcular_analise_kpi

    t0 = time.time()
    df, dims, diag = gerar_demanda_mensal()
    cota = calcular_cota_mensal(agregar_por_setor(df, dims))
    kpi = calcular_analise_kpi(df, dims)
    print(f"insumos prontos em {time.time() - t0:.1f}s")

    t1 = time.time()
    tabelas = executar_backfill(df, cota, kpi, dims)
    print(f"backfill em {time.time() - t1:.1f}s\n")

    for nome, t in tabelas.items():
        print(f"  {nome:<32s} {len(t):>10,} linhas")

    # reconciliacao semanal = mensal
    sem = tabelas["fato_demanda_aberta_semanal"]
    a = sem.groupby(["periodo_mensal", "id_produto", "id_outlet"])["demanda_unidades"].sum().sum()
    b = df["demanda_unidades"].sum()
    print(f"\nreconciliacao semanal vs mensal: {a:,.0f} vs {b:,.0f} "
          f"({'OK' if abs(a - b) < 1 else 'DIVERGENTE'})")

    dsp = tabelas["fato_desempenho_semanal"]
    print(f"cobertura_cota: mediana {dsp.cobertura_cota.median():.3f} | "
          f"projetada {dsp.cobertura_cota_projetada.median():.3f}")
