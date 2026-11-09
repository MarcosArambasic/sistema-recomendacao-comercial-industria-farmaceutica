"""***************************************************************************************
SCRIPT    : cota.py
OBJETIVO  : Calculo da cota mensal (fato_cota_mensal), no grao
            mes x setor x produto - secoes 11 e 13 do prompt mestre.

REGRA (parametrizavel, nada hard-coded):

    cota(M) = media(demanda_realizada[M-janela .. M-1])
              x (1 + crescimento)
              x ajuste_sazonal(M)

    * janela      : 6 meses por padrao (COTA_JANELA_MESES)
    * crescimento : +4,25% por padrao (COTA_CRESCIMENTO), que e a razao
                    objetivo/demanda observada nos dados reais da empresa
                    (15.397 / 14.769). O prompt sugeria +3%; ambos ficam
                    disponiveis por parametro.
    * ajuste_sazonal(M) : corrige o fato de a media dos 6 meses anteriores
                    carregar a sazonalidade DAQUELES meses, e nao a do mes
                    alvo. Sem isso, a cota de um mes de pico seria sempre
                    subestimada e a de um vale, superestimada.

BOOTSTRAP: os primeiros meses da serie nao tem 6 meses de historico. Em vez
de ficar sem cota, usa-se janela expansivel (o que houver, respeitando
COTA_JANELA_MINIMA). Isso e sinalizado na coluna `janela_utilizada` para que
a validacao e o TCC possam tratar esses meses como periodo de aquecimento.

AUTOR     : Marcos Arambasic
DATA      : AGO/2026
****************************************************************************************/
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import config as cfg
from .calendario import dias_no_mes, semanas_do_mes


def calcular_cota_mensal(
    demanda_por_setor: pd.DataFrame,
    janela_meses: int = cfg.COTA_JANELA_MESES,
    crescimento: float = cfg.COTA_CRESCIMENTO,
    usar_ajuste_sazonal: bool = cfg.COTA_USAR_AJUSTE_SAZONAL,
    janela_minima: int = cfg.COTA_JANELA_MINIMA,
    perfil_sazonal: dict[tuple[int, int], float] | None = None,
    extrapolar_tendencia: bool = cfg.COTA_EXTRAPOLAR_TENDENCIA,
    limite_extrapolacao: float = cfg.COTA_LIMITE_EXTRAPOLACAO,
) -> pd.DataFrame:
    """
    Calcula a cota mensal por (periodo, setor, produto).

    Parametros
    ----------
    demanda_por_setor : DataFrame com colunas
        ['periodo_demanda', 'id_setor', 'id_produto', 'demanda_unidades']
        ja consolidado no grao setor (soma dos outlets do setor).
    janela_meses : tamanho da janela de historico.
    crescimento : fator de ambicao aplicado sobre a media historica.
    usar_ajuste_sazonal : corrige a sazonalidade do mes alvo vs a da janela.
    janela_minima : minimo de meses para conseguir calcular alguma cota.
    perfil_sazonal : opcional, {(id_setor, mes): indice}. Se None, e estimado
        da propria serie de demanda.

    Retorna
    -------
    DataFrame com ['periodo_demanda','id_setor','id_produto','cota_unidades',
                   'janela_utilizada','bootstrap']
    """
    df = demanda_por_setor.copy()
    df["periodo_demanda"] = pd.to_datetime(df["periodo_demanda"])
    df = df.sort_values(["id_setor", "id_produto", "periodo_demanda"])

    # ---- media movel dos `janela_meses` meses ANTERIORES (nao inclui o mes) --
    grupo = df.groupby(["id_setor", "id_produto"], sort=False)["demanda_unidades"]
    df["media_hist"] = grupo.transform(
        lambda s: s.shift(1).rolling(janela_meses, min_periods=janela_minima).mean()
    )
    df["janela_utilizada"] = grupo.transform(
        lambda s: s.shift(1).rolling(janela_meses, min_periods=janela_minima).count()
    )
    df["bootstrap"] = df["janela_utilizada"] < janela_meses

    # ---- ajuste sazonal ---------------------------------------------------
    if usar_ajuste_sazonal:
        if perfil_sazonal is None:
            perfil = _estimar_perfil_sazonal(df)
        else:
            perfil = perfil_sazonal

        mes_alvo = df["periodo_demanda"].dt.month
        idx_alvo = pd.Series(
            [perfil.get((s, m), 1.0) for s, m in zip(df["id_setor"], mes_alvo)],
            index=df.index,
        )
        # indice medio da janela usada como base
        idx_base = (
            df.assign(_i=idx_alvo)
            .groupby(["id_setor", "id_produto"], sort=False)["_i"]
            .transform(lambda s: s.shift(1).rolling(janela_meses, min_periods=janela_minima).mean())
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            ajuste = np.where(idx_base > 0, idx_alvo / idx_base, 1.0)
    else:
        ajuste = 1.0

    # ---- extrapolacao da tendencia ----------------------------------------
    # A media da janela esta centrada ~ (janela+1)/2 meses ANTES do mes alvo.
    # Numa serie com tendencia forte (concorrentes crescem ~79% a.a.) usar a
    # media crua deixaria a cota sistematicamente defasada - a razao
    # cota/demanda cai para ~0,94 em vez do 1,0425 observado na empresa.
    # Projetamos a media para o mes alvo comparando as duas metades da janela.
    if extrapolar_tendencia and janela_meses >= 4:
        metade = janela_meses // 2
        g2 = df.groupby(["id_setor", "id_produto"], sort=False)["demanda_unidades"]
        recente = g2.transform(
            lambda s: s.shift(1).rolling(metade, min_periods=1).mean()
        )
        antiga = g2.transform(
            lambda s: s.shift(1 + metade).rolling(metade, min_periods=1).mean()
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            razao = np.where((antiga > 0) & (recente > 0), recente / antiga, 1.0)
        # razao cobre `metade` meses; projetamos ate o mes alvo
        gap = (janela_meses + 1) / 2.0
        fator = np.power(np.clip(razao, 1e-6, None), gap / metade)
        fator = np.clip(fator, 1.0 / limite_extrapolacao, limite_extrapolacao)
        fator = np.nan_to_num(fator, nan=1.0)
    else:
        fator = 1.0

    df["cota_unidades"] = df["media_hist"] * fator * (1.0 + crescimento) * ajuste

    # meses sem historico algum ficam sem cota (serao descartados)
    df = df[df["cota_unidades"].notna()].copy()
    df["cota_unidades"] = df["cota_unidades"].clip(lower=0).round(0)

    return df[[
        "periodo_demanda", "id_setor", "id_produto",
        "cota_unidades", "janela_utilizada", "bootstrap",
    ]].reset_index(drop=True)


def _estimar_perfil_sazonal(df: pd.DataFrame) -> dict[tuple[int, int], float]:
    """
    Indice sazonal por (setor, mes-calendario), estimado da propria serie de
    demanda depois de remover a tendencia - mesma logica usada na calibracao.
    """
    total = (
        df.groupby(["id_setor", "periodo_demanda"], as_index=False)["demanda_unidades"]
        .sum()
        .sort_values(["id_setor", "periodo_demanda"])
    )
    perfil: dict[tuple[int, int], float] = {}
    for setor, g in total.groupby("id_setor"):
        y = g["demanda_unidades"].to_numpy(dtype=float)
        if len(y) < 4 or np.any(y <= 0):
            continue
        t = np.arange(len(y), dtype=float)
        coef = np.polyfit(t, np.log(y), 1)
        destendencia = y / np.exp(np.polyval(coef, t))
        aux = g.copy()
        aux["_d"] = destendencia
        por_mes = aux.groupby(aux["periodo_demanda"].dt.month)["_d"].mean()
        por_mes = por_mes / por_mes.mean()
        for m, v in por_mes.items():
            perfil[(int(setor), int(m))] = float(v)
    return perfil


def cota_semanal(cota_mensal: float, ano: int, mes: int) -> pd.DataFrame:
    """
    Rateio da cota mensal entre as semanas comerciais, proporcional ao numero
    de dias de cada semana (regra informada pelo grupo no Excel:
    "objetivo_mensal / nr_dias de cada semana").

    A soma das cotas semanais e exatamente a cota mensal.
    """
    semanas = semanas_do_mes(ano, mes)
    total = dias_no_mes(ano, mes)
    return pd.DataFrame([
        {
            "nr_semana": s.nr_semana,
            "periodo_demanda": s.periodo_demanda,
            "nr_dias": s.nr_dias,
            "cota_semanal": cota_mensal * s.nr_dias / total,
        }
        for s in semanas
    ])


def adicionar_metadados(df_cota: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta a data de disponibilizacao (fim do mes + latencia do pipeline)."""
    df = df_cota.copy()
    df["data_atualizacao_base"] = df["periodo_demanda"].apply(
        lambda p: _fim_do_mes(p.date() if hasattr(p, "date") else p) + timedelta(days=cfg.LATENCIA_DIAS)
    )
    return df


def _fim_do_mes(d: date) -> date:
    from calendar import monthrange
    return date(d.year, d.month, monthrange(d.year, d.month)[1])


if __name__ == "__main__":
    from .gerador import agregar_por_setor, gerar_demanda_mensal

    df_dem, dims, _ = gerar_demanda_mensal()
    por_setor = agregar_por_setor(df_dem, dims)
    cota = calcular_cota_mensal(por_setor)

    print(f"Linhas de cota: {len(cota):,}")
    print(f"  meses com bootstrap (janela < {cfg.COTA_JANELA_MESES}): "
          f"{cota[cota.bootstrap].periodo_demanda.nunique()}")
    print(f"  cota negativa: {(cota.cota_unidades < 0).sum()}")
    print("\nAmostra:")
    print(cota.head(5).to_string(index=False))

    # sanidade: cota / demanda realizada deve orbitar 1 + crescimento
    comp = cota.merge(
        por_setor.assign(periodo_demanda=pd.to_datetime(por_setor.periodo_demanda)),
        on=["periodo_demanda", "id_setor", "id_produto"], how="inner",
    )
    razao = (comp.cota_unidades / comp.demanda_unidades.replace(0, np.nan)).dropna()
    print(f"\nRazao cota/demanda: mediana {razao.median():.3f} "
          f"(esperado ~{1 + cfg.COTA_CRESCIMENTO:.3f})")
