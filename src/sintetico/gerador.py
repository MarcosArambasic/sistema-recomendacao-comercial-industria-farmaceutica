"""***************************************************************************************
SCRIPT    : gerador.py
OBJETIVO  : Gerador hierarquico de demanda mensal sintetica (secoes 2, 5-10 do
            prompt mestre). Produz fato_demanda_aberta_mensal no grao
            mes x produto x outlet.

MODELO (multiplicativo, top-down):

    demanda(produto, outlet, mes) =
          nivel_nacional(tipo_produto, mes)      -- ancora + tendencia propria
        / n_produtos(tipo)                       -- particao igual (decisao do grupo)
        x peso_territorio(uf, tipo)              -- escala REAL por UF (do Excel)
        x sazonalidade(uf, tipo, mes)            -- indice REAL por UF (do Excel)
        x perturbacao_sazonal(produto, mes)      -- diferencia SKUs entre si
        x ruido_ar1(produto, uf, mes)            -- choque comum da UF, CV residual
        x peso_outlet(outlet | uf)               -- Pareto, top20% ~ 65%
        x ruido_idiossincratico(par, mes)        -- variacao entre outlets

    O rateio e top-down com renormalizacao, entao valem POR CONSTRUCAO:
        soma(outlets de uma UF) = total da UF
        soma(produtos)          = total do mercado

DECISOES DE MODELAGEM (justificadas no diagnostico):
  * Multiplicativo e nao aditivo: a sazonalidade real vem como INDICE
    (demanda/media) e a variabilidade como CV - ambos relativos. Um modelo
    aditivo daria residuos de mesmo tamanho absoluto em SP e em RR, que
    diferem por um fator de 318x. O modelo multiplicativo equivale ao
    aditivo classico Yt = Tt + St + Rt aplicado sobre log(Y).
  * Proprios e concorrentes tem tendencias OPOSTAS (-9,09% vs +47,47%). O
    mercado (+35,14%) EMERGE da soma, em vez de ser imposto - e isso que
    reproduz a erosao de market share observada (25,85% -> 11,70%).
  * O ruido injetado usa o CV RESIDUAL (4,40% / 7,26%), nao o CV bruto
    (9,69% / 26,03%), porque o bruto ja embute tendencia e sazonalidade.
  * AR(1) com phi=0,57 porque o residuo real tem autocorrelacao lag-1
    positiva - ruido i.i.d. produziria serie sem memoria.

AUTOR     : Marcos Arambasic 
DATA      : AGO/2026
****************************************************************************************/
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import config as cfg
from .calendario import meses_do_periodo
from .calibracao import carregar_calibracao


# ============================================================================
# Estruturas auxiliares
# ============================================================================
@dataclass
class Dimensoes:
    """Espelha o que o supabase/scripts/popular_banco.sql carregou no banco."""
    produtos: pd.DataFrame     # id_produto, nome, tipo_produto, grupo_produto, id_mercado
    ufs: pd.DataFrame          # id_uf, sigla_uf
    setores: pd.DataFrame      # id_setor, id_uf
    outlets: pd.DataFrame      # id_outlet, codigo_outlet, id_uf
    alocacao: pd.DataFrame     # id_alocacao, id_outlet, id_uf, id_setor


def montar_dimensoes(siglas_uf: list[str]) -> Dimensoes:
    """
    Reconstroi as dimensoes na MESMA ordem do popular_banco.sql
    (UFs ordenadas por sigla; outlets sequenciais a partir de 100000),
    para que os ids batam com os do banco.
    """
    siglas = sorted(siglas_uf)

    produtos = []
    id_produto = 1
    for i in range(1, cfg.N_PRODUTOS_PROPRIOS + 1):
        produtos.append({
            "id_produto": id_produto,
            "nome_sintetico": f"Produto Proprio {i:02d}",
            "grupo_produto": "Grupo A" if i <= 5 else "Grupo B",
            "tipo_produto": "PROPRIO",
            "id_mercado": 1,
        })
        id_produto += 1
    for i in range(1, cfg.N_PRODUTOS_CONCORRENTES + 1):
        produtos.append({
            "id_produto": id_produto,
            "nome_sintetico": f"Produto Concorrente {i:02d}",
            "grupo_produto": None,
            "tipo_produto": "CONCORRENTE",
            "id_mercado": 1,
        })
        id_produto += 1

    ufs = pd.DataFrame({
        "id_uf": range(1, len(siglas) + 1),
        "sigla_uf": siglas,
    })
    setores = pd.DataFrame({
        "id_setor": range(1, len(siglas) + 1),
        "id_uf": range(1, len(siglas) + 1),
    })

    outlets, alocacao = [], []
    id_outlet = 1
    codigo = 100000
    for row in ufs.itertuples():
        for _ in range(cfg.N_OUTLETS_POR_UF):
            outlets.append({
                "id_outlet": id_outlet,
                "codigo_outlet": f"{codigo:06d}",
                "id_uf": row.id_uf,
            })
            alocacao.append({
                "id_alocacao": id_outlet,      # 1:1 com outlet
                "id_outlet": id_outlet,
                "id_uf": row.id_uf,
                "id_setor": row.id_uf,          # 1 setor por UF
            })
            id_outlet += 1
            codigo += 1

    return Dimensoes(
        produtos=pd.DataFrame(produtos),
        ufs=ufs,
        setores=setores,
        outlets=pd.DataFrame(outlets),
        alocacao=pd.DataFrame(alocacao),
    )


# ============================================================================
# Componentes do modelo
# ============================================================================
def _sigma_lognormal(cv: float) -> float:
    """sigma da log-normal que produz o CV desejado com media 1."""
    return float(np.sqrt(np.log(1.0 + cv ** 2)))


def _ruido_ar1(shape: tuple, cv: float, phi: float, rng: np.random.Generator,
               eixo_tempo: int = -1, estado: dict | None = None,
               chave: str | None = None) -> np.ndarray:
    """
    Ruido multiplicativo log-normal com media 1, CV alvo e autocorrelacao AR(1)
    ao longo do eixo temporal.

    z_t = phi*z_{t-1} + sqrt(1-phi^2)*eps_t   -> z ~ N(0,1) estacionario
    ruido = exp(sigma*z - sigma^2/2)          -> E[ruido] = 1
    """
    sigma = _sigma_lognormal(cv)
    n_t = shape[eixo_tempo]
    z = np.empty(shape, dtype=np.float64)

    idx = [slice(None)] * len(shape)
    idx[eixo_tempo] = 0
    z[tuple(idx)] = rng.standard_normal(np.delete(np.array(shape), eixo_tempo))

    coef = np.sqrt(1.0 - phi ** 2)
    for t in range(1, n_t):
        atual, anterior = list(idx), list(idx)
        atual[eixo_tempo] = t
        anterior[eixo_tempo] = t - 1
        z[tuple(atual)] = (
            phi * z[tuple(anterior)]
            + coef * rng.standard_normal(np.delete(np.array(shape), eixo_tempo))
        )

    if estado is not None:
        # guarda o ultimo z de cada serie: e o ponto de partida para estender
        # o processo AR(1) para meses posteriores (extensao.py)
        ultimo = [slice(None)] * len(shape)
        ultimo[eixo_tempo] = n_t - 1
        estado[chave] = z[tuple(ultimo)].copy()

    return np.exp(sigma * z - sigma ** 2 / 2.0)


def _nivel_nacional(meses: list[date], ancora_mes: float, cagr: float,
                    ancora_ano: int) -> np.ndarray:
    """
    Serie nacional mensal do agregado (proprios ou concorrentes), com
    tendencia exponencial calibrada para que a MEDIA do ano-ancora seja
    exatamente `ancora_mes`.
    """
    idx = np.array([(m.year - ancora_ano) * 12 + (m.month - 1) for m in meses], dtype=float)
    bruto = (1.0 + cagr) ** (idx / 12.0)

    mask_ancora = np.array([m.year == ancora_ano for m in meses])
    if mask_ancora.any():
        fator = ancora_mes / bruto[mask_ancora].mean()
    else:                                   # ano-ancora fora do periodo
        fator = ancora_mes
    return bruto * fator


def _matriz_sazonal(perfil: dict[str, dict[str, float]], siglas: list[str],
                    meses: list[date]) -> np.ndarray:
    """(n_uf, n_meses) com o indice sazonal real de cada UF por mes-calendario."""
    out = np.ones((len(siglas), len(meses)), dtype=np.float64)
    for i, uf in enumerate(siglas):
        p = perfil.get(uf, {})
        for j, m in enumerate(meses):
            out[i, j] = float(p.get(str(m.month), p.get(m.month, 1.0)))
    return out


def _perturbacao_sazonal_produto(n_produtos: int, meses: list[date],
                                 amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """
    (n_produtos, n_meses) - faz cada SKU ter um desvio sazonal proprio em
    torno do indice agregado (exigencia da secao 6 do prompt: "nao aplicar
    simplesmente um fator fixo para todos os produtos").

    Normalizado por produto para que a media dos 12 meses-calendario seja 1,
    o que impede a perturbacao de deslocar o nivel anual do produto.
    """
    por_mes_cal = 1.0 + rng.uniform(-amplitude, amplitude, size=(n_produtos, 12))
    por_mes_cal /= por_mes_cal.mean(axis=1, keepdims=True)
    meses_cal = np.array([m.month - 1 for m in meses])
    return por_mes_cal[:, meses_cal]


def _pesos_outlet_pareto(n_por_uf: int, n_uf: int, sigma: float,
                         rng: np.random.Generator) -> np.ndarray:
    """
    (n_uf, n_por_uf) com os pesos relativos de cada outlet dentro da sua UF,
    normalizados para somar 1 por UF. Log-normal com sigma calibrado para
    top 20% ~ 65% do volume (decisao do grupo).

    Os pesos sao ESTAVEIS no tempo: um outlet grande e grande em todos os
    meses. A variacao mes a mes vem do ruido idiossincratico, nao daqui.
    """
    pesos = rng.lognormal(mean=0.0, sigma=sigma, size=(n_uf, n_por_uf))
    return pesos / pesos.sum(axis=1, keepdims=True)


def _mascara_esparsidade(pesos_outlet: np.ndarray, n_produtos: int,
                         rng: np.random.Generator) -> np.ndarray:
    """
    (n_produtos, n_uf, n_por_uf) booleana - quais pares produto x outlet estao
    ativos. ESTRUTURAL e estavel no tempo: um produto esta ou nao esta num
    outlet; nao pisca de mes para mes (isso destruiria a serie temporal).

    A probabilidade de presenca cresce com o tamanho do outlet - PDVs maiores
    carregam mais SKUs, que e o comportamento real. Com p(u) = min + (max-min)*u^2
    sobre o rank percentil u, a densidade media resulta em
    min + (max-min)/3 = 0,10 + 0,30 = 0,40, batendo com DENSIDADE_ALVO.
    """
    n_uf, n_por_uf = pesos_outlet.shape

    # rank percentil do outlet dentro da UF (0 = menor, 1 = maior)
    ordem = np.argsort(np.argsort(pesos_outlet, axis=1), axis=1)
    percentil = ordem / max(n_por_uf - 1, 1)

    lo, hi = cfg.DENSIDADE_MIN_OUTLET, cfg.DENSIDADE_MAX_OUTLET
    gama = (hi - lo) / max(cfg.DENSIDADE_ALVO - lo, 1e-9) - 1.0   # = 2 nos defaults
    prob = lo + (hi - lo) * percentil ** gama

    sorteio = rng.random(size=(n_produtos, n_uf, n_por_uf))
    mascara = sorteio < prob[None, :, :]

    # garante cobertura minima: todo (produto, UF) precisa de outlets ativos,
    # senao a demanda da UF nao teria onde ser distribuida
    minimo = max(5, int(0.05 * n_por_uf))
    for p in range(n_produtos):
        for u in range(n_uf):
            if mascara[p, u].sum() < minimo:
                maiores = np.argsort(pesos_outlet[u])[::-1][:minimo]
                mascara[p, u, maiores] = True
    return mascara


# ============================================================================
# Gerador principal
# ============================================================================
def gerar_demanda_mensal(seed: int | None = None, calib: dict | None = None,
                         periodo_inicio: str | None = None,
                         periodo_fim: str | None = None) -> tuple[pd.DataFrame, Dimensoes, dict]:
    """
    Gera fato_demanda_aberta_mensal no grao mes x produto x outlet.

    Retorna (dataframe_longo, dimensoes, diagnostico_da_geracao).
    """
    df, dims, diag, _ = gerar_com_estado(seed, calib, periodo_inicio, periodo_fim)
    return df, dims, diag


def gerar_com_estado(seed: int | None = None, calib: dict | None = None,
                     periodo_inicio: str | None = None, periodo_fim: str | None = None,
                     ) -> tuple[pd.DataFrame, Dimensoes, dict, dict]:
    """
    Igual a gerar_demanda_mensal, mas devolve tambem o ESTADO interno do
    modelo (pesos de outlet, mascara de esparsidade, perturbacao sazonal por
    SKU e o ultimo valor de cada processo AR(1)). Com ele, extensao.py gera
    os meses seguintes como continuacao exata da mesma serie, sem alterar
    nenhum valor ja gerado.
    """
    estado: dict = {}
    calib = calib or carregar_calibracao()
    rng = np.random.default_rng(cfg.RANDOM_SEED if seed is None else seed)

    meses = meses_do_periodo(periodo_inicio or cfg.PERIODO_INICIO,
                             periodo_fim or cfg.PERIODO_FIM)
    n_t = len(meses)

    siglas = sorted(calib["peso_territorio"]["proprios"].keys())
    dims = montar_dimensoes(siglas)
    n_uf = len(siglas)
    n_out = cfg.N_OUTLETS_POR_UF

    prod = dims.produtos
    idx_prop = prod.index[prod.tipo_produto == "PROPRIO"].to_numpy()
    idx_conc = prod.index[prod.tipo_produto == "CONCORRENTE"].to_numpy()
    n_prod = len(prod)

    # ---------------- 1) nivel nacional por tipo, com tendencias opostas ----
    if cfg.USAR_TENDENCIA_CALIBRADA:
        chave = ("crescimento_para_cagr_excel" if cfg.ALVO_TENDENCIA == "cagr_excel"
                 else "crescimento_log_linear")
        taxas = calib["tendencia"][chave]
        cagr_prop, cagr_conc = taxas["proprios"], taxas["concorrentes"]
    else:
        cagr_prop, cagr_conc = cfg.CAGR_PROPRIOS, cfg.CAGR_CONCORRENTES

    nac_prop = _nivel_nacional(meses, cfg.ANCORA_PROPRIOS_MES, cagr_prop, cfg.ANCORA_ANO)
    nac_conc = _nivel_nacional(meses, cfg.ANCORA_CONCORRENTES_MES, cagr_conc, cfg.ANCORA_ANO)

    # particao igual entre os produtos de cada tipo (decisao do grupo)
    nivel_produto = np.zeros((n_prod, n_t))
    nivel_produto[idx_prop, :] = nac_prop / len(idx_prop)
    nivel_produto[idx_conc, :] = nac_conc / len(idx_conc)

    # ---------------- 2) peso e sazonalidade por territorio ----------------
    peso_uf = np.zeros((n_prod, n_uf))
    peso_uf[idx_prop, :] = np.array([calib["peso_territorio"]["proprios"][u] for u in siglas])
    peso_uf[idx_conc, :] = np.array([calib["peso_territorio"]["concorrentes"][u] for u in siglas])

    saz_prop = _matriz_sazonal(calib["sazonalidade"]["proprios"], siglas, meses)
    saz_merc = _matriz_sazonal(calib["sazonalidade"]["mercado"], siglas, meses)
    sazonal = np.zeros((n_prod, n_uf, n_t))
    sazonal[idx_prop, :, :] = saz_prop[None, :, :]
    sazonal[idx_conc, :, :] = saz_merc[None, :, :]

    # perturbacao sazonal por SKU
    pert = _perturbacao_sazonal_produto(n_prod, meses, cfg.PERTURBACAO_SAZONAL_PRODUTO, rng)
    sazonal *= pert[:, None, :]
    pert_cal = np.ones((n_prod, 12))
    for j, m in enumerate(meses):
        pert_cal[:, m.month - 1] = pert[:, j]

    # ---------------- 3) ruido AR(1) comum por (produto, UF) ---------------
    ruido = np.empty((n_prod, n_uf, n_t))
    ruido[idx_prop] = _ruido_ar1((len(idx_prop), n_uf, n_t),
                                 cfg.CV_RESIDUAL_PROPRIOS, cfg.AR1_PHI, rng,
                                 estado=estado, chave="z_uf_proprios")
    ruido[idx_conc] = _ruido_ar1((len(idx_conc), n_uf, n_t),
                                 cfg.CV_RESIDUAL_CONCORRENTES, cfg.AR1_PHI, rng,
                                 estado=estado, chave="z_uf_concorrentes")

    # ---------------- 4) alvo por (produto, UF, mes) -----------------------
    alvo = nivel_produto[:, None, :] * peso_uf[:, :, None] * sazonal * ruido

    # ---------------- 5) outlets: Pareto + esparsidade ---------------------
    pesos_out = _pesos_outlet_pareto(n_out, n_uf, cfg.SIGMA_LOGNORMAL_OUTLET, rng)
    mascara = _mascara_esparsidade(pesos_out, n_prod, rng)

    p_idx, u_idx, o_idx = np.nonzero(mascara)
    n_pares = p_idx.size

    peso_par = pesos_out[u_idx, o_idx]
    # renormaliza dentro de cada (produto, UF) considerando so os ativos,
    # para que a soma dos outlets reproduza exatamente o alvo da UF
    grupo = p_idx * n_uf + u_idx
    soma_grupo = np.bincount(grupo, weights=peso_par, minlength=n_prod * n_uf)
    peso_par = peso_par / soma_grupo[grupo]

    # ---------------- 6) ruido idiossincratico por par ---------------------
    ruido_par = _ruido_ar1((n_pares, n_t), cfg.CV_IDIOSSINCRATICO_OUTLET, cfg.AR1_PHI, rng,
                           estado=estado, chave="z_par")

    # ---------------- 7) demanda + renormalizacao (reconciliacao exata) ----
    alvo_par = alvo[p_idx, u_idx, :]                      # (n_pares, n_t)
    bruta = alvo_par * peso_par[:, None] * ruido_par

    soma_bruta = np.zeros((n_prod * n_uf, n_t))
    for t in range(n_t):
        soma_bruta[:, t] = np.bincount(grupo, weights=bruta[:, t], minlength=n_prod * n_uf)

    alvo_grupo = alvo.reshape(n_prod * n_uf, n_t)
    with np.errstate(divide="ignore", invalid="ignore"):
        fator = np.where(soma_bruta > 0, alvo_grupo / soma_bruta, 0.0)
    demanda = bruta * fator[grupo, :]

    # unidades sao inteiras
    demanda = np.rint(demanda)
    demanda[demanda < 0] = 0.0

    # ---------------- 8) formato longo ------------------------------------
    id_produto = prod.id_produto.to_numpy()[p_idx]
    id_uf = dims.ufs.id_uf.to_numpy()[u_idx]
    id_outlet = (u_idx * n_out + o_idx) + 1

    periodos = np.array([np.datetime64(m) for m in meses])
    atualizacao = np.array([
        np.datetime64(_fim_do_mes(m) + timedelta(days=cfg.LATENCIA_DIAS)) for m in meses
    ])

    df = pd.DataFrame({
        "data_atualizacao_base": np.repeat(atualizacao[None, :], n_pares, axis=0).ravel(),
        "periodo_demanda": np.repeat(periodos[None, :], n_pares, axis=0).ravel(),
        "id_produto": np.repeat(id_produto, n_t),
        "id_outlet": np.repeat(id_outlet, n_t),
        "id_uf": np.repeat(id_uf, n_t),
        "demanda_unidades": demanda.ravel(),
    })

    diagnostico = {
        "n_meses": n_t,
        "n_produtos": n_prod,
        "n_ufs": n_uf,
        "n_outlets": n_uf * n_out,
        "n_pares_ativos": int(n_pares),
        "densidade": float(n_pares / (n_prod * n_uf * n_out)),
        "n_linhas": int(df.shape[0]),
        "seed": cfg.RANDOM_SEED if seed is None else seed,
        "periodo": [str(meses[0]), str(meses[-1])],
        "crescimento_aplicado": {"proprios": cagr_prop, "concorrentes": cagr_conc},
    }
    estado.update({
        "seed": cfg.RANDOM_SEED if seed is None else seed,
        "periodo_inicio": str(meses[0])[:7], "periodo_fim": str(meses[-1])[:7],
        "siglas": siglas, "idx_prop": idx_prop, "idx_conc": idx_conc,
        "cagr_prop": cagr_prop, "cagr_conc": cagr_conc,
        "peso_uf": peso_uf, "pert_cal": pert_cal,
        "p_idx": p_idx, "u_idx": u_idx, "o_idx": o_idx, "peso_par": peso_par,
        "grupo": grupo,
    })
    return df, dims, diagnostico, estado


def _fim_do_mes(d: date) -> date:
    from calendar import monthrange
    return date(d.year, d.month, monthrange(d.year, d.month)[1])


def agregar_por_setor(df_demanda: pd.DataFrame, dims: Dimensoes,
                      apenas_proprios: bool | None = None) -> pd.DataFrame:
    """
    Consolida a demanda por (periodo, setor, produto) - insumo do calculo da
    cota e base da validacao de reconciliacao soma(outlets) = setor.

    apenas_proprios: se True, filtra so os produtos PROPRIO. A empresa nao
    define meta para produto de concorrente, entao a cota so faz sentido para
    o portfolio proprio. Default vem de config.COTA_APENAS_PROPRIOS.
    """
    if apenas_proprios is None:
        apenas_proprios = cfg.COTA_APENAS_PROPRIOS

    df = df_demanda
    if apenas_proprios:
        proprios = dims.produtos.loc[
            dims.produtos.tipo_produto == "PROPRIO", "id_produto"
        ]
        df = df[df.id_produto.isin(proprios)]

    mapa = dims.alocacao[["id_outlet", "id_setor"]]
    return (
        df.merge(mapa, on="id_outlet", how="left")
        .groupby(["periodo_demanda", "id_setor", "id_produto"], as_index=False)["demanda_unidades"]
        .sum()
    )


if __name__ == "__main__":
    import time

    t0 = time.time()
    df, dims, diag = gerar_demanda_mensal()
    print(f"Gerado em {time.time() - t0:.1f}s")
    for k, v in diag.items():
        print(f"  {k}: {v}")
    print("\nAmostra:")
    print(df.head(5).to_string(index=False))
    print(f"\nDemanda total no periodo: {df.demanda_unidades.sum():,.0f} unidades")
