"""***************************************************************************************
SCRIPT    : extensao.py
OBJETIVO  : Gerar a demanda de meses POSTERIORES a base historica, como
            continuacao exata do mesmo modelo.

            E o que permite ao executar_semana.py rodar qualquer semana futura
            sem que a demanda mensal exista: a demanda do mes e criada pelo
            modelo calibrado, repartida nas semanas, e so vira linha da
            fato_demanda_aberta_mensal na ULTIMA semana (consolidacao).

COMO A CONTINUIDADE E GARANTIDA:
    1. o gerador e re-executado com o MESMO periodo e a MESMA semente da base
       historica -> reproduz a base bit a bit (verificado: diferenca maxima 0)
       e devolve o estado interno: pesos e mascara de outlets, perturbacao
       sazonal de cada SKU e o ultimo valor z de cada processo AR(1);
    2. para cada mes seguinte, os processos AR(1) avancam um passo:
           z(t) = phi * z(t-1) + sqrt(1 - phi^2) * eps(t)
       com eps(t) sorteado de uma semente propria do MES
       (SeedSequence([seed, ano*100+mes])). Por isso o mes M sai sempre
       igual, nao importa quantas vezes, em que ordem ou em que dia ele e
       gerado;
    3. nivel nacional (tendencia), peso do territorio, sazonalidade e
       renormalizacao seguem exatamente as formulas do gerador.

    Os outlets ativos, a concentracao Pareto e a erosao de share continuam
    as mesmas - o mes novo e indistinguivel de um mes que tivesse sido gerado
    junto com a base.

CACHE: o estado e salvo em data/calibracao/estado_gerador_*.pkl (refazer custa
~7 s); e invalidado se a calibracao mudar.

AUTOR     : Marcos Arambasic
DATA      : OUT/2026
****************************************************************************************/
"""

from __future__ import annotations

import hashlib
import pickle
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import config as cfg
from .calibracao import carregar_calibracao
from .gerador import (_fim_do_mes, _matriz_sazonal, _nivel_nacional, _sigma_lognormal,
                      gerar_com_estado, montar_dimensoes)

SAL_EXTENSAO = 7919          # separa as sementes da extensao das do gerador


def _proximo_mes(d: date) -> date:
    return date(d.year + (d.month == 12), d.month % 12 + 1, 1)


def _assinatura_calibracao() -> str:
    p = cfg.DIR_CALIBRACAO / "calibracao.json"
    return hashlib.sha1(p.read_bytes()).hexdigest()[:10] if p.exists() else "sem"


def carregar_estado(periodo_inicio: str, periodo_fim: str, seed: int | None = None) -> dict:
    """Estado do gerador ao fim da base historica (com cache em disco)."""
    seed = cfg.RANDOM_SEED if seed is None else seed
    cfg.DIR_CALIBRACAO.mkdir(parents=True, exist_ok=True)
    arq = cfg.DIR_CALIBRACAO / (f"estado_gerador_s{seed}_{periodo_inicio}_{periodo_fim}_"
                               f"{_assinatura_calibracao()}.pkl")
    if arq.exists():
        with open(arq, "rb") as f:
            return pickle.load(f)
    _, _, _, estado = gerar_com_estado(seed=seed, periodo_inicio=periodo_inicio,
                                       periodo_fim=periodo_fim)
    with open(arq, "wb") as f:
        pickle.dump(estado, f)
    return estado


def _avancar_ar(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    phi = cfg.AR1_PHI
    return phi * z + np.sqrt(1.0 - phi ** 2) * rng.standard_normal(z.shape)


def gerar_meses(estado: dict, ate: date, calib: dict | None = None) -> pd.DataFrame:
    """
    Demanda mensal (mesmo formato da fato_demanda_aberta_mensal) de todos os
    meses entre o fim da base e `ate`, inclusive.
    """
    calib = calib or carregar_calibracao()
    fim = date.fromisoformat(estado["periodo_fim"] + "-01")
    ate = date(ate.year, ate.month, 1)
    if ate <= fim:
        return pd.DataFrame(columns=["data_atualizacao_base", "periodo_demanda", "id_produto",
                                     "id_outlet", "id_uf", "demanda_unidades"])

    siglas = estado["siglas"]
    dims = montar_dimensoes(siglas)
    prod = dims.produtos
    idx_prop, idx_conc = estado["idx_prop"], estado["idx_conc"]
    n_prod, n_uf, n_out = len(prod), len(siglas), cfg.N_OUTLETS_POR_UF
    p_idx, u_idx, o_idx = estado["p_idx"], estado["u_idx"], estado["o_idx"]
    peso_par, grupo = estado["peso_par"], estado["grupo"]

    z_prop = estado["z_uf_proprios"].copy()
    z_conc = estado["z_uf_concorrentes"].copy()
    z_par = estado["z_par"].copy()

    sig_prop = _sigma_lognormal(cfg.CV_RESIDUAL_PROPRIOS)
    sig_conc = _sigma_lognormal(cfg.CV_RESIDUAL_CONCORRENTES)
    sig_par = _sigma_lognormal(cfg.CV_IDIOSSINCRATICO_OUTLET)
    meses_ancora = [date(cfg.ANCORA_ANO, m, 1) for m in range(1, 13)]

    id_produto = prod.id_produto.to_numpy()[p_idx]
    id_uf = dims.ufs.id_uf.to_numpy()[u_idx]
    id_outlet = (u_idx * n_out + o_idx) + 1

    saidas = []
    m = _proximo_mes(fim)
    while m <= ate:
        rng = np.random.default_rng(np.random.SeedSequence([estado["seed"], SAL_EXTENSAO,
                                                            m.year * 100 + m.month]))
        z_prop = _avancar_ar(z_prop, rng)
        z_conc = _avancar_ar(z_conc, rng)
        z_par = _avancar_ar(z_par, rng)

        # 1) nivel nacional por tipo (mesma ancora e tendencia do gerador)
        nivel = np.zeros(n_prod)
        nivel[idx_prop] = _nivel_nacional(meses_ancora + [m], cfg.ANCORA_PROPRIOS_MES,
                                          estado["cagr_prop"], cfg.ANCORA_ANO)[-1] / len(idx_prop)
        nivel[idx_conc] = _nivel_nacional(meses_ancora + [m], cfg.ANCORA_CONCORRENTES_MES,
                                          estado["cagr_conc"], cfg.ANCORA_ANO)[-1] / len(idx_conc)

        # 2) sazonalidade do territorio x perturbacao do SKU
        sazonal = np.zeros((n_prod, n_uf))
        sazonal[idx_prop] = _matriz_sazonal(calib["sazonalidade"]["proprios"], siglas, [m])[:, 0]
        sazonal[idx_conc] = _matriz_sazonal(calib["sazonalidade"]["mercado"], siglas, [m])[:, 0]
        sazonal *= estado["pert_cal"][:, m.month - 1][:, None]

        # 3) ruido AR(1) por (produto, UF)
        ruido = np.empty((n_prod, n_uf))
        ruido[idx_prop] = np.exp(sig_prop * z_prop - sig_prop ** 2 / 2)
        ruido[idx_conc] = np.exp(sig_conc * z_conc - sig_conc ** 2 / 2)

        alvo = nivel[:, None] * estado["peso_uf"] * sazonal * ruido          # (n_prod, n_uf)

        # 4) outlets + ruido idiossincratico + renormalizacao exata
        bruta = alvo[p_idx, u_idx] * peso_par * np.exp(sig_par * z_par - sig_par ** 2 / 2)
        soma = np.bincount(grupo, weights=bruta, minlength=n_prod * n_uf)
        with np.errstate(divide="ignore", invalid="ignore"):
            fator = np.where(soma > 0, alvo.reshape(-1) / soma, 0.0)
        demanda = np.clip(np.rint(bruta * fator[grupo]), 0, None)

        saidas.append(pd.DataFrame({
            "data_atualizacao_base": pd.Timestamp(_fim_do_mes(m) + timedelta(days=cfg.LATENCIA_DIAS)),
            "periodo_demanda": pd.Timestamp(m),
            "id_produto": id_produto,
            "id_outlet": id_outlet,
            "id_uf": id_uf,
            "demanda_unidades": demanda,
        }))
        m = _proximo_mes(m)

    return pd.concat(saidas, ignore_index=True)
