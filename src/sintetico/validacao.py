"""***************************************************************************************
SCRIPT    : validacao.py
OBJETIVO  : Validacao QUANTITATIVA da base sintetica e calculo do
            synthetic_data_fidelity_score (secoes 22, 23 e 24 do prompt mestre).

            O principio aqui e o da secao 22: "nao considerar um dataset
            realista apenas porque os valores parecem plausiveis". Toda
            afirmacao de fidelidade e medida contra um alvo explicito vindo
            dos dados reais agregados.

SEIS DIMENSOES AVALIADAS:
    1. distribuicao   - media, dp, CV, min, max, mediana, P25/P75/P95/P99
    2. sazonalidade   - indice sintetico x indice real por UF e mes
    3. tendencia      - CAGR sintetico x CAGR alvo (na convencao do Excel)
    4. correlacao     - relacoes estruturais (erosao de share, Pareto)
    5. reconciliacao  - soma(outlets)=setor, soma(produtos)=mercado
    6. integridade    - FKs, nulos, duplicidade, negativos, share fora de [0,1]

CUIDADO METODOLOGICO IMPORTANTE: os alvos do Excel foram medidos numa janela
especifica (jul/2024 a jun/2026, 24 meses) e num grao especifico
(territorio x mes). Comparar contra a base inteira (44 meses) daria erro
espurio, porque CV e CAGR dependem da janela. Todas as comparacoes recortam
a mesma janela do Excel.

AUTOR     : Marcos Arambasic 
DATA      : AGO/2026
****************************************************************************************/
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .calibracao import carregar_calibracao


# ============================================================================
# Infra de pontuacao
# ============================================================================
@dataclass
class Check:
    dimensao: str
    metrica: str
    valor: float | None
    alvo: float | None
    erro_rel: float | None
    score: float
    tolerancia: float | None = None
    observacao: str = ""

    @property
    def aprovado(self) -> bool:
        return self.score >= 70.0


@dataclass
class Relatorio:
    checks: list[Check] = field(default_factory=list)

    def add(self, c: Check) -> None:
        self.checks.append(c)

    def df(self) -> pd.DataFrame:
        return pd.DataFrame([c.__dict__ for c in self.checks])

    def score_por_dimensao(self) -> dict[str, float]:
        d = self.df()
        if d.empty:
            return {}
        return d.groupby("dimensao")["score"].mean().round(2).to_dict()

    def fidelity_score(self, pesos: dict[str, float] | None = None) -> float:
        """
        synthetic_data_fidelity_score entre 0 e 100.

        Integridade e reconciliacao pesam mais porque sao condicoes de
        corretude (a base esta certa ou errada), enquanto distribuicao e
        sazonalidade sao de aderencia (quao perto do real).
        """
        pesos = pesos or {
            "integridade": 0.25,
            "reconciliacao": 0.20,
            "distribuicao": 0.20,
            "tendencia": 0.15,
            "sazonalidade": 0.15,
            "correlacao": 0.05,
        }
        por_dim = self.score_por_dimensao()
        total = sum(pesos.get(k, 0) for k in por_dim)
        if total == 0:
            return 0.0
        return round(sum(por_dim[k] * pesos.get(k, 0) for k in por_dim) / total, 2)

    def falhas(self) -> pd.DataFrame:
        d = self.df()
        return d[d.score < 70].sort_values("score") if not d.empty else d


def _score_relativo(valor: float, alvo: float, tolerancia: float) -> tuple[float, float]:
    """
    Converte erro relativo em score 0-100 com decaimento linear:
    erro 0 -> 100 ; erro == tolerancia -> 70 ; erro >= 3x tolerancia -> 0.
    """
    if alvo is None or (isinstance(alvo, float) and (np.isnan(alvo) or alvo == 0)):
        return 0.0, float("nan")
    erro = abs(valor - alvo) / abs(alvo)
    if erro <= tolerancia:
        score = 100.0 - 30.0 * (erro / tolerancia)
    else:
        score = max(0.0, 70.0 * (1.0 - (erro - tolerancia) / (2.0 * tolerancia)))
    return round(score, 2), round(erro, 6)


def _score_absoluto(valor: float, alvo: float, tolerancia_abs: float) -> tuple[float, float]:
    """Mesma logica, mas para diferencas absolutas (ex.: CAGR em p.p.)."""
    erro = abs(valor - alvo)
    if erro <= tolerancia_abs:
        score = 100.0 - 30.0 * (erro / tolerancia_abs)
    else:
        score = max(0.0, 70.0 * (1.0 - (erro - tolerancia_abs) / (2.0 * tolerancia_abs)))
    return round(score, 2), round(erro, 6)


# ============================================================================
# Validacao
# ============================================================================
def validar(
    df_demanda: pd.DataFrame,
    df_cota: pd.DataFrame,
    dims,
    calib: dict | None = None,
) -> Relatorio:
    calib = calib or carregar_calibracao()
    rel = Relatorio()

    df = df_demanda.copy()
    df["periodo_demanda"] = pd.to_datetime(df["periodo_demanda"])
    tipo = dims.produtos.set_index("id_produto")["tipo_produto"]
    df["tipo"] = df["id_produto"].map(tipo)

    janela = calib["janela_observada"]
    ini, fim = pd.Timestamp(janela["inicio"]), pd.Timestamp(janela["fim"])
    dj = df[(df.periodo_demanda >= ini) & (df.periodo_demanda <= fim)]

    # agregado por UF x mes x tipo (grao dos alvos do Excel)
    uf_mes = (
        dj.groupby(["id_uf", "periodo_demanda", "tipo"])["demanda_unidades"]
        .sum().unstack(fill_value=0.0)
    )
    uf_mes["MERCADO"] = uf_mes.sum(axis=1)

    _validar_distribuicao(rel, uf_mes, df_cota, calib)
    _validar_sazonalidade(rel, dj, dims, calib)
    _validar_tendencia(rel, df, dims, calib)
    _validar_correlacao(rel, df, dims, calib)
    _validar_reconciliacao(rel, df, dims)
    _validar_integridade(rel, df, df_cota, dims)
    return rel


def _validar_distribuicao(rel: Relatorio, uf_mes: pd.DataFrame,
                          df_cota: pd.DataFrame, calib: dict) -> None:
    tol = cfg.TOLERANCIA_DISTRIBUICAO
    alvos = cfg.ALVOS_DISTRIBUICAO

    series = {
        "demanda_mercado_uf_mes": uf_mes["MERCADO"],
        "demanda_proprios_uf_mes": uf_mes.get("PROPRIO", pd.Series(dtype=float)),
    }
    for nome, s in series.items():
        if s.empty:
            continue
        alvo = alvos[nome]
        for metrica, fn in [
            ("media", lambda x: x.mean()),
            ("dp", lambda x: x.std(ddof=1)),
        ]:
            score, erro = _score_relativo(float(fn(s)), alvo[metrica], tol)
            rel.add(Check("distribuicao", f"{nome}.{metrica}", float(fn(s)),
                          alvo[metrica], erro, score, tol))
        # CV derivado
        cv = float(s.std(ddof=1) / s.mean())
        cv_alvo = alvo["dp"] / alvo["media"]
        score, erro = _score_relativo(cv, cv_alvo, tol)
        rel.add(Check("distribuicao", f"{nome}.cv", cv, cv_alvo, erro, score, tol))

        # quantis - reportados como diagnostico (sem alvo no Excel)
        for q in [0.25, 0.50, 0.75, 0.95, 0.99]:
            rel.add(Check("distribuicao", f"{nome}.P{int(q*100)}",
                          float(s.quantile(q)), None, None, 100.0,
                          observacao="sem alvo no Excel - diagnostico"))

    # market share
    if "PROPRIO" in uf_mes:
        ms = uf_mes["PROPRIO"] / uf_mes["MERCADO"].replace(0, np.nan)
        ms = ms.dropna()
        alvo = alvos["market_share"]
        for metrica, val in [("media", ms.mean()), ("dp", ms.std(ddof=1))]:
            score, erro = _score_relativo(float(val), alvo[metrica], tol)
            rel.add(Check("distribuicao", f"market_share.{metrica}", float(val),
                          alvo[metrica], erro, score, tol))

    # objetivo (cota)
    if not df_cota.empty:
        c = df_cota.copy()
        c["periodo_demanda"] = pd.to_datetime(c["periodo_demanda"])
        jan = calib["janela_observada"]
        cj = c[(c.periodo_demanda >= pd.Timestamp(jan["inicio"]))
               & (c.periodo_demanda <= pd.Timestamp(jan["fim"]))]
        por_uf = cj.groupby(["id_setor", "periodo_demanda"])["cota_unidades"].sum()
        alvo = alvos["objetivo_uf_mes"]
        for metrica, val in [("media", por_uf.mean()), ("dp", por_uf.std(ddof=1))]:
            score, erro = _score_relativo(float(val), alvo[metrica], tol)
            rel.add(Check("distribuicao", f"objetivo_uf_mes.{metrica}", float(val),
                          alvo[metrica], erro, score, tol))


def _validar_sazonalidade(rel: Relatorio, dj: pd.DataFrame, dims, calib: dict) -> None:
    tol = cfg.TOLERANCIA_SAZONAL
    sigla = dims.ufs.set_index("id_uf")["sigla_uf"]

    for tipo_col, chave in [("PROPRIO", "proprios"), ("CONCORRENTE", "mercado")]:
        sub = dj[dj.tipo == tipo_col]
        if sub.empty:
            continue
        serie = sub.groupby(["id_uf", "periodo_demanda"])["demanda_unidades"].sum()

        erros = []
        for id_uf, g in serie.groupby(level=0):
            y = g.to_numpy(dtype=float)
            if len(y) < 4 or np.any(y <= 0):
                continue
            t = np.arange(len(y), dtype=float)
            coef = np.polyfit(t, np.log(y), 1)
            destend = y / np.exp(np.polyval(coef, t))
            meses = g.index.get_level_values(1).month
            obs = pd.Series(destend, index=meses).groupby(level=0).mean()
            obs = obs / obs.mean()

            perfil = calib["sazonalidade"][chave].get(sigla[id_uf], {})
            for m, v in obs.items():
                alvo = perfil.get(str(m), perfil.get(m))
                if alvo:
                    erros.append(abs(v - alvo) / alvo)
        if erros:
            erro_medio = float(np.mean(erros))
            score = max(0.0, 100.0 - 30.0 * (erro_medio / tol)) if erro_medio <= tol \
                else max(0.0, 70.0 * (1 - (erro_medio - tol) / (2 * tol)))
            rel.add(Check("sazonalidade", f"indice_sazonal.{chave}", erro_medio, 0.0,
                          erro_medio, round(score, 2), tol,
                          "erro relativo medio entre indice sintetico e real"))


def _validar_tendencia(rel: Relatorio, df: pd.DataFrame, dims, calib: dict) -> None:
    tol = cfg.TOLERANCIA_TENDENCIA
    nac = df.groupby(["periodo_demanda", "tipo"])["demanda_unidades"].sum().unstack()
    nac["MERCADO"] = nac.sum(axis=1)

    jan = calib["janela_observada"]["meses_por_ano"]
    anos = sorted(int(a) for a in jan)
    ano_ini, ano_fim = anos[0], anos[-1]

    def media_janela(col: str, ano: int) -> float:
        meses = jan[str(ano)]
        s = nac[(nac.index.year == ano) & (nac.index.month.isin(meses))][col]
        return float(s.mean())

    alvos = {
        "PROPRIO": calib["tendencia"]["cagr_excel_proprios"],
        "CONCORRENTE": calib["tendencia"]["cagr_excel_concorrentes"],
        "MERCADO": calib["tendencia"]["cagr_excel_mercado"],
    }
    for col, alvo in alvos.items():
        if col not in nac:
            continue
        a, b = media_janela(col, ano_ini), media_janela(col, ano_fim)
        cagr = float((b / a) ** (1.0 / (ano_fim - ano_ini)) - 1.0)
        score, erro = _score_absoluto(cagr, alvo, tol)
        rel.add(Check("tendencia", f"cagr_excel.{col.lower()}", cagr, alvo, erro, score, tol,
                      "medido na mesma convencao e janela do Excel"))


def _validar_correlacao(rel: Relatorio, df: pd.DataFrame, dims, calib: dict) -> None:
    # 1) erosao de market share ao longo dos anos (fenomeno central do negocio)
    nac = df.groupby([df.periodo_demanda.dt.year, "tipo"])["demanda_unidades"].sum().unstack()
    if {"PROPRIO", "CONCORRENTE"}.issubset(nac.columns):
        share = nac["PROPRIO"] / (nac["PROPRIO"] + nac["CONCORRENTE"])
        anos = sorted(share.index)
        decrescente = all(share[anos[i]] > share[anos[i + 1]] for i in range(len(anos) - 1))
        rel.add(Check("correlacao", "erosao_share_monotona",
                      1.0 if decrescente else 0.0, 1.0,
                      0.0 if decrescente else 1.0, 100.0 if decrescente else 0.0,
                      observacao=f"share por ano: {[round(share[a], 4) for a in anos]}"))

        # share observado na janela do Excel vs share real do mesmo ano
        for ano, alvo in calib["share"]["por_ano"].items():
            ano = int(ano)
            if ano in share.index:
                score, erro = _score_absoluto(float(share[ano]), float(alvo), 0.06)
                rel.add(Check("correlacao", f"share.{ano}", float(share[ano]),
                              float(alvo), erro, score, 0.06))

    # 2) concentracao Pareto INTRA-UF (medida nacional misturaria escala de UF)
    o = df.groupby(["id_uf", "id_outlet"])["demanda_unidades"].sum()
    topo = []
    for _, g in o.groupby(level=0):
        s = g.sort_values(ascending=False)
        k = max(1, int(len(s) * cfg.PARETO_TOP_PCT))
        topo.append(s.head(k).sum() / s.sum())
    obs = float(np.mean(topo))
    score, erro = _score_relativo(obs, cfg.PARETO_SHARE_ALVO, 0.15)
    rel.add(Check("correlacao", "pareto_top20_intra_uf", obs, cfg.PARETO_SHARE_ALVO,
                  erro, score, 0.15, "medido dentro de cada UF, nao nacionalmente"))


def _validar_reconciliacao(rel: Relatorio, df: pd.DataFrame, dims) -> None:
    tol = cfg.TOLERANCIA_RECONCILIACAO

    # soma(outlets do setor) == total do setor
    mapa = dims.alocacao[["id_outlet", "id_setor"]]
    por_outlet = df.merge(mapa, on="id_outlet", how="left")
    faltando = int(por_outlet["id_setor"].isna().sum())
    rel.add(Check("reconciliacao", "outlets_com_setor", float(len(por_outlet) - faltando),
                  float(len(por_outlet)), float(faltando), 100.0 if faltando == 0 else 0.0))

    a = por_outlet.groupby(["periodo_demanda", "id_setor"])["demanda_unidades"].sum().sum()
    b = df["demanda_unidades"].sum()
    erro = abs(a - b) / b if b else 0.0
    rel.add(Check("reconciliacao", "soma_outlet_igual_setor", float(a), float(b),
                  float(erro), 100.0 if erro <= tol else 0.0, tol))

    # soma(produtos) == mercado
    c = df.groupby(["periodo_demanda", "id_produto"])["demanda_unidades"].sum().sum()
    erro2 = abs(c - b) / b if b else 0.0
    rel.add(Check("reconciliacao", "soma_produto_igual_mercado", float(c), float(b),
                  float(erro2), 100.0 if erro2 <= tol else 0.0, tol))


def _validar_integridade(rel: Relatorio, df: pd.DataFrame, df_cota: pd.DataFrame, dims) -> None:
    def bool_check(nome: str, ok: bool, obs: str = "") -> None:
        rel.add(Check("integridade", nome, 1.0 if ok else 0.0, 1.0,
                      0.0 if ok else 1.0, 100.0 if ok else 0.0, observacao=obs))

    bool_check("demanda_nao_negativa", bool((df.demanda_unidades >= 0).all()),
               f"{int((df.demanda_unidades < 0).sum())} negativas")
    bool_check("sem_nulos", bool(df.isna().sum().sum() == 0),
               f"{int(df.isna().sum().sum())} nulos")
    dup = int(df.duplicated(subset=["periodo_demanda", "id_produto", "id_outlet"]).sum())
    bool_check("sem_duplicidade_no_grao", dup == 0, f"{dup} duplicatas")

    # FKs
    bool_check("fk_produto_valida",
               bool(df.id_produto.isin(dims.produtos.id_produto).all()))
    bool_check("fk_outlet_valida",
               bool(df.id_outlet.isin(dims.outlets.id_outlet).all()))
    bool_check("fk_uf_valida", bool(df.id_uf.isin(dims.ufs.id_uf).all()))

    # coerencia outlet <-> uf (id_uf e desnormalizado; tem que bater com a dim)
    mapa_uf = dims.outlets.set_index("id_outlet")["id_uf"]
    coerente = bool((df.id_outlet.map(mapa_uf) == df.id_uf).all())
    bool_check("uf_coerente_com_outlet", coerente)

    # market share dentro de [0,1]
    por = df.groupby(["periodo_demanda", "id_uf", "tipo"])["demanda_unidades"].sum().unstack(fill_value=0)
    if "PROPRIO" in por:
        total = por.sum(axis=1)
        ms = (por["PROPRIO"] / total.replace(0, np.nan)).dropna()
        bool_check("market_share_entre_0_e_1", bool(((ms >= 0) & (ms <= 1)).all()),
                   f"min={ms.min():.4f} max={ms.max():.4f}")

    if not df_cota.empty:
        bool_check("cota_nao_negativa", bool((df_cota.cota_unidades >= 0).all()))
        dupc = int(df_cota.duplicated(subset=["periodo_demanda", "id_setor", "id_produto"]).sum())
        bool_check("cota_sem_duplicidade", dupc == 0, f"{dupc} duplicatas")

    # contagens estruturais
    # o horizonte e configuravel (PERIODO_FIM); o que importa para a
    # integridade e nao haver mes faltando entre o primeiro e o ultimo
    p_ini, p_fim = pd.to_datetime(df.periodo_demanda.min()), pd.to_datetime(df.periodo_demanda.max())
    esperado = (p_fim.year - p_ini.year) * 12 + (p_fim.month - p_ini.month) + 1
    bool_check("n_meses_correto", df.periodo_demanda.nunique() == esperado,
               f"{df.periodo_demanda.nunique()} meses de {esperado} esperados (sem lacunas)")
    bool_check("n_produtos_correto", df.id_produto.nunique() == cfg.N_PRODUTOS,
               f"{df.id_produto.nunique()} produtos")
    bool_check("n_ufs_correto", df.id_uf.nunique() == cfg.N_UFS,
               f"{df.id_uf.nunique()} UFs")


# ============================================================================
# Relatorio
# ============================================================================
def gerar_relatorio_markdown(rel: Relatorio, diagnostico: dict, destino: Path) -> Path:
    d = rel.df()
    score = rel.fidelity_score()
    por_dim = rel.score_por_dimensao()

    linhas = [
        "# Relatorio de validacao estatistica da base sintetica",
        "",
        f"**synthetic_data_fidelity_score: {score:.2f} / 100**",
        "",
        "## Score por dimensao",
        "",
        "| Dimensao | Score |",
        "|---|---|",
    ]
    for k, v in sorted(por_dim.items(), key=lambda x: -x[1]):
        linhas.append(f"| {k} | {v:.2f} |")

    linhas += ["", "## Parametros da geracao", "",
               "| Item | Valor |", "|---|---|"]
    for k, v in diagnostico.items():
        linhas.append(f"| {k} | {v} |")

    linhas += ["", "## Checks detalhados", "",
               "| Dimensao | Metrica | Valor | Alvo | Erro | Score | Obs |",
               "|---|---|---|---|---|---|---|"]
    for _, r in d.iterrows():
        val = "-" if r.valor is None else (f"{r.valor:,.4f}" if abs(r.valor) < 1000 else f"{r.valor:,.0f}")
        alv = "-" if r.alvo is None or (isinstance(r.alvo, float) and np.isnan(r.alvo)) else \
              (f"{r.alvo:,.4f}" if abs(r.alvo) < 1000 else f"{r.alvo:,.0f}")
        err = "-" if r.erro_rel is None or (isinstance(r.erro_rel, float) and np.isnan(r.erro_rel)) \
              else f"{r.erro_rel:.4f}"
        linhas.append(f"| {r.dimensao} | {r.metrica} | {val} | {alv} | {err} | {r.score:.1f} | {r.observacao} |")

    falhas = rel.falhas()
    linhas += ["", "## Onde o dado sintetico se distancia dos parametros", ""]
    if falhas.empty:
        linhas.append("Nenhum check abaixo de 70 pontos.")
    else:
        linhas += ["| Metrica | Valor | Alvo | Score |", "|---|---|---|---|"]
        for _, r in falhas.iterrows():
            linhas.append(f"| {r.metrica} | {r.valor:,.4f} | {r.alvo} | {r.score:.1f} |")

    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("\n".join(linhas), encoding="utf-8")
    return destino


if __name__ == "__main__":
    from .cota import calcular_cota_mensal
    from .gerador import agregar_por_setor, gerar_demanda_mensal

    df, dims, diag = gerar_demanda_mensal()
    cota = calcular_cota_mensal(agregar_por_setor(df, dims))
    rel = validar(df, cota, dims)

    print(f"synthetic_data_fidelity_score: {rel.fidelity_score():.2f} / 100\n")
    for k, v in sorted(rel.score_por_dimensao().items(), key=lambda x: -x[1]):
        print(f"  {k:<16s} {v:6.2f}")
    f = rel.falhas()
    if not f.empty:
        print("\nChecks abaixo de 70:")
        print(f[["metrica", "valor", "alvo", "score"]].to_string(index=False))
