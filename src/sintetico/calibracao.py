"""***************************************************************************************
SCRIPT    : calibracao.py
OBJETIVO  : Extrair do Parametros_Dados_Sinteticos.xlsx os parametros REAIS e
            agregados da empresa e congela-los num JSON auditavel
            (data/calibracao/calibracao.json).

            Por que congelar em JSON em vez de ler o Excel em runtime:
              1) reprodutibilidade - a base gerada fica amarrada a uma versao
                 explicita dos parametros;
              2) auditoria - da para versionar e revisar o que foi usado;
              3) o gerador nao depende do Excel (nem do openpyxl) para rodar.

            IMPORTANTE (privacidade): o Excel contem apenas AGREGADOS
            (medias, indices, CAGR). Nenhum registro transacional real da
            empresa e lido, gravado ou versionado por este modulo.

AUTOR     : Marcos Arambasic
DATA      : AGO/2026
****************************************************************************************/
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ANCORA_ANO as ANCORA_ANO_CALIB
from .config import ARQ_CALIBRACAO, CAMINHO_EXCEL, DIR_CALIBRACAO

ABA_SAZONAL = "Índice Sazonal"

# Blocos da aba 'Indice Sazonal' (linha inicial, linha final, rotulo).
# A aba tem 4 blocos empilhados, cada um com seu proprio cabecalho.
BLOCOS = {
    "proprios_nacional": (2, 26),
    "mercado_nacional": (29, 53),
    "proprios_territorio": (56, 704),
    "mercado_territorio": (707, 1355),
}

COLUNAS = ["chave", "periodo", "media_anual", "media_mensal", "indice_sazonal", "variacao"]


def _ler_bloco(raw: pd.DataFrame, ini: int, fim: int) -> pd.DataFrame:
    b = raw.iloc[ini:fim].copy()
    b.columns = COLUNAS
    b = b.dropna(subset=["periodo"])
    b["periodo"] = pd.to_datetime(b["periodo"])
    for c in ["media_anual", "media_mensal", "indice_sazonal"]:
        b[c] = pd.to_numeric(b[c], errors="coerce")
    return b.dropna(subset=["media_mensal"])


def _cv_residual_e_phi(bloco: pd.DataFrame) -> tuple[float, float]:
    """
    Decompoe a serie de cada territorio em tendencia x sazonalidade x residuo
    e devolve (CV do residuo, autocorrelacao lag-1 do residuo).

    Esse e o passo que evita o erro de injetar o CV bruto como ruido: o CV
    observado (ex.: 26,03%) ja embute tendencia e sazonalidade, entao o ruido
    a injetar precisa ser menor.
    """
    cvs, phis = [], []
    for _, g in bloco.groupby("chave"):
        g = g.sort_values("periodo")
        y = g["media_mensal"].to_numpy(dtype=float)
        if len(y) < 4 or np.any(y <= 0):
            continue
        t = np.arange(len(y))
        coef = np.polyfit(t, np.log(y), 1)
        destendencia = y / np.exp(np.polyval(coef, t))

        aux = g.copy()
        aux["_d"] = destendencia
        sazonal = aux.groupby(aux.periodo.dt.month)["_d"].transform("mean").to_numpy()
        residuo = destendencia / sazonal

        cvs.append(residuo.std(ddof=1) / residuo.mean())
        if len(residuo) > 2:
            phis.append(float(np.corrcoef(residuo[:-1], residuo[1:])[0, 1]))
    return float(np.mean(cvs)), float(np.nanmean(phis))


def _perfil_sazonal(bloco: pd.DataFrame) -> dict[str, dict[int, float]]:
    """
    Indice sazonal medio por (territorio, mes-calendario), normalizado para
    que a media dos 12 meses seja exatamente 1,0.

    IMPORTANTE - por que NAO usamos a coluna 'indice_sazonal' do Excel
    diretamente: ela foi calculada como demanda_mensal / media_do_ano, e os
    anos 2024 e 2026 sao PARCIAIS na base (jul-dez e jan-jun). Como a serie
    tem tendencia forte, dividir pela media de meio ano contamina o indice
    com o crescimento do periodo - meses do fim de um ano parcial ficam com
    indice inflado e os do inicio, deflacionado.

    Se esse indice contaminado fosse aplicado no gerador, ele cancelaria
    parte da tendencia e a taxa de crescimento sairia menor do que a injetada
    (efeito medido: mercado saindo a 48% quando 54% foi o input).

    A correcao e fazer a decomposicao correta: remover a tendencia log-linear
    ANTES de estimar o componente sazonal.
    """
    perfil: dict[str, dict[int, float]] = {}
    for uf, g in bloco.groupby("chave"):
        g = g.sort_values("periodo")
        y = g["media_mensal"].to_numpy(dtype=float)
        if len(y) < 4 or np.any(y <= 0):
            por_mes = pd.Series(1.0, index=range(1, 13))
        else:
            t = np.arange(len(y), dtype=float)
            coef = np.polyfit(t, np.log(y), 1)
            destendencia = y / np.exp(np.polyval(coef, t))
            aux = g.copy()
            aux["_d"] = destendencia
            por_mes = aux.groupby(aux.periodo.dt.month)["_d"].mean().reindex(range(1, 13))
            por_mes = por_mes.fillna(por_mes.mean() if por_mes.notna().any() else 1.0)

        por_mes = por_mes / por_mes.mean()
        perfil[str(uf)] = {int(m): round(float(v), 6) for m, v in por_mes.items()}
    return perfil


def _cagr(serie_por_ano: pd.Series) -> float:
    anos = sorted(serie_por_ano.index)
    n = anos[-1] - anos[0]
    if n <= 0:
        return 0.0
    return float((serie_por_ano[anos[-1]] / serie_por_ano[anos[0]]) ** (1 / n) - 1)


def _crescimento_log_linear(serie_mensal: pd.Series) -> float:
    """
    Taxa de crescimento ANUAL estimada por regressao log-linear sobre a serie
    mensal completa.

    Por que nao usar o CAGR da aba do Excel diretamente: as "medias anuais"
    de 2024 e 2026 cobrem apenas 6 meses cada (jul-dez e jan-jun). Os pontos
    medios distam 18 meses, mas a formula do Excel divide por 2 anos, o que
    subestima a taxa real. A regressao usa os 24 pontos mensais observados e
    nao depende de janelas parciais.

    A relacao algebrica entre as duas e (1 + cagr_excel) = (1 + g_real)^0.75.
    O modulo de validacao confere que a base gerada, medida com a MESMA
    convencao do Excel, reproduz o CAGR informado.
    """
    s = serie_mensal.dropna().astype(float)
    y = np.log(s.to_numpy())
    t = np.arange(len(y), dtype=float)
    slope = float(np.polyfit(t, y, 1)[0])
    return float(np.exp(slope * 12.0) - 1.0)


def _expoente_correcao_excel(serie_mensal: pd.Series) -> float:
    """
    Fator que converte o CAGR na convencao do Excel para a taxa anual real.

    O Excel faz (media_ano_final / media_ano_inicial)^(1/n_anos), mas os anos
    das pontas sao parciais. Para uma exponencial de taxa g:

        cagr_excel = g^(gap_meses / 12 / n_anos) - 1
        =>  g = (1 + cagr_excel)^(12 * n_anos / gap_meses)

    onde gap_meses e a distancia real entre os pontos medios das duas janelas.
    Com jul-dez/2024 e jan-jun/2026 o gap e 18 meses e n_anos = 2, entao o
    expoente vale 24/18 = 4/3. Calculado a partir dos dados, nao fixado.
    """
    s = serie_mensal.dropna()
    anos = sorted(s.index.year.unique())
    ini, fim = anos[0], anos[-1]

    def _ponto_medio(ano: int) -> float:
        meses = s[s.index.year == ano].index
        return float(np.mean([(d.year - ini) * 12 + (d.month - 1) for d in meses]))

    gap_meses = _ponto_medio(fim) - _ponto_medio(ini)
    n_anos = fim - ini
    if gap_meses <= 0 or n_anos <= 0:
        return 1.0
    return float(12.0 * n_anos / gap_meses)


def _inverter_cagr_excel(cagr_excel: float, expoente: float) -> float:
    """
    Taxa anual real que reproduz `cagr_excel` sob a convencao do Excel,
    ignorando sazonalidade (solucao algebrica aproximada).
    """
    return float((1.0 + cagr_excel) ** expoente - 1.0)


def _perfil_nacional_efetivo(perfil_por_uf: dict, pesos_uf: dict) -> np.ndarray:
    """
    Perfil sazonal nacional = media dos perfis das UFs ponderada pelo volume.

    As chaves de mes podem vir como int (objeto em memoria) ou str (apos ida
    e volta pelo JSON), entao a busca aceita as duas formas.
    """
    eff = np.zeros(12)
    for uf, peso in pesos_uf.items():
        p = perfil_por_uf[uf]
        eff += peso * np.array([float(p.get(m, p.get(str(m), 1.0))) for m in range(1, 13)])
    return eff


def _cagr_excel_do_sinal(g: float, eff: np.ndarray, janela: dict, ancora_ano: int) -> float:
    """
    Mede o CAGR na convencao do Excel sobre o sinal DETERMINISTICO
    (tendencia x sazonalidade), sem ruido.
    """
    anos = sorted(int(a) for a in janela["meses_por_ano"])
    ano_ini, ano_fim = anos[0], anos[-1]
    meses_ini = janela["meses_por_ano"][str(ano_ini)]
    meses_fim = janela["meses_por_ano"][str(ano_fim)]

    def media(ano: int, meses: list[int]) -> float:
        vals = []
        for m in meses:
            idx = (ano - ancora_ano) * 12 + (m - 1)
            vals.append((1.0 + g) ** (idx / 12.0) * eff[m - 1])
        return float(np.mean(vals))

    a = media(ano_ini, meses_ini)
    b = media(ano_fim, meses_fim)
    return float((b / a) ** (1.0 / (ano_fim - ano_ini)) - 1.0)


def _resolver_taxa(cagr_alvo: float, eff: np.ndarray, janela: dict,
                   ancora_ano: int, iteracoes: int = 200) -> float:
    """
    Encontra por bissecao a taxa anual g cujo sinal deterministico, medido com
    a convencao do Excel, devolve exatamente `cagr_alvo`.

    Necessario porque a solucao algebrica ignora a sazonalidade, e o perfil
    sazonal do mercado e assimetrico entre semestres (H2/H1 ~ 1,10). As duas
    janelas do Excel caem em semestres diferentes (jul-dez vs jan-jun), entao
    essa assimetria vaza direto para a taxa medida - no caso dos concorrentes,
    um vies de -4,7 p.p.
    """
    lo, hi = -0.95, 10.0
    for _ in range(iteracoes):
        mid = (lo + hi) / 2.0
        if _cagr_excel_do_sinal(mid, eff, janela, ancora_ano) < cagr_alvo:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def extrair_calibracao(caminho_excel: Path | None = None) -> dict:
    """Le o Excel e devolve o dicionario de calibracao completo."""
    caminho = Path(caminho_excel or CAMINHO_EXCEL)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Excel de parametros nao encontrado em {caminho}. "
            "Ajuste CAMINHO_EXCEL em config.py."
        )

    raw = pd.read_excel(caminho, sheet_name=ABA_SAZONAL, header=None)
    blocos = {nome: _ler_bloco(raw, ini, fim) for nome, (ini, fim) in BLOCOS.items()}

    prop_terr = blocos["proprios_territorio"]
    merc_terr = blocos["mercado_territorio"]
    prop_nac = blocos["proprios_nacional"]
    merc_nac = blocos["mercado_nacional"]

    # ---- escala por territorio (media anual media dos anos disponiveis) ----
    escala_prop = prop_terr.groupby("chave")["media_anual"].mean()
    escala_merc = merc_terr.groupby("chave")["media_anual"].mean()
    escala_conc = (escala_merc - escala_prop)

    if (escala_conc <= 0).any():
        ufs = list(escala_conc[escala_conc <= 0].index)
        raise ValueError(
            f"Inconsistencia nos dados de origem: mercado < proprios nas UFs {ufs}. "
            "Concorrentes ficariam negativos."
        )

    # ---- tendencia por ano (nacional) ----
    prop_ano = prop_nac.groupby(prop_nac.periodo.dt.year)["media_anual"].first()
    merc_ano = merc_nac.groupby(merc_nac.periodo.dt.year)["media_anual"].first()
    conc_ano = merc_ano - prop_ano

    # ---- tendencia por regressao log-linear na serie mensal (preferida) ----
    prop_mensal = prop_nac.set_index("periodo")["media_mensal"].sort_index()
    merc_mensal = merc_nac.set_index("periodo")["media_mensal"].sort_index()
    conc_mensal = (merc_mensal - prop_mensal).dropna()
    if (conc_mensal <= 0).any():
        raise ValueError("Serie de concorrentes ficou <= 0 em algum mes (mercado < proprios).")

    expoente_excel = _expoente_correcao_excel(prop_mensal)

    # janela de observacao usada pelo Excel - a validacao mede nela para
    # comparar como-com-como
    janela = {
        "inicio": str(prop_mensal.index.min().date()),
        "fim": str(prop_mensal.index.max().date()),
        "meses_por_ano": {
            str(int(ano)): sorted(int(m) for m in g.index.month)
            for ano, g in prop_mensal.groupby(prop_mensal.index.year)
        },
    }

    # perfis sazonais nacionais efetivos, usados para resolver a taxa
    perfil_prop = _perfil_sazonal(prop_terr)
    perfil_merc = _perfil_sazonal(merc_terr)
    pesos_prop = {str(k): float(v / escala_prop.sum()) for k, v in escala_prop.items()}
    pesos_conc = {str(k): float(v / escala_conc.sum()) for k, v in escala_conc.items()}
    eff_prop = _perfil_nacional_efetivo(perfil_prop, pesos_prop)
    eff_conc = _perfil_nacional_efetivo(perfil_merc, pesos_conc)

    # ---- variabilidade residual + autocorrelacao ----
    cv_res_prop, phi_prop = _cv_residual_e_phi(prop_terr)
    cv_res_merc, phi_merc = _cv_residual_e_phi(merc_terr)

    # ---- CV bruto (alvo de validacao) ----
    def _cv_bruto(bloco):
        return float(np.mean([
            g.sort_values("periodo")["media_mensal"].std(ddof=1)
            / g["media_mensal"].mean()
            for _, g in bloco.groupby("chave")
        ]))

    calib = {
        "_meta": {
            "fonte": caminho.name,
            "aba": ABA_SAZONAL,
            "descricao": (
                "Parametros agregados derivados de dados reais da empresa. "
                "Contem apenas medias, indices e taxas - nenhum registro transacional."
            ),
            "n_territorios": int(escala_prop.shape[0]),
            "meses_observados": int(prop_terr.periodo.nunique()),
            "periodo_observado": [
                str(prop_terr.periodo.min().date()),
                str(prop_terr.periodo.max().date()),
            ],
        },
        "escala_territorio": {
            "proprios": {str(k): float(v) for k, v in escala_prop.items()},
            "concorrentes": {str(k): float(v) for k, v in escala_conc.items()},
            "mercado": {str(k): float(v) for k, v in escala_merc.items()},
        },
        "peso_territorio": {
            "proprios": {str(k): float(v / escala_prop.sum()) for k, v in escala_prop.items()},
            "concorrentes": {str(k): float(v / escala_conc.sum()) for k, v in escala_conc.items()},
        },
        "sazonalidade": {
            "proprios": perfil_prop,
            "mercado": perfil_merc,
        },
        "sazonalidade_nacional_efetiva": {
            "proprios": [float(x) for x in eff_prop],
            "concorrentes": [float(x) for x in eff_conc],
        },
        "janela_observada": janela,
        "tendencia": {
            "media_mensal_por_ano": {
                "proprios": {int(k): float(v) for k, v in prop_ano.items()},
                "mercado": {int(k): float(v) for k, v in merc_ano.items()},
                "concorrentes": {int(k): float(v) for k, v in conc_ano.items()},
            },
            # convencao do Excel (janelas parciais, divisor de 2 anos).
            # E o ALVO DE VALIDACAO, nao o input do gerador.
            "cagr_excel_proprios": _cagr(prop_ano),
            "cagr_excel_mercado": _cagr(merc_ano),
            "cagr_excel_concorrentes": _cagr(conc_ano),
            # taxa real estimada na serie mensal completa
            "crescimento_log_linear": {
                "proprios": _crescimento_log_linear(prop_mensal),
                "concorrentes": _crescimento_log_linear(conc_mensal),
                "mercado": _crescimento_log_linear(merc_mensal),
            },
            # aproximacao algebrica (ignora sazonalidade) - mantida para
            # comparacao e documentacao
            "crescimento_algebrico_cagr_excel": {
                "proprios": _inverter_cagr_excel(_cagr(prop_ano), expoente_excel),
                "concorrentes": _inverter_cagr_excel(_cagr(conc_ano), expoente_excel),
                "mercado": _inverter_cagr_excel(_cagr(merc_ano), expoente_excel),
            },
            # taxa resolvida numericamente sobre o sinal deterministico, ja
            # considerando a assimetria sazonal entre semestres.
            # E O INPUT DO GERADOR.
            "crescimento_para_cagr_excel": {
                "proprios": _resolver_taxa(_cagr(prop_ano), eff_prop, janela, ANCORA_ANO_CALIB),
                "concorrentes": _resolver_taxa(_cagr(conc_ano), eff_conc, janela, ANCORA_ANO_CALIB),
            },
            "expoente_correcao_excel": expoente_excel,
            "assimetria_sazonal_h2_h1": {
                "proprios": float(eff_prop[6:].mean() / eff_prop[:6].mean()),
                "concorrentes": float(eff_conc[6:].mean() / eff_conc[:6].mean()),
            },
        },
        "variabilidade": {
            "cv_bruto_proprios": _cv_bruto(prop_terr),
            "cv_bruto_mercado": _cv_bruto(merc_terr),
            "cv_residual_proprios": cv_res_prop,
            "cv_residual_mercado": cv_res_merc,
            "phi_ar1_proprios": phi_prop,
            "phi_ar1_mercado": phi_merc,
        },
        "share": {
            "por_ano": {int(a): float(prop_ano[a] / merc_ano[a]) for a in prop_ano.index},
            "por_territorio": {
                str(k): float(escala_prop[k] / escala_merc[k]) for k in escala_prop.index
            },
        },
    }
    return calib


def salvar_calibracao(calib: dict | None = None, caminho: Path | None = None) -> Path:
    calib = calib or extrair_calibracao()
    caminho = Path(caminho or ARQ_CALIBRACAO)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(calib, indent=2, ensure_ascii=False), encoding="utf-8")
    return caminho


def carregar_calibracao(caminho: Path | None = None) -> dict:
    """Carrega o JSON congelado; gera na hora se ainda nao existir."""
    caminho = Path(caminho or ARQ_CALIBRACAO)
    if not caminho.exists():
        salvar_calibracao(caminho=caminho)
    return json.loads(caminho.read_text(encoding="utf-8"))


if __name__ == "__main__":
    DIR_CALIBRACAO.mkdir(parents=True, exist_ok=True)
    calib = extrair_calibracao()
    destino = salvar_calibracao(calib)

    print(f"Calibracao salva em: {destino}")
    m = calib["_meta"]
    print(f"  territorios: {m['n_territorios']} | meses observados: {m['meses_observados']} "
          f"| periodo: {m['periodo_observado'][0]} a {m['periodo_observado'][1]}")
    t = calib["tendencia"]
    print("\n  CAGR na convencao do Excel (alvo de validacao):")
    print(f"    proprios={t['cagr_excel_proprios']:+.4%}  concorrentes={t['cagr_excel_concorrentes']:+.4%}"
          f"  mercado={t['cagr_excel_mercado']:+.4%}")
    ll = t["crescimento_log_linear"]
    print("  Crescimento log-linear na serie mensal (input do gerador):")
    print(f"    proprios={ll['proprios']:+.4%}  concorrentes={ll['concorrentes']:+.4%}"
          f"  mercado={ll['mercado']:+.4%}")
    v = calib["variabilidade"]
    print(f"\n  CV bruto proprios/mercado    = {v['cv_bruto_proprios']:.2%} / {v['cv_bruto_mercado']:.2%}")
    print(f"  CV residual proprios/mercado = {v['cv_residual_proprios']:.2%} / {v['cv_residual_mercado']:.2%}")
    print(f"  phi AR(1) proprios/mercado   = {v['phi_ar1_proprios']:.3f} / {v['phi_ar1_mercado']:.3f}")
    print(f"\n  share por ano: { {k: f'{x:.2%}' for k, x in calib['share']['por_ano'].items()} }")
