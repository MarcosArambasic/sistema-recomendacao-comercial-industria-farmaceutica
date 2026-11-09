"""***************************************************************************************
SCRIPT    : avaliar_agentes.py
OBJETIVO  : Calcula, num so lugar, as metricas de validacao e desempenho de
            cada agente e grava data/agentes/metricas_agentes.json.
            As definicoes de cada metrica estao em docs/Arquitetura/METRICAS_AGENTES.md.

USO:
    python scripts/avaliar_agentes.py
    python scripts/avaliar_agentes.py --particoes 10

O QUE MEDE:
    Base sintetica  fidelity score e score por dimensao (RELATORIO_VALIDACAO.md)
    Agente 1        precisao, recall e F1 por classe, F1 macro, acuracia e kappa
                    de Cohen contra o rotulo consolidado do corpus (validacao
                    repetida 80/20), com as baselines classe majoritaria e regras;
                    kappa de Fleiss dos 3 avaliadores (teto de referencia)
    Agente 2        precisao/recall/F1 da decisao "manter (alto/medio) x descartar
                    (baixo)", taxa de retencao e cobertura de vinculo
    Agente 3        MAE, RMSE, R2, acerto de direcao, ganho sobre baselines, WAPE
                    da previsao de mercado, importancia SHAP (do agente3_saida.json)
    Agente 4        cobertura de setores, achados por categoria, % com regra aplicada
    Agente 5        destaques por setor, diversidade de escopo, repeticao de
                    oportunidades, distribuicao de urgencia
    Contratos       erros de validacao dos JSON dos Agentes 3, 4 e 5
****************************************************************************************/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.noticias.classificacao import (                         # noqa: E402
    CAMINHO_CORPUS, MAPA_PADRAO, NaiveBayesImpacto, carregar_mapa, impacto_heuristico,
    normalizar, rotulos_do_corpus, vincular_entidades,
)

DIR_AG = RAIZ / "data" / "agentes"
CLASSES = ["alto", "medio", "baixo"]


# ============================================================================
# Metricas genericas
# ============================================================================
def metricas_classificacao(y: list[str], p: list[str], classes=CLASSES) -> dict:
    y, p = np.array(y), np.array(p)
    por_classe = {}
    for c in classes:
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        por_classe[c] = {"precisao": round(prec, 4), "recall": round(rec, 4),
                         "f1": round(f1, 4), "suporte": int((y == c).sum())}
    acc = float((y == p).mean())
    # kappa de Cohen (modelo x referencia)
    pe = sum(((y == c).mean()) * ((p == c).mean()) for c in classes)
    kappa = (acc - pe) / (1 - pe) if pe < 1 else 0.0
    return {
        "acuracia": round(acc, 4),
        "f1_macro": round(float(np.mean([v["f1"] for v in por_classe.values()])), 4),
        "kappa_cohen": round(kappa, 4),
        "por_classe": por_classe,
        "matriz_confusao": {r: {c: int(((y == r) & (p == c)).sum()) for c in classes} for r in classes},
    }


def fleiss_kappa(tabela: np.ndarray) -> float:
    """tabela: (n_itens, n_categorias) com a contagem de avaliadores por categoria."""
    n = tabela.sum(axis=1)[0]
    p_j = tabela.sum(axis=0) / tabela.sum()
    p_i = ((tabela ** 2).sum(axis=1) - n) / (n * (n - 1))
    p_bar, p_e = p_i.mean(), (p_j ** 2).sum()
    return float((p_bar - p_e) / (1 - p_e))


# ============================================================================
# Agentes 1 e 2
# ============================================================================
def avaliar_agentes_1_e_2(particoes: int) -> dict:
    df = pd.read_excel(CAMINHO_CORPUS)
    cols = [c for c in df.columns if str(c).lower().startswith("classificacao")
            and "kappa" not in str(c).lower()]

    # Fleiss dos avaliadores humanos (teto de referencia da tarefa)
    norm = {"ALTO": "alto", "MÉDIO": "medio", "MEDIO": "medio", "BAIXO": "baixo"}
    votos = df[cols].apply(lambda s: s.astype(str).str.strip().str.upper().map(norm))
    completos = votos.dropna()
    tabela = np.stack([(completos == c).sum(axis=1).to_numpy() for c in CLASSES], axis=1)
    kappa_humanos = fleiss_kappa(tabela)
    concord_total = float((completos.nunique(axis=1) == 1).mean())
    # corpus ampliado: kappa separado por origem (anotacao humana x simulada)
    kappa_por_origem = {}
    if "origem_anotacao" in df.columns:
        for origem, idx in df.loc[completos.index].groupby("origem_anotacao").groups.items():
            sub = completos.loc[idx]
            tab = np.stack([(sub == c).sum(axis=1).to_numpy() for c in CLASSES], axis=1)
            kappa_por_origem[str(origem)] = {"kappa_fleiss": round(fleiss_kappa(tab), 4), "n_itens": int(len(sub))}

    rot = rotulos_do_corpus()
    df["y"] = df["url"].map(rot)
    df = df.dropna(subset=["y"])
    df["t"] = [normalizar(f"{a} {b}") for a, b in zip(df["titulo"], df["descricao"])]
    df["t_long"] = [normalizar(f"{a} {b} {c}") for a, b, c in
                    zip(df["titulo"], df["descricao"], df["conteudo"])]
    df = df.drop_duplicates("t").reset_index(drop=True)
    mapa = carregar_mapa() if (RAIZ / "data" / "config").exists() else MAPA_PADRAO

    y_all, p_nb, p_maj, p_reg = [], [], [], []
    for semente in range(particoes):
        idx = np.random.default_rng(semente).permutation(len(df))
        corte = int(0.8 * len(df))
        tr, te = df.iloc[idx[:corte]], df.iloc[idx[corte:]]
        nb = NaiveBayesImpacto().treinar(tr["t"].tolist(), tr["y"].tolist())
        maj = tr["y"].value_counts().index[0]
        for r in te.itertuples():
            y_all.append(r.y)
            p_nb.append(nb.prever(r.t))
            p_maj.append(maj)
            esc = vincular_entidades(r.t_long, mapa)[0]
            p_reg.append(impacto_heuristico(r.t_long) if esc != "nenhum" else "baixo")

    a1 = {
        "protocolo": f"{particoes} particoes aleatorias 80/20 sobre {len(df)} noticias sem duplicatas",
        "referencia": "voto majoritario dos 3 avaliadores (empate total = medio)",
        "naive_bayes_corpus": metricas_classificacao(y_all, p_nb),
        "baseline_classe_majoritaria": metricas_classificacao(y_all, p_maj),
        "baseline_regras": metricas_classificacao(y_all, p_reg),
        "humanos": {"kappa_fleiss_3_avaliadores": round(kappa_humanos, 4),
                    "concordancia_total": round(concord_total, 4), "n_itens": int(len(completos)),
                    "por_origem": kappa_por_origem},
    }

    # Agente 2: decisao binaria manter (alto/medio) x descartar (baixo)
    bin_ = lambda v: ["manter" if x in ("alto", "medio") else "descartar" for x in v]
    a2 = {"decisao_manter_descartar": metricas_classificacao(bin_(y_all), bin_(p_nb),
                                                               classes=["manter", "descartar"])}
    base = RAIZ / "data" / "noticias" / "noticias_classificadas.csv"
    if base.exists():
        b = pd.read_csv(base)
        mantidas = b["nivel_impacto"].isin(["alto", "medio"]) & (b["escopo"] != "nenhum")
        a2["operacao"] = {
            "noticias_na_base": int(len(b)),
            "taxa_retencao": round(float(mantidas.mean()), 4),
            "cobertura_vinculo": round(float((b["escopo"] != "nenhum").mean()), 4),
            "por_escopo": {k: int(v) for k, v in b["escopo"].value_counts().items()},
            "por_origem_classificacao": {k: int(v) for k, v in b["origem_classificacao"].value_counts().items()},
            "por_impacto": {k: int(v) for k, v in b["nivel_impacto"].value_counts().items()},
        }
    return {"agente1": a1, "agente2": a2}


# ============================================================================
# Agentes 3, 4 e 5
# ============================================================================
def _json(nome):
    p = DIR_AG / nome
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def avaliar_agente3() -> dict | None:
    s = _json("agente3_saida.json")
    if not s:
        return None
    from src.agentes.agente3_oportunidades_mercado import validar_saida
    m = s["modelo"]
    return {
        "algoritmo": m["algoritmo"], "periodo_teste": m["periodo_teste"],
        "n_treino": m["n_treino"], "n_teste": m["n_teste"],
        "metricas_teste": m["metricas_teste"], "baselines": m["baselines"],
        "ganho_sobre_baseline_pct": m["ganho_sobre_baseline_pct"],
        "supera_baseline": m["supera_baseline"],
        "previsao_mercado": m["previsao_mercado"],
        "top5_shap": m["importancia_global"][:5],
        "recomendacoes": s["resumo"],
        "erros_contrato": validar_saida(s),
    }


def avaliar_agente4() -> dict | None:
    s = _json("agente4_saida.json")
    if not s:
        return None
    from src.agentes.agente4_riscos_oportunidades import validar_saida
    ach = [a for st in s["setores"] for a in st["achados"]]
    return {
        "noticias_avaliadas": s["n_noticias_avaliadas"],
        "noticias_com_achado": len(s["noticias"]),
        "setores_com_achado": sum(1 for st in s["setores"] if st["achados"]),
        "achados": len(ach),
        "por_categoria": dict(Counter(a["categoria"] for a in ach)),
        "pct_com_regra_aplicada": round(np.mean([a["regra_aplicada"] is not None for a in ach]), 4) if ach else 0,
        "multiplicador_medio": round(float(np.mean([a["multiplicador"] for a in ach])), 4) if ach else 0,
        "score_medio": round(float(np.mean([a["score"] for a in ach])), 4) if ach else 0,
        "erros_contrato": validar_saida(s),
    }


def avaliar_agente5() -> dict | None:
    s = _json("agente5_saida.json")
    if not s:
        return None
    from src.agentes.agente5_recomendacoes import validar_saida
    dest = [st["destaques"] for st in s["setores"]]
    escopos = [len({d["noticia"]["escopo"] for d in ds}) for ds in dest if ds]
    repet = []
    for ds in dest:
        ops = [(o["id_alocacao"], o["id_produto"]) for d in ds for o in d["oportunidades"]]
        if ops:
            repet.append(1 - len(set(ops)) / len(ops))
    todas = [d for ds in dest for d in ds]
    return {
        "setores_com_destaque": sum(1 for ds in dest if ds),
        "destaques_por_setor_medio": round(float(np.mean([len(ds) for ds in dest])), 2) if dest else 0,
        "escopos_distintos_no_top4_medio": round(float(np.mean(escopos)), 2) if escopos else 0,
        "taxa_repeticao_oportunidades": round(float(np.mean(repet)), 4) if repet else 0,
        "por_urgencia": dict(Counter(d["recomendacao"]["urgencia"] for d in todas)),
        "por_modo": dict(Counter(d["recomendacao"].get("modo", "?") for d in todas)),
        "noticias_distintas_no_painel": len({d["noticia"]["id_noticia"] for d in todas}),
        "erros_contrato": validar_saida(s),
    }


def avaliar_base() -> dict | None:
    p = RAIZ / "docs" / "Arquitetura" / "RELATORIO_VALIDACAO.md"
    if not p.exists():
        return None
    txt = p.read_text(encoding="utf-8")
    score = re.search(r"fidelity_score:\s*([\d.]+)", txt)
    dims = dict(re.findall(r"\|\s*(integridade|reconciliacao|distribuicao|tendencia|sazonalidade|correlacao)\s*\|\s*([\d.]+)\s*\|", txt))
    return {"fidelity_score": float(score.group(1)) if score else None,
            "por_dimensao": {k: float(v) for k, v in dims.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description="Metricas de validacao e desempenho dos agentes")
    ap.add_argument("--particoes", type=int, default=5)
    a = ap.parse_args()
    # Agentes 1, 2 e Desempate: protocolo do Orquestrador (missao "corpus"):
    # validacao cruzada estratificada de 5 folds, corpus sem duplicatas,
    # discordancia total fora do treino ate o Desempate resolver.
    orq = _json("metricas_agentes_1_2_desempate.json")
    mbert = None
    p_mbert = DIR_AG / "agente1" / "agente1_metricas.json"
    if p_mbert.exists():
        mbert = json.loads(p_mbert.read_text(encoding="utf-8"))
    humanos = avaliar_agentes_1_e_2(a.particoes)["agente1"]["humanos"]
    res = {"gerado_em": datetime.now().isoformat(timespec="seconds"),
           "base_sintetica": avaliar_base(),
           "concordancia_humanos": humanos,
           "orquestrador_corpus": orq,
           "agente1_mbert": mbert,
           "agente3": avaliar_agente3(), "agente4": avaliar_agente4(), "agente5": avaliar_agente5()}
    DIR_AG.mkdir(parents=True, exist_ok=True)
    (DIR_AG / "metricas_agentes.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
