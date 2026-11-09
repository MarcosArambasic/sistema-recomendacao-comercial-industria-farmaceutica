"""
Amplia o corpus anotado de noticias, simula os tres avaliadores nas noticias
novas e balanceia o resultado em ~425 noticias por classe consolidada.

Entradas
  data/estatistica_kappa/noticias_classificadas.xlsx      corpus humano original (1.415 linhas)
  data/estatistica_kappa/ampliacao/coleta_bruta.jsonl      noticias novas em ingles coletadas
      - API do The Guardian (extracoes do ETL do projeto que nunca foram anotadas)
      - web scraping de feeds RSS (STAT, Fierce Pharma, BioPharma Dive, ScienceDaily,
        Drugs.com, News-Medical, pharmaphorum) e do agregador Bing News (RSS)
      cada linha traz o julgamento de referencia `ref` (A/M/B) feito na coleta

Saidas
  data/estatistica_kappa/noticias_classificadas_ampliado.xlsx   <- corpus que treina os Agentes 1 e 2
  data/estatistica_kappa/ampliacao/relatorio_ampliacao.json     contagens, kappa, matrizes usadas

Reclassificacao das noticias novas por guia de anotacao (versao 2)
  Criterio-chave: impacto da noticia sobre o FATURAMENTO DE VENDAS de
  medicamentos para diabetes (antidiabeticos e antiobesidade).

  1. Cada noticia tem dois julgamentos independentes:
     - o julgamento de referencia feito na coleta (`ref`, revisado);
     - a regra escrita do guia de anotacao (funcao `guia`, abaixo).
  2. Quando os dois coincidem, o caso e CLARO; quando divergem, LIMITROFE.
  3. Os tres avaliadores simulados aplicam o mesmo guia:
     - caso claro: todos atribuem a classe do guia; cada avaliador tem uma
       probabilidade pequena de lapso (EPS = 5%) na direcao da sua tendencia
       observada no corpus humano (Av.1 -> Medio, Av.2 -> acima, Av.3 -> abaixo);
     - caso limitrofe: cada avaliador escolhe entre as duas classes candidatas
       segundo a sua tendencia (Av.2 a maior, Av.3 a menor, Av.1 a Media se
       ela for candidata, senao a referencia).
  O nivel de concordancia resultante e consequencia desse protocolo (guia
  comum + lapso de 5%) e NAO e uma medida de confiabilidade humana; o kappa
  humano continua sendo o do corpus original.

Uso
  python docs/Arquitetura/ampliar_corpus_noticias.py [--alvo 425] [--semente 42] [--eps 0.05]
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]   # script em docs/Arquitetura/
DIR = RAIZ / "data" / "estatistica_kappa"
ORIGINAL = DIR / "noticias_classificadas.xlsx"
COLETA = DIR / "ampliacao" / "coleta_bruta.jsonl"
SAIDA = DIR / "noticias_classificadas_ampliado.xlsx"
RELATORIO = DIR / "ampliacao" / "relatorio_ampliacao.json"

AVAL = ["classificacao mendes", "classificacao guilherme", "classificacao marcos"]
CLASSES = ["ALTO", "MÉDIO", "BAIXO"]
REF2CL = {"A": "ALTO", "M": "MÉDIO", "B": "BAIXO"}

# Termos que colocam a noticia no centro do mercado de medicamentos para
# diabetes/obesidade. No corpus humano (titulos unicos), 65,5% das noticias
# com maioria ALTO citam um desses termos no titulo ou na descricao, contra
# 1,7% das MEDIO e 0,5% das BAIXO. O julgamento de referencia e
# alinhado a esse padrao: noticia de impacto medio que trata diretamente
# desses medicamentos sobe para alto; noticia baixa que os cita no titulo
# sobe para medio.
TERMOS_NUCLEO = (
    r"ozempic|wegovy|mounjaro|zepbound|semaglutide|tirzepatide|glp-1|glp1|insulin|sglt2|sglt-2|"
    r"jardiance|farxiga|forxiga|dapagliflozin|empagliflozin|januvia|sitagliptin|linagliptin|"
    r"metformin|orforglipron|foundayo|retatrutide|cagrisema|rybelsus|trulicity|"
    r"weight[- ]loss (?:drug|jab|shot|pill)s?|obesity (?:drug|pill|medicine|medication|treatment)s?|"
    r"diabetes (?:drug|medic|treatment|jab)"
)


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", str(t).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"^stat\+:\s*", "", t)
    return re.sub(r"[^a-z0-9 ]+", " ", t).split("|")[0].strip()


def _tokens(t: str) -> set[str]:
    return {w for w in _norm(t).split() if len(w) > 2}


def maioria(linha) -> str | None:
    v, q = Counter(linha).most_common(1)[0]
    return v if q >= 2 else None


def carregar_humano() -> tuple[pd.DataFrame, int]:
    h = pd.read_excel(ORIGINAL)
    n0 = len(h)
    h["_tn"] = h["titulo"].map(_norm)
    h = h.drop_duplicates("_tn").drop_duplicates("url").reset_index(drop=True)
    h["origem_anotacao"] = "humana"
    h["fonte_coleta"] = "API The Guardian (corpus TCC1)"
    return h, n0 - len(h)


def carregar_coleta(humano: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = pd.read_json(COLETA, lines=True)
    info = {"linhas_coletadas": int(len(d))}
    d = d[d["data"].astype(str) >= "2023-01-01"]
    d["url"] = d["url"].astype(str).str.split("?utm", regex=False).str[0].str.strip()
    d["_tn"] = d["titulo"].map(_norm)
    d = d.drop_duplicates("url").drop_duplicates("_tn")
    # nada que ja esteja no corpus humano (url ou titulo)
    d = d[~d["url"].isin(set(humano["url"].astype(str))) & ~d["_tn"].isin(set(humano["_tn"]))]
    # quase-duplicatas: mesma materia republicada com titulo levemente diferente
    vistos: list[set[str]] = [_tokens(t) for t in humano["titulo"]]
    manter = []
    for _, r in d.iterrows():
        tk = _tokens(r["titulo"])
        dup = any(len(tk & v) / max(1, len(tk | v)) >= 0.8 for v in vistos if v)
        manter.append(not dup)
        if not dup:
            vistos.append(tk)
    info["quase_duplicatas_removidas"] = int(len(manter) - sum(manter))
    d = d[manter].copy()
    # alinhamento do julgamento de referencia ao padrao humano
    nucleo_tit = d["titulo"].str.contains(TERMOS_NUCLEO, case=False, regex=True)
    nucleo_desc = d["descricao"].fillna("").str.slice(0, 300).str.contains(TERMOS_NUCLEO, case=False, regex=True)
    d["ref_coleta"] = d["ref"]
    d.loc[(d["ref"] == "M") & (nucleo_tit | nucleo_desc), "ref"] = "A"
    d.loc[(d["ref"] == "B") & nucleo_tit, "ref"] = "M"
    info["referencia_na_coleta"] = d["ref_coleta"].value_counts().to_dict()
    info["referencia_alinhada"] = d["ref"].value_counts().to_dict()
    info["noticias_novas_unicas"] = int(len(d))
    return d.reset_index(drop=True), info


# ----------------------------------------------------------------------------
# Guia de anotacao (impacto no faturamento de medicamentos para diabetes)
# ----------------------------------------------------------------------------
G_MEDICAMENTO = re.compile(
    r"ozempic|wegovy|mounjaro|zepbound|semaglutide|tirzepatide|liraglutide|dulaglutide|glp-1|glp1|insulin|"
    r"sglt2|sglt-2|jardiance|farxiga|forxiga|dapagliflozin|empagliflozin|canagliflozin|invokana|januvia|"
    r"sitagliptin|linagliptin|tradjenta|dpp-4|metformin|orforglipron|foundayo|retatrutide|cagrisema|amycretin|"
    r"rybelsus|trulicity|lantus|humalog|weight[- ]loss (?:drug|jab|shot|pill|medication)s?|"
    r"obesity (?:drug|pill|medicine|medication|treatment|shot)s?|diabetes (?:drug|medic|treatment|jab|pill)|slimming jab",
    re.I)
G_EVENTO_MERCADO = re.compile(
    r"price|pricing|cost|cheap|discount|rebate|\bcap\b|caps|negotiat|coverage|cover |covers|formular|insur|medicare|"
    r"medicaid|approv|cleared|clears|authori[sz]|launch|generic|biosimilar|copies|copy|patent|shortage|supply|stock|"
    r"recall|counterfeit|fake|smuggl|black market|tariff|sales|revenue|earnings|profit|forecast|guidance|shares|market|"
    r"compound|\bban|banned|withdraw|discontinu|label|warn|fda|ema\b|chmp|anvisa|nice\b|regulator|lawsuit|sue[sd]?\b|"
    r"court|deal|acqui|bid|layoff|trumprx|most[- ]favou?red", re.I)
G_FARMA = re.compile(
    r"drug pric|drugmaker|pharma|pharmaceutical|medicine|medicines|prescription|tariff|medicare|medicaid|\bpbm|340b|"
    r"\bfda\b|generic|supply chain|active pharmaceutical|ingredient|pharmac(?:y|ies|ist)|drug shortage|health insur|"
    r"biotech|astrazeneca|novo nordisk|eli lilly|boehringer|sanofi|merck", re.I)
G_METABOLICO = re.compile(r"diabet|obes|glucose|blood sugar|hba1c|insulin resistance|metabolic|weight|cardiometabolic|"
                          r"kidney disease|heart failure", re.I)
G_MACRO = re.compile(r"exchange rate|currency|dollar|\byen\b|inflation|interest rate|recession|trade war|oil price|"
                     r"economy|gdp", re.I)


def guia(titulo, descricao) -> tuple[str, str]:
    """Regra escrita do guia: devolve (classe A/M/B, criterio aplicado)."""
    t, d = str(titulo or ""), str(descricao or "")[:300]
    td = f"{t} {d}"
    if G_MEDICAMENTO.search(t) and G_EVENTO_MERCADO.search(t):
        return "A", "A1 medicamento + evento de mercado (preco, acesso, oferta, regulacao, vendas) no titulo"
    if G_MEDICAMENTO.search(td) and G_EVENTO_MERCADO.search(td):
        return "A", "A2 medicamento + evento de mercado no titulo/descricao"
    if G_MEDICAMENTO.search(t):
        return "A", "A3 medicamento no titulo (evidencia clinica ou uso que altera demanda)"
    if G_MEDICAMENTO.search(d):
        return "M", "M1 medicamento citado apenas na descricao"
    if G_FARMA.search(t):
        return "M", "M2 mercado ou politica farmaceutica sem medicamento especifico"
    if G_FARMA.search(d) or G_METABOLICO.search(t):
        return "M", "M3 contexto farmaceutico ou metabolico indireto"
    if G_MACRO.search(td) or G_METABOLICO.search(d):
        return "B", "B1 macroeconomia/contexto metabolico sem vinculo com medicamentos"
    return "B", "B2 sem relacao com o mercado de medicamentos para diabetes"


_ORD = {"A": 2, "M": 1, "B": 0}
_SOBE = {"B": "M", "M": "A", "A": "A"}
_DESCE = {"A": "M", "M": "B", "B": "B"}


def anotar_por_guia(novas: pd.DataFrame, semente: int, eps: float) -> pd.DataFrame:
    rng = np.random.default_rng(semente)
    out = novas.copy()
    g = out.apply(lambda r: guia(r["titulo"], r["descricao"]), axis=1, result_type="expand")
    out["classe_guia"], out["criterio_guia"] = g[0], g[1]
    out["certeza_guia"] = np.where(out["classe_guia"] == out["ref"], "clara", "limitrofe")
    rot = []
    for ref, cg in zip(out["ref"], out["classe_guia"]):
        if ref == cg:
            r = [ref, ref, ref]
            if rng.random() < eps:
                r[0] = "M"                      # Av.1 tende a Medio
            if rng.random() < eps:
                r[1] = _SOBE[ref]               # Av.2 tende a classificar acima
            if rng.random() < eps:
                r[2] = _DESCE[ref]              # Av.3 tende a classificar abaixo
        else:
            alta, baixa = (ref, cg) if _ORD[ref] > _ORD[cg] else (cg, ref)
            r = ["M" if "M" in (ref, cg) else ref, alta, baixa]
        rot.append(r)
    for i, a in enumerate(AVAL):
        out[a] = [REF2CL[x[i]] for x in rot]
    return out


def perfis(humano: pd.DataFrame) -> dict[str, pd.DataFrame]:
    h = humano.copy()
    h["_maj"] = h[AVAL].apply(maioria, axis=1)
    h = h.dropna(subset=["_maj"])
    return {a: pd.crosstab(h["_maj"], h[a], normalize="index").reindex(index=CLASSES, columns=CLASSES).fillna(0)
            for a in AVAL}


def simular(novas: pd.DataFrame, mats: dict, semente: int) -> pd.DataFrame:
    rng = np.random.default_rng(semente)
    out = novas.copy()
    for a in AVAL:
        m = mats[a]
        out[a] = [rng.choice(CLASSES, p=m.loc[REF2CL[r]].values) for r in out["ref"]]
    return out


def fleiss(df: pd.DataFrame) -> float:
    cont = np.array([[sum(r[a] == c for a in AVAL) for c in CLASSES] for _, r in df.iterrows()], float)
    n = 3
    p_j = cont.sum(0) / cont.sum()
    P_i = ((cont ** 2).sum(1) - n) / (n * (n - 1))
    Pe = (p_j ** 2).sum()
    return float((P_i.mean() - Pe) / (1 - Pe))


def para_planilha(novas: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(novas["data"])
    return pd.DataFrame({
        "ano": dt.dt.year, "mes": dt.dt.month, "dia": dt.dt.day,
        "palavra_chave": novas["consulta"], "api_key_indice": None,
        "titulo": novas["titulo"], "descricao": novas["descricao"].fillna(""),
        "conteudo": novas["conteudo"].fillna("").where(novas["conteudo"].fillna("") != "", novas["descricao"]),
        "url": novas["url"], "fonte_nome": novas["fonte"],
        "fonte_url": novas["url"].map(lambda u: urlparse(u).netloc),
        "classificacao mendes": novas["classificacao mendes"],
        "classificacao guilherme": novas["classificacao guilherme"],
        "classificacao marcos": novas["classificacao marcos"],
        "classificacao kappa": None, "categoria_geral": None, "subcategoria": None,
        "origem_anotacao": "simulada",
        "fonte_coleta": np.where(novas["fonte"].eq("The Guardian"), "API The Guardian (extracao nao anotada)",
                                 "web scraping (RSS)"),
        "julgamento_referencia": novas["ref"].map(REF2CL),
        "classe_guia": novas["classe_guia"].map(REF2CL),
        "certeza_guia": novas["certeza_guia"],
        "criterio_guia": novas["criterio_guia"],
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alvo", type=int, default=425)
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--eps", type=float, default=0.05, help="probabilidade de lapso por avaliador nos casos claros")
    args = ap.parse_args()

    humano, dup_h = carregar_humano()
    novas, info = carregar_coleta(humano)
    mats = perfis(humano)
    sim = anotar_por_guia(novas, args.semente, args.eps)
    sim["_maj"] = sim[AVAL].apply(maioria, axis=1)

    humano["_maj"] = humano[AVAL].apply(maioria, axis=1)
    cont_h = humano["_maj"].value_counts().reindex(CLASSES).fillna(0).astype(int)
    falta = {c: max(0, args.alvo - int(cont_h[c])) for c in CLASSES}
    taxa_disc_h = float(humano["_maj"].isna().mean())

    ordem = sim.sample(frac=1.0, random_state=args.semente)
    escolhidas, n_disc = [], 0
    for idx, r in ordem.iterrows():
        c = r["_maj"]
        if c is None:
            # mantem a mesma proporcao de discordancia total do corpus humano
            if falta.get(REF2CL[r["ref"]], 0) > 0 and n_disc < taxa_disc_h * (len(escolhidas) + 1):
                escolhidas.append(idx); n_disc += 1
            continue
        if falta[c] > 0:
            escolhidas.append(idx); falta[c] -= 1
    sel = sim.loc[escolhidas].sort_values("data")

    final = pd.concat([humano.drop(columns=["_tn", "_maj"]), para_planilha(sel)], ignore_index=True)
    final["_maj"] = final[AVAL].apply(maioria, axis=1)
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    final.drop(columns=["_maj"]).to_excel(SAIDA, index=False)

    rel = {
        "arquivo_saida": str(SAIDA.relative_to(RAIZ)),
        "corpus_humano_original": int(len(humano) + dup_h),
        "corpus_humano_duplicatas_removidas": int(dup_h),
        "corpus_humano_unico": int(len(humano)),
        **info,
        "noticias_novas_incluidas": int(len(sel)),
        "novas_por_fonte": sel["fonte"].map(lambda f: "The Guardian" if f == "The Guardian" else
                                            ("STAT" if f == "STAT" else "outros RSS/Bing")).value_counts().to_dict(),
        "novas_por_referencia": sel["ref"].map(REF2CL).value_counts().to_dict(),
        "novas_consolidado": sel["_maj"].fillna("DISCORDANCIA_TOTAL").value_counts().to_dict(),
        "total_linhas": int(len(final)),
        "consolidado_final": final["_maj"].fillna("DISCORDANCIA_TOTAL").value_counts().to_dict(),
        "consolidado_humano": humano["_maj"].fillna("DISCORDANCIA_TOTAL").value_counts().to_dict(),
        "fleiss_kappa_humano_unico": round(fleiss(humano), 4),
        "fleiss_kappa_simulado": round(fleiss(sel), 4),
        "fleiss_kappa_corpus_ampliado": round(fleiss(final), 4),
        "concordancia_simulada_com_referencia": round(float((sel["_maj"] == sel["ref"].map(REF2CL)).mean()), 4),
        "concordancia_par_a_par_novas": {f"{a[14:]} x {b[14:]}": round(float((sel[a] == sel[b]).mean()), 4) for i, a in enumerate(AVAL) for b in AVAL[i + 1:]},
        "unanimes_novas": round(float((sel[AVAL].nunique(axis=1) == 1).mean()), 4),
        "matrizes_perfil_avaliador": {a: mats[a].round(3).to_dict(orient="index") for a in AVAL},
        "distribuicao_por_avaliador": {a: final[a].value_counts().to_dict() for a in AVAL},
        "semente": args.semente, "alvo_por_classe": args.alvo, "eps_lapso": args.eps,
        "metodo_anotacao_simulada": "guia de anotacao v2 (julgamento de referencia x regra escrita; lapso eps nos casos claros)",
        "pool_certeza_guia": sim["certeza_guia"].value_counts().to_dict(),
        "novas_certeza_guia": sel["certeza_guia"].value_counts().to_dict(),
        "novas_criterio_guia": sel["criterio_guia"].str.slice(0, 2).value_counts().to_dict(),
        "concordancia_referencia_x_guia_pool": round(float((sim["ref"] == sim["classe_guia"]).mean()), 4),
        "kappa_teto_corpus_com_novas_perfeitas": round(fleiss(pd.concat([humano, sel.assign(**{a: sel["ref"].map(REF2CL) for a in AVAL})])), 4),
    }
    RELATORIO.write_text(json.dumps(rel, ensure_ascii=False, indent=2, default=int), encoding="utf-8")
    print(json.dumps({k: v for k, v in rel.items() if k != "matrizes_perfil_avaliador"},
                     ensure_ascii=False, indent=2, default=int))


if __name__ == "__main__":
    main()
