"""***************************************************************************************
SCRIPT    : dados.py
OBJETIVO  : Camada de dados do painel. Tudo o que a tela mostra sai daqui, ja
            agregado por SETOR - a interface nao le CSV nem faz conta.

DE ONDE VEM CADA COISA:
    indicadores mes a mes   fato_analise_kpi_final + fato_cota_mensal
    projecao das semanas    fato_desempenho_semanal
    recomendacoes do setor  data/agentes/agente3_saida.json
    noticias do setor       data/agentes/agente5_saida.json

    Cada tabela fato e lida de DOIS lugares e unida: o historico gerado uma vez
    (data/sintetico/) e as execucoes semanais (data/semanal/AAAA-MM-SN/). Assim
    cada "Gerar nova semana" aparece no painel sem reprocessar o historico.
    Quando o mesmo periodo existe nos dois, vale o historico.

CACHE EM DISCO: os CSVs do historico tem centenas de MB. Eles sao agregados por
setor uma unica vez e gravados em data/painel/; o agregado so e refeito se o
arquivo de origem mudar (tamanho ou data de modificacao). As pastas semanais
sao pequenas e entram sempre.

AUTOR     : Marcos Arambasic
DATA      : OUT/2026
****************************************************************************************/
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
DIR_SINT = RAIZ / "data" / "sintetico"
DIR_SEMANAL = RAIZ / "data" / "semanal"
DIR_AGENTES = RAIZ / "data" / "agentes"
DIR_CACHE = RAIZ / "data" / "painel"

INDICADORES = {
    "market_share": "Market share (próprios)",
    "cobertura_cota": "Cobertura da cota",
    "cagr_medio": "CAGR médio",
    "cv": "Coeficiente de variação",
    "indice_sazonal": "Índice sazonal",
}
# faixas de cobertura usadas pela empresa (dim_classifica_kpis)
FAIXAS_COBERTURA = [(0.65, "Crítico"), (0.75, "Atenção"), (0.80, "Próximo"), (np.inf, "Bateu o plano")]


# ============================================================================
# Utilitarios
# ============================================================================
def _assinatura(arquivos: list[Path]) -> list:
    return [[str(a.name), a.stat().st_size, int(a.stat().st_mtime)] for a in arquivos if a.exists()]


def _cache(nome: str, fontes: list[Path], construir) -> pd.DataFrame:
    """Le data/painel/<nome>.csv se as fontes nao mudaram; senao reconstroi."""
    DIR_CACHE.mkdir(parents=True, exist_ok=True)
    csv, meta = DIR_CACHE / f"{nome}.csv", DIR_CACHE / f"{nome}.meta.json"
    assinatura = _assinatura(fontes)
    if csv.exists() and meta.exists():
        try:
            if json.loads(meta.read_text(encoding="utf-8")) == assinatura:
                return pd.read_csv(csv)
        except (ValueError, OSError):
            pass
    df = construir()
    df.to_csv(csv, index=False)
    meta.write_text(json.dumps(assinatura), encoding="utf-8")
    return df


def _arquivos_historico(prefixo: str) -> list[Path]:
    return sorted(a for a in DIR_SINT.glob(f"{prefixo}*.csv")
                  if a.name[len(prefixo):].startswith((".", "_2")) )


def _arquivos_semanais(nome: str) -> list[Path]:
    return sorted(DIR_SEMANAL.glob(f"*/{nome}.csv")) if DIR_SEMANAL.exists() else []


def _ler(arquivos: list[Path], usecols: list[str]) -> pd.DataFrame:
    if not arquivos:
        return pd.DataFrame(columns=usecols)
    return pd.concat((pd.read_csv(a, usecols=usecols) for a in arquivos), ignore_index=True)


# ============================================================================
# Dimensoes
# ============================================================================
@dataclass
class Dimensoes:
    setores: pd.DataFrame        # id_setor, codigo_setor, nome, sigla_uf, desc_uf, rotulo
    produtos: pd.DataFrame       # id_produto, nome_sintetico, grupo_produto, tipo_produto
    outlets: pd.DataFrame        # id_outlet, codigo_outlet

    @property
    def proprios(self) -> set:
        return set(self.produtos.loc[self.produtos["tipo_produto"] == "PROPRIO", "id_produto"])

    def nome_produto(self, i) -> str:
        m = self.produtos.set_index("id_produto")["nome_sintetico"]
        return str(m.get(i, f"Produto {i}"))

    def nome_outlet(self, i) -> str:
        m = self.outlets.set_index("id_outlet")["codigo_outlet"]
        return f"Brick {m[i]}" if i in m.index else f"Outlet {i}"


def carregar_dimensoes() -> Dimensoes:
    setor = pd.read_csv(DIR_SINT / "dim_setor.csv")
    uf = pd.read_csv(DIR_SINT / "dim_uf.csv")
    setor = setor.merge(uf, on="id_uf", how="left")
    setor["rotulo"] = setor["sigla_uf"] + " · " + setor["desc_uf"] + " (" + setor["codigo_setor"] + ")"
    setor = setor.sort_values("sigla_uf").reset_index(drop=True)
    return Dimensoes(
        setores=setor,
        produtos=pd.read_csv(DIR_SINT / "dim_produto.csv"),
        outlets=pd.read_csv(DIR_SINT / "dim_outlet.csv"),
    )


# ============================================================================
# Indicadores mes a mes (por setor)
# ============================================================================
COLS_KPI = ["periodo_demanda", "id_produto", "id_setor", "id_outlet",
            "demanda_produto", "demanda_mercado", "cagr_medio", "cv", "indice_sazonal"]


def _agregar_kpi(kpi: pd.DataFrame, proprios: set) -> pd.DataFrame:
    if kpi.empty:
        return pd.DataFrame(columns=["id_setor", "periodo_demanda"])
    kpi = kpi.copy()
    kpi["periodo_demanda"] = pd.to_datetime(kpi["periodo_demanda"]).dt.strftime("%Y-%m-01")
    chave = ["id_setor", "periodo_demanda"]

    # mercado: um valor por outlet/mes (repetido em todas as linhas de produto)
    mercado = (kpi.groupby(chave + ["id_outlet"])["demanda_mercado"].max()
               .groupby(level=[0, 1]).sum().rename("demanda_mercado"))

    p = kpi[kpi["id_produto"].isin(proprios)].copy()
    w = p["demanda_produto"].clip(lower=0)
    for c in ("cagr_medio", "cv", "indice_sazonal"):
        p[f"_{c}"] = p[c] * w
    g = p.assign(_w=w).groupby(chave)
    agg = pd.DataFrame({
        "demanda_proprios": g["demanda_produto"].sum(),
        "_w": g["_w"].sum(),
        **{f"_{c}": g[f"_{c}"].sum() for c in ("cagr_medio", "cv", "indice_sazonal")},
    })
    out = agg.join(mercado, how="outer").fillna(0).reset_index()
    with np.errstate(divide="ignore", invalid="ignore"):
        for c in ("cagr_medio", "cv", "indice_sazonal"):
            out[c] = np.where(out["_w"] > 0, out[f"_{c}"] / out["_w"], np.nan)
    return out[chave + ["demanda_proprios", "demanda_mercado", "cagr_medio", "cv", "indice_sazonal"]]


def _agregar_cota(cota: pd.DataFrame) -> pd.DataFrame:
    if cota.empty:
        return pd.DataFrame(columns=["id_setor", "periodo_demanda", "cota"])
    cota = cota.copy()
    cota["periodo_demanda"] = pd.to_datetime(cota["periodo_demanda"]).dt.strftime("%Y-%m-01")
    return (cota.groupby(["id_setor", "periodo_demanda"], as_index=False)["cota_unidades"].sum()
            .rename(columns={"cota_unidades": "cota"}))


def indicadores_mensais(dims: Dimensoes) -> pd.DataFrame:
    """Uma linha por setor x mes com os indicadores do painel."""
    hist_kpi = _arquivos_historico("fato_analise_kpi_final")
    hist_cota = _arquivos_historico("fato_cota_mensal")
    base_kpi = _cache("kpi_setor_mes", hist_kpi,
                      lambda: _agregar_kpi(_ler(hist_kpi, COLS_KPI), dims.proprios))
    base_cota = _cache("cota_setor_mes", hist_cota,
                       lambda: _agregar_cota(_ler(hist_cota, ["periodo_demanda", "id_setor",
                                                               "id_produto", "cota_unidades"])))

    sem_kpi = _agregar_kpi(_ler(_arquivos_semanais("fato_analise_kpi_final"), COLS_KPI), dims.proprios)
    sem_cota = _agregar_cota(_ler(_arquivos_semanais("fato_cota_mensal"),
                                  ["periodo_demanda", "id_setor", "id_produto", "cota_unidades"]))
    chave = ["id_setor", "periodo_demanda"]
    # o historico tem precedencia: a pasta semanal so entra com periodos que o
    # historico nao cobre (reprocessar uma semana antiga nao mistura versoes)
    kpi = pd.concat([sem_kpi, base_kpi]).drop_duplicates(chave, keep="last")
    cota = pd.concat([sem_cota, base_cota]).drop_duplicates(chave, keep="last")

    out = kpi.merge(cota, on=chave, how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        out["market_share"] = np.where(out["demanda_mercado"] > 0,
                                       out["demanda_proprios"] / out["demanda_mercado"], np.nan)
        out["cobertura_cota"] = np.where(out["cota"] > 0, out["demanda_proprios"] / out["cota"], np.nan)
    out["periodo_demanda"] = pd.to_datetime(out["periodo_demanda"])
    return out.sort_values(chave).reset_index(drop=True)


def serie_indicadores_setor(ind: pd.DataFrame, id_setor: int, meses: int | None = 12) -> pd.DataFrame:
    """Formato longo (periodo, indicador, valor) para o grafico de evolucao."""
    d = ind[ind["id_setor"] == id_setor].sort_values("periodo_demanda")
    if meses:
        d = d.tail(meses)
    longo = d.melt(id_vars="periodo_demanda", value_vars=list(INDICADORES),
                   var_name="indicador", value_name="valor").dropna(subset=["valor"])
    longo["indicador"] = longo["indicador"].map(INDICADORES)
    return longo


# ============================================================================
# Projecao das semanas do mes (por setor)
# ============================================================================
COLS_DESEMP = ["periodo_demanda", "nr_semana", "id_setor", "id_produto", "id_alocacao",
               "demanda_unidades_acumulada", "demanda_unidades_projetada", "cota_mensal"]


def _agregar_desempenho(d: pd.DataFrame) -> pd.DataFrame:
    if d.empty:
        return pd.DataFrame(columns=["id_setor", "periodo_demanda", "nr_semana",
                                     "acumulado", "projetado", "cota"])
    d = d.drop_duplicates(["periodo_demanda", "nr_semana", "id_produto", "id_alocacao"], keep="last")
    return (d.groupby(["id_setor", "periodo_demanda", "nr_semana"], as_index=False)
             .agg(acumulado=("demanda_unidades_acumulada", "sum"),
                  projetado=("demanda_unidades_projetada", "sum"),
                  cota=("cota_mensal", "sum")))


def projecao_semanal(dims: Dimensoes | None = None) -> pd.DataFrame:
    hist = _arquivos_historico("fato_desempenho_semanal")
    base = _cache("desempenho_setor_semana", hist, lambda: _agregar_desempenho(_ler(hist, COLS_DESEMP)))
    sem = _agregar_desempenho(_ler(_arquivos_semanais("fato_desempenho_semanal"), COLS_DESEMP))
    chave = ["id_setor", "periodo_demanda", "nr_semana"]
    # historico tem precedencia sobre reprocessamentos de semanas antigas
    out = pd.concat([sem, base]).drop_duplicates(chave, keep="last")
    out["periodo_demanda"] = pd.to_datetime(out["periodo_demanda"])
    out["periodo_mensal"] = out["periodo_demanda"].dt.to_period("M").dt.to_timestamp()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["cobertura_projetada"] = np.where(out["cota"] > 0, out["projetado"] / out["cota"], np.nan)
    return out.sort_values(chave).reset_index(drop=True)


def semanas_do_mes_setor(proj: pd.DataFrame, id_setor: int, periodo_mensal: pd.Timestamp) -> pd.DataFrame:
    """Todas as semanas comerciais do mes, preenchidas com o que ja foi calculado."""
    import sys
    sys.path.insert(0, str(RAIZ))
    from src.sintetico.calendario import semanas_do_mes

    cal = pd.DataFrame([{"nr_semana": s.nr_semana, "inicio": s.inicio, "fim": s.fim}
                        for s in semanas_do_mes(periodo_mensal.year, periodo_mensal.month)])
    d = proj[(proj["id_setor"] == id_setor) & (proj["periodo_mensal"] == periodo_mensal)]
    out = cal.merge(d[["nr_semana", "acumulado", "projetado", "cota", "cobertura_projetada"]],
                    on="nr_semana", how="left")
    out["cota"] = out["cota"].ffill().bfill()
    out["rotulo"] = out.apply(lambda r: f"S{r.nr_semana} · {r.inicio:%d/%m}–{r.fim:%d/%m}", axis=1)
    out["calculada"] = out["acumulado"].notna()
    return out


def classificar_cobertura(c: float) -> str:
    if c is None or pd.isna(c):
        return "—"
    for limite, nome in FAIXAS_COBERTURA:
        if c < limite:
            return nome
    return FAIXAS_COBERTURA[-1][1]


# ============================================================================
# Semana vigente
# ============================================================================
def semana_vigente(proj: pd.DataFrame) -> dict | None:
    """Ultima semana com dados calculados e a proxima a rodar."""
    import sys
    sys.path.insert(0, str(RAIZ))
    from src.sintetico.calendario import semanas_do_mes

    if proj.empty:
        return None
    fim = proj["periodo_demanda"].max()
    nr = int(proj.loc[proj["periodo_demanda"] == fim, "nr_semana"].iloc[0])
    sems = semanas_do_mes(fim.year, fim.month)
    atual = next(s for s in sems if s.nr_semana == nr)
    if nr < len(sems):
        prox = sems[nr]
    else:
        a, m = (fim.year + 1, 1) if fim.month == 12 else (fim.year, fim.month + 1)
        prox = semanas_do_mes(a, m)[0]
    return {
        "ano": fim.year, "mes": fim.month, "nr_semana": nr, "total": len(sems),
        "inicio": atual.inicio, "fim": atual.fim,
        "periodo_mensal": pd.Timestamp(fim.year, fim.month, 1),
        "proxima": {"ano": prox.ano, "mes": prox.mes, "nr_semana": prox.nr_semana,
                    "inicio": prox.inicio, "fim": prox.fim},
    }


# ============================================================================
# Saidas dos agentes
# ============================================================================
def _json(nome: str) -> dict | None:
    p = DIR_AGENTES / nome
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def recomendacoes_setor(id_setor: int, dims: Dimensoes, n: int = 8) -> tuple[pd.DataFrame, dict | None]:
    a3 = _json("agente3_saida.json")
    if not a3:
        return pd.DataFrame(), None
    s = next((x for x in a3["setores"] if x["id_setor"] == id_setor), None)
    meta = {"periodo_previsto": a3["periodo_previsto"], "algoritmo": a3["modelo"]["algoritmo"],
            "avisos": a3.get("avisos", []),
            "oportunidade_un": s["oportunidade_total_un"] if s else 0,
            "risco_un": s["risco_total_un"] if s else 0}
    if not s:
        return pd.DataFrame(), meta
    linhas = [{
        "Prioridade": it["prioridade"],
        "Recomendação": it["tipo_recomendacao"].replace("_", " ").capitalize(),
        "Produto": dims.nome_produto(it["id_produto"]),
        "Local": dims.nome_outlet(it["id_outlet"]),
        "Unidades em jogo": it["previsao"]["valor_em_jogo_un"],
        "Ação sugerida": it["acao_sugerida"],
        "natureza": it["natureza"],
    } for it in s["itens"][:n]]
    return pd.DataFrame(linhas), meta


# ============================================================================
# Filtro de noticias por UF (somente exibicao - nao altera nenhum agente)
# ============================================================================
# Brasilia nao conta como DF: noticias federais (Anvisa, Ministerio da Saude)
# citam a capital sem ser regionais.
# Cada UF e reconhecida pelo nome do estado, por capitais de nome inequivoco e
# pela sigla em contexto tipico de noticia ("em MG", "(SP)", "-RJ", "/PR").
# Os nomes sao comparados sem acento e sem diferenciar maiusculas, EXCETO
# "Pará" e "Acre", que colidem com palavras comuns ("para", "acre"): esses
# dois so contam escritos com a grafia propria do estado.
_UF_NOMES: dict[str, list[str]] = {
    "AC": ["rio branco"], "AL": ["alagoas", "maceio"], "AP": ["amapa", "macapa"],
    "AM": ["amazonas", "manaus"], "BA": ["bahia"], "CE": ["ceara", "fortaleza"],
    "DF": ["distrito federal"], "ES": ["espirito santo"],
    "GO": ["goias", "goiania"], "MA": ["maranhao", "sao luis"],
    "MT": ["mato grosso", "cuiaba"], "MS": ["mato grosso do sul"],
    "MG": ["minas gerais", "belo horizonte"], "PA": ["belem"],
    "PB": ["paraiba", "joao pessoa"], "PR": ["parana", "curitiba"],
    "PE": ["pernambuco", "recife"], "PI": ["piaui", "teresina"],
    "RJ": ["rio de janeiro"], "RN": ["rio grande do norte"],
    "RS": ["rio grande do sul", "porto alegre"], "RO": ["rondonia", "porto velho"],
    "RR": ["roraima"], "SC": ["santa catarina", "florianopolis"],
    "SP": ["sao paulo"], "SE": ["sergipe", "aracaju"], "TO": ["tocantins"],
}
_UF_GRAFIA_EXATA = {"PA": re.compile(r"\bPar[áÁ]\b|\bPAR[ÁA]\b(?=\s)"), "AC": re.compile(r"\bAcre\b")}
_UF_SIGLA = {uf: re.compile(rf"(?:\(|\b(?:em|no|na|do|da|de|in)\s+|[-/–]\s?){uf}\b|\b{uf}\)")
             for uf in _UF_NOMES}


def _sem_acento(t: str) -> str:
    import unicodedata
    t = unicodedata.normalize("NFKD", t or "")
    return re.sub(r"\s+", " ", "".join(c for c in t if not unicodedata.combining(c)).lower())


def ufs_citadas(texto: str) -> set[str]:
    """Siglas das UFs mencionadas em um texto."""
    texto = texto or ""
    norm = _sem_acento(texto)
    achadas: set[str] = set()
    # "mato grosso do sul" precisa sair antes de testar "mato grosso"
    if re.search(r"\bmato grosso do sul\b", norm):
        achadas.add("MS")
        norm = re.sub(r"\bmato grosso do sul\b", " ", norm)
    for uf, nomes in _UF_NOMES.items():
        if uf == "MS":
            continue
        if any(re.search(rf"\b{re.escape(n)}\b", norm) for n in nomes):
            achadas.add(uf)
    for uf, rx in _UF_GRAFIA_EXATA.items():
        if rx.search(texto):
            achadas.add(uf)
    for uf, rx in _UF_SIGLA.items():
        if rx.search(texto):
            achadas.add(uf)
    return achadas


def noticia_pertence_a_uf(texto: str, sigla: str) -> bool:
    """
    Fica a noticia que nao cita UF nenhuma (alcance nacional) ou que cita a UF
    do setor. Sai a que cita apenas outras UFs.
    """
    citadas = ufs_citadas(texto)
    return not citadas or sigla in citadas


def _conteudo_por_url() -> dict:
    """Corpo das noticias (os JSONs dos agentes guardam so titulo e descricao)."""
    chave = tuple(int(p.stat().st_mtime) for p in sorted((RAIZ / "data").glob("transformacao_noticias_*.xlsx")))
    if getattr(_conteudo_por_url, "_chave", None) != chave:
        mapa = {}
        for p in sorted((RAIZ / "data").glob("transformacao_noticias_*.xlsx")):
            try:
                d = pd.read_excel(p, usecols=["url", "conteudo"])
                mapa.update(dict(zip(d["url"], d["conteudo"].fillna("").astype(str))))
            except (ValueError, KeyError):
                continue
        _conteudo_por_url._mapa, _conteudo_por_url._chave = mapa, chave
    return _conteudo_por_url._mapa


def _texto_noticia(n: dict, corpos: dict) -> str:
    return " ".join(str(n.get(k) or "") for k in ("titulo", "descricao")) + " " + corpos.get(n.get("url"), "")


def _complementos_agente4(id_setor: int, sigla: str, excluir: set, faltam: int,
                          dims: "Dimensoes | None", corpos: dict) -> list[dict]:
    """
    Quando o filtro remove destaques do Agente 5, completa a lista com as
    proximas noticias que o Agente 4 ja ranqueou para o setor (mesma ordem de
    score). Esses cards nao tem texto do Agente 5: mostram a acao do Agente 3.
    """
    a4 = _json("agente4_saida.json")
    if not a4 or faltam <= 0:
        return []
    cat = {n["id_noticia"]: n for n in a4["noticias"]}
    s = next((x for x in a4["setores"] if x["id_setor"] == id_setor), None)
    if not s:
        return []
    por_noticia: dict[str, list] = {}
    for a in s["achados"]:
        por_noticia.setdefault(a["id_noticia"], []).append(a)
    saida = []
    for idn, achs in sorted(por_noticia.items(), key=lambda kv: -max(a["score"] for a in kv[1])):
        n = cat.get(idn)
        if not n or idn in excluir or not noticia_pertence_a_uf(_texto_noticia(n, corpos), sigla):
            continue
        achs = sorted(achs, key=lambda a: -a["score"])
        vistos, ops = set(), []
        for a in achs:
            k = (a["id_alocacao"], a["id_produto"])
            if k in vistos:
                continue
            vistos.add(k)
            ops.append({
                "id_alocacao": a["id_alocacao"], "id_outlet": a["id_outlet"], "id_produto": a["id_produto"],
                "produto": dims.nome_produto(a["id_produto"]) if dims else f"Produto {a['id_produto']}",
                "local": dims.nome_outlet(a["id_outlet"]) if dims else f"Outlet {a['id_outlet']}",
                "tipo_recomendacao": a["tipo_recomendacao"], "natureza": a["natureza"],
                "categoria": a["categoria"], "valor_em_jogo_un": a["valor_em_jogo_un"],
                "efeito": a["efeito"], "score": a["score"]})
            if len(ops) == 3:
                break
        top = achs[0]
        saida.append({
            "score": top["score"], "noticia": n, "oportunidades": ops, "origem": "agente4",
            "recomendacao": {
                "acao_sugerida": top["acao_agente3"],
                "justificativa": top["justificativa"],
                "urgencia": {"alta": "alta", "media": "media"}.get(top.get("prioridade_agente3"), "baixa"),
                "impacto_esperado": f"~{sum((o['valor_em_jogo_un'] or 0) for o in ops):,.0f} unidades em jogo"
                                    .replace(",", "."),
            }})
        if len(saida) == faltam:
            break
    return saida


def noticias_setor(id_setor: int, sigla_uf: str | None = None, dims: "Dimensoes | None" = None,
                   n: int = 4) -> tuple[list[dict], dict | None]:
    """
    Destaques de noticias do setor. Com `sigla_uf`, aplica o filtro de exibicao
    por UF e completa a lista com as proximas noticias do Agente 4, se preciso.
    A saida dos agentes nao e alterada.
    """
    a5 = _json("agente5_saida.json")
    if not a5:
        return [], None
    s = next((x for x in a5["setores"] if x["id_setor"] == id_setor), None)
    destaques = [dict(d, origem="agente5") for d in (s["destaques"] if s else [])]
    meta = {"gerado_em": a5.get("gerado_em"), "ocultadas": 0, "complementadas": 0, "filtro_uf": sigla_uf}
    if not sigla_uf:
        return destaques, meta

    corpos = _conteudo_por_url()
    mantidos, ocultadas = [], []
    for d in destaques:
        texto = _texto_noticia(d["noticia"], corpos)
        if noticia_pertence_a_uf(texto, sigla_uf):
            mantidos.append(d)
        else:
            ocultadas.append(sorted(ufs_citadas(texto)))
    excluir = {d["noticia"]["id_noticia"] for d in destaques}
    extra = _complementos_agente4(id_setor, sigla_uf, excluir, n - len(mantidos), dims, corpos) \
        if len(mantidos) < n else []
    final = (mantidos + extra)[:n]
    for i, d in enumerate(final, start=1):
        d["posicao"] = i
    meta.update(ocultadas=len(ocultadas), complementadas=len(extra),
                ufs_ocultadas=sorted({u for lst in ocultadas for u in lst}))
    return final, meta


def ultima_execucao_noticias() -> dict | None:
    p = RAIZ / "data" / "noticias" / "_execucoes.json"
    if not p.exists():
        return None
    h = json.loads(p.read_text(encoding="utf-8"))
    return h[-1] if h else None
