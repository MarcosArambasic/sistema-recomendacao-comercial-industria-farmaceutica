"""***************************************************************************************
SCRIPT    : gerar_figuras_artigo.py
OBJETIVO  : Gera as figuras do artigo final (docs/artigo/figuras/) a partir dos
            MESMOS dados e da MESMA camada de dados do painel (src/painel/dados.py)
            e dos JSONs de metricas dos agentes.

            As figuras da tela do painel sao REPRODUCOES estaticas do layout do
            Streamlit (app/painel.py) desenhadas com matplotlib: mesmo setor,
            mesmos numeros, mesmas recomendacoes. Para substituir por capturas
            reais: streamlit run app/painel.py, setor SP, e salvar com os mesmos
            nomes de arquivo.

USO:
    python scripts/gerar_figuras_artigo.py [--setor SP]
****************************************************************************************/
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
from src.painel import dados as D  # noqa: E402

DEST = RAIZ / "docs" / "artigo" / "figuras"
DIR_AG = RAIZ / "data" / "agentes"
MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
AZUL, LARANJA, VERMELHO, VERDE, CINZA = "#2a78d6", "#eb6834", "#d62728", "#1baf7a", "#6b6b66"
TEMAS = {
    "claro": dict(fundo="#ffffff", lateral="#f0f2f6", texto="#31333f", suave="#6b7280",
                  borda="#d9dce3", campo="#ffffff", primario="#1f6feb"),
    "escuro": dict(fundo="#0e1117", lateral="#1a1d26", texto="#fafafa", suave="#a3a8b8",
                   borda="#3a3f4b", campo="#262a35", primario="#1f6feb"),
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def _j(nome):
    p = DIR_AG / nome
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def mes_ano(ts) -> str:
    ts = pd.Timestamp(ts)
    return f"{MESES[ts.month - 1]}/{ts.year}"


def _salvar(fig, nome):
    DEST.mkdir(parents=True, exist_ok=True)
    fig.savefig(DEST / nome, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  ", DEST / nome)


# ============================================================================
# Reproducao da tela do painel
# ============================================================================
def _caixa(fig, x, y, w, h, cor, borda=None, raio=0.006, lw=0.8):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={raio}",
                                      transform=fig.transFigure, facecolor=cor,
                                      edgecolor=borda or cor, linewidth=lw, zorder=0))


def _txt(fig, x, y, s, t, size=9, cor=None, peso="normal", ha="left", va="center"):
    fig.text(x, y, s, fontsize=size, color=cor or t["texto"], fontweight=peso, ha=ha, va=va)


def tela_painel(sigla: str, tema: str, nome: str):
    t = TEMAS[tema]
    dims = D.carregar_dimensoes()
    setor = dims.setores[dims.setores["sigla_uf"] == sigla].iloc[0]
    sid = int(setor["id_setor"])
    ind = D.indicadores_mensais(dims)
    proj = D.projecao_semanal()
    vig = D.semana_vigente(proj)
    recs, meta3 = D.recomendacoes_setor(sid, dims)
    noticias, _ = D.noticias_setor(sid)
    ult = D.ultima_execucao_noticias()

    fig = plt.figure(figsize=(16, 11.5), facecolor=t["fundo"])
    # ---------------- barra lateral ----------------
    _caixa(fig, 0, 0, 0.185, 1, t["lateral"], raio=0.0)
    _txt(fig, 0.012, 0.965, "Painel do setor", t, 13, peso="bold")
    _caixa(fig, 0.135, 0.955, 0.03, 0.02, t["primario"] if tema == "escuro" else t["borda"], raio=0.01)
    fig.patches.append(plt.Circle((0.158 if tema == "escuro" else 0.142, 0.965), 0.0075,
                                  transform=fig.transFigure, color="white"))
    _txt(fig, 0.135, 0.94, "Escuro", t, 8, t["suave"])
    _txt(fig, 0.012, 0.905, "Simular setor", t, 9)
    _caixa(fig, 0.012, 0.865, 0.16, 0.028, t["campo"], t["borda"])
    _txt(fig, 0.02, 0.879, setor["rotulo"], t, 9)
    _txt(fig, 0.165, 0.879, "v", t, 8, t["suave"], ha="right")
    fig.lines.append(plt.Line2D([0.012, 0.172], [0.84, 0.84], transform=fig.transFigure, color=t["borda"]))
    _caixa(fig, 0.012, 0.79, 0.16, 0.032, t["primario"])
    _txt(fig, 0.092, 0.806, "▶ Gerar nova semana", t, 10, "white", "bold", ha="center")
    if vig:
        _txt(fig, 0.012, 0.765, f"Semana vigente: {MESES[vig['mes'] - 1]}/{vig['ano']} · "
                                f"semana {vig['nr_semana']} de {vig['total']}", t, 8.5, peso="bold")
        p = vig["proxima"]
        _txt(fig, 0.012, 0.747, f"{vig['inicio']:%d/%m} a {vig['fim']:%d/%m/%Y} · próxima: "
                                f"semana {p['nr_semana']} de {MESES[p['mes'] - 1]}/{p['ano']}", t, 7.5, t["suave"])
    _caixa(fig, 0.012, 0.69, 0.16, 0.032, t["campo"], t["borda"])
    _txt(fig, 0.092, 0.706, "Gerar mais notícias", t, 10, ha="center")
    if ult:
        _txt(fig, 0.012, 0.672, f"Última coleta: {pd.Timestamp(ult['executado_em']):%d/%m/%Y %H:%M} · "
                                f"{ult.get('noticias_validas', 0)} notícias", t, 7.5, t["suave"])
        _txt(fig, 0.012, 0.656, "de impacto alto/médio", t, 7.5, t["suave"])

    # ---------------- cabecalho ----------------
    x0 = 0.205
    _txt(fig, x0, 0.965, f"{setor['desc_uf']} · {setor['codigo_setor']}", t, 20, peso="bold")
    if meta3 and vig:
        _txt(fig, x0, 0.937, f"Dados até {MESES[vig['mes'] - 1]}/{vig['ano']} (semana {vig['nr_semana']}) · "
                             f"recomendações para {mes_ano(meta3['periodo_previsto'] + '-01')} · "
                             f"modelo do Agente 3: {meta3['algoritmo']}", t, 9, t["suave"])
    _txt(fig, x0, 0.905, "Indicadores mês a mês", t, 13, peso="bold")
    _txt(fig, 0.98, 0.905, "○ 6 meses   ● 12 meses   ○ 24 meses   ○ Tudo", t, 8.5, t["suave"], ha="right")

    serie = D.serie_indicadores_setor(ind, sid, 12)
    fmt = {"Market share (próprios)": (True, 1), "Cobertura da cota": (True, 0), "CAGR médio": (True, 1),
           "Coeficiente de variação": (True, 1), "Índice sazonal": (False, 2)}
    largura = (0.98 - x0) / 5
    for i, nome_ind in enumerate(D.INDICADORES.values()):
        s = serie[serie["indicador"] == nome_ind].sort_values("periodo_demanda")
        cx = x0 + i * largura
        if s.empty:
            continue
        pct, casas = fmt[nome_ind]
        atual = s["valor"].iloc[-1]
        ant = s["valor"].iloc[-2] if len(s) > 1 else np.nan
        _txt(fig, cx, 0.875, nome_ind, t, 8.5, t["suave"])
        _txt(fig, cx, 0.85, f"{atual:.{casas}%}" if pct else f"{atual:.2f}", t, 17)
        dlt = (atual - ant) * (100 if pct else 1)
        cor = VERDE if dlt >= 0 else VERMELHO
        _txt(fig, cx, 0.826, f"{'▲' if dlt >= 0 else '▼'} {dlt:+.{1 if pct else 2}f}{' p.p.' if pct else ''}",
             t, 8.5, cor)
        ax = fig.add_axes([cx + 0.008, 0.705, largura - 0.022, 0.1])
        ax.plot(s["periodo_demanda"], s["valor"], color=AZUL, marker="o", ms=2.5, lw=1.6)
        if nome_ind == "Cobertura da cota":
            ax.axhline(1.0, color=VERMELHO, ls="--", lw=0.9)
        _eixo(ax, t, pct)
        ax.set_xticks(s["periodo_demanda"].iloc[::4])
        ax.set_xticklabels([f"{MESES[d.month - 1]}/{str(d.year)[2:]}" for d in s["periodo_demanda"].iloc[::4]])

    fig.lines.append(plt.Line2D([x0, 0.98], [0.675, 0.675], transform=fig.transFigure, color=t["borda"]))

    # ---------------- projecao ----------------
    _txt(fig, x0, 0.65, "Projeção das semanas do mês", t, 13, peso="bold")
    mes = vig["periodo_mensal"] if vig else None
    _caixa(fig, 0.5, 0.638, 0.1, 0.024, t["campo"], t["borda"])
    _txt(fig, 0.507, 0.65, mes_ano(mes), t, 9)
    sem = D.semanas_do_mes_setor(proj, sid, pd.Timestamp(mes))
    ultima = sem[sem["calculada"]].tail(1)
    if not ultima.empty:
        cob = float(ultima["cobertura_projetada"].iloc[0])
        for k, (rot, val, sub) in enumerate([
            ("Cobertura projetada", f"{cob:.0%}", D.classificar_cobertura(cob)),
            ("Projeção de fechamento", f"{ultima['projetado'].iloc[0]:,.0f} un".replace(",", "."), None),
            ("Cota do mês", f"{ultima['cota'].iloc[0]:,.0f} un".replace(",", "."), None)]):
            cx = x0 + k * 0.135
            _txt(fig, cx, 0.618, rot, t, 8.5, t["suave"])
            _txt(fig, cx, 0.595, val, t, 15)
            if sub:
                _txt(fig, cx, 0.574, f"▲ {sub}", t, 8.5, VERDE if cob >= 0.8 else VERMELHO)
    ax = fig.add_axes([x0 + 0.025, 0.4, 0.375, 0.155])
    xs = np.arange(len(sem))
    ax.bar(xs, sem["acumulado"].fillna(0), color="#4c78a8", alpha=0.85, width=0.6, label="realizado acumulado")
    calc = sem["calculada"].to_numpy()
    ax.plot(xs[calc], sem["projetado"][calc], color="#f58518", marker="o", lw=2.5, label="projeção de fechamento")
    ax.axhline(sem["cota"].iloc[0], color=VERMELHO, ls="--", lw=1.6, label="cota do mês")
    ax.set_xticks(xs)
    ax.set_xticklabels(sem["rotulo"], fontsize=7)
    _eixo(ax, t, False, unidades=True)
    leg = ax.legend(fontsize=7, loc="lower right", frameon=False)
    for tx in leg.get_texts():
        tx.set_color(t["suave"])

    # ---------------- recomendacoes ----------------
    xr = 0.645
    _txt(fig, xr, 0.65, "Recomendações do setor", t, 13, peso="bold")
    if meta3:
        _txt(fig, xr, 0.618, "Oportunidade prevista", t, 8.5, t["suave"])
        _txt(fig, xr, 0.595, f"{meta3['oportunidade_un']:,.0f} un".replace(",", "."), t, 15)
        _txt(fig, xr + 0.17, 0.618, "Risco previsto", t, 8.5, t["suave"])
        _txt(fig, xr + 0.17, 0.595, f"{meta3['risco_un']:,.0f} un".replace(",", "."), t, 15)
    cores_p = {"alta": VERMELHO, "media": LARANJA, "baixa": CINZA}
    for k, r in enumerate(recs.head(3).to_dict("records")):
        y = 0.51 - k * 0.058
        _caixa(fig, xr, y, 0.335, 0.052, t["fundo"], t["borda"])
        _txt(fig, xr + 0.006, y + 0.04, f"{r['Recomendação']}", t, 9, peso="bold")
        _txt(fig, xr + 0.13, y + 0.04, r["Prioridade"].upper(), t, 8.5, cores_p.get(r["Prioridade"]), "bold")
        _txt(fig, xr + 0.006, y + 0.024, f"{r['Produto']} · {r['Local']} · "
                                         f"{(r['Unidades em jogo'] or 0):,.0f} un em jogo".replace(",", "."), t, 8)
        _txt(fig, xr + 0.006, y + 0.009, textwrap.shorten(r["Ação sugerida"], 92, placeholder="..."),
             t, 7, t["suave"])

    fig.lines.append(plt.Line2D([x0, 0.98], [0.355, 0.355], transform=fig.transFigure, color=t["borda"]))

    # ---------------- noticias ----------------
    _txt(fig, x0, 0.33, "Notícias mais impactantes e as oportunidades que elas afetam", t, 13, peso="bold")
    cores_i = {"alto": VERMELHO, "medio": LARANJA, "baixo": CINZA}
    for k, d in enumerate(noticias[:4]):
        cx = x0 + (k % 2) * 0.39
        cy = 0.165 if k < 2 else 0.01
        _caixa(fig, cx, cy, 0.38, 0.145, t["fundo"], t["borda"])
        n, rec = d["noticia"], d["recomendacao"]
        linhas = textwrap.wrap(f"{d['posicao']}. {n['titulo']}", 78)[:2]
        for j, ln in enumerate(linhas):
            _txt(fig, cx + 0.008, cy + 0.128 - j * 0.016, ln, t, 8.5, AZUL, "bold")
        data = pd.to_datetime(n.get("data_publicacao"), errors="coerce")
        _txt(fig, cx + 0.008, cy + 0.093, f"{n.get('fonte') or '—'} · "
                                          f"{data:%d/%m/%Y} · escopo: {n['escopo']}" if pd.notna(data)
             else f"escopo: {n['escopo']}", t, 7.5, t["suave"])
        _txt(fig, cx + 0.008, cy + 0.077, f"Impacto {n['nivel_impacto'].upper()}", t, 8,
             cores_i.get(n["nivel_impacto"]), "bold")
        _txt(fig, cx + 0.062 if n["nivel_impacto"] != "medio" else cx + 0.068, cy + 0.077,
             f"· {n['sentimento']}", t, 8)
        for j, o in enumerate(d["oportunidades"][:2]):
            _txt(fig, cx + 0.008, cy + 0.06 - j * 0.014,
                 f"• {o['produto']} · {o['local']} — {o['tipo_recomendacao'].replace('_', ' ').lower()}, "
                 f"{(o['valor_em_jogo_un'] or 0):,.0f} un".replace(",", "."), t, 7.5)
        _txt(fig, cx + 0.008, cy + 0.03, f"Recomendação · urgência {rec['urgencia']}", t, 8,
             cores_p.get(rec["urgencia"], CINZA), "bold")
        _txt(fig, cx + 0.008, cy + 0.014, textwrap.shorten(rec["acao_sugerida"], 100, placeholder="..."),
             t, 7.2)
    _salvar(fig, nome)


def _eixo(ax, t, pct, unidades=False):
    ax.set_facecolor(t["fundo"])
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(t["borda"])
    ax.tick_params(colors=t["suave"], labelsize=7)
    ax.grid(axis="y", color=t["borda"], lw=0.5, alpha=0.7)
    if pct:
        lo, hi = ax.get_ylim()
        casas = 1 if (hi - lo) < 0.05 else 0
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=casas))
    if unidades:
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".")))


# ============================================================================
# Figuras de resultados
# ============================================================================
def _limpo(ax):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    ax.set_axisbelow(True)


def fig_arquitetura():
    fig = plt.figure(figsize=(13, 6.2), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 6.2)

    def bloco(x, y, w, h, titulo, sub, cor):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=cor, edgecolor="#2b2b28", lw=0.8))
        ax.text(x + w / 2, y + h * 0.64, titulo, ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=10.5, color="#3a3a36")

    def seta(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, color="#2b2b28", lw=1))

    ax.add_patch(FancyBboxPatch((0.2, 5.1), 12.6, 0.9, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor="#e8eefc", edgecolor="#1f6feb", lw=1.2))
    ax.text(6.5, 5.67, "AGENTE ORQUESTRADOR", ha="center", fontsize=15, fontweight="bold", color="#1f3f8f")
    ax.text(6.5, 5.32, "planeja · delega · estado compartilhado · valida contratos · escolhe o motor de cada agente "
                       "· registra log e motores usados", ha="center", fontsize=11.5, color="#1f3f8f")
    # trilha de noticias
    ax.text(0.25, 4.65, "Trilha de notícias (missões corpus e noticias)", fontsize=11.5, color=CINZA, style="italic")
    y = 3.35
    bloco(0.2, y, 1.9, 1.05, "ETL", "GNews · Guardian\n+ RSS (ampliação)", "#f3f4f6")
    bloco(2.45, y, 1.9, 1.05, "Desempate", "LLM local (Ollama)\ndiscordância total", "#fdf2e9")
    bloco(4.7, y, 1.9, 1.05, "Agente 1", "mBERT | Naive Bayes\nimpacto + entidades", "#fdf2e9")
    bloco(6.95, y, 1.9, 1.05, "Agente 2", "regras R1–R6 + k-NN\nsemântico | lexical", "#fdf2e9")
    seta(2.1, y + 0.52, 2.45, y + 0.52); seta(4.35, y + 0.52, 4.7, y + 0.52); seta(6.6, y + 0.52, 6.95, y + 0.52)
    # trilha de vendas
    ax.text(0.25, 2.75, "Trilha de vendas (missão semana)", fontsize=11.5, color=CINZA, style="italic")
    y2 = 1.45
    bloco(0.2, y2, 1.9, 1.05, "Base sintética", "gerador calibrado\n+ extensão por mês", "#eaf5ef")
    bloco(2.45, y2, 1.9, 1.05, "Pipeline semanal", "demanda · cota · KPI\nconsolidação mensal", "#eaf5ef")
    bloco(4.7, y2, 1.9, 1.05, "Agente 3", "XGBoost + SHAP\noportunidades/riscos", "#eaf5ef")
    seta(2.1, y2 + 0.52, 2.45, y2 + 0.52); seta(4.35, y2 + 0.52, 4.7, y2 + 0.52)
    # convergencia
    bloco(9.25, 2.4, 1.6, 1.05, "Agente 4", "cruza notícia ×\noportunidade", "#ede9fe")
    bloco(11.15, 2.4, 1.65, 1.05, "Agente 5", "recomendação\ntática", "#ede9fe")
    seta(8.85, y + 0.52, 9.6, 3.45); seta(6.6, y2 + 0.52, 9.6, 2.4); seta(10.85, 2.92, 11.15, 2.92)
    bloco(9.6, 0.25, 3.2, 0.95, "Painel Streamlit", "setor · indicadores · projeção · notícias", "#f3f4f6")
    seta(11.97, 2.4, 11.6, 1.2)
    for x in (1.15, 3.4, 5.65, 7.9, 10.05, 11.97):
        ax.plot([x, x], [5.1, 4.45 if x < 9 else 3.45], color="#1f6feb", lw=0.6, ls=":")
    _salvar(fig, "fig01_arquitetura_orquestrador.png")


def fig_anotacao(orq):
    a = orq["anotacao"]
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ordem = [("UNANIME", "Unânime (3 iguais)"), ("MAIORIA", "Maioria (2 de 3)"),
             ("DISCORDANCIA_TOTAL", "Discordância total")]
    vals = [a.get(k, 0) for k, _ in ordem]
    cores = [VERDE, AZUL, LARANJA]
    b = ax.barh([r for _, r in ordem], vals, color=cores)
    tot = sum(vals)
    for barra, v in zip(b, vals):
        ax.text(v + 8, barra.get_y() + barra.get_height() / 2, f"{v} ({v / tot:.1%})", va="center", fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, max(vals) * 1.25); ax.set_xlabel("notícias")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    _salvar(fig, "fig06_anotacao_concordancia.png")


def fig_matriz_confusao(orq):
    mc = orq["agente1"]["out_of_fold"]["matriz_confusao"]
    cl = ["Alto", "Medio", "Baixo"]
    m = np.array([[mc[r][c] for c in cl] for r in cl])
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    ax.imshow(m, cmap="Blues")
    for i in range(3):
        for j in range(3):
            pct = m[i, j] / m[i].sum()
            ax.text(j, i, f"{m[i, j]}\n{pct:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if pct > 0.5 else "#2b2b28")
    rot = ["Alto", "Médio", "Baixo"]
    ax.set_xticks(range(3)); ax.set_xticklabels(rot); ax.set_yticks(range(3)); ax.set_yticklabels(rot)
    ax.set_xlabel("previsto"); ax.set_ylabel("referência")
    _salvar(fig, "fig07_agente1_matriz_confusao_substituto.png")


def fig_agente2(orq):
    st = orq["agente2"]["status"]
    nomes = {"APROVADO": "Aprovado", "SEM_ANCORA_DE_MERCADO": "Sem âncora de mercado",
             "DESCARTADO_BAIXO_IMPACTO": "Descartado (baixo impacto)",
             "QUARENTENA_REVISAO_HUMANA": "Quarentena (revisão humana)", "DUPLICADO": "Duplicata"}
    ks = sorted(st, key=lambda k: -st[k])
    fig, ax = plt.subplots(figsize=(6.5, 3))
    cores = [VERDE if k == "APROVADO" else CINZA for k in ks]
    b = ax.barh([nomes.get(k, k) for k in ks], [st[k] for k in ks], color=cores)
    for barra, k in zip(b, ks):
        ax.text(st[k] + 5, barra.get_y() + barra.get_height() / 2, str(st[k]), va="center", fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel("notícias do corpus (predições out-of-fold)")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    _salvar(fig, "fig08_agente2_status.png")


def fig_agente3(a3):
    m = a3["modelo"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.4]})
    nomes = ["XGBoost", "Persistência", "Média 3 meses"]
    maes = [m["metricas_teste"]["mae"], m["baselines"]["persistencia"]["mae"], m["baselines"]["media_3m"]["mae"]]
    b = ax1.bar(nomes, maes, color=[AZUL, CINZA, CINZA])
    for barra, v in zip(b, maes):
        ax1.text(barra.get_x() + barra.get_width() / 2, v, f"{v:.5f}", ha="center", va="bottom", fontsize=8.5)
    ax1.set_ylabel("MAE do share t+1"); ax1.set_title(f"Ganho de {m['ganho_sobre_baseline_pct']:+.1f}% sobre a "
                                                     "melhor baseline", fontsize=9.5)
    _limpo(ax1)
    imp = m["importancia_global"][:8][::-1]
    ax2.barh([i["variavel"] for i in imp], [i["shap_medio_abs"] for i in imp], color=AZUL)
    ax2.set_xlabel("média de |SHAP|"); ax2.set_title("Importância global das variáveis", fontsize=9.5)
    for lado in ("top", "right"):
        ax2.spines[lado].set_visible(False)
    fig.tight_layout()
    _salvar(fig, "fig09_agente3_desempenho_shap.png")


def fig_agente45(a4, a5):
    cat = {"oportunidade": 0, "risco_medio_prazo": 0, "risco_imediato": 0}
    for s in a4["setores"]:
        for a in s["achados"]:
            cat[a["categoria"]] += 1
    urg = {"alta": 0, "media": 0, "baixa": 0}
    for s in a5["setores"]:
        for d in s["destaques"]:
            urg[d["recomendacao"]["urgencia"]] += 1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))
    ax1.bar(["Oportunidade", "Risco médio prazo", "Risco imediato"], list(cat.values()),
            color=[VERDE, LARANJA, VERMELHO])
    ax1.set_title("Agente 4 – achados por categoria", fontsize=9.5); _limpo(ax1)
    for i, v in enumerate(cat.values()):
        ax1.text(i, v, str(v), ha="center", va="bottom", fontsize=8.5)
    ax2.bar(["Alta", "Média", "Baixa"], list(urg.values()), color=[VERMELHO, LARANJA, CINZA])
    ax2.set_title("Agente 5 – urgência das recomendações", fontsize=9.5); _limpo(ax2)
    for i, v in enumerate(urg.values()):
        ax2.text(i, v, str(v), ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout()
    _salvar(fig, "fig10_agentes4_5.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setor", default="SP")
    a = ap.parse_args()
    print("Figuras do artigo:")
    fig_arquitetura()
    tela_painel(a.setor, "claro", "fig02_painel_tema_claro.png")
    tela_painel(a.setor, "escuro", "fig03_painel_tema_escuro.png")
    orq = _j("metricas_agentes_1_2_desempate.json")
    if orq:
        fig_anotacao(orq); fig_matriz_confusao(orq); fig_agente2(orq)
    a3, a4, a5 = _j("agente3_saida.json"), _j("agente4_saida.json"), _j("agente5_saida.json")
    if a3:
        fig_agente3(a3)
    if a4 and a5:
        fig_agente45(a4, a5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
