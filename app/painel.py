"""***************************************************************************************
SCRIPT    : painel.py
OBJETIVO  : Painel de insights por setor (Streamlit).

COMO RODAR (na raiz do projeto):
    pip install -r requirements.txt
    streamlit run app/painel.py

TELA:
    barra lateral   seletor do setor simulado
                    botao "Gerar nova semana"  (+ semana vigente logo abaixo)
                    botao "Gerar mais noticias"
    area principal  1. indicadores mes a mes do setor
                    2. projecao das semanas do mes  |  recomendacoes do Agente 3
                    3. as 4 noticias mais impactantes e as oportunidades que elas afetam

Tudo na area principal se refere ao setor escolhido na barra lateral.
A logica fica em src/painel/ (dados.py e acoes.py); este arquivo so desenha.
****************************************************************************************/
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.painel import acoes, dados as D  # noqa: E402

st.set_page_config(page_title="Painel do Setor · Inteligência Prescritiva",
                   page_icon="💊", layout="wide")

COR_PRIORIDADE = {"alta": "red", "media": "orange", "baixa": "gray"}
COR_IMPACTO = {"alto": "red", "medio": "orange", "baixo": "gray"}
ICONE_SENTIMENTO = {"positivo": "🟢 positivo", "neutro": "⚪ neutro", "negativo": "🔴 negativo"}
FORMATO_IND = {  # indicador -> (formato do eixo/valor, e percentual?)
    "Market share (próprios)": (".1%", True),
    "Cobertura da cota": (".0%", True),
    "CAGR médio": (".1%", True),
    "Coeficiente de variação": (".1%", True),
    "Índice sazonal": (".2f", False),
}
MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]

# ----------------------------------------------------------------------------
# Tema claro / escuro. O Streamlit nao troca o tema pelo codigo, entao o modo
# escuro e aplicado por CSS (fundo, textos, cartoes, campos) e os graficos
# recebem cores proprias - por isso todos passam por `grafico()`.
# ----------------------------------------------------------------------------
TEMAS = {
    "claro": {"fundo": "#ffffff", "lateral": "#f0f2f6", "texto": "#31333f", "suave": "#6b7280",
              "borda": "#e5e7eb", "grade": "#e5e7eb", "campo": "#ffffff"},
    "escuro": {"fundo": "#0e1117", "lateral": "#1a1d26", "texto": "#fafafa", "suave": "#a3a8b8",
               "borda": "#3a3f4b", "grade": "#2c313c", "campo": "#262a35"},
}

CSS_ESCURO = """
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stMain"] {{
    background-color: {fundo}; color: {texto}; }}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {{ background-color: {lateral}; }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp li, .stApp label, .stApp span,
[data-testid="stMarkdownContainer"], [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
[data-testid="stWidgetLabel"] {{ color: {texto}; }}
[data-testid="stCaptionContainer"], .stApp small {{ color: {suave} !important; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: {borda} !important; }}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, [data-baseweb="popover"] li {{
    background-color: {campo} !important; color: {texto} !important; }}
[data-testid="stExpander"] details, [data-testid="stExpander"] summary,
[data-testid="stStatusWidget"], .stCode, pre {{ background-color: {lateral} !important; color: {texto}; }}
.stButton > button[kind="secondary"] {{ background-color: {campo}; color: {texto}; border-color: {borda}; }}
hr {{ border-color: {borda}; }}
</style>
"""


def grafico(chart, escuro: bool, altura: int | None = None):
    """Aplica as cores do tema ao grafico altair e desenha."""
    t = TEMAS["escuro" if escuro else "claro"]
    if altura:
        chart = chart.properties(height=altura)
    chart = (chart.configure(background="transparent")
             .configure_axis(labelColor=t["suave"], titleColor=t["suave"],
                             gridColor=t["grade"], domainColor=t["borda"], tickColor=t["borda"])
             .configure_view(strokeWidth=0)
             .configure_legend(labelColor=t["texto"], titleColor=t["texto"]))
    st.altair_chart(chart, use_container_width=True, theme=None)


def mes_ano(ts) -> str:
    ts = pd.Timestamp(ts)
    return f"{MESES[ts.month - 1]}/{ts.year}"


# ============================================================================
# Dados (cache em memoria; limpo apos cada botao)
# ============================================================================
@st.cache_data(show_spinner="Carregando dimensões...")
def dimensoes():
    return D.carregar_dimensoes()


@st.cache_data(show_spinner="Agregando indicadores por setor (só na primeira vez)...")
def indicadores():
    return D.indicadores_mensais(dimensoes())


@st.cache_data(show_spinner="Agregando projeção semanal por setor (só na primeira vez)...")
def projecao():
    return D.projecao_semanal()


def limpar_cache():
    st.cache_data.clear()


try:
    dims = dimensoes()
    ind = indicadores()
    proj = projecao()
except FileNotFoundError as e:
    st.error(f"Arquivo de dados não encontrado: {e}.\n\n"
             "Gere a base antes de abrir o painel: `python scripts/gerar_base_inicial.py`, "
             "`python scripts/gerar_backfill_semanal.py` e `python scripts/gerar_kpi_final.py`.")
    st.stop()

vigente = D.semana_vigente(proj)

# area reservada no topo para o andamento dos botoes
topo = st.container()

# ============================================================================
# Barra lateral: setor simulado + botoes
# ============================================================================
with st.sidebar:
    cab, tog = st.columns([3, 2])
    cab.markdown("### 💊 Painel do setor")
    escuro = tog.toggle("🌙 Escuro", key="tema_escuro", help="Alterna entre o tema claro e o escuro")
    rotulos = dims.setores["rotulo"].tolist()
    escolha = st.selectbox("Simular setor", rotulos, key="setor",
                           help="Todos os gráficos, recomendações e notícias se referem a este setor.")
    setor = dims.setores.iloc[rotulos.index(escolha)]
    id_setor = int(setor["id_setor"])

    st.divider()
    clicou_semana = st.button("▶ Gerar nova semana", type="primary", use_container_width=True,
                              disabled=vigente is None)
    if vigente:
        st.markdown(f"**Semana vigente:** {MESES[vigente['mes'] - 1]}/{vigente['ano']} · "
                    f"semana {vigente['nr_semana']} de {vigente['total']}")
        st.caption(f"{vigente['inicio']:%d/%m} a {vigente['fim']:%d/%m/%Y} · próxima: "
                   f"semana {vigente['proxima']['nr_semana']} de "
                   f"{MESES[vigente['proxima']['mes'] - 1]}/{vigente['proxima']['ano']}")
    else:
        st.caption("Nenhuma semana calculada ainda.")

    st.write("")
    clicou_noticias = st.button("📰 Gerar mais notícias", type="primary", use_container_width=True)
    ult = D.ultima_execucao_noticias()
    if ult:
        st.caption(f"Última coleta: {pd.Timestamp(ult['executado_em']):%d/%m/%Y %H:%M} · "
                   f"{ult.get('noticias_validas', 0)} notícias de impacto alto/médio")
    else:
        st.caption("Notícias ainda não coletadas.")

    if "msg" in st.session_state:
        tipo, texto = st.session_state.pop("msg")
        (st.success if tipo == "ok" else st.error)(texto)


if escuro:
    st.markdown(CSS_ESCURO.format(**TEMAS["escuro"]), unsafe_allow_html=True)


def executar_acao(titulo: str, funcao, *args):
    with topo:
        with st.status(titulo, expanded=True) as status:
            linhas: list[str] = []
            caixa = st.empty()

            def log(l: str):
                linhas.append(l)
                caixa.code("\n".join(linhas[-25:]), language=None)

            ok, msg = funcao(*args, log)
            status.update(label=msg, state="complete" if ok else "error", expanded=not ok)
    if ok:
        limpar_cache()
        st.session_state["msg"] = ("ok", msg)
        st.rerun()
    else:
        st.session_state["msg"] = ("erro", msg)


if clicou_semana and vigente:
    executar_acao(f"Gerando a semana {vigente['proxima']['nr_semana']} de "
                  f"{MESES[vigente['proxima']['mes'] - 1]}/{vigente['proxima']['ano']}...",
                  acoes.gerar_nova_semana, vigente["proxima"])
if clicou_noticias:
    executar_acao("Coletando notícias e cruzando com as oportunidades...", acoes.gerar_mais_noticias)

# ============================================================================
# Cabecalho
# ============================================================================
recs, meta3 = D.recomendacoes_setor(id_setor, dims)
st.title(f"{setor['desc_uf']} · {setor['codigo_setor']}")
if meta3 and vigente:
    st.caption(f"Dados até {MESES[vigente['mes'] - 1]}/{vigente['ano']} (semana {vigente['nr_semana']}) · "
               f"recomendações para {mes_ano(meta3['periodo_previsto'] + '-01')} · "
               f"modelo do Agente 3: {meta3['algoritmo']}")
    for aviso in meta3["avisos"]:
        st.warning(aviso, icon="⚠️")

# ============================================================================
# 1. Indicadores mes a mes
# ============================================================================
c1, c2 = st.columns([3, 2])
c1.subheader("Indicadores mês a mês")
janela = c2.radio("Período", ["6 meses", "12 meses", "24 meses", "Tudo"], index=1,
                  horizontal=True, label_visibility="collapsed")
n_meses = {"6 meses": 6, "12 meses": 12, "24 meses": 24, "Tudo": None}[janela]
serie = D.serie_indicadores_setor(ind, id_setor, n_meses)

if serie.empty:
    st.info("Sem indicadores para este setor.")
else:
    cols = st.columns(len(D.INDICADORES))
    for col, nome in zip(cols, D.INDICADORES.values()):
        s = serie[serie["indicador"] == nome].sort_values("periodo_demanda")
        if s.empty:
            col.caption(nome)
            continue
        fmt, pct = FORMATO_IND[nome]
        atual = s["valor"].iloc[-1]
        ant = s["valor"].iloc[-2] if len(s) > 1 else None
        if pct:
            valor = f"{atual:.1%}"
            delta = f"{(atual - ant) * 100:+.1f} p.p." if ant is not None else None
        else:
            valor = f"{atual:.2f}"
            delta = f"{atual - ant:+.2f}" if ant is not None else None
        col.metric(nome, valor, delta, help=f"Último mês: {mes_ano(s['periodo_demanda'].iloc[-1])}; "
                                            "variação contra o mês anterior.")
        graf = (alt.Chart(s).mark_line(point=True, strokeWidth=2)
                .encode(x=alt.X("periodo_demanda:T", title=None,
                                axis=alt.Axis(format="%b/%y", labelAngle=0, tickCount=4)),
                        y=alt.Y("valor:Q", title=None, scale=alt.Scale(zero=False),
                                axis=alt.Axis(format=fmt, tickCount=4)),
                        tooltip=[alt.Tooltip("periodo_demanda:T", title="Mês", format="%m/%Y"),
                                 alt.Tooltip("valor:Q", title=nome, format=fmt)]))
        if nome == "Cobertura da cota":     # meta = 100% da cota
            graf = graf + alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(
                strokeDash=[4, 4], color="#d62728").encode(y="y:Q")
        with col:
            grafico(graf, escuro, 150)

st.divider()

# ============================================================================
# 2. Projecao das semanas | recomendacoes do setor
# ============================================================================
esq, dir_ = st.columns([3, 2], gap="large")

with esq:
    meses_disp = sorted(proj.loc[proj["id_setor"] == id_setor, "periodo_mensal"].unique())[-12:]
    a, b = st.columns([3, 2])
    a.subheader("Projeção das semanas do mês")
    mes_sel = b.selectbox("Mês", list(reversed(meses_disp)), format_func=mes_ano,
                          label_visibility="collapsed") if meses_disp else None
    if mes_sel is None:
        st.info("Sem semanas calculadas para este setor.")
    else:
        sem = D.semanas_do_mes_setor(proj, id_setor, pd.Timestamp(mes_sel))
        ultima = sem[sem["calculada"]].tail(1)
        if not ultima.empty:
            cob = float(ultima["cobertura_projetada"].iloc[0])
            k1, k2, k3 = st.columns(3)
            k1.metric("Cobertura projetada", f"{cob:.0%}", D.classificar_cobertura(cob),
                      delta_color="normal" if cob >= 0.80 else "inverse")
            k2.metric("Projeção de fechamento", f"{ultima['projetado'].iloc[0]:,.0f} un".replace(",", "."))
            k3.metric("Cota do mês", f"{ultima['cota'].iloc[0]:,.0f} un".replace(",", "."))

        dados_graf = sem[["rotulo", "acumulado", "projetado", "cota", "cobertura_projetada"]]
        base = alt.Chart(dados_graf).encode(x=alt.X("rotulo:N", title=None, sort=None,
                                             axis=alt.Axis(labelAngle=0)))
        barras = base.mark_bar(color="#4c78a8", opacity=0.85).encode(
            y=alt.Y("acumulado:Q", title="Unidades"),
            tooltip=[alt.Tooltip("rotulo:N", title="Semana"),
                     alt.Tooltip("acumulado:Q", title="Realizado acumulado", format=",.0f"),
                     alt.Tooltip("projetado:Q", title="Projeção de fechamento", format=",.0f"),
                     alt.Tooltip("cobertura_projetada:Q", title="Cobertura projetada", format=".0%")])
        linha = base.mark_line(color="#f58518", point=alt.OverlayMarkDef(size=70), strokeWidth=3).encode(
            y="projetado:Q")
        cota = alt.Chart(dados_graf.head(1)).mark_rule(color="#d62728", strokeDash=[6, 4], size=2).encode(
            y="cota:Q")
        rot_cota = alt.Chart(dados_graf.head(1)).mark_text(align="left", dx=4, dy=-8, color="#d62728").encode(
            y="cota:Q", x=alt.value(0), text=alt.value("cota do mês"))
        grafico(barras + linha + cota + rot_cota, escuro, 320)
        st.caption("Barras: realizado acumulado no mês · linha laranja: projeção de fechamento "
                   "(ritmo diário × dias do mês) · tracejado vermelho: cota.")

with dir_:
    st.subheader("Recomendações do setor")
    if meta3:
        o1, o2 = st.columns(2)
        o1.metric("Oportunidade prevista", f"{meta3['oportunidade_un']:,.0f} un".replace(",", "."))
        o2.metric("Risco previsto", f"{meta3['risco_un']:,.0f} un".replace(",", "."))
    if recs.empty:
        st.info("O Agente 3 não apontou oportunidades ou riscos relevantes para este setor.")
    else:
        for r in recs.head(5).to_dict("records"):
            with st.container(border=True):
                cor = COR_PRIORIDADE.get(r["Prioridade"], "gray")
                icone = "⚠️" if r["natureza"] == "risco" else "📈"
                un = f"{(r['Unidades em jogo'] or 0):,.0f}".replace(",", ".")
                st.markdown(f"{icone} **{r['Recomendação']}** · :{cor}[**{r['Prioridade'].upper()}**]  \n"
                            f"{r['Produto']} · {r['Local']} · {un} un em jogo")
                st.caption(r["Ação sugerida"])

st.divider()

# ============================================================================
# 3. Noticias mais impactantes x oportunidades
# ============================================================================
st.subheader("Notícias mais impactantes e as oportunidades que elas afetam")
sigla = str(setor["sigla_uf"])
destaques, meta5 = D.noticias_setor(id_setor, sigla, dims)
st.info(
    f"**Filtro de notícias por UF:** para o setor **{sigla}**, são exibidas apenas as notícias que não "
    f"citam nenhuma UF no título ou no conteúdo (alcance nacional) ou que citam a própria UF **{sigla}**. "
    "Notícias que citam somente outras UFs são ocultadas. O filtro atua apenas na exibição: a "
    "classificação e o cruzamento feitos pelos agentes não são alterados.",
    icon="🔎",
)
if meta5 and meta5.get("ocultadas"):
    outras = ", ".join(meta5.get("ufs_ocultadas") or [])
    st.caption(f"{meta5['ocultadas']} notícia(s) do ranking do setor ocultada(s) por citar outra UF ({outras})"
               + (f"; {meta5['complementadas']} substituída(s) pela(s) próxima(s) do ranking do Agente 4."
                  if meta5.get("complementadas") else "."))
if not destaques:
    if meta5 and recs.empty:
        st.info("O Agente 3 não encontrou oportunidades relevantes neste setor (volumes pequenos), "
                "então não há o que cruzar com as notícias.")
    else:
        st.info("Ainda não há notícias cruzadas para este setor. Clique em **Gerar mais notícias**.")
else:
    for linha in (destaques[:2], destaques[2:4]):
        cols = st.columns(2, gap="large")
        for col, d in zip(cols, linha):
            n, rec = d["noticia"], d["recomendacao"]
            with col.container(border=True):
                data = pd.to_datetime(n.get("data_publicacao"), errors="coerce")
                st.markdown(f"**{d['posicao']}. [{n['titulo']}]({n['url']})**")
                st.caption(f"{n.get('fonte') or 'fonte não informada'}"
                           f"{' · ' + data.strftime('%d/%m/%Y') if pd.notna(data) else ''} · "
                           f"escopo: {n['escopo']}")
                cor = COR_IMPACTO.get(n["nivel_impacto"], "gray")
                st.markdown(f":{cor}[**Impacto {n['nivel_impacto'].upper()}**] · "
                            f"{ICONE_SENTIMENTO.get(n['sentimento'], n['sentimento'])}")
                st.markdown("**Oportunidades afetadas**")
                for o in d["oportunidades"]:
                    icone = "⚠️" if o["natureza"] == "risco" else "📈"
                    un = f"{(o['valor_em_jogo_un'] or 0):,.0f}".replace(",", ".")
                    st.markdown(f"- {icone} {o['produto']} · {o['local']} — "
                                f"{o['tipo_recomendacao'].replace('_', ' ').lower()}, {un} un")
                urg = COR_PRIORIDADE.get(rec["urgencia"], "gray")
                rotulo_rec = "Recomendação" if d.get("origem") != "agente4" else "Ação do Agente 3"
                st.markdown(f"**{rotulo_rec}** · :{urg}[urgência {rec['urgencia']}]  \n{rec['acao_sugerida']}")
                if d.get("origem") == "agente4":
                    st.caption("Notícia incluída após o filtro de UF, a partir do ranking do Agente 4 "
                               "(sem texto do Agente 5).")
                with st.expander("Por quê?"):
                    st.write(rec["justificativa"])
                    st.caption(f"Impacto esperado: {rec['impacto_esperado']} · "
                               f"classificação da notícia: {n.get('origem_classificacao', '—')}")
