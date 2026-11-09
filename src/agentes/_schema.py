"""
NORMALIZACAO DE SCHEMA E PREPARO DE TEXTO
==========================================
Camada de defesa entre a planilha bruta e os agentes.

MOTIVACAO: hoje o Agente 1 quebra com KeyError se a planilha tiver
"Titulo" em vez de "titulo", ou "Avaliador1" em vez de "Avaliador 1".
Corpus montado a partir de API de noticias + anotacao manual em Excel
quase nunca tem cabecalho estavel. Centralizar a normalizacao aqui
significa que os agentes assumem um contrato fixo de colunas e nao
precisam mais tratar variacao - Principio Aberto/Fechado (SOLID-O):
para suportar um novo nome de coluna, adiciona-se um alias neste
modulo, sem tocar nos agentes.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

import pandas as pd

# ---------------------------------------------------------------------
# Contrato de colunas que os agentes esperam
# ---------------------------------------------------------------------
COL_ID = "doc_id"
COL_TITULO = "titulo"
COL_DESCRICAO = "descricao"
COL_CONTEUDO = "conteudo"
COL_TEXTO = "texto_completo"
COL_EXTRACAO = "texto_extracao"
COL_LABEL = "classificacao_final"
COLS_AVALIADORES = ["Avaliador 1", "Avaliador 2", "Avaliador 3"]

# Aliases aceitos por coluna canonica. A chave e o nome canonico; os
# valores sao variantes ja normalizadas (minusculas, sem acento).
_ALIASES: dict[str, set[str]] = {
    COL_TITULO: {"titulo", "title", "headline", "manchete", "titulo_noticia", "nome"},
    COL_DESCRICAO: {"descricao", "description", "resumo", "summary", "subtitulo", "lide", "lead"},
    COL_CONTEUDO: {"conteudo", "content", "texto", "corpo", "materia", "body", "noticia", "full_text"},
    COL_TEXTO: {"texto_completo", "textocompleto", "full_text", "texto_final"},
    COL_LABEL: {
        "classificacao_final", "classificacaofinal", "classificacao", "label", "rotulo",
        "impacto", "classe_final", "gabarito", "consenso", "classe", "y",
    },
    "url": {"url", "link", "fonte_url", "endereco"},
    "data": {"data", "date", "published_at", "data_publicacao", "publishedat", "datapublicacao"},
    "fonte": {"fonte", "source", "veiculo", "portal"},
}

# Variantes de nome dos avaliadores (Avaliador 1, avaliador_1, av1, ...)
_RE_AVALIADOR = re.compile(r"^(?:avaliador|aval|av|anotador|rater|juiz)[\s_\-]*([123])$")

# Mapeamento de rotulos para a forma canonica. Corpus anotado a mao
# costuma ter "ALTO", "alto", "Alto impacto", "3", "A" para a mesma classe.
_MAPA_ROTULOS: dict[str, str] = {
    "alto": "Alto", "alta": "Alto", "a": "Alto", "3": "Alto", "high": "Alto",
    "alto impacto": "Alto", "impacto alto": "Alto",
    "medio": "Medio", "media": "Medio", "m": "Medio", "2": "Medio", "medium": "Medio",
    "medio impacto": "Medio", "impacto medio": "Medio", "moderado": "Medio",
    "baixo": "Baixo", "baixa": "Baixo", "b": "Baixo", "1": "Baixo", "low": "Baixo",
    "baixo impacto": "Baixo", "impacto baixo": "Baixo", "nenhum": "Baixo", "0": "Baixo",
}

ROTULOS_CANONICOS = ("Alto", "Medio", "Baixo")


def _slug(texto: str) -> str:
    """Reduz um nome de coluna a sua forma comparavel."""
    texto = unicodedata.normalize("NFKD", str(texto))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "_", texto.lower()).strip("_")


def normalizar_rotulo(valor) -> str | None:
    """
    Converte um rotulo bruto para {Alto, Medio, Baixo}.

    Retorna None quando o valor e nulo ou nao reconhecido - deliberado:
    e melhor o agente ver um None explicito e descartar a linha do que
    treinar sobre um rotulo mal interpretado. Rotulo silenciosamente
    errado contamina a metrica reportada no TCC.
    """
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return None
    chave = _slug(valor).replace("_", " ").strip()
    return _MAPA_ROTULOS.get(chave)


def detectar_colunas(df: pd.DataFrame) -> dict[str, str]:
    """
    Mapeia nomes reais -> nomes canonicos, sem alterar o DataFrame.

    Retorna {nome_canonico: nome_original}, util para log e auditoria
    (o TCC pode citar exatamente como o schema foi interpretado).
    """
    encontrado: dict[str, str] = {}
    for col in df.columns:
        s = _slug(col)

        m = _RE_AVALIADOR.match(s)
        if m:
            encontrado[f"Avaliador {m.group(1)}"] = col
            continue

        for canonico, aliases in _ALIASES.items():
            if s in aliases and canonico not in encontrado:
                encontrado[canonico] = col
                break
    return encontrado


def montar_texto_completo(
    df: pd.DataFrame,
    mapa: dict[str, str],
    max_caracteres: int = 1500,
) -> pd.Series:
    """
    Concatena titulo + descricao + conteudo com TRUNCAMENTO PRIORIZADO.

    Por que nao concatenar tudo e deixar o tokenizer truncar: o mBERT
    aceita no maximo 512 tokens (~350 palavras em portugues). Uma
    materia completa passa disso com folga, entao o `truncation=True`
    corta o FIM do texto. Se o conteudo vier primeiro na concatenacao,
    o titulo - que e a parte mais densa em sinal - pode ser o pedaco
    descartado.

    Aqui a ordem garante que titulo e descricao sempre sobrevivem, e
    apenas o corpo da materia e cortado. Ganho tipico de F1 observado
    em tarefas de classificacao de noticia: 2 a 5 pontos, so por causa
    da ordem (SUN et al., 2019, "How to Fine-Tune BERT for Text
    Classification").
    """
    def _col(nome: str) -> pd.Series:
        original = mapa.get(nome)
        if original is None or original not in df.columns:
            return pd.Series([""] * len(df), index=df.index)
        return df[original].fillna("").astype(str).str.strip()

    titulo, descricao, conteudo = _col(COL_TITULO), _col(COL_DESCRICAO), _col(COL_CONTEUDO)

    def _juntar(t: str, d: str, c: str) -> str:
        cabecalho = ". ".join(p for p in (t, d) if p)
        folga = max_caracteres - len(cabecalho)
        if folga <= 0:
            return cabecalho[:max_caracteres]
        return (cabecalho + ". " + c[:folga]).strip(". ").strip() if c else cabecalho

    return pd.Series(
        [_juntar(t, d, c) for t, d, c in zip(titulo, descricao, conteudo)],
        index=df.index,
    )


def gerar_doc_id(df: pd.DataFrame, mapa: dict[str, str]) -> pd.Series:
    """
    Cria um identificador estavel por documento (hash do titulo+url).

    ESSENCIAL para o Agente 2: sem um id, nao ha como o validador saber
    que uma noticia esta se comparando consigo mesma no indice de
    eventos historicos - que e a origem do vazamento de dados corrigido
    naquele agente. Hash em vez de indice posicional porque o indice
    muda quando o DataFrame e filtrado ou reordenado.
    """
    titulo = df[mapa[COL_TITULO]].fillna("").astype(str) if COL_TITULO in mapa else pd.Series([""] * len(df), index=df.index)
    url = df[mapa["url"]].fillna("").astype(str) if "url" in mapa else pd.Series([""] * len(df), index=df.index)
    base = (titulo + "|" + url).where(lambda s: s.str.strip() != "|", other=pd.Series(df.index.astype(str), index=df.index))
    return base.map(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()[:16])


def consolidar_avaliadores(
    df: pd.DataFrame,
    coluna_saida: str = COL_LABEL,
    marcar_discordancia: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Consolida os rotulos dos tres avaliadores por MAIORIA SIMPLES.

    PECA QUE FALTAVA NO PROJETO: o Agente de Desempate (secao 3.5) so
    atua nos casos de discordancia TOTAL - quando nenhum par de
    avaliadores concorda. Ele nao resolve, e nem deveria resolver, os
    casos de maioria (2 de 3 concordam), que sao a grande maioria do
    corpus. Nao havia nada no pacote que calculasse esse voto, entao um
    corpus com apenas 'Avaliador 1/2/3' e sem coluna consolidada nao
    tinha rotulo de treino e o pipeline parava na primeira etapa.

    Divisao de responsabilidade:
      - maioria simples (2 ou 3 iguais) -> resolvida aqui, deterministica
      - discordancia total (3 diferentes) -> deixada como nula, para o
        Agente de Desempate resolver lendo o conteudo integral

    Deixar o caso dificil explicitamente nulo, em vez de escolher um
    rotulo qualquer, e o que garante que o Agente de Desempate receba
    exatamente os casos que existem para ele - e que o TCC possa
    reportar quantos foram.

    Retorna (df_com_coluna_consolidada, resumo_das_contagens).
    """
    faltando = [c for c in COLS_AVALIADORES if c not in df.columns]
    if faltando:
        raise KeyError(f"Colunas de avaliador ausentes: {faltando}")

    out = df.copy()
    votos = out[COLS_AVALIADORES]

    consolidado, situacao = [], []
    for _, linha in votos.iterrows():
        validos = [v for v in linha.tolist() if isinstance(v, str) and v]
        if not validos:
            consolidado.append(None)
            situacao.append("SEM_AVALIACAO")
            continue

        contagem: dict[str, int] = {}
        for v in validos:
            contagem[v] = contagem.get(v, 0) + 1
        rotulo, n = max(contagem.items(), key=lambda kv: kv[1])

        if n >= 2:
            consolidado.append(rotulo)
            situacao.append("UNANIME" if n == len(validos) else "MAIORIA")
        elif len(validos) == 1:
            # Um unico avaliador respondeu: nao ha maioria a apurar, mas
            # tambem nao ha discordancia. Aceita-se o voto, marcado.
            consolidado.append(validos[0])
            situacao.append("AVALIADOR_UNICO")
        else:
            consolidado.append(None)
            situacao.append("DISCORDANCIA_TOTAL")

    out[coluna_saida] = consolidado
    if marcar_discordancia:
        out["situacao_anotacao"] = situacao

    resumo: dict[str, int] = {}
    for s in situacao:
        resumo[s] = resumo.get(s, 0) + 1
    return out, resumo


def preparar_corpus(
    df: pd.DataFrame,
    max_caracteres: int = 1500,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Ponto de entrada unico: recebe a planilha crua e devolve um
    DataFrame que respeita o contrato de colunas dos agentes.

    Garante a presenca de: doc_id, texto_completo e (quando houver
    rotulo) classificacao_final ja canonizada. Nao remove linhas -
    quem decide o que descartar sao os agentes, para que a contagem de
    descarte seja auditavel e reportavel no TCC.
    """
    mapa = detectar_colunas(df)
    out = df.copy()

    if COL_TEXTO in mapa and out[mapa[COL_TEXTO]].fillna("").astype(str).str.strip().ne("").any():
        out[COL_TEXTO] = out[mapa[COL_TEXTO]].fillna("").astype(str)
    else:
        out[COL_TEXTO] = montar_texto_completo(out, mapa, max_caracteres=max_caracteres)

    # Texto para EXTRACAO DE ENTIDADES - sem o truncamento agressivo.
    #
    # O corte em `max_caracteres` existe por causa do limite de tokens do
    # mBERT, nao por causa do gazetteer: casar regex custa ~0,3 ms por
    # documento e nao tem limite de tamanho. Rodar a extracao sobre o
    # texto ja truncado descartava entidades que aparecem no meio da
    # materia - medido no corpus real, 273 documentos perdiam a
    # classificacao de escopo so por isso.
    #
    # Duas colunas, dois propositos: `texto_completo` alimenta o modelo
    # neural, `texto_extracao` alimenta o dicionario de dominio.
    out[COL_EXTRACAO] = montar_texto_completo(out, mapa, max_caracteres=30000)
    out[COL_ID] = gerar_doc_id(out, mapa)

    if COL_LABEL in mapa:
        out[COL_LABEL] = out[mapa[COL_LABEL]].map(normalizar_rotulo)

    for i in (1, 2, 3):
        canonico = f"Avaliador {i}"
        if canonico in mapa:
            out[canonico] = out[mapa[canonico]].map(normalizar_rotulo)

    # Consolidacao automatica quando a planilha traz os tres avaliadores
    # mas nao a coluna consolidada. Sem isso o pipeline nao tem rotulo
    # de treino e para logo na primeira etapa.
    tem_avaliadores = all(f"Avaliador {i}" in out.columns for i in (1, 2, 3))
    precisa_consolidar = COL_LABEL not in out.columns or out[COL_LABEL].isna().all()
    if tem_avaliadores and precisa_consolidar:
        out, resumo = consolidar_avaliadores(out)
        if verbose:
            print(f"[schema] classificacao_final ausente -> consolidada por maioria simples: {resumo}")

    if verbose:
        print("[schema] Colunas reconhecidas:")
        for canonico, original in sorted(mapa.items()):
            print(f"         {canonico:<22} <- '{original}'")
        faltando = [c for c in (COL_TITULO, COL_DESCRICAO, COL_CONTEUDO) if c not in mapa]
        if faltando:
            print(f"[schema] AVISO: colunas de texto ausentes: {faltando}")
        vazios = int((out[COL_TEXTO].str.strip() == "").sum())
        if vazios:
            print(f"[schema] AVISO: {vazios} documento(s) com texto vazio.")
        if COL_LABEL in mapa:
            nao_mapeados = int(out[COL_LABEL].isna().sum())
            dist = out[COL_LABEL].value_counts().to_dict()
            print(f"[schema] Distribuicao de rotulos: {dist}")
            if nao_mapeados:
                print(f"[schema] AVISO: {nao_mapeados} rotulo(s) nao reconhecido(s) -> None.")

    return out, mapa
