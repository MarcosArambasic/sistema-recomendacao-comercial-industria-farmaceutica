"""
RELATORIO DE AVALIACAO DO AGENTE 1
====================================
Artefatos de avaliacao gerados automaticamente ao fim de cada treino do
Agente 1, em resposta a orientacao recebida: relatorio de acuracia ao
final de cada ciclo, avaliacao POR CLASSE e matriz de confusao do teste
final.

--------------------------------------------------------------------
DE ONDE VEM CADA NUMERO (e por que isso importa)
--------------------------------------------------------------------
Todas as tabelas sao calculadas sobre as predicoes OUT-OF-FOLD: cada
documento do corpus foi classificado exatamente uma vez, pelo modelo do
fold em que ele estava ESCONDIDO. Nenhuma linha avaliada aqui foi vista
em treino pelo modelo que a classificou.

A distincao nao e academica. Avaliar o modelo final - treinado sobre
100% do corpus - no proprio corpus produz metrica inflada, porque o
modelo ja decorou boa parte daquelas linhas. No corpus deste projeto a
diferenca medida foi de 0,66 (out-of-fold) para 0,79 (dentro da
amostra): treze pontos de ilusao. E o erro mais comum em TCC de
aprendizado de maquina e o primeiro lugar onde uma banca vai cutucar.

--------------------------------------------------------------------
"CICLO" EM DOIS NIVEIS
--------------------------------------------------------------------
O termo e ambiguo, entao o relatorio cobre os dois:
  - CICLO = FOLD: `metricas_por_fold.csv`, uma linha por particao, com
    acuracia, F1-macro e F1 de cada classe, mais a linha de media +/-
    desvio. E a tabela de resultado do TCC.
  - CICLO = EPOCA: `curva_treino.csv` e `curva_treino.png`, com perda de
    treino, perda de validacao e F1 de validacao a cada epoca, em todos
    os folds. E o grafico que evidencia o sobreajuste - perda de treino
    caindo enquanto a de validacao sobe - e sustenta empiricamente o
    argumento de corpus insuficiente.

--------------------------------------------------------------------
DECISOES DE APRESENTACAO DAS FIGURAS
--------------------------------------------------------------------
- Matriz de confusao em RAMPA DE UMA COR SO (azul claro -> escuro). Mapa
  de cores tipo arco-iris nao tem ordem perceptual: o leitor nao sabe se
  verde e mais ou menos que laranja. Com uma unica matiz, mais escuro e
  sempre mais.
- Cada celula traz a contagem absoluta E o percentual da classe real, de
  modo que a figura continua legivel impressa em preto e branco - caso
  real de um TCC encadernado.
- Curva de treino em DOIS PAINEIS (perda e F1), nunca dois eixos y no
  mesmo grafico: eixo duplo permite sugerir qualquer relacao entre as
  series so escolhendo as escalas.
- Texto das figuras com acentuacao correta, porque elas vao impressas no
  trabalho; o codigo e as mensagens de console seguem o padrao sem
  acento do projeto.

Dependencia opcional: matplotlib. Sem ele, as tabelas (.csv) saem
normalmente e apenas as figuras sao puladas, com aviso.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

# --- Paleta -----------------------------------------------------------
# Matizes conferidos para daltonismo (separacao deuteranopia dE 9,2) e
# para contraste sobre fundo claro.
AZUL = "#2a78d6"       # perda de treino
LARANJA = "#eb6834"    # perda de validacao
VERDE = "#1baf7a"      # F1-macro de validacao
TINTA = "#2b2b28"      # texto principal
TINTA_FRACA = "#6b6b66"  # texto secundario, grade
RAMPA_AZUL = [
    "#ffffff", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
    "#104281", "#0d366b",
]

# Rotulos acentuados APENAS nas figuras: elas vao impressas no TCC. O
# codigo, os CSV e o console seguem o padrao sem acento do projeto.
ROTULO_IMPRESSO = {"Alto": "Alto", "Medio": "Médio", "Baixo": "Baixo"}

ARQUIVOS = {
    "metricas_por_fold": "metricas_por_fold.csv",
    "relatorio_por_classe": "relatorio_por_classe.csv",
    "matriz_confusao": "matriz_confusao.csv",
    "matriz_confusao_percentual": "matriz_confusao_percentual.csv",
    "curva_treino": "curva_treino.csv",
    "figura_matriz": "matriz_confusao.png",
    "figura_curva": "curva_treino.png",
}


# =====================================================================
# Tabelas
# =====================================================================
def tabela_por_fold(metricas_folds: list[dict], classes: list[str]) -> pd.DataFrame:
    """
    Uma linha por fold com acuracia, F1-macro e F1 de cada classe, mais
    as linhas de media e desvio padrao.

    O desvio entre folds e o que responde a pergunta "esse resultado nao
    foi sorte da particao?". Reportar a media sem ele e reportar metade
    da informacao.
    """
    df = pd.DataFrame(metricas_folds)
    colunas = ["fold", "n_treino", "n_validacao", "acuracia", "f1_macro", "f1_weighted"]
    colunas += [f"f1_{c.lower()}" for c in classes]
    df = df[[c for c in colunas if c in df.columns]]

    numericas = df.columns.drop("fold")
    resumo = pd.DataFrame(
        [
            {"fold": "media", **df[numericas].mean().to_dict()},
            {"fold": "desvio", **df[numericas].std(ddof=0).to_dict()},
        ]
    )
    if {"n_treino", "n_validacao"} <= set(resumo.columns):
        resumo.loc[1, ["n_treino", "n_validacao"]] = np.nan   # desvio de n nao significa nada
    saida = pd.concat([df, resumo], ignore_index=True).round(4)
    for coluna in ("n_treino", "n_validacao"):
        if coluna in saida.columns:
            saida[coluna] = saida[coluna].round().astype("Int64")
    return saida


def tabela_por_classe(y: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> pd.DataFrame:
    """
    Precisao, revocacao, F1 e suporte de cada classe, nas predicoes
    out-of-fold, com as medias macro e ponderada ao final.

    E a tabela que mostra ONDE o modelo erra. A media esconde que uma
    classe especifica pode estar sendo ignorada - exatamente o que o
    desbalanceio do corpus favorece.
    """
    idx = np.arange(len(classes))
    p, r, f1, sup = precision_recall_fscore_support(y, y_pred, labels=idx, zero_division=0)
    linhas = [
        {"classe": c, "precisao": p[i], "revocacao": r[i], "f1": f1[i], "suporte": int(sup[i])}
        for i, c in enumerate(classes)
    ]
    pm, rm, fm, _ = precision_recall_fscore_support(y, y_pred, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y, y_pred, average="weighted", zero_division=0)
    linhas += [
        {"classe": "media_macro", "precisao": pm, "revocacao": rm, "f1": fm, "suporte": int(sup.sum())},
        {"classe": "media_ponderada", "precisao": pw, "revocacao": rw, "f1": fw, "suporte": int(sup.sum())},
        {"classe": "acuracia", "precisao": np.nan, "revocacao": np.nan,
         "f1": accuracy_score(y, y_pred), "suporte": int(sup.sum())},
        {"classe": "kappa_cohen", "precisao": np.nan, "revocacao": np.nan,
         "f1": cohen_kappa_score(y, y_pred), "suporte": int(sup.sum())},
    ]
    return pd.DataFrame(linhas).round(4)


def tabelas_confusao(
    y: np.ndarray, y_pred: np.ndarray, classes: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Matriz de confusao em contagem e em percentual POR LINHA.

    Normalizar por linha (sobre a classe real) e o que torna a figura
    legivel num corpus desbalanceado: sem isso, a linha da classe mais
    numerosa domina a escala de cor e as demais somem.
    """
    cm = confusion_matrix(y, y_pred, labels=np.arange(len(classes)))
    linhas = [f"real: {c}" for c in classes]
    colunas = [f"previsto: {c}" for c in classes]
    contagem = pd.DataFrame(cm, index=linhas, columns=colunas)
    total = cm.sum(axis=1, keepdims=True)
    total[total == 0] = 1
    percentual = pd.DataFrame((cm / total * 100).round(2), index=linhas, columns=colunas)
    return contagem, percentual


def tabela_por_epoca(historicos: list[tuple[int, list[dict]]]) -> pd.DataFrame:
    """
    Reconstroi a curva de treino a partir do `log_history` do Trainer.

    O Trainer registra a perda de treino a cada `logging_steps` (varias
    vezes por epoca) e as metricas de validacao uma vez por epoca. Aqui
    a perda de treino e agregada por epoca e casada com a avaliacao
    correspondente, para que as duas series fiquem na mesma escala de x.
    """
    linhas = []
    for fold, historico in historicos:
        treino: dict[int, list[float]] = {}
        avaliacao: dict[int, dict] = {}
        for registro in historico:
            epoca = registro.get("epoch")
            if epoca is None:
                continue
            if "loss" in registro:
                treino.setdefault(int(np.ceil(epoca)), []).append(float(registro["loss"]))
            if "eval_loss" in registro:
                avaliacao[int(round(epoca))] = registro
        for epoca in sorted(set(treino) | set(avaliacao)):
            registro = avaliacao.get(epoca, {})
            linhas.append({
                "fold": fold,
                "epoca": epoca,
                "perda_treino": float(np.mean(treino[epoca])) if epoca in treino else np.nan,
                "perda_validacao": registro.get("eval_loss", np.nan),
                "f1_macro_validacao": registro.get("eval_f1_macro", np.nan),
                "acuracia_validacao": registro.get("eval_acuracia", np.nan),
            })
    return pd.DataFrame(linhas).round(4)


# =====================================================================
# Figuras
# =====================================================================
def _matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")           # sem interface grafica: serve no Colab e em servidor
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("[Agente 1] matplotlib ausente - figuras puladas (as tabelas .csv foram geradas). "
              "Para gerar: pip install matplotlib")
        return None


def _limpar_eixo(ax, plt) -> None:
    """Remove o excesso: so grade horizontal fraca e dois eixos."""
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(TINTA_FRACA)
        ax.spines[lado].set_linewidth(0.8)
    ax.tick_params(colors=TINTA_FRACA, labelsize=9, length=3)
    ax.grid(axis="y", color=TINTA_FRACA, alpha=0.18, linewidth=0.7)
    ax.set_axisbelow(True)


def figura_matriz_confusao(
    contagem: pd.DataFrame, percentual: pd.DataFrame, classes: list[str], caminho: str | Path
) -> str | None:
    """Mapa de calor em uma unica matiz, com contagem e percentual em cada celula."""
    plt = _matplotlib()
    if plt is None:
        return None
    from matplotlib.colors import LinearSegmentedColormap

    rampa = LinearSegmentedColormap.from_list("azul_tcc", RAMPA_AZUL)
    pct = percentual.to_numpy(dtype=float)
    cnt = contagem.to_numpy(dtype=int)
    n = len(classes)

    fig, ax = plt.subplots(figsize=(1.6 * n + 2.2, 1.35 * n + 1.8), dpi=200)
    im = ax.imshow(pct, cmap=rampa, vmin=0, vmax=100, aspect="equal")

    nomes = [ROTULO_IMPRESSO.get(c, c) for c in classes]
    ax.set_xticks(range(n), nomes)
    ax.set_yticks(range(n), nomes)
    ax.set_xlabel("Classe prevista pelo modelo", color=TINTA, fontsize=10, labelpad=8)
    ax.set_ylabel("Classe real (anotação humana)", color=TINTA, fontsize=10, labelpad=8)
    ax.set_title("Matriz de confusão — predições fora da amostra",
                 color=TINTA, fontsize=11.5, pad=12, loc="left")
    ax.tick_params(colors=TINTA, labelsize=9.5, length=0)
    for lado in ("top", "right", "bottom", "left"):
        ax.spines[lado].set_visible(False)

    # Separador de 2px entre celulas: a figura e uma grade, nao um bloco.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    for i in range(n):
        for j in range(n):
            # Texto claro sobre celula escura, escuro sobre clara: o
            # numero precisa continuar legivel em toda a rampa.
            cor = "white" if pct[i, j] > 55 else TINTA
            ax.text(j, i - 0.09, f"{cnt[i, j]}", ha="center", va="center",
                    color=cor, fontsize=13, fontweight="bold")
            ax.text(j, i + 0.22, f"{pct[i, j]:.1f}%", ha="center", va="center",
                    color=cor, fontsize=9.5, alpha=0.85)

    barra = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    barra.set_label("% da classe real", color=TINTA_FRACA, fontsize=9)
    barra.ax.tick_params(colors=TINTA_FRACA, labelsize=8.5, length=2)
    barra.outline.set_visible(False)

    fig.text(0.0, -0.02,
             "Diagonal = acertos. Cada linha soma 100% da classe real.",
             color=TINTA_FRACA, fontsize=8.5, ha="left")
    fig.tight_layout()
    fig.savefig(caminho, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(caminho)


def figura_curva_treino(curva: pd.DataFrame, caminho: str | Path) -> str | None:
    """
    Dois paineis empilhados: perda (treino x validacao) e F1-macro de
    validacao, por epoca. Linhas finas = folds individuais; linha grossa
    = media entre folds.
    """
    plt = _matplotlib()
    if plt is None or curva.empty:
        return None

    media = curva.groupby("epoca").mean(numeric_only=True).reset_index()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 6.4), dpi=200, sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.18},
    )

    for _, g in curva.groupby("fold"):
        ax1.plot(g["epoca"], g["perda_treino"], color=AZUL, alpha=0.22, linewidth=1.1)
        ax1.plot(g["epoca"], g["perda_validacao"], color=LARANJA, alpha=0.22, linewidth=1.1)
        ax2.plot(g["epoca"], g["f1_macro_validacao"], color=VERDE, alpha=0.22, linewidth=1.1)

    ax1.plot(media["epoca"], media["perda_treino"], color=AZUL, linewidth=2,
             marker="o", markersize=4.5, label="Perda de treino")
    ax1.plot(media["epoca"], media["perda_validacao"], color=LARANJA, linewidth=2,
             marker="o", markersize=4.5, label="Perda de validação")
    ax2.plot(media["epoca"], media["f1_macro_validacao"], color=VERDE, linewidth=2,
             marker="o", markersize=4.5)

    _limpar_eixo(ax1, plt)
    _limpar_eixo(ax2, plt)
    ax1.set_ylabel("Perda", color=TINTA, fontsize=10)
    ax2.set_ylabel("F1-macro (validação)", color=TINTA, fontsize=10)
    ax2.set_xlabel("Época", color=TINTA, fontsize=10)
    ax1.set_title("Curva de treino por época — média dos folds (linhas finas: cada fold)",
                  color=TINTA, fontsize=11.5, pad=28, loc="left")
    ax2.margins(x=0.06)
    legenda = ax1.legend(frameon=False, fontsize=9.5, ncol=2,
                         loc="lower left", bbox_to_anchor=(0.0, 1.0))
    for texto in legenda.get_texts():
        texto.set_color(TINTA)
    ax1.set_xticks(sorted(curva["epoca"].dropna().unique()))

    # Rotulo direto no ultimo ponto: a serie do painel inferior nao tem
    # legenda (serie unica) e a cor sozinha nao pode carregar a leitura.
    if media["f1_macro_validacao"].notna().any():
        ultimo = media.dropna(subset=["f1_macro_validacao"]).iloc[-1]
        ax2.annotate(f"{ultimo['f1_macro_validacao']:.3f}",
                     xy=(ultimo["epoca"], ultimo["f1_macro_validacao"]),
                     xytext=(6, 0), textcoords="offset points",
                     color=TINTA, fontsize=9.5, va="center", fontweight="bold")

    fig.text(0.0, -0.01,
             "Perda de treino caindo enquanto a de validação sobe indica sobreajuste.",
             color=TINTA_FRACA, fontsize=8.5, ha="left")
    fig.savefig(caminho, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(caminho)


# =====================================================================
# Ponto de entrada
# =====================================================================
def gerar(
    destino: str | Path,
    y: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    metricas_folds: list[dict],
    historicos: list[tuple[int, list[dict]]] | None = None,
    imprimir: bool = True,
) -> dict:
    """
    Gera todas as tabelas e figuras em `destino` e devolve o dicionario
    de caminhos, para registro no relatorio de treino (JSON).
    """
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    caminhos: dict[str, str] = {}

    por_fold = tabela_por_fold(metricas_folds, classes)
    por_classe = tabela_por_classe(y, y_pred, classes)
    contagem, percentual = tabelas_confusao(y, y_pred, classes)

    for chave, tabela, indice in (
        ("metricas_por_fold", por_fold, False),
        ("relatorio_por_classe", por_classe, False),
        ("matriz_confusao", contagem, True),
        ("matriz_confusao_percentual", percentual, True),
    ):
        caminho = destino / ARQUIVOS[chave]
        tabela.to_csv(caminho, index=indice, encoding="utf-8-sig")
        caminhos[chave] = str(caminho)

    curva = tabela_por_epoca(historicos or [])
    if not curva.empty:
        caminho = destino / ARQUIVOS["curva_treino"]
        curva.to_csv(caminho, index=False, encoding="utf-8-sig")
        caminhos["curva_treino"] = str(caminho)

    figura = figura_matriz_confusao(contagem, percentual, classes, destino / ARQUIVOS["figura_matriz"])
    if figura:
        caminhos["figura_matriz"] = figura
    if not curva.empty:
        figura = figura_curva_treino(curva, destino / ARQUIVOS["figura_curva"])
        if figura:
            caminhos["figura_curva"] = figura

    if imprimir:
        _imprimir(por_fold, por_classe, contagem, percentual, classes, destino)
    return {
        "caminhos": caminhos,
        "por_fold": por_fold,
        "por_classe": por_classe,
        "confusao_contagem": contagem,
        "confusao_percentual": percentual,
        "curva": curva,
    }


def _imprimir(por_fold, por_classe, contagem, percentual, classes, destino) -> None:
    largura = 74
    print("\n" + "=" * largura)
    print("[Agente 1] RELATORIO DE AVALIACAO - predicoes fora da amostra (out-of-fold)")
    print("=" * largura)
    print("\n-- Acuracia e F1 por ciclo (fold) " + "-" * (largura - 34))
    print(por_fold.to_string(index=False))
    print("\n-- Avaliacao por classe " + "-" * (largura - 24))
    print(por_classe.to_string(index=False))
    print("\n-- Matriz de confusao (contagem | % da classe real) " + "-" * (largura - 52))
    combinada = contagem.astype(str) + percentual.map(lambda v: f" ({v:.1f}%)")
    print(combinada.to_string())
    erros = []
    for i, real in enumerate(classes):
        for j, previsto in enumerate(classes):
            if i != j:
                erros.append((percentual.iat[i, j], real, previsto))
    if erros:
        pct, real, previsto = max(erros)
        print(f"\n   Confusao dominante: {pct:.1f}% dos '{real}' foram classificados como '{previsto}'.")
    print(f"\n[Agente 1] Tabelas e figuras em: {destino}")
    print("=" * largura)
