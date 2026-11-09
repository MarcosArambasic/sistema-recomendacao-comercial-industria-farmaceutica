"""***************************************************************************************
SCRIPT    : diagnostico.py
OBJETIVO  : Gerar uma proposta de dim_regras_diagnostico - as combinacoes dos
            cinco indicadores e o texto de diagnostico correspondente.

            IMPORTANTE: isto e uma SUGESTAO para o grupo revisar, nao uma
            regra de negocio validada. O prompt mestre pede que os
            diagnosticos nao sejam inventados pelo codigo; entao o que este
            modulo faz e montar uma grade COMPLETA e coerente (4^5 = 1024
            combinacoes) a partir de uma hierarquia explicita de regras, para
            voces ajustarem os textos em vez de escreverem 1024 linhas na mao.

LOGICA DA HIERARQUIA:

    A COBERTURA PROJETADA e a espinha do diagnostico - e por ela que o
    representante e cobrado. Os outros quatro indicadores entram como
    CONTEXTO, explicando por que aquele nivel de cobertura e mais ou menos
    grave:

      * market share  -> posicao competitiva (perdendo ou ganhando espaco)
      * CAGR          -> o mercado esta crescendo ou encolhendo
      * indice sazonal-> o mes e naturalmente forte ou fraco
      * CV            -> a demanda e previsivel ou erratica

    Exemplos do que a hierarquia captura:
      - cobertura critica + mes de pico  -> pior cenario (perda concentrada
        no melhor periodo do ano)
      - cobertura critica + mercado em expansao + share baixo -> perda de
        espaco para a concorrencia, nao so falha de execucao
      - cota batida + share baixo + mercado crescendo -> a meta provavelmente
        estava subdimensionada; bater nao significa bom desempenho

AUTOR     : Marcos Arambasic
DATA      : SET/2026
****************************************************************************************/
"""

from __future__ import annotations

from itertools import product

import pandas as pd

# ============================================================================
# Rotulos - precisam bater com desc_grupo da dim_classifica_kpis
# ============================================================================
ROTULOS = {
    "cob_proj": {1: "critico", 2: "atencao", 3: "proximo", 4: "bateu_plano"},
    "mkt": {1: "baixo", 2: "medio", 3: "alto", 4: "muito_alto"},
    "cagr": {1: "queda", 2: "estavel_ou_sem_historico", 3: "crescimento",
             4: "crescimento_forte"},
    "cv": {1: "baixo", 2: "medio", 3: "alto", 4: "muito_alto"},
    "sazonal": {1: "baixo", 2: "medio", 3: "alto", 4: "muito_alto"},
}

LIMITE_DIAGNOSTICO = 255   # varchar(255) no DDL


def _contexto(mkt: int, cagr: int, cv: int, saz: int, ja_dito: set[str]) -> str:
    """
    Escolhe o fator de contexto mais relevante, em ordem de gravidade
    comercial. So um entra no texto - empilhar os quatro produziria frases
    longas e sem foco, que atrapalham o Agente 5 na hora de recomendar.

    `ja_dito` traz os temas que a parte principal do diagnostico ja cobriu,
    para o contexto nao repetir a mesma ideia com outras palavras (ex.:
    "cota atingida, porem com share baixo em mercado crescente - mercado em
    expansao com share baixo...").
    """
    candidatos = [
        ("share_x_mercado", cagr >= 3 and mkt <= 2,
         "mercado em expansao com share baixo: espaco sendo cedido a concorrencia"),
        ("defesa", cagr <= 1 and mkt >= 3,
         "mercado em retracao, mas com share alto: defender a posicao conquistada"),
        ("saz_alta", saz >= 3,
         "mes de alta sazonalidade: o periodo mais decisivo do ano"),
        ("saz_baixa", saz <= 1,
         "mes de baixa sazonalidade: desempenho deve ser lido com esse desconto"),
        ("cv_alto", cv >= 3,
         "demanda erratica: previsibilidade baixa compromete o planejamento"),
        ("mercado_sobe", cagr >= 3,
         "mercado em expansao: ha demanda disponivel para capturar"),
        ("mercado_cai", cagr <= 1,
         "mercado em retracao: parte do desvio vem do ambiente, nao da execucao"),
        ("share_alto", mkt >= 3, "share consolidado no territorio"),
        ("share_baixo", mkt <= 1, "share baixo no territorio"),
    ]
    for tema, condicao, texto in candidatos:
        if condicao and tema not in ja_dito:
            return texto
    return "demais indicadores dentro do padrao"


def _cabeca(cob: int, mkt: int, cagr: int, saz: int) -> tuple[str, set[str]]:
    """
    Parte principal do diagnostico, guiada pela cobertura projetada.
    Devolve tambem os temas ja mencionados, para o contexto nao repeti-los.
    """
    if cob == 1:
        if saz >= 3:
            return "Risco critico de nao atingir a cota no mes de pico", {"saz_alta"}
        if cagr >= 3 and mkt <= 2:
            return ("Risco critico com perda de participacao em mercado aquecido",
                    {"share_x_mercado", "mercado_sobe", "share_baixo"})
        return "Risco critico de nao atingir a cota", set()

    if cob == 2:
        if saz >= 3:
            return "Cota em risco justamente no mes mais forte", {"saz_alta"}
        if cagr <= 1:
            return ("Cota em risco em mercado que esta encolhendo",
                    {"mercado_cai", "defesa"})
        return "Cota em risco", set()

    if cob == 3:
        if cagr >= 3:
            return ("Proximo da cota, com mercado favoravel para fechar a diferenca",
                    {"mercado_sobe", "share_x_mercado"})
        return "Proximo da cota", set()

    # cob == 4
    if mkt <= 2 and cagr >= 3:
        return ("Cota atingida, porem com share baixo em mercado crescente",
                {"share_x_mercado", "mercado_sobe", "share_baixo"})
    if saz <= 1:
        return ("Cota atingida em mes fraco: desempenho acima do esperado",
                {"saz_baixa"})
    if mkt >= 3 and cagr <= 2:
        return ("Cota atingida com lideranca em mercado estavel",
                {"share_alto", "defesa", "mercado_cai"})
    return "Cota atingida", set()


def _acao(cob: int, mkt: int, cagr: int, cv: int, saz: int) -> str:
    """Encaminhamento sugerido - e o que o Agente 5 transforma em recomendacao."""
    if cob == 1:
        if saz >= 3:
            return "acionar plano de contingencia ainda dentro do mes"
        if cagr >= 3:
            return "priorizar cobertura dos outlets de maior potencial"
        return "revisar execucao com o representante"
    if cob == 2:
        if cv >= 3:
            return "acompanhar semanalmente ate estabilizar"
        return "reforcar visitacao nos outlets com maior desvio"
    if cob == 3:
        return "manter o ritmo e monitorar o fechamento"
    if mkt <= 2 and cagr >= 3:
        return "reavaliar o dimensionamento da cota no proximo ciclo"
    return "manter execucao e replicar as praticas do territorio"


def montar_texto(cob: int, mkt: int, cagr: int, cv: int, saz: int) -> str:
    cabeca, ja_dito = _cabeca(cob, mkt, cagr, saz)
    texto = f"{cabeca} - {_contexto(mkt, cagr, cv, saz, ja_dito)}. " \
            f"Acao: {_acao(cob, mkt, cagr, cv, saz)}."
    if len(texto) > LIMITE_DIAGNOSTICO:
        texto = texto[:LIMITE_DIAGNOSTICO - 3].rstrip() + "..."
    return texto


def gerar_dim_regras_diagnostico() -> pd.DataFrame:
    """
    Grade completa: 4 niveis x 5 indicadores = 1.024 combinacoes.

    Cobrir tudo garante que nenhuma linha de fato_analise_semanal_final fique
    sem diagnostico, qualquer que seja a combinacao que aparecer na base.
    """
    linhas = []
    for mkt, cagr, cv, saz, cob in product(range(1, 5), repeat=5):
        linhas.append({
            "cd_mkt_indicator_padrao": mkt,
            "mkt_indicator": ROTULOS["mkt"][mkt],
            "cd_cagr_indicator_padrao": cagr,
            "cagr_indicator": ROTULOS["cagr"][cagr],
            "cd_cv_indicator_padrao": cv,
            "cv_indicator": ROTULOS["cv"][cv],
            "cd_sazonal_indicator_padrao": saz,
            "sazonal_indicator": ROTULOS["sazonal"][saz],
            "cd_cob_proj_indicator_padrao": cob,
            "cob_proj_indicator": ROTULOS["cob_proj"][cob],
            "diagnostico": montar_texto(cob, mkt, cagr, cv, saz),
        })
    return pd.DataFrame(linhas)


def sql_insert(df: pd.DataFrame) -> str:
    """Gera o INSERT pronto para colar no editor SQL do Supabase."""
    cols = [c for c in df.columns]
    def _v(v):
        return str(v) if isinstance(v, (int,)) else "'" + str(v).replace("'", "''") + "'"
    valores = ",\n".join(
        "  (" + ", ".join(_v(r[c]) for c in cols) + ")" for _, r in df.iterrows()
    )
    return (
        "begin;\n"
        "truncate table public.dim_regras_diagnostico restart identity cascade;\n\n"
        f"insert into public.dim_regras_diagnostico ({', '.join(cols)}) values\n"
        f"{valores};\n\ncommit;\n"
    )


if __name__ == "__main__":
    from . import config as cfg

    df = gerar_dim_regras_diagnostico()
    destino = cfg.DIR_SAIDA
    destino.mkdir(parents=True, exist_ok=True)

    df.to_csv(destino / "dim_regras_diagnostico_SUGERIDO.csv", index=False)
    (destino / "dim_regras_diagnostico_SUGERIDO.sql").write_text(
        sql_insert(df), encoding="utf-8"
    )

    print(f"{len(df):,} combinacoes geradas")
    print(f"diagnosticos distintos: {df.diagnostico.nunique()}")
    print(f"tamanho max do texto: {df.diagnostico.str.len().max()} (limite {LIMITE_DIAGNOSTICO})")
    print(f"\nsalvo em: {destino}")
    print("\nAmostra dos cenarios extremos:\n")
    for cob in [1, 4]:
        for mkt, cagr, saz in [(1, 4, 4), (4, 1, 1)]:
            print(f"  cob={ROTULOS['cob_proj'][cob]:<12s} mkt={ROTULOS['mkt'][mkt]:<10s} "
                  f"cagr={ROTULOS['cagr'][cagr]:<24s}")
            print(f"    -> {montar_texto(cob, mkt, cagr, 2, saz)}\n")
