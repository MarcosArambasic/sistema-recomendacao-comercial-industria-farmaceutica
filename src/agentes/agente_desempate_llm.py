"""
AGENTE DE DESEMPATE (via LLM local) - secao 3.5 do TCC
=========================================================
Atua SOMENTE nos casos de discordancia total (nenhum par de avaliadores
concorda) e usa uma LLM local (Ollama, llama3.1) que le a noticia e
produz a classificacao de referencia com justificativa.

CUSTO: R$ 0,00 (LLM local via Ollama, no lugar da API paga da OpenAI).

CORRECOES (OUT/2026):
  - nao depende mais da biblioteca `ollama` (ver _llm_local.py): o
    Desempate deixava de enxergar o Llama porque o import falhava em
    silencio;
  - mostra o MOTIVO quando a LLM nao esta disponivel;
  - valida o rotulo devolvido (Alto/Medio/Baixo, com ou sem acento);
    resposta fora do padrao vira REVISAO_MANUAL em vez de rotulo errado;
  - textos nulos (NaN) nao vao mais para o prompt como "nan";
  - progresso a cada caso e opcoes --modelo, --limite e --diagnostico.

Requisitos: pandas (+ Ollama instalado e um modelo baixado).

USO:
    python src/agentes/agente_desempate_llm.py --diagnostico
    python src/agentes/agente_desempate_llm.py data/estatistica_kappa/noticias_classificadas_ampliado.xlsx --limite 5
"""

from __future__ import annotations

import time
import unicodedata

import pandas as pd

from _llm_local import (MODELO_PADRAO, chamar_llm, diagnosticar, llm_disponivel,
                        modelo_em_uso, motivo_indisponivel)

CRITERIOS_IMPACTO = """Criterios de classificacao (contexto: faturamento de medicamentos para diabetes):
- ALTO: decisao regulatoria direta (ANVISA/CONITEC/FDA/EMA), desabastecimento/recall,
  reajuste de preco CMED, fusao/aquisicao/parceria no setor, ou geopolitica com impacto
  explicito na cadeia de suprimentos/IFA/importacao de insumos.
- MEDIO: tema regulatorio/economico geral plausivel para o setor (cambio, PBM),
  guerra/conflito/desastre sem mencao direta mas com rota relevante, estudo pre-clinico/fase 1.
- BAIXO: saude publica geral, campanha, conscientizacao, perfil/entrevista institucional sem fato novo.
"""

PROMPT_TEMPLATE = """Voce e um especialista em regulacao e mercado farmaceutico, atuando como
arbitro de desempate para um corpus de anotacao cientifica.

{criterios}

Tres avaliadores classificaram a noticia abaixo em impacto no faturamento de
medicamentos para diabetes, e OS TRES DISCORDARAM TOTALMENTE entre si (nenhum par bateu):
Avaliador 1: {av1} | Avaliador 2: {av2} | Avaliador 3: {av3}

NOTICIA (conteudo integral):
Titulo: {titulo}
Descricao: {descricao}
Conteudo: {conteudo}

TAREFA: leia a noticia por completo e produza a classificacao de referencia (a que voce
julga correta), com uma justificativa curta (ate 2 frases, em portugues) baseada nos criterios acima.

Responda SOMENTE com um JSON no formato:
{{"classificacao_final": "Alto" ou "Medio" ou "Baixo", "justificativa": "..."}}
"""

ROTULOS_VALIDOS = {"alto": "Alto", "medio": "Medio", "baixo": "Baixo",
                   "high": "Alto", "medium": "Medio", "low": "Baixo"}


def _texto(valor, limite: int | None = None) -> str:
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return ""
    s = str(valor).strip()
    return s[:limite] if limite else s


def normalizar_classificacao(valor) -> str | None:
    """'Médio', 'MEDIO', ' medio.' -> 'Medio'; qualquer outra coisa -> None."""
    s = unicodedata.normalize("NFKD", _texto(valor)).encode("ascii", "ignore").decode().lower()
    s = s.strip(" .!\"'")
    return ROTULOS_VALIDOS.get(s)


class AgenteDesempateLLM:
    """Agente de Desempate fiel a secao 3.5 (so discordancia total, via LLM local)."""

    def __init__(self, modelo_llm: str = MODELO_PADRAO, verbose: bool = True):
        self.modelo_llm = modelo_llm
        self.verbose = verbose

    # ------------------------------------------------------------------
    @staticmethod
    def identificar_discordancia_total(df: pd.DataFrame) -> pd.DataFrame:
        """Linhas em que os 3 avaliadores deram notas diferentes (nenhum par concorda)."""
        a1, a2, a3 = df["Avaliador 1"], df["Avaliador 2"], df["Avaliador 3"]
        completos = a1.notna() & a2.notna() & a3.notna()
        nenhum_par_concorda = ~((a1 == a2) | (a1 == a3) | (a2 == a3))
        return df[completos & nenhum_par_concorda].copy()

    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[Agente de Desempate] {msg}", flush=True)

    def resolver(self, df_discordantes: pd.DataFrame) -> pd.DataFrame:
        """Aplica o desempate via LLM (REVISAO_MANUAL se a LLM nao estiver disponivel)."""
        if df_discordantes.empty:
            self._log("Nenhum caso de discordancia total encontrado - nada a fazer.")
            return df_discordantes.assign(classificacao_final=[], justificativa_desempate=[])

        usar_llm = llm_disponivel(self.modelo_llm)
        if usar_llm:
            self._log(f"LLM local ativa: {modelo_em_uso(self.modelo_llm)} | "
                      f"{len(df_discordantes)} caso(s) a resolver.")
        else:
            motivo = motivo_indisponivel(self.modelo_llm)
            self._log(f"AVISO: LLM local indisponivel -> {motivo}")
            self._log("Todos os casos serao marcados como REVISAO_MANUAL.")

        classificacoes, justificativas = [], []
        t0 = time.time()
        total = len(df_discordantes)
        for i, (_, row) in enumerate(df_discordantes.iterrows(), start=1):
            if not usar_llm:
                classificacoes.append("REVISAO_MANUAL")
                justificativas.append(f"LLM local indisponivel: {motivo_indisponivel(self.modelo_llm)}")
                continue

            prompt = PROMPT_TEMPLATE.format(
                criterios=CRITERIOS_IMPACTO,
                av1=_texto(row.get("Avaliador 1")), av2=_texto(row.get("Avaliador 2")),
                av3=_texto(row.get("Avaliador 3")),
                titulo=_texto(row.get("titulo")), descricao=_texto(row.get("descricao")),
                conteudo=_texto(row.get("conteudo"), 4000) or "(sem conteudo integral; use titulo e descricao)",
            )
            try:
                resultado = chamar_llm(prompt, modelo=self.modelo_llm, esperar_json=True)
                rotulo = normalizar_classificacao(resultado.get("classificacao_final"))
                if rotulo is None:
                    classificacoes.append("REVISAO_MANUAL")
                    justificativas.append(f"Rotulo invalido devolvido pela LLM: {resultado!r}"[:500])
                else:
                    classificacoes.append(rotulo)
                    justificativas.append(_texto(resultado.get("justificativa")))
            except Exception as e:
                classificacoes.append("REVISAO_MANUAL")
                justificativas.append(f"Falha ao consultar a LLM: {e}"[:500])
            decorrido = time.time() - t0
            self._log(f"{i}/{total} -> {classificacoes[-1]} "
                      f"({decorrido:.0f}s, ~{decorrido / i * (total - i):.0f}s restantes)")

        df_resultado = df_discordantes.copy()
        df_resultado["classificacao_final"] = classificacoes
        df_resultado["justificativa_desempate"] = justificativas
        n_ok = sum(c != "REVISAO_MANUAL" for c in classificacoes)
        self._log(f"Concluido: {n_ok}/{total} resolvidos pela LLM, {total - n_ok} em REVISAO_MANUAL.")
        return df_resultado


if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Agente de Desempate (LLM local) sobre uma planilha anotada.")
    parser.add_argument("planilha", nargs="?")
    parser.add_argument("--saida", default="outputs/desempate_llm.xlsx")
    parser.add_argument("--modelo", default=MODELO_PADRAO)
    parser.add_argument("--limite", type=int, help="resolve so os N primeiros casos (teste rapido)")
    parser.add_argument("--diagnostico", action="store_true", help="so verifica o Ollama e sai")
    args = parser.parse_args()

    if args.diagnostico or not args.planilha:
        d = diagnosticar(args.modelo)
        print(json.dumps(d, indent=2, ensure_ascii=False))
        sys.exit(0 if d["ok"] else 1)

    caminho = Path(args.planilha)
    df = pd.read_csv(caminho) if caminho.suffix.lower() == ".csv" else pd.read_excel(caminho)
    if not {"Avaliador 1", "Avaliador 2", "Avaliador 3"} <= set(df.columns):
        # planilhas do projeto usam nomes diferentes; reaproveita o normalizador do pipeline
        from _schema import preparar_corpus  # type: ignore
        df, _ = preparar_corpus(df, verbose=False)
    agente = AgenteDesempateLLM(modelo_llm=args.modelo)
    disc = agente.identificar_discordancia_total(df)
    print(f"[Agente de Desempate] {len(disc)} caso(s) de discordancia total identificados.")
    if args.limite:
        disc = disc.head(args.limite)
    res = agente.resolver(disc)
    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    res.to_excel(args.saida, index=False)
    print(f"[Agente de Desempate] Resultado salvo em: {args.saida}")
