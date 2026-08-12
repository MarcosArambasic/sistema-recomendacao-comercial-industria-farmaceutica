"""
Agente 5 - Gerador de Recomendacoes (secao 3.4.5 do TCC)

Sintetiza os outputs de todos os agentes anteriores e, via GPT-4 com
prompts estruturados, produz recomendacoes taticas priorizadas para
gestores e diretores comerciais. Cada recomendacao inclui: justificativa
rastreavel, nivel de urgencia, produtos/territorios afetados, acao
especifica sugerida e impacto esperado no faturamento quando quantificavel.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

Urgencia = Literal["alta", "media", "baixa"]

PROMPT_SISTEMA = """Voce e um consultor senior de inteligencia comercial \
para a industria farmaceutica, especializado no mercado de diabetes.

Dado um achado de risco ou oportunidade (com justificativa, produtos e \
territorios afetados), gere UMA recomendacao tatica acionavel para \
gestores e diretores comerciais.

Responda em JSON:
{
  "acao_sugerida": "acao especifica e concreta, uma frase",
  "justificativa": "por que essa acao, baseada nas evidencias fornecidas",
  "urgencia": "alta|media|baixa",
  "impacto_esperado": "descricao qualitativa ou quantitativa do impacto no faturamento"
}
"""


@dataclass
class Recomendacao:
    origem_categoria: str
    produtos_afetados: list
    territorios_afetados: list
    acao_sugerida: str
    justificativa: str
    urgencia: Urgencia
    impacto_esperado: str
    fonte_rastreavel: str


class AgenteGeradorRecomendacoes:
    def __init__(self, usar_llm: bool = False, api_key: str | None = None, modelo: str = "gpt-4"):
        self.usar_llm = usar_llm
        self.modelo = modelo
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = None
        if usar_llm:
            try:
                from openai import OpenAI
                if self.api_key:
                    self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                pass

    # ------------------------------------------------------------------
    def _gerar_heuristico(self, achado: dict) -> Recomendacao:
        """Template baseado em regras - usado para testar o fluxo sem API."""
        categoria = achado["categoria"]
        urgencia_map = {
            "risco_imediato": "alta",
            "risco_medio_prazo": "media",
            "oportunidade": "baixa",
        }
        acao_map = {
            "risco_imediato": "Acionar time comercial para plano de contingencia imediato nos territorios afetados",
            "risco_medio_prazo": "Monitorar indicadores semanalmente e preparar plano de mitigacao",
            "oportunidade": "Reforcar cobertura de visitacao medica para capturar crescimento identificado",
        }

        return Recomendacao(
            origem_categoria=categoria,
            produtos_afetados=achado["produtos_afetados"],
            territorios_afetados=achado["territorios_afetados"],
            acao_sugerida=acao_map.get(categoria, "Avaliar situacao com time comercial"),
            justificativa=(
                f"[heuristico] baseado em '{achado['noticia_titulo'][:80]}' "
                f"com desvio medio de vendas de {achado['desvio_medio_pct']}%"
            ),
            urgencia=urgencia_map.get(categoria, "media"),
            impacto_esperado=f"Desvio medio observado: {achado['desvio_medio_pct']}%",
            fonte_rastreavel=achado["noticia_titulo"],
        )

    def _gerar_via_llm(self, achado: dict) -> Recomendacao:
        if self.client is None:
            raise RuntimeError(
                "Cliente OpenAI nao configurado. Defina OPENAI_API_KEY ou "
                "use usar_llm=False para o modo heuristico de teste."
            )
        resposta = self.client.chat.completions.create(
            model=self.modelo,
            messages=[
                {"role": "system", "content": PROMPT_SISTEMA},
                {"role": "user", "content": json.dumps(achado, ensure_ascii=False)},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        saida = json.loads(resposta.choices[0].message.content)
        return Recomendacao(
            origem_categoria=achado["categoria"],
            produtos_afetados=achado["produtos_afetados"],
            territorios_afetados=achado["territorios_afetados"],
            acao_sugerida=saida["acao_sugerida"],
            justificativa=saida["justificativa"],
            urgencia=saida["urgencia"],
            impacto_esperado=saida["impacto_esperado"],
            fonte_rastreavel=achado["noticia_titulo"],
        )

    # ------------------------------------------------------------------
    def gerar(self, achados: list[dict]) -> list[Recomendacao]:
        gerar_um = self._gerar_via_llm if self.usar_llm else self._gerar_heuristico
        recomendacoes = [gerar_um(a) for a in achados]
        # prioriza por urgencia (alta > media > baixa)
        ordem = {"alta": 0, "media": 1, "baixa": 2}
        recomendacoes.sort(key=lambda r: ordem.get(r.urgencia, 3))
        return recomendacoes

    @staticmethod
    def exportar(recomendacoes: list[Recomendacao]) -> list[dict]:
        return [asdict(r) for r in recomendacoes]


if __name__ == "__main__":
    from agente4_riscos_oportunidades import AgenteRiscosOportunidades
    import pandas as pd

    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    df_noticias = pd.read_excel(
        BASE_DIR / "data" / "Nova_classificação_kappa_0_68_aprox.xlsx",
        sheet_name="Dados",
    ).head(50)
    df_desvios = pd.read_csv(BASE_DIR / "outputs" / "agente3_desvios.csv")

    agente4 = AgenteRiscosOportunidades(usar_llm=False)
    achados = agente4.correlacionar(df_noticias, df_desvios)
    achados_dict = agente4.exportar(achados)

    agente5 = AgenteGeradorRecomendacoes(usar_llm=False)
    recomendacoes = agente5.gerar(achados_dict)

    print(f"{len(recomendacoes)} recomendacoes geradas, ordenadas por urgencia:\n")
    for r in recomendacoes:
        print(f"[{r.urgencia.upper()}] {r.acao_sugerida}")
        print(f"  Produtos: {r.produtos_afetados} | Territorios: {r.territorios_afetados}")
        print(f"  Justificativa: {r.justificativa}")
        print(f"  Impacto esperado: {r.impacto_esperado}\n")
