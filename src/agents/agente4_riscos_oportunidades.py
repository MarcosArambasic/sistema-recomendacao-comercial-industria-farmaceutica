"""
Agente de Desempate (secao 3.5 do TCC)

Recebe os casos de discordancia total entre os 3 anotadores humanos (nenhum
par concorda) e produz uma classificacao de referencia fundamentada no
conteudo integral da noticia, no contexto farmaceutico (mercado de diabetes)
e nos criterios de impacto definidos pelo projeto.

Requer variavel de ambiente OPENAI_API_KEY para rodar de fato. Sem a key,
o modulo ainda pode ser importado e testado com mocks (ver
`_classificar_mock`), o que permite validar toda a logica de identificacao
de casos de discordancia total sem gastar chamadas de API.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    from openai import OpenAI
    OPENAI_DISPONIVEL = True
except ImportError:
    OPENAI_DISPONIVEL = False


PROMPT_SISTEMA = """Voce e um especialista em analise de impacto de noticias \
sobre o mercado farmaceutico de medicamentos para diabetes (insulinas, \
metformina, inibidores de SGLT2, agonistas de GLP-1, etc).

Classifique o IMPACTO da noticia no faturamento de laboratorios \
farmaceuticos do segmento de diabetes em uma das tres classes: \
ALTO, MEDIO ou BAIXO.

Criterios:
- ALTO: eventos com potencial de alterar significativamente vendas/demanda \
no curto prazo (aprovacoes regulatorias, mudancas de reembolso, recalls, \
rupturas de estoque, resultados de trials decisivos).
- MEDIO: eventos relevantes ao setor mas sem efeito direto e imediato em \
vendas (financiamento de pesquisa, tendencias de mercado, parcerias).
- BAIXO: mencoes tangenciais, contexto geral de saude publica sem relacao \
direta e imediata com o faturamento de medicamentos para diabetes.

Responda APENAS em JSON, no formato:
{"classificacao": "ALTO|MEDIO|BAIXO", "justificativa": "..."}
"""


@dataclass
class ResultadoDesempate:
    indice: int
    titulo: str
    classificacoes_originais: tuple
    classificacao_final: str
    justificativa: str


class AgenteDesempate:
    def __init__(self, api_key: str | None = None, modelo: str = "gpt-4"):
        self.modelo = modelo
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = None
        if OPENAI_DISPONIVEL and self.api_key:
            self.client = OpenAI(api_key=self.api_key)

    # ------------------------------------------------------------------
    @staticmethod
    def identificar_discordancia_total(
        df: pd.DataFrame,
        col_a1: str,
        col_a2: str,
        col_a3: str,
    ) -> pd.DataFrame:
        """
        Retorna apenas as linhas em que os tres avaliadores divergem
        totalmente entre si (nenhum par concorda) - conforme secao 3.5.
        """
        mask = (
            (df[col_a1] != df[col_a2])
            & (df[col_a1] != df[col_a3])
            & (df[col_a2] != df[col_a3])
        )
        return df[mask].copy()

    # ------------------------------------------------------------------
    def _classificar_via_api(self, titulo: str, descricao: str) -> dict:
        if self.client is None:
            raise RuntimeError(
                "Cliente OpenAI nao configurado. Defina OPENAI_API_KEY ou "
                "use _classificar_mock() para testes sem custo de API."
            )
        resposta = self.client.chat.completions.create(
            model=self.modelo,
            messages=[
                {"role": "system", "content": PROMPT_SISTEMA},
                {
                    "role": "user",
                    "content": f"Titulo: {titulo}\n\nDescricao: {descricao[:2000]}",
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return json.loads(resposta.choices[0].message.content)

    @staticmethod
    def _classificar_mock(titulo: str, descricao: str) -> dict:
        """
        Heuristica simples para testar o pipeline sem gastar chamadas de
        API. NAO substitui a classificacao real via LLM - serve apenas
        para validar a integracao end-to-end.
        """
        texto = f"{titulo} {descricao}".lower()
        palavras_alto = ["aprova", "recall", "desabastec", "ruptura", "reembolso"]
        palavras_medio = ["financiamento", "pesquisa", "trial", "parceria"]

        if any(p in texto for p in palavras_alto):
            classe = "ALTO"
        elif any(p in texto for p in palavras_medio):
            classe = "MEDIO"
        else:
            classe = "BAIXO"

        return {
            "classificacao": classe,
            "justificativa": "[MOCK] classificacao heuristica para testes sem API",
        }

    # ------------------------------------------------------------------
    def resolver(
        self,
        df_discordancia: pd.DataFrame,
        col_a1: str,
        col_a2: str,
        col_a3: str,
        col_titulo: str = "titulo",
        col_descricao: str = "descricao",
        usar_mock: bool = False,
    ) -> list[ResultadoDesempate]:
        resultados = []
        classificar = self._classificar_mock if usar_mock else self._classificar_via_api

        for idx, row in df_discordancia.iterrows():
            saida = classificar(row[col_titulo], row.get(col_descricao, ""))
            resultados.append(
                ResultadoDesempate(
                    indice=idx,
                    titulo=row[col_titulo],
                    classificacoes_originais=(row[col_a1], row[col_a2], row[col_a3]),
                    classificacao_final=saida["classificacao"],
                    justificativa=saida["justificativa"],
                )
            )
        return resultados


if __name__ == "__main__":
    # Exemplo de uso com a base real de anotacoes (Nova_classificacao_kappa...).
    # Coloque o arquivo .xlsx real em data/ antes de rodar este script.
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    df = pd.read_excel(
        BASE_DIR / "data" / "Nova_classificação_kappa_0_68_aprox.xlsx",
        sheet_name="Dados",
    )

    agente = AgenteDesempate()
    df_discord = agente.identificar_discordancia_total(
        df, "Avaliador 1", "Avaliador 2", "Avaliador 3"
    )
    print(f"{len(df_discord)} casos de discordancia total identificados")

    # roda em modo mock (sem gastar API) para demonstrar o fluxo completo
    resultados = agente.resolver(
        df_discord.head(5),
        "Avaliador 1", "Avaliador 2", "Avaliador 3",
        col_titulo="titulo", col_descricao="descricao",
        usar_mock=True,
    )
    for r in resultados:
        print(f"\n> {r.titulo[:80]}")
        print(f"  Originais: {r.classificacoes_originais} -> Final: {r.classificacao_final}")
        print(f"  Justificativa: {r.justificativa}")
