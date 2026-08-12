"""
Agente Orquestrador - nucleo de coordenacao (secao 3.3 do TCC)

Implementa o padrao orquestrador-executor: planeja, delega tarefas aos
agentes especializados (1 a 5), mantem um estado compartilhado entre eles
(contexto acessivel a agentes subsequentes) e consolida os outputs para o
painel de insights final.

Este modulo NAO reimplementa cada agente; ele importa e orquestra os
modulos ja construidos em agents/.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).parent / "agents"))


@dataclass
class EstadoPipeline:
    """Estado compartilhado entre os agentes ao longo da execucao do pipeline."""

    noticias_classificadas: Any = None       # output Agente 1
    noticias_validadas: Any = None           # output Agente 2
    desvios_vendas: Any = None               # output Agente 3
    riscos_oportunidades: Any = None         # output Agente 4
    recomendacoes: Any = None                # output Agente 5
    log: list = field(default_factory=list)

    def registrar(self, etapa: str, detalhe: str) -> None:
        self.log.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "etapa": etapa,
                "detalhe": detalhe,
            }
        )


class AgenteOrquestrador:
    """
    Nucleo de coordenacao. Planeja a ordem de acionamento dos agentes,
    delega tarefas e integra os resultados.

    Uso tipico:
        orq = AgenteOrquestrador()
        orq.executar_pipeline(
            df_noticias=df_noticias,
            df_vendas=df_vendas,
            usar_llm=False,  # True se houver OPENAI_API_KEY configurada
        )
        orq.exportar_painel("outputs/painel_insights.json")
    """

    def __init__(self):
        self.estado = EstadoPipeline()

    # ------------------------------------------------------------------
    def _etapa1_classificar_noticias(self, df_noticias, coluna_texto: str) -> None:
        """
        Agente 1 - Classificador de Impacto (mBERT).

        Nota de implementacao: o fine-tuning completo de mBERT exige GPU e
        tempo de treino que nao cabem neste ambiente. Aqui o Agente 1 usa
        as classificacoes JA CONSOLIDADAS do corpus anotado (apos o Agente
        de Desempate) como proxy do output do classificador treinado -
        exatamente os rotulos que alimentariam o fine-tuning em producao.
        """
        self.estado.noticias_classificadas = df_noticias
        self.estado.registrar(
            "Agente 1 - Classificador de Impacto",
            f"{len(df_noticias)} noticias classificadas (rotulos consolidados)",
        )

    def _etapa2_validar_noticias(self, df_noticias) -> None:
        """
        Agente 2 - Validador & Ajuste.
        Remove itens de baixo impacto (regra de negocio descrita na secao
        3.4.2) e mantem apenas ALTO/MEDIO para as etapas seguintes.
        """
        col_impacto = self._detectar_coluna_impacto(df_noticias)
        df_validado = df_noticias[
            df_noticias[col_impacto].str.upper() != "BAIXO"
        ].copy()
        self.estado.noticias_validadas = df_validado
        self.estado.registrar(
            "Agente 2 - Validador & Ajuste",
            f"{len(df_validado)}/{len(df_noticias)} noticias mantidas apos filtro de baixo impacto",
        )

    @staticmethod
    def _detectar_coluna_impacto(df) -> str:
        for col in df.columns:
            if "avaliador" in col.lower() or "impacto" in col.lower() or "classific" in col.lower():
                return col
        raise ValueError("Nao foi possivel detectar a coluna de impacto/classificacao.")

    def _etapa3_comparar_vendas(self, df_vendas) -> None:
        """Agente 3 - Comparador de Vendas (XGBoost)."""
        from agente3_comparador_vendas import AgenteComparadorVendas

        agente3 = AgenteComparadorVendas()
        metricas = agente3.treinar(df_vendas)

        ultimo_periodo = df_vendas["data"].max()
        amostra = df_vendas[df_vendas["data"] == ultimo_periodo]
        resultados = agente3.analisar(amostra)
        df_desvios = agente3.exportar_serie_temporal(resultados)

        self.estado.desvios_vendas = df_desvios
        self.estado.registrar(
            "Agente 3 - Comparador de Vendas",
            f"XGBoost treinado (MAE={metricas['mae']}, R2={metricas['r2']}); "
            f"{len(df_desvios)} desvios analisados no periodo {ultimo_periodo}",
        )

    def _etapa4_riscos_oportunidades(self, usar_llm: bool) -> None:
        """Agente 4 - Identificador de Riscos e Oportunidades."""
        from agents.agente4_riscos_oportunidades import AgenteRiscosOportunidades

        agente4 = AgenteRiscosOportunidades(usar_llm=usar_llm)
        achados = agente4.correlacionar(
            self.estado.noticias_validadas,
            self.estado.desvios_vendas,
        )
        self.estado.riscos_oportunidades = agente4.exportar(achados)
        self.estado.registrar(
            "Agente 4 - Identificador de Riscos e Oportunidades",
            f"{len(achados)} achados gerados (llm={'on' if usar_llm else 'heuristico'})",
        )

    def _etapa5_gerar_recomendacoes(self, usar_llm: bool) -> None:
        """Agente 5 - Gerador de Recomendacoes."""
        from agente5_gerador_recomendacoes import AgenteGeradorRecomendacoes

        agente5 = AgenteGeradorRecomendacoes(usar_llm=usar_llm)
        recomendacoes = agente5.gerar(self.estado.riscos_oportunidades)
        # Guardamos como dicts (via exportar/asdict), nao como objetos dataclass:
        # json.dump nao serializa dataclasses nativamente e, com default=str,
        # cada recomendacao virava uma string "Recomendacao(...)" em vez de um
        # objeto JSON estruturado no painel final.
        self.estado.recomendacoes = agente5.exportar(recomendacoes)
        self.estado.registrar(
            "Agente 5 - Gerador de Recomendacoes",
            f"{len(recomendacoes)} recomendacoes tacitas geradas",
        )

    # ------------------------------------------------------------------
    def executar_pipeline(
        self,
        df_noticias,
        df_vendas,
        usar_llm: bool = False,
    ) -> EstadoPipeline:
        """Executa o pipeline completo, na ordem definida pelo orquestrador."""
        self._etapa1_classificar_noticias(df_noticias, coluna_texto="titulo")
        self._etapa2_validar_noticias(self.estado.noticias_classificadas)
        self._etapa3_comparar_vendas(df_vendas)
        self._etapa4_riscos_oportunidades(usar_llm=usar_llm)
        self._etapa5_gerar_recomendacoes(usar_llm=usar_llm)
        return self.estado

    def exportar_painel(self, caminho: str) -> None:
        """Consolida os outputs em um JSON para alimentar o painel de insights."""
        painel = {
            "gerado_em": datetime.now().isoformat(timespec="seconds"),
            "resumo": {
                "noticias_classificadas": len(self.estado.noticias_classificadas)
                if self.estado.noticias_classificadas is not None else 0,
                "noticias_validadas": len(self.estado.noticias_validadas)
                if self.estado.noticias_validadas is not None else 0,
                "desvios_analisados": len(self.estado.desvios_vendas)
                if self.estado.desvios_vendas is not None else 0,
                "achados_riscos_oportunidades": len(self.estado.riscos_oportunidades)
                if self.estado.riscos_oportunidades is not None else 0,
                "recomendacoes_geradas": len(self.estado.recomendacoes)
                if self.estado.recomendacoes is not None else 0,
            },
            "recomendacoes": self.estado.recomendacoes,
            "log_execucao": self.estado.log,
        }
        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(painel, f, ensure_ascii=False, indent=2, default=str)
        print(f"Painel exportado -> {caminho}")
