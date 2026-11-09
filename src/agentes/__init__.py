"""Agentes especializados do sistema de recomendacoes comerciais.

Os agentes 1, 2 e de Desempate (e seus modulos de apoio _schema,
dominio_farma, _relatorio_agente1 e _llm_local) usam imports absolutos
entre si ("from _schema import ..."). Colocar esta pasta no sys.path
permite importa-los tanto como `src.agentes.agente2_validador_ajuste`
quanto rodando os arquivos diretamente.
"""

import sys as _sys
from pathlib import Path as _Path

_DIR = str(_Path(__file__).resolve().parent)
if _DIR not in _sys.path:
    _sys.path.insert(0, _DIR)
