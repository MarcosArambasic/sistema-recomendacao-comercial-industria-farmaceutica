"""***************************************************************************************
SCRIPT    : acoes.py
OBJETIVO  : O que os dois botoes do painel executam - ambos delegam ao
            Agente Orquestrador (scripts/orquestrar.py).

    GERAR NOVA SEMANA
      1. scripts/executar_semana.py --ano --mes --semana  (proxima semana do calendario,
         sem trava de data; se o mes ainda nao tem demanda, ela e criada pelo
         modelo e consolidada na ultima semana)
      2. se for a 1a semana do mes, retreina o Agente 3 (os KPIs do mes
         anterior acabaram de ser calculados)
      3. Agentes 4 e 5 recruzam as noticias ja classificadas com as novas
         oportunidades (scripts/gerar_noticias.py --so-cruzar)

    GERAR MAIS NOTICIAS
      scripts/gerar_noticias.py  (ETL das APIs -> Agentes 1/2 -> 4 -> 5)

Cada etapa roda em um processo separado (subprocess): uma falha nao derruba a
tela, a memoria dos calculos pesados e liberada ao final e a saida de cada
script aparece ao vivo no painel.
****************************************************************************************/
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

RAIZ = Path(__file__).resolve().parent.parent.parent
PY = sys.executable


def _rodar(cmd: list[str], log: Callable[[str], None]) -> int:
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(cmd, cwd=RAIZ, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", env=env)
    for linha in proc.stdout:            # type: ignore[union-attr]
        linha = linha.rstrip()
        if linha:
            log(linha)
    return proc.wait()


def _usar_llm() -> list[str]:
    return ["--usar-llm"] if os.environ.get("PAINEL_USAR_LLM") == "1" else []


def gerar_nova_semana(proxima: dict, log: Callable[[str], None]) -> tuple[bool, str]:
    """Delegado ao Agente Orquestrador (missao 'semana')."""
    rotulo = f"{proxima['ano']}-{proxima['mes']:02d} semana {proxima['nr_semana']}"
    rc = _rodar([PY, "scripts/orquestrar.py", "semana", "--ano", str(proxima["ano"]),
                 "--mes", str(proxima["mes"]), "--semana", str(proxima["nr_semana"]), *_usar_llm()], log)
    return (rc == 0, f"Semana {rotulo} gerada." if rc == 0
            else f"O orquestrador falhou ao gerar {rotulo}. Veja o log abaixo.")


def gerar_mais_noticias(log: Callable[[str], None]) -> tuple[bool, str]:
    """Delegado ao Agente Orquestrador (missao 'noticias')."""
    rc = _rodar([PY, "scripts/orquestrar.py", "noticias", *_usar_llm()], log)
    return (rc == 0, "Notícias atualizadas." if rc == 0 else "O pipeline de notícias falhou.")
