"""***************************************************************************************
SCRIPT    : gerar_noticias.py
OBJETIVO  : Roda o pipeline de noticias completo (ETL -> Agentes 1/2 -> 4 -> 5).
            E o mesmo que o botao "Gerar mais noticias" do painel executa.

USO:
    python scripts/gerar_noticias.py                 # extrai, classifica e cruza
    python scripts/gerar_noticias.py --sem-extracao  # so reclassifica/cruza o que ja existe
    python scripts/gerar_noticias.py --usar-llm      # classificacao e textos via OpenAI
    python scripts/gerar_noticias.py --so-cruzar     # so Agentes 4 e 5 (ex.: apos nova semana)
    python scripts/gerar_noticias.py --sem-extracao --reclassificar
****************************************************************************************/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.noticias.pipeline_noticias import cruzar_e_recomendar, executar  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Pipeline de noticias + Agentes 4 e 5")
    ap.add_argument("--sem-extracao", action="store_true")
    ap.add_argument("--usar-llm", action="store_true")
    ap.add_argument("--so-cruzar", action="store_true")
    ap.add_argument("--reclassificar", action="store_true",
                    help="refaz a classificacao de todas as noticias (ex.: apos mudar o mapa de entidades)")
    a = ap.parse_args()
    if a.so_cruzar:
        print(cruzar_e_recomendar(usar_llm=a.usar_llm))
    else:
        # o fluxo completo passa pelo Agente Orquestrador (Agentes 1, 2, 4 e 5)
        from src.agentes.orquestrador import AgenteOrquestrador
        print(AgenteOrquestrador(usar_llm=a.usar_llm).missao_noticias(
            fazer_extracao=not a.sem_extracao, reclassificar=a.reclassificar))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
