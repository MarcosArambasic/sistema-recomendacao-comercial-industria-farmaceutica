"""***************************************************************************************
SCRIPT    : orquestrar.py
OBJETIVO  : Ponto de entrada unico do Agente Orquestrador.

USO:
    python scripts/orquestrar.py corpus                 # Desempate + treino/avaliacao dos Agentes 1 e 2
    python scripts/orquestrar.py noticias               # ETL + Agentes 1, 2, 4 e 5 (botao "Gerar mais noticias")
    python scripts/orquestrar.py noticias --sem-extracao
    python scripts/orquestrar.py semana                 # proxima semana + Agente 3 + 4 e 5 (botao "Gerar nova semana")
    python scripts/orquestrar.py semana --ano 2026 --mes 10 --semana 2
    python scripts/orquestrar.py cruzar                 # so Agentes 4 e 5
    python scripts/orquestrar.py status                 # motores disponiveis e ultima execucao

OPCOES GERAIS:
    --usar-llm            Agente 5 (e classificacao) via LLM quando houver chave
    --substitutos         forca Naive Bayes / encoder lexical mesmo com mBERT instalado
****************************************************************************************/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.agentes.orquestrador import (  # noqa: E402
    DIR_ORQ, MODELO_MBERT, AgenteOrquestrador, encoder_semantico_disponivel, mbert_disponivel,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Agente Orquestrador")
    ap.add_argument("missao", choices=["corpus", "noticias", "semana", "cruzar", "status"])
    ap.add_argument("--usar-llm", action="store_true")
    ap.add_argument("--substitutos", action="store_true")
    ap.add_argument("--sem-extracao", action="store_true")
    ap.add_argument("--reclassificar", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epocas", type=int, default=5)
    ap.add_argument("--ano", type=int)
    ap.add_argument("--mes", type=int)
    ap.add_argument("--semana", type=int)
    a = ap.parse_args()

    if a.missao == "status":
        from _llm_local import llm_disponivel
        print(json.dumps({
            "agente1_mbert_instalado": mbert_disponivel(),
            "agente1_modelo_treinado": MODELO_MBERT.exists(),
            "agente2_encoder_semantico": encoder_semantico_disponivel(),
            "desempate_llm_local": llm_disponivel(),
        }, indent=2))
        p = DIR_ORQ / "estado.json"
        if p.exists():
            est = json.loads(p.read_text(encoding="utf-8"))
            for missao, info in est.items():
                if missao != "historico_log":
                    print(f"{missao:<9} {info['executado_em']}  motores={info['motores']}")
        return 0

    orq = AgenteOrquestrador(usar_llm=a.usar_llm, forcar_substitutos=a.substitutos)
    if a.missao == "corpus":
        r = orq.missao_corpus(n_folds=a.folds, epocas=a.epocas)
        print(json.dumps({k: r[k] for k in ("motores", "anotacao", "desempate")}, indent=2))
    elif a.missao == "noticias":
        print(orq.missao_noticias(fazer_extracao=not a.sem_extracao, reclassificar=a.reclassificar))
    elif a.missao == "semana":
        print(orq.missao_semana(a.ano, a.mes, a.semana))
    else:
        print(orq.etapa4_5_cruzar())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
