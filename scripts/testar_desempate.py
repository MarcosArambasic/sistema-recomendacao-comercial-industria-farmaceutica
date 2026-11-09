"""***************************************************************************************
SCRIPT    : testar_desempate.py
OBJETIVO  : Teste guiado do Agente de Desempate com a LLM local (Ollama).

            Passo 1  diagnostico: servidor Ollama responde? modelo baixado?
            Passo 2  chamada simples ao modelo (mede o tempo de carga)
            Passo 3  um caso de discordancia total ficticio (Alto x Medio x Baixo)
            Passo 4  (opcional) N casos reais do corpus, sem mexer nos arquivos
                     do pipeline - saida em outputs/teste_desempate.csv

USO:
    python scripts/testar_desempate.py              # passos 1 a 3
    python scripts/testar_desempate.py --amostra 5  # + 5 casos reais do corpus
    python scripts/testar_desempate.py --modelo llama3.2
****************************************************************************************/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import pandas as pd  # noqa: E402

import src.agentes  # noqa: E402,F401  (coloca src/agentes no sys.path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Teste do Agente de Desempate (LLM local)")
    ap.add_argument("--modelo", help="ex.: llama3.1, llama3.2, mistral")
    ap.add_argument("--amostra", type=int, default=0, help="casos reais do corpus a testar")
    a = ap.parse_args()
    if a.modelo:
        os.environ["DESEMPATE_MODELO"] = a.modelo

    import _llm_local as llm
    from agente_desempate_llm import AgenteDesempateLLM

    modelo = a.modelo or llm.MODELO_PADRAO
    print("=" * 70, "\nPASSO 1 - Diagnostico do Ollama\n" + "=" * 70)
    d = llm.diagnosticar(modelo)
    print(json.dumps(d, indent=2, ensure_ascii=False))
    if not d["ok"]:
        print(f"\nFALHOU: {d['motivo']}\nCOMO RESOLVER: {d['solucao']}")
        return 1
    print(f"OK - usando o modelo '{d['modelo']}'.")

    print("\n" + "=" * 70, "\nPASSO 2 - Chamada simples (1a vez carrega o modelo)\n" + "=" * 70)
    t0 = time.time()
    try:
        r = llm.chamar_llm('Responda so com JSON: {"status": "ok"}', modelo=modelo, esperar_json=True)
    except Exception as e:
        print(f"FALHOU: {e}\nDica: aumente o tempo limite com  set OLLAMA_TIMEOUT=600  "
              "ou use um modelo menor (--modelo llama3.2).")
        return 2
    print(f"OK - resposta {r} em {time.time() - t0:.1f}s")

    print("\n" + "=" * 70, "\nPASSO 3 - Caso ficticio de discordancia total\n" + "=" * 70)
    caso = pd.DataFrame([{
        "doc_id": "teste-1",
        "titulo": "FDA approves first generic of semaglutide for type 2 diabetes",
        "descricao": "Regulator clears lower-cost version, expected to cut prices of GLP-1 drugs.",
        "conteudo": "The US Food and Drug Administration approved the first generic version of "
                    "semaglutide, a GLP-1 drug used for type 2 diabetes, opening the market to "
                    "lower-priced competitors.",
        "Avaliador 1": "Alto", "Avaliador 2": "Medio", "Avaliador 3": "Baixo",
    }])
    agente = AgenteDesempateLLM(modelo_llm=modelo)
    disc = agente.identificar_discordancia_total(caso)
    assert len(disc) == 1, "o caso ficticio deveria ser discordancia total"
    res = agente.resolver(disc)
    linha = res.iloc[0]
    print(f"Classificacao: {linha['classificacao_final']}\nJustificativa: {linha['justificativa_desempate']}")
    if linha["classificacao_final"] == "REVISAO_MANUAL":
        print("FALHOU: a LLM respondeu, mas o resultado nao foi aproveitado (veja a justificativa).")
        return 3
    print("OK - esperado 'Alto' (decisao regulatoria direta da FDA).")

    if a.amostra > 0:
        print("\n" + "=" * 70, f"\nPASSO 4 - {a.amostra} caso(s) reais do corpus\n" + "=" * 70)
        from src.agentes.orquestrador import AgenteOrquestrador
        orq = AgenteOrquestrador(log=lambda *_: None)
        df = orq.preparar()
        disc = agente.identificar_discordancia_total(df)
        print(f"{len(disc)} casos de discordancia total no corpus; testando {min(a.amostra, len(disc))}.")
        res = agente.resolver(disc.head(a.amostra))
        destino = RAIZ / "outputs" / "teste_desempate.csv"
        destino.parent.mkdir(parents=True, exist_ok=True)
        cols = [c for c in ["doc_id", "titulo", "Avaliador 1", "Avaliador 2", "Avaliador 3",
                            "classificacao_final", "justificativa_desempate"] if c in res]
        res[cols].to_csv(destino, index=False, encoding="utf-8-sig")
        print(res["classificacao_final"].value_counts().to_string())
        print(f"Salvo em: {destino}")

    print("\nTUDO OK. Agora rode o pipeline completo:  python scripts\\orquestrar.py corpus")
    return 0


if __name__ == "__main__":
    sys.exit(main())
