"""***************************************************************************************
SCRIPT    : agente5_recomendacoes.py
OBJETIVO  : Agente 5 - Gerador de Recomendacoes.

            Para cada setor, escolhe as N noticias mais impactantes (maior
            score de achado no Agente 4) e, para cada uma, as oportunidades /
            riscos mais importantes que ela afeta. Sobre esse par
            noticia x oportunidades gera UMA recomendacao tatica com acao,
            justificativa rastreavel, urgencia e impacto esperado.

            Dois modos:
              heuristico (padrao)  modelos de texto por categoria, sem custo
              llm                  OPENAI_API_KEY + usar_llm=True

SAIDA: data/agentes/agente5_saida.json - e o que o painel exibe na lista
       "noticias mais impactantes".

AUTOR     : Marcos Arambasic
DATA      : OUT/2026
****************************************************************************************/
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

AGENTE = "agente5_recomendacoes"
VERSAO_SCHEMA = "1.0"
URGENCIAS = ("alta", "media", "baixa")

PROMPT_LLM = """Voce e consultor senior de inteligencia comercial da industria \
farmaceutica (mercado de diabetes). Recebe uma noticia e as oportunidades/riscos \
de mercado que ela afeta em um setor comercial. Gere UMA recomendacao tatica \
para o gerente do setor.

Responda APENAS em JSON:
{"acao_sugerida": "uma frase concreta", "justificativa": "ligando noticia e dados",
 "urgencia": "alta|media|baixa", "impacto_esperado": "texto curto, com unidades se houver"}
"""

ROTULO_TIPO = {
    "CAPTURAR_CRESCIMENTO": "capturar crescimento",
    "AMPLIAR_PENETRACAO": "ampliar penetração",
    "ANTECIPAR_PICO_SAZONAL": "antecipar pico sazonal",
    "DEFENDER_SHARE": "defender share",
    "RECUPERAR_SHARE": "recuperar share",
    "MANTER_POSICAO": "manter posição",
}


def _urgencia(top: dict, noticia: dict) -> str:
    if top["categoria"] == "risco_imediato":
        return "alta"
    if noticia["nivel_impacto"] == "alto" and top["multiplicador"] > 1:
        return "alta"
    if noticia["nivel_impacto"] == "alto" or top["multiplicador"] > 1:
        return "media"
    return "baixa"


class AgenteRecomendacoes:

    def __init__(self, nomes_produto: dict | None = None, nomes_outlet: dict | None = None,
                 usar_llm: bool = False):
        self.np = nomes_produto or {}
        self.no = nomes_outlet or {}
        self.client = None
        if usar_llm and os.environ.get("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                self.modelo = os.environ.get("OPENAI_MODELO", "gpt-4o-mini")
            except Exception:           # noqa: BLE001
                self.client = None

    def _p(self, i):
        return self.np.get(i, f"Produto {i}")

    def _o(self, i):
        return f"Brick {self.no[i]}" if i in self.no else f"Outlet {i}"

    # ------------------------------------------------------------------
    def _locais(self, ops: list[dict]) -> str:
        """'Brick 102544 (Produto Proprio 04 e 06) e Brick 102534 (Produto Proprio 04)'."""
        por_local: dict[str, list] = {}
        for o in ops:
            por_local.setdefault(self._o(o["id_outlet"]), []).append(self._p(o["id_produto"]))
        partes = [f"{loc} ({' e '.join(dict.fromkeys(prods))})" for loc, prods in por_local.items()]
        return partes[0] if len(partes) == 1 else ", ".join(partes[:-1]) + " e " + partes[-1]

    def _heuristico(self, noticia: dict, ops: list[dict]) -> dict:
        top = ops[0]
        locais = self._locais(ops)
        un = sum((o["valor_em_jogo_un"] or 0) for o in ops)
        tipo, regra = top["tipo_recomendacao"], top.get("regra_aplicada") or ""

        if regra == "noticia_negativa_concorrente" and top["natureza"] == "risco":
            acao = (f"Aproveitar a notícia desfavorável ao concorrente para recuperar share em {locais}: "
                    f"retomar visitas aos prescritores que migraram e reforçar o diferencial do produto.")
        elif regra == "noticia_negativa_concorrente":
            acao = (f"Ocupar o espaço aberto pela notícia desfavorável ao concorrente em {locais}: "
                    f"ampliar a cobertura de prescritores nas próximas semanas.")
        elif regra == "noticia_positiva_concorrente":
            acao = (f"Reforçar a defesa da posição em {locais}: visitas de manutenção aos principais "
                    f"prescritores e argumentação frente ao avanço do concorrente.")
        elif regra == "noticia_negativa_produto":
            acao = (f"Preparar resposta técnica para a notícia junto aos prescritores de {locais} "
                    f"e monitorar a demanda semanalmente.")
        elif regra == "noticia_ruptura_ou_abastecimento" or tipo == "ANTECIPAR_PICO_SAZONAL":
            acao = f"Garantir abastecimento e antecipar visitas em {locais} antes do pico do mês."
        elif top["natureza"] == "risco":
            acao = (f"Proteger a posição ({ROTULO_TIPO.get(tipo, tipo)}) em {locais}: reforçar visitas "
                    f"a prescritores e acompanhar os concorrentes nas próximas semanas.")
        else:
            acao = (f"Direcionar esforço para {ROTULO_TIPO.get(tipo, tipo)} em {locais}, usando a "
                    f"notícia como argumento junto aos prescritores.")

        rotulo = ROTULO_TIPO.get(tipo, tipo)
        contexto = (f"Notícia de impacto {noticia['nivel_impacto']} e sentimento {noticia['sentimento']} "
                    f"sobre {noticia['escopo']}")
        if regra:
            justificativa = (f"{contexto}: pelas regras do Agente 3, {top['efeito'].rstrip('.')} "
                             f"a recomendação '{rotulo}' nestes pontos de venda.")
        else:
            justificativa = (f"{contexto}: não altera diretamente a recomendação '{rotulo}' do "
                             f"Agente 3, mas serve de contexto para a ação nestes pontos de venda.")
        return {
            "acao_sugerida": acao,
            "justificativa": justificativa,
            "urgencia": _urgencia(top, noticia),
            "impacto_esperado": f"~{un:,.0f} unidades em jogo no próximo mês".replace(",", "."),
            "modo": "heuristico",
        }

    def _llm(self, noticia: dict, ops: list[dict]) -> dict:
        entrada = {
            "noticia": {k: noticia.get(k) for k in ("titulo", "descricao", "nivel_impacto", "sentimento", "escopo")},
            "oportunidades": [{
                "produto": self._p(o["id_produto"]), "local": self._o(o["id_outlet"]),
                "tipo": o["tipo_recomendacao"], "natureza": o["natureza"],
                "unidades_em_jogo": o["valor_em_jogo_un"], "acao_dados": o["acao_agente3"],
                "efeito_da_noticia": o["efeito"]} for o in ops],
        }
        r = self.client.chat.completions.create(
            model=self.modelo, temperature=0.3, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": PROMPT_LLM},
                      {"role": "user", "content": json.dumps(entrada, ensure_ascii=False)}])
        d = json.loads(r.choices[0].message.content)
        urg = str(d.get("urgencia", "media")).lower()
        return {"acao_sugerida": d.get("acao_sugerida", ""), "justificativa": d.get("justificativa", ""),
                "urgencia": urg if urg in URGENCIAS else "media",
                "impacto_esperado": d.get("impacto_esperado", ""), "modo": "llm"}

    # ------------------------------------------------------------------
    def gerar(self, saida_agente4: dict, n_noticias: int = 4, n_oportunidades: int = 3) -> dict:
        cat = {n["id_noticia"]: n for n in saida_agente4["noticias"]}
        setores = []
        for s in saida_agente4["setores"]:
            por_noticia: dict[str, list] = {}
            for a in s["achados"]:                       # ja ordenados por score
                por_noticia.setdefault(a["id_noticia"], []).append(a)
            # ranking guloso com diversidade: cada noticia ja escolhida do mesmo
            # escopo reduz a nota em 30%, e do mesmo tema em 15% - evita 4
            # noticias sobre o mesmo assunto quando ha alternativas relevantes
            restantes = dict(por_noticia)
            ranking, escopos, temas = [], {}, {}
            while restantes and len(ranking) < n_noticias:
                def nota(kv):
                    n = cat[kv[0]]
                    return (kv[1][0]["score"] * 0.70 ** escopos.get(n["escopo"], 0)
                            * 0.85 ** temas.get(n.get("categoria"), 0))
                idn, achs = max(restantes.items(), key=nota)
                ranking.append((idn, achs))
                n = cat[idn]
                escopos[n["escopo"]] = escopos.get(n["escopo"], 0) + 1
                temas[n.get("categoria")] = temas.get(n.get("categoria"), 0) + 1
                del restantes[idn]

            destaques = []
            usados: set = set()          # evita repetir a mesma oportunidade em noticias diferentes
            for pos, (idn, achs) in enumerate(ranking, start=1):
                # oportunidades distintas (uma por alocacao x produto), maior score primeiro;
                # so reaproveita item ja exibido se a noticia nao tiver outro
                ordem = sorted(achs, key=lambda a: -a["score"])
                vistos, ops = set(), []
                for permitir_repeticao in (False, True):
                    for a in ordem:
                        k = (a["id_alocacao"], a["id_produto"])
                        if k in vistos or (k in usados and not permitir_repeticao):
                            continue
                        vistos.add(k)
                        ops.append(a)
                        if len(ops) == n_oportunidades:
                            break
                    if len(ops) == n_oportunidades:
                        break
                usados |= vistos
                ops.sort(key=lambda a: -a["score"])
                noticia = cat[idn]
                try:
                    rec = self._llm(noticia, ops) if self.client else self._heuristico(noticia, ops)
                except Exception:               # noqa: BLE001
                    rec = self._heuristico(noticia, ops)
                destaques.append({
                    "posicao": pos,
                    "score": achs[0]["score"],
                    "noticia": noticia,
                    "oportunidades": [{
                        "id_alocacao": o["id_alocacao"], "id_outlet": o["id_outlet"],
                        "id_produto": o["id_produto"], "produto": self._p(o["id_produto"]),
                        "local": self._o(o["id_outlet"]),
                        "tipo_recomendacao": o["tipo_recomendacao"], "natureza": o["natureza"],
                        "categoria": o["categoria"], "valor_em_jogo_un": o["valor_em_jogo_un"],
                        "efeito": o["efeito"], "score": o["score"],
                    } for o in ops],
                    "recomendacao": rec,
                })
            destaques.sort(key=lambda d: -d["score"])          # exibicao: mais impactante primeiro
            for i, d in enumerate(destaques, start=1):
                d["posicao"] = i
            setores.append({"id_setor": s["id_setor"], "id_uf": s["id_uf"], "destaques": destaques})

        return {
            "schema_version": VERSAO_SCHEMA,
            "agente": AGENTE,
            "gerado_em": datetime.now().isoformat(timespec="seconds"),
            "periodo_previsto": saida_agente4["periodo_previsto"],
            "setores": setores,
        }


def validar_saida(saida: dict) -> list[str]:
    erros = [f"campo ausente: {k}" for k in ("schema_version", "agente", "setores") if k not in saida]
    for s in saida.get("setores", []):
        for d in s["destaques"]:
            if d["recomendacao"]["urgencia"] not in URGENCIAS:
                erros.append(f"setor {s['id_setor']}: urgencia invalida")
            if not d["oportunidades"]:
                erros.append(f"setor {s['id_setor']}: destaque sem oportunidade")
    return erros[:50]


def salvar_json(saida: dict, caminho: str | Path) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(saida, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return caminho
