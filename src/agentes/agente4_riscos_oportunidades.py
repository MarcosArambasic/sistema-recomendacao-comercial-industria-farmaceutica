"""***************************************************************************************
SCRIPT    : agente4_riscos_oportunidades.py
OBJETIVO  : Agente 4 - Identificador de Riscos e Oportunidades.

            Cruza as noticias de impacto ALTO e MEDIO (saida dos Agentes 1 e 2)
            com as recomendacoes de mercado do Agente 3, setor a setor, e
            produz uma lista de ACHADOS: "esta noticia afeta esta oportunidade
            (ou risco) nesta alocacao, neste sentido, com esta intensidade".

COMO UMA NOTICIA SE LIGA A UM ITEM DO AGENTE 3 (chaves_noticia do contrato v2.0):

    escopo da noticia    liga com o item quando...                  relevancia
    -----------------    ---------------------------------------   ----------
    proprio              o grupo citado = grupo do produto do item   1,0
    concorrente          sempre (o item lista os concorrentes fortes) 0,5 + share
                                                                     dos concorrentes
    mercado              sempre (mesmo id_mercado)                   0,5

COMO A NOTICIA MUDA O ITEM: pelas `regras_de_sensibilidade` que o proprio
Agente 3 publica em `instrucoes_agente4`. A frase da regra vira um
multiplicador (amplifica/confirma/agrava > 1; atenua < 1). Assim o criterio
fica num lugar so - quem define como cada tipo de recomendacao reage a uma
noticia e o Agente 3, e este agente apenas aplica.

SCORE DO ACHADO:
    score = peso_impacto x relevancia x multiplicador x valor_relativo x recencia
      peso_impacto   alto 1,0 | medio 0,6
      valor_relativo 0,5 + 0,5 x (unidades em jogo / maior valor do setor)
      recencia       exp(-dias/180), com piso de 0,3

SAIDA: data/agentes/agente4_saida.json (contrato versionado, por setor).

AUTOR     : Marcos Arambasic
DATA      : OUT/2026
****************************************************************************************/
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

AGENTE = "agente4_riscos_oportunidades"
VERSAO_SCHEMA = "1.0"

PESO_IMPACTO = {"alto": 1.0, "medio": 0.6}
MEIA_VIDA_DIAS = 180
PISO_RECENCIA = 0.3
MAX_NOTICIAS_POR_ESCOPO = 10    # noticias mais relevantes guardadas por setor, por escopo
MAX_ACHADOS_POR_NOTICIA = 8     # itens do Agente 3 guardados por noticia

# frase da regra -> multiplicador (ordem importa: a primeira que casar vale)
MULTIPLICADORES = [
    ("prioridade alta", 1.5), ("confirma", 1.4), ("agrava", 1.4), ("eleva", 1.4),
    ("aumenta a urgencia", 1.3), ("abre janela", 1.3), ("amplifica", 1.3),
    ("atenua", 0.7), ("reavaliar", 1.0),
]
TERMOS_ABASTECIMENTO = ("falta", "shortage", "escassez", "desabastec", "ruptura", "ifa", "insumo")


def _multiplicador(frase: str) -> float:
    f = frase.lower()
    for termo, m in MULTIPLICADORES:
        if termo in f:
            return m
    return 1.0


def _chave_regra(noticia: dict, tipo: str) -> str | None:
    """Traduz (escopo, sentimento, tema) da noticia na chave da regra do Agente 3."""
    esc, sent = noticia["escopo"], noticia["sentimento"]
    texto = f"{noticia.get('titulo', '')} {noticia.get('descricao', '')}".lower()
    if tipo == "ANTECIPAR_PICO_SAZONAL" and any(t in texto for t in TERMOS_ABASTECIMENTO):
        return "noticia_ruptura_ou_abastecimento"
    if tipo == "MANTER_POSICAO" and noticia["nivel_impacto"] == "alto":
        return "qualquer_noticia_alto_impacto"
    if esc == "concorrente":
        return {"positivo": "noticia_positiva_concorrente",
                "negativo": "noticia_negativa_concorrente"}.get(sent)
    if sent == "positivo":
        return "noticia_positiva_produto_ou_mercado" if tipo != "ANTECIPAR_PICO_SAZONAL" \
            else "noticia_positiva_mercado"
    if sent == "negativo":
        return "noticia_negativa_produto"
    return None


def _recencia(data_pub, data_ref: pd.Timestamp) -> float:
    d = pd.to_datetime(data_pub, errors="coerce", utc=True)
    if pd.isna(d):
        return PISO_RECENCIA
    dias = max(0, (data_ref - d.tz_convert(None)).days)
    return max(PISO_RECENCIA, math.exp(-dias / MEIA_VIDA_DIAS))


class AgenteRiscosOportunidades:

    def __init__(self, saida_agente3: dict):
        self.a3 = saida_agente3
        self.regras = saida_agente3["instrucoes_agente4"]["regras_de_sensibilidade"]
        self.filtro = set(saida_agente3["instrucoes_agente4"]["filtro_noticias"]["nivel_impacto"])

    # ------------------------------------------------------------------
    def _relevancia(self, noticia: dict, item: dict) -> float:
        esc = noticia["escopo"]
        if esc == "proprio":
            grupos = [g for g in str(noticia.get("grupos_alvo") or "").split("|") if g]
            return 1.0 if item.get("grupo") in grupos else 0.0
        if esc == "concorrente":
            soma = sum(c.get("mrkt_share", 0) for c in item.get("concorrentes_relevantes", []))
            return min(1.0, 0.5 + soma)
        if esc == "mercado":
            return 0.5 if noticia.get("id_mercado") in (None, item.get("id_mercado")) else 0.0
        return 0.0

    def cruzar(self, noticias: pd.DataFrame, data_referencia: str | None = None) -> dict:
        ref = pd.Timestamp(data_referencia) if data_referencia else \
            pd.Timestamp(self.a3["periodo_previsto"] + "-01")
        nots = noticias[noticias["nivel_impacto"].isin(self.filtro)].to_dict("records")

        setores = []
        for s in self.a3["setores"]:
            itens = s["itens"]
            vmax = max((i["previsao"]["valor_em_jogo_un"] or 0) for i in itens) if itens else 0
            achados = []
            for n in nots:
                rec = _recencia(n.get("data_publicacao"), ref)
                for it in itens:
                    rel = self._relevancia(n, it)
                    if rel <= 0:
                        continue
                    chave = _chave_regra(n, it["tipo_recomendacao"])
                    frase = self.regras.get(it["tipo_recomendacao"], {}).get(chave, "") if chave else ""
                    mult = _multiplicador(frase) if frase else 1.0
                    v = (it["previsao"]["valor_em_jogo_un"] or 0) / vmax if vmax else 0
                    score = PESO_IMPACTO[n["nivel_impacto"]] * rel * mult * (0.5 + 0.5 * v) * rec

                    if it["natureza"] == "risco":
                        categoria = "risco_imediato" if (n["nivel_impacto"] == "alto" and mult > 1) \
                            else "risco_medio_prazo"
                    else:
                        categoria = "oportunidade"

                    achados.append({
                        "id_achado": f"{n['id_noticia']}|{it['id_item']}",
                        "id_noticia": n["id_noticia"],
                        "id_item": it["id_item"],
                        "id_alocacao": it["id_alocacao"],
                        "id_outlet": it["id_outlet"],
                        "id_produto": it["id_produto"],
                        "grupo": it.get("grupo"),
                        "tipo_recomendacao": it["tipo_recomendacao"],
                        "natureza": it["natureza"],
                        "prioridade_agente3": it["prioridade"],
                        "valor_em_jogo_un": it["previsao"]["valor_em_jogo_un"],
                        "acao_agente3": it["acao_sugerida"],
                        "regra_aplicada": chave,
                        "efeito": frase or "contexto (sem regra especifica para este sentimento)",
                        "multiplicador": mult,
                        "relevancia": round(rel, 4),
                        "recencia": round(rec, 4),
                        "score": round(score, 6),
                        "categoria": categoria,
                        "justificativa": (
                            f"Noticia de impacto {n['nivel_impacto']} ({n['sentimento']}, escopo "
                            f"{n['escopo']}) sobre item {it['tipo_recomendacao']} com "
                            f"{(it['previsao']['valor_em_jogo_un'] or 0):,.0f} un em jogo"
                            + (f": {frase}." if frase else ".")
                        ),
                    })
            # guarda as N noticias mais relevantes do setor e, de cada uma, os M
            # itens mais afetados (um corte global por score deixaria so 2 ou 3
            # noticias, cada uma repetida em dezenas de itens)
            por_noticia: dict[str, list] = {}
            for a in sorted(achados, key=lambda a: -a["score"]):
                por_noticia.setdefault(a["id_noticia"], []).append(a)
            # o corte e feito por escopo (proprio/concorrente/mercado) para que um
            # tipo de noticia nao elimine os demais do cardapio do Agente 5
            escopo_de = {n["id_noticia"]: n["escopo"] for n in nots}
            top, por_escopo = [], {}
            for lista in sorted(por_noticia.values(), key=lambda l: -l[0]["score"]):
                e = escopo_de[lista[0]["id_noticia"]]
                if por_escopo.get(e, 0) < MAX_NOTICIAS_POR_ESCOPO:
                    top.append(lista)
                    por_escopo[e] = por_escopo.get(e, 0) + 1
            mantidos = [a for l in top for a in l[:MAX_ACHADOS_POR_NOTICIA]]
            mantidos.sort(key=lambda a: -a["score"])
            setores.append({"id_setor": s["id_setor"], "id_uf": s["id_uf"],
                            "n_achados": len(achados),
                            "achados": mantidos})

        noticias_usadas = {a["id_noticia"] for s in setores for a in s["achados"]}
        cols = ["id_noticia", "titulo", "descricao", "url", "fonte", "data_publicacao",
                "nivel_impacto", "sentimento", "score_sentimento", "escopo", "grupos_alvo",
                "categoria", "origem_classificacao"]
        cat = [{k: (None if pd.isna(n.get(k)) else (str(n.get(k)) if k == "data_publicacao" else n.get(k)))
                for k in cols if k in n}
               for n in nots if n["id_noticia"] in noticias_usadas]

        return {
            "schema_version": VERSAO_SCHEMA,
            "agente": AGENTE,
            "gerado_em": datetime.now().isoformat(timespec="seconds"),
            "periodo_previsto": self.a3["periodo_previsto"],
            "data_referencia": f"{ref:%Y-%m-%d}",
            "n_noticias_avaliadas": len(nots),
            "noticias": cat,
            "setores": setores,
        }


def validar_saida(saida: dict) -> list[str]:
    erros = []
    for k in ("schema_version", "agente", "periodo_previsto", "noticias", "setores"):
        if k not in saida:
            erros.append(f"campo ausente: {k}")
    if erros:
        return erros
    ids = {n["id_noticia"] for n in saida["noticias"]}
    for s in saida["setores"]:
        for a in s["achados"]:
            if a["id_noticia"] not in ids:
                erros.append(f"{a['id_achado']}: noticia nao catalogada")
            if a["categoria"] not in ("oportunidade", "risco_imediato", "risco_medio_prazo"):
                erros.append(f"{a['id_achado']}: categoria invalida")
            if a["score"] < 0:
                erros.append(f"{a['id_achado']}: score negativo")
    return erros[:50]


def salvar_json(saida: dict, caminho: str | Path) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(saida, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return caminho
