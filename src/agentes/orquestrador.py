"""***************************************************************************************
SCRIPT    : orquestrador.py
OBJETIVO  : Agente Orquestrador - nucleo de coordenacao (secao 3.3 do TCC1).

            Implementa o padrao orquestrador-executor sobre o pipeline que
            existe no projeto: planeja a ordem dos agentes, delega, mantem um
            ESTADO COMPARTILHADO entre eles, valida o contrato de cada saida
            antes de passar adiante e consolida tudo para o painel.

FLUXOS (cada um e uma "missao" do orquestrador):

    corpus    Preparo do corpus anotado -> Agente de Desempate (so
              discordancia total) -> Agente 1 (treino com validacao
              cruzada, predicoes out-of-fold) -> Agente 2 (validacao das
              predicoes out-of-fold) -> metricas dos Agentes 1, 2 e Desempate.
              Roda uma vez, ou quando o corpus muda.

    noticias  ETL das APIs (GNews, Guardian) -> preparo -> Agente 1
              (classifica as noticias novas) -> Agente 2 (regras + k-NN)
              -> Agentes 4 e 5. E o botao "Gerar mais noticias".

    semana    executar_semana.py (proxima semana, sem trava de data) ->
              Agente 3 (retreino na 1a semana do mes) -> Agentes 4 e 5.
              E o botao "Gerar nova semana".

    cruzar    so Agentes 4 e 5 sobre o que ja existe.

MOTORES E SUBSTITUTOS: o orquestrador escolhe, a cada execucao, o motor
de cada agente conforme o ambiente, e REGISTRA a escolha:

    Agente 1   mBERT fine-tuned (modelos/agente1_mbert)  | Naive Bayes do corpus
    Agente 2   sentence-transformers multilingue          | encoder lexical TF-IDF
    Desempate  LLM local via Ollama (llama3.1)            | REVISAO_MANUAL (caso
                                                          |  sai do treino)
    Agente 3   XGBoost                                    | ridge
    Agente 5   LLM (OpenAI)                               | modelos de texto

ESTADO: data/orquestrador/estado.json (ultima execucao de cada fluxo,
motores usados, contagens e log) e data/orquestrador/painel_insights.json.

AUTOR     : Marcos Arambasic
DATA      : OUT/2026
****************************************************************************************/
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import src.agentes  # noqa: E402,F401  (coloca src/agentes no sys.path)
from _schema import COL_ID, COL_LABEL, COL_TEXTO, normalizar_rotulo, preparar_corpus  # noqa: E402

DIR_DATA = RAIZ / "data"
DIR_AG = DIR_DATA / "agentes"
DIR_ORQ = DIR_DATA / "orquestrador"
CORPUS_PADRAO = DIR_DATA / "estatistica_kappa" / "noticias_classificadas_ampliado.xlsx"
MODELO_MBERT = RAIZ / "modelos" / "agente1_mbert"

# Nomes das colunas de avaliador no corpus do projeto -> nome do contrato
MAPA_AVALIADORES = {
    "Avaliador 1": "Avaliador 1",
    "Avaliador 2": "Avaliador 2",
    "Avaliador 3": "Avaliador 3",
}

MAPA_DIRECAO = {"POSITIVA": "positivo", "NEGATIVA": "negativo"}


def _tem(*modulos: str) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in modulos)


def mbert_disponivel() -> bool:
    return _tem("torch", "transformers", "sklearn", "accelerate")


def encoder_semantico_disponivel() -> bool:
    return _tem("sentence_transformers")


# ============================================================================
# Estado compartilhado
# ============================================================================
@dataclass
class EstadoPipeline:
    corpus_anotado: Any = None          # corpus normalizado (Avaliador 1/2/3 + consolidacao)
    corpus_desempatado: Any = None      # apos o Agente de Desempate
    predicoes_oof: Any = None           # Agente 1 sobre o corpus (out-of-fold)
    noticias_classificadas: Any = None  # Agente 1 sobre noticias novas
    noticias_validadas: Any = None      # Agente 2 (tabela completa, auditavel)
    noticias_aprovadas: Any = None      # Agente 2 -> seguem para o Agente 4
    motores: dict = field(default_factory=dict)
    contagens: dict = field(default_factory=dict)
    log: list = field(default_factory=list)

    def registrar(self, etapa: str, detalhe: str) -> None:
        self.log.append({"timestamp": datetime.now().isoformat(timespec="seconds"),
                         "etapa": etapa, "detalhe": detalhe})


class AgenteOrquestrador:
    """Planeja, delega, valida contratos e consolida."""

    def __init__(self, log: Callable[[str], None] = print, usar_llm: bool = False,
                 forcar_substitutos: bool = False):
        self.estado = EstadoPipeline()
        self._log = log
        self.usar_llm = usar_llm
        self.forcar_substitutos = forcar_substitutos
        self._agente1 = None
        self._agente2 = None

    def _info(self, etapa: str, msg: str) -> None:
        self._log(f"[{etapa}] {msg}")
        self.estado.registrar(etapa, msg)

    # ------------------------------------------------------------------
    # Preparo e Agente de Desempate
    # ------------------------------------------------------------------
    def preparar(self, df_bruto: pd.DataFrame | None = None) -> pd.DataFrame:
        """Normaliza o corpus anotado (doc_id, texto_completo, rotulos canonicos, maioria)."""
        if df_bruto is None:
            df_bruto = pd.read_excel(CORPUS_PADRAO)
        if not any(c.startswith("Avaliador") for c in df_bruto.columns):
            df_bruto = df_bruto.rename(columns=MAPA_AVALIADORES)
        df, _ = preparar_corpus(df_bruto, verbose=False)
        self.estado.corpus_anotado = df
        sit = df["situacao_anotacao"].value_counts().to_dict() if "situacao_anotacao" in df else {}
        self.estado.contagens["anotacao"] = {k: int(v) for k, v in sit.items()}
        self._info("Preparo", f"{len(df)} documentos normalizados | {sit}")
        return df

    def etapa0_desempate(self) -> pd.DataFrame:
        """Agente de Desempate: so atua onde os 3 avaliadores discordam totalmente."""
        from agente_desempate_llm import AgenteDesempateLLM
        from _llm_local import llm_disponivel, modelo_em_uso, motivo_indisponivel

        df = self.estado.corpus_anotado
        if df is None:
            df = self.preparar()
        agente = AgenteDesempateLLM()
        disc = agente.identificar_discordancia_total(df)
        if llm_disponivel():
            motor = f"llm_local_ollama:{modelo_em_uso()}"
        else:
            motor = "revisao_manual"
            self._info("Agente de Desempate", f"LLM local indisponivel -> {motivo_indisponivel()}")
        self.estado.motores["desempate"] = motor
        final = df.copy()
        if not disc.empty:
            resolvidos = agente.resolver(disc)
            rot = resolvidos["classificacao_final"].map(normalizar_rotulo)   # REVISAO_MANUAL -> None
            final.loc[resolvidos.index, COL_LABEL] = rot
            final.loc[resolvidos.index, "justificativa_desempate"] = resolvidos["justificativa_desempate"]
            destino = DIR_AG / "desempate" / "desempate_resultado.csv"
            destino.parent.mkdir(parents=True, exist_ok=True)
            cols = [c for c in ["doc_id", "titulo", "Avaliador 1", "Avaliador 2", "Avaliador 3",
                                "classificacao_final", "justificativa_desempate", "url"] if c in resolvidos]
            resolvidos[cols].to_csv(destino, index=False)
        n_res = int(final.loc[disc.index, COL_LABEL].notna().sum()) if not disc.empty else 0
        self.estado.corpus_desempatado = final
        self.estado.contagens["desempate"] = {"discordancia_total": int(len(disc)),
                                              "resolvidos": n_res,
                                              "pendentes_revisao_manual": int(len(disc) - n_res)}
        self._info("Agente de Desempate",
                   f"{len(disc)} casos de discordancia total | {n_res} resolvidos ({motor})")
        return final

    # ------------------------------------------------------------------
    # Agente 1
    # ------------------------------------------------------------------
    def etapa1_treinar(self, n_folds: int = 5, **kwargs) -> dict:
        """Treina o Agente 1 sobre o corpus desempatado e guarda as predicoes out-of-fold."""
        corpus = self.estado.corpus_desempatado
        if corpus is None:
            corpus = self.etapa0_desempate()
        # O corpus tem a mesma materia coletada mais de uma vez (311 titulos
        # repetidos). Sem remover, a mesma noticia cai no treino de um fold e
        # na validacao de outro e a metrica out-of-fold fica otimista.
        antes = len(corpus)
        corpus = corpus.drop_duplicates(subset=COL_TEXTO).reset_index(drop=True)
        self.estado.contagens["agente1_duplicatas_removidas"] = int(antes - len(corpus))
        destino = DIR_AG / "agente1"
        destino.mkdir(parents=True, exist_ok=True)

        if mbert_disponivel() and not self.forcar_substitutos:
            from agente1_classificador_mbert import AgenteClassificadorImpacto
            ag = AgenteClassificadorImpacto()
            rel = ag.treinar(corpus, n_folds=n_folds, output_dir=str(destino / "treino"),
                             salvar_relatorio=str(destino / "agente1_metricas.json"),
                             dir_relatorio=str(destino / "relatorio"), **kwargs)
            ag.salvar(str(MODELO_MBERT))
            motor = "mbert"
        else:
            from substitutos import ClassificadorNBImpacto
            ag = ClassificadorNBImpacto()
            rel = ag.treinar(corpus, n_folds=n_folds)
            (destino / "agente1_metricas_substituto.json").write_text(
                json.dumps(rel, ensure_ascii=False, indent=2), encoding="utf-8")
            motor = "naive_bayes"
        self._agente1 = ag
        self.estado.predicoes_oof = ag.predicoes_oof_
        ag.predicoes_oof_.drop(columns=[c for c in ("texto_extracao",) if c in ag.predicoes_oof_],
                               errors="ignore").to_csv(destino / f"agente1_oof_{motor}.csv", index=False)
        self.estado.motores["agente1"] = motor
        self._info("Agente 1", f"motor={motor} | F1-macro {rel.get('f1_macro_media', 0):.4f} "
                               f"+/- {rel.get('f1_macro_desvio', 0):.4f} ({rel.get('n_folds')} folds)")
        return rel

    def carregar_agente1(self):
        """Modelo de producao: mBERT salvo, se existir; senao o substituto treinado no corpus."""
        if self._agente1 is not None:
            return self._agente1
        if MODELO_MBERT.exists() and mbert_disponivel() and not self.forcar_substitutos:
            from agente1_classificador_mbert import AgenteClassificadorImpacto
            self._agente1 = AgenteClassificadorImpacto.carregar(str(MODELO_MBERT))
            self.estado.motores["agente1"] = "mbert"
        else:
            if self.estado.corpus_desempatado is None:
                self.etapa0_desempate()
            from substitutos import ClassificadorNBImpacto
            self._agente1 = ClassificadorNBImpacto()
            self._agente1.treinar(self.estado.corpus_desempatado)
            self.estado.motores["agente1"] = "naive_bayes"
        return self._agente1

    # ------------------------------------------------------------------
    # Agente 2
    # ------------------------------------------------------------------
    def _novo_agente2(self, textos_extra: list[str]):
        from agente2_validador_ajuste import AgenteValidadorAjuste
        corpus = self.estado.corpus_desempatado
        cache = str(DIR_AG / "agente2" / "cache_embeddings")
        if encoder_semantico_disponivel() and not self.forcar_substitutos:
            ag = AgenteValidadorAjuste(cache_dir=cache)
            motor = "sentence_transformers"
        else:
            from substitutos import EncoderLexical
            enc = EncoderLexical().ajustar_idf(corpus[COL_TEXTO].astype(str).tolist() + textos_extra)
            ag = AgenteValidadorAjuste(modelo_embedding="encoder_lexical_tfidf_hashing",
                                       encoder=enc, cache_dir=None)
            motor = "encoder_lexical"
        rotulados = corpus.dropna(subset=[COL_LABEL])
        ag.indexar_eventos_historicos(rotulados, coluna_texto=COL_TEXTO, coluna_impacto=COL_LABEL)
        ag.calibrar_limiar()
        self.estado.motores["agente2"] = motor
        return ag

    def etapa2_validar(self, df_classificado: pd.DataFrame, nome: str = "noticias") -> pd.DataFrame:
        from agente2_validador_ajuste import AgenteValidadorAjuste
        if self.estado.corpus_desempatado is None:
            self.etapa0_desempate()
        self._agente2 = self._novo_agente2(df_classificado[COL_TEXTO].astype(str).tolist())
        destino = DIR_AG / "agente2"
        destino.mkdir(parents=True, exist_ok=True)
        val = self._agente2.validar(df_classificado, coluna_texto=COL_TEXTO,
                                    salvar_relatorio=str(destino / f"agente2_relatorio_{nome}.json"))
        self.estado.noticias_validadas = val
        self.estado.noticias_aprovadas = AgenteValidadorAjuste.aprovados(val)
        r = self._agente2.relatorio
        self.estado.contagens[f"agente2_{nome}"] = r.to_dict()
        self._info("Agente 2", f"motor={self.estado.motores['agente2']} | {r.total_entrada} entrada | "
                               f"{r.aprovados} aprovadas | {r.descartados_baixo} baixo | "
                               f"{r.duplicados} duplicatas | {r.quarentena} quarentena | "
                               f"{r.sem_ancora} sem ancora | {r.promovidos_por_regra} promovidas")
        return val

    # ------------------------------------------------------------------
    # Missao: corpus (avaliacao dos Agentes 1, 2 e Desempate)
    # ------------------------------------------------------------------
    def missao_corpus(self, n_folds: int = 5, **kwargs) -> dict:
        self.preparar()
        self.etapa0_desempate()
        rel1 = self.etapa1_treinar(n_folds=n_folds, **kwargs)
        oof = self.estado.predicoes_oof
        val = self.etapa2_validar(oof, nome="corpus_oof")
        metricas = {
            "gerado_em": datetime.now().isoformat(timespec="seconds"),
            "motores": dict(self.estado.motores),
            "anotacao": self.estado.contagens.get("anotacao"),
            "desempate": self.estado.contagens.get("desempate"),
            "agente1": rel1,
            "agente2": avaliar_agente2(val),
        }
        texto = json.dumps(metricas, ensure_ascii=False, indent=2, default=str)
        (DIR_AG / "metricas_agentes_1_2_desempate.json").write_text(texto, encoding="utf-8")
        # copia por motor: o artigo compara mBERT x substituto sem que uma
        # execucao sobrescreva a outra
        motor = self.estado.motores.get("agente1", "desconhecido")
        (DIR_AG / f"metricas_agentes_1_2_desempate_{motor}.json").write_text(texto, encoding="utf-8")
        self._salvar_estado("corpus")
        return metricas

    # ------------------------------------------------------------------
    # Missao: noticias
    # ------------------------------------------------------------------
    def missao_noticias(self, fazer_extracao: bool = True, reclassificar: bool = False) -> dict:
        from src.noticias import pipeline_noticias as pn
        inicio = datetime.now()
        self._info("ETL", "extracao e transformacao das fontes")
        status = pn.extrair_e_transformar(self._log, fazer_extracao)
        brutas = pn.consolidar()

        base_ant = pd.read_csv(pn.BASE_CLASSIFICADA) if (pn.BASE_CLASSIFICADA.exists() and not reclassificar) \
            else pd.DataFrame()
        from src.noticias.classificacao import id_noticia
        brutas["id_noticia"] = brutas["url"].map(id_noticia)
        ja = set(base_ant["id_noticia"]) if not base_ant.empty else set()
        novas = brutas[~brutas["id_noticia"].isin(ja)].reset_index(drop=True)
        self._info("ETL", f"{len(brutas)} noticias consolidadas | {len(novas)} novas")

        if not novas.empty:
            self.preparar()
            self.etapa0_desempate()
            prep, _ = preparar_corpus(novas, verbose=False)
            a1 = self.carregar_agente1()
            classif = a1.classificar_dataframe(prep, coluna_texto=COL_TEXTO)
            # noticias que ja estao no corpus anotado: vale o rotulo humano consolidado
            humano = self.estado.corpus_desempatado.dropna(subset=[COL_LABEL]).drop_duplicates("url") \
                .set_index("url")[COL_LABEL]
            no_corpus = classif["url"].isin(humano.index)
            classif.loc[no_corpus, "classe_predita"] = classif.loc[no_corpus, "url"].map(humano)
            classif["origem_classificacao"] = self.estado.motores["agente1"]
            classif.loc[no_corpus, "origem_classificacao"] = "corpus_anotado"
            self.estado.noticias_classificadas = classif
            self._info("Agente 1", f"motor={self.estado.motores['agente1']} | {len(classif)} classificadas "
                                   f"({int(no_corpus.sum())} com rotulo humano)")
            val = self.etapa2_validar(classif, nome="noticias")
            nova_base = para_contrato_agente4(val, self.estado.motores)
            base = pd.concat([base_ant, nova_base], ignore_index=True) if not base_ant.empty else nova_base
            base.to_csv(pn.BASE_CLASSIFICADA, index=False)
        else:
            base = base_ant

        res = pn.cruzar_e_recomendar(self.usar_llm, self._log)
        self.estado.motores["agente5"] = "llm" if self.usar_llm else "heuristico"
        registro = {"executado_em": inicio.isoformat(timespec="seconds"),
                    "duracao_s": round((datetime.now() - inicio).total_seconds(), 1),
                    "fontes": status, "noticias_na_base": int(len(base)),
                    "motores": dict(self.estado.motores), **res}
        hist_p = pn.DIR_NOTICIAS / "_execucoes.json"
        hist = json.loads(hist_p.read_text(encoding="utf-8")) if hist_p.exists() else []
        hist.append(registro)
        hist_p.write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")
        self._salvar_estado("noticias", registro)
        return registro

    # ------------------------------------------------------------------
    # Missao: semana
    # ------------------------------------------------------------------
    def _rodar(self, cmd: list[str]) -> int:
        proc = subprocess.Popen(cmd, cwd=RAIZ, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace",
                                env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"})
        for linha in proc.stdout:          # type: ignore[union-attr]
            if linha.strip():
                self._log(linha.rstrip())
        return proc.wait()

    def missao_semana(self, ano: int | None = None, mes: int | None = None,
                      semana: int | None = None) -> dict:
        cmd = [sys.executable, "scripts/executar_semana.py"]
        if ano and mes and semana:
            cmd += ["--ano", str(ano), "--mes", str(mes), "--semana", str(semana)]
        self._info("Pipeline semanal", " ".join(cmd[1:]))
        if self._rodar(cmd) != 0:
            raise RuntimeError("o pipeline semanal falhou")

        u = _ultima_semana()
        if u and u["nr_semana"] == 1:
            self.etapa3_oportunidades()
        if (DIR_DATA / "noticias" / "noticias_classificadas.csv").exists():
            self.etapa4_5_cruzar()
        registro = {"semana": u and f"{u['ano']}-{u['mes']:02d} S{u['nr_semana']}"}
        self._salvar_estado("semana", registro)
        return registro

    def etapa3_oportunidades(self) -> None:
        self._info("Agente 3", "retreino com os KPIs do mes que fechou")
        if self._rodar([sys.executable, "-m", "src.agentes.agente3_oportunidades_mercado"]) != 0:
            raise RuntimeError("o Agente 3 falhou")
        s = json.loads((DIR_AG / "agente3_saida.json").read_text(encoding="utf-8"))
        self.estado.motores["agente3"] = s["modelo"]["algoritmo"]

    def etapa4_5_cruzar(self) -> dict:
        from src.noticias import pipeline_noticias as pn
        res = pn.cruzar_e_recomendar(self.usar_llm, self._log)
        self.estado.motores["agente5"] = "llm" if self.usar_llm else "heuristico"
        self._info("Agentes 4 e 5", f"{res['achados']} achados | {res['noticias_validas']} noticias validas")
        return res

    # ------------------------------------------------------------------
    # Consolidacao
    # ------------------------------------------------------------------
    def _salvar_estado(self, missao: str, extra: dict | None = None) -> None:
        DIR_ORQ.mkdir(parents=True, exist_ok=True)
        p = DIR_ORQ / "estado.json"
        est = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        est[missao] = {"executado_em": datetime.now().isoformat(timespec="seconds"),
                       "motores": dict(self.estado.motores),
                       "contagens": self.estado.contagens, **(extra or {})}
        est.setdefault("historico_log", []).extend(self.estado.log)
        est["historico_log"] = est["historico_log"][-500:]
        p.write_text(json.dumps(est, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        self.exportar_painel()

    def exportar_painel(self) -> Path:
        def _j(nome):
            q = DIR_AG / nome
            return json.loads(q.read_text(encoding="utf-8")) if q.exists() else None
        a3, a4, a5 = _j("agente3_saida.json"), _j("agente4_saida.json"), _j("agente5_saida.json")
        painel = {
            "gerado_em": datetime.now().isoformat(timespec="seconds"),
            "motores": dict(self.estado.motores),
            "resumo": {
                "recomendacoes_agente3": a3["resumo"]["n_itens"] if a3 else 0,
                "achados_agente4": sum(len(s["achados"]) for s in a4["setores"]) if a4 else 0,
                "destaques_agente5": sum(len(s["destaques"]) for s in a5["setores"]) if a5 else 0,
            },
            "log_execucao": self.estado.log,
        }
        DIR_ORQ.mkdir(parents=True, exist_ok=True)
        p = DIR_ORQ / "painel_insights.json"
        p.write_text(json.dumps(painel, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return p


# ============================================================================
# Adaptadores
# ============================================================================
def para_contrato_agente4(val: pd.DataFrame, motores: dict) -> pd.DataFrame:
    """
    Converte a saida do Agente 2 no contrato de noticia que o Agente 4 consome
    (mesmas colunas da tabela `noticia` do banco + colunas de auditoria).
    """
    from src.noticias.classificacao import (carregar_mapa, categoria, id_noticia, normalizar,
                                            sentimento_heuristico, vincular_entidades)
    mapa = carregar_mapa()
    linhas = []
    for r in val.to_dict("records"):
        texto = normalizar(r.get("texto_extracao") or r.get(COL_TEXTO, ""))
        escopo, grupos, termos = vincular_entidades(texto, mapa)
        if escopo == "nenhum" and r.get("escopo_relevancia") in ("SISTEMICO", "ESPECIFICO"):
            escopo = "mercado"
        sent_lex, score = sentimento_heuristico(normalizar(f"{r.get('titulo', '')} {r.get('descricao', '')}"))
        sentimento = MAPA_DIRECAO.get(r.get("direcao_impacto"), sent_lex)
        classe = str(r.get("classe_final_validada") or r.get("classe_predita") or "").strip()
        linhas.append({
            "id_noticia": r.get("id_noticia") or id_noticia(r["url"]),
            "doc_id": r.get(COL_ID),
            "data_publicacao": r.get("data_publicacao"),
            "fonte": r.get("fonte_nome", "") or "",
            "titulo": r.get("titulo", ""), "descricao": r.get("descricao", ""), "url": r.get("url"),
            "palavra_chave": r.get("palavra_chave", ""),
            "categoria": categoria(texto),
            "nivel_impacto": {"Alto": "alto", "Medio": "medio", "Baixo": "baixo"}.get(classe, "baixo"),
            "sentimento": sentimento, "score_sentimento": score,
            "escopo": escopo, "grupos_alvo": "|".join(grupos),
            "termos_encontrados": "|".join(sorted(set(termos)))[:300],
            "id_mercado": mapa.get("id_mercado", 1) if escopo != "nenhum" else None,
            "origem_classificacao": r.get("origem_classificacao"),
            "classe_agente1": r.get("classe_predita"), "score_agente1": r.get("score"),
            "score_final_agente2": r.get("score_final"),
            "status_validacao": r.get("status_validacao"), "acoes_regras": r.get("acoes_regras"),
            "escopo_relevancia": r.get("escopo_relevancia"), "evento_principal": r.get("evento_principal"),
            "direcao_impacto": r.get("direcao_impacto"),
            "principios_ativos": r.get("principios_ativos"),
            "motor_agente1": motores.get("agente1"), "motor_agente2": motores.get("agente2"),
            "processada_em": datetime.now().isoformat(timespec="seconds"),
        })
    return pd.DataFrame(linhas)


def avaliar_agente2(val: pd.DataFrame) -> dict:
    """
    Qualidade do Agente 2 sobre o corpus (entrada = predicoes out-of-fold do Agente 1,
    referencia = classificacao_final consolidada/desempatada).
    """
    from agente2_validador_ajuste import STATUS_APROVADO
    from substitutos import metricas_classificacao
    v = val.dropna(subset=[COL_LABEL]).copy()
    ref_manter = v[COL_LABEL].isin(["Alto", "Medio"])
    aprovado = v["status_validacao"] == STATUS_APROVADO
    tp = int((aprovado & ref_manter).sum())
    fp = int((aprovado & ~ref_manter).sum())
    fn = int((~aprovado & ref_manter).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    alto = v[COL_LABEL] == "Alto"
    promovidos = v["acoes_regras"].fillna("").str.contains("R2_promovido")
    return {
        "n_avaliado": int(len(v)),
        "status": {k: int(n) for k, n in v["status_validacao"].value_counts().items()},
        "aprovacao_vs_referencia": {
            "precisao": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
            "recall_alto": round(float((aprovado & alto).sum() / max(alto.sum(), 1)), 4),
        },
        "promocao_por_evento_critico": {
            "n": int(promovidos.sum()),
            "acerto_ref_alto": round(float((promovidos & alto).sum() / max(promovidos.sum(), 1)), 4),
        },
        "classe_agente1_vs_ref": metricas_classificacao(v[COL_LABEL], v["classe_predita"],
                                                        ["Alto", "Medio", "Baixo"])["resumo"],
        "classe_validada_vs_ref": metricas_classificacao(v[COL_LABEL], v["classe_final_validada"],
                                                         ["Alto", "Medio", "Baixo"])["resumo"],
        "classe_knn_vs_ref": metricas_classificacao(v[COL_LABEL], v["classe_knn"],
                                                    ["Alto", "Medio", "Baixo"])["resumo"],
        "divergencias": int(v["divergencia"].sum()),
        "escopo": {k: int(n) for k, n in v["escopo_relevancia"].value_counts().items()},
    }


def _ultima_semana() -> dict | None:
    spec = importlib.util.spec_from_file_location("_es", RAIZ / "scripts" / "executar_semana.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ultima_semana_calculada(DIR_DATA / "sintetico", DIR_DATA / "semanal")
