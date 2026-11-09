"""***************************************************************************************
SCRIPT    : pipeline_noticias.py
OBJETIVO  : Pipeline completo de noticias, do ETL ate as recomendacoes:

            1. EXTRACAO      extract_gnews.py + extract_guardian_news.py
                             (so roda a fonte cuja chave de API esta no .env)
            2. TRANSFORMACAO transformation_and_load_gnews/guardian.py
            3. CONSOLIDACAO  junta as duas fontes, remove duplicadas por URL
            4. AGENTES 1 e 2 classifica impacto/sentimento/escopo e filtra
                             ALTO e MEDIO (src/noticias/classificacao.py).
                             Noticias ja classificadas em execucoes anteriores
                             nao sao reclassificadas.
            5. AGENTE 4      cruza com data/agentes/agente3_saida.json
            6. AGENTE 5      gera as recomendacoes por setor

SAIDAS:
    data/noticias/noticias_classificadas.csv   base acumulada (formato da tabela `noticia`)
    data/agentes/agente4_saida.json
    data/agentes/agente5_saida.json
    data/noticias/_execucoes.json              historico das execucoes

AUTOR     : Marcos Arambasic
DATA      : OUT/2026
****************************************************************************************/
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ))

from src.agentes import agente4_riscos_oportunidades as a4      # noqa: E402
from src.agentes import agente5_recomendacoes as a5             # noqa: E402
from src.noticias.classificacao import (                        # noqa: E402
    classificar_noticias, id_noticia, validar_noticias,
)

DIR_DATA = RAIZ / "data"
DIR_SCRIPTS_ETL = RAIZ / "src" / "data"
DIR_NOTICIAS = DIR_DATA / "noticias"
BASE_CLASSIFICADA = DIR_NOTICIAS / "noticias_classificadas.csv"
SAIDA_A3 = DIR_DATA / "agentes" / "agente3_saida.json"
SAIDA_A4 = DIR_DATA / "agentes" / "agente4_saida.json"
SAIDA_A5 = DIR_DATA / "agentes" / "agente5_saida.json"

FONTES = {
    # nome: (script de extracao, script de transformacao, xlsx transformado, variavel(is) de chave)
    "gnews": ("extract_gnews.py", "transformation_and_load_gnews.py",
              "transformacao_noticias_gnews.xlsx", [f"API_KEY_GNEWS_PRODUCAO_{i}" for i in range(1, 12)]),
    "guardian": ("extract_guardian_news.py", "transformation_and_load_guardian.py",
                 "transformacao_noticias_guardian.xlsx", ["API_KEY_GUARDIAN"]),
}


def _carregar_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(RAIZ / ".env")
    except ImportError:
        pass


def _rodar(script: str, log) -> bool:
    """Roda um script de ETL com cwd = src/data (os caminhos deles sao relativos a essa pasta)."""
    r = subprocess.run([sys.executable, script], cwd=DIR_SCRIPTS_ETL,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    saida = (r.stdout or "").strip().splitlines()[-3:]
    for linha in saida:
        log(f"    {linha}")
    if r.returncode != 0:
        log(f"    ERRO em {script}: {(r.stderr or '').strip().splitlines()[-1:]}")
    return r.returncode == 0


def extrair_e_transformar(log=print, fazer_extracao: bool = True) -> dict:
    _carregar_env()
    status = {}
    for nome, (ext, trans, _, chaves) in FONTES.items():
        tem_chave = any(os.environ.get(c) for c in chaves)
        if fazer_extracao and tem_chave:
            log(f"  [{nome}] extraindo...")
            ok = _rodar(ext, log)
            if ok:
                log(f"  [{nome}] transformando...")
                ok = _rodar(trans, log)
            status[nome] = "atualizada" if ok else "erro"
        else:
            motivo = "extracao desligada" if not fazer_extracao else f"sem chave ({chaves[0]}) no .env"
            log(f"  [{nome}] pulada - {motivo}; usando a base ja transformada")
            status[nome] = "pulada"
    return status


def consolidar() -> pd.DataFrame:
    partes = []
    for nome, (_, _, xlsx, _) in FONTES.items():
        p = DIR_DATA / xlsx
        if not p.exists():
            continue
        d = pd.read_excel(p)
        d["data_publicacao"] = pd.to_datetime(
            dict(year=d["ano"], month=d["mes"], day=d["dia"]), errors="coerce")
        d["origem_fonte"] = nome
        partes.append(d)
    if not partes:
        return pd.DataFrame(columns=["titulo", "descricao", "url"])
    df = pd.concat(partes, ignore_index=True)
    df = df.dropna(subset=["url"]).drop_duplicates(subset="url", keep="last")
    return df.reset_index(drop=True)


def classificar_incremental(df: pd.DataFrame, usar_llm: bool, log=print,
                            reclassificar: bool = False) -> pd.DataFrame:
    """Classifica so as noticias novas e acumula na base (reclassificar=True refaz tudo)."""
    DIR_NOTICIAS.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(BASE_CLASSIFICADA) if (BASE_CLASSIFICADA.exists() and not reclassificar) \
        else pd.DataFrame()
    ja = set(base["id_noticia"]) if not base.empty else set()
    df = df.assign(_id=df["url"].map(id_noticia))
    novas = df[~df["_id"].isin(ja)].drop(columns="_id")
    log(f"  {len(df):,} noticias na base consolidada; {len(novas):,} novas para classificar")
    if not novas.empty:
        c = classificar_noticias(novas, usar_llm=usar_llm, log=log)
        c["processada_em"] = datetime.now().isoformat(timespec="seconds")
        base = pd.concat([base, c], ignore_index=True) if not base.empty else c
        base.to_csv(BASE_CLASSIFICADA, index=False)
    return base


def nomes_dimensoes() -> tuple[dict, dict]:
    sint = DIR_DATA / "sintetico"
    np_, no_ = {}, {}
    if (sint / "dim_produto.csv").exists():
        d = pd.read_csv(sint / "dim_produto.csv")
        np_ = dict(zip(d["id_produto"], d["nome_sintetico"]))
    if (sint / "dim_outlet.csv").exists():
        d = pd.read_csv(sint / "dim_outlet.csv")
        no_ = dict(zip(d["id_outlet"], d["codigo_outlet"]))
    return np_, no_


def cruzar_e_recomendar(usar_llm: bool = False, log=print) -> dict:
    """Agentes 4 e 5 sobre a base classificada atual. Rapido (segundos)."""
    if not SAIDA_A3.exists():
        raise FileNotFoundError("data/agentes/agente3_saida.json nao existe - rode o Agente 3 antes")
    if not BASE_CLASSIFICADA.exists():
        raise FileNotFoundError("nenhuma noticia classificada ainda - rode o pipeline de noticias")
    a3 = json.loads(SAIDA_A3.read_text(encoding="utf-8"))
    validas = validar_noticias(pd.read_csv(BASE_CLASSIFICADA))
    log(f"  Agente 2: {len(validas):,} noticias de impacto alto/medio ligadas ao mercado")

    s4 = a4.AgenteRiscosOportunidades(a3).cruzar(validas)
    erros = a4.validar_saida(s4)
    if erros:
        raise ValueError(f"contrato do Agente 4 invalido: {erros[:3]}")
    a4.salvar_json(s4, SAIDA_A4)
    log(f"  Agente 4: {sum(len(s['achados']) for s in s4['setores']):,} achados em {len(s4['setores'])} setores")

    np_, no_ = nomes_dimensoes()
    s5 = a5.AgenteRecomendacoes(np_, no_, usar_llm=usar_llm).gerar(s4)
    erros = a5.validar_saida(s5)
    if erros:
        raise ValueError(f"contrato do Agente 5 invalido: {erros[:3]}")
    a5.salvar_json(s5, SAIDA_A5)
    log(f"  Agente 5: recomendacoes geradas para {len(s5['setores'])} setores")
    return {"achados": sum(len(s["achados"]) for s in s4["setores"]),
            "noticias_validas": int(len(validas))}


def executar(fazer_extracao: bool = True, usar_llm: bool = False, log=print,
             reclassificar: bool = False) -> dict:
    inicio = datetime.now()
    log("1-2. ETL de noticias")
    status = extrair_e_transformar(log, fazer_extracao)
    log("3. Consolidacao")
    df = consolidar()
    log("4. Agentes 1 e 2 (classificacao e validacao)")
    base = classificar_incremental(df, usar_llm, log, reclassificar)
    log("5-6. Agentes 4 e 5 (cruzamento e recomendacoes)")
    res = cruzar_e_recomendar(usar_llm, log)

    registro = {
        "executado_em": inicio.isoformat(timespec="seconds"),
        "duracao_s": round((datetime.now() - inicio).total_seconds(), 1),
        "fontes": status,
        "noticias_na_base": int(len(base)),
        **res,
    }
    hist_p = DIR_NOTICIAS / "_execucoes.json"
    hist = json.loads(hist_p.read_text(encoding="utf-8")) if hist_p.exists() else []
    hist.append(registro)
    hist_p.write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")
    return registro
