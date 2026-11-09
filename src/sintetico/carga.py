"""***************************************************************************************
SCRIPT    : carga.py
OBJETIVO  : Carga das tabelas fato no PostgreSQL/Supabase (secoes 19, 25, 26
            e 28 do prompt mestre).

PRINCIPIOS APLICADOS:
  * COPY em vez de INSERT linha a linha - a diferenca em 1,4 milhao de linhas
    e de horas para segundos;
  * TRANSACAO por etapa, com ROLLBACK automatico: ou a tabela inteira entra,
    ou nada entra. Nunca deixa o banco meio atualizado;
  * IDEMPOTENCIA por recorte de periodo: antes de inserir, apaga o intervalo
    que sera reescrito. Rodar duas vezes produz o mesmo estado final.
    (Estrategia escolhida porque as tabelas fato do schema atual nao tem
    constraint UNIQUE de grao - sem ela, ON CONFLICT nao e possivel. O
    script supabase/scripts/alter_schema_constraints.sql propoe adiciona-las;
    com elas, da para migrar para UPSERT.)
  * CREDENCIAIS so por variavel de ambiente. Nada de chave no codigo.

CONEXAO: use DATABASE_URL (string do Postgres). A SERVICE_ROLE_KEY do
Supabase serve para a API REST, nao para conexao direta - e para carga em
lote a conexao Postgres e muito mais eficiente.

AUTOR     : Marcos Arambasic
DATA      : AGO/2026
****************************************************************************************/
"""

from __future__ import annotations

import io
import os
from contextlib import contextmanager

import pandas as pd

from . import config as cfg
from .log_conf import configurar, etapa

log = configurar(__name__)

COLUNAS_DEMANDA_MENSAL = [
    "data_atualizacao_base", "periodo_demanda",
    "id_produto", "id_outlet", "id_uf", "demanda_unidades",
]
COLUNAS_COTA_MENSAL = [
    "periodo_demanda", "id_setor", "id_produto", "cota_unidades",
]


def _database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL nao definida. Copie .env.example para .env e preencha. "
            "Nunca coloque a string de conexao no codigo."
        )
    if "[YOUR-PASSWORD]" in url or "YOUR-PASSWORD" in url:
        raise RuntimeError(
            "DATABASE_URL ainda esta com o placeholder de senha. "
            "Substitua pela senha real do banco no arquivo .env."
        )
    return url


@contextmanager
def conexao():
    """Conexao psycopg2 com transacao explicita e rollback em caso de erro."""
    try:
        import psycopg2
    except ImportError as e:
        raise RuntimeError(
            "psycopg2 nao instalado. Rode: pip install psycopg2-binary"
        ) from e

    conn = psycopg2.connect(_database_url())
    conn.autocommit = False
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        log.exception("Transacao revertida (ROLLBACK) - banco nao foi alterado")
        raise
    finally:
        conn.close()


def _copy(cur, df: pd.DataFrame, tabela: str, colunas: list[str]) -> int:
    """Envia um DataFrame para o Postgres via COPY (formato CSV em memoria)."""
    buf = io.StringIO()
    df[colunas].to_csv(buf, index=False, header=False, na_rep="\\N")
    buf.seek(0)
    cur.copy_expert(
        f"COPY public.{tabela} ({', '.join(colunas)}) FROM STDIN WITH (FORMAT csv, NULL '\\N')",
        buf,
    )
    return len(df)


def _apagar_periodo(cur, tabela: str, periodo_min, periodo_max) -> int:
    cur.execute(
        f"DELETE FROM public.{tabela} WHERE periodo_demanda BETWEEN %s AND %s",
        (periodo_min, periodo_max),
    )
    return cur.rowcount


def carregar_tabela(df: pd.DataFrame, tabela: str, colunas: list[str],
                    substituir_periodo: bool = True,
                    batch_size: int = cfg.BATCH_SIZE) -> dict:
    """
    Carrega um DataFrame numa tabela fato, de forma idempotente e transacional.

    substituir_periodo=True apaga previamente o intervalo de periodos contido
    no DataFrame, garantindo que reprocessar nao duplique.
    """
    if df.empty:
        log.warning(f"{tabela}: DataFrame vazio, nada a carregar")
        return {"tabela": tabela, "inseridos": 0, "removidos": 0}

    p_min = pd.to_datetime(df["periodo_demanda"]).min()
    p_max = pd.to_datetime(df["periodo_demanda"]).max()
    removidos = inseridos = 0

    with etapa(log, f"carga.{tabela}", f"{p_min:%Y-%m} a {p_max:%Y-%m}") as ctx:
        with conexao() as conn:
            with conn.cursor() as cur:
                if substituir_periodo:
                    removidos = _apagar_periodo(cur, tabela, p_min, p_max)
                    if removidos:
                        log.info(f"{tabela}: {removidos:,} linhas do periodo removidas "
                                 f"antes da recarga (idempotencia)")
                for i in range(0, len(df), batch_size):
                    inseridos += _copy(cur, df.iloc[i:i + batch_size], tabela, colunas)
        ctx["registros"] = inseridos

    return {"tabela": tabela, "inseridos": inseridos, "removidos": removidos}


def carregar_demanda_mensal(df: pd.DataFrame, **kw) -> dict:
    return carregar_tabela(df, "fato_demanda_aberta_mensal", COLUNAS_DEMANDA_MENSAL, **kw)


def carregar_cota_mensal(df: pd.DataFrame, **kw) -> dict:
    return carregar_tabela(df, "fato_cota_mensal", COLUNAS_COTA_MENSAL, **kw)


def conferir_contagens() -> pd.DataFrame:
    """SELECT count(*) nas tabelas do escopo - conferencia pos-carga."""
    tabelas = [
        "dim_mercado", "dim_produto", "dim_uf", "dim_setor", "dim_outlet",
        "fato_alocacao", "fato_demanda_aberta_mensal", "fato_cota_mensal",
    ]
    linhas = []
    with conexao() as conn:
        with conn.cursor() as cur:
            for t in tabelas:
                cur.execute(f"SELECT count(*) FROM public.{t}")
                linhas.append({"tabela": t, "linhas": cur.fetchone()[0]})
    return pd.DataFrame(linhas)


# Exportacao para CSV local mora em exportacao.py - este modulo cuida so da
# carga direta no banco, que e o caminho OPCIONAL do pipeline.


if __name__ == "__main__":
    print(conferir_contagens().to_string(index=False))
