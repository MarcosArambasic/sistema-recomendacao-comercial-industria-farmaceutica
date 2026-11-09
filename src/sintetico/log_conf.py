"""***************************************************************************************
SCRIPT    : log_conf.py
OBJETIVO  : Logging estruturado do pipeline (secao 27 do prompt mestre).
            Registra data/hora, etapa, periodo, quantidade de registros,
            tempo de execucao, erros, warnings e validacoes - em texto no
            console e em JSON Lines em arquivo, para permitir auditoria
            posterior sem precisar reparsear texto livre.

AUTOR     : Marcos Arambasic 
DATA      : AGO/2026
****************************************************************************************/
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .config import DIR_LOGS


class FormatoJSONL(logging.Formatter):
    """Uma linha JSON por evento - facil de carregar num DataFrame depois."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "nivel": record.levelname,
            "modulo": record.name,
            "mensagem": record.getMessage(),
        }
        for chave in ("etapa", "periodo", "registros", "duracao_s", "extra"):
            if hasattr(record, chave):
                payload[chave] = getattr(record, chave)
        if record.exc_info:
            payload["erro"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configurar(nome: str = "sintetico", nivel: int = logging.INFO,
               arquivo: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(nome)
    if logger.handlers:
        return logger

    logger.setLevel(nivel)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                           datefmt="%H:%M:%S"))
    logger.addHandler(console)

    DIR_LOGS.mkdir(parents=True, exist_ok=True)
    destino = arquivo or DIR_LOGS / f"pipeline_{datetime.now():%Y%m%d}.jsonl"
    arq = logging.FileHandler(destino, encoding="utf-8")
    arq.setFormatter(FormatoJSONL())
    logger.addHandler(arq)

    logger.propagate = False
    return logger


@contextmanager
def etapa(logger: logging.Logger, nome: str, periodo: str | None = None):
    """
    Context manager que cronometra uma etapa e loga inicio/fim/erro.

    Uso:
        with etapa(log, "geracao_demanda", "2023-01..2026-08") as ctx:
            ...
            ctx["registros"] = len(df)
    """
    ctx: dict = {"registros": None}
    inicio = time.time()
    logger.info(f"[inicio] {nome}", extra={"etapa": nome, "periodo": periodo})
    try:
        yield ctx
    except Exception:
        logger.exception(
            f"[erro] {nome} apos {time.time() - inicio:.1f}s",
            extra={"etapa": nome, "periodo": periodo,
                   "duracao_s": round(time.time() - inicio, 2)},
        )
        raise
    else:
        dur = round(time.time() - inicio, 2)
        logger.info(
            f"[fim] {nome} em {dur}s"
            + (f" - {ctx['registros']:,} registros" if ctx.get("registros") else ""),
            extra={"etapa": nome, "periodo": periodo, "duracao_s": dur,
                   "registros": ctx.get("registros")},
        )
