"""***************************************************************************************
SCRIPT    : gerar_kpi_final.py
OBJETIVO  : Gera a fato_analise_kpi_final (base de treino do Agente 3) a partir
            dos CSVs ja exportados - nao regenera nada.

ENTRADAS (em data/sintetico/, ou --dir-base):
    fato_analise_kpi*.csv     (gerado pelo gerar_backfill_semanal.py)
    fato_alocacao.csv
    dim_classifica_kpis.csv   (ou --classifica-kpis CAMINHO)

USO:
    python scripts/gerar_kpi_final.py
    python scripts/gerar_kpi_final.py --particionar

AUTOR     : Marcos Arambasic
DATA      : OUT/2026
****************************************************************************************/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.sintetico import config as cfg                    # noqa: E402
from src.sintetico.exportacao import exportar              # noqa: E402
from src.sintetico.kpi_final import montar_kpi_final       # noqa: E402
from src.sintetico.log_conf import configurar, etapa       # noqa: E402

log = configurar("kpi_final")


def _ler(dir_base: Path, padrao: str) -> pd.DataFrame:
    arquivos = sorted(a for a in dir_base.glob(padrao) if "_final" not in a.name)
    if not arquivos:
        raise SystemExit(f"nao encontrei {padrao} em {dir_base}")
    return pd.concat((pd.read_csv(a) for a in arquivos), ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Gera a fato_analise_kpi_final")
    ap.add_argument("--dir-base", type=Path, default=None)
    ap.add_argument("--dir-saida", type=Path, default=None)
    ap.add_argument("--classifica-kpis", type=Path, default=None)
    ap.add_argument("--particionar", action="store_true")
    args = ap.parse_args()

    dir_base = args.dir_base or cfg.DIR_SAIDA
    dir_saida = args.dir_saida or dir_base
    faixas_csv = args.classifica_kpis or (dir_base / "dim_classifica_kpis.csv")
    if not faixas_csv.exists():
        raise SystemExit(f"dim_classifica_kpis nao encontrada: {faixas_csv}")

    with etapa(log, "kpi_final", str(dir_base)) as ctx:
        kpi = _ler(dir_base, "fato_analise_kpi*.csv")
        aloc = pd.read_csv(dir_base / "fato_alocacao.csv")
        final = montar_kpi_final(kpi, aloc, pd.read_csv(faixas_csv))
        ctx["registros"] = len(final)
        exportar(final, "fato_analise_kpi_final", dir_saida,
                 particionar_por_ano=args.particionar)

    print(f"\nfato_analise_kpi_final: {len(final):,} linhas "
          f"({final.periodo_demanda.min():%m/%Y} a {final.periodo_demanda.max():%m/%Y}) "
          f"em {dir_saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
