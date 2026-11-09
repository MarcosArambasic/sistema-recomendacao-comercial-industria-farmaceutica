"""***************************************************************************************
SCRIPT    : gerar_backfill_semanal.py
OBJETIVO  : Roda o pipeline semanal sobre TODO o historico (229 semanas) e o
            fato_analise_kpi mensal, exportando tudo em CSV local.

            E o passo que produz os dados de TREINO do XGBoost. Sem ele,
            fato_analise_semanal_final so acumularia 1 conjunto de linhas por
            semana daqui pra frente - insuficiente para treinar o modelo e
            para a validacao de 12 meses prometida na secao 3.7 do TCC1.

ETAPAS:
    1. gera a demanda mensal (mesma seed -> base identica a ja exportada)
    2. calcula a cota mensal
    3. calcula o fato_analise_kpi mensal
    4. roda o pipeline semanal completo em backfill
    5. classifica (se dim_classifica_kpis estiver disponivel)
    6. exporta tudo em CSV local

USO:
    python scripts/gerar_backfill_semanal.py
    python scripts/gerar_backfill_semanal.py --particionar
    python scripts/gerar_backfill_semanal.py --sem-demanda-semanal   # pula a maior tabela
    python scripts/gerar_backfill_semanal.py --classificar data/dim_classifica_kpis.csv
    python scripts/gerar_backfill_semanal.py --ate 2026-09   # historico ate 09/2026

VOLUME (referencia):
    fato_analise_kpi              1,4 M linhas
    fato_demanda_aberta_semanal   7,4 M linhas  <- a maior; transiente no banco
    fato_demanda_alocada_semanal  2,5 M linhas
    fato_desempenho_semanal       2,4 M linhas
    fato_analise_semanal          2,4 M linhas  <- base de treino

AUTOR     : Marcos Arambasic
DATA      : SET/2026
****************************************************************************************/
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.sintetico import config as cfg                          # noqa: E402
from src.sintetico.calibracao import carregar_calibracao         # noqa: E402
from src.sintetico.cota import calcular_cota_mensal              # noqa: E402
from src.sintetico.exportacao import (                           # noqa: E402
    exportar, gerar_instrucoes_importacao, gerar_manifesto,
)
from src.sintetico.gerador import agregar_por_setor, gerar_demanda_mensal  # noqa: E402
from src.sintetico.kpi import calcular_analise_kpi, sugerir_faixas_kpi     # noqa: E402
from src.sintetico.log_conf import configurar, etapa             # noqa: E402
from src.sintetico.pipeline_semanal import classificar, executar_backfill  # noqa: E402

log = configurar("backfill")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Backfill do pipeline semanal + KPI mensal, exportando CSV local"
    )
    ap.add_argument("--particionar", action="store_true",
                    help="um CSV por ano em cada tabela")
    ap.add_argument("--comprimir", action="store_true", help="grava em .gz")
    ap.add_argument("--dir-saida", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sem-demanda-semanal", action="store_true",
                    help="nao exporta fato_demanda_aberta_semanal (a maior tabela; "
                         "e transiente no banco por desenho)")
    ap.add_argument("--classificar", type=Path, default=None,
                    help="CSV de dim_classifica_kpis para gerar fato_analise_semanal_final")
    ap.add_argument("--ate", metavar="AAAA-MM", default=None,
                    help="ultimo mes do historico semanal/KPI. Use quando a base mensal vai "
                         "alem disso e as semanas seguintes serao geradas uma a uma pelo "
                         "executar_semana.py (ex.: base ate 2027-12, historico ate 2026-09)")
    args = ap.parse_args()

    periodo = f"{cfg.PERIODO_INICIO}..{cfg.PERIODO_FIM}"
    dir_saida = args.dir_saida or cfg.DIR_SAIDA

    # ---------------- 1-2. base mensal ------------------------------------
    with etapa(log, "base_mensal", periodo) as ctx:
        calib = carregar_calibracao()
        df_dem, dims, diag = gerar_demanda_mensal(seed=args.seed, calib=calib)
        if args.ate:
            # o historico para no mes informado; o resto da base mensal fica
            # disponivel para o executar_semana.py gerar semana a semana
            corte = pd.Timestamp(args.ate + "-01")
            df_dem = df_dem[pd.to_datetime(df_dem["periodo_demanda"]) <= corte].copy()
            periodo = f"{cfg.PERIODO_INICIO}..{args.ate}"
            log.info(f"historico limitado a {args.ate} (--ate)")
        df_cota = calcular_cota_mensal(agregar_por_setor(df_dem, dims))
        ctx["registros"] = len(df_dem)

    # ---------------- 3. KPI mensal ---------------------------------------
    # As tabelas somam ~1,5 GB e os merges chegam a dobrar isso. Por isso cada
    # tabela e exportada assim que fica pronta e liberada em seguida, em vez
    # de manter todas vivas ate o fim (a versao anterior estourava a memoria).
    exportacoes: dict[str, list[dict]] = {}
    faixas_sugeridas = []

    def _exportar_e_liberar(nome: str, df: pd.DataFrame) -> None:
        if nome == "fato_demanda_aberta_semanal" and args.sem_demanda_semanal:
            log.info(f"   {nome}: {len(df):,} linhas (nao exportada, --sem-demanda-semanal)")
            return
        exportacoes[nome] = exportar(
            df, nome, dir_saida,
            particionar_por_ano=args.particionar,
            comprimir=args.comprimir,
        )
        gc.collect()

    with etapa(log, "analise_kpi_mensal", periodo) as ctx:
        df_kpi = calcular_analise_kpi(df_dem, dims)
        ctx["registros"] = len(df_kpi)
        _exportar_e_liberar("fato_analise_kpi", df_kpi)
        if not args.classificar:
            faixas_sugeridas.append(sugerir_faixas_kpi(df_kpi[df_kpi.demanda_produto > 0]))

    # ---------------- 4. pipeline semanal em backfill ----------------------
    analise_final = {}

    def _ao_concluir(nome: str, df: pd.DataFrame) -> None:
        log.info(f"   {nome:<32s} {len(df):>10,} linhas")
        _exportar_e_liberar(nome, df)

        if nome == "fato_analise_semanal":
            # classificacao (etapa 7) aproveita o DataFrame ainda em memoria
            if args.classificar:
                faixas = pd.read_csv(args.classificar)
                final = classificar(df, faixas)
                _exportar_e_liberar("fato_analise_semanal_final", final)
                analise_final["linhas"] = len(final)
                del final
                gc.collect()
            else:
                faixas_sugeridas.append(
                    sugerir_faixas_kpi(df, kpis=["cobertura_cota_projetada"])
                )

    if args.classificar and not args.classificar.exists():
        log.error(f"arquivo nao encontrado: {args.classificar}")
        return 2

    with etapa(log, "backfill_semanal", periodo) as ctx:
        contagens = executar_backfill(
            df_dem, df_cota, df_kpi, dims, seed=args.seed,
            ao_concluir=_ao_concluir, manter_em_memoria=False,
        )
        ctx["registros"] = sum(contagens.values())

    if not args.classificar:
        log.info("dim_classifica_kpis nao informada - fato_analise_semanal_final nao gerada. "
                 "Use --classificar CAMINHO.csv quando as faixas estiverem definidas.")

    # ---------------- 5. manifesto e instrucoes ----------------------------
    with etapa(log, "manifesto", periodo):
        gerar_manifesto(exportacoes, diag, dir_saida)
        gerar_instrucoes_importacao(exportacoes, dir_saida)

    # ---------------- 6. faixas sugeridas ----------------------------------
    if faixas_sugeridas:
        sugestao = pd.concat(faixas_sugeridas, ignore_index=True)
        destino = Path(dir_saida) / "dim_classifica_kpis.csv"
        sugestao.to_csv(destino, index=False)
        log.info(f"faixas sugeridas (a partir dos quantis reais): {destino}")

    print(f"\nTabelas exportadas em {dir_saida}:")
    for tabela, metas in exportacoes.items():
        total = sum(m["linhas"] for m in metas)
        mb = round(sum(m["tamanho_mb"] for m in metas), 2)
        print(f"  {tabela:<32s} {total:>10,} linhas  {mb:>8.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
