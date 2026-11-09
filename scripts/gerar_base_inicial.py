"""***************************************************************************************
SCRIPT    : gerar_base_inicial.py
OBJETIVO  : Orquestrador da geracao da base historica (secoes 20, 26, 27 e 28
            do prompt mestre). Executa, em ordem e com logging por etapa:

              1. calibracao   - le o Excel e congela os parametros em JSON
              2. geracao      - fato_demanda_aberta_mensal (mes x produto x outlet)
              3. cota         - fato_cota_mensal (mes x setor x produto)
              4. validacao    - fidelity score + relatorio de desvios
              5. exportacao   - CSVs locais em data/sintetico/  <- SAIDA PADRAO
              6. carga        - OPCIONAL, direto no Supabase (--carregar)

            A saida padrao e CSV LOCAL. O banco so e tocado se voce pedir
            explicitamente com --carregar. Junto com os CSVs sao gerados:
              * manifesto_exportacao.json  - linhas, tamanho e SHA-256
              * INSTRUCOES_IMPORTACAO.md   - comandos \\copy prontos

USO:
    python scripts/gerar_base_inicial.py                  # gera, valida e exporta CSV
    python scripts/gerar_base_inicial.py --particionar    # um CSV por ano (import mais facil)
    python scripts/gerar_base_inicial.py --comprimir      # CSVs em .gz
    python scripts/gerar_base_inicial.py --seed 7         # outra semente
    python scripts/gerar_base_inicial.py --recalibrar     # forca releitura do Excel
    python scripts/gerar_base_inicial.py --carregar       # (opcional) sobe no Supabase

A carga, quando pedida, so acontece se a validacao passar do score minimo
(--score-minimo), para nao subir base ruim para o banco.

AUTOR     : Marcos Arambasic
DATA      : AGO/2026
****************************************************************************************/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.sintetico import config as cfg                       # noqa: E402
from src.sintetico.calibracao import salvar_calibracao, carregar_calibracao  # noqa: E402
from src.sintetico.cota import adicionar_metadados, calcular_cota_mensal  # noqa: E402
from src.sintetico.exportacao import exportar_tudo            # noqa: E402
from src.sintetico.gerador import agregar_por_setor, gerar_demanda_mensal  # noqa: E402
from src.sintetico.log_conf import configurar, etapa          # noqa: E402
from src.sintetico.validacao import gerar_relatorio_markdown, validar  # noqa: E402

log = configurar("pipeline")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Gera a base sintetica historica e exporta em CSV local"
    )
    ap.add_argument("--particionar", action="store_true",
                    help="gera um CSV por ano (arquivos menores, import mais facil)")
    ap.add_argument("--comprimir", action="store_true", help="grava os CSVs em .gz")
    ap.add_argument("--dir-saida", type=Path, default=None,
                    help=f"pasta de saida (default: {cfg.DIR_SAIDA})")
    ap.add_argument("--seed", type=int, default=None, help="semente aleatoria")
    ap.add_argument("--recalibrar", action="store_true", help="rele o Excel de parametros")
    ap.add_argument("--sem-csv", action="store_true", help="nao exporta CSVs")
    ap.add_argument("--carregar", action="store_true",
                    help="OPCIONAL: alem do CSV, carrega direto no Supabase")
    ap.add_argument("--score-minimo", type=float, default=80.0,
                    help="score minimo de fidelidade para permitir a carga")
    args = ap.parse_args()

    periodo = f"{cfg.PERIODO_INICIO}..{cfg.PERIODO_FIM}"

    # ---------------- 1. calibracao ---------------------------------------
    with etapa(log, "calibracao", periodo):
        if args.recalibrar or not cfg.ARQ_CALIBRACAO.exists():
            salvar_calibracao()
        calib = carregar_calibracao()
        log.info(f"calibracao: {calib['_meta']['n_territorios']} territorios, "
                 f"{calib['_meta']['meses_observados']} meses observados")

    # ---------------- 2. geracao da demanda -------------------------------
    with etapa(log, "geracao_demanda", periodo) as ctx:
        df_dem, dims, diag = gerar_demanda_mensal(seed=args.seed, calib=calib)
        ctx["registros"] = len(df_dem)
        log.info(f"densidade produto x outlet: {diag['densidade']:.2%} "
                 f"({diag['n_pares_ativos']:,} pares ativos)")

    # ---------------- 3. cota ---------------------------------------------
    with etapa(log, "calculo_cota", periodo) as ctx:
        por_setor = agregar_por_setor(df_dem, dims)
        df_cota = calcular_cota_mensal(por_setor)
        ctx["registros"] = len(df_cota)
        n_boot = int(df_cota["bootstrap"].sum())
        if n_boot:
            log.warning(f"{n_boot:,} linhas de cota usaram janela incompleta "
                        f"(primeiros meses da serie, sem 6 meses de historico)")

    # ---------------- 4. validacao ----------------------------------------
    with etapa(log, "validacao", periodo):
        rel = validar(df_dem, df_cota, dims, calib)
        score = rel.fidelity_score()
        destino = cfg.BASE_DIR / "docs" / "Arquitetura" / "RELATORIO_VALIDACAO.md"
        gerar_relatorio_markdown(rel, diag, destino)

        log.info(f"synthetic_data_fidelity_score = {score:.2f}/100")
        for dim, val in sorted(rel.score_por_dimensao().items(), key=lambda x: -x[1]):
            log.info(f"   {dim:<16s} {val:6.2f}")
        falhas = rel.falhas()
        if not falhas.empty:
            log.warning(f"{len(falhas)} checks abaixo de 70 - ver {destino.name}")
            for _, r in falhas.iterrows():
                log.warning(f"   {r.metrica}: {r.valor} (alvo {r.alvo}) score {r.score}")

    # ---------------- 5. exportacao CSV local (saida padrao) ---------------
    exportacoes = {}
    if not args.sem_csv:
        with etapa(log, "exportacao_csv", periodo) as ctx:
            exportacoes = exportar_tudo(
                df_dem, df_cota, diag,
                dir_saida=args.dir_saida,
                particionar_por_ano=args.particionar,
                comprimir=args.comprimir,
            )
            ctx["registros"] = sum(
                m["linhas"] for metas in exportacoes.values() for m in metas
            )

    # ---------------- 6. carga no banco (opcional) -------------------------
    if args.carregar:
        if score < args.score_minimo:
            log.error(f"carga abortada: score {score:.2f} < minimo {args.score_minimo}. "
                      "Ajuste os parametros ou baixe --score-minimo conscientemente.")
            return 2
        from src.sintetico.carga import carregar_cota_mensal, carregar_demanda_mensal
        with etapa(log, "carga_supabase", periodo):
            r1 = carregar_demanda_mensal(df_dem)
            r2 = carregar_cota_mensal(adicionar_metadados(df_cota))
            log.info(f"carga concluida: {r1['inseridos']:,} + {r2['inseridos']:,} linhas")

    # ---------------- resumo ----------------------------------------------
    print(f"\nsynthetic_data_fidelity_score: {score:.2f}/100")
    print(f"demanda: {len(df_dem):,} linhas | cota: {len(df_cota):,} linhas")
    if exportacoes:
        destino = Path(args.dir_saida or cfg.DIR_SAIDA)
        print(f"\nCSVs em {destino}:")
        for tabela, metas in exportacoes.items():
            total = sum(m["linhas"] for m in metas)
            mb = round(sum(m["tamanho_mb"] for m in metas), 2)
            print(f"  {tabela}: {len(metas)} arquivo(s), {total:,} linhas, {mb} MB")
        print(f"\nComandos de importacao prontos em {destino / 'INSTRUCOES_IMPORTACAO.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
