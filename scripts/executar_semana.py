"""***************************************************************************************
SCRIPT    : executar_semana.py
OBJETIVO  : Execucao SEMANAL do pipeline (secoes 12 a 18 do prompt mestre).
            Roda todas as etapas para UMA semana comercial especifica.

            Este e o script que voce roda toda semana. O
            gerar_backfill_semanal.py e o oposto: roda uma vez so, para
            criar o historico.

COMO A SEMANA E DETERMINADA (nesta ordem de precedencia):

    1) --ano/--mes/--semana   -> semana explicita
    2) --data AAAA-MM-DD      -> a semana comercial que CONTEM essa data
    3) --hoje                 -> a semana comercial que contem HOJE
    4) nada                   -> a PROXIMA semana depois da ultima calculada
                                 (e o que o botao "Gerar nova semana" usa)

    Nao ha trava de data: qualquer semana pode ser gerada, inclusive
    semanas futuras em relacao ao dia de hoje.

    O calendario segue a regra do grupo: domingo->sabado, com a primeira e a
    ultima semana truncadas nos limites do mes. Use --listar-semanas AAAA-MM
    para ver a numeracao de um mes antes de rodar.

O QUE RODA EM CADA SEMANA:

    sempre                 ETAPA 1  demanda semanal do periodo
                           ETAPA 3  alocacao + acumulado no mes
                           ETAPA 5  desempenho (cobertura e projetada)
                           ETAPA 6  analise semanal (KPI + desempenho)
                           ETAPA 7  classificacao -> analise semanal final
    se PRIMEIRA semana     ETAPA 2  cota do mes (historico de 6 meses)
                           ETAPA 4  KPIs do mes anterior consolidado
    se ULTIMA semana       consolidacao semanal -> fato_demanda_aberta_mensal

    Nas semanas que nao sao a primeira, a cota e os KPIs vigentes sao os que
    ja foram calculados no inicio do mes - eles nao mudam no meio do mes, e
    por isso o script os recalcula de forma deterministica em vez de guardar
    estado entre execucoes. Rodar a mesma semana duas vezes produz
    exatamente o mesmo resultado.

COMO RODAR:
    python scripts/executar_semana.py                                   # proxima semana depois da ultima calculada
    python scripts/executar_semana.py --hoje                            # semana que contém hoje
    python scripts/executar_semana.py --data 2026-03-15                 # semana que contém a data
    python scripts/executar_semana.py --ano 2026 --mes 3 --semana 2
    python scripts/executar_semana.py --listar-semanas 2026-03          # so mostra o calendario de marco/2026 e sai
    python scripts/executar_semana.py --ultima-semana

DE ONDE VEM A DEMANDA DA SEMANA:

    mes dentro da base historica  -> data/sintetico/fato_demanda_aberta_mensal.csv
                                     (reprocessamento de semana antiga)
    mes ALEM da base historica    -> a demanda do mes e CRIADA pelo mesmo modelo
                                     calibrado (src/sintetico/extensao.py),
                                     repartida nas semanas, e a semana alvo e
                                     recortada. O mes so entra na
                                     fato_demanda_aberta_mensal na ULTIMA semana,
                                     pela soma das semanas (consolidacao).

    A criacao e deterministica por mes (semente propria do mes + continuacao
    exata dos processos AR(1) da base), entao as semanas 1..N de um mes sempre
    somam o mesmo total, nao importa em que ordem ou em que dia sao geradas.
    Para o historico de cota (6 meses) e KPI (12 meses), meses ja consolidados
    sao lidos das pastas data/semanal/*/fato_demanda_aberta_mensal.csv.

AUTOR     : Marcos Arambasic
DATA      : SET/2026
****************************************************************************************/
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.sintetico import config as cfg                              # noqa: E402
from src.sintetico.calendario import semanas_do_mes                  # noqa: E402
from src.sintetico.cota import calcular_cota_mensal                  # noqa: E402
from src.sintetico.exportacao import exportar                        # noqa: E402
from src.sintetico.gerador import agregar_por_setor, montar_dimensoes  # noqa: E402
from src.sintetico.calibracao import carregar_calibracao             # noqa: E402
from src.sintetico.kpi import calcular_analise_kpi                   # noqa: E402
from src.sintetico.log_conf import configurar, etapa                 # noqa: E402
from src.sintetico.pipeline_semanal import (                         # noqa: E402
    alocar_semanal, aplicar_diagnostico, calcular_desempenho, classificar,
    desagregar_semanal, montar_analise_semanal,
)

log = configurar("semanal")

ARQ_BASE_MENSAL = "fato_demanda_aberta_mensal.csv"
# Historico completo: o CAGR compara com a media do PRIMEIRO ano da serie, entao
# um recorte de 18 meses mudaria o indicador em relacao ao historico gerado.
MESES_HISTORICO_NECESSARIO = None


# ============================================================================
# Resolucao da semana alvo
# ============================================================================
def resolver_semana(args) -> dict:
    """Traduz os parametros da linha de comando na semana comercial alvo."""
    if args.ano and args.mes and args.semana:
        semanas = semanas_do_mes(args.ano, args.mes)
        alvo = [s for s in semanas if s.nr_semana == args.semana]
        if not alvo:
            raise SystemExit(
                f"{args.mes:02d}/{args.ano} tem {len(semanas)} semanas comerciais "
                f"(1 a {len(semanas)}), nao existe semana {args.semana}.\n"
                f"Rode com --listar-semanas {args.ano}-{args.mes:02d} para ver."
            )
        s = alvo[0]
    elif not args.data and not args.hoje:
        # padrao: proxima semana depois da ultima calculada
        u = ultima_semana_calculada(args.dir_base or cfg.DIR_SAIDA,
                                    cfg.BASE_DIR / "data" / "semanal")
        if u is None:
            raise SystemExit("nenhuma semana calculada ainda - informe --ano/--mes/--semana, "
                             "--data ou --hoje")
        s = u["proxima"]
    else:
        if args.data:
            try:
                ref = datetime.strptime(args.data, "%Y-%m-%d").date()
            except ValueError:
                raise SystemExit(f"--data invalida: '{args.data}'. Use AAAA-MM-DD.")
        else:
            ref = date.today()

        semanas = semanas_do_mes(ref.year, ref.month)
        alvo = [x for x in semanas if x.inicio <= ref <= x.fim]
        if not alvo:
            raise SystemExit(f"nao achei semana comercial contendo {ref}")
        s = alvo[0]

    todas = semanas_do_mes(s.ano, s.mes)
    return {
        "ano": s.ano, "mes": s.mes, "nr_semana": s.nr_semana,
        "inicio": s.inicio, "fim": s.fim,
        "periodo_demanda": pd.Timestamp(s.fim),
        "periodo_mensal": pd.Timestamp(s.periodo_mensal),
        "nr_dias": s.nr_dias,
        "primeira_semana": s.nr_semana == 1,
        "ultima_semana": s.nr_semana == len(todas),
        "total_semanas_no_mes": len(todas),
    }


def _arquivos_analise_semanal(dir_base: Path, dir_semanal: Path) -> list[Path]:
    """
    Todos os fato_analise_semanal*.csv(.gz) existentes: o backfill (arquivo unico
    ou particionado por ano, em dir_base) e as execucoes semanais
    (dir_semanal/AAAA-MM-SN/). Exclui a tabela _final.
    """
    achados: list[Path] = []
    padroes = ["fato_analise_semanal*.csv", "fato_analise_semanal*.csv.gz"]
    pastas = [Path(dir_base)]
    if Path(dir_semanal).exists():
        pastas += [p for p in sorted(Path(dir_semanal).iterdir()) if p.is_dir()]
    for pasta in pastas:
        for padrao in padroes:
            achados += [a for a in pasta.glob(padrao) if "_final" not in a.name]
    return achados


def ultima_semana_calculada(dir_base: Path, dir_semanal: Path) -> dict | None:
    """
    Descobre a semana comercial mais recente que tem linhas em
    fato_analise_semanal. Retorna None se nao houver nenhuma.
    """
    arquivos = _arquivos_analise_semanal(dir_base, dir_semanal)
    if not arquivos:
        return None

    melhor = None   # (periodo_demanda, nr_semana, arquivo, linhas)
    for arq in arquivos:
        df = pd.read_csv(arq, usecols=["periodo_demanda", "nr_semana"])
        if df.empty:
            continue
        df["periodo_demanda"] = pd.to_datetime(df["periodo_demanda"], errors="coerce")
        df = df.dropna(subset=["periodo_demanda"])
        if df.empty:
            continue
        maximo = df["periodo_demanda"].max()
        if melhor is None or maximo > melhor[0]:
            nr = int(df.loc[df["periodo_demanda"] == maximo, "nr_semana"].iloc[0])
            linhas = int((df["periodo_demanda"] == maximo).sum())
            melhor = (maximo, nr, arq, linhas)

    if melhor is None:
        return None

    fim, nr, arq, linhas = melhor
    ref = fim.date()
    semanas = semanas_do_mes(ref.year, ref.month)
    s = next((x for x in semanas if x.nr_semana == nr), None)

    # proxima semana a rodar (a seguinte no calendario comercial)
    if nr < len(semanas):
        prox = next(x for x in semanas if x.nr_semana == nr + 1)
    else:
        ano_p, mes_p = (ref.year + 1, 1) if ref.month == 12 else (ref.year, ref.month + 1)
        prox = semanas_do_mes(ano_p, mes_p)[0]

    return {
        "ano": ref.year, "mes": ref.month, "nr_semana": nr,
        "inicio": s.inicio if s else None, "fim": ref,
        "total_semanas_no_mes": len(semanas),
        "linhas": linhas, "arquivo": arq,
        "proxima": prox,
    }


def mostrar_ultima_semana(dir_base: Path, dir_semanal: Path) -> int:
    u = ultima_semana_calculada(dir_base, dir_semanal)
    if u is None:
        print("\nNenhuma semana calculada: nao ha fato_analise_semanal*.csv em\n"
              f"  {dir_base}\n  {dir_semanal}\n"
              "Rode o gerar_backfill_semanal.py ou o executar_semana.py.\n")
        return 1
    p = u["proxima"]
    print("\nULTIMA SEMANA CALCULADA (fato_analise_semanal)")
    print(f"  semana          : {u['ano']}-{u['mes']:02d} semana {u['nr_semana']} "
          f"de {u['total_semanas_no_mes']}")
    if u["inicio"]:
        print(f"  periodo         : {u['inicio']:%d/%m/%Y} a {u['fim']:%d/%m/%Y}")
    print(f"  periodo_demanda : {u['fim']:%Y-%m-%d}")
    print(f"  linhas          : {u['linhas']:,}")
    print(f"  arquivo         : {u['arquivo']}")
    print(f"\nPROXIMA A RODAR: {p.ano}-{p.mes:02d} semana {p.nr_semana} "
          f"({p.inicio:%d/%m/%Y} a {p.fim:%d/%m/%Y})")
    print(f"  python scripts/executar_semana.py --ano {p.ano} --mes {p.mes} "
          f"--semana {p.nr_semana}\n")
    return 0


def listar_semanas(aaaa_mm: str, dir_base: Path, dir_semanal: Path) -> None:
    try:
        ano, mes = (int(x) for x in aaaa_mm.split("-"))
    except ValueError:
        raise SystemExit(f"formato invalido: '{aaaa_mm}'. Use AAAA-MM.")

    ultima = ultima_semana_calculada(dir_base, dir_semanal)
    hoje = date.today()

    semanas = semanas_do_mes(ano, mes)
    print(f"\nSemanas comerciais de {mes:02d}/{ano} ({len(semanas)} no total):\n")
    print(f"  {'semana':>6}  {'inicio':<12} {'fim':<12} {'dias':>4}  periodo_demanda")
    for s in semanas:
        marca = ""
        if s.nr_semana == 1:
            marca = "  <- primeira (calcula cota e KPI)"
        if s.nr_semana == len(semanas):
            marca += "  <- ultima (consolida o mes)"
        if s.inicio <= hoje <= s.fim:
            marca += "  <- semana ATUAL (hoje)"
        if ultima and (ultima["ano"], ultima["mes"], ultima["nr_semana"]) == (ano, mes, s.nr_semana):
            marca += "  <- ULTIMA CALCULADA"
        print(f"  {s.nr_semana:>6}  {s.inicio:%d/%m/%Y}   {s.fim:%d/%m/%Y}   "
              f"{s.nr_dias:>4}  {s.fim:%Y-%m-%d}{marca}")

    # semana comercial que contem hoje (pode ser de outro mes)
    atual = next((x for x in semanas_do_mes(hoje.year, hoje.month)
                  if x.inicio <= hoje <= x.fim), None)
    print()
    if atual:
        print(f"Semana mais atual (hoje, {hoje:%d/%m/%Y}): {atual.ano}-{atual.mes:02d} "
              f"semana {atual.nr_semana} ({atual.inicio:%d/%m/%Y} a {atual.fim:%d/%m/%Y})")
    if ultima:
        print(f"Ultima semana calculada: {ultima['ano']}-{ultima['mes']:02d} "
              f"semana {ultima['nr_semana']} (periodo_demanda {ultima['fim']:%Y-%m-%d})")
    else:
        print("Ultima semana calculada: nenhuma encontrada em fato_analise_semanal")
    print()


# ============================================================================
# Leitura da base mensal
# ============================================================================
def carregar_base_mensal(dir_base: Path, ate: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """
    Monta a demanda mensal de (ate - 18 meses) ate o mes alvo, juntando:
      1. a base historica (data/sintetico/fato_demanda_aberta_mensal*.csv);
      2. meses ja consolidados por execucoes semanais (data/semanal/*/);
      3. meses ainda nao consolidados -> criados pelo modelo (extensao.py).
    Retorna (df, origem_por_mes).
    """
    from datetime import date as _date
    from src.sintetico.extensao import carregar_estado, gerar_meses

    arquivos = sorted(dir_base.glob("fato_demanda_aberta_mensal*.csv"))
    if not arquivos:
        raise SystemExit(
            f"base mensal nao encontrada em {dir_base}.\n"
            "Rode antes: python scripts/gerar_base_inicial.py"
        )
    hist = pd.concat((pd.read_csv(a) for a in arquivos), ignore_index=True)
    hist["periodo_demanda"] = pd.to_datetime(hist["periodo_demanda"])
    ini_base, fim_base = hist["periodo_demanda"].min(), hist["periodo_demanda"].max()

    desde = (ini_base if MESES_HISTORICO_NECESSARIO is None
             else ate - pd.DateOffset(months=MESES_HISTORICO_NECESSARIO))
    partes = [hist[(hist.periodo_demanda >= desde) & (hist.periodo_demanda <= ate)]]
    origem = {m: "historico" for m in sorted(partes[0].periodo_demanda.unique())}

    if ate > fim_base:
        # meses ja consolidados pela propria execucao semanal
        consolidados = {}
        for a in sorted((cfg.BASE_DIR / "data" / "semanal").glob("*/fato_demanda_aberta_mensal.csv")):
            d = pd.read_csv(a)
            d["periodo_demanda"] = pd.to_datetime(d["periodo_demanda"])
            for m, g in d.groupby("periodo_demanda"):
                if fim_base < m <= ate:
                    consolidados[m] = g
        # meses que faltam (inclusive o mes alvo) -> criados pelo modelo
        estado = carregar_estado(f"{ini_base:%Y-%m}", f"{fim_base:%Y-%m}")
        criados = gerar_meses(estado, _date(ate.year, ate.month, 1))
        m = fim_base + pd.DateOffset(months=1)
        while m <= ate:
            if m in consolidados and m != ate:
                partes.append(consolidados[m]); origem[m] = "consolidado_semanal"
            else:
                partes.append(criados[criados.periodo_demanda == m]); origem[m] = "criado_pelo_modelo"
            m += pd.DateOffset(months=1)

    df = pd.concat(partes, ignore_index=True)
    df = df[df.periodo_demanda >= desde]
    cols = ["data_atualizacao_base", "periodo_demanda", "id_produto", "id_outlet", "id_uf",
            "demanda_unidades"]
    df = df[[c for c in cols if c in df.columns]]
    # ordem estavel: o rateio semanal sorteia ruido linha a linha
    df = df.sort_values(["periodo_demanda", "id_produto", "id_outlet"]).reset_index(drop=True)
    return df, {pd.Timestamp(k): v for k, v in origem.items() if pd.Timestamp(k) >= desde}


# ============================================================================
# Execucao
# ============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Roda o pipeline semanal para uma semana comercial especifica",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
exemplos:
  python scripts/executar_semana.py
      roda a proxima semana depois da ultima calculada

  python scripts/executar_semana.py --hoje
      roda a semana comercial que contem hoje

  python scripts/executar_semana.py --data 2026-03-15
      roda a semana que contem 15/03/2026

  python scripts/executar_semana.py --ano 2026 --mes 3 --semana 2
      roda a 2a semana comercial de marco/2026

  python scripts/executar_semana.py --listar-semanas 2026-03
      so mostra o calendario de marco/2026 e sai
        """,
    )
    ap.add_argument("--data", help="qualquer data dentro da semana alvo (AAAA-MM-DD)")
    ap.add_argument("--hoje", action="store_true", help="a semana comercial que contem hoje")
    ap.add_argument("--ano", type=int)
    ap.add_argument("--mes", type=int)
    ap.add_argument("--semana", type=int, help="numero da semana comercial no mes")
    ap.add_argument("--listar-semanas", metavar="AAAA-MM",
                    help="mostra as semanas comerciais do mes e sai")
    ap.add_argument("--ultima-semana", action="store_true",
                    help="mostra a ultima semana com dados em fato_analise_semanal "
                         "e a proxima a rodar, e sai")
    ap.add_argument("--dir-base", type=Path, default=None,
                    help=f"pasta com a base mensal (default: {cfg.DIR_SAIDA})")
    ap.add_argument("--dir-saida", type=Path, default=None,
                    help="pasta de saida (default: data/semanal/AAAA-MM-SEMANA)")
    ap.add_argument("--classifica-kpis", type=Path, default=None,
                    help="CSV de dim_classifica_kpis (default: procura em dir-base)")
    ap.add_argument("--regras-diagnostico", type=Path, default=None,
                    help="CSV de dim_regras_diagnostico (opcional: anexa o texto)")
    ap.add_argument("--carregar", action="store_true",
                    help="alem do CSV, grava no Supabase")
    args = ap.parse_args()

    dir_semanal = cfg.BASE_DIR / "data" / "semanal"
    if args.listar_semanas:
        listar_semanas(args.listar_semanas, args.dir_base or cfg.DIR_SAIDA, dir_semanal)
        return 0
    if args.ultima_semana:
        return mostrar_ultima_semana(args.dir_base or cfg.DIR_SAIDA, dir_semanal)

    if any([args.ano, args.mes, args.semana]) and not all([args.ano, args.mes, args.semana]):
        raise SystemExit("--ano, --mes e --semana precisam ser usados juntos.")

    sem = resolver_semana(args)
    dir_base = args.dir_base or cfg.DIR_SAIDA
    dir_saida = args.dir_saida or (cfg.BASE_DIR / "data" / "semanal" /
                                   f"{sem['ano']}-{sem['mes']:02d}-S{sem['nr_semana']}")
    rotulo = f"{sem['ano']}-{sem['mes']:02d} semana {sem['nr_semana']}"

    log.info("=" * 68)
    log.info(f"SEMANA ALVO: {rotulo}")
    log.info(f"  periodo      : {sem['inicio']:%d/%m/%Y} a {sem['fim']:%d/%m/%Y} "
             f"({sem['nr_dias']} dias)")
    log.info(f"  periodo_demanda (chave no banco): {sem['periodo_demanda']:%Y-%m-%d}")
    log.info(f"  primeira do mes: {'SIM - calcula cota e KPI' if sem['primeira_semana'] else 'nao'}")
    log.info(f"  ultima do mes  : {'SIM - consolida o mes' if sem['ultima_semana'] else 'nao'}")
    log.info("=" * 68)

    calib = carregar_calibracao()
    dims = montar_dimensoes(sorted(calib["peso_territorio"]["proprios"].keys()))
    proprios = dims.produtos.loc[dims.produtos.tipo_produto == "PROPRIO", "id_produto"]

    # ---------------- base mensal ------------------------------------------
    with etapa(log, "leitura_base_mensal", rotulo) as ctx:
        base, origem = carregar_base_mensal(dir_base, sem["periodo_mensal"])
        ctx["registros"] = len(base)
        fonte_mes = origem.get(sem["periodo_mensal"], "?")
        log.info(f"  demanda de {sem['periodo_mensal']:%m/%Y}: {fonte_mes.replace('_', ' ')}")
        criados = [f"{m:%m/%Y}" for m, o in origem.items() if o == "criado_pelo_modelo"]
        if len(criados) > 1:
            log.warning(f"  meses nao consolidados por execucao semanal, criados agora: {criados}")

    saidas: dict[str, pd.DataFrame] = {}

    # ---------------- ETAPA 2: cota (so na 1a semana) ----------------------
    with etapa(log, "etapa2_cota", rotulo) as ctx:
        cota = calcular_cota_mensal(agregar_por_setor(base, dims))
        cota_mes = cota[cota.periodo_demanda == sem["periodo_mensal"]]
        ctx["registros"] = len(cota_mes)
        if sem["primeira_semana"]:
            saidas["fato_cota_mensal"] = cota_mes
            log.info(f"  cota do mes calculada: {len(cota_mes):,} linhas (setor x produto)")
        else:
            log.info(f"  nao e a 1a semana - cota do mes ja vigente ({len(cota_mes):,} linhas), "
                     "nao sera reexportada")

    # ---------------- ETAPA 4: KPI do mes anterior (so na 1a semana) -------
    mes_anterior = sem["periodo_mensal"] - pd.DateOffset(months=1)
    with etapa(log, "etapa4_kpi", rotulo) as ctx:
        # O KPI usado na semana e o do MES ANTERIOR. Ele e calculado na 1a
        # semana (sobre o historico completo, sem o mes corrente, que ainda
        # nao fechou) e reaproveitado nas semanas seguintes.
        kpi_s1 = (cfg.BASE_DIR / "data" / "semanal" /
                  f"{sem['ano']}-{sem['mes']:02d}-S1" / "fato_analise_kpi.csv")
        if not sem["primeira_semana"] and kpi_s1.exists():
            kpi = pd.read_csv(kpi_s1, parse_dates=["periodo_demanda", "data_atualizacao_base"])
            log.info(f"  KPIs de {mes_anterior:%m/%Y} reaproveitados da 1a semana")
        else:
            kpi = calcular_analise_kpi(base[base.periodo_demanda < sem["periodo_mensal"]], dims)
        if sem["primeira_semana"]:
            kpi_mes = kpi[kpi.periodo_demanda == mes_anterior]
            saidas["fato_analise_kpi"] = kpi_mes
            ctx["registros"] = len(kpi_mes)
            log.info(f"  KPIs de {mes_anterior:%m/%Y}: {len(kpi_mes):,} linhas")
        else:
            ctx["registros"] = 0
            log.info("  nao e a 1a semana - KPIs do mes anterior ja calculados")

    # ---------------- ETAPA 4b: kpi_final do mes anterior (base do Agente 3)
    faixas_kpi_csv = args.classifica_kpis or _achar(
        dir_base, ["dim_classifica_kpis.csv", "dim_classifica_kpis_SUGERIDO.csv"]
    )
    if sem["primeira_semana"] and faixas_kpi_csv:
        from src.sintetico.kpi_final import montar_kpi_final
        with etapa(log, "etapa4b_kpi_final", rotulo) as ctx:
            kpi_final = montar_kpi_final(saidas["fato_analise_kpi"], dims.alocacao,
                                         pd.read_csv(faixas_kpi_csv))
            saidas["fato_analise_kpi_final"] = kpi_final
            ctx["registros"] = len(kpi_final)
            log.info(f"  kpi_final de {mes_anterior:%m/%Y}: {len(kpi_final):,} linhas "
                     "(acrescentar a base de treino do Agente 3)")

    # ---------------- ETAPA 1: demanda semanal -----------------------------
    with etapa(log, "etapa1_demanda_semanal", rotulo) as ctx:
        base_mes = base[base.periodo_demanda == sem["periodo_mensal"]]
        semanal_mes = desagregar_semanal(base_mes)
        semana = semanal_mes[semanal_mes.nr_semana == sem["nr_semana"]]
        saidas["fato_demanda_aberta_semanal"] = semana
        ctx["registros"] = len(semana)
        log.info(f"  demanda da semana: {semana.demanda_unidades.sum():,.0f} unidades")

    # ---------------- ETAPA 3: alocacao com acumulado ----------------------
    with etapa(log, "etapa3_alocacao", rotulo) as ctx:
        # o acumulado precisa das semanas 1..N do mes, por isso alocamos o mes
        # inteiro e depois recortamos a semana alvo
        alocado_mes = alocar_semanal(
            semanal_mes[semanal_mes.id_produto.isin(proprios)], dims
        )
        alocado = alocado_mes[alocado_mes.nr_semana == sem["nr_semana"]]
        saidas["fato_demanda_alocada_semanal"] = alocado
        ctx["registros"] = len(alocado)
        log.info(f"  acumulado ate a semana {sem['nr_semana']}: "
                 f"{alocado.demanda_unidades_acumulada.sum():,.0f} unidades")

    # ---------------- ETAPA 5: desempenho ----------------------------------
    with etapa(log, "etapa5_desempenho", rotulo) as ctx:
        desempenho_mes = calcular_desempenho(alocado_mes, cota, base, dims)
        desempenho = desempenho_mes[desempenho_mes.nr_semana == sem["nr_semana"]]
        saidas["fato_desempenho_semanal"] = desempenho
        ctx["registros"] = len(desempenho)
        if len(desempenho):
            log.info(f"  cobertura projetada: mediana "
                     f"{desempenho.cobertura_cota_projetada.median():.3f}")

    # ---------------- ETAPA 6: analise semanal -----------------------------
    with etapa(log, "etapa6_analise", rotulo) as ctx:
        analise = montar_analise_semanal(desempenho, kpi, dims)
        saidas["fato_analise_semanal"] = analise
        ctx["registros"] = len(analise)

    # ---------------- ETAPA 7: classificacao -------------------------------
    caminho_faixas = args.classifica_kpis or _achar(
        dir_base, ["dim_classifica_kpis.csv", "dim_classifica_kpis_SUGERIDO.csv"]
    )
    if caminho_faixas:
        with etapa(log, "etapa7_classificacao", rotulo) as ctx:
            faixas = pd.read_csv(caminho_faixas)
            final = classificar(analise, faixas)

            caminho_regras = args.regras_diagnostico or _achar(
                dir_base, ["dim_regras_diagnostico.csv",
                           "dim_regras_diagnostico_SUGERIDO.csv"]
            )
            if caminho_regras:
                com_texto = aplicar_diagnostico(final, pd.read_csv(caminho_regras))
                cobertura = com_texto["diagnostico"].notna().mean()
                log.info(f"  diagnostico atribuido a {cobertura:.1%} das linhas")
                saidas["_analise_semanal_com_diagnostico"] = com_texto

            saidas["fato_analise_semanal_final"] = final
            ctx["registros"] = len(final)
    else:
        log.warning("dim_classifica_kpis nao encontrada - etapa 7 pulada. "
                    "Use --classifica-kpis CAMINHO.csv")

    # ---------------- consolidacao mensal (so na ultima semana) ------------
    if sem["ultima_semana"]:
        with etapa(log, "consolidacao_mensal", rotulo) as ctx:
            consolidado = (
                semanal_mes.groupby(
                    ["periodo_mensal", "id_produto", "id_outlet", "id_uf"], as_index=False
                )["demanda_unidades"].sum()
                .rename(columns={"periodo_mensal": "periodo_demanda"})
            )
            consolidado["data_atualizacao_base"] = (
                sem["periodo_demanda"] + pd.Timedelta(days=cfg.LATENCIA_DIAS)
            )
            esperado = base_mes.demanda_unidades.sum()
            obtido = consolidado.demanda_unidades.sum()
            if abs(esperado - obtido) > 1:
                log.error(f"  consolidacao NAO reconcilia: {obtido:,.0f} vs {esperado:,.0f}")
            else:
                log.info(f"  consolidacao reconcilia: {obtido:,.0f} unidades")
            saidas["fato_demanda_aberta_mensal"] = consolidado
            ctx["registros"] = len(consolidado)

    # ---------------- exportacao -------------------------------------------
    dir_saida.mkdir(parents=True, exist_ok=True)
    with etapa(log, "exportacao", rotulo) as ctx:
        total = 0
        for nome, df in saidas.items():
            if nome.startswith("_"):
                df.to_csv(dir_saida / f"{nome.lstrip('_')}.csv", index=False)
                continue
            metas = exportar(df, nome, dir_saida)
            total += sum(m["linhas"] for m in metas)
        ctx["registros"] = total

    # ---------------- carga opcional ---------------------------------------
    if args.carregar:
        from src.sintetico.carga import carregar_tabela
        from src.sintetico.exportacao import COLUNAS_POR_TABELA
        with etapa(log, "carga_supabase", rotulo):
            for nome, df in saidas.items():
                if nome.startswith("_"):
                    continue
                carregar_tabela(df, nome, COLUNAS_POR_TABELA[nome])

    print(f"\nSemana {rotulo} processada.")
    print(f"periodo_demanda = {sem['periodo_demanda']:%Y-%m-%d}")
    print(f"\nArquivos em {dir_saida}:")
    for nome, df in saidas.items():
        print(f"  {nome.lstrip('_'):<34s} {len(df):>9,} linhas")
    return 0


def _achar(pasta: Path, nomes: list[str]) -> Path | None:
    for n in nomes:
        p = Path(pasta) / n
        if p.exists():
            return p
    return None


if __name__ == "__main__":
    raise SystemExit(main())
