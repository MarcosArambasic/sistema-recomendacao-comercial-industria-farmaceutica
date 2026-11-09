"""***************************************************************************************
SCRIPT    : exportacao.py
OBJETIVO  : Exportar a base sintetica para CSV LOCAL, pronto para importacao
            posterior no PostgreSQL/Supabase.

            Este e o caminho PADRAO do pipeline. A carga direta no banco
            (carga.py) existe, mas so roda com a flag --carregar.

DECISOES DE FORMATO (importam para o import funcionar de primeira):
  * Colunas na ORDEM EXATA da tabela de destino - assim o \\copy funciona
    sem precisar listar coluna por coluna.
  * Datas como texto AAAA-MM-DD, sem hora. O pandas escreveria
    "2023-01-01 00:00:00"; o Postgres aceita, mas polui o arquivo e cresce
    o tamanho a toa.
  * demanda_unidades gravada como INTEIRO. Os valores ja sao inteiros
    (arredondados na geracao); escrever "16" em vez de "16.0" reduz o
    arquivo e o tempo de escrita. A coluna no banco continua float e
    aceita inteiro normalmente.
  * Sem indice do pandas.
  * Particionamento opcional por ano: o painel do Supabase engasga com
    arquivo unico de ~50 MB, e 4 arquivos de ~13 MB sobem sem problema.

AUDITORIA: cada exportacao gera um manifesto (JSON) com contagem de linhas,
tamanho e SHA-256 de cada arquivo, para conferir depois se o que entrou no
banco e exatamente o que saiu daqui.

AUTOR     : Marcos Arambasic
DATA      : AGO/2026
****************************************************************************************/
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import config as cfg
from .log_conf import configurar

log = configurar(__name__)


# ============================================================================
# Contrato de colunas - tem que espelhar o DDL (criacao_banco_v2.sql)
# ============================================================================
COLUNAS_POR_TABELA: dict[str, list[str]] = {
    "fato_demanda_aberta_mensal": [
        "data_atualizacao_base", "periodo_demanda",
        "id_produto", "id_outlet", "id_uf", "demanda_unidades",
    ],
    "fato_cota_mensal": [
        "periodo_demanda", "id_setor", "id_produto", "cota_unidades",
    ],
    "fato_analise_kpi": [
        "data_atualizacao_base", "periodo_demanda",
        "id_produto", "grupo", "id_mercado", "id_outlet", "id_uf",
        "demanda_produto", "demanda_mercado",
        "demanda_media_produto", "demanda_media_mercado",
        "demanda_produto_total", "demanda_mercado_total",
        "mrkt_share", "cagr_medio", "cv", "indice_sazonal",
    ],
    "fato_analise_kpi_final": [
        "periodo_demanda", "nr_ano", "nr_mes",
        "id_produto", "grupo", "id_mercado", "id_setor", "id_alocacao", "id_outlet", "id_uf",
        "demanda_produto", "demanda_mercado",
        "mrkt_share", "cagr_medio", "cv", "indice_sazonal",
        "cd_mrkt_share", "cd_cagr_medio", "cd_cv", "cd_indice_sazonal",
    ],
    "fato_demanda_aberta_semanal": [
        "data_atualizacao_base", "periodo_demanda", "nr_semana",
        "id_produto", "id_outlet", "id_uf", "demanda_unidades",
    ],
    "fato_demanda_alocada_semanal": [
        "data_atualizacao_base", "periodo_demanda", "nr_semana",
        "id_produto", "id_alocacao", "id_setor", "demanda_unidades",
    ],
    "fato_desempenho_semanal": [
        "periodo_demanda", "nr_semana", "id_produto", "id_setor", "id_alocacao",
        "demanda_unidades_acumulada", "demanda_unidades_projetada", "cota_mensal",
        "cobertura_cota", "cobertura_cota_projetada",
    ],
    "fato_analise_semanal": [
        "data_atualizacao_base", "periodo_demanda", "nr_semana",
        "id_produto", "id_mercado", "id_setor", "id_alocacao", "id_uf",
        "cobertura_cota_projetada", "mrkt_share", "cagr_medio", "cv", "indice_sazonal",
    ],
    "fato_analise_semanal_final": [
        "nr_ano", "nr_mes", "nr_semana",
        "id_produto", "id_mercado", "id_setor", "id_alocacao", "id_uf",
        "cd_cobertura_cota_projetada", "cd_mrkt_share", "cd_cagr_medio",
        "cd_cv", "cd_indice_sazonal",
    ],
}

# Renomeacoes necessarias entre o nome usado no pipeline e o nome da coluna
# no banco. Caso relevante: na fato_demanda_alocada_semanal, a coluna
# 'demanda_unidades' do DDL guarda o valor ACUMULADO no mes (o proprio
# comentario da tabela diz "acumulando semana anterior").
RENOMEAR_PARA_BANCO: dict[str, dict[str, str]] = {
    "fato_demanda_alocada_semanal": {"demanda_unidades_acumulada": "demanda_unidades"},
}

# colunas que devem sair como data pura (AAAA-MM-DD)
COLUNAS_DATA = {"periodo_demanda", "data_atualizacao_base"}

# colunas de contagem que devem sair como inteiro
COLUNAS_INTEIRAS = {"demanda_unidades", "demanda_unidades_acumulada"}

# casas decimais dos floats no CSV
FORMATO_FLOAT = "%.6f"

# Faixa aceitavel para qualquer coluna de data exportada. Serve de rede de
# seguranca: datas fora disso indicam conversao de tipo corrompida, nao dado
# legitimo (o periodo do projeto e 2023-2026).
ANO_MINIMO_ACEITAVEL = pd.Timestamp("2000-01-01")
ANO_MAXIMO_ACEITAVEL = pd.Timestamp("2100-01-01")


def preparar_para_csv(df: pd.DataFrame, tabela: str) -> pd.DataFrame:
    """
    Deixa o DataFrame no formato exato de importacao: colunas certas, na
    ordem certa, com datas e numeros formatados para o Postgres.

    Levanta erro se faltar alguma coluna obrigatoria - melhor falhar aqui do
    que descobrir na hora do \\copy.
    """
    colunas = COLUNAS_POR_TABELA.get(tabela)
    if colunas is None:
        raise ValueError(
            f"Tabela '{tabela}' nao mapeada em COLUNAS_POR_TABELA. "
            f"Conhecidas: {sorted(COLUNAS_POR_TABELA)}"
        )

    out = df.copy()

    # aplica as renomeacoes pipeline -> banco, se houver.
    # Se o nome de destino ja existir no DataFrame, a coluna original e
    # descartada primeiro - caso contrario ficariam duas colunas homonimas e
    # df[coluna] passaria a devolver DataFrame em vez de Series.
    # Caso concreto: em fato_demanda_alocada_semanal o pipeline carrega tanto
    # 'demanda_unidades' (da semana) quanto 'demanda_unidades_acumulada';
    # a coluna do banco guarda a ACUMULADA.
    renomear = RENOMEAR_PARA_BANCO.get(tabela)
    if renomear:
        efetivas = {k: v for k, v in renomear.items() if k in out.columns}
        conflitos = [v for v in efetivas.values() if v in out.columns]
        if conflitos:
            out = out.drop(columns=conflitos)
        out = out.rename(columns=efetivas)

    faltando = [c for c in colunas if c not in out.columns]
    if faltando:
        raise ValueError(f"{tabela}: colunas ausentes no DataFrame: {faltando}")

    out = out[colunas].copy()

    for c in colunas:
        if c in COLUNAS_DATA:
            datas = pd.to_datetime(out[c])

            # Guarda contra datas corrompidas. Ja aconteceu de um round-trip
            # datetime -> int64 -> datetime zerar tudo para 1970-01-01 (epoch)
            # em pandas 3.x, e o CSV saiu silenciosamente errado. Melhor
            # quebrar aqui do que carregar lixo no banco.
            fora = datas[(datas < ANO_MINIMO_ACEITAVEL) | (datas > ANO_MAXIMO_ACEITAVEL)]
            if len(fora):
                raise ValueError(
                    f"{tabela}.{c}: {len(fora)} datas fora da faixa esperada "
                    f"({ANO_MINIMO_ACEITAVEL.date()} a {ANO_MAXIMO_ACEITAVEL.date()}). "
                    f"Exemplos: {sorted(fora.unique())[:3]}. "
                    "Sinal classico de conversao de tipo perdida."
                )

            out[c] = datas.dt.strftime("%Y-%m-%d")
        elif c in COLUNAS_INTEIRAS:
            # astype('int64') estoura com NaN. A geracao nao produz nulos
            # (validado), mas se algum aparecer e melhor falhar com mensagem
            # clara do que gravar um CSV silenciosamente corrompido.
            n_nulos = int(out[c].isna().sum())
            if n_nulos:
                raise ValueError(
                    f"{tabela}.{c}: {n_nulos} valores nulos - nao da para gravar "
                    "como inteiro. Investigue a geracao antes de exportar."
                )
            out[c] = out[c].round(0).astype("int64")

    return out


# ============================================================================
# Exportacao
# ============================================================================
def _sha256(caminho: Path, bloco: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for pedaco in iter(lambda: f.read(bloco), b""):
            h.update(pedaco)
    return h.hexdigest()


def _escrever(df: pd.DataFrame, destino: Path, comprimir: bool) -> dict:
    destino.parent.mkdir(parents=True, exist_ok=True)

    # Limita as casas decimais dos floats. Sem isso o pandas grava a
    # representacao completa (ex.: 0.07638888888888888), o que infla o
    # arquivo e o tempo de escrita sem nenhum ganho analitico - indicadores
    # como market share e CV nao tem significado alem da 6a casa.
    kwargs = {"index": False, "encoding": "utf-8", "float_format": FORMATO_FLOAT}

    if comprimir:
        destino = destino.with_suffix(destino.suffix + ".gz")
        df.to_csv(destino, compression="gzip", **kwargs)
    else:
        df.to_csv(destino, **kwargs)

    return {
        "arquivo": destino.name,
        "caminho": str(destino),
        "linhas": int(len(df)),
        "tamanho_mb": round(destino.stat().st_size / 1024 ** 2, 2),
        "sha256": _sha256(destino),
    }


def exportar(
    df: pd.DataFrame,
    tabela: str,
    dir_saida: Path | None = None,
    particionar_por_ano: bool = False,
    comprimir: bool = False,
) -> list[dict]:
    """
    Exporta um DataFrame para CSV local no formato de importacao.

    particionar_por_ano=True gera um arquivo por ano
    (ex.: fato_demanda_aberta_mensal_2023.csv), o que facilita subir pelo
    painel do Supabase, que tem limite pratico de tamanho por upload.

    Retorna a lista de metadados dos arquivos gerados.
    """
    dir_saida = Path(dir_saida or cfg.DIR_SAIDA)
    pronto = preparar_para_csv(df, tabela)

    if not particionar_por_ano:
        meta = _escrever(pronto, dir_saida / f"{tabela}.csv", comprimir)
        log.info(f"{tabela}: {meta['linhas']:,} linhas -> {meta['arquivo']} ({meta['tamanho_mb']} MB)")
        return [meta]

    # particiona pelo ano do periodo_demanda (ja e string AAAA-MM-DD)
    anos = pronto["periodo_demanda"].str.slice(0, 4)
    metas = []
    for ano in sorted(anos.unique()):
        parte = pronto[anos == ano]
        meta = _escrever(parte, dir_saida / f"{tabela}_{ano}.csv", comprimir)
        meta["ano"] = ano
        metas.append(meta)
        log.info(f"{tabela} {ano}: {meta['linhas']:,} linhas -> {meta['arquivo']} "
                 f"({meta['tamanho_mb']} MB)")
    return metas


def gerar_manifesto(exportacoes: dict[str, list[dict]], diagnostico: dict,
                    dir_saida: Path | None = None) -> Path:
    """
    Manifesto JSON da exportacao: o que foi gerado, quantas linhas, checksum.
    Serve para conferir depois se o que esta no banco bate com o que saiu daqui.
    """
    dir_saida = Path(dir_saida or cfg.DIR_SAIDA)
    manifesto = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "seed": diagnostico.get("seed"),
        "periodo": diagnostico.get("periodo"),
        "parametros_geracao": diagnostico,
        "tabelas": {
            tabela: {
                "arquivos": metas,
                "linhas_total": sum(m["linhas"] for m in metas),
                "tamanho_mb_total": round(sum(m["tamanho_mb"] for m in metas), 2),
                "colunas": COLUNAS_POR_TABELA[tabela],
            }
            for tabela, metas in exportacoes.items()
        },
    }
    destino = dir_saida / "manifesto_exportacao.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"manifesto: {destino}")
    return destino


def gerar_instrucoes_importacao(exportacoes: dict[str, list[dict]],
                                dir_saida: Path | None = None) -> Path:
    """
    Gera um .md com os comandos prontos de importacao (psql \\copy e painel
    do Supabase), ja com os nomes de arquivo e a ordem de colunas corretos.
    """
    dir_saida = Path(dir_saida or cfg.DIR_SAIDA)
    linhas = [
        "# Como importar os CSVs no Supabase",
        "",
        "Arquivos gerados por `scripts/gerar_base_inicial.py`.",
        "",
        "> Ordem obrigatoria: as tabelas de dimensao e a `fato_alocacao` precisam",
        "> estar populadas ANTES (via `supabase/scripts/popular_banco.sql`), senao",
        "> as foreign keys rejeitam a carga.",
        "",
        "## Resumo do que foi gerado",
        "",
        "| Tabela | Arquivos | Linhas | Tamanho |",
        "|---|---|---|---|",
    ]
    for tabela, metas in exportacoes.items():
        linhas.append(
            f"| `{tabela}` | {len(metas)} | {sum(m['linhas'] for m in metas):,} | "
            f"{round(sum(m['tamanho_mb'] for m in metas), 2)} MB |"
        )

    linhas += [
        "",
        "## Opcao A - psql (recomendada)",
        "",
        "Mais rapida e a unica confiavel para arquivos grandes. O `\\copy` roda",
        "no cliente, entao o caminho do arquivo e o da SUA maquina.",
        "",
        "```bash",
        "psql \"$DATABASE_URL\"",
        "```",
        "",
        "Ja dentro do psql:",
        "",
        "```sql",
    ]

    tem_gz = any(m["arquivo"].endswith(".gz")
                 for metas in exportacoes.values() for m in metas)

    for tabela, metas in exportacoes.items():
        cols = ", ".join(COLUNAS_POR_TABELA[tabela])
        for m in metas:
            arq = m["arquivo"]
            if arq.endswith(".gz"):
                # psql nao le gzip direto; PROGRAM descomprime no caminho
                origem = f"PROGRAM 'gzip -dc \"{arq}\"'"
            else:
                origem = f"'{arq}'"
            linhas.append(
                f"\\copy public.{tabela} ({cols}) FROM {origem} WITH (FORMAT csv, HEADER true);"
            )
    linhas += ["```", ""]

    if tem_gz:
        linhas += [
            "> Os arquivos estao compactados, entao o comando usa `PROGRAM 'gzip -dc'`",
            "> em vez de ler o arquivo direto - o psql nao abre `.gz` sozinho.",
            "> No Windows sem `gzip` no PATH, descompacte antes e troque pelo nome",
            "> do `.csv`, ou rode pelo Git Bash / WSL.",
            "",
        ]

    linhas += [
        "## Opcao B - painel do Supabase",
        "",
        "Table Editor -> selecione a tabela -> Insert -> Import data from CSV.",
        "",
        "Limitacoes conhecidas: fica lento acima de ~10 MB e pode expirar. Por isso",
        "vale gerar particionado por ano (`--particionar`), que produz arquivos",
        "menores, e subir um de cada vez.",
        "",
        "## Reimportacao (idempotencia)",
        "",
        "O `\\copy` sempre INSERE - rodar duas vezes duplica. Antes de reimportar,",
        "limpe o intervalo:",
        "",
        "```sql",
        "-- apaga so o periodo que sera recarregado",
        "delete from public.fato_demanda_aberta_mensal",
        " where periodo_demanda between '2023-01-01' and '2026-08-01';",
        "",
        "delete from public.fato_cota_mensal",
        " where periodo_demanda between '2023-01-01' and '2026-08-01';",
        "```",
        "",
        "Ou, se ja tiver aplicado `alter_schema_constraints.sql` (que cria as",
        "constraints UNIQUE de grao), da para importar numa tabela temporaria e",
        "fazer `INSERT ... ON CONFLICT DO UPDATE`.",
        "",
        "## Conferencia pos-importacao",
        "",
        "```sql",
        "select 'fato_demanda_aberta_mensal' as tabela, count(*) from public.fato_demanda_aberta_mensal",
        "union all",
        "select 'fato_cota_mensal', count(*) from public.fato_cota_mensal;",
        "```",
        "",
        "Compare com as contagens do `manifesto_exportacao.json`. O manifesto tambem",
        "traz o SHA-256 de cada arquivo, caso precise verificar integridade.",
    ]

    destino = dir_saida / "INSTRUCOES_IMPORTACAO.md"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("\n".join(linhas), encoding="utf-8")
    log.info(f"instrucoes de importacao: {destino}")
    return destino


def exportar_tudo(df_demanda: pd.DataFrame, df_cota: pd.DataFrame,
                  diagnostico: dict, dir_saida: Path | None = None,
                  particionar_por_ano: bool = False,
                  comprimir: bool = False) -> dict[str, list[dict]]:
    """Exporta as duas tabelas + manifesto + instrucoes, numa chamada so."""
    exportacoes = {
        "fato_demanda_aberta_mensal": exportar(
            df_demanda, "fato_demanda_aberta_mensal", dir_saida,
            particionar_por_ano, comprimir,
        ),
        "fato_cota_mensal": exportar(
            df_cota, "fato_cota_mensal", dir_saida,
            particionar_por_ano, comprimir,
        ),
    }
    gerar_manifesto(exportacoes, diagnostico, dir_saida)
    gerar_instrucoes_importacao(exportacoes, dir_saida)
    return exportacoes


if __name__ == "__main__":
    from .cota import adicionar_metadados, calcular_cota_mensal
    from .gerador import agregar_por_setor, gerar_demanda_mensal

    df, dims, diag = gerar_demanda_mensal()
    cota = calcular_cota_mensal(agregar_por_setor(df, dims))
    exportacoes = exportar_tudo(df, cota, diag, particionar_por_ano=True)

    for tabela, metas in exportacoes.items():
        total = sum(m["linhas"] for m in metas)
        mb = round(sum(m["tamanho_mb"] for m in metas), 2)
        print(f"{tabela}: {len(metas)} arquivo(s), {total:,} linhas, {mb} MB")
