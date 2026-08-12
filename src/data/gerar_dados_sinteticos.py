"""***************************************************************************************
SCRIPT    : gerar_dados_sinteticos.py
OBJETIVO  : Gerar base farmaceutica sintetica (produtos, mercados, territorios, outlets,
            representantes e series mensais de vendas/metas/market share) parametrizavel,
            sem usar nenhum dado real da empresa. Preencher o bloco CONFIG abaixo com os
            valores levantados na planilha 'Parametros_Dados_Sinteticos.xlsx' (agregados,
            nunca dados brutos/identificaveis).

            Gera dois niveis de saida:
              1) Tabelas relacionais no grao do schema (supabase/scripts/criacao_banco.sql):
                 dim_produto, dim_mercado, produto_mercado, dim_setor, dim_outlet,
                 dim_representante, fato_objetivo_mensal, fato_mercado_mensal.
              2) CSV achatado (vendas_sinteticas.csv) no grao produto x territorio x mes,
                 com exatamente as colunas que src/agents/agente3_comparador_vendas.py
                 espera: produto, territorio, representante, data, mes, meta_mil_r$,
                 market_share_relativo, tendencia_prescricao, desvio_pct.

AUTOR     : Marcos Arambasic (gerado com apoio de Claude)
DATA      : AGO/2026
--------------------------------ALTERACOES----------------------------------------------
DATA       AUTOR       OBJETIVO
12/08/2026 Claude      Criacao inicial do gerador parametrizavel
****************************************************************************************/
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from faker import Faker
    FAKER_DISPONIVEL = True
except ImportError:
    FAKER_DISPONIVEL = False


# ============================================================================
# CONFIG — preencha com os valores da planilha Parametros_Dados_Sinteticos.xlsx
# Cada bloco abaixo corresponde a uma aba da planilha. Valores nao preenchidos
# ficam no default (plausivel, mas generico) e o gerador roda normalmente.
# ============================================================================
CONFIG = {
    "seed": 42,

    "periodo": {
        "inicio": "2025-01",   # AAAA-MM
        "meses": 12,
    },

    "saida": {
        "diretorio": "data/dados_sinteticos",
        "gerar_tabelas_relacionais": True,   # dim_*/fato_* no grao do schema SQL
        "gerar_csv_plano": True,             # vendas_sinteticas.csv para o Agente 3
    },

    # --- Aba 1: Estrutura Dimensional ---------------------------------------
    "estrutura": {
        "n_produtos": 30,
        "pct_produto_proprio": 0.40,
        "n_mercados": 5,
        "n_territorios": 15,
        "outlets_por_territorio": (8, 20),      # (min, max) por territorio
        "representantes_por_territorio": 1,     # geralmente 1:1 territorio-representante
        "canais": ["Farmacia", "Hospital", "Drogaria"],
        "subcanais": ["Independente", "Rede", "Popular"],
        "ufs": ["SP", "RJ", "MG", "PR", "RS", "BA", "SC", "GO", "PE", "CE"],
        "pct_outlet_cnpj": 0.70,
    },

    # --- Aba 2: Distribuicoes Numericas -------------------------------------
    # media / dp_pct (desvio como % da media) / min / max quando fizer sentido limitar
    "distribuicoes": {
        "demanda_mercado_mensal": {"media": 5000, "dp_pct": 0.24, "min": 500, "max": 30000},
        "market_share_produto": {"media": 0.18, "dp": 0.05, "min": 0.01, "max": 0.60},
        "preco_medio": {"media": 45.0, "dp_pct": 0.08},
        "objetivo_mensal_unid": {"media": 1000, "dp_pct": 0.15},
        "investimento_promocional": {"media": 8000, "dp_pct": 0.25},
        "estoque_disponivel": {"media": 400, "dp_pct": 0.30},
    },

    # --- Aba 3: Sazonalidade e Tendencia ------------------------------------
    "sazonalidade": {  # indice por mes-calendario, 1.0 = media do ano
        1: 0.95, 2: 0.97, 3: 1.00, 4: 1.02, 5: 1.05, 6: 1.03,
        7: 1.00, 8: 0.98, 9: 0.96, 10: 0.99, 11: 1.04, 12: 1.08,
    },
    "tendencia_crescimento_anual_pct": 0.05,

    # --- Aba 4: Regras de Negocio --------------------------------------------
    "regras_negocio": {
        "fator_ambicao_meta": 1.05,   # meta = valor esperado (sem ruido) * fator_ambicao
        "cota_semanas_por_mes": 4.33,
        "threshold_atencao": 0.90,
        "threshold_critico": 0.75,
        "limite_desvio_baixo": 10,     # |desvio_pct| < 10 -> magnitude baixa
        "limite_desvio_medio": 25,     # |desvio_pct| < 25 -> media; acima -> alta
    },

    # --- Aba 5: Correlacoes (0 a 1 = forca aproximada) ----------------------
    "correlacoes": {
        "market_share_vs_investimento_promocional": 0.40,
        "demanda_vs_estoque": 0.25,
    },

    # --- Aba 6: Qualidade e Cadencia -----------------------------------------
    "qualidade": {
        "pct_nulos": 0.02,
        "pct_outliers": 0.01,
    },
}


# ============================================================================
# Utilidades
# ============================================================================
def _seed_tudo(seed: int) -> tuple[np.random.Generator, "Faker | None"]:
    random.seed(seed)
    rng = np.random.default_rng(seed)
    faker = None
    if FAKER_DISPONIVEL:
        faker = Faker("pt_BR")
        Faker.seed(seed)
    return rng, faker


def _periodos(cfg: dict) -> list[pd.Timestamp]:
    inicio = pd.Period(cfg["periodo"]["inicio"], freq="M")
    n = cfg["periodo"]["meses"]
    return [(inicio + i).to_timestamp() for i in range(n)]


def _distribuir_proporcional(total: np.ndarray, n_partes: int, rng: np.random.Generator) -> np.ndarray:
    """Divide um array de totais em n_partes usando pesos Dirichlet (somam 1)."""
    pesos = rng.dirichlet(alpha=np.ones(n_partes), size=len(total))
    return total[:, None] * pesos


def _injetar_qualidade(df: pd.DataFrame, colunas: list[str], cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Aplica nulos e outliers nas colunas numericas indicadas, conforme CONFIG['qualidade']."""
    pct_nulos = cfg["qualidade"]["pct_nulos"]
    pct_outliers = cfg["qualidade"]["pct_outliers"]
    df = df.copy()
    n = len(df)
    for col in colunas:
        if pct_nulos > 0:
            idx_nulos = rng.choice(n, size=int(n * pct_nulos), replace=False)
            df.loc[df.index[idx_nulos], col] = np.nan
        if pct_outliers > 0:
            idx_out = rng.choice(n, size=int(n * pct_outliers), replace=False)
            fator = rng.choice([-1, 1], size=len(idx_out)) * rng.uniform(3, 6, size=len(idx_out))
            df.loc[df.index[idx_out], col] = df.loc[df.index[idx_out], col].astype(float) * (1 + fator / 10)
    return df


def _nome_sintetico_simples(prefixo: str, i: int, rng: np.random.Generator) -> str:
    """Fallback sem Faker: nomes sinteticos legiveis tipo 'Territorio Sintetico 07'."""
    return f"{prefixo} Sintetico {i:02d}"


# ============================================================================
# 1. Dimensoes
# ============================================================================
def gerar_dimensoes(cfg: dict, rng: np.random.Generator, faker) -> dict[str, pd.DataFrame]:
    est = cfg["estrutura"]

    # --- dim_produto ---------------------------------------------------
    n_prod = est["n_produtos"]
    n_proprio = int(round(n_prod * est["pct_produto_proprio"]))
    tipos = ["PROPRIO"] * n_proprio + ["CONCORRENTE"] * (n_prod - n_proprio)
    rng.shuffle(tipos)
    dim_produto = pd.DataFrame({
        "id_produto": range(1, n_prod + 1),
        "nome_sintetico": [f"Produto {t.title()} {i:02d}" for i, t in enumerate(tipos, start=1)],
        "tipo_produto": tipos,
    })

    # --- dim_mercado -----------------------------------------------------
    n_merc = est["n_mercados"]
    dim_mercado = pd.DataFrame({
        "id_mercado": range(1, n_merc + 1),
        "nome_sintetico": [f"Mercado Sintetico {i:02d}" for i in range(1, n_merc + 1)],
    })

    # --- produto_mercado (cada produto pertence a 1 mercado, atribuicao aleatoria) ---
    produto_mercado = pd.DataFrame({
        "id_produto": dim_produto["id_produto"],
        "id_mercado": rng.integers(1, n_merc + 1, size=n_prod),
    })

    # --- dim_setor (territorios) ------------------------------------------
    n_terr = est["n_territorios"]
    ufs = est["ufs"]
    dim_setor = pd.DataFrame({
        "id_setor": range(1, n_terr + 1),
        "codigo_setor": [f"TERR{i:03d}" for i in range(1, n_terr + 1)],
        "nome_sintetico": [f"Territorio Sintetico {i:02d}" for i in range(1, n_terr + 1)],
        "uf": rng.choice(ufs, size=n_terr),
    })
    # "potencial" relativo do territorio (fator multiplicativo de demanda) — log-normal
    dim_setor["potencial"] = rng.lognormal(mean=0.0, sigma=0.35, size=n_terr)
    dim_setor["potencial"] = dim_setor["potencial"] / dim_setor["potencial"].mean()  # normaliza em torno de 1.0

    # --- dim_outlet ---------------------------------------------------------
    outlets_rows = []
    outlet_id = 1
    min_out, max_out = est["outlets_por_territorio"]
    for _, setor in dim_setor.iterrows():
        n_outlets = rng.integers(min_out, max_out + 1)
        for _ in range(n_outlets):
            tipo_outlet = "CNPJ" if rng.random() < est["pct_outlet_cnpj"] else "BRICK"
            outlets_rows.append({
                "id_outlet": outlet_id,
                "codigo_outlet": f"OUT{outlet_id:06d}",
                "id_setor": setor["id_setor"],
                "tipo_outlet": tipo_outlet,
                "uf": setor["uf"],
                "cidade": faker.city() if faker else _nome_sintetico_simples("Cidade", outlet_id % 50, rng),
                "canal": rng.choice(est["canais"]),
                "subcanal": rng.choice(est["subcanais"]),
            })
            outlet_id += 1
    dim_outlet = pd.DataFrame(outlets_rows)

    # --- dim_representante (nao existe no schema SQL atual — ver observacao no README) ---
    reps_rows = []
    rep_id = 1
    n_reps_por_terr = est["representantes_por_territorio"]
    for _, setor in dim_setor.iterrows():
        for _ in range(n_reps_por_terr):
            nome = faker.name() if faker else _nome_sintetico_simples("Representante", rep_id, rng)
            reps_rows.append({"id_representante": rep_id, "nome_sintetico": nome, "id_setor": setor["id_setor"]})
            rep_id += 1
    dim_representante = pd.DataFrame(reps_rows)

    return {
        "dim_produto": dim_produto,
        "dim_mercado": dim_mercado,
        "produto_mercado": produto_mercado,
        "dim_setor": dim_setor,
        "dim_outlet": dim_outlet,
        "dim_representante": dim_representante,
    }


# ============================================================================
# 2. Fato mensal (grao territorio x produto x mes) + distribuicao por outlet
# ============================================================================
def gerar_fato_territorio(cfg: dict, dims: dict, periodos: list, rng: np.random.Generator) -> pd.DataFrame:
    dist = cfg["distribuicoes"]
    saz = cfg["sazonalidade"]
    crescimento_anual = cfg["tendencia_crescimento_anual_pct"]
    corr = cfg["correlacoes"]["market_share_vs_investimento_promocional"]

    # produto ja carregando seu id_mercado (via produto_mercado) para poder ser usado
    # nas chaves de fato_objetivo_mensal/fato_mercado_mensal, que sao por mercado tambem
    dim_produto = dims["dim_produto"].merge(dims["produto_mercado"], on="id_produto", how="left")
    dim_setor = dims["dim_setor"]

    # combinacao completa produto x territorio (todo produto vendido em todo territorio)
    base = dim_produto.merge(dim_setor, how="cross")

    # share "base" por produto (fixo ao longo do ano, com leve ruido por territorio)
    share_media = dist["market_share_produto"]["media"]
    share_dp = dist["market_share_produto"]["dp"]
    share_min, share_max = dist["market_share_produto"]["min"], dist["market_share_produto"]["max"]
    base["share_base"] = np.clip(
        rng.normal(share_media, share_dp, size=len(base)), share_min, share_max
    )

    linhas = []
    for i_mes, periodo in enumerate(periodos):
        mes_cal = periodo.month
        fator_sazonal = saz.get(mes_cal, 1.0)
        fator_tendencia = (1 + crescimento_anual) ** (i_mes / 12)

        df_mes = base.copy()
        df_mes["periodo"] = periodo
        df_mes["mes"] = mes_cal

        # demanda de mercado (territorio-produto-mes): media global * potencial territorio * sazonalidade * tendencia
        media_mkt = dist["demanda_mercado_mensal"]["media"]
        dp_pct_mkt = dist["demanda_mercado_mensal"]["dp_pct"]
        ruido_mkt = rng.normal(1.0, dp_pct_mkt, size=len(df_mes))
        demanda_mercado_esperada = (
            media_mkt * df_mes["potencial"] * fator_sazonal * fator_tendencia
        )
        df_mes["demanda_mercado_und"] = np.clip(
            demanda_mercado_esperada * ruido_mkt,
            dist["demanda_mercado_mensal"]["min"],
            dist["demanda_mercado_mensal"]["max"],
        )

        # investimento promocional e estoque (drivers de correlacao)
        media_inv = dist["investimento_promocional"]["media"]
        dp_pct_inv = dist["investimento_promocional"]["dp_pct"]
        df_mes["investimento_promocional"] = np.clip(
            rng.normal(media_inv, media_inv * dp_pct_inv, size=len(df_mes)), 0, None
        )
        inv_normalizado = np.clip(df_mes["investimento_promocional"] / (2 * media_inv), 0, 1)

        media_est = dist["estoque_disponivel"]["media"]
        dp_pct_est = dist["estoque_disponivel"]["dp_pct"]
        df_mes["estoque_disponivel"] = np.clip(
            rng.normal(media_est, media_est * dp_pct_est, size=len(df_mes)), 0, None
        )

        # share ajustado pela correlacao com investimento promocional
        share_ajustado = df_mes["share_base"] * (1 + corr * (inv_normalizado - 0.5))
        share_ajustado = np.clip(share_ajustado, 0.005, 0.95)

        # demanda "esperada" do produto (sem ruido de execucao) -> usada para meta
        demanda_produto_esperada = demanda_mercado_esperada * share_ajustado
        df_mes["_demanda_produto_esperada"] = demanda_produto_esperada

        # demanda "realizada" do produto (com ruido de execucao mensal)
        ruido_exec = rng.normal(1.0, 0.12, size=len(df_mes))
        demanda_produto_real = np.minimum(
            demanda_produto_esperada * ruido_exec, df_mes["demanda_mercado_und"]
        )
        df_mes["demanda_produto_und"] = np.clip(demanda_produto_real, 0, None)

        # preco medio
        media_preco = dist["preco_medio"]["media"]
        dp_pct_preco = dist["preco_medio"]["dp_pct"]
        df_mes["preco_medio"] = np.clip(
            rng.normal(media_preco, media_preco * dp_pct_preco, size=len(df_mes)), 1, None
        )

        df_mes["demanda_mercado_val"] = df_mes["demanda_mercado_und"] * df_mes["preco_medio"]
        df_mes["demanda_produto_val"] = df_mes["demanda_produto_und"] * df_mes["preco_medio"]

        linhas.append(df_mes)

    fato = pd.concat(linhas, ignore_index=True)
    return fato


def gerar_objetivo_territorio(cfg: dict, fato_territorio: pd.DataFrame) -> pd.DataFrame:
    """Meta = demanda esperada (sem ruido de execucao) * fator de ambicao — evita vazamento
    do proprio ruido do 'realizado' para dentro da meta, o que inflaria artificialmente o R2."""
    fator_ambicao = cfg["regras_negocio"]["fator_ambicao_meta"]
    obj = fato_territorio[[
        "id_produto", "id_mercado", "id_setor", "periodo", "mes", "_demanda_produto_esperada"
    ]].copy()
    obj["objetivo_unidades"] = obj["_demanda_produto_esperada"] * fator_ambicao
    return obj.drop(columns=["_demanda_produto_esperada"])


def distribuir_fato_por_outlet(cfg: dict, fato_territorio: pd.DataFrame, dim_outlet: pd.DataFrame,
                                rng: np.random.Generator) -> pd.DataFrame:
    """Espalha os totais territorio-produto-mes pelos outlets do territorio (pesos Dirichlet),
    respeitando demanda_produto_und <= demanda_mercado_und em cada outlet."""
    linhas = []
    outlets_por_setor = dim_outlet.groupby("id_setor")["id_outlet"].apply(list)

    for (id_setor, id_produto, periodo), grupo in fato_territorio.groupby(["id_setor", "id_produto", "periodo"]):
        outlets = outlets_por_setor.get(id_setor, [])
        if not outlets:
            continue
        n_out = len(outlets)
        row = grupo.iloc[0]
        pesos = rng.dirichlet(alpha=np.ones(n_out))
        demanda_mercado_out = row["demanda_mercado_und"] * pesos
        demanda_produto_out = np.minimum(row["demanda_produto_und"] * pesos, demanda_mercado_out)
        for id_outlet, dm, dp in zip(outlets, demanda_mercado_out, demanda_produto_out):
            linhas.append({
                "periodo": periodo,
                "id_produto": id_produto,
                "id_mercado": row["id_mercado"],
                "id_setor": id_setor,
                "id_outlet": id_outlet,
                "preco_medio": row["preco_medio"],
                "demanda_mercado_und": dm,
                "demanda_produto_und": dp,
                "demanda_mercado_val": dm * row["preco_medio"],
                "demanda_produto_val": dp * row["preco_medio"],
            })
    return pd.DataFrame(linhas)


# ============================================================================
# 3. CSV plano para o Agente 3 (produto x territorio x mes)
# ============================================================================
def gerar_csv_plano_agente3(cfg: dict, dims: dict, fato_territorio: pd.DataFrame,
                             objetivo_territorio: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    dim_produto = dims["dim_produto"][["id_produto", "nome_sintetico"]].rename(
        columns={"nome_sintetico": "produto"}
    )
    dim_setor = dims["dim_setor"][["id_setor", "nome_sintetico"]].rename(
        columns={"nome_sintetico": "territorio"}
    )
    dim_rep = dims["dim_representante"][["id_setor", "nome_sintetico"]].rename(
        columns={"nome_sintetico": "representante"}
    )
    # se houver >1 representante por territorio, usa o primeiro por simplicidade
    dim_rep_unico = dim_rep.groupby("id_setor").first().reset_index()

    df = fato_territorio.merge(objetivo_territorio, on=["id_produto", "id_mercado", "id_setor", "periodo", "mes"])
    df = df.merge(dim_produto, on="id_produto").merge(dim_setor, on="id_setor").merge(dim_rep_unico, on="id_setor")

    df["meta_mil_r$"] = (df["objetivo_unidades"] * df["preco_medio"]) / 1000.0
    df["market_share_relativo"] = df["demanda_produto_und"] / df["demanda_mercado_und"].replace(0, np.nan)
    df["desvio_pct"] = (
        (df["demanda_produto_und"] - df["objetivo_unidades"]) / df["objetivo_unidades"].replace(0, np.nan)
    ) * 100

    # tendencia_prescricao: variacao do realizado vs. media movel de ate 3 meses anteriores
    df = df.sort_values(["id_produto", "id_setor", "periodo"])
    df["_media_movel_3m"] = (
        df.groupby(["id_produto", "id_setor"])["demanda_produto_und"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    df["tendencia_prescricao"] = (
        (df["demanda_produto_und"] - df["_media_movel_3m"]) / df["_media_movel_3m"].replace(0, np.nan)
    ).fillna(0.0)

    df["data"] = df["periodo"].dt.strftime("%Y-%m-%d")

    saida = df[[
        "produto", "territorio", "representante", "data", "mes",
        "meta_mil_r$", "market_share_relativo", "tendencia_prescricao", "desvio_pct",
    ]].copy()

    # nulos/outliers so nas features (nunca no target 'desvio_pct' — treinar com o alvo
    # faltante quebraria o XGBoost do Agente 3; XGBoost lida nativamente com NaN em features)
    saida = _injetar_qualidade(
        saida, ["meta_mil_r$", "market_share_relativo"], cfg, rng
    )
    return saida.reset_index(drop=True)


# ============================================================================
# Execucao principal
# ============================================================================
def gerar_tudo(cfg: dict = CONFIG) -> dict[str, pd.DataFrame]:
    rng, faker = _seed_tudo(cfg["seed"])
    periodos = _periodos(cfg)

    dims = gerar_dimensoes(cfg, rng, faker)
    fato_territorio = gerar_fato_territorio(cfg, dims, periodos, rng)
    objetivo_territorio = gerar_objetivo_territorio(cfg, fato_territorio)

    resultado = {**dims}

    if cfg["saida"]["gerar_tabelas_relacionais"]:
        resultado["fato_objetivo_mensal"] = objetivo_territorio
        resultado["fato_mercado_mensal"] = distribuir_fato_por_outlet(cfg, fato_territorio, dims["dim_outlet"], rng)

    if cfg["saida"]["gerar_csv_plano"]:
        resultado["vendas_sinteticas"] = gerar_csv_plano_agente3(
            cfg, dims, fato_territorio, objetivo_territorio, rng
        )

    return resultado


def salvar(resultado: dict[str, pd.DataFrame], cfg: dict = CONFIG) -> Path:
    out_dir = Path(cfg["saida"]["diretorio"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for nome, df in resultado.items():
        df.to_csv(out_dir / f"{nome}.csv", index=False)
    return out_dir


if __name__ == "__main__":
    if not FAKER_DISPONIVEL:
        print("[aviso] biblioteca 'faker' nao encontrada — usando nomes sinteticos simples. "
              "Para nomes/cidades mais variados: pip install faker\n")

    resultado = gerar_tudo(CONFIG)
    out_dir = salvar(resultado)

    print(f"Dados sinteticos gerados em: {out_dir.resolve()}\n")
    for nome, df in resultado.items():
        print(f"  {nome:<24s} {len(df):>7d} linhas")

    vendas = resultado["vendas_sinteticas"]
    print("\nAmostra de vendas_sinteticas.csv:")
    print(vendas.head(5).to_string(index=False))

    print("\nColunas geradas:", list(vendas.columns))
    print("Meses distintos:", sorted(vendas['data'].unique())[:3], "...")

    # copia direta para data/vendas_sinteticas.csv, no caminho que
    # src/agents/agente3_comparador_vendas.py ja espera (__main__ do proprio agente)
    caminho_agente3 = Path("data") / "vendas_sinteticas.csv"
    caminho_agente3.parent.mkdir(parents=True, exist_ok=True)
    vendas.to_csv(caminho_agente3, index=False)
    print(f"\nCopia pronta para o Agente 3 em: {caminho_agente3.resolve()}")
