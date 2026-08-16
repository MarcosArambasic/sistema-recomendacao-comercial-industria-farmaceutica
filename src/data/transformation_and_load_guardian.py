# %%
import os
import pandas as pd

# Caminhos independentes do diretório a partir do qual o script é executado.
CAMINHO_DIRETORIO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data")
)

#ENTRADA
CSV_ENTRADA = os.path.join(CAMINHO_DIRETORIO, "extracao_noticias_guardian.csv")

#SAÍDA
EXCEL_SAIDA = os.path.join(CAMINHO_DIRETORIO, "transformacao_noticias_guardian.xlsx")

# %%
# E - EXTRACT
df_noticias = pd.read_csv(CSV_ENTRADA)

#================================================================================================

# T - TRANSFORM
df_noticias.drop_duplicates(inplace=True)

# Campos opcionais podem não existir em arquivos gerados por versões anteriores.
for coluna in ("imagem", "fonte_url"):
    if coluna not in df_noticias.columns:
        df_noticias[coluna] = ""
    df_noticias[coluna] = df_noticias[coluna].fillna("").astype(str)

df_noticias["data_publicacao"] = pd.to_datetime(
    df_noticias["data_publicacao"], errors="coerce"
)
df_noticias.dropna(subset=["data_publicacao", "url"], inplace=True)
df_noticias["url"] = df_noticias["url"].astype(str).str.strip()
df_noticias = df_noticias[df_noticias["url"] != ""].copy()

df_noticias.insert(0, "ano", df_noticias["data_publicacao"].dt.year)
df_noticias.insert(1, "mes", df_noticias["data_publicacao"].dt.month)
df_noticias.insert(2, "dia", df_noticias["data_publicacao"].dt.day)
df_noticias.drop(columns=["data_publicacao", "imagem"], inplace=True)

#================================================================================================

# L - LOAD
# Se já existe um Excel, adiciona somente URLs ainda não registradas.
if os.path.exists(EXCEL_SAIDA):
    df_existing = pd.read_excel(EXCEL_SAIDA)

    # Garante compatibilidade com planilhas criadas antes de novas colunas.
    for coluna in df_noticias.columns:
        if coluna not in df_existing.columns:
            df_existing[coluna] = ""
    df_existing = df_existing[df_noticias.columns]

    df_existing["url"] = df_existing["url"].fillna("").astype(str).str.strip()
    urls_existentes = set(df_existing.loc[df_existing["url"] != "", "url"])
    df_noticias_novas = df_noticias[
        ~df_noticias["url"].isin(urls_existentes)
    ].copy()

    if df_noticias_novas.empty:
        print("Nenhuma notícia nova para adicionar. O arquivo já está atualizado.")
    else:
        df_atualiza = pd.concat([df_existing, df_noticias_novas], ignore_index=True)
        df_atualiza.to_excel(EXCEL_SAIDA, index=False)
        print(f"{len(df_noticias_novas)} notícia(s) novas adicionadas ao arquivo com sucesso!")

#Se não existe um arquivo Excel, cria um novo com os dados transformados
else:
    df_noticias.to_excel(EXCEL_SAIDA, index=False)
    print(f"Arquivo criado com sucesso, com inserção de {len(df_noticias)} notícia(s)!")



