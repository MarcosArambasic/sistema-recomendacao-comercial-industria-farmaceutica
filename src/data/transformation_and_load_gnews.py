# %%
import os
import pandas as pd

# %%
#CAMINHO DOS ARQUIVOS
CAMINHO_DIRETORIO = os.path.abspath(os.path.join(os.path.dirname("sistema-inteligencia-prescritiva-para-industria-farmaceutica"), "..","..", "data"))

#ENTRADA
CSV_ENTRADA = os.path.join(CAMINHO_DIRETORIO, "extracao_noticias_gnews.csv")

#SAÍDA
EXCEL_SAIDA = os.path.join(CAMINHO_DIRETORIO, "transformacao_noticias_gnews.xlsx")

# %%
#E - EXTRACT
df_noticias = pd.read_csv(CSV_ENTRADA)

#================================================================================================

#T - TRANSFORM Tratamento do que está no excel
    # 1 - Remover duplicatas
df_noticias.drop_duplicates(inplace=True)

    # 2 - Tratar valores nulos
df_noticias.dropna(inplace=True)

    # 3 - Converter data para formato datetime
df_noticias['data_publicacao'] = pd.to_datetime(df_noticias['data_publicacao'], errors='coerce')

    # 4 - Remover linhas onde a conversão falhou (valores inválidos)
df_noticias.dropna(subset=['data_publicacao'], inplace=True)

    # 5 - Separar a data em colunas
        #ANO
ano = df_noticias['data_publicacao'].dt.year
df_noticias.insert(0, 'ano', ano)
        #MÊS
mes = df_noticias['data_publicacao'].dt.month
df_noticias.insert(1, 'mes', mes)
        #DIA
dia = df_noticias['data_publicacao'].dt.day
df_noticias.insert(2, 'dia', dia)

    # 6 - Remover coluna publishedAt
df_noticias.drop(columns=['data_publicacao','imagem'], inplace=True)  

#================================================================================================

#L - LOAD
    #Se já existe um arquivo Excel, compara os dados e adiciona apenas as notícias novas
if os.path.exists(EXCEL_SAIDA):
    df_existing = pd.read_excel(EXCEL_SAIDA)
    
    #Se há dados idênticos
        # O indicator cria uma coluna '_merge' que indica a origem de cada linha: 'left_only' para df_existing, 'right_only' para df_noticias, e 'both' para linhas idênticas em ambos os DataFrames.
    df_comparado = df_existing.merge(df_noticias,how='right', indicator=True)
    df_apenas_noticias_novas = df_comparado[df_comparado['_merge'] == 'right_only'].drop(columns=['_merge'])

    if len(df_apenas_noticias_novas) <= 0:
        print("Nenhuma notícia nova para adicionar. O arquivo 'transformacao_noticias_gnews.xlsx' já está atualizado.")
    else:
        #Concatena os dados existentes com as notícias novas e salva no Excel
        df_atualiza = pd.concat([df_existing, df_apenas_noticias_novas], ignore_index=True)
        df_atualiza.to_excel(EXCEL_SAIDA, index=False)
        
        print(f"{len(df_apenas_noticias_novas)} notícia(s) novas adicionadas ao arquivo 'transformacao_noticias_gnews.xlsx' com sucesso!")

#Se não existe um arquivo Excel, cria um novo com os dados transformados
else:
    df_noticias.to_excel(EXCEL_SAIDA, index=False)

    if(os.path.exists(EXCEL_SAIDA)):
        print(f"Arquivo 'transformacao_noticias_gnews.xlsx' criado com sucesso, com inserção de {len(df_noticias)} notícia(s)!")



