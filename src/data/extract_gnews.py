# %%
import os
import csv
import time
import logging
import requests
from dotenv import load_dotenv
from urllib.parse import urlencode

# =========================
# CONFIGURAÇÕES INICIAIS
# =========================
load_dotenv()

LANG = "pt"
PAIS = "br"
QNTD_NOTICIAS = 100

# vetor de palavras-chave

TEXTOS = [
    'Diabetes Mellitus Tipo 2',
    'PBM',
    'empagliflozina',
    'linagliptina',
    'Inibidores de SGLT2',
    'Anvisa',
    'obesidade',
    'GLP-1',
    'Insulina',
    'Degludeca',
    'Semaglutida'
    'Hemoglobina Glicada',
    'Gestão Metabólica',
    'Metformina',
    'taxa de câmbio',
    'dólar',
    'IFA',
    'reajuste CMED',
    'CONITEC'
    'MACE'
]


# carregar as 12 API keys do .env
#API_KEYS = [os.getenv(f"API_KEY_GNEWS_TESTE")]

API_KEYS = [os.getenv(f"API_KEY_GNEWS_PRODUCAO_{i}") for i in range(1, 12)]
API_KEYS = [key for key in API_KEYS if key]  # remove None/vazias

#CAMINHO DOS ARQUIVOS
CAMINHO_DIRETORIO = os.path.abspath(os.path.join(os.path.dirname("sistema-inteligencia-prescritiva-para-industria-farmaceutica"), "..","..", "data"))

CSV_SAIDA = os.path.join(CAMINHO_DIRETORIO, "extracao_noticias_gnews.csv")
LOG_ARQUIVO = os.path.join(CAMINHO_DIRETORIO, "pipeline_extracao_gnews.log")

# =========================
# LOGGING
# =========================
logging.basicConfig(
    filename=LOG_ARQUIVO,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# %%
# =========================
# FUNÇÕES AUXILIARES
# =========================
def limpar_texto(texto):
    """
    Limpeza básica de texto:
    - remove quebra de linha
    - remove espaços duplicados
    - trata None
    """
    if not texto:
        return ""
    texto = texto.replace("\n", " ").replace("\r", " ")
    texto = " ".join(texto.split())
    return texto.strip()

def buscar_noticias(api_key, palavra_chave, lang="pt", pais="br", max_noticias=100, page=3, tentativas=3):
    """
    Busca notícias na API GNews com tratamento de erros e tentativas.
    """
    base_url = "https://gnews.io/api/v4/search"

    params = {
        "q": palavra_chave,
        "lang": lang,
        "country": pais,
        "max": max_noticias,
        "page": page,
        "apikey": api_key
    }

    url = f"{base_url}?{urlencode(params)}"

    for tentativa in range(1, tentativas + 1):
        try:
            logging.info(f"Buscando notícias | palavra='{palavra_chave}' | tentativa={tentativa}")
            
            response = requests.get(url, timeout=20)
            response.raise_for_status()

            data = response.json()

            # garante que articles exista
            articles = data.get("articles", [])
            logging.info(f"{len(articles)} notícias encontradas para '{palavra_chave}'")
            return articles

        except requests.exceptions.HTTPError as e:
            status = response.status_code if response is not None else "desconhecido"
            logging.error(f"Erro HTTP {status} para '{palavra_chave}' | tentativa={tentativa} | erro={e}")

            # se for erro 401, 403 ou 429, talvez seja melhor parar logo para essa chave
            if status in [401, 403, 429]:
                break

        except requests.exceptions.Timeout:
            logging.warning(f"Timeout para '{palavra_chave}' | tentativa={tentativa}")

        except requests.exceptions.RequestException as e:
            logging.error(f"Erro de requisição para '{palavra_chave}' | tentativa={tentativa} | erro={e}")

        except ValueError as e:
            logging.error(f"Erro ao converter JSON para '{palavra_chave}' | tentativa={tentativa} | erro={e}")
            break

        except Exception as e:
            logging.exception(f"Erro inesperado para '{palavra_chave}' | tentativa={tentativa} | erro={e}")
            break

        time.sleep(2)  # pequena espera entre tentativas

    return []

def transformar_artigo(article, palavra_chave, api_key_indice):
    """
    Normaliza os campos do artigo para salvar no CSV.
    """
    source = article.get("source", {}) or {}

    return {
        "palavra_chave": palavra_chave,
        "api_key_indice": api_key_indice,
        "titulo": limpar_texto(article.get("title")),
        "descricao": limpar_texto(article.get("description")),
        "conteudo": limpar_texto(article.get("content")),
        "url": article.get("url", ""),
        "imagem": article.get("image", ""),
        "data_publicacao": article.get("publishedAt", ""),
        "fonte_nome": source.get("name", ""),
        "fonte_url": source.get("url", "")
    }

def salvar_csv(lista_dados, arquivo_saida):
    """
    Salva os dados em CSV.
    """
    if not lista_dados:
        logging.warning("Nenhum dado para salvar no CSV.")
        return

    colunas = [
        "palavra_chave",
        "api_key_indice",
        "titulo",
        "descricao",
        "conteudo",
        "url",
        "imagem",
        "data_publicacao",
        "fonte_nome",
        "fonte_url"
    ]

    with open(arquivo_saida, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=colunas)
        writer.writeheader()
        writer.writerows(lista_dados)

    logging.info(f"CSV salvo com sucesso em: {arquivo_saida}")

# %%
# =========================
# PIPELINE PRINCIPAL
# =========================
def executar_pipeline():
    if not API_KEYS:
        raise ValueError("Nenhuma API key foi encontrada no .env")

    todos_os_resultados = []
    urls_vistas = set()

    for indice_key, api_key in enumerate(API_KEYS, start=1):
        logging.info(f"Iniciando processamento com API_KEY_{indice_key}")

        for palavra_chave in TEXTOS:
            artigos = buscar_noticias(
                api_key=api_key,
                palavra_chave=palavra_chave,
                lang=LANG,
                pais=PAIS,
                max_noticias=QNTD_NOTICIAS,
                page=1
            )

            for article in artigos:
                registro = transformar_artigo(article, palavra_chave, indice_key)

                # remover duplicadas pela URL
                if registro["url"] and registro["url"] not in urls_vistas:
                    urls_vistas.add(registro["url"])
                    todos_os_resultados.append(registro)

        # pequena pausa entre chaves
        time.sleep(1)

    salvar_csv(todos_os_resultados, CSV_SAIDA)

    print(f"Processo finalizado com sucesso.")
    print(f"Total de registros únicos salvos: {len(todos_os_resultados)}")
    print(f"Arquivo CSV: {CSV_SAIDA}")
    print(f"Arquivo de log: {LOG_ARQUIVO}")

if __name__ == "__main__":
    try:
        executar_pipeline()
    except Exception as e:
        logging.exception(f"Falha geral no pipeline: {e}")
        print("Erro na execução. Verifique o arquivo de log.")


