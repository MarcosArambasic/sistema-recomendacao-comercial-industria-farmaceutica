from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import pandas as pd
import requests
import os
from dotenv import load_dotenv
load_dotenv()

#----------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------#


# CHAVE DA OPENAI
API_KEY = os.getenv("OPENAI_API_KEY")

#CAMINHO DOS ARQUIVOS
CAMINHO_DIRETORIO = os.path.abspath(os.path.join(os.path.dirname("sistema-inteligencia-prescritiva-para-industria-farmaceutica"), "..","..", "data"))

EXCEL_SAIDA = os.path.join(CAMINHO_DIRETORIO, "avaliacoes_desempatadas.xlsx")


# INICIALIZA O LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# TEMPLATE DO PROMPT 
prompt_template = PromptTemplate(
    input_variables=["texto_noticia"],
    template="""
    Você é um especialista em inteligência de mercado na indústria farmacêutica.
    Leia a notícia abaixo e avalie o impacto que os fatos relatados terão sobre a VENDA DE MEDICAMENTOS PARA DIABETES.
    
    Notícia:
    {texto_noticia}
    
    Regras de Classificação:
    - ALTO: Impacto direto, imediato ou em larga escala nas vendas, tratamentos, regulamentação ou perfil de pacientes com diabetes.
    - MÉDIO: Impacto indireto, em longo prazo, ou que afeta a saúde de forma geral podendo respingar no mercado de diabetes.
    - BAIXO: Nenhuma relação com diabetes, saúde metabólica ou impacto nulo nas vendas.
    
    Responda APENAS com uma das três palavras: ALTO, MÉDIO ou BAIXO. Não inclua nenhuma outra palavra ou pontuação.
    """
)

# CHAIN DO LANGCHAIN
chain = prompt_template | llm

def extrair_texto_url(url):
    """Acessa a URL e extrai o texto principal da notícia."""
    try:
        # EVITAR BLOQUEIOS 
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        # EXTRAI PARÁGRAFOS
        paragrafos = soup.find_all('p')
        texto = ' '.join([p.get_text() for p in paragrafos])
        
        # LIMITAR O TEXTO
        return texto[:4000]
    except Exception as e:
        print(f"Erro ao acessar {url}: {e}")
        return None

def processar_avaliacoes(caminho_arquivo):
    print("Carregando planilha...")
    df = pd.read_excel(caminho_arquivo)
    
    # PADRONIZAÇÃO
    df.columns = df.columns.str.lower().str.strip()
    
    for index, row in df.iterrows():
        aval_mendes = str(row['classificacao mendes']).strip().upper()
        aval_guilherme = str(row['classificacao guilherme']).strip().upper()
        aval_marcos = str(row['classificacao marcos']).strip().upper()
        url = row['url']
        
        # CONDIÇÕES PARA DETERMINAR O DESEMPATE OU ACORDO
            # ACORDO ENTRE TODOS
        if aval_mendes == aval_guilherme and aval_marcos == aval_guilherme and aval_marcos == aval_mendes:
            df.at[index, 'classificacao kappa'] = aval_mendes
            print(f"Linha {index}: Acordo Entre Todos({aval_mendes}).")

            # ACORDO ENTRE DOIS AVALIADORES MENDES-MARCOS
        elif aval_mendes == aval_marcos and aval_guilherme != aval_mendes:
            df.at[index, 'classificacao kappa'] = aval_mendes
            print(f"Linha {index}: Acordo Mendes-Marcos ({aval_mendes}).")
        
            # ACORDO ENTRE DOIS AVALIADORES GUILHERME-MARCOS
        elif aval_guilherme == aval_marcos and aval_mendes != aval_guilherme:
            df.at[index, 'classificacao kappa'] = aval_guilherme
            print(f"Linha {index}: Acordo Guilherme-Marcos ({aval_guilherme}).")

            # ACORDO ENTRE DOIS AVALIADORES MENDES-GUILHERME
        elif aval_mendes == aval_guilherme and aval_marcos != aval_mendes:
            df.at[index, 'classificacao kappa'] = aval_mendes
            print(f"Linha {index}: Acordo Mendes-Guilherme ({aval_mendes}).")
            
        # EM DIVERGÊNCIA, ACIONA O LLM PARA DESEMPATE
        else:
            print(f"Linha {index}: Divergência ({aval_mendes} vs {aval_guilherme} vs {aval_marcos}). Acionando LLM...")
            texto_noticia = extrair_texto_url(url)
            
            if texto_noticia:
                # CHAMA O MODELO DE IA
                resposta = chain.invoke({"texto_noticia": texto_noticia})
                decisao_ia = resposta.content.strip().upper()
                
                if decisao_ia in ['ALTO', 'MÉDIO', 'BAIXO']:
                    df.at[index, 'classificacao kappa'] = decisao_ia
                    print(f" -> LLM decidiu: {decisao_ia}")
                else:
                    df.at[index, 'classificacao kappa'] = "ERRO_IA"
                    print(f" -> LLM retornou valor inválido: {decisao_ia}")
            else:
                df.at[index, 'classificacao kappa'] = "ERRO_URL"
                print(" -> Falha ao ler a notícia.")
 
    df.to_excel(EXCEL_SAIDA, index=False)
    print(f"\nProcesso concluído! Arquivo salvo como: {EXCEL_SAIDA}")

#----------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    try:
        processar_avaliacoes()