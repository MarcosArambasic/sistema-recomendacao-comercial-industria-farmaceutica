# REPRODUÇÃO DOS RESULTADOS

Os resultados podem ser reproduzidos a partir do repositório do projeto, na ordem a seguir; os números citados no texto são lidos dos arquivos gerados por esses comandos. A base sintética, o Agente 3, o Naive Bayes e as métricas rodam em CPU; o treino do mBERT exige GPU (sem ela, o orquestrador usa o Naive Bayes); o Desempate depende do Llama 3.1 servido localmente pelo Ollama (sem ele, os casos vão para revisão manual); o encoder semântico do Agente 2 requer o download de um modelo de cerca de 1 GB; e o corpus anotado e os parâmetros de calibração são arquivos previamente gerados, versionados no repositório.


```bash
python scripts/gerar\_base\_inicial.py       # 1. base sintética mensal
python scripts/gerar\_backfill\_semanal.py   # 2. histórico semanal
python scripts/gerar\_kpi\_final.py          # 3. base de treino do Agente 3
python -m src.agentes.agente3\_oportunidades\_mercado #4. 
python scripts/orquestrar.py corpus        # 5. Desempate e Agentes 1 e 2
python scripts/orquestrar.py noticias      # 6. notícias e Agentes 4 e 5
python scripts/avaliar\_agentes.py         # 7. métricas consolidadas
streamlit run app/painel.py                # 8. painel de insights
```


