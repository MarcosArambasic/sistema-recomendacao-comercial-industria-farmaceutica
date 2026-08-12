# Roadmap de 10 dias — Sprint final do TCC2

Prazo alvo: **09/08/2026**. Plano de 10 dias corridos (26/07 a 04/08), deixando **05 a 09/08 como buffer** para redação do TCC2, ensaio de banca e correção de imprevistos.

**Divisão de trabalho:** você conduz a construção do banco de dados farmacêutico sintético (base real com variação de até 10%) em paralelo. Eu construo os Agentes 1–5, o orquestrador e o painel, usando um schema combinado no Dia 1 para não ficar bloqueado esperando o banco terminar. **O ponto crítico é o Dia 5**: o Agente 3 precisa de dados reais de vendas para treinar de verdade — se o banco não estiver populado até lá, ele continua rodando sobre dados fictícios e isso vira risco para o restante do roadmap.

Decisões que valem para todo o plano: Agente 2 simplificado (remove apenas notícias BAIXO impacto, sem recalibração por embeddings por ora); orquestrador em LangGraph desde o início, com prova de conceito isolada no Dia 6 antes de plugar os 5 agentes reais.

Como usar esta tabela: no dia combinado, copie o texto da coluna "Prompt sugerido" (ajuste se quiser) e me envie. Depois de eu entregar, confira a coluna "Critérios de validação" antes de aprovar e seguir para o próximo dia.

---

## Dia 1 — Sáb 26/07 — Contrato de dados + fundação do repositório

**Você:** define/traz a estrutura da base real que vai usar de referência para os dados sintéticos.

**Prompt sugerido:**
> "Vamos fechar o schema do banco farmacêutico (tabelas de produto, território, representante, vendas, metas e market share, com nomes e tipos de coluna) com base na estrutura real que vou usar de referência. Depois disso, atualize o `requirements.txt` do projeto com as bibliotecas que faltam (transformers, torch, xgboost, shap, langgraph, streamlit, psycopg2-binary ou sqlalchemy) e monte o script de split do corpus anotado (`data/estatistica_kappa/...`) em treino/validação/teste (70/15/15)."

**Entregável:** documento de schema (`docs/Arquitetura/schema_banco.md` ou similar) + `requirements.txt` atualizado + script `src/models/split_corpus.py` gerando `train.csv`, `val.csv`, `test.csv`.

**Critérios de validação:**
- Schema tem nomes de tabela/coluna que batem com o que você vai usar no banco real (combine antes de eu escrever código em cima disso).
- `pip install -r requirements.txt` roda sem erro.
- Os três arquivos gerados somam o total de notícias anotadas, sem sobreposição entre eles.

---

## Dia 2 — Dom 27/07 — Agente 1: treino do mBERT

**Prompt sugerido:**
> "Monte o notebook de fine-tuning do mBERT (`bert-base-multilingual-cased`) para rodar no Google Colab, usando `train.csv`/`val.csv`, salvando checkpoint a cada época e logando loss/accuracy de treino e validação."

**Entregável:** notebook (`src/models/train_agente1_mbert.ipynb`) pronto para rodar no Colab.

**Critérios de validação:**
- O notebook sobe no Colab sem erro de dependência.
- Salva pelo menos um checkpoint por época.
- Mostra métricas de treino/validação a cada época (mesmo que você ainda não tenha rodado o treino completo).

---

## Dia 3 — Seg 28/07 — Agente 1: avaliação e empacotamento

**Prompt sugerido:**
> "Depois do treino, gere o script de avaliação do modelo no `test.csv` com matriz de confusão e precisão/recall/F1 por classe (Alto/Médio/Baixo), e empacote a inferência numa função `classificar_noticia(texto) -> (classe, score)` reutilizável pelos outros agentes."

**Entregável:** `src/models/avaliar_agente1.py` + `src/models/agente1_classificador.py` (função de inferência).

**Critérios de validação:**
- Relatório de F1 por classe é impresso/salvo.
- A função `classificar_noticia` roda com uma notícia de teste e devolve classe + score entre 0 e 1.

---

## Dia 4 — Ter 29/07 — Agente 2 (validador) + esqueleto do Agente 3

**Prompt sugerido:**
> "Implemente o Agente 2 como função pura que recebe a lista de notícias classificadas pelo Agente 1 e remove as de impacto BAIXO. Em paralelo, comece o Agente 3 (XGBoost) treinando sobre dados fictícios no formato do schema combinado no Dia 1, para eu não ficar bloqueado esperando o banco."

**Entregável:** `src/models/agente2_validador.py` + `src/models/agente3_comparador.py` (versão inicial com dados mock).

**Critérios de validação:**
- Agente 2 testável isoladamente: lista de entrada com 3 classes → saída sem nenhum item BAIXO.
- Agente 3 treina sem erro sobre os dados mock e gera uma previsão de desvio de meta por produto/território.

---

## Dia 5 — Qua 30/07 — Agente 3: SHAP + conexão com banco real

**Ponto crítico:** se o seu banco já estiver populado até aqui, conectamos o Agente 3 aos dados reais hoje. Se não, ele continua sobre dados mock e isso empurra risco pros dias seguintes.

**Prompt sugerido:**
> "Adicione explicabilidade SHAP às previsões do Agente 3 e troque a fonte de dados mock pela conexão real com o PostgreSQL (ou me avise se o banco ainda não estiver pronto para eu manter o mock por mais um dia)."

**Entregável:** `src/models/agente3_comparador.py` atualizado com SHAP + conexão real (ou mock, se combinado assim).

**Critérios de validação:**
- Cada previsão de desvio vem acompanhada dos SHAP values das variáveis mais relevantes.
- Se conectado ao banco real: a query retorna dados de pelo menos um produto/território de verdade.

---

## Dia 6 — Qui 31/07 — Prova de conceito do LangGraph + início do Agente 4

**Prompt sugerido:**
> "Crie uma prova de conceito do LangGraph com 2 nós fictícios só para validar como o estado (notícia, classificação, dados de vendas) trafega entre eles. Em paralelo, comece o Agente 4: correlação entre notícia validada e desvio de vendas usando chain-of-thought prompting (GPT-4o), categorizando em risco imediato (0–30d), risco de médio prazo (30–90d) ou oportunidade."

**Entregável:** `src/pipeline/poc_langgraph.py` + `src/models/agente4_riscos_oportunidades.py`.

**Critérios de validação:**
- POC do LangGraph roda e imprime o estado passando de um nó para o outro sem erro.
- Agente 4 retorna categoria (risco imediato/médio prazo/oportunidade) com o raciocínio intermediário explícito no output, para pelo menos 2 exemplos manuais.

---

## Dia 7 — Sex 01/08 — Agente 5 + grafo ligando os Agentes 1–4

**Prompt sugerido:**
> "Implemente o Agente 5 (gerador de recomendações) retornando JSON estruturado com justificativa, nível de urgência, produtos/territórios afetados, ação sugerida e impacto esperado. Depois, monte o grafo LangGraph ligando os Agentes 1 a 4, com a aresta condicional que pula os Agentes 3–5 quando o Agente 2 filtra a notícia como BAIXO."

**Entregável:** `src/models/agente5_recomendacoes.py` + `src/pipeline/orquestrador.py` (grafo com 1–4 integrados).

**Critérios de validação:**
- JSON do Agente 5 tem todos os 5 campos obrigatórios preenchidos.
- Rodando o grafo com 3 notícias de teste (incluindo 1 que deveria ser filtrada no Agente 2), o fluxo pula corretamente os agentes seguintes para essa notícia.

---

## Dia 8 — Sáb 02/08 — Orquestrador completo + persistência no banco

**Prompt sugerido:**
> "Inclua o Agente 5 no grafo (fechando o pipeline 1→2→3→4→5) e adicione a gravação de cada execução nas tabelas `agent_runs` e `recomendacoes` do PostgreSQL, para termos rastreabilidade completa."

**Entregável:** `src/pipeline/orquestrador.py` completo + tabelas `agent_runs`/`recomendacoes` criadas.

**Critérios de validação:**
- Rodar o pipeline completo em lote com 20–50 notícias reais e conferir que as recomendações aparecem no banco.
- Cada linha de `agent_runs` permite reconstruir por que uma recomendação foi gerada (rastreabilidade).

---

## Dia 9 — Dom 03/08 — Painel Streamlit

**Prompt sugerido:**
> "Construa o painel Streamlit com quatro páginas: (1) visão geral de vendas x metas, (2) notícias classificadas com filtro por produto/território, (3) recomendações táticas em formato de lista priorizada por urgência, (4) métricas de confiabilidade do modelo (Kappa, F1 do Agente 1, SHAP do Agente 3)."

**Entregável:** `src/app/painel.py` (ou `app.py`) rodável via `streamlit run`.

**Critérios de validação:**
- `streamlit run` sobe sem erro.
- As quatro páginas carregam dados reais do PostgreSQL, sem dados fixos/hardcoded.

---

## Dia 10 — Seg 04/08 — Integração final e validação com histórico de 12 meses

**Prompt sugerido:**
> "Rode a validação do sistema completo com os dados históricos sintéticos (12 meses), conforme a metodologia da seção 3.7 do TCC1, e gere um relatório final com F1 do Agente 1, métricas de negócio dos Agentes 3–5 e principais achados para eu usar na redação dos resultados do TCC2."

**Entregável:** relatório de validação (`docs/Arquitetura/resultados_validacao.md` ou `.xlsx`).

**Critérios de validação:**
- Relatório cobre as duas fases de validação descritas no TCC1 (classificação e recomendações).
- Pipeline completo roda ponta a ponta sobre os 12 meses sem erro crítico (falhas pontuais em notícias específicas são aceitáveis, desde que logadas).

---

## Buffer — 05 a 09/08

Sem tarefas técnicas novas alocadas aqui de propósito. Use para: redigir os resultados finais no TCC2, revisar consistência entre o que foi implementado e o que está descrito no texto, ensaiar a apresentação de banca e corrigir o que os testes do Dia 10 tiverem revelado.

---

## Observações finais

- Se o banco de dados atrasar além do Dia 5, me avise no prompt do dia — eu sigo com dados mock no schema combinado e trocamos pela fonte real assim que estiver pronta, sem travar o resto do pipeline.
- Cada entrega é um arquivo/script novo ou modificado no repositório — se algo não bater com os critérios de validação, é mais rápido pedir ajuste no mesmo dia do que deixar acumular para o dia seguinte.
- Guarde as chaves de API que forem realmente usadas em `.env`; não precisa manter as demais listadas em `docs/Auxiliador.md`.
