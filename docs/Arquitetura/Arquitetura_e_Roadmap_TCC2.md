# Arquitetura Técnica e Roadmap — Sistema de Inteligência Prescritiva para a Indústria Farmacêutica

Documento de apoio ao desenvolvimento do TCC2, elaborado a partir da leitura do TCC1 e do estado atual do repositório (`sistema-inteligencia-prescritiva-para-industria-farmaceutica`). Data de referência: 24/07/2026.

---

## 1. Parecer de viabilidade

**O sistema é viável.** Diferente da maioria dos projetos nesta fase, vocês já não estão partindo do zero: o repositório já contém ETL funcional para as duas fontes de notícias (GNews e The Guardian, com extração, tratamento e carga incremental) e o agente de desempate baseado em Kappa de Fleiss + GPT-4o-mini já está implementado e rodando. Isso reduz consideravelmente o risco técnico das partes mais incertas (integração com APIs externas e uso de LLM para decisão), que já foram validadas na prática.

O que resta a construir é bem delimitado pelo próprio TCC1: banco de dados farmacêutico sintético, Agentes 1 a 5, orquestrador e painel de insights. São peças de engenharia bem conhecidas na literatura (fine-tuning de BERT, XGBoost, orquestração com LLMs, dashboards em Python), então o risco não é "isso é possível?", e sim "cabe no prazo até setembro com a equipe de 4 pessoas?". Esse é o ponto que o roadmap abaixo tenta endereçar com honestidade — o cronograma original do TCC1 (Mar–Set) já é apertado, e a decisão de usar LangGraph/AutoGen no orquestrador (em vez de um pipeline determinístico simples) aumenta esse risco. Recomendo uma versão "controlada" dessa autonomia (seção 3.3) para não comprometer o prazo.

### Pontos de atenção que merecem decisão do grupo

- **Concordância fraca (κ = 0,1858).** O rótulo que vai treinar o Agente 1 (mBERT) é, em boa parte, produto do agente de desempate, não de consenso humano direto. Isso é metodologicamente defensável (já está documentado e justificado no TCC1), mas vale registrar no TCC2 essa dependência como limitação conhecida, e não só como um número de qualidade de anotação.
- **Regras de negócio do Agente 2 (Validador) ainda não estão formalizadas em código.** O TCC1 as descreve em alto nível ("filtra classificações que contradizem histórico" e "remove notícias de baixo impacto"), mas não há uma lista concreta de regras. Isso precisa ser definido antes da Sprint 2 do roadmap.
- **Geração dos dados sintéticos farmacêuticos (CRM/SFV/market share) ainda não existe no repositório.** Não há script de geração nem schema de banco — é o maior bloco de trabalho não iniciado.
- **Hospedagem do PostgreSQL** entre 4 integrantes: recomendo uma instância gratuita compartilhada (Neon ou Supabase, ambos com free tier e suporte à extensão `pgvector`) em vez de Postgres local em cada máquina, para evitar dessincronização de dados entre vocês. Se já houver preferência ou restrição institucional, isso muda a seção de infraestrutura.
- **Data da banca/entrega final.** O roadmap abaixo assume entrega/defesa no final de setembro de 2026, espelhando o cronograma do TCC1. Se a data real for diferente, o roadmap precisa ser recalculado.
- **Compute para o mBERT.** Como ainda não está decidido, o roadmap assume Google Colab (gratuito, com limite de sessão) como padrão — isso significa treinar em blocos curtos e salvar checkpoints com frequência.
- **Divisão de responsabilidades entre os 4 integrantes** por agente/etapa não é algo que eu deva decidir, mas o roadmap na seção 4 foi desenhado em blocos paralelizáveis justamente para facilitar essa divisão.

---

## 2. O que já existe no repositório (não precisa ser refeito)

| Componente | Status | Arquivo(s) |
|---|---|---|
| Extração de notícias GNews | Pronto | `src/data/extract_gnews.py` |
| Extração de notícias The Guardian | Pronto | `src/data/extract_guardian_news.py` |
| Tratamento/carga de notícias (dedup, nulos, split de data) | Pronto | `src/data/transformation_and_load_*.py` |
| Cálculo do Kappa de Fleiss | Pronto | `src/estatistica_kappa/kappa_fleiss.ipynb` |
| Agente de desempate (GPT-4o-mini via LangChain) | Pronto | `src/estatistica_kappa/agent_kappa.py` |
| Corpus anotado consolidado (1.415 notícias) | Pronto | `data/estatistica_kappa/*.xlsx` |
| Banco de dados farmacêutico sintético | **Não iniciado** | — |
| Agente 1 (Classificador mBERT) | **Não iniciado** | pasta `src/models` vazia |
| Agente 2 (Validador) | **Não iniciado** | — |
| Agente 3 (Comparador XGBoost) | **Não iniciado** | — |
| Agente 4 (Riscos/Oportunidades) | **Não iniciado** | — |
| Agente 5 (Gerador de Recomendações) | **Não iniciado** | — |
| Orquestrador | **Não iniciado** | `src/main.py` vazio |
| Painel de insights | **Não iniciado** | — |

---

## 3. Arquitetura técnica proposta

### 3.1. Visão em camadas

```
┌─────────────────────────────────────────────────────────────────────┐
│  CAMADA DE APRESENTAÇÃO — Streamlit                                  │
│  KPIs de vendas · Notícias classificadas · Recomendações · Métricas  │
└───────────────────────────────▲───────────────────────────────────────┘
                                 │ lê de
┌───────────────────────────────┴───────────────────────────────────────┐
│  CAMADA DE PERSISTÊNCIA — PostgreSQL (+ extensão pgvector)            │
│  noticias · vendas · metas · market_share · embeddings ·             │
│  recomendacoes · agent_runs (auditoria)                              │
└───────────────────────────────▲───────────────────────────────────────┘
                                 │ grava em / consulta
┌───────────────────────────────┴───────────────────────────────────────┐
│  CAMADA DE AGENTES — LangGraph (Python)                              │
│                                                                        │
│  Agente 1 → Agente 2 → [rota condicional] → Agente 3 → Agente 4 → 5  │
│  Classif.   Validador                        Comparador  Risco/Op.  Gerador │
│  mBERT      regras + embeddings              XGBoost     CoT GPT-4o  GPT-4o │
└───────────────────────────────▲───────────────────────────────────────┘
                                 │ consome
┌───────────────────────────────┴───────────────────────────────────────┐
│  CAMADA DE INGESTÃO — ETL (já implementado para notícias)             │
│  API GNews · API The Guardian · Gerador de dados sintéticos farma    │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2. Detalhamento por agente

| Agente | Função | Modelo/Técnica | Entrada | Saída |
|---|---|---|---|---|
| 1 – Classificador de Impacto | Classifica notícia em Alto/Médio/Baixo | mBERT (`bert-base-multilingual-cased`) fine-tuned no corpus anotado | Texto da notícia | Classe + score softmax |
| 2 – Validador e Ajuste | Aplica regras de negócio e recalibra confiança | Regras Python + embeddings OpenAI + similaridade via `pgvector` | Classificação do Agente 1 | Classificação validada/ajustada |
| 3 – Comparador de Vendas | Detecta desvios de meta por produto/território/representante | XGBoost + SHAP | Dados de vendas (Postgres) | Séries de desvio, ranking de causas |
| 4 – Identificador de Riscos/Oportunidades | Correlaciona notícias validadas com desvios de vendas | Embeddings + chain-of-thought (GPT-4o) | Saída dos Agentes 2 e 3 | Riscos (0–30d, 30–90d) e oportunidades |
| 5 – Gerador de Recomendações | Sintetiza tudo em recomendação tática | GPT-4o com prompt estruturado (JSON) | Saída do Agente 4 | Recomendação: justificativa, urgência, produto/território, ação, impacto esperado |

### 3.3. Orquestrador: autonomia controlada via LangGraph

Optar por LangGraph (em vez de um pipeline puramente sequencial em código) é a escolha certa para aderir ao referencial teórico do TCC1 e ganhar recursos prontos de estado compartilhado, retries e rastreabilidade. Para não estourar o prazo, sugiro implementar o grafo como **majoritariamente linear, com poucas arestas condicionais** — não um agente que replaneja livremente:

- Aresta condicional 1: se o Agente 2 classificar a notícia como baixo impacto, o fluxo pula direto para o registro em `agent_runs` sem acionar os Agentes 3, 4 e 5 (economiza custo de API e tempo).
- Aresta condicional 2: se o Agente 3 não encontrar desvio relevante de vendas para o produto/território da notícia, o Agente 4 ainda roda (para capturar oportunidades futuras), mas com prioridade baixa marcada no estado.
- Todo o estado (notícia, classificações, scores, séries de vendas, riscos, recomendação final) trafega como um único objeto tipado (`TypedDict` ou `pydantic.BaseModel`) entre os nós — isso facilita tanto a implementação quanto a auditoria exigida no TCC1.

Recomendo reservar 3–5 dias no início da Sprint 3 (seção 4) só para uma prova de conceito pequena do LangGraph (2 nós fake) antes de plugar os 5 agentes reais — isso evita descobrir problemas de infraestrutura tarde demais.

### 3.4. Stack técnico consolidado

| Camada | Tecnologia | Observação |
|---|---|---|
| Linguagem | Python 3.11+ | Já em uso |
| ETL notícias | `requests`, `beautifulsoup4`, `pandas` | Já implementado |
| Orquestração | `langgraph`, `langchain-core`, `langchain-openai` | Adicionar `langgraph` ao requirements |
| Agente 1 | `transformers`, `torch` | Adicionar ao requirements; treino via Colab |
| Agente 2 | `openai` (embeddings), `pgvector` | Trocar `text-embedding-ada-002` (citado no TCC1) por `text-embedding-3-small`, mais barato e atual |
| Agente 3 | `xgboost`, `shap` | Adicionar ao requirements |
| Agentes 4 e 5 | `langchain-openai` (GPT-4o / GPT-4o-mini) | Usar `gpt-4o-mini` sempre que possível para custo |
| Banco de dados | PostgreSQL + extensão `pgvector` | Hospedagem sugerida: Neon ou Supabase (free tier) |
| Painel | `streamlit`, `plotly` | Conforme decidido |
| Segredos | `.env` + `python-dotenv` | Já em uso — **atenção**: `docs/Auxiliador.md` tem chaves de API em texto puro; recomendo mover para `.env` e adicionar o arquivo ao `.gitignore` se ainda não estiver |
| Testes | `pytest` | Cobrir ETL e regras do Agente 2 no mínimo |

---

## 4. Roadmap (24/07/2026 → 30/09/2026)

Assume-se entrega/banca no final de setembro. Ajustar se a data real for outra.

### Sprint 0 — Fundação (24/07 a 31/07)
- Formalizar o schema do banco farmacêutico sintético (dimensões: produto, território, representante; fato: vendas, metas, market share).
- Construir o gerador de dados sintéticos (regras de negócio + biblioteca `Faker` ou similar) e carregar no PostgreSQL.
- Provisionar o PostgreSQL compartilhado (Neon/Supabase) com extensão `pgvector` habilitada.
- Atualizar `requirements.txt` com as bibliotecas novas da seção 3.4.
- Consolidar o dataset anotado final (com desempate) como conjunto de treino/validação/teste do Agente 1.

### Sprint 1 — Agente 1: Classificador (01/08 a 14/08)
- Fine-tuning do mBERT no corpus anotado (split sugerido 70/15/15).
- Avaliação por precisão, recall e F1-score por classe (Alto/Médio/Baixo), conforme metodologia já definida na seção 3.7 do TCC1.
- Empacotar o modelo treinado para uso em inferência dentro do pipeline.

### Sprint 2 — Agentes 2 e 3 (15/08 a 28/08)
- Formalizar e implementar as regras de negócio do Validador (Agente 2).
- Implementar recalibração por similaridade semântica usando `pgvector`.
- Treinar o XGBoost (Agente 3) sobre os dados sintéticos de vendas; gerar explicabilidade via SHAP.
- Testes unitários dos dois agentes isoladamente, antes de qualquer integração.

### Sprint 3 — Orquestrador + Agentes 4 e 5 (29/08 a 11/09)
- Prova de conceito do LangGraph (3–5 dias) antes de plugar os agentes reais.
- Implementar chain-of-thought no Agente 4 (correlação notícia × desvio de vendas).
- Implementar geração estruturada (JSON) de recomendações no Agente 5.
- Montar o grafo completo com as arestas condicionais descritas na seção 3.3.
- Primeiro teste ponta a ponta com uma amostra pequena e real de notícias.

### Sprint 4 — Painel e Validação (12/09 a 21/09)
- Construir o painel Streamlit consumindo diretamente o PostgreSQL.
- Rodar a validação com 12 meses de dados históricos sintéticos (seção 3.7 do TCC1).
- Consolidar métricas finais (precisão/recall/F1 do Agente 1, métricas de negócio dos Agentes 3–5).

### Sprint 5 — Fechamento (22/09 a 30/09)
- Ajustes finais de UX do painel e revisão de bugs.
- Redação dos resultados finais no TCC2.
- Preparação da apresentação de banca e gravação/roteiro de demonstração.
- Buffer para imprevistos — não alocar trabalho novo aqui.

### Riscos que podem atrasar o roadmap

- Limite de sessão do Google Colab pode exigir mais de uma sessão de treino do mBERT — mitigar salvando checkpoints a cada época.
- LangGraph é tecnologia nova para a equipe — a prova de conceito da Sprint 3 existe justamente para isolar esse risco cedo.
- Regras de negócio do Agente 2 ainda não definidas — se não forem fechadas até o fim da Sprint 0, isso atrasa a Sprint 2 inteira.
- Geração de dados sintéticos de vendas é o único bloco totalmente novo sem nenhum código de partida — vale começar por aí.

---

## 5. Dúvidas em aberto para o grupo

1. Qual a data real de entrega/banca do TCC2? O roadmap acima assume 30/09/2026.
2. Quais são, concretamente, as regras de negócio do Agente 2 (Validador)? O TCC1 as descreve de forma genérica.
3. Existe alguma referência real (benchmark do setor) para calibrar a geração dos dados sintéticos de vendas, ou a lógica será inteiramente definida pelo grupo?
4. Como será a divisão de responsabilidades entre os 4 integrantes por agente/sprint?
5. As chaves de API listadas em `docs/Auxiliador.md` estão sendo usadas em produção — vale migrar todas para `.env` por segurança antes de compartilhar o repositório mais amplamente?
