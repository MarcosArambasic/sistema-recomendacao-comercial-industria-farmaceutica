# Plano final do TCC2 — de 12/08 a final de agosto/2026

Atualizado a partir da leitura do `TCC 1 - FINAL.pdf`, dos roadmaps já existentes (`Arquitetura_e_Roadmap_TCC2.md`, `Roadmap_10_dias_TCC2.md`) e do estado real do código no repositório em 12/08/2026. Diferente dos documentos anteriores, este assume o prazo real informado: **entrega/banca no final de agosto/2026** — ou seja, cerca de **2,5 semanas**, bem mais apertado do que os ~2 meses que os documentos anteriores assumiam (30/09).

Isso muda a estratégia: a prioridade agora é **entregar uma versão funcional e defensável do sistema completo**, cortando ou simplificando partes que não são essenciais para o TCC1 já escrito. As seções 4 e 5 marcam claramente o que é obrigatório vs. o que pode virar "trabalho futuro" no texto do TCC2 se o tempo apertar.

---

## 1. O que o TCC exige (resumo do TCC1)

**Objetivo geral:** sistema de inteligência prescritiva com arquitetura multiagente (LLMs) que cruza performance de vendas, indicadores de mercado e notícias externas, gerando recomendações táticas para gestores comerciais da indústria farmacêutica (nicho: mercado de diabetes).

**Os 5 agentes prometidos no TCC1** (seção 3.4), coordenados por um Agente Orquestrador:

| Agente | Função | Técnica prometida no TCC1 |
|---|---|---|
| 1 — Classificador de Impacto | Classifica notícia em Alto/Médio/Baixo | mBERT fine-tuned no corpus anotado |
| 2 — Validador & Ajuste | Filtra/recalibra classificações | Regras de negócio + embeddings + similaridade semântica |
| 3 — Comparador de Vendas | Detecta desvios de meta | XGBoost + SHAP |
| 4 — Riscos/Oportunidades | Correlaciona notícias com desvios | Embeddings + chain-of-thought (GPT-4) |
| 5 — Gerador de Recomendações | Sintetiza recomendação tática | GPT-4 com prompt estruturado (JSON) |

**Validação prometida (seção 3.7, obrigatória para o capítulo de Resultados do TCC2):**
- Fase 1: Agentes 1 e 2 avaliados com o corpus de 1.415 notícias — precisão, recall e F1 por classe.
- Fase 2: recomendações avaliadas sobre 12 meses de dados históricos sintéticos.

**Já validado e escrito no TCC1** (não precisa refazer): pipelines ETL de notícias, anotação manual de 1.415 notícias, cálculo do Kappa de Fleiss (κ = 0,1858 — concordância fraca), justificativa metodológica do Agente de Desempate.

---

## 2. Estado real do repositório em 12/08/2026

Verificado lendo o código, não apenas os relatórios anteriores — há avanços desde a última análise (24/07) e também divergências importantes.

| Componente | Status real | Observação |
|---|---|---|
| ETL notícias (GNews + Guardian) | **Pronto** | `src/data/*.py` |
| Kappa de Fleiss + Agente de Desempate | **Pronto e funcional** | `src/estatistica_kappa/agent_kappa.py` (128 linhas, completo) |
| `src/agents/agente_desempate.py` | **Vazio (0 bytes)** | Parece um stub duplicado — o desempate real está em `estatistica_kappa/agent_kappa.py`. Decidir se descartam este arquivo ou consolidam nele. |
| Agente 1 (Classificador mBERT) | **Não implementado** | `src/models/` está vazio. O orquestrador usa os rótulos já consolidados do corpus como *proxy* do classificador (comentário explícito no código admitindo isso), o que não cumpre a promessa de fine-tuning do TCC1. |
| Agente 2 (Validador) | **Parcial** | Existe embutido no `orquestrador.py` como filtro simples (remove BAIXO impacto). Não é um arquivo próprio, não tem regras adicionais, não tem recalibração por embeddings — mas isso bate com a simplificação combinada no `Roadmap_10_dias`. |
| Agente 3 (Comparador XGBoost) | **Código pronto, sem dados reais** | `src/agents/agente3_comparador_vendas.py` (158 linhas) — treino, SHAP e classificação de desvio implementados. Espera um CSV (`data/vendas_sinteticas.csv`) que **não existe no repositório**. |
| Agente 4 (Riscos/Oportunidades) | **Pronto** | `src/agents/agente4_riscos_oportunidades.py` (186 linhas) |
| Agente 5 (Gerador de Recomendações) | **Pronto** | `src/agents/agente5_gerador_recomendacoes.py` (158 linhas), modo heurístico + modo LLM (GPT) |
| Orquestrador | **Pronto, mas não é LangGraph** | `src/orquestrador.py` — pipeline sequencial em Python puro (dataclass de estado + métodos por etapa). Funciona e é rastreável, mas diverge do texto do TCC1 ("Implementado com o framework LangChain") e do plano de LangGraph do roadmap anterior. |
| Persistência dos runs (`agent_runs`, `recomendacoes`) | **Não implementado** | O orquestrador exporta para JSON (`exportar_painel`), não grava no Postgres. |
| Banco de dados farmacêutico sintético | **Não iniciado — maior bloco em aberto** | Existe apenas o schema SQL (`supabase/scripts/criacao_banco.sql`), e ele **não bate** com o que o Agente 3 espera (ver risco abaixo). Não há gerador de dados sintéticos (nenhum script com Faker ou similar) nem CSV populado. |
| Painel de insights (Streamlit) | **Não iniciado** | Nenhum arquivo `app.py`/`painel.py`, `streamlit` não está no `requirements.txt`. |
| `requirements.txt` | **Desatualizado** | Não tem `xgboost`, `shap`, `streamlit`, `transformers`, `torch` — apesar de `agente3_comparador_vendas.py` já importar `xgboost` e `shap` diretamente. |
| `src/pipeline/`, `src/features/`, `src/utils/` | **Vazios** | Pastas criadas, sem conteúdo. |

### Riscos técnicos que precisam de decisão do grupo antes de codar mais

1. **Schema do banco não bate com o Agente 3.** `criacao_banco.sql` modela `dim_produto`/`dim_mercado`/`dim_setor`/`dim_outlet`/`fato_mercado_mensal` (granularidade de mercado/outlet), mas `agente3_comparador_vendas.py` espera colunas simples: `produto`, `territorio`, `representante`, `meta_mil_r$`, `market_share_relativo`, `tendencia_prescricao`, `mes`, `desvio_pct`. Além disso, o SQL tem *views* que referenciam uma tabela `fato_desempenho_semanal` que **nunca é criada** no próprio script — isso vai quebrar ao rodar. Precisa reconciliar os dois antes de gerar dados.
2. **Chaves de API em texto puro no repositório.** `docs/Auxiliador.md` lista e-mails, senhas e API keys de GNews/Guardian de 13 contas de integrantes do grupo. Isso já estava sinalizado no roadmap anterior e continua sem correção. Antes de qualquer entrega/banca (e idealmente já), mover para `.env` (que já está no projeto) e não deixar senhas em arquivo versionado.
3. **Agente 1 não cumpre a promessa metodológica do TCC1.** A seção 3.7 promete avaliar Agente 1 com precisão/recall/F1 por classe sobre um modelo mBERT fine-tuned de verdade. Hoje ele só repassa rótulos já prontos. Isso precisa virar, no mínimo, um fine-tuning real (mesmo que simplificado) para o capítulo de Resultados ter dados verdadeiros — ou uma limitação explicitamente assumida no texto (mais arriscado numa banca).
4. **Orquestrador não usa LangGraph.** Não é necessariamente um problema — a versão sequencial funciona e é mais simples de defender sob pressão de prazo — mas o texto do TCC1 promete LangChain/LangGraph. Recomendo **manter a versão sequencial atual** e ajustar o texto do TCC2 para descrever a decisão como simplificação arquitetural deliberada (ganho real de tempo, ver seção 4).

---

## 3. Por que o prazo mudou o jogo

Os documentos anteriores (`Roadmap_10_dias_TCC2.md`) previam um sprint terminando em 09/08 com banco de dados populado, mBERT treinado, LangGraph funcionando e painel Streamlit pronto. Nenhuma dessas quatro entregas está feita, e restam ~2,5 semanas até o prazo real (final de agosto). Isso não é recuperável mantendo o escopo original do TCC1 à risca — a estratégia precisa ser **cortar/simplificar deliberadamente** o que não compromete a defesa, e ser transparente sobre isso no texto do TCC2 (bancas costumam valorizar mais a honestidade metodológica do que a promessa não cumprida).

---

## 4. Cortes de escopo recomendados (para caber no prazo)

| Corte proposto | Por quê | Como documentar no TCC2 |
|---|---|---|
| **Não migrar para LangGraph** — manter o orquestrador sequencial já implementado | Já funciona; LangGraph exigiria dias de POC + refatoração de 5 agentes, sem ganho perceptível para a banca | "Optou-se por uma implementação determinística do padrão orquestrador-executor em vez de um framework de grafo de estados, por restrição de cronograma; a migração para LangGraph é indicada como trabalho futuro." |
| **Painel lê de um JSON/CSV exportado pelo orquestrador, não do Postgres ao vivo** | Elimina a dependência de terminar a integração com o Supabase a tempo; o painel ainda mostra dados reais do pipeline | Mencionar como decisão de implementação; não compromete os resultados apresentados |
| **Sem gravação em `agent_runs`/`recomendacoes` no Postgres** | `exportar_painel()` já gera rastreabilidade completa em JSON — atende ao requisito de auditoria sem precisar do banco | Rastreabilidade demonstrada via log estruturado, não via tabelas relacionais |
| **Agente 2 continua simples (sem embeddings/pgvector)** | Já era a decisão combinada no roadmap anterior; recalibração semântica é incremento, não requisito central | Já é consistente com o texto atual do TCC1 |
| **Agente 1: fine-tuning enxuto, não extensivo** | Um mBERT com poucas épocas, métricas reais (mesmo que modestas) é mais defensável do que zero treino real | Reportar F1 real, mesmo que baixo — é resultado, não falha |

Se ainda sobrar tempo depois de tudo isso funcionando ponta a ponta, aí sim vale investir em LangGraph ou Postgres ao vivo — mas só depois do MVP fechado.

---

## 5. Plano de tarefas (12/08 → ~29/08, buffer até fim de agosto)

Dividido em 4 blocos paralelizáveis (útil para os 4 integrantes trabalharem simultaneamente, como já era a lógica do roadmap anterior). Ordem de prioridade **dentro de cada bloco** é a ordem de execução.

### Bloco A — Dados sintéticos (CAMINHO CRÍTICO — bloqueia B e D)
- [ ] Reconciliar o schema SQL (`criacao_banco.sql`) com as colunas que `agente3_comparador_vendas.py` espera — decidir se o Agente 3 muda para consumir o schema relacional ou se o schema é simplificado para bater com o CSV plano.
- [ ] Corrigir a inconsistência do `fato_desempenho_semanal` (referenciado em views mas nunca criado).
- [ ] Escrever gerador de dados sintéticos (Faker ou regras de negócio) cobrindo produto, território, representante, vendas, metas, market share — 12 meses de histórico.
- [ ] Popular `data/vendas_sinteticas.csv` (ou o banco, se a hospedagem Postgres estiver pronta) para o Agente 3 treinar de verdade.
- [ ] Provisionar Postgres (Neon ou Supabase, free tier) **somente se sobrar tempo** — não é bloqueante se o painel ler de CSV/JSON (corte da seção 4).

### Bloco B — Agente 1 (Classificador mBERT)
- [ ] Script de split do corpus anotado em treino/validação/teste (70/15/15).
- [ ] Notebook de fine-tuning do mBERT no Colab, salvando checkpoints por época.
- [ ] Rodar o treino (timebox: se não convergir em ~2 dias de tentativas, documentar como limitação e manter o proxy atual, mas registrar isso explicitamente no texto).
- [ ] Avaliação: matriz de confusão, precisão/recall/F1 por classe sobre o conjunto de teste.
- [ ] Empacotar função `classificar_noticia(texto) -> (classe, score)` e plugar no orquestrador no lugar do proxy atual.

### Bloco C — Orquestrador, integração e testes
- [ ] Excluir ou consolidar `src/agents/agente_desempate.py` (está vazio; decidir se o desempate real de `estatistica_kappa/agent_kappa.py` deve morar em `src/agents/` por organização).
- [ ] Atualizar `requirements.txt` com `xgboost`, `shap`, `streamlit`, `transformers`, `torch` (e remover o que não é mais usado).
- [ ] Rodar o pipeline completo ponta a ponta assim que o Bloco A entregar dados reais — trocar o Agente 3 de mock para dados reais.
- [ ] Testar Agentes 4 e 5 em modo `usar_llm=True` com uma amostra pequena (10–20 notícias) para conferir custo e qualidade antes de rodar em lote.
- [ ] Mover as chaves de `docs/Auxiliador.md` para `.env` e remover/anonimizar o arquivo antes da entrega final (risco de segurança, não é opcional).

### Bloco D — Painel Streamlit
- [ ] Pode começar em paralelo usando o JSON de exemplo que o orquestrador já gera (`exportar_painel`), sem esperar o Bloco A terminar.
- [ ] 4 páginas conforme já planejado: visão geral vendas x metas; notícias classificadas (filtro produto/território); recomendações priorizadas por urgência; métricas de confiabilidade (Kappa, F1 do Agente 1, SHAP do Agente 3).
- [ ] Trocar fonte de dados mock pelo JSON/CSV real assim que o pipeline completo rodar (Bloco C).

### Convergência final (últimos ~4-5 dias antes da entrega)
- [ ] Rodar validação com 12 meses de dados sintéticos (fase 2 da seção 3.7 do TCC1) e gerar relatório de métricas de negócio.
- [ ] Consolidar métricas finais (F1 do Agente 1, MAE/R² do Agente 3, exemplos de recomendações do Agente 5) para o capítulo de Resultados do TCC2.
- [ ] Redigir o capítulo de Resultados e as limitações metodológicas (proxy do Agente 1 se não convergir, orquestrador sequencial em vez de LangGraph, etc.).
- [ ] Ensaiar a apresentação de banca.
- [ ] Buffer para imprevistos — não alocar trabalho técnico novo aqui.

---

## 6. Como rastrear o progresso

Você escolheu **GitHub Projects/Issues**, que faz sentido — vocês já usam o GitHub para o código, e um board Kanban ligado a commits/PRs facilita ver o que cada um dos 4 integrantes está fazendo sem sair do fluxo de trabalho.

**Não tenho um conector de GitHub conectado nesta conversa para criar o board automaticamente**, então aqui vai o passo a passo (leva ~10 minutos):

1. No repositório no GitHub, vá em **Projects → New project → Board** (template Kanban).
2. Crie as colunas: `Backlog` · `Em andamento` · `Bloqueado` · `Em revisão` · `Concluído`.
3. Ative **Issues** no repositório (Settings → Features → Issues, se ainda não estiver ativado).
4. Crie uma Issue para cada item das checklists da seção 5 acima (dá pra copiar/colar os itens `- [ ]` direto — o GitHub já reconhece esse formato se vocês colarem tudo dentro de uma Issue "mãe" por bloco, ou cada item vira uma Issue própria se preferirem granularidade maior).
5. Adicione um label por bloco (`bloco-a-dados`, `bloco-b-agente1`, `bloco-c-orquestrador`, `bloco-d-painel`) e atribua um responsável por bloco.
6. Arraste as Issues para o board conforme o progresso — isso já fica visível pra vocês 4 sem precisar de reunião de status.

Se preferirem, o arquivo `docs/Arquitetura/Plano_Final_TCC2_Agosto2026.md` (este aqui) já serve como checklist de apoio dentro do próprio repositório, útil para quem for atualizando os itens `- [ ]` → `- [x]` direto no Markdown como registro histórico, mesmo usando o board do GitHub como ferramenta do dia a dia.

---

## 7. Perguntas em aberto para o grupo

1. O schema do banco (SQL) fica como está e o Agente 3 se adapta a ele, ou simplificamos o schema para bater com o CSV que o Agente 3 já espera? (Bloco A depende dessa decisão antes de começar.)
2. Se o mBERT não convergir bem no prazo, o grupo topa assumir isso como limitação explícita no texto, ou prefere insistir mais tempo nele mesmo com risco pro resto do cronograma?
3. Quem fica responsável por qual bloco (A/B/C/D)? O plano foi desenhado para paralelizar entre os 4 integrantes.
4. As chaves de API do `Auxiliador.md` — pode apagar as que não estão em uso, ou algum integrante ainda depende de alguma delas?
