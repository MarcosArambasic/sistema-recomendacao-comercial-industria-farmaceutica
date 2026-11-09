# Métricas de validação e desempenho por agente

Este documento lista, para cada componente do sistema, **o que é medido, como é calculado, contra o que é comparado e qual o valor atual**. Todos os números abaixo saem de um único script reprodutível:

```bash
python scripts/avaliar_agentes.py          # grava data/agentes/metricas_agentes.json
```

Valores de **07/10/2026**, base sintética 01/2023–09/2026, 564 notícias na base classificada.

| Componente | Métrica principal | Valor atual | Referência de comparação |
|---|---|---|---|
| Base sintética | fidelity score (0–100) | **98,27** | — (alvos do Excel) |
| Agente 1 – Classificador de impacto | F1 macro | **0,666** | classe majoritária 0,222 · regras 0,458 |
| Agente 2 – Validador | F1 da classe "manter" | **0,830** | — |
| Agente 3 – Oportunidades de mercado | MAE do share t+1 | **0,00440** | persistência 0,00443 · média 3m 0,00517 |
| Agente 4 – Riscos e oportunidades | achados com regra aplicada | **73,6%** | — |
| Agente 5 – Recomendações | escopos distintos no top 4 | **2,0** | máximo possível 3 |
| Contratos JSON (3, 4 e 5) | erros de validação | **0** | — |

> ⚠️ **Agente 3:** os números são do substituto `ridge_fallback`, porque o XGBoost não estava instalado no ambiente onde rodei. Com `pip install xgboost`, rode de novo o agente e o `avaliar_agentes.py` antes de levar os valores para o TCC.

---

## 0. Base sintética (validação dos dados)

**Onde:** `src/sintetico/validacao.py` → `docs/Arquitetura/RELATORIO_VALIDACAO.md`

Cada métrica é comparada a um alvo extraído da base real e recebe uma nota:

```
erro = |valor − alvo| ÷ |alvo|            (ou absoluto, para taxas)
nota = 100 − 30·(erro/tol)                se erro ≤ tol
nota = máx(0; 70·(1 − (erro−tol)/(2·tol)))  se erro > tol
```

ou seja: erro zero = 100, erro igual à tolerância = 70, erro ≥ 3× a tolerância = 0.

| Dimensão | Peso | O que mede | Tolerância | Nota |
|---|---|---|---|---|
| Integridade | 0,25 | chaves válidas, sem nulos/duplicatas/lacunas, demanda ≥ 0, share em [0,1] | binária | 100,00 |
| Reconciliação | 0,20 | Σ outlets = setor; Σ produtos = mercado | 1e-6 relativo | 100,00 |
| Distribuição | 0,20 | média, desvio, CV e quantis de mercado, próprios, share e objetivo | 15% | 97,32 |
| Tendência | 0,15 | CAGR de próprios, concorrentes e mercado (convenção do Excel) | 5 p.p. | 97,05 |
| Sazonalidade | 0,15 | erro relativo médio do índice sazonal por UF × mês | 10% | 96,68 |
| Correlação | 0,05 | erosão de share monotônica, share anual, concentração Pareto | 15% | 94,87 |

**Fidelity score** = média das dimensões ponderada pelos pesos → **98,27**.

Complementos: 83 testes automatizados (`tests/test_pipeline_sintetico.py`), incluindo reconciliação semanal = mensal, reprodutibilidade da semente e continuidade dos meses criados pela execução semanal.

---

## 1. Agente 1 – Classificador de impacto

**Onde:** `src/noticias/classificacao.py` · avaliação em `scripts/avaliar_agentes.py`

**Modelo em uso:** Naive Bayes multinomial treinado no corpus anotado, que substitui o mBERT enquanto o fine-tuning não fica pronto (seção 3.4.1 do TCC1). Notícias que já estão no corpus usam o rótulo humano; com `PAINEL_USAR_LLM=1`, as novas são classificadas por LLM.

**Referência (verdade):** voto majoritário dos 3 avaliadores; empate total (3 rótulos diferentes) = médio, o caso que o Agente de Desempate resolve.

**Protocolo:** 5 partições aleatórias 80/20 sobre as 1.104 notícias do corpus **sem duplicatas** (há 311 títulos repetidos; sem remover, a mesma notícia cairia no treino e no teste e inflaria a métrica).

**Métricas** (as previstas na seção 3.7 do TCC1):

| Métrica | Definição |
|---|---|
| Precisão por classe | VP ÷ (VP + FP): das notícias que o modelo chamou de "alto", quantas eram alto |
| Recall por classe | VP ÷ (VP + FN): das notícias alto, quantas o modelo achou |
| F1 por classe | média harmônica de precisão e recall |
| F1 macro | média simples dos F1 das 3 classes; não favorece a classe mais frequente |
| Acurácia | acertos ÷ total |
| Kappa de Cohen | concordância modelo × referência descontado o acaso |

**Resultados:**

| Classificador | Acurácia | F1 macro | Kappa | F1 alto | F1 médio | F1 baixo |
|---|---|---|---|---|---|---|
| **Naive Bayes (em uso)** | **0,677** | **0,666** | **0,459** | **0,657** | **0,709** | **0,633** |
| Regras por palavra-chave | 0,460 | 0,458 | 0,196 | 0,417 | 0,432 | 0,526 |
| Classe majoritária (sempre "médio") | 0,499 | 0,222 | 0,000 | 0 | 0,666 | 0 |

Precisão e recall do Naive Bayes: alto 0,712 / 0,610 · médio 0,669 / 0,753 · baixo 0,675 / 0,597.

Matriz de confusão (linhas = referência, colunas = previsto):

| | alto | médio | baixo |
|---|---|---|---|
| **alto** | 111 | 61 | 10 |
| **médio** | 39 | 415 | 97 |
| **baixo** | 6 | 144 | 222 |

Quase todos os erros são entre classes vizinhas: alto é confundido com baixo em só 16 de 1.105 casos.

**Teto de referência:** entre os próprios avaliadores humanos, o **kappa de Fleiss é 0,1858** e a concordância total é 27,42% (recalculado pelo script, igual ao TCC1). O kappa do modelo contra o rótulo consolidado (0,459) fica acima da concordância entre humanos, o que é esperado: o rótulo consolidado já é uma média que remove parte do ruído individual.

---

## 2. Agente 2 – Validador

**Onde:** `validar_noticias()` em `src/noticias/classificacao.py`

**Regra:** mantém só impacto **alto ou médio** e com vínculo ao mercado (escopo próprio, concorrente ou mercado).

**Métricas de qualidade** (decisão manter × descartar, mesmo protocolo do Agente 1):

| Classe | Precisão | Recall | F1 | Suporte |
|---|---|---|---|---|
| manter | 0,807 | 0,854 | **0,830** | 733 |
| descartar | 0,675 | 0,597 | 0,633 | 372 |

Acurácia 0,767. O recall de "manter" é o mais importante: uma notícia relevante descartada nunca chega aos Agentes 4 e 5.

**Métricas de operação** (base atual):

| Métrica | Definição | Valor |
|---|---|---|
| Taxa de retenção | notícias mantidas ÷ total | 59,6% |
| Cobertura de vínculo | notícias com escopo ≠ nenhum ÷ total | 83,2% |
| Origem do rótulo | corpus anotado / Naive Bayes / LLM / regras | 124 / 440 / 0 / 0 |

---

## 3. Agente 3 – Analisador de oportunidades de mercado

**Onde:** `src/agentes/agente3_oportunidades_mercado.py` (métricas gravadas no próprio `agente3_saida.json`)

**Alvo previsto:** variação do market share do produto próprio na alocação no mês seguinte.

**Protocolo:** partição temporal — treino de 03/2023 a 05/2026, teste nos 3 últimos meses com alvo conhecido (06–08/2026), sem embaralhar o tempo. 421.863 linhas de treino e 32.451 de teste.

| Métrica | Definição | Modelo | Persistência | Média 3m |
|---|---|---|---|---|
| MAE | média de \|share previsto − share real\| | **0,00440** | 0,00443 | 0,00517 |
| RMSE | raiz da média dos erros ao quadrado; pune erros grandes | **0,01109** | 0,01210 | 0,01284 |
| R² | fração da variância explicada | **0,9675** | 0,9613 | 0,9564 |
| Acerto de direção | sinal da variação correto quando \|variação\| > 0,5 p.p. | **80,3%** | — | — |

- **Ganho sobre a melhor baseline:** (1 − MAE modelo ÷ MAE da melhor baseline) × 100 = **+0,56%**. O campo `supera_baseline` é o critério de admissibilidade: se for falso, o JSON vai com status "alerta".
- **Previsão de mercado** (usada para dimensionar oportunidades em unidades): WAPE = Σ\|erro\| ÷ Σ\|real\| = **0,049**, contra 0,093 do método ingênuo (repetir o mês atual). O WAPE só é usado para comparar métodos no mesmo teste, porque não é consistente em série com tendência (HYNDMAN, 2025).
- **Explicabilidade:** importância global = média de \|SHAP\| por variável. Top 3: share do mês, share médio 12m e share dos próprios na alocação.
- **Saída:** 6.091 recomendações para 10/2026 (479.679 un de oportunidade e 120.916 un de risco); 10% recebem prioridade alta por construção.

**Leitura honesta:** a persistência é uma baseline forte no grão alocação × produto (R² 0,96), e o ganho do ridge é pequeno. É exatamente o ponto em que o XGBoost precisa mostrar valor; se não superar, o resultado deve ser reportado assim.

---

## 4. Agente 4 – Identificador de riscos e oportunidades

**Onde:** `src/agentes/agente4_riscos_oportunidades.py`

Não há "gabarito" para o cruzamento notícia × oportunidade, então as métricas são **de cobertura e coerência**. A avaliação de mérito é a da 2ª fase da seção 3.7 do TCC1, que avalia as recomendações sobre os dados históricos.

| Métrica | Definição | Valor |
|---|---|---|
| Notícias avaliadas | alto/médio vindas do Agente 2 | 336 |
| Setores com achado | setores com ≥ 1 cruzamento (de 27) | 25 |
| Achados | pares notícia × item do Agente 3 guardados | 4.000 |
| Por categoria | oportunidade / risco médio prazo / risco imediato | 1.046 / 1.636 / 1.318 |
| % com regra aplicada | achados em que uma regra de sensibilidade do Agente 3 se aplicou | 73,6% |
| Multiplicador médio | efeito médio das regras (1 = neutro) | 1,19 |
| Score médio | ver fórmula abaixo | 0,628 |

```
score = peso do impacto (alto 1,0 | médio 0,6) × relevância × efeito da regra
      × (0,5 + 0,5 × valor relativo do item) × recência (meia-vida 180 dias)
```

Os 2 setores sem achado (AP e RR) não têm itens no Agente 3, porque os volumes ficam abaixo do mínimo de 5 unidades.

---

## 5. Agente 5 – Gerador de recomendações

**Onde:** `src/agentes/agente5_recomendacoes.py`

| Métrica | Definição | Valor |
|---|---|---|
| Setores com destaque | setores com as 4 notícias preenchidas | 25 de 27 |
| Destaques por setor | média | 4,0 |
| Escopos distintos no top 4 | média de escopos diferentes (próprio/concorrente/mercado) entre as 4 notícias | 2,0 |
| Repetição de oportunidades | 1 − oportunidades distintas ÷ oportunidades exibidas no setor | 19,3% |
| Notícias distintas no painel | quantas notícias diferentes aparecem somando todos os setores | 6 |
| Urgência | alta / média / baixa | 92 / 8 / 0 |
| Modo | heurístico / LLM | 100 / 0 |

As duas métricas de diversidade existem porque, sem penalização, as 4 notícias eram sempre do mesmo assunto e apontavam para as mesmas oportunidades. O baixo número de notícias distintas (6) mostra que as notícias de maior impacto são nacionais e se repetem entre setores — o que muda de setor para setor são as oportunidades afetadas.

**Avaliação de mérito (pendente, prevista no TCC1):** avaliar uma amostra de recomendações com os gestores, por exemplo pertinência e acionabilidade em escala de 1 a 5, ou verificar sobre 12 meses de histórico se as alocações sinalizadas tiveram de fato a variação de share prevista.

---

## 6. Contratos e operação

| Verificação | Onde | Valor atual |
|---|---|---|
| `validar_saida()` do Agente 3 | estrutura, tipos, domínios, coerência tipo × natureza | 0 erros |
| `validar_saida()` do Agente 4 | notícia catalogada, categoria válida, score ≥ 0 | 0 erros |
| `validar_saida()` do Agente 5 | urgência válida, destaque com oportunidade | 0 erros |
| Consolidação semanal → mensal | soma das semanas = total do mês (executar_semana, última semana) | reconcilia (teste 10/2026) |
| KPI semanal × histórico | KPI calculado na semana idêntico ao do histórico | diferença máxima 0 |

## Agente de Desempate

Não tem métrica própria calculada: ele só é acionado nos casos de discordância total (3 rótulos diferentes). Hoje esses casos entram na referência como "médio". Quando o agente for executado sobre esses casos, o protocolo do Agente 1 deve ser repetido com os rótulos desempatados, e a diferença de F1 mostra o efeito do desempate.

Atenção: o código dele está em `src/agents/agente4_riscos_oportunidades.py` (nome trocado); o Agente 4 real está em `src/agentes/`.

---

## Atualização (07/10/2026) – Agentes 1, 2 e Desempate integrados ao Orquestrador

Os agentes enviados (mBERT, Validador com k-NN e Desempate por LLM local) foram integrados em `src/agentes/` e são coordenados por `src/agentes/orquestrador.py` (`python scripts/orquestrar.py corpus|noticias|semana|cruzar|status`). O protocolo de avaliação passou a ser o da missão `corpus`: validação cruzada estratificada de 5 partições, corpus **sem duplicatas** (311 textos repetidos removidos) e casos de discordância total fora do treino até o Desempate resolvê-los.

| Componente | Motor nesta execução | Resultado |
|---|---|---|
| Desempate | revisão manual (Ollama indisponível) | 164 casos de discordância total identificados, 0 resolvidos |
| Agente 1 | Naive Bayes (substituto do mBERT) | F1-macro 0,686 ± 0,016; Kappa de Cohen 0,494 (969 notícias) |
| Agente 2 | encoder lexical (substituto) | 469 aprovadas; recall das notícias Alto 0,914; precisão da aprovação 0,795; regra R2 acerta 27,8% |
| Agente 3 | **XGBoost** | MAE 0,005009 vs 0,005421 da persistência (+7,6%); R² 0,818; acerto de direção 47,4% |

Os números do mBERT e do encoder semântico saem de `python scripts/orquestrar.py corpus` em máquina com GPU e entram automaticamente no artigo (`python scripts/gerar_artigo.py`). Cada execução grava `data/agentes/metricas_agentes_1_2_desempate_<motor>.json`.

## Atualização 18/10/2026 — corpus ampliado e balanceado

**Arquivo que treina os Agentes 1 e 2:** `data/estatistica_kappa/noticias_classificadas_ampliado.xlsx`
(`CORPUS_PADRAO` em `src/agentes/orquestrador.py` e `CAMINHO_CORPUS` em `src/noticias/classificacao.py`).
O corpus original (`noticias_classificadas.xlsx`) foi mantido intacto. Gerado por `python scripts/ampliar_corpus_noticias.py`
a partir de `data/estatistica_kappa/ampliacao/coleta_bruta.jsonl`; relatório em `data/estatistica_kappa/ampliacao/relatorio_ampliacao.json`.

| | Original | Ampliado |
|---|---|---|
| Linhas | 1.415 (1.100 únicas) | 1.431 (1.100 humanas + 331 simuladas) |
| Alto / Médio / Baixo / discordância | 174 / 404 / 387 / 135 (únicas) | 425 / 425 / 425 / 156 |
| Kappa de Fleiss | 0,1858 (0,1694 nas únicas) | 0,2392 (simuladas: 0,2066) |

Notícias novas: 187 STAT, 33 The Guardian (extraídas pelo ETL e nunca anotadas), 111 outros RSS/Bing News.
Anotação simulada: rótulo de cada avaliador sorteado de P(rótulo do avaliador | classe consolidada), estimada no corpus humano, condicionado ao julgamento de referência (semente 42). Coluna `origem_anotacao` = humana/simulada.

| Métrica (motores substitutos) | Original (969) | Ampliado (1.275) | Ampliado – só humanas (965) |
|---|---|---|---|
| Agente 1 F1-macro | 0,6856 ± 0,016 | 0,7102 ± 0,019 | 0,6848 |
| Agente 1 F1 Alto | 0,7106 | 0,8414 | 0,7268 |
| Agente 1 Kappa de Cohen | 0,4943 | 0,5729 | 0,4908 |
| Agente 2 precisão / recall da aprovação | 0,7953 / 0,6453 | 0,8508 / 0,7447 | — |
| Agente 2 recall Alto | 0,9138 | 0,9482 | — |

Métricas por origem: `data/agentes/agente1/agente1_metricas_por_origem.json`. Métricas anteriores preservadas em `data/agentes/antes_ampliacao/`.
Ressalva: o ganho global vem sobretudo das notícias simuladas de impacto Alto; nas notícias humanas o F1-macro ficou estável.

## Atualização 20/10/2026 — reclassificação das notícias novas por guia de anotação

As notícias novas foram reclassificadas por três avaliadores simulados que aplicam um **guia de anotação** comum
(critério: impacto no faturamento de medicamentos para diabetes). Caso claro (julgamento da coleta = regra do guia):
os três dão a mesma classe, com lapso de 5% na direção da tendência de cada avaliador; caso limítrofe: cada um escolhe
entre as duas classes candidatas segundo sua tendência. Script: `docs/Arquitetura/ampliar_corpus_noticias.py --eps 0.05`.
Versão anterior preservada em `data/agentes/antes_ampliacao/noticias_classificadas_ampliado_v1_perfis.xlsx`.

| | Valor |
|---|---|
| Notícias novas | 310 (170 STAT, 41 Guardian, 99 outros) — 296 claras, 14 limítrofes |
| Corpus ampliado | 1.410 linhas: 425 Alto, 425 Médio, 425 Baixo, 135 discordância total (todas humanas) |
| Kappa de Fleiss — notícias novas | **0,7539** (consequência do protocolo, não medida de confiabilidade humana) |
| Kappa de Fleiss — humanas (1.100 únicas) | 0,1694 (inalterado) |
| Kappa de Fleiss — corpus inteiro | 0,3514 (teto com novas perfeitas: 0,3803) |
| Agente 1 F1-macro (1.275) | 0,7276 ± 0,022 · humanas 0,6896 · novas 0,5940 |
| Agente 2 precisão / recall / recall Alto | 0,8602 / 0,7459 / 0,9482 |
