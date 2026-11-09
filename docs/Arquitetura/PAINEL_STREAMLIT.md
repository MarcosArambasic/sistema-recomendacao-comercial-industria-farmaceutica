# Painel do setor (Streamlit)

```bash
pip install -r requirements.txt
streamlit run app/painel.py
```

A primeira abertura agrega os CSVs grandes por setor e grava o resultado em `data/painel/` (uns 15 s). Depois disso o painel abre em segundos; o agregado só é refeito se o arquivo de origem mudar.

## A tela

| Área | O que mostra | De onde vem |
|---|---|---|
| Barra lateral – **Simular setor** | escolhe o setor; tudo na tela passa a ser dele | `dim_setor` + `dim_uf` |
| Barra lateral – **Gerar nova semana** | roda a próxima semana do calendário; abaixo, a semana vigente e a próxima | `scripts/executar_semana.py` |
| Barra lateral – **Gerar mais notícias** | ETL das APIs + Agentes 1, 2, 4 e 5 | `scripts/gerar_noticias.py` |
| Indicadores mês a mês | market share dos próprios, cobertura da cota, CAGR, CV e índice sazonal, com variação contra o mês anterior | `fato_analise_kpi_final` + `fato_cota_mensal` |
| Projeção das semanas do mês | barras = realizado acumulado, linha = projeção de fechamento, tracejado = cota | `fato_desempenho_semanal` |
| Recomendações do setor | as 5 maiores oportunidades/riscos do Agente 3 | `data/agentes/agente3_saida.json` |
| Notícias mais impactantes | 4 notícias, as oportunidades que cada uma afeta e a recomendação do Agente 5 | `data/agentes/agente5_saida.json` |

## O que cada botão executa

**Gerar nova semana**
1. `executar_semana.py --ano --mes --semana` com a **próxima** semana depois da última calculada.
2. Se for a 1ª semana do mês: retreina o Agente 3 (os KPIs do mês anterior acabaram de fechar).
3. Agentes 4 e 5 recruzam as notícias já classificadas com as novas oportunidades.

**Gerar mais notícias**
1. Extração GNews e Guardian — só roda a fonte cuja chave está no `.env` (`API_KEY_GNEWS_PRODUCAO_1..11`, `API_KEY_GUARDIAN`). Sem chave, usa as notícias já transformadas.
2. Transformação (os scripts de `src/data/`).
3. Agentes 1 e 2: impacto, sentimento e vínculo com produto/concorrente/mercado; só alto e médio seguem.
4. Agente 4 cruza com o Agente 3; Agente 5 escreve as recomendações.

Os dois botões rodam cada etapa como processo separado e mostram o log ao vivo no topo da tela.

## Classificação das notícias (Agentes 1 e 2)

O impacto vem da melhor fonte disponível, nesta ordem, e a coluna `origem_classificacao` registra qual foi usada:

| Ordem | Fonte | Quando |
|---|---|---|
| 1 | rótulo humano do corpus (voto majoritário dos 3 avaliadores) | a URL já está em `noticias_classificadas.xlsx` |
| 2 | LLM (OpenAI) | `PAINEL_USAR_LLM=1` e `OPENAI_API_KEY` no `.env` |
| 3 | Naive Bayes treinado no corpus | padrão — 67,7% de acurácia em validação (classe majoritária: 48,9%) |
| 4 | regras por palavra-chave | só se o corpus não existir (46% de acurácia) |

O vínculo com os produtos fica em `data/config/mapa_entidades_noticias.json`. Os produtos sintéticos não têm nome real, então o mapa diz qual molécula/marca corresponde a cada **grupo próprio** — por padrão empagliflozina → Grupo A e linagliptina → Grupo B. **Ajuste para o portfólio real** e rode `python scripts/gerar_noticias.py --sem-extracao --reclassificar`.

## Como o Agente 4 pontua um cruzamento

```
score = peso do impacto (alto 1,0 | médio 0,6)
      × relevância (próprio do mesmo grupo 1,0 | concorrente 0,5 + share dos concorrentes | mercado 0,5)
      × efeito (regra de sensibilidade publicada pelo Agente 3: confirma/agrava 1,4; amplifica 1,3; atenua 0,7)
      × valor relativo do item no setor
      × recência (meia-vida de 180 dias)
```

O Agente 5 escolhe as 4 notícias de maior score no setor, penalizando repetição do mesmo escopo e tema (para não mostrar 4 notícias sobre o mesmo assunto), e evita repetir a mesma oportunidade em notícias diferentes.

## Geração semana a semana (sem trava de data)

Cada clique em **Gerar nova semana** gera a semana seguinte à última calculada, independentemente do dia de hoje — dá para avançar quantas semanas quiser.

| Mês da semana | De onde vem a demanda |
|---|---|
| dentro da base histórica (até 09/2026) | `fato_demanda_aberta_mensal.csv` (reprocessamento) |
| além da base | **criada pelo modelo calibrado** (`src/sintetico/extensao.py`), repartida nas semanas; o mês só entra na `fato_demanda_aberta_mensal` na **última semana**, pela soma das semanas |

A criação é a continuação exata do gerador: mesmos pontos de venda ativos, mesma concentração, mesma sazonalidade por SKU e os processos AR(1) continuam do último valor da base. Cada mês tem sua própria semente, então as semanas de um mês sempre somam o mesmo total, em qualquer ordem ou dia. Testado de 10/2026 S1 até 11/2026 S1: a consolidação de outubro reconciliou (soma das semanas = total do mês) e os KPIs calculados na semana são idênticos aos do histórico.

Tempo por clique: ~50 s; na 1ª semana do mês, mais o retreino do Agente 3.

Linha de comando equivalente:

```bash
python scripts/executar_semana.py              # próxima semana
python scripts/executar_semana.py --hoje       # semana que contém hoje
python scripts/executar_semana.py --ano 2026 --mes 12 --semana 2
```

Para voltar o relógio, apague as pastas `data/semanal/AAAA-MM-SN` das semanas que quer desfazer.

Quando um período existe tanto no histórico quanto numa pasta semanal (reprocessamento de semana antiga), o painel e o Agente 3 usam o **histórico**.

## Tema claro / escuro

O seletor **🌙 Escuro** no topo da barra lateral troca o tema. O Streamlit não permite trocar o tema por código, então o modo escuro é aplicado via CSS e os gráficos recebem as cores do tema escolhido.

## Filtro de notícias por UF (só exibição)

A lista "Notícias mais impactantes" mostra apenas notícias que **não citam UF nenhuma** (alcance nacional) ou que **citam a UF do setor simulado**. Notícias que citam somente outras UFs, no título, na descrição ou no conteúdo, são ocultadas. O painel exibe essa regra em um aviso fixo acima dos cards.

- A detecção usa o nome do estado, capitais de nome inequívoco e a sigla em contexto de notícia ("em MG", "(SP)", "-RJ"). "Pará" e "Acre" só contam com a grafia própria do estado, para não confundir com "para" e "acre". "Brasília" não conta como DF, porque notícias federais (Anvisa, Ministério da Saúde) citam a capital sem serem regionais.
- Se o filtro remover algum dos 4 destaques do Agente 5, a lista é completada com as próximas notícias do ranking do Agente 4 para o mesmo setor. Esses cards mostram a ação do Agente 3 e uma nota de que não têm texto do Agente 5.
- Nenhum agente foi alterado: o filtro está em `src/painel/dados.py` (`noticias_setor`, `ufs_citadas`).
