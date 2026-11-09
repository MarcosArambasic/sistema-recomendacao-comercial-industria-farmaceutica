# Como rodar o pipeline

Guia operacional. Atualizado em 20/09/2026.

---

## Visão geral: três scripts, três momentos

| Script                        | Quando roda           | O que faz                                                                     |
| ----------------------------- | --------------------- | ----------------------------------------------------------------------------- |
| `gerar_base_inicial.py`     | **Uma vez**     | Gera a base mensal (demanda + cota) dos 44 meses                              |
| `gerar_backfill_semanal.py` | **Uma vez**     | Roda o pipeline semanal sobre todo o histórico → dados de treino do XGBoost |
| `executar_semana.py`        | **Toda semana** | Roda as 7 etapas para**uma** semana comercial                           |

A ordem importa: o backfill consome a base mensal, e a execução semanal consome a base mensal também. Rode nessa sequência na primeira vez.

---

## ⚠️ Antes de tudo: regere a base

As tabelas semanais que você gerou estão com `periodo_demanda = 1970-01-01`. Era um bug meu: eu convertia as datas para inteiro e de volta para caber numa operação de pivot, e esse round-trip quebra no **pandas 3.x** (que é a versão do seu `requirements.txt`). No pandas 2.x funcionava, por isso passou nos meus testes.

Já corrigi — as datas agora vêm por `merge` com o calendário, sem conversão de tipo. Também pus uma trava no export que **recusa** gravar qualquer data fora de 2000–2100, para isso não passar silencioso de novo.

Os arquivos antigos não são recuperáveis (a data foi perdida, não embaralhada). Regere:

```bash
python scripts/gerar_base_inicial.py --particionar
python scripts/gerar_backfill_semanal.py --particionar --sem-demanda-semanal
```

---

## Execução semanal

### Como dizer qual semana

```bash
# 1. a PRÓXIMA semana depois da última calculada (padrão; é o que o botão do painel faz)
python scripts/executar_semana.py

# 2. semana que contém hoje
python scripts/executar_semana.py --hoje

# 3. semana que contém uma data específica
python scripts/executar_semana.py --data 2026-03-15

# 4. semana explícita pelo número
python scripts/executar_semana.py --ano 2026 --mes 3 --semana 2
```

**Sem trava de data:** qualquer semana pode ser gerada, inclusive à frente do dia de hoje. Se o mês ainda não tem demanda mensal (está além da base histórica), a demanda é **criada pelo modelo calibrado** (`src/sintetico/extensao.py`) e repartida nas semanas; a semana pedida é recortada e o pipeline roda normalmente. Na **última semana** do mês, a soma das semanas é consolidada na `fato_demanda_aberta_mensal` (pasta da semana). A criação é a continuação exata da base (mesmos outlets, mesma sazonalidade por SKU, processos AR(1) continuam do último valor) e determinística por mês: as semanas de um mês sempre somam o mesmo total.

### Antes de rodar, veja a numeração do mês

O número da semana não é óbvio, porque a primeira e a última são truncadas nos limites do mês:

```bash
python scripts/executar_semana.py --listar-semanas 2026-04
```

```
Semanas comerciais de 04/2026 (5 no total):

  semana  inicio       fim          dias  periodo_demanda
       1  01/04/2026   04/04/2026      4  2026-04-04  <- primeira (calcula cota e KPI)
       2  05/04/2026   11/04/2026      7  2026-04-11
       3  12/04/2026   18/04/2026      7  2026-04-18
       4  19/04/2026   25/04/2026      7  2026-04-25
       5  26/04/2026   30/04/2026      5  2026-04-30  <- ultima (consolida o mes)
```

Repare que abril/2026 começa numa quarta, então a semana 1 tem só 4 dias. Alguns meses têm 6 semanas; fevereiro/2026 tem exatamente 4.

### Qual foi a última semana que eu rodei?

```bash
python scripts/executar_semana.py --ultima-semana
```

```
ULTIMA SEMANA CALCULADA (fato_analise_semanal)
  semana          : 2026-09 semana 5 de 5
  periodo         : 27/09/2026 a 30/09/2026
  periodo_demanda : 2026-09-30
  linhas          : 10,817
  arquivo         : .../data/sintetico/fato_analise_semanal.csv

PROXIMA A RODAR: 2026-10 semana 1 (01/10/2026 a 03/10/2026)
  python scripts/executar_semana.py --ano 2026 --mes 10 --semana 1
```

A resposta vem dos próprios dados: o script lê todos os `fato_analise_semanal*.csv` (o backfill em `data/sintetico/` e as pastas `data/semanal/AAAA-MM-SN/`) e pega o maior `periodo_demanda`.

O `--listar-semanas AAAA-MM` marca a semana que contém hoje (`<- semana ATUAL (hoje)`) e a última calculada (`<- ULTIMA CALCULADA`), e mostra as duas no rodapé mesmo quando caem em outro mês.

### Todos os parâmetros

| Parâmetro                           | Default                     | Para quê                                      |
| ------------------------------------ | --------------------------- | ---------------------------------------------- |
| `--data AAAA-MM-DD`                | hoje                        | Qualquer dia dentro da semana alvo             |
| `--ano` / `--mes` / `--semana` | —                          | Semana explícita (os três juntos)            |
| `--listar-semanas AAAA-MM`         | —                          | Só mostra o calendário e sai                 |
| `--ultima-semana`                  | —                          | Mostra a última calculada e a próxima, e sai |
| `--dir-base`                       | `data/sintetico`          | Onde está a base mensal                       |
| `--dir-saida`                      | `data/semanal/AAAA-MM-SN` | Onde gravar                                    |
| `--classifica-kpis`                | procura no`dir-base`      | CSV com as faixas dos KPIs                     |
| `--regras-diagnostico`             | procura no`dir-base`      | CSV com os textos de diagnóstico              |
| `--carregar`                       | desligado                   | Além do CSV, grava no Supabase                |

---

## O que acontece em cada semana

O script decide sozinho o que rodar, olhando se a semana é a primeira ou a última do mês:

| Etapa                                | Roda quando                  | Tabela                           |
| ------------------------------------ | ---------------------------- | -------------------------------- |
| 1 — demanda semanal                 | sempre                       | `fato_demanda_aberta_semanal`  |
| 3 — alocação + acumulado          | sempre                       | `fato_demanda_alocada_semanal` |
| 5 — desempenho                      | sempre                       | `fato_desempenho_semanal`      |
| 6 — análise semanal                | sempre                       | `fato_analise_semanal`         |
| 7 — classificação                 | sempre                       | `fato_analise_semanal_final`   |
| **2 — cota do mês**          | **só 1ª semana**     | `fato_cota_mensal`             |
| **4 — KPIs do mês anterior** | **só 1ª semana**     | `fato_analise_kpi`             |
| **consolidação mensal**      | **só última semana** | `fato_demanda_aberta_mensal`   |

O log deixa isso explícito logo no começo:

```
SEMANA ALVO: 2026-03 semana 1
  periodo      : 01/03/2026 a 07/03/2026 (7 dias)
  periodo_demanda (chave no banco): 2026-03-07
  primeira do mes: SIM - calcula cota e KPI
  ultima do mes  : nao
```

### Por que a cota e os KPIs só na primeira semana

A cota vale para o mês inteiro — ela não muda na semana 3. Os KPIs também são mensais e usam o **mês anterior fechado**, porque o mês corrente ainda não terminou. Essa defasagem não é só realismo operacional: ela evita vazamento temporal no treino do XGBoost.

Nas semanas 2 em diante o script ainda **recalcula** cota e KPI internamente (precisa deles para a análise), mas não os reexporta, já que não mudaram.

---

## Idempotência

Rodar a mesma semana duas vezes produz exatamente o mesmo resultado. Não há estado guardado entre execuções — tudo é recalculado de forma determinística a partir da base mensal e da seed.

Se você rodar com `--carregar`, a carga apaga o período antes de reinserir, então também não duplica.

**Uma ressalva sobre o `\copy`:** se você importar os CSVs manualmente via psql, aí sim duplica — o `\copy` sempre insere. Limpe antes:

```sql
delete from public.fato_analise_semanal where periodo_demanda = '2026-03-14';
```

---

## Saída

```
data/semanal/2026-03-S2/
├── fato_demanda_aberta_semanal.csv
├── fato_demanda_alocada_semanal.csv
├── fato_desempenho_semanal.csv
├── fato_analise_semanal.csv
├── fato_analise_semanal_final.csv
└── analise_semanal_com_diagnostico.csv   ← com o texto do diagnóstico anexado
```

O último arquivo não corresponde a nenhuma tabela do banco — é a análise já cruzada com `dim_regras_diagnostico`, para você conferir se os diagnósticos fazem sentido antes de alimentar o Agente 5.

Tempo de execução: ~40 s por semana (a maior parte é o recálculo dos KPIs sobre 18 meses de histórico).

---

## As duas dimensões sugeridas

### `dim_classifica_kpis`

Já estava pronta (`dim_classifica_kpis_SUGERIDO.csv`). Converte os valores numéricos em códigos de 1 a 4.

### `dim_regras_diagnostico` — nova

`data/sintetico/dim_regras_diagnostico_SUGERIDO.csv` (e o `.sql` equivalente para colar no Supabase).

**1.024 combinações** (4 níveis × 5 indicadores), **80 textos distintos**, cobertura de 100% — nenhuma linha da análise fica sem diagnóstico.

A lógica é hierárquica, não aleatória. A **cobertura projetada** é a espinha, porque é por ela que o representante é cobrado. Os outros quatro indicadores entram como contexto, explicando por que aquele nível de cobertura é mais ou menos grave:

| Indicador       | Papel no diagnóstico                                       |
| --------------- | ----------------------------------------------------------- |
| market share    | posição competitiva — está perdendo ou ganhando espaço |
| CAGR            | o mercado está crescendo ou encolhendo                     |
| índice sazonal | o mês é naturalmente forte ou fraco                       |
| CV              | a demanda é previsível ou errática                       |

Exemplos do que a hierarquia captura:

> **Risco critico de nao atingir a cota no mes de pico** — mercado em expansao com share baixo: espaco sendo cedido a concorrencia. Acao: acionar plano de contingencia ainda dentro do mes.

> **Cota atingida, porem com share baixo em mercado crescente** — mes de alta sazonalidade: o periodo mais decisivo do ano. Acao: reavaliar o dimensionamento da cota no proximo ciclo.

Esse segundo caso é o que eu acho mais interessante para a banca: bater a meta **não** significa bom desempenho quando o share está caindo num mercado em expansão — a meta é que estava subdimensionada.

**Isto é uma sugestão, não regra validada.** O prompt mestre pede que os diagnósticos não sejam inventados pelo código, e concordo. O que fiz foi montar a grade completa e coerente para vocês ajustarem os textos, em vez de escreverem 1.024 linhas na mão. Revise principalmente os verbos de ação — eles viram recomendação no Agente 5.

Para carregar:

```bash
psql "$DATABASE_URL" -f data/sintetico/dim_regras_diagnostico_SUGERIDO.sql
```

---

## Agente 3 — Analisador de Oportunidades de Mercado

`src/agentes/agente3_oportunidades_mercado.py` (o antigo `agente3_comparador_vendas.py` virou um atalho para este módulo)

### Base de treino: `fato_analise_kpi_final`

É a `fato_analise_kpi` mensal com duas coisas a mais: **setor e alocação** de cada outlet (via `fato_alocacao`) e os quatro indicadores de mercado **classificados** pelas faixas da `dim_classifica_kpis` (`cd_mrkt_share`, `cd_cagr_medio`, `cd_cv`, `cd_indice_sazonal`). Os valores numéricos ficam ao lado dos códigos.

```bash
# DDL (uma vez, no Supabase)
supabase/scripts/criacao_fato_analise_kpi_final.sql

# gerar a partir dos CSVs ja exportados (~1,5 min)
python scripts/gerar_kpi_final.py
```

Na operação semanal, a 1ª semana de cada mês já gera a `fato_analise_kpi_final` do mês anterior junto com a `fato_analise_kpi`.

### O que o modelo prevê

A **variação do market share** do produto próprio na alocação no mês seguinte: `mrkt_share(t+1) − mrkt_share(t)`. O XGBoost é comparado em toda execução com duas baselines — persistência (share não muda) e média de 3 meses — e o JSON informa `ganho_sobre_baseline_pct` e `supera_baseline`.

### Da previsão à recomendação

| Grandeza               | Cálculo                                                                |
| ---------------------- | ----------------------------------------------------------------------- |
| `share_previsto`     | share atual + variação prevista                                       |
| `mercado_previsto`   | mercado do mesmo mês do ano anterior × crescimento interanual recente |
| `share_benchmark_uf` | percentil 75 do share do produto entre as alocações da UF             |
| `oportunidade_un`    | mercado previsto × (benchmark − share previsto), se positivo          |
| `risco_un`           | mercado previsto × (share médio 3m − share previsto), se positivo    |

| Tipo                       | Regra                                                                           | Natureza     |
| -------------------------- | ------------------------------------------------------------------------------- | ------------ |
| `DEFENDER_SHARE`         | queda prevista 10 p.p. pior que a mediana dos pares (produto × UF), share alto | risco        |
| `RECUPERAR_SHARE`        | idem, share baixo                                                               | risco        |
| `CAPTURAR_CRESCIMENTO`   | gap para o benchmark e mercado crescendo acima da mediana da UF                 | oportunidade |
| `AMPLIAR_PENETRACAO`     | gap para o benchmark                                                            | oportunidade |
| `ANTECIPAR_PICO_SAZONAL` | mês previsto na faixa sazonal mais alta                                        | oportunidade |
| `MANTER_POSICAO`         | nenhum dos anteriores                                                           | neutro       |

Prioridade: 10% maiores "unidades em jogo" = alta, próximos 20% = média, resto = baixa. Todos os limiares ficam em `LIMIARES`, no topo do módulo.

**Vazamento evitado:** o `indice_sazonal` da `fato_analise_kpi` usa a média do ano inteiro, que no histórico inclui o mês t+1. Por isso o índice do mês corrente **não** entra como variável; entra o do mesmo mês do ano anterior (t−11).

### Rodar

```bash
pip install xgboost
python -m src.agentes.agente3_oportunidades_mercado                 # referência = último mês da base
python -m src.agentes.agente3_oportunidades_mercado --periodo 2026-09 --top 20
```

Saídas: `data/agentes/agente3_saida.json` (top N por setor, para o Agente 4) e `data/agentes/agente3_recomendacoes_completas.csv` (todas as linhas, para auditoria e painel).

### Contrato JSON (v2.0) — o que o Agente 4 usa

| Bloco                  | Conteúdo                                                                                                                                                                                    |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `modelo`             | algoritmo, métricas no teste temporal, baselines, ganho, WAPE da previsão de mercado, importância SHAP global                                                                             |
| `instrucoes_agente4` | filtro de notícias (`alto`, `medio`), chaves de cruzamento e **regras de sensibilidade** por tipo (ex.: notícia positiva de concorrente + `DEFENDER_SHARE` = confirma o risco) |
| `resumo`             | contagens por tipo e prioridade, unidades totais de oportunidade e risco                                                                                                                     |
| `setores[]`          | por setor: totais e os`itens` mais relevantes                                                                                                                                              |
| `setores[].itens[]`  | alocação, produto, tipo, prioridade, ação sugerida, indicadores (valor + faixa), previsão, 3 fatores SHAP, 3 concorrentes mais fortes na alocação e`chaves_noticia`                 |

`chaves_noticia` tem `id_produto`, `id_mercado`, `grupo`, `id_uf` e `ids_concorrentes` — o Agente 4 casa `noticia.id_produto` tanto com o produto próprio quanto com os concorrentes fortes daquela alocação.

`validar_saida(saida)` retorna a lista de erros (vazia = válido), em Python puro.

## Rotina

**Uma vez:**

1. `gerar_base_inicial.py --particionar`
2. `gerar_backfill_semanal.py --particionar --sem-demanda-semanal`
3. Revisar e carregar as duas dimensões sugeridas
4. `gerar_kpi_final.py` e treinar o Agente 3 sobre `fato_analise_kpi_final`

**Toda semana:**

```bash
python scripts/executar_semana.py
```

---

## Se der erro

| Mensagem                                       | O que fazer                                                                |
| ---------------------------------------------- | -------------------------------------------------------------------------- |
| `base mensal nao encontrada`                 | Rode`gerar_base_inicial.py` antes                                        |
| `o mes MM/AAAA nao existe na base mensal`    | O horizonte acabou. Ajuste`PERIODO_FIM` em `config.py` e regere a base |
| `datas fora da faixa esperada`               | A trava nova pegou uma conversão de tipo corrompida — me avise, é bug   |
| `dim_classifica_kpis nao encontrada`         | Passe`--classifica-kpis CAMINHO.csv`                                     |
| `MM/AAAA tem N semanas, nao existe semana X` | Use`--listar-semanas` para ver a numeração                             |

**Sobre estender o horizonte:** mudar `PERIODO_FIM` re-sorteia o ruído AR(1) e **altera todo o histórico**, porque a série é gerada de uma vez. Se você vai rodar semanalmente por meses, defina um `PERIODO_FIM` folgado (ex.: `2027-12`) **de uma vez** e regere tudo uma única vez.
