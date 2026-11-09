# Primeira Entrega — Diagnóstico técnico do pipeline de dados sintéticos

Sistema de Recomendações Comerciais para a Indústria Farmacêutica — TCC2
Data: 25/08/2026 · Fontes analisadas: `CRIACAO DO BANCO.txt`, `POPULA BANCO.txt`, `Parametros_Dados_Sinteticos.xlsx`

> Documento de análise **anterior à implementação**, conforme a seção 31 do prompt mestre. Nenhum código de geração foi escrito ainda. As seções J e K contêm bloqueios que precisam de decisão antes da codificação.

---

## ⚠️ 0. Alerta de segurança (ler primeiro)

As credenciais do Supabase foram coladas em texto no prompt. Três providências antes de qualquer coisa:

1. **Rotacionar as chaves** no painel do Supabase (Settings → API). Uma chave que trafegou em texto deve ser considerada comprometida.
2. **A chave informada não é a service_role.** O valor `sb_publishable_...` tem prefixo de chave *publishable/anon*. Com RLS ativo, ela **não consegue** fazer a carga em lote — a `service_role` real é obrigatória para o ETL, e deve viver apenas no `.env` local (nunca versionado).
3. A `DATABASE_URL` está com o placeholder `[YOUR-PASSWORD]` não preenchido — a senha do banco também deve ficar só no `.env`.

Nenhuma dessas credenciais foi gravada em arquivo neste projeto.

---

## A. Diagnóstico da arquitetura

**Parecer geral: a modelagem está bem construída e é adequada ao objetivo.** O desenho separa corretamente (i) dados de origem, (ii) consolidações, (iii) KPIs, (iv) classificação para o XGBoost, e a cadeia `demanda → cota → desempenho → KPI → análise → classificação` é coerente e auditável. As tabelas têm FKs declaradas e comentários — está acima da média para um TCC.

Há, porém, **quatro problemas estruturais** que precisam de decisão antes da implementação (detalhados em J):

1. **Volumetria incompatível com o Supabase free tier.** A granularidade especificada gera ~100 milhões de linhas / ~12,7 GB (~19 GB com índices), contra 500 MB do plano gratuito. Isso é 25x acima do limite — é o bloqueio mais grave e muda o desenho do projeto, não só a configuração.
2. **Nenhuma tabela fato tem constraint UNIQUE de grão.** Só existem PKs *identity*. Isso torna a idempotência exigida na seção 26 impossível de garantir no banco: reprocessar duplica silenciosamente.
3. **Nenhum índice foi criado** (`create index` aparece 0 vezes no DDL). Com dezenas de milhões de linhas, as consultas de consolidação e os JOINs do pipeline ficam inviáveis.
4. **A cota é definida por setor × produto, mas o desempenho é medido por alocação (outlet).** Falta a regra de rateio da cota entre os 100 outlets de cada setor — sem ela, `cobertura_cota` não é calculável no grão da tabela.

Um quinto ponto, menos grave mas relevante: **a aba "Correlações" do Excel ficou em branco e, na prática, tornou-se obsoleta** — as variáveis que ela mediria (`investimento_promocional`, `estoque_disponivel`) **não existem** no schema novo. As correlações que restam possíveis são as estruturais (produto × mercado, setor × demanda, sazonalidade × demanda), que o modelo hierárquico da seção G reproduz por construção.

---

## B. Mapa completo das tabelas

### Dimensões (estáticas)

| Tabela | PK | FKs | Volume | Status |
|---|---|---|---|---|
| `dim_mercado` | `id_mercado` | — | 1 | Populada |
| `dim_produto` | `id_produto` | `id_mercado` | 30 | Populada (10 próprios / 20 concorrentes) |
| `dim_uf` | `id_uf` | — | 27 | Populada |
| `dim_setor` | `id_setor` | `id_uf` | 27 | Populada (1 por UF) |
| `dim_outlet` | `id_outlet` | `id_uf` | 2.700 | Populada (100 por UF, 100% BRICK) |
| `fato_alocacao` | `id_alocacao` | `id_outlet`, `id_uf`, `id_setor` | 2.700 | Populada (1:1 com outlet) |
| `dim_classifica_kpis` | `id_classifica_kpis` | — | ? | **Pendente (manual)** |
| `dim_regras_diagnostico` | `id_regras_diagnostico` | — | ? | **Pendente (manual)** |
| `noticia` | `id_noticia` | `id_produto`, `id_mercado` | 1.415 | Pipeline separado (ETL de notícias já existente) |

Observação: `fato_alocacao` está nomeada como fato mas é estrutural (1:1 com outlet, sem dimensão temporal). Funciona, mas é uma dimensão de relacionamento — vale registrar isso na documentação do TCC para não confundir a banca.

### Fatos (gerados pelo pipeline)

| Tabela | Grão | Linhas estimadas | Origem |
|---|---|---|---|
| `fato_demanda_aberta_semanal` | semana × produto × outlet | 18.549.000 | **Gerador sintético** (raiz) |
| `fato_demanda_aberta_mensal` | mês × produto × outlet | 3.564.000 | Consolidação da semanal |
| `fato_demanda_alocada_semanal` | semana × produto × alocação | 18.549.000 | Semanal + alocação (acumulado) |
| `fato_cota_mensal` | mês × setor × produto | 35.640 | Histórico 6 meses + crescimento |
| `fato_analise_kpi` | mês × produto × outlet | 3.564.000 | Mensal consolidada |
| `fato_desempenho_semanal` | semana × produto × setor × alocação | 18.549.000 | Alocada + cota |
| `fato_analise_semanal` | semana × produto × alocação | 18.549.000 | KPI + desempenho |
| `fato_analise_semanal_final` | semana × produto × alocação | 18.549.000 | Análise + classificação |
| | **TOTAL** | **~99,9 milhões** | **~12,7 GB sem índices** |

---

## C. Granularidade e chave natural de cada tabela

A coluna "chave natural" é o que **deveria** virar constraint `UNIQUE` (hoje não existe em nenhuma):

| Tabela | Chave natural proposta |
|---|---|
| `fato_demanda_aberta_semanal` | (`periodo_demanda`, `nr_semana`, `id_produto`, `id_outlet`) |
| `fato_demanda_aberta_mensal` | (`periodo_demanda`, `id_produto`, `id_outlet`) |
| `fato_demanda_alocada_semanal` | (`periodo_demanda`, `nr_semana`, `id_produto`, `id_alocacao`) |
| `fato_cota_mensal` | (`periodo_demanda`, `id_setor`, `id_produto`) |
| `fato_analise_kpi` | (`periodo_demanda`, `id_produto`, `id_outlet`) |
| `fato_desempenho_semanal` | (`periodo_demanda`, `nr_semana`, `id_produto`, `id_alocacao`) |
| `fato_analise_semanal` | (`periodo_demanda`, `nr_semana`, `id_produto`, `id_alocacao`) |
| `fato_analise_semanal_final` | (`nr_ano`, `nr_mes`, `nr_semana`, `id_produto`, `id_alocacao`) |

Notas de granularidade:

- `id_uf` em `fato_demanda_aberta_*` é **redundante** (derivável de `dim_outlet`). Não é erro, é desnormalização para performance — mas cria risco de inconsistência se não for populado a partir do outlet.
- `id_setor` em `fato_desempenho_semanal` e `fato_analise_semanal` também é redundante (derivável de `fato_alocacao`). Mesma observação.
- Como `fato_alocacao` é 1:1 com `dim_outlet`, o grão "alocação" **é** o grão "outlet". Isso significa que `fato_desempenho_semanal` tem 2.700 linhas por produto/semana, não 27.

---

## D. Fluxo ETL

```
                  ┌─────────────────────────────────────────┐
   CARGA INICIAL  │ 0. Dimensões + fato_alocacao (prontas)   │
                  └────────────────────┬────────────────────┘
                                       │
                  ┌────────────────────▼────────────────────┐
                  │ 1. GERADOR SINTÉTICO (raiz de tudo)      │
                  │    modelo hierárquico → demanda semanal  │
                  │    por produto × outlet × semana         │
                  └────────────────────┬────────────────────┘
                                       │
       ┌───────────────────────────────┼───────────────────────────────┐
       │                               │                               │
┌──────▼──────────┐          ┌─────────▼─────────┐          ┌─────────▼─────────┐
│ 2. ETAPA 1      │          │ 3. ETAPA 3        │          │ 4. ETAPA 2        │
│ demanda semanal │─────────▶│ alocada semanal   │          │ cota mensal       │
│ (+ consolidação │          │ (acumulado no mês)│          │ (1ª semana do mês)│
│  mensal no fim  │          └─────────┬─────────┘          └─────────┬─────────┘
│  do mês)        │                    │                              │
└──────┬──────────┘                    └──────────┬───────────────────┘
       │                                          │
┌──────▼──────────────┐                 ┌─────────▼─────────┐
│ 5. ETAPA 4          │                 │ 6. ETAPA 5        │
│ KPIs (1ª semana,    │                 │ desempenho semanal│
│ sobre mês anterior) │                 │ (cobertura/projeç)│
└──────┬──────────────┘                 └─────────┬─────────┘
       │                                          │
       └──────────────────┬───────────────────────┘
                          │
              ┌───────────▼───────────┐
              │ 7. ETAPA 6            │
              │ fato_analise_semanal  │
              └───────────┬───────────┘
                          │
              ┌───────────▼────────────────────────────┐
              │ 8. ETAPA 7 — classificação             │
              │ dim_classifica_kpis → códigos          │
              │ dim_regras_diagnostico → diagnóstico   │
              │ → fato_analise_semanal_final           │
              └───────────┬────────────────────────────┘
                          │
              ┌───────────▼───────────┐
              │ 9. VALIDAÇÃO + SCORE  │
              │ de fidelidade (0-100) │
              └───────────────────────┘
```

---

## E. Dependências entre as etapas

| Etapa | Depende de | Cadência | Pode paralelizar? |
|---|---|---|---|
| 0. Dimensões | — | Uma vez | — |
| 1. Gerador | 0 | Uma vez (histórico) | Sim, por produto |
| 2. Demanda semanal | 1 | Toda semana | Sim, por semana |
| 2b. Consolidação mensal | 2 (mês completo) | Última semana do mês | Não |
| 3. Cota mensal | 2b (6 meses de histórico) | 1ª semana do mês | Sim, por setor |
| 4. Alocada semanal | 2, `fato_alocacao` | Toda semana | Não (acumulado é sequencial dentro do mês) |
| 5. KPIs | 2b (mês anterior) | 1ª semana do mês | Sim, por produto |
| 6. Desempenho | 3, 4 | Toda semana | Sim, por setor |
| 7. Análise semanal | 5, 6 | Toda semana | Sim |
| 8. Classificação | 7, `dim_classifica_kpis`, `dim_regras_diagnostico` | Toda semana | Sim |
| 9. Validação | 1–8 | Sob demanda | — |

**Dependência circular na carga inicial (atenção):** a cota do mês *M* precisa dos 6 meses anteriores consolidados, mas na carga histórica todos os meses são gerados de uma vez. A ordem obrigatória é: gerar **toda** a demanda (44 meses) → consolidar **todos** os meses → só então calcular cotas mês a mês em ordem cronológica. Os **6 primeiros meses (jan–jun/2023) não terão histórico suficiente** para a regra de cota — precisa de uma regra de *bootstrap* (ver K5).

---

## F. Fórmulas de cada KPI

### Market share (mensal)
```
mrkt_share = demanda_produto / demanda_mercado
```
Restrição: `0 <= mrkt_share <= 1`. Deve **resultar** da demanda gerada, nunca ser sorteado. O denominador é a demanda de **todos os 30 produtos** do mercado no mesmo grão/período.

### CAGR médio (anual)
Conforme a definição documentada no Excel (aba 3: *"CAGR com o valor da média mensal do ano início e fim"*):
```
CAGR = (media_mensal_ano_final / media_mensal_ano_inicial)^(1/n_anos) - 1
```
Validado contra os dados reais: mercado 1.648.252 → 3.009.953 em 2 anos ⇒ **0,3514**, batendo exatamente com o valor informado no Excel. Confirma a fórmula.

### Coeficiente de variação (anual)
```
CV = desvio_padrao / media
```
Calculado sobre a série mensal dos últimos 12 meses no grão da análise.

### Índice sazonal (mensal)
Conforme a metodologia do Excel (aba 3: *"demanda mensal / média mensal do ano"*):
```
indice_sazonal(m) = demanda_mensal(m) / media_mensal_do_ano(m)
```
Propriedade que o gerador deve preservar: **a média dos índices sazonais de um ano fechado é 1,0** (verificado nos dados reais: média = 1,0000 nos quatro blocos da aba `Índice Sazonal`).

### Cobertura de cota (semanal)
```
cobertura_cota           = demanda_acumulada_no_mes / cota_mensal
cobertura_cota_projetada = (demanda_acumulada / dias_corridos_decorridos * dias_do_mes) / cota_mensal
```
`dias_corridos_decorridos` = dias do início do mês até o fim da semana corrente.

### Cota mensal (parametrizável, seção 11 do prompt)
```
cota(M) = media(demanda_realizada[M-6 .. M-1]) * (1 + crescimento) * ajuste_sazonal(M)
```
com `crescimento = 0,03` por padrão. **Conflito detectado:** o Excel indica objetivo médio 15.397 contra demanda média 14.769, o que implica **+4,25%**, não +3%. Ver K4.

---

## G. Modelo estatístico proposto

Esta é a parte central do diagnóstico, e a análise dos dados reais mudou o desenho de forma importante.

### G.1. Descoberta crítica: o CV de 146% é transversal, não temporal

A aba "Distribuições Numéricas" informa demanda de mercado com média 88.740 e desvio 129.934 — **CV = 146%**. Gerar a demanda com uma distribuição de CV 146% por linha produziria uma série absurda (um território saltando de 300 para 100.000 de um mês para o outro).

Cruzando com a aba `Índice Sazonal` (24 meses reais por UF), a decomposição real é:

| Fonte de variação | CV real | Interpretação |
|---|---|---|
| **Entre territórios** (transversal) | 146,4% (próprios) / 142,0% (mercado) | SP = 103.434 vs RR = 325 → razão de **318x** |
| **Dentro do território ao longo do tempo** | **9,7%** (próprios) / **26,0%** (mercado) | volatilidade mensal real |

O CV de 26,03% informado na aba 3 como "Variabilidade" bate **exatamente** com o CV temporal do mercado calculado a partir da aba `Índice Sazonal` (26,02%). Isso confirma a leitura: **são duas variabilidades diferentes e o gerador precisa reproduzir as duas separadamente.**

### G.2. Descoberta crítica: produtos próprios caem enquanto o mercado cresce

| Ano | Mercado (média mensal) | Próprios (média mensal) | Share |
|---|---|---|---|
| 2024 | 1.648.252 | 426.124 | **25,85%** |
| 2025 | 2.462.863 | 408.351 | **16,58%** |
| 2026 | 3.009.953 | 352.177 | **11,70%** |
| | CAGR **+35,14%** | CAGR **−9,09%** | erosão de share |

Aplicar a tendência de +35,14% a tudo (leitura ingênua do campo "Tendência" do Excel) **inverteria o fenômeno central do negócio** e destruiria o KPI de market share, que é justamente o insumo do motor de diagnóstico. O gerador precisa de **duas tendências distintas**: mercado `+35,14%` a.a. e produtos próprios `−9,09%` a.a.

### G.3. Modelo hierárquico multiplicativo proposto

```
demanda(produto, outlet, semana) =
      escala_base(produto)                    -- nível do produto
    × fator_territorio(uf)                    -- log-normal, calibrado no real
    × fator_outlet(outlet | uf)               -- concentração intra-território
    × tendencia(t | tipo_produto)             -- +35,14% mercado / −9,09% próprios
    × sazonalidade(mes, produto, uf)          -- da aba Índice Sazonal
    × ruido_temporal(t)                       -- AR(1), CV 9,7% / 26,0%
    × fracao_semana(semana)                   -- rateio por dias úteis da semana
```

Escolhas e justificativas técnicas:

| Componente | Distribuição | Justificativa |
|---|---|---|
| `fator_territorio` | **Log-normal** (`σ_log` = 1,43 próprios / 1,31 mercado) | Verificado nos dados reais: o log das médias por UF tem assimetria ≈ −0,27, próximo de simétrico ⇒ a variável original é log-normal. Reproduz o CV de 146% e a razão de 318x sem gerar negativos. |
| `fator_outlet` | **Log-normal + Pareto** | Demanda de PDV é notoriamente concentrada. **Parâmetro ausente** — ver I3. |
| `ruido_temporal` | **AR(1)** com CV alvo | Ruído i.i.d. produziria série sem memória; a autocorrelação é exigida na seção 2 do prompt. |
| `sazonalidade` | **Empírica** (tabela real) | Preferível a paramétrica: os 24 meses reais por UF já estão disponíveis. |
| Demanda em unidades | **Contínua → arredondada** | Volumes altos (milhares) tornam Poisson/NegBin desnecessário; abaixo de ~30 unidades, arredondamento com preservação de soma. |

**Por que multiplicativo e não aditivo:** o Excel documenta a decomposição clássica `Yt = Tt + St + Rt` (aditiva). Mas a sazonalidade real está expressa como **índice** (`demanda / média = 1,0234`), que é multiplicativo por definição, e a variabilidade está expressa em **CV** (relativo), também multiplicativo. Um modelo aditivo com esses parâmetros produziria resíduos de tamanho constante em SP e em RR — o que é irreal, dado o fator de 318x entre eles. **Recomendação: usar o modelo multiplicativo, que é equivalente ao aditivo aplicado sobre `log(Y)`,** e registrar essa justificativa no TCC.

### G.4. Reconciliação hierárquica

Para garantir as identidades exigidas na seção 5 do prompt, o gerador deve funcionar **top-down com rateio proporcional**, nunca bottom-up somando peças independentes:

```
nacional(produto, mês) → rateio por UF → rateio por outlet → rateio por semana
```
Assim `soma(outlets) = setor`, `soma(semanas) = mês` e `soma(produtos) = mercado` valem **por construção**, com erro apenas de arredondamento (corrigido pelo maior resto).

---

## H. Parâmetros extraídos do Excel (com validação cruzada)

| Parâmetro | Valor | Onde usar | Validação |
|---|---|---|---|
| Produtos | 30 (10 próprios / 20 concorrentes) | `dim_produto` | ✅ já populado |
| Grupos de próprios | 2 (5 produtos cada) | `dim_produto.grupo_produto` | ✅ |
| Mercado | 1 — ISGLT2 | `dim_mercado` | ✅ |
| Territórios / UFs / representantes | 27 (1:1:1) | `dim_setor` | ✅ |
| Outlets | 100 por UF = 2.700, **100% BRICK** | `dim_outlet` | ✅ |
| Demanda mercado (mês × território) | avg 88.740 · dp 129.934 · min 1.695 · max 854.084 | calibração nacional | ✅ bate com 2.462.863/27 = 91.217 |
| Demanda próprios (mês × território) | avg 14.769 · dp 21.387 · min 250 · max 131.778 | calibração próprios | ✅ bate com 408.351/27 = 15.124 |
| Objetivo (mês × território) | avg 15.397 · dp 22.604 | cota | ⚠️ implica +4,25% vs +3% do prompt |
| Market share | avg 0,1766 · dp 0,0885 · min 0,0241 · max 0,5033 | validação | ✅ bate com share 2025 = 0,1658 |
| CAGR mercado | **+35,14%** a.a. | tendência mercado | ✅ recalculado: 0,3514 exato |
| CAGR próprios | **−9,09%** a.a. | tendência próprios | ⚠️ derivado por mim, não está explícito no Excel |
| CV temporal mercado | 26,02% | ruído | ✅ bate com "cv=26,03%" da aba 3 |
| CV temporal próprios | 9,69% | ruído | ✅ derivado da aba Índice Sazonal |
| σ_log entre territórios | 1,43 próprios / 1,31 mercado | escala territorial | ✅ derivado |
| Índice sazonal | 24 meses × 27 UFs × 2 visões (próprio/mercado) | sazonalidade | ✅ 648 linhas por bloco, média = 1,0000 |
| Rateio da cota semanal | `objetivo_mensal / nr_dias de cada semana` | cota semanal | ✅ |
| Thresholds de cobertura | ≤65% crítico · ≤75% atenção · <80% próximo · ≥80% bateu plano | `dim_classifica_kpis` | ✅ |
| Nulos | 0% | qualidade | ✅ não injetar nulos |
| Latência | 5 dias | `data_atualizacao_base` | ✅ |
| Cadência | semanal | pipeline | ✅ |

**Granularidade dos parâmetros da aba 2 (descoberta importante):** os valores são por **território × mês agregando todos os produtos**, não por produto individual. Verificado pelas duas divisões acima. Isso significa que o Excel **não** especifica como repartir a demanda entre os 10 próprios e os 20 concorrentes — ver I1.

---

## I. Parâmetros ainda ausentes

| # | Parâmetro ausente | Impacto | Encaminhamento proposto |
|---|---|---|---|
| I1 | **Distribuição da demanda entre produtos** (o Excel dá o agregado, não o split entre os 10 próprios / 20 concorrentes) | Alto — define a cara do dataset | Log-normal com concentração tipo Pareto; **precisa de decisão** (K1) |
| I2 | **Sazonalidade por produto individual** (o Excel dá por mercado e por território, não por SKU) | Médio — a seção 6 do prompt exige que produtos tenham sazonalidades diferentes | Perturbar o índice do mercado com ruído por produto |
| I3 | **Concentração da demanda entre outlets** dentro da UF | Alto — 100 outlets por UF, sem regra de repartição | Assumir Pareto (top 20% ≈ 60-70% do volume); **precisa de validação** (K2) |
| I4 | **Dados anteriores a jul/2024** (o Excel cobre 24 meses; o período pedido são 44) | Alto — 18 meses sem referência | Retropolação pela tendência invertida; ver K3 |
| I5 | **Aba Correlações vazia** | Baixo — **as variáveis não existem mais no schema** (`investimento_promocional`, `estoque`) | Considerar obsoleta; documentar no TCC |
| I6 | `dim_classifica_kpis` e `dim_regras_diagnostico` | Bloqueia a Etapa 7 | Arquitetura lê do banco, sem hard-code (já previsto na seção 18) |
| I7 | Faixas de CV (baixo/médio/alto) | Bloqueia parte da classificação | Excel diz "a determinar depois da base criada" — definir por quantis da própria base |
| I8 | % de outliers | Baixo | Excel em branco; nulos = 0. Assumir 0 e documentar |
| I9 | **Rateio da cota (setor) para o grão de alocação (outlet)** | Alto — bloqueia a Etapa 5 | Ver K6 |

---

## J. Riscos e inconsistências identificadas

### J1. 🔴 CRÍTICO — Volumetria 25x acima do Supabase free tier

| Tabela | Linhas | Tamanho estimado |
|---|---|---|
| `fato_analise_semanal` | 18,5 M | 2,76 GB |
| `fato_analise_semanal_final` | 18,5 M | 2,63 GB |
| `fato_desempenho_semanal` | 18,5 M | 2,35 GB |
| `fato_demanda_aberta_semanal` | 18,5 M | 1,93 GB |
| `fato_demanda_alocada_semanal` | 18,5 M | 1,93 GB |
| `fato_analise_kpi` | 3,6 M | 0,69 GB |
| `fato_demanda_aberta_mensal` | 3,6 M | 0,35 GB |
| **Total** | **~100 M** | **12,65 GB** (≈19 GB com índices) |

Free tier = 0,5 GB. **Não cabe, e não é questão de otimizar** — é ordem de grandeza. Opções em K7.

### J2. 🔴 Ausência de constraints UNIQUE nas tabelas fato
A seção 26 exige idempotência, mas nenhuma tabela fato tem chave natural declarada. Sem isso, `ON CONFLICT DO UPDATE` é impossível e reprocessar duplica dados silenciosamente. **Correção necessária no DDL** (chaves propostas na seção C).

### J3. 🔴 Nenhum índice no schema
Zero `create index`. Os JOINs de consolidação e as janelas de 6/12 meses farão *sequential scan* em tabelas de dezenas de milhões de linhas.

### J4. 🟠 Cota (setor) × desempenho (alocação): grão incompatível
`fato_cota_mensal` é setor × produto (27 × 30). `fato_desempenho_semanal` é alocação × produto (2.700 × 30). Sem regra de rateio, `cobertura_cota` fica indefinida — se a cota do setor for repetida em cada um dos 100 outlets, a cobertura individual dará ~1% e o KPI perde sentido.

### J5. 🟠 Tendência única destruiria o market share
Já detalhado em G.2. Usar +35,14% para tudo faria o share ficar constante, quando o comportamento real é de erosão (25,9% → 11,7%). O motor de diagnóstico perderia o sinal mais importante.

### J6. 🟠 CV de 146% aplicado como volatilidade temporal
Já detalhado em G.1. Confundir dispersão transversal com temporal geraria séries sem nenhuma estabilidade.

### J7. 🟡 Semanas degeneradas de 1–2 dias
A regra do calendário produz **25 semanas de 1 ou 2 dias** no período (ex.: 01/04/2023 é sábado ⇒ semana 1 tem 1 dia; 30/04/2023 é domingo ⇒ semana 6 tem 1 dia). O período tem **229 semanas** (33 meses com 5, 10 meses com 6, e fev/2026 com exatamente 4). Não é erro — é consequência da regra — mas afeta a projeção de cobertura, que fica instável na 1ª semana. Confirmar em K8.

### J8. 🟡 `fato_analise_kpi` calcula CAGR e CV no grão de outlet
CAGR anual e CV com base em série de outlet individual (valores baixos, muitos zeros) é estatisticamente frágil. Recomendo calcular no grão de **produto × UF** e propagar para o outlet, documentando a decisão.

### J9. 🟡 Ambiguidade em `periodo_demanda` nas tabelas mensais
Para as semanais o prompt define claramente (data de encerramento da semana). Para `fato_cota_mensal`, `fato_demanda_aberta_mensal` e `fato_analise_kpi` não está definido se é o primeiro ou o último dia do mês. Padronizar (sugiro **primeiro dia**, que é a convenção usual de dimensão temporal e já era a do schema antigo).

### J10. 🟡 `noticia` desconectada do modelo comercial
A tabela não tem FK para setor/UF nem para período de demanda. O Agente 4 (correlação notícia × desvio de vendas) vai precisar cruzar por data e produto/mercado apenas — o que é viável, mas vale registrar a limitação.

### J11. 🟢 `tipo_outlet` 100% BRICK
O `POPULA BANCO.txt` fixa `'BRICK'` para todos (você alterou o sorteio 70/30 original), e o Excel confirma "100% Brick". Consistente — só registrando que a coluna fica sem variabilidade e não serve como feature para o XGBoost.

---

## K. Perguntas que precisam de resposta antes da implementação

**Bloqueantes (impedem começar):**

- **K7 — Volumetria.** Qual caminho seguir?
  (a) reduzir escopo (ex.: 10 produtos × 30 outlets/UF × 24 meses ⇒ ~2 GB);
  (b) gerar tudo local em Parquet/DuckDB e subir ao Supabase só os agregados que o painel consome;
  (c) plano pago do Supabase;
  (d) Postgres local/Docker para o volume completo.
  *Minha recomendação: (b)* — preserva o rigor estatístico exigido, cabe no free tier e o painel continua lendo do Supabase.

- **K6 — Rateio da cota.** A cota de setor × produto deve ser distribuída entre os 100 outlets do setor? Se sim, por qual critério — proporcional à demanda histórica do outlet (recomendado) ou igualitário? Ou o desempenho deveria ser medido no grão de **setor**, e não de alocação?

- **K1 — Split entre produtos.** Como repartir a demanda entre os 10 próprios e os 20 concorrentes? Há concentração conhecida (ex.: 2 produtos = 60% do volume) ou reparto aproximadamente uniforme?

**Importantes (podem começar com premissa documentada):**

- **K2 — Concentração entre outlets.** Faz sentido assumir Pareto (top 20% dos outlets ≈ 65% do volume) dentro de cada UF?
- **K3 — Retropolação 2023-01 a 2024-06.** Os 18 meses sem referência devem ser gerados projetando a tendência para trás (mercado menor em 2023, share dos próprios maior)? Isso é coerente com a erosão observada, mas é uma extrapolação — confirma?
- **K4 — Crescimento da cota.** +3% (prompt) ou +4,25% (implícito no Excel)? Ambos ficam parametrizáveis; a pergunta é qual vira o default.
- **K5 — Bootstrap da cota.** Os 6 primeiros meses não têm histórico de 6 meses. Usar janela expansível (o que houver) ou começar as cotas só a partir de jul/2023?
- **K8 — Semanas de 1 dia.** Confirma que semanas de 1–2 dias são aceitáveis, ou a primeira semana deve ser fundida com a seguinte quando tiver menos de N dias?

**De confirmação:**

- **K9 — Alterações no DDL.** Posso propor um script `alter_schema.sql` adicionando as constraints UNIQUE (J2) e os índices (J3)? A seção 30 proíbe modificar a estrutura *silenciosamente* — a proposta seria explícita e revisável.
- **K10 — Sazonalidade por produto.** Confirma perturbar o índice do mercado por produto (I2), já que o Excel só tem o agregado?

---

## Próximo passo

Respondidas **K7, K6 e K1** (as três bloqueantes), implemento na sequência:

1. `alter_schema.sql` (constraints + índices, se aprovado em K9)
2. Módulo de calendário comercial + `dim_calendario`
3. Gerador hierárquico com os parâmetros calibrados da seção H
4. Pipeline das 7 etapas, com transação e logging por etapa
5. Módulo de validação + `synthetic_data_fidelity_score`
6. Testes automatizados (seção 24)
7. `.env.example`, `requirements.txt` e documentação final em Markdown

As demais perguntas (K2–K5, K8, K10) posso assumir com o default recomendado e deixar documentado como premissa, se você preferir não travar o cronograma.
