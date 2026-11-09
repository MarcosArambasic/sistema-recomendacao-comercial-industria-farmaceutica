# Pipeline de dados sintéticos — documentação

Sistema de Recomendações Comerciais para a Indústria Farmacêutica — TCC2
Gerado em 25/08/2026 · `synthetic_data_fidelity_score` atual: **98,28 / 100**

---

## 1. O que este pipeline faz

Gera a base histórica sintética das duas tabelas mensais — `fato_demanda_aberta_mensal` e `fato_cota_mensal` — cobrindo **44 meses (jan/2023 a ago/2026)**, calibrada em parâmetros **agregados e reais** da empresa. As tabelas semanais não são geradas aqui: elas são populadas pelo pipeline semanal, alimentado por esta base.

| Saída                         | Grão                             | Linhas    |
| ------------------------------ | --------------------------------- | --------- |
| `fato_demanda_aberta_mensal` | mês × produto × outlet         | 1.431.012 |
| `fato_cota_mensal`           | mês × setor × produto próprio | 11.610    |

Tempo total de execução: ~41 s (4 s de geração + validação, 37 s de exportação CSV).

---

## 2. Arquivos criados

```
sistema-recomendacao-comercial-industria-farmaceutica/
├── .env.example                                  ← nomes das variáveis (sem valores)
├── requirements.txt                              ← + Faker, psycopg2-binary, pytest, SQLAlchemy
│
├── src/sintetico/
│   ├── __init__.py
│   ├── config.py                                 ← TODOS os parâmetros centralizados
│   ├── calendario.py                             ← semanas comerciais (domingo→sábado, corte no mês)
│   ├── calibracao.py                             ← extrai o Excel → calibracao.json
│   ├── gerador.py                                ← modelo hierárquico da demanda
│   ├── cota.py                                   ← cálculo da cota mensal parametrizável
│   ├── validacao.py                              ← 6 dimensões + fidelity score
│   ├── exportacao.py                             ← CSV local (SAÍDA PADRÃO) + manifesto
│   ├── carga.py                                  ← COPY/transação no Supabase (OPCIONAL)
│   └── log_conf.py                               ← logging estruturado (console + JSONL)
│
├── scripts/
│   └── gerar_base_inicial.py                     ← orquestrador da carga inicial
│
├── tests/
│   └── test_pipeline_sintetico.py                ← 41 testes automatizados
│
├── supabase/scripts/
│   ├── criacao_banco_v2.sql                      ← DDL atual (cópia do enviado)
│   ├── popular_banco.sql                         ← dimensões + fato_alocacao
│   └── alter_schema_constraints.sql              ← PROPOSTA: UNIQUE de grão + índices
│
├── data/
│   ├── calibracao/calibracao.json                ← parâmetros congelados (versionar!)
│   └── sintetico/                                ← saídas (fora do git)
│       ├── fato_demanda_aberta_mensal*.csv
│       ├── fato_cota_mensal*.csv
│       ├── manifesto_exportacao.json             ← linhas, tamanho e SHA-256
│       └── INSTRUCOES_IMPORTACAO.md              ← comandos \copy prontos
│
├── logs/pipeline_AAAAMMDD.jsonl                  ← log estruturado por etapa
│
└── docs/Arquitetura/
    ├── PIPELINE_DADOS_SINTETICOS.md              ← este documento
    └── RELATORIO_VALIDACAO.md                    ← gerado a cada execução
```

---

## 3. Modelo estatístico

### 3.1. Estrutura multiplicativa hierárquica

```
demanda(produto, outlet, mês) =
      nível_nacional(tipo_produto, mês)     ← âncora 2025 + tendência própria
    / n_produtos(tipo)                      ← partição igual
    × peso_território(uf, tipo)             ← escala REAL por UF
    × sazonalidade(uf, tipo, mês)           ← índice REAL por UF
    × perturbação_sazonal(produto, mês)     ← diferencia SKUs entre si
    × ruído_AR(1)(produto, uf, mês)         ← choque comum, CV residual
    × peso_outlet(outlet | uf)              ← Pareto (top 20% ≈ 65%)
    × ruído_idiossincrático(par, mês)       ← variação entre outlets
```

**Por que multiplicativo e não aditivo.** O Excel documenta a decomposição clássica `Yt = Tt + St + Rt`. Mas a sazonalidade real está expressa como **índice** (`demanda/média = 1,0234`) e a variabilidade como **CV** — ambos relativos, portanto multiplicativos. Um modelo aditivo produziria resíduos de mesmo tamanho absoluto em SP e em RR, que diferem por um fator de 318×. O modelo multiplicativo é equivalente ao aditivo aplicado sobre `log(Y)`.

### 3.2. Distribuições escolhidas

| Componente             | Distribuição        | Justificativa                                                                                                                          |
| ---------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Escala por território | Log-normal            | O log das médias reais por UF tem assimetria ≈ −0,27 (quase simétrico) ⇒ a variável original é log-normal. Não gera negativos. |
| Peso por outlet        | Log-normal σ=0,70    | Calibrado para top 20% ≈ 65% **dentro de cada UF**.                                                                             |
| Ruído temporal        | AR(1), φ=0,57        | O resíduo real tem autocorrelação lag-1 de +0,54 a +0,59. Ruído i.i.d. daria série sem memória.                                  |
| Sazonalidade           | Empírica             | 24 meses reais × 27 UFs já disponíveis — melhor que qualquer forma paramétrica.                                                   |
| Demanda                | Contínua arredondada | Volumes na casa de milhares tornam Poisson/NegBin desnecessário.                                                                      |

### 3.3. Três descobertas que mudaram o modelo

**(a) O CV de 146% é transversal, não temporal.** A aba "Distribuições Numéricas" traz média 88.740 e desvio 129.934 (CV 146%). Injetar isso como volatilidade produziria um território saltando de 300 para 100.000 de um mês para o outro. Decompondo os dados reais:

| Fonte de variação              | CV real                                  |
| -------------------------------- | ---------------------------------------- |
| Entre territórios (transversal) | 146,4% próprios / 142,0% mercado        |
| Dentro do território no tempo   | **9,7% próprios / 26,0% mercado** |

O CV de 26,03% anotado no Excel bate exatamente com o CV temporal do mercado (26,02%), confirmando a leitura.

**(b) O ruído a injetar é menor que o CV observado.** O CV bruto já contém tendência e sazonalidade. Decompondo:

|           | CV bruto | sem tendência | **resíduo (injetado)** |
| --------- | -------- | -------------- | ----------------------------- |
| Próprios | 9,69%    | 7,42%          | **4,40%**               |
| Mercado   | 26,02%   | 10,64%         | **7,26%**               |

**(c) Produtos próprios caem enquanto o mercado cresce.** Aplicar +35,14% a tudo apagaria a erosão de share — o fenômeno central do negócio:

| Ano  | Mercado   | Próprios | Share  |
| ---- | --------- | --------- | ------ |
| 2024 | 1.648.252 | 426.124   | 25,85% |
| 2025 | 2.462.863 | 408.351   | 16,58% |
| 2026 | 3.009.953 | 352.177   | 11,70% |

O gerador aplica tendências **separadas** a próprios e concorrentes; o mercado emerge da soma, então o CAGR de +35,14% é reproduzido por construção.

### 3.4. Duas armadilhas de calibração corrigidas

**Convenção do CAGR.** As "médias anuais" do Excel usam janelas parciais (2024 = jul–dez, 2026 = jan–jun). Os pontos médios distam 18 meses, mas a fórmula divide por 2 anos — o que **subestima** a taxa real. Relação: `(1 + cagr_excel) = (1 + g_real)^0,75`.

**Índice sazonal contaminado.** A coluna `ÍNDICE SAZONAL` do Excel foi calculada como `demanda / média_do_ano`, e com anos parciais + tendência forte ela absorveu parte do crescimento. Usá-la direto cancelava a tendência (mercado saía a 48% quando o input era 54%). A correção é remover a tendência log-linear **antes** de estimar a sazonalidade.

Como o perfil sazonal dos concorrentes é assimétrico entre semestres (H2/H1 = 1,102) e as duas janelas do Excel caem em semestres diferentes, a taxa é resolvida por **bisseção numérica** sobre o sinal determinístico, não por álgebra.

---

## 4. Fórmulas implementadas

### Market share

```
mrkt_share = demanda_produto / demanda_mercado          (0 ≤ ms ≤ 1)
```

Resulta da demanda gerada, nunca é sorteado.

### CAGR (convenção do Excel — usada na validação)

```
CAGR = (média_mensal_ano_final / média_mensal_ano_inicial)^(1/n_anos) − 1
```

### Coeficiente de variação

```
CV = desvio_padrão / média
```

### Índice sazonal

```
índice(m) = demanda_mensal(m) / média_mensal_do_ano
```

Normalizado para que a média dos 12 meses seja exatamente 1,0.

### Cota mensal (`calcular_cota_mensal`, tudo parametrizável)

```
cota(M) = média(demanda[M−6 … M−1])
        × fator_extrapolação_tendência
        × (1 + crescimento)              ← default +4,25%
        × ajuste_sazonal(M)
```

O `fator_extrapolação_tendência` existe porque a média da janela fica ~3,5 meses atrás do mês alvo. Sem ele, numa série que cresce 79% a.a. a cota fica defasada e a razão cota/demanda cai para 0,94. Com ele, a razão fica em **1,042** — exatamente o 1,0425 observado nos dados reais (15.397 / 14.769).

**Cota só para produtos próprios.** Não se define meta de venda para produto de concorrente, e os dados confirmam: o objetivo médio do Excel é exatamente a demanda dos próprios × 1,0425, no mesmo grão.

**Bootstrap.** Os primeiros meses não têm 6 meses de histórico; usa-se janela expansível, sinalizada na coluna `bootstrap` (1.350 linhas, todas antes de ago/2023).

### Cobertura de cota (para o pipeline semanal)

```
cobertura_cota           = realizado_acumulado / cota_mensal
cobertura_cota_projetada = (acumulado / dias_corridos × dias_do_mês) / cota_mensal
```

Thresholds do Excel: ≤65% crítico · ≤75% atenção · <80% próximo · ≥80% bateu plano.

---

## 5. Calendário comercial

Implementado em `calendario.py` conforme a regra do grupo:

- semana normal: **domingo → sábado**;
- a primeira semana começa no **dia 1** e termina no primeiro sábado ≥ dia 1;
- a última termina exatamente no **último dia do mês**;
- `periodo_demanda` da semana = **data de encerramento**.

No período de 44 meses são **229 semanas comerciais** (33 meses com 5, 10 meses com 6, e fev/2026 com exatamente 4). A regra produz **25 semanas de 1–2 dias** — ex.: 01/04/2023 cai num sábado, então a semana 1 tem 1 dia. É consequência esperada da regra, não um bug, mas convém ter em mente porque a *cobertura projetada* fica instável na primeira semana desses meses.

---

## 6. Como usar

### Instalação

```bash
pip install -r requirements.txt
cp .env.example .env      # preencha DATABASE_URL com a senha real
```

### Geração — a saída padrão é CSV local

O script **não toca no banco** por padrão. Ele gera, valida e escreve os CSVs em `data/sintetico/`.

```bash
# padrão: gera, valida e exporta CSV local
python scripts/gerar_base_inicial.py

# um CSV por ano — arquivos de ~13 MB, bem mais fáceis de subir pelo painel
python scripts/gerar_base_inicial.py --particionar

# CSVs compactados em .gz
python scripts/gerar_base_inicial.py --particionar --comprimir

# pasta de saída alternativa
python scripts/gerar_base_inicial.py --dir-saida /caminho/que/eu/quiser

# outra semente / recalibrar a partir do Excel
python scripts/gerar_base_inicial.py --seed 7 --recalibrar
```

Cada execução também gera:

- **`manifesto_exportacao.json`** — contagem de linhas, tamanho e SHA-256 de cada arquivo, mais os parâmetros da geração. Serve para conferir depois se o que está no banco é exatamente o que saiu daqui.
- **`INSTRUCOES_IMPORTACAO.md`** — os comandos `\copy` já prontos, com os nomes de arquivo e a ordem de colunas corretos.

### Formato dos CSVs

Pensado para o `\copy` funcionar de primeira:

- colunas na **ordem exata** da tabela de destino;
- datas como `AAAA-MM-DD`, sem hora;
- `demanda_unidades` como inteiro (`16`, não `16.0`) — arquivo menor e escrita mais rápida;
- sem índice do pandas, com header.

### Importar no banco depois

```bash
# 1. criar as tabelas
psql "$DATABASE_URL" -f supabase/scripts/criacao_banco_v2.sql
# 2. popular dimensões + alocação (obrigatório ANTES: as FKs dependem disso)
psql "$DATABASE_URL" -f supabase/scripts/popular_banco.sql
# 3. gerar os CSVs
python scripts/gerar_base_inicial.py --particionar
# 4. importar (comandos exatos em data/sintetico/INSTRUCOES_IMPORTACAO.md)
psql "$DATABASE_URL"
#    \copy public.fato_demanda_aberta_mensal (...) FROM 'arquivo.csv' WITH (FORMAT csv, HEADER true);
# 5. constraints e índices (DEPOIS da carga — é bem mais rápido)
psql "$DATABASE_URL" -f supabase/scripts/alter_schema_constraints.sql
```

### Carga direta (opcional)

Se preferir pular o CSV e escrever direto no banco:

```bash
python scripts/gerar_base_inicial.py --carregar
```

Só carrega se o fidelity score ≥ 80 (ajustável com `--score-minimo`). Usa COPY dentro de transação, com rollback em caso de erro, e apaga o intervalo de períodos antes de reinserir — então rodar duas vezes produz o mesmo estado final, sem duplicar.

### Reimportação a partir do CSV

Diferente da carga direta, o `\copy` **sempre insere** — rodar duas vezes duplica. Antes de reimportar, limpe o intervalo (o `INSTRUCOES_IMPORTACAO.md` já traz o `delete` pronto).

### Testes

```bash
pytest tests/ -v          # 41 testes
```

### Módulos isolados

```bash
python -m src.sintetico.calibracao   # recalibra e mostra os parâmetros
python -m src.sintetico.calendario   # inspeciona o calendário
python -m src.sintetico.gerador      # só a geração
python -m src.sintetico.cota         # só a cota
python -m src.sintetico.validacao    # só a validação
python -m src.sintetico.exportacao   # só a exportação em CSV
```

---

## 7. Resultado da validação

| Dimensão                               | Score           |
| --------------------------------------- | --------------- |
| Integridade                             | 100,00          |
| Reconciliação                         | 100,00          |
| Distribuição                          | 97,33           |
| Tendência                              | 97,11           |
| Sazonalidade                            | 96,54           |
| Correlação                            | 95,33           |
| **synthetic_data_fidelity_score** | **98,28** |

Aderência aos alvos reais:

| Métrica                   | Gerado   | Alvo    | Erro       |
| -------------------------- | -------- | ------- | ---------- |
| CAGR mercado (conv. Excel) | +35,34%  | +35,14% | 0,21 p.p.  |
| CAGR concorrentes          | +47,24%  | +47,47% | 0,22 p.p.  |
| CAGR próprios             | −10,10% | −9,09% | 1,01 p.p.  |
| CV temporal próprios      | 9,93%    | 9,69%   | 2,5% rel.  |
| CV temporal mercado        | 26,21%   | 26,02%  | 0,7% rel.  |
| Market share médio        | 17,19%   | 17,66%  | 2,7% rel.  |
| Market share (desvio)      | 0,0878   | 0,0885  | 0,8% rel.  |
| Pareto top 20% intra-UF    | 65,8%    | 65,0%   | 1,2% rel.  |
| Razão cota/demanda        | 1,042    | 1,0425  | 0,05% rel. |

**Cuidado metodológico:** todos os alvos foram medidos na janela do Excel (jul/2024–jun/2026) e no grão território × mês. A validação recorta exatamente a mesma janela — comparar contra os 44 meses daria erro espúrio, porque CV e CAGR dependem da janela.

---

## 8. Premissas assumidas (registrar no TCC)

| # | Premissa                                                             | Origem                                |
| - | -------------------------------------------------------------------- | ------------------------------------- |
| 1 | Partição igual entre os 10 próprios e entre os 20 concorrentes    | Decisão do grupo                     |
| 2 | Concentração Pareto: top 20% dos outlets ≈ 65% do volume intra-UF | Decisão do grupo                     |
| 3 | Esparsidade ~40%: nem todo produto existe em todo outlet             | Decisão do grupo                     |
| 4 | Jan/2023–jun/2024 retropolado projetando a tendência para trás    | Decisão do grupo                     |
| 5 | Crescimento da cota +4,25%                                           | Excel (objetivo/demanda)              |
| 6 | Cota apenas para produtos próprios                                  | Excel + regra de negócio             |
| 7 | 0% de nulos e 0% de outliers                                         | Excel (nulos = 0; outliers em branco) |
| 8 | `periodo_demanda` mensal = primeiro dia do mês                    | Convenção adotada                   |
| 9 | Sazonalidade por SKU = índice agregado ± 5%                        | Excel não tem por SKU                |

**Sobre a premissa 4:** a retropolação produz share de ~44% em 2023, caindo para 10,6% em 2026. É coerente com a erosão observada, mas são **18 meses extrapolados além dos dados**. Vale citar como limitação no texto.

**Sobre a premissa 1:** com todos os produtos idênticos em escala, a variação entre SKUs vem só da sazonalidade perturbada e do ruído. Se o XGBoost precisar de mais sinal por produto, basta trocar a partição igual por pesos configuráveis em `config.py`.

---

## 9. Pontos em aberto

1. **`dim_regras_diagnostico` e `dim_classifica_kpis`** continuam pendentes (população manual). A arquitetura já lê as regras do banco — nenhum diagnóstico está hard-coded no Python.
2. **Faixas de CV** (baixo/médio/alto) — o Excel diz "a determinar depois da base criada". Agora a base existe; sugiro definir por quantis dela.
3. **Rateio da cota para o grão de alocação** — pendente, mas só bloqueia o pipeline **semanal** (`fato_desempenho_semanal`), não esta carga inicial.
4. **Constraints e índices** — `alter_schema_constraints.sql` está pronto mas não foi aplicado; precisa da decisão de vocês.
5. **Rotacionar as credenciais do Supabase** que foram expostas em texto.

---

## 10. Segurança

- Nenhuma credencial no código ou em arquivo versionado. Só `.env.example` com os **nomes** das variáveis.
- `.env` adicionado ao `.gitignore`.
- A chave `sb_publishable_...` informada é *publishable/anon*, não `service_role` — com RLS ativo ela **não** consegue fazer a carga em lote. Use a `service_role` real, apenas no `.env`.
- O `calibracao.json` contém somente **agregados** (médias, índices, taxas). Nenhum registro transacional real da empresa é lido, gravado ou versionado.
