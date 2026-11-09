# Relatorio de validacao estatistica da base sintetica

**synthetic_data_fidelity_score: 98.27 / 100**

## Score por dimensao

| Dimensao | Score |
|---|---|
| integridade | 100.00 |
| reconciliacao | 100.00 |
| distribuicao | 97.32 |
| tendencia | 97.05 |
| sazonalidade | 96.68 |
| correlacao | 94.87 |

## Parametros da geracao

| Item | Valor |
|---|---|
| n_meses | 45 |
| n_produtos | 30 |
| n_ufs | 27 |
| n_outlets | 2700 |
| n_pares_ativos | 32532 |
| densidade | 0.4016296296296296 |
| n_linhas | 1463940 |
| seed | 42 |
| periodo | ['2023-01-01', '2026-09-01'] |
| crescimento_aplicado | {'proprios': -0.11848223488430348, 'concorrentes': 0.7909979720229032} |

## Checks detalhados

| Dimensao | Metrica | Valor | Alvo | Erro | Score | Obs |
|---|---|---|---|---|---|---|
| distribuicao | demanda_mercado_uf_mes.media | 94,218 | 88,740 | 0.0617 | 87.7 |  |
| distribuicao | demanda_mercado_uf_mes.dp | 138,229 | 129,934 | 0.0638 | 87.2 |  |
| distribuicao | demanda_mercado_uf_mes.cv | 1.4671 | 1.4642 | 0.0020 | 99.6 |  |
| distribuicao | demanda_mercado_uf_mes.P25 | 20,553 | - | - | 100.0 | sem alvo no Excel - diagnostico |
| distribuicao | demanda_mercado_uf_mes.P50 | 43,756 | - | - | 100.0 | sem alvo no Excel - diagnostico |
| distribuicao | demanda_mercado_uf_mes.P75 | 108,708 | - | - | 100.0 | sem alvo no Excel - diagnostico |
| distribuicao | demanda_mercado_uf_mes.P95 | 379,358 | - | - | 100.0 | sem alvo no Excel - diagnostico |
| distribuicao | demanda_mercado_uf_mes.P99 | 750,132 | - | - | 100.0 | sem alvo no Excel - diagnostico |
| distribuicao | demanda_proprios_uf_mes.media | 15,160 | 14,769 | 0.0265 | 94.7 |  |
| distribuicao | demanda_proprios_uf_mes.dp | 22,014 | 21,387 | 0.0293 | 94.1 |  |
| distribuicao | demanda_proprios_uf_mes.cv | 1.4521 | 1.4481 | 0.0028 | 99.4 |  |
| distribuicao | demanda_proprios_uf_mes.P25 | 2,957 | - | - | 100.0 | sem alvo no Excel - diagnostico |
| distribuicao | demanda_proprios_uf_mes.P50 | 7,586 | - | - | 100.0 | sem alvo no Excel - diagnostico |
| distribuicao | demanda_proprios_uf_mes.P75 | 17,475 | - | - | 100.0 | sem alvo no Excel - diagnostico |
| distribuicao | demanda_proprios_uf_mes.P95 | 49,608 | - | - | 100.0 | sem alvo no Excel - diagnostico |
| distribuicao | demanda_proprios_uf_mes.P99 | 113,837 | - | - | 100.0 | sem alvo no Excel - diagnostico |
| distribuicao | market_share.media | 0.1717 | 0.1766 | 0.0276 | 94.5 |  |
| distribuicao | market_share.dp | 0.0879 | 0.0885 | 0.0069 | 98.6 |  |
| distribuicao | objetivo_uf_mes.media | 15,836 | 15,397 | 0.0285 | 94.3 |  |
| distribuicao | objetivo_uf_mes.dp | 23,017 | 22,604 | 0.0183 | 96.3 |  |
| sazonalidade | indice_sazonal.proprios | 0.0100 | 0.0000 | 0.0100 | 97.0 | erro relativo medio entre indice sintetico e real |
| sazonalidade | indice_sazonal.mercado | 0.0121 | 0.0000 | 0.0121 | 96.4 | erro relativo medio entre indice sintetico e real |
| tendencia | cagr_excel.proprio | -0.1010 | -0.0909 | 0.0101 | 93.9 | medido na mesma convencao e janela do Excel |
| tendencia | cagr_excel.concorrente | 0.4718 | 0.4747 | 0.0029 | 98.3 | medido na mesma convencao e janela do Excel |
| tendencia | cagr_excel.mercado | 0.3531 | 0.3514 | 0.0017 | 99.0 | medido na mesma convencao e janela do Excel |
| correlacao | erosao_share_monotona | 1.0000 | 1.0000 | 0.0000 | 100.0 | share por ano: [np.float64(0.448), np.float64(0.2862), np.float64(0.1642), np.float64(0.0968)] |
| correlacao | share.2024 | 0.2862 | 0.2585 | 0.0276 | 86.2 |  |
| correlacao | share.2025 | 0.1642 | 0.1658 | 0.0016 | 99.2 |  |
| correlacao | share.2026 | 0.0968 | 0.1170 | 0.0202 | 89.9 |  |
| correlacao | pareto_top20_intra_uf | 0.6469 | 0.6500 | 0.0048 | 99.0 | medido dentro de cada UF, nao nacionalmente |
| reconciliacao | outlets_com_setor | 1,463,940 | 1,463,940 | 0.0000 | 100.0 |  |
| reconciliacao | soma_outlet_igual_setor | 97,012,659 | 97,012,659 | 0.0000 | 100.0 |  |
| reconciliacao | soma_produto_igual_mercado | 97,012,659 | 97,012,659 | 0.0000 | 100.0 |  |
| integridade | demanda_nao_negativa | 1.0000 | 1.0000 | 0.0000 | 100.0 | 0 negativas |
| integridade | sem_nulos | 1.0000 | 1.0000 | 0.0000 | 100.0 | 0 nulos |
| integridade | sem_duplicidade_no_grao | 1.0000 | 1.0000 | 0.0000 | 100.0 | 0 duplicatas |
| integridade | fk_produto_valida | 1.0000 | 1.0000 | 0.0000 | 100.0 |  |
| integridade | fk_outlet_valida | 1.0000 | 1.0000 | 0.0000 | 100.0 |  |
| integridade | fk_uf_valida | 1.0000 | 1.0000 | 0.0000 | 100.0 |  |
| integridade | uf_coerente_com_outlet | 1.0000 | 1.0000 | 0.0000 | 100.0 |  |
| integridade | market_share_entre_0_e_1 | 1.0000 | 1.0000 | 0.0000 | 100.0 | min=0.0173 max=0.7752 |
| integridade | cota_nao_negativa | 1.0000 | 1.0000 | 0.0000 | 100.0 |  |
| integridade | cota_sem_duplicidade | 1.0000 | 1.0000 | 0.0000 | 100.0 | 0 duplicatas |
| integridade | n_meses_correto | 1.0000 | 1.0000 | 0.0000 | 100.0 | 45 meses de 45 esperados (sem lacunas) |
| integridade | n_produtos_correto | 1.0000 | 1.0000 | 0.0000 | 100.0 | 30 produtos |
| integridade | n_ufs_correto | 1.0000 | 1.0000 | 0.0000 | 100.0 | 27 UFs |

## Onde o dado sintetico se distancia dos parametros

Nenhum check abaixo de 70 pontos.