"""***************************************************************************************
SCRIPT    : config.py
OBJETIVO  : Parametros centrais do pipeline de geracao de dados sinteticos.
            TODOS os numeros que governam a geracao vivem aqui - nada de valor
            magico espalhado pelo codigo. Os valores marcados como [CALIBRADO]
            foram derivados dos dados reais agregados do Excel
            (Parametros_Dados_Sinteticos.xlsx) e estao documentados em
            docs/Arquitetura/PRIMEIRA_ENTREGA_Diagnostico_Pipeline.md.

AUTOR     : Marcos Arambasic
DATA      : AGO/2026
****************************************************************************************/
"""

from __future__ import annotations

import os
from pathlib import Path

# ============================================================================
# CAMINHOS
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent

CAMINHO_EXCEL = BASE_DIR / "docs" / "Arquitetura" / "Parametros_Dados_Sinteticos_PREENCHIDO.xlsx"
DIR_CALIBRACAO = BASE_DIR / "data" / "calibracao"
DIR_SAIDA = BASE_DIR / "data" / "sintetico"
DIR_LOGS = BASE_DIR / "logs"

ARQ_CALIBRACAO = DIR_CALIBRACAO / "calibracao.json"

# ============================================================================
# REPRODUTIBILIDADE (secao 21 do prompt mestre)
# ============================================================================
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# ============================================================================
# PERIODO (secao 3 do prompt mestre) - 44 meses
# ============================================================================
PERIODO_INICIO = "2023-01"
PERIODO_FIM = "2027-120"

# ============================================================================
# ESTRUTURA DIMENSIONAL - precisa espelhar o que foi carregado pelo
# supabase/scripts/popular_banco.sql
# ============================================================================
N_PRODUTOS_PROPRIOS = 10
N_PRODUTOS_CONCORRENTES = 20
N_PRODUTOS = N_PRODUTOS_PROPRIOS + N_PRODUTOS_CONCORRENTES
N_UFS = 27
N_SETORES = 27               # 1 por UF
N_OUTLETS_POR_UF = 100
N_OUTLETS = N_UFS * N_OUTLETS_POR_UF

# ============================================================================
# ANCORAS DE NIVEL [CALIBRADO] - media mensal NACIONAL de 2025, em unidades.
# Fonte: aba 'Indice Sazonal', coluna MEDIA_ANUAL do ano de 2025.
# ============================================================================
ANCORA_ANO = 2025
ANCORA_PROPRIOS_MES = 408_350.833333      # soma dos 10 produtos proprios
ANCORA_MERCADO_MES = 2_462_862.666667     # mercado total (proprios + concorrentes)
ANCORA_CONCORRENTES_MES = ANCORA_MERCADO_MES - ANCORA_PROPRIOS_MES

# ============================================================================
# TENDENCIA (secao 7) - crescimento anual por tipo de produto.
#
# Descoberta central do diagnostico: os produtos proprios PERDEM volume
# enquanto o mercado cresce. Usar uma unica tendencia (+35,14%) para tudo
# apagaria a erosao de market share (25,85% -> 11,70%), que e justamente o
# fenomeno que o motor de diagnostico precisa detectar.
#
# O crescimento do mercado NAO e aplicado diretamente: ele EMERGE da soma
# proprios + concorrentes, garantindo consistencia interna por construcao.
#
# ATENCAO A DUAS CONVENCOES DIFERENTES:
#   * CAGR_EXCEL_* : como o Excel calcula (medias anuais de janelas parciais -
#     2024 cobre jul-dez e 2026 cobre jan-jun - divididas por 2 anos). Os
#     pontos medios distam 18 meses, nao 24, entao essa convencao SUBESTIMA a
#     taxa real. Serve como ALVO DE VALIDACAO, medido do mesmo jeito.
#   * O input do gerador vem da regressao log-linear sobre os 24 pontos
#     mensais reais (calibracao.json -> tendencia.crescimento_log_linear),
#     que usa toda a serie e independe de janelas parciais.
#     Relacao algebrica: (1 + cagr_excel) = (1 + g_real)^0.75
# ============================================================================
USAR_TENDENCIA_CALIBRADA = True   # le de calibracao.json; False usa os fallbacks abaixo

# Qual alvo o gerador persegue:
#   "cagr_excel"  -> reproduz exatamente o CAGR documentado no Excel (+35,14%)
#                    quando medido com a convencao do Excel. RECOMENDADO,
#                    porque e o numero que vai citado no texto do TCC.
#   "log_linear"  -> reproduz a inclinacao log-linear da serie mensal real.
#                    Estatisticamente mais preciso para a janela observada,
#                    mas nao bate com o numero citado no TCC.
ALVO_TENDENCIA = "cagr_excel"

CAGR_EXCEL_MERCADO = 0.351400     # alvo de validacao (convencao do Excel)
CAGR_EXCEL_PROPRIOS = -0.090934   # alvo de validacao (convencao do Excel)

# Fallbacks usados apenas se USAR_TENDENCIA_CALIBRADA = False
CAGR_PROPRIOS = -0.121500         # -12,15% a.a. (log-linear)
CAGR_CONCORRENTES = 0.737400      # +73,74% a.a. (log-linear)

# ============================================================================
# VARIABILIDADE [CALIBRADO] (secao 8)
#
# Atencao ao que cada numero significa:
#   - CV_BRUTO_*  : CV da serie mensal observada (ja contem tendencia +
#                   sazonalidade + ruido). E o numero que aparece no Excel
#                   e o ALVO da validacao.
#   - CV_RESIDUAL_*: CV do residuo depois de remover tendencia e sazonalidade.
#                   E o que efetivamente injetamos como ruido. Injetar o CV
#                   bruto estouraria o alvo, porque tendencia e sazonalidade
#                   ja contribuem com boa parte da variacao.
# ============================================================================
CV_BRUTO_PROPRIOS = 0.0969       # alvo de validacao
CV_BRUTO_MERCADO = 0.2603        # alvo de validacao (Excel: cv = 26,03%)

CV_RESIDUAL_PROPRIOS = 0.0440    # ruido injetado
CV_RESIDUAL_CONCORRENTES = 0.0726

# Autocorrelacao lag-1 do residuo -> justifica AR(1) em vez de ruido i.i.d.
AR1_PHI = 0.57

# Ruido idiossincratico por outlet (acima do choque comum da UF).
# Diversifica no agregado, entao nao contamina o CV alvo no nivel de UF.
CV_IDIOSSINCRATICO_OUTLET = 0.18

# ============================================================================
# CONCENTRACAO ENTRE OUTLETS (decisao do grupo: Pareto)
#
# sigma calibrado para que o volume FINAL (ja com a esparsidade aplicada)
# tenha top 20% dos outlets ~ 65% do volume.
#
# Nota: sigma=1,227 seria o valor correto se a concentracao viesse so dos
# pesos. Mas a esparsidade e correlacionada com o tamanho (outlets maiores
# carregam mais SKUs), entao os dois efeitos se compoem e o resultado final
# fica bem mais concentrado. Calibrado empiricamente sobre o resultado final.
# ============================================================================
SIGMA_LOGNORMAL_OUTLET = 0.70
PARETO_TOP_PCT = 0.20
PARETO_SHARE_ALVO = 0.65

# ============================================================================
# ESPARSIDADE (decisao do grupo: ~40% dos pares produto x outlet ativos)
#
# A esparsidade e ESTRUTURAL, nao aleatoria por linha: um produto esta ou nao
# esta em um outlet, e isso NAO muda de mes para mes (senao a serie temporal
# de cada par ficaria sem sentido). Outlets maiores carregam mais SKUs.
# ============================================================================
DENSIDADE_ALVO = 0.40
# Outlets no topo carregam praticamente todo o portfolio; a cauda carrega pouco.
DENSIDADE_MIN_OUTLET = 0.10      # menor outlet: 10% dos SKUs
DENSIDADE_MAX_OUTLET = 1.00      # maior outlet: 100% dos SKUs

# ============================================================================
# SAZONALIDADE POR PRODUTO (secao 6)
# O Excel fornece sazonalidade agregada (proprios e mercado) por UF, nao por
# SKU. Perturbamos o indice agregado por produto para que produtos diferentes
# tenham comportamentos sazonais diferentes, como exige o prompt.
# ============================================================================
PERTURBACAO_SAZONAL_PRODUTO = 0.05   # +/- 5% sobre o indice do agregado

# ============================================================================
# COTA (secoes 11 e 13) - todos parametrizaveis, nada hard-coded
# ============================================================================
COTA_JANELA_MESES = 6
COTA_CRESCIMENTO = 0.0425        # +4,25% (razao objetivo/demanda observada no Excel)
COTA_USAR_AJUSTE_SAZONAL = True
COTA_JANELA_MINIMA = 1           # bootstrap: janela expansivel nos primeiros meses

# A media da janela fica ~3,5 meses atras do mes alvo. Sem projetar a
# tendencia, a cota fica defasada em series de crescimento forte e a razao
# cota/demanda cai para ~0,94, contra o 1,0425 observado nos dados da empresa.
COTA_EXTRAPOLAR_TENDENCIA = True
COTA_LIMITE_EXTRAPOLACAO = 2.0   # trava o fator entre 0,5x e 2,0x (evita explosao)

# A empresa define meta apenas para os PRODUTOS PROPRIOS - nao se estabelece
# cota de venda para produto de concorrente. Confirmado pelos dados: o
# objetivo medio do Excel (15.397) e exatamente a demanda dos proprios
# (14.769) x 1,0425, no mesmo grao territorio x mes.
COTA_APENAS_PROPRIOS = True

# ============================================================================
# REGRAS DE NEGOCIO [DO EXCEL] - thresholds de cobertura de cota
# ============================================================================
THRESHOLDS_COBERTURA = {
    "critico": 0.65,       # <= 65%
    "atencao": 0.75,       # <= 75%
    "proximo": 0.80,       # <  80%
    # >= 80% -> "bateu plano"
}

# ============================================================================
# QUALIDADE [DO EXCEL]
# ============================================================================
PCT_NULOS = 0.0                  # Excel informa 0% de nulos
PCT_OUTLIERS = 0.0               # Excel em branco -> assumido 0 e documentado
LATENCIA_DIAS = 5                # data_atualizacao_base = fim do periodo + 5 dias

# ============================================================================
# CONVENCOES DE PERIODO
# ============================================================================
# periodo_demanda nas tabelas MENSAIS = primeiro dia do mes.
# (Nas semanais, por definicao do prompt, e a data de encerramento da semana.)
PERIODO_MENSAL_PRIMEIRO_DIA = True

# ============================================================================
# CARGA / PERFORMANCE (secoes 19 e 25)
# ============================================================================
BATCH_SIZE = 50_000
USAR_COPY = True

# ============================================================================
# ALVOS DE DISTRIBUICAO [DO EXCEL, aba 2] (secao 22)
#
# Granularidade: TERRITORIO x MES, agregando os produtos do tipo.
# Descoberto por validacao cruzada: 408.351/27 = 15.124 ~ 14.769 (proprios)
# e 2.462.863/27 = 91.217 ~ 88.740 (mercado). Nao e por produto individual.
#
# Medidos sobre a janela observada (jul/2024 a jun/2026), entao a validacao
# precisa recortar a mesma janela para comparar como-com-como.
# ============================================================================
ALVOS_DISTRIBUICAO = {
    "demanda_mercado_uf_mes": {"media": 88_740, "dp": 129_934, "min": 1_695, "max": 854_084},
    "demanda_proprios_uf_mes": {"media": 14_769, "dp": 21_387, "min": 250, "max": 131_778},
    "objetivo_uf_mes": {"media": 15_397, "dp": 22_604, "min": 239, "max": 198_424},
    "market_share": {"media": 0.1766, "dp": 0.0885, "min": 0.0241, "max": 0.5033},
}

# ============================================================================
# TOLERANCIAS DE VALIDACAO (secoes 22 e 23)
# ============================================================================
TOLERANCIA_RECONCILIACAO = 1e-6   # erro relativo aceitavel nas somas
TOLERANCIA_DISTRIBUICAO = 0.15    # 15% de desvio relativo aceitavel na media/CV
TOLERANCIA_TENDENCIA = 0.05       # 5 p.p. no CAGR
TOLERANCIA_SAZONAL = 0.10         # 10% no indice sazonal
