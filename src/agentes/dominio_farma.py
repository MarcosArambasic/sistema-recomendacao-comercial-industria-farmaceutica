"""
CONHECIMENTO DE DOMINIO - MERCADO FARMACEUTICO DE DIABETES (BRASIL)
====================================================================
Este modulo concentra TODO o conhecimento de dominio do sistema em um
unico lugar, separado da logica dos agentes. Isso atende ao Principio
da Responsabilidade Unica (SOLID-S): os agentes cuidam de *como*
classificar; este modulo cuida de *o que* existe no mercado.

Beneficio pratico: quando o corpus mudar (novo farmaco, novo
laboratorio, nova regra da CMED), voce edita SO este arquivo, sem
tocar em nenhum agente. Beneficio academico: o gazetteer e a taxonomia
de eventos viram um artefato descritivel na metodologia do TCC - sao
uma contribuicao propria, nao apenas uso de biblioteca pronta.

--------------------------------------------------------------------
JUSTIFICATIVA METODOLOGICA (para a fundamentacao teorica)
--------------------------------------------------------------------
A extracao de entidades aqui e feita por *gazetteer* (dicionario
controlado) + regex, e nao por um modelo de NER neural, por tres
razoes defensaveis em banca:

  1. AUSENCIA DE CORPUS ANOTADO PARA NER. Os avaliadores humanos
     rotularam impacto (Alto/Medio/Baixo), nao spans de entidade.
     Treinar NER supervisionado exigiria uma segunda rodada de
     anotacao que o cronograma nao comporta.

  2. ALTA PRECISAO EM DOMINIO FECHADO. O universo de farmacos
     antidiabeticos comercializados no Brasil e finito e conhecido
     (~40 principios ativos). Em dominios fechados, gazetteers
     alcancam precisao superior a modelos neurais generalistas
     (NADEAU; SEKINE, 2007).

  3. AUDITABILIDADE. Toda extracao e rastreavel ate um termo
     explicito do dicionario - requisito relevante em sistemas de
     apoio a decisao comercial, onde o gestor precisa saber POR QUE
     o sistema sinalizou algo.

LIMITACAO ASSUMIDA: gazetteers nao generalizam para termos fora do
dicionario (ex.: farmaco lancado apos a ultima atualizacao). Isso deve
constar como limitacao no TCC, com a mitigacao de que o dicionario e
versionado e atualizavel sem retreino de modelo.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum

# =====================================================================
# 1. NORMALIZACAO DE TEXTO
# =====================================================================


def normalizar(texto: str) -> str:
    """
    Normaliza texto para casamento robusto: minusculas, sem acento,
    espacos colapsados.

    Necessario porque o corpus vem de APIs de noticias com grafias
    inconsistentes ("Anvisa", "ANVISA", "AnVisa") e acentuacao
    irregular ("metformina" vs "Metformina").
    """
    if not isinstance(texto, str):
        return ""
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", texto.lower()).strip()


# =====================================================================
# 2. GAZETTEER - ENTIDADES DO MERCADO
# =====================================================================

# Principios ativos antidiabeticos, agrupados por classe terapeutica.
# A classe importa porque o impacto de uma noticia costuma se propagar
# para a classe inteira (ex.: alerta de seguranca em um iSGLT2 respinga
# nos demais).
# NOTA IMPORTANTE SOBRE IDIOMA
# ------------------------------
# O corpus deste projeto e coletado por API internacional e e
# majoritariamente EM INGLES (~85% dos titulos). Uma versao anterior
# deste modulo listava apenas as grafias em portugues, e o efeito foi
# severo: "metformin" nao casa com "metformina", "shortage" nao casa
# com "desabastecimento". O extrator reconhecia so 21,8% do corpus, e a
# regra de ancora de mercado do Agente 2 teria descartado 484 noticias
# que os avaliadores humanos classificaram como ALTO impacto.
#
# Por isso cada termo aparece nas DUAS grafias. A Denominacao Comum
# Internacional muda so o sufixo entre os idiomas (-ina/-in,
# -ida/-ide, -ona/-one), o que torna a duplicacao curta e auditavel.
# Escrever as duas formas explicitamente e preferivel a gerar variantes
# por regra: um TCC precisa que o dicionario seja inspecionavel.
PRINCIPIOS_ATIVOS: dict[str, list[str]] = {
    "biguanida": ["metformina", "metformin", "cloridrato de metformina"],
    "sulfonilureia": [
        "glibenclamida", "glibenclamide", "glyburide",
        "gliclazida", "gliclazide",
        "glimepirida", "glimepiride",
        "clorpropamida", "chlorpropamide",
    ],
    "isglt2": [
        "dapagliflozina", "dapagliflozin",
        "empagliflozina", "empagliflozin",
        "canagliflozina", "canagliflozin",
        "ertugliflozina", "ertugliflozin",
        "sglt2", "sglt-2",
    ],
    "idpp4": [
        "sitagliptina", "sitagliptin",
        "vildagliptina", "vildagliptin",
        "linagliptina", "linagliptin",
        "saxagliptina", "saxagliptin",
        "alogliptina", "alogliptin",
        "dpp-4", "dpp4",
    ],
    "agonista_glp1": [
        "liraglutida", "liraglutide",
        "semaglutida", "semaglutide",
        "dulaglutida", "dulaglutide",
        "exenatida", "exenatide",
        "lixisenatida", "lixisenatide",
        "tirzepatida", "tirzepatide",
        "retatrutida", "retatrutide",
        "glp-1", "glp1",
    ],
    "insulina": [
        "insulina", "insulin",
        "glargina", "glargine",
        "degludeca", "degludec",
        "detemir", "lispro",
        "asparte", "aspart",
        "glulisina", "glulisine",
        "insulina nph", "nph insulin",
    ],
    "tiazolidinediona": [
        "pioglitazona", "pioglitazone",
        "rosiglitazona", "rosiglitazone",
    ],
    "inibidor_alfaglicosidase": ["acarbose"],
}

# Grafia em ingles -> forma canonica em portugues.
#
# POR QUE ISSO E NECESSARIO: a base de vendas da IQVIA usa a nomenclatura
# em portugues. Se o extrator devolvesse "metformin" para uma noticia em
# ingles e "metformina" para uma em portugues, o Agente 4 trataria as
# duas como produtos diferentes e a juncao com as vendas falharia
# silenciosamente na metade dos casos. Canonizar aqui garante que o
# idioma da noticia nao vaze para a chave de juncao.
SINONIMOS_ATIVO: dict[str, str] = {
    "metformin": "metformina",
    "glibenclamide": "glibenclamida", "glyburide": "glibenclamida",
    "gliclazide": "gliclazida", "glimepiride": "glimepirida",
    "chlorpropamide": "clorpropamida",
    "dapagliflozin": "dapagliflozina", "empagliflozin": "empagliflozina",
    "canagliflozin": "canagliflozina", "ertugliflozin": "ertugliflozina",
    "sitagliptin": "sitagliptina", "vildagliptin": "vildagliptina",
    "linagliptin": "linagliptina", "saxagliptin": "saxagliptina",
    "alogliptin": "alogliptina",
    "liraglutide": "liraglutida", "semaglutide": "semaglutida",
    "dulaglutide": "dulaglutida", "exenatide": "exenatida",
    "lixisenatide": "lixisenatida", "tirzepatide": "tirzepatida",
    "retatrutide": "retatrutida",
    "insulin": "insulina", "glargine": "glargina", "degludec": "degludeca",
    "aspart": "asparte", "glulisine": "glulisina", "nph insulin": "insulina nph",
    "pioglitazone": "pioglitazona", "rosiglitazone": "rosiglitazona",
    # classes terapeuticas citadas como sigla
    "sglt-2": "sglt2", "dpp4": "dpp-4", "glp1": "glp-1",
}

# Marcas comerciais -> principio ativo. Noticia de varejo cita marca
# ("Ozempic"), nao molecula ("semaglutida"); o mapeamento permite
# resolver as duas grafias para a mesma entidade canonica.
MARCAS_COMERCIAIS: dict[str, str] = {
    "glifage": "metformina", "glifage xr": "metformina", "dimefor": "metformina",
    "forxiga": "dapagliflozina", "xigduo": "dapagliflozina",
    "jardiance": "empagliflozina", "glyxambi": "empagliflozina", "synjardy": "empagliflozina",
    "januvia": "sitagliptina", "janumet": "sitagliptina", "xelevia": "sitagliptina",
    "galvus": "vildagliptina", "galvus met": "vildagliptina",
    "trayenta": "linagliptina", "jentadueto": "linagliptina",
    "onglyza": "saxagliptina",
    "victoza": "liraglutida", "saxenda": "liraglutida",
    "ozempic": "semaglutida", "wegovy": "semaglutida", "rybelsus": "semaglutida",
    "trulicity": "dulaglutida",
    "mounjaro": "tirzepatida", "zepbound": "tirzepatida",
    "lantus": "glargina", "toujeo": "glargina", "basaglar": "glargina",
    "tresiba": "degludeca", "xultophy": "degludeca",
    "levemir": "detemir",
    "humalog": "lispro", "novorapid": "asparte", "fiasp": "asparte", "apidra": "glulisina",
    "diamicron": "gliclazida", "amaryl": "glimepirida", "daonil": "glibenclamida",
    "actos": "pioglitazona", "glucobay": "acarbose",
}

# Laboratorios: os detentores de marca definem quem ganha ou perde com
# o evento. Separados entre multinacionais (portfolio de referencia) e
# nacionais (forte em genericos/similares) porque a polaridade de um
# mesmo evento se inverte entre os dois grupos - ex.: expiracao de
# patente e negativa para a referencia e positiva para o generico.
LABORATORIOS_REFERENCIA: list[str] = [
    "novo nordisk", "eli lilly", "lilly", "astrazeneca", "boehringer ingelheim",
    "boehringer", "merck", "msd", "sanofi", "novartis", "servier", "takeda",
    "abbott", "roche", "bayer", "johnson", "pfizer", "glaxosmithkline", "gsk",
]

LABORATORIOS_NACIONAIS: list[str] = [
    "ems", "hypera", "eurofarma", "ache", "cristalia", "medley", "germed",
    "biolab", "libbs", "mantecorp", "cimed", "neo quimica", "prati-donaduzzi",
    "prati donaduzzi", "legrand", "nova quimica", "sandoz", "teuto", "geolab",
]

# Orgaos com poder de alterar diretamente receita, preco ou acesso.
ORGAOS_REGULADORES: list[str] = [
    "anvisa", "cmed", "conitec", "ministerio da saude", "ministry of health",
    "sus", "ans", "fda", "food and drug administration", "ema",
    "european medicines agency", "inpi", "farmacia popular", "cade",
    "sindusfarma", "interfarma", "cff", "sbd",
    "sociedade brasileira de diabetes", "who", "world health organization",
    "nice", "ich", "usp",
]

# Canal de varejo - relevante porque a base IQVIA e de varejo (PMB).
CANAIS_VAREJO: list[str] = [
    "farmacia", "farmacias", "pharmacy", "pharmacies",
    "drogaria", "drugstore", "drug store", "retail pharmacy",
    "raia drogasil", "rd saude", "droga raia", "drogasil",
    "pague menos", "panvel", "drogaria sao paulo", "ultrafarma",
    "venancio", "extrafarma", "pbm", "pharmacy benefit manager",
    "farmacia popular", "abrafarma", "febrafar", "cvs", "walgreens",
]

# Termos de cadeia de suprimentos - antecipam ruptura de oferta, que na
# base de vendas aparece como queda abrupta sem causa comercial.
# Estes sao o gatilho do escopo SISTEMICO no Agente 2: noticia macro sem
# farmaco citado, mas com efeito plausivel sobre custo ou oferta.
CADEIA_SUPRIMENTOS: list[str] = [
    # insumo
    "ifa", "insumo farmaceutico ativo", "principio ativo", "materia-prima",
    "active pharmaceutical ingredient", "api", "raw material", "excipient",
    # oferta
    "importacao", "import", "imports", "export", "exports",
    "desabastecimento", "desabastecido", "shortage", "shortages",
    "falta de estoque", "stockout", "out of stock", "ruptura",
    "supply chain", "cadeia de suprimentos", "supply disruption",
    "recall", "recolhimento", "lote", "batch", "producao", "production",
    "fabrica", "factory", "plant", "manufacturing",
    "logistica", "logistics", "frete", "freight", "shipping",
    # custo
    "cambio", "exchange rate", "dolar", "dollar", "currency",
    "tarifa", "tariff", "tariffs", "sobretaxa", "surcharge",
    "inflacao", "inflation", "interest rate", "juros",
]

# Regioes - permitem recortar o impacto por territorio de vendas.
REGIOES_BRASIL: dict[str, list[str]] = {
    "norte": ["acre", "amapa", "amazonas", "para", "rondonia", "roraima", "tocantins", "regiao norte"],
    "nordeste": ["alagoas", "bahia", "ceara", "maranhao", "paraiba", "pernambuco",
                 "piaui", "rio grande do norte", "sergipe", "nordeste"],
    "centro-oeste": ["distrito federal", "goias", "mato grosso", "mato grosso do sul",
                     "brasilia", "centro-oeste"],
    "sudeste": ["espirito santo", "minas gerais", "rio de janeiro", "sao paulo", "sudeste"],
    "sul": ["parana", "rio grande do sul", "santa catarina", "regiao sul"],
}


# =====================================================================
# 3. TAXONOMIA DE EVENTOS E POLARIDADE
# =====================================================================


class Polaridade(str, Enum):
    """Direcao esperada do efeito sobre o faturamento."""

    POSITIVA = "POSITIVA"
    NEGATIVA = "NEGATIVA"
    NEUTRA = "NEUTRA"
    AMBIVALENTE = "AMBIVALENTE"  # efeito depende de qual lado se olha


@dataclass(frozen=True)
class TipoEvento:
    """
    Um tipo de evento de mercado, com sua polaridade esperada.

    'peso_impacto' (0..1) expressa a magnitude tipica do evento e serve
    de PRIOR para o Agente 2 quando o classificador neural fica
    inseguro - nao substitui a predicao do mBERT, apenas a contextualiza.
    """

    codigo: str
    descricao: str
    polaridade: Polaridade
    peso_impacto: float
    padroes: tuple[str, ...] = field(default_factory=tuple)
    # Quando a polaridade se inverte para o concorrente (ex.: recall do
    # lider abre espaco para o segundo colocado).
    inverte_para_concorrente: bool = False


# A ordem importa: eventos mais especificos e de maior severidade vem
# primeiro, porque a extracao para no primeiro padrao satisfeito por
# categoria (evita que "aprovacao" generica sobreponha "recall").
TAXONOMIA_EVENTOS: tuple[TipoEvento, ...] = (
    TipoEvento(
        codigo="RECALL_SUSPENSAO",
        descricao="Recolhimento, suspensao de lote ou cassacao de registro",
        polaridade=Polaridade.NEGATIVA,
        peso_impacto=0.95,
        padroes=(
            r"\brecall\b", r"\brecalled\b", r"\brecalls\b",
            r"\brecolhiment\w+", r"suspens\w+ (?:d[eo] )?(?:lote|venda|registro|fabricac)",
            r"cassac\w+ (?:d[eo] )?registro", r"interdic\w+", r"\bapreens\w+",
            r"\bwithdraw\w*\b[^.]{0,30}\bmarket\b", r"\bpull\w*\b[^.]{0,20}\bfrom (?:the )?market\b",
            r"suspend\w*\b[^.]{0,30}\b(?:sale|licen[cs]e|registration|production)",
            r"\bseiz\w+\b", r"\bhalt\w*\b[^.]{0,25}\bproduction\b",
        ),
        inverte_para_concorrente=True,
    ),
    TipoEvento(
        codigo="DESABASTECIMENTO",
        descricao="Falta de produto, ruptura de estoque ou de insumo",
        polaridade=Polaridade.NEGATIVA,
        peso_impacto=0.90,
        padroes=(
            r"desabastec\w+", r"\bfalta d[eo]\b", r"\bruptura\b", r"escassez",
            r"sem estoque", r"estoque zerado", r"\bdesabastecid\w+",
            r"\bshortage[s]?\b", r"\bstockout\b", r"\bout of stock\b",
            r"\bsupply (?:disruption|crisis|constraint)", r"\brunning (?:low|out)\b",
            r"\bscarcit\w+", r"\bunable to (?:supply|meet demand)",
        ),
        inverte_para_concorrente=True,
    ),
    TipoEvento(
        codigo="REGULATORIO_APROVACAO",
        descricao="Aprovacao de registro, nova indicacao ou ampliacao de bula",
        polaridade=Polaridade.POSITIVA,
        peso_impacto=0.85,
        padroes=(
            r"\baprov\w+", r"autoriz\w+ (?:a )?(?:comercializac|uso|venda)",
            r"registr\w+ (?:concedid|aprovad)", r"nova indicac\w+",
            r"amplia\w+ (?:a )?bula", r"libera\w+ (?:o )?uso",
            r"\bapprov\w+\b", r"\bauthoris\w+\b", r"\bauthoriz\w+\b",
            r"\bclear\w*\b[^.]{0,20}\bfor (?:use|sale|market)",
            r"\bgreen ?light\b", r"\bnew indication\b", r"\blabel expansion\b",
            r"\bgranted (?:approval|marketing authorisation|authorization)",
        ),
    ),
    TipoEvento(
        codigo="REGULATORIO_NEGATIVA",
        descricao="Indeferimento, negativa de registro ou restricao de uso",
        polaridade=Polaridade.NEGATIVA,
        peso_impacto=0.85,
        padroes=(
            r"indeferi\w+", r"negou (?:o )?registro", r"reprov\w+",
            r"restric\w+ (?:de |ao )?uso", r"proib\w+", r"veta\w+", r"contraindicac\w+ nova",
            r"\breject\w+\b", r"\bdenie[ds]\b", r"\brefus\w+ (?:approval|authoris|authoriz)",
            r"\bban\w*\b[^.]{0,25}\b(?:drug|medicine|sale)", r"\brestrict\w+ (?:the )?use",
            r"\bcomplete response letter\b", r"\bnegative opinion\b",
        ),
    ),
    TipoEvento(
        codigo="INCORPORACAO_SUS",
        descricao="Incorporacao ao SUS/CONITEC ou inclusao na Farmacia Popular",
        polaridade=Polaridade.POSITIVA,
        peso_impacto=0.90,
        padroes=(
            r"incorporac\w+", r"incorpor\w+ ao sus", r"conitec (?:aprov|recomend)",
            r"farmacia popular", r"distribuic\w+ gratuita", r"passa a ser oferecid",
            r"protocolo clinico", r"\bpcdt\b",
            r"\breimburs\w+\b", r"\bcoverage (?:expansion|decision)",
            r"\badded to\b[^.]{0,25}\b(?:formulary|list)", r"\bpublicly funded\b",
            r"\bfree (?:of charge )?(?:on|through) the\b", r"\bwill (?:now )?cover\b",
        ),
    ),
    TipoEvento(
        codigo="EXCLUSAO_SUS",
        descricao="Exclusao de lista publica ou fim de fornecimento gratuito",
        polaridade=Polaridade.NEGATIVA,
        peso_impacto=0.85,
        padroes=(r"exclu\w+ (?:d[oa] )?(?:sus|lista|rename)", r"fim d[oa] (?:distribuic|fornecimento)"),
    ),
    TipoEvento(
        codigo="PRECO_REAJUSTE",
        descricao="Reajuste de preco autorizado pela CMED",
        polaridade=Polaridade.POSITIVA,
        peso_impacto=0.75,
        padroes=(
            r"reajust\w+", r"aument\w+ (?:d[eo] )?preco", r"\bcmed\b", r"teto de preco",
            r"\bprice (?:increase|hike|rise|adjustment)", r"\braise[sd]? (?:the )?price",
            r"\bprice cap\b", r"\bmore expensive\b",
        ),
    ),
    TipoEvento(
        codigo="PRECO_REDUCAO",
        descricao="Reducao de preco, guerra de precos ou pressao de desconto",
        polaridade=Polaridade.NEGATIVA,
        peso_impacto=0.70,
        padroes=(
            r"reduc\w+ (?:d[eo] )?preco", r"queda (?:n[oa] )?preco", r"desconto agressiv",
            r"guerra de precos", r"barateamento",
            r"\bprice (?:cut|drop|reduction|war|pressure)", r"\bcheaper\b",
            r"\blower (?:the )?price", r"\bdiscount\w*\b", r"\bprice erosion\b",
        ),
    ),
    TipoEvento(
        codigo="PATENTE_EXPIRACAO",
        descricao="Expiracao de patente, abrindo espaco para genericos",
        polaridade=Polaridade.NEGATIVA,
        peso_impacto=0.90,
        # Padroes de PROXIMIDADE, e nao de adjacencia estrita: entre
        # "patente" e o verbo quase sempre aparece o nome do produto
        # ("Patente de Forxiga expira"). Exigir as palavras coladas
        # fazia o padrao falhar na construcao mais comum do noticiario.
        # O limite [^.]{0,50} mantem o casamento dentro da mesma frase,
        # evitando falso positivo entre sentencas distantes.
        padroes=(
            r"\bpatente\b[^.]{0,50}?\b(?:expir|vence|venci|caduc)\w*",
            r"\b(?:expirac|vencimento|quebra|perda|fim)\w*\b[^.]{0,50}?\bpatente\b",
            r"perda d[ae] exclusividade",
            r"\bpatente\b[^.]{0,50}?\bdominio publico\b",
            r"\bpatent\b[^.]{0,50}?\b(?:expir|lapse|end|run out)\w*",
            r"\b(?:expir|lapse|loss|end)\w*\b[^.]{0,50}?\bpatent\b",
            r"\bloss of exclusivity\b", r"\bpatent cliff\b", r"\boff-?patent\b",
        ),
        inverte_para_concorrente=True,
    ),
    TipoEvento(
        codigo="GENERICO_ENTRADA",
        descricao="Lancamento de generico ou similar concorrente",
        polaridade=Polaridade.NEGATIVA,
        peso_impacto=0.85,
        padroes=(
            r"\bgeneric\w+", r"\bsimilar\b", r"biossimilar", r"biosimilar",
            r"lanc\w+ (?:a )?versao mais barata", r"\bcopycat\b",
            r"\bgeneric (?:version|entry|competition|launch)",
        ),
        inverte_para_concorrente=True,
    ),
    TipoEvento(
        codigo="CONCORRENTE_LANCAMENTO",
        descricao="Lancamento de produto concorrente ou entrada de novo player",
        polaridade=Polaridade.NEGATIVA,
        peso_impacto=0.80,
        padroes=(r"lanca\w* (?:no |o |a )?(?:mercado|produto|medicament)", r"chega ao mercado",
                 r"novo (?:concorrente|player|entrante)", r"estreia no mercado"),
        inverte_para_concorrente=True,
    ),
    TipoEvento(
        codigo="ENSAIO_CLINICO_POSITIVO",
        descricao="Resultado favoravel de estudo clinico",
        polaridade=Polaridade.POSITIVA,
        peso_impacto=0.60,
        padroes=(
            r"estudo (?:mostr|revel|apont|demonstr)\w+ (?:eficac|beneficio|reduc)",
            r"resultado[s]? positiv", r"fase 3 (?:bem-?sucedid|positiv)", r"eficacia comprovada",
            r"\bstudy (?:shows|found|finds|reveals|suggests)",
            r"\btrial (?:success|met|showed)", r"\bphase (?:3|iii)\b[^.]{0,30}\b(?:success|positive|met)",
            r"\bmet (?:the )?primary endpoint", r"\bsignificant(?:ly)? reduc\w+",
            r"\bcould (?:prevent|stop|reduce|help)",
        ),
    ),
    TipoEvento(
        codigo="SEGURANCA_ALERTA",
        descricao="Alerta de seguranca, evento adverso ou farmacovigilancia",
        polaridade=Polaridade.NEGATIVA,
        peso_impacto=0.85,
        padroes=(
            r"efeito[s]? colateral", r"evento[s]? advers", r"alerta de seguranca",
            r"risco de (?:morte|cancer|amputac|cetoacidose|pancreatite)", r"farmacovigilancia",
            r"\bside effect[s]?\b", r"\badverse (?:event|reaction|effect)",
            r"\bsafety (?:alert|signal|concern|warning|review)",
            r"\bblack box warning\b", r"\bpharmacovigilance\b",
            r"\brisk of (?:death|cancer|amputation|ketoacidosis|pancreatitis|blindness)",
            r"\blinked to\b[^.]{0,30}\b(?:risk|death|cancer)",
        ),
        inverte_para_concorrente=True,
    ),
    TipoEvento(
        codigo="CADEIA_SUPRIMENTO_RISCO",
        descricao="Risco na cadeia de insumos, importacao ou logistica",
        polaridade=Polaridade.NEGATIVA,
        peso_impacto=0.70,
        padroes=(
            r"\bifa\b", r"insumo farmaceutico", r"importac\w+ (?:de |d[eo] )?(?:insumo|materia)",
            r"cadeia de suprimento", r"gargalo (?:logistic|de producao)", r"tarifa\w* sobre",
            r"\bsupply chain\b", r"\bactive pharmaceutical ingredient\b",
            r"\braw material[s]?\b", r"\bbottleneck\b", r"\btariff[s]?\b",
            r"\bimport (?:ban|restriction|dependency|reliance)",
            r"\bmanufacturing (?:issue|problem|delay|disruption)",
            r"\bport (?:closure|congestion)", r"\bexport control",
        ),
    ),
    TipoEvento(
        codigo="CAMBIO_CUSTO",
        descricao="Variacao cambial ou pressao de custo sobre margem",
        polaridade=Polaridade.NEGATIVA,
        peso_impacto=0.55,
        padroes=(
            r"\bdolar\b", r"\bcambi\w+", r"alta d[oa] moeda", r"pressao de custo", r"inflac\w+",
            r"\bdollar\b", r"\bexchange rate\b", r"\bcurrency\b", r"\bforex\b",
            r"\binflation\b", r"\bcost pressure\b", r"\brising costs?\b",
            r"\binterest rate[s]?\b", r"\bdevaluation\b", r"\bweaker (?:real|currency)",
        ),
    ),
    TipoEvento(
        codigo="FUSAO_AQUISICAO",
        descricao="Fusao, aquisicao, joint venture ou parceria comercial",
        polaridade=Polaridade.AMBIVALENTE,
        peso_impacto=0.75,
        padroes=(
            r"aquisic\w+", r"\bfusao\b", r"compra d[ae] (?:empresa|laboratorio|farmaceutica)",
            r"joint venture", r"parceria (?:estrategica|comercial)", r"\bmerger\b",
            r"\bacquisition\b", r"\bacquir\w+\b", r"\bbuyout\b", r"\btakeover\b",
            r"\bstrategic partnership\b", r"\blicensing deal\b", r"\bm&a\b",
        ),
    ),
    TipoEvento(
        codigo="CAPACIDADE_EXPANSAO",
        descricao="Expansao de capacidade produtiva ou novo investimento fabril",
        polaridade=Polaridade.POSITIVA,
        peso_impacto=0.65,
        padroes=(r"expans\w+ (?:d[ae] )?(?:producao|capacidade|fabrica)", r"nova fabrica",
                 r"investiment\w+ (?:de |em )?(?:r\$|us\$|producao)", r"amplia\w+ (?:a )?producao"),
    ),
    TipoEvento(
        codigo="DEMANDA_AUMENTO",
        descricao="Crescimento de demanda, prevalencia ou procura pelo produto",
        polaridade=Polaridade.POSITIVA,
        peso_impacto=0.65,
        padroes=(
            r"aument\w+ (?:d[ae] )?(?:demanda|procura|consumo|vendas)", r"disparad\w+ (?:as )?vendas",
            r"cresc\w+ (?:o |a )?(?:mercado|demanda|prevalencia)", r"boom d[eo]", r"recorde de vendas",
            r"\bsurg\w+ (?:in )?demand", r"\bsoaring (?:sales|demand)",
            r"\brecord (?:sales|revenue|quarter)", r"\bblockbuster\b",
            r"\bsales (?:jump|surge|soar|climb|rise|grew|growth)",
            r"\bdemand (?:surge|boom|spike)", r"\bbest-?sell\w+",
        ),
    ),
    TipoEvento(
        codigo="EPIDEMIOLOGIA",
        descricao="Dado epidemiologico de prevalencia ou incidencia de diabetes",
        polaridade=Polaridade.POSITIVA,
        peso_impacto=0.45,
        padroes=(
            r"prevalencia", r"incidencia", r"numero de (?:diabetic|cas)", r"\bidf\b",
            r"atlas do diabetes", r"milhoes de (?:brasileiros|pessoas) com diabetes",
            r"\bprevalence\b", r"\bincidence\b", r"\bmillion (?:people|adults)\b[^.]{0,30}\bdiabet",
            r"\bdiabetes (?:atlas|epidemic)", r"\bcases? (?:of|are) (?:diabetes|rising)",
        ),
    ),
    TipoEvento(
        codigo="INSTITUCIONAL",
        descricao="Campanha, conscientizacao, entrevista ou conteudo sem fato de mercado",
        polaridade=Polaridade.NEUTRA,
        peso_impacto=0.15,
        padroes=(r"campanha", r"conscientizac\w+", r"dia mundial", r"novembro azul",
                 r"entrevista", r"\bdicas?\b", r"saiba como", r"especialista (?:explica|alerta|orienta)"),
    ),
)

# Indice por codigo, para lookup O(1).
EVENTOS_POR_CODIGO: dict[str, TipoEvento] = {e.codigo: e for e in TAXONOMIA_EVENTOS}

# Eventos que, por si so, justificam tratamento prioritario mesmo que o
# classificador neural tenha atribuido impacto moderado. Usado pelo
# Agente 2 como regra de negocio de PROMOCAO (override auditavel).
EVENTOS_CRITICOS: frozenset[str] = frozenset({
    "RECALL_SUSPENSAO",
    "DESABASTECIMENTO",
    "PATENTE_EXPIRACAO",
    "INCORPORACAO_SUS",
    "SEGURANCA_ALERTA",
})


# =====================================================================
# 4. EXTRATOR
# =====================================================================


@dataclass
class EntidadesExtraidas:
    """Resultado da extracao de entidades e eventos de um documento."""

    principios_ativos: list[str] = field(default_factory=list)
    classes_terapeuticas: list[str] = field(default_factory=list)
    marcas: list[str] = field(default_factory=list)
    laboratorios: list[str] = field(default_factory=list)
    laboratorios_nacionais: list[str] = field(default_factory=list)
    orgaos: list[str] = field(default_factory=list)
    canais: list[str] = field(default_factory=list)
    regioes: list[str] = field(default_factory=list)
    tipos_evento: list[str] = field(default_factory=list)
    termos_cadeia: list[str] = field(default_factory=list)

    @property
    def tem_ancora_de_mercado(self) -> bool:
        """
        Indica se a noticia pode ser LIGADA a uma linha da base de
        vendas. Sem ao menos um principio ativo, marca ou laboratorio,
        o Agente 4 nao tem chave de juncao com os dados IQVIA e a
        'recomendacao tatica' resultante seria generica.
        """
        return bool(self.principios_ativos or self.marcas or self.laboratorios or self.laboratorios_nacionais)

    @property
    def escopo_relevancia(self) -> str:
        """
        Classifica COMO a noticia pode ser usada pelo Agente 4.

        Uma versao anterior tratava isso como sim/nao: ou a noticia
        citava um farmaco, ou era descartada. Medido no corpus real,
        esse criterio eliminaria 78% dos documentos - incluindo 484 que
        os avaliadores humanos marcaram como ALTO impacto. O erro foi
        confundir "nao da para atribuir a um produto" com "nao serve
        para nada".

        A coleta deste projeto e deliberadamente multi-eixo: alem de
        farmacos, busca cambio, dolar, desastres naturais e cadeia de
        suprimentos, porque esses eventos afetam custo e oferta de TODO
        o mercado. Sao sinais legitimos - so nao sao atribuiveis a um
        SKU especifico.

        Os tres escopos:
          ESPECIFICO - cita farmaco ou laboratorio. O Agente 4 consegue
                       juntar com uma linha da base de vendas e gerar
                       recomendacao dirigida ("reforce estoque de X em Y").
          SISTEMICO  - nao cita produto, mas traz evento macro (cambio,
                       IFA, logistica, tarifa). Vira alerta de contexto
                       para a carteira inteira, nao para um item.
          SEM_ANCORA - nem produto nem evento macro. Nao sustenta
                       recomendacao tatica de nenhum tipo.
        """
        if self.tem_ancora_de_mercado:
            return "ESPECIFICO"
        macro = {"CADEIA_SUPRIMENTO_RISCO", "CAMBIO_CUSTO", "DESABASTECIMENTO"}
        if set(self.tipos_evento) & macro or self.termos_cadeia:
            return "SISTEMICO"
        return "SEM_ANCORA"

    @property
    def evento_principal(self) -> str | None:
        """
        Evento de maior peso entre os detectados. Serve de rotulo
        sintetico para o painel e para o prompt dos Agentes 4 e 5.
        """
        if not self.tipos_evento:
            return None
        return max(self.tipos_evento, key=lambda c: EVENTOS_POR_CODIGO[c].peso_impacto)

    def to_dict(self) -> dict:
        return {
            "principios_ativos": "; ".join(self.principios_ativos),
            "classes_terapeuticas": "; ".join(self.classes_terapeuticas),
            "marcas": "; ".join(self.marcas),
            "laboratorios": "; ".join(self.laboratorios + self.laboratorios_nacionais),
            "orgaos": "; ".join(self.orgaos),
            "canais": "; ".join(self.canais),
            "regioes": "; ".join(self.regioes),
            "tipos_evento": "; ".join(self.tipos_evento),
            "evento_principal": self.evento_principal or "",
            "tem_ancora_de_mercado": self.tem_ancora_de_mercado,
            "escopo_relevancia": self.escopo_relevancia,
        }


def _compilar_termos(termos: list[str]) -> re.Pattern:
    """
    Compila uma lista de termos em um unico regex com fronteira de
    palavra. Um regex alternado e ~10x mais rapido que N buscas
    independentes, o que importa em corpus de milhares de documentos.
    """
    alternativas = sorted((re.escape(t) for t in termos), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(alternativas) + r")\b")


# Regexes pre-compilados no import (custo pago uma vez, nao por documento).
_RE_ATIVOS = {classe: _compilar_termos(termos) for classe, termos in PRINCIPIOS_ATIVOS.items()}
_RE_MARCAS = _compilar_termos(list(MARCAS_COMERCIAIS.keys()))
_RE_LAB_REF = _compilar_termos(LABORATORIOS_REFERENCIA)
_RE_LAB_NAC = _compilar_termos(LABORATORIOS_NACIONAIS)
_RE_ORGAOS = _compilar_termos(ORGAOS_REGULADORES)
_RE_CANAIS = _compilar_termos(CANAIS_VAREJO)
_RE_CADEIA = _compilar_termos(CADEIA_SUPRIMENTOS)
_RE_REGIOES = {reg: _compilar_termos(termos) for reg, termos in REGIOES_BRASIL.items()}
_RE_EVENTOS = {e.codigo: [re.compile(p) for p in e.padroes] for e in TAXONOMIA_EVENTOS}


def extrair_entidades(texto: str) -> EntidadesExtraidas:
    """
    Extrai entidades e tipos de evento de um documento.

    Complexidade: O(n) no tamanho do texto, com constante baixa por
    causa dos regexes alternados pre-compilados. Na pratica, ~0,3 ms
    por noticia - desprezivel frente ao forward pass do mBERT.
    """
    t = normalizar(texto)
    res = EntidadesExtraidas()

    for classe, padrao in _RE_ATIVOS.items():
        achados = set(padrao.findall(t))
        if achados:
            res.principios_ativos.extend(sorted(achados))
            res.classes_terapeuticas.append(classe)

    marcas = sorted(set(_RE_MARCAS.findall(t)))
    res.marcas = marcas
    # Marca implica principio ativo: resolve "Ozempic" -> "semaglutida"
    # para que a juncao com a base de vendas funcione nos dois casos.
    for marca in marcas:
        ativo = MARCAS_COMERCIAIS[marca]
        if ativo not in res.principios_ativos:
            res.principios_ativos.append(ativo)

    res.laboratorios = sorted(set(_RE_LAB_REF.findall(t)))
    res.laboratorios_nacionais = sorted(set(_RE_LAB_NAC.findall(t)))
    res.orgaos = sorted(set(_RE_ORGAOS.findall(t)))
    res.canais = sorted(set(_RE_CANAIS.findall(t)))
    res.termos_cadeia = sorted(set(_RE_CADEIA.findall(t)))
    res.regioes = sorted(reg for reg, padrao in _RE_REGIOES.items() if padrao.search(t))

    res.tipos_evento = sorted(
        codigo for codigo, padroes in _RE_EVENTOS.items() if any(p.search(t) for p in padroes)
    )
    # Canoniza para a grafia em portugues: e ela que a base de vendas usa.
    res.principios_ativos = sorted(
        {SINONIMOS_ATIVO.get(a, a) for a in res.principios_ativos}
    )
    res.classes_terapeuticas = sorted(set(res.classes_terapeuticas))
    return res


# =====================================================================
# 5. DIRECAO DO IMPACTO
# =====================================================================


@dataclass
class DirecaoImpacto:
    """
    Direcao esperada do efeito sobre o faturamento, com rastreabilidade.

    Por que isso e necessario: 'Alto impacto' e ambiguo do ponto de
    vista da decisao comercial. Um recall do concorrente e alto impacto
    POSITIVO (oportunidade de captura de share); um recall do proprio
    produto e alto impacto NEGATIVO (risco de receita). Sem a direcao,
    os Agentes 4 e 5 nao conseguem distinguir 'risco' de 'oportunidade'
    - que e literalmente o nome do Agente 4.
    """

    polaridade: str
    confianca: float
    evento_base: str | None
    justificativa: str

    def to_dict(self) -> dict:
        return {
            "direcao_impacto": self.polaridade,
            "direcao_confianca": round(self.confianca, 4),
            "direcao_evento_base": self.evento_base or "",
            "direcao_justificativa": self.justificativa,
        }


def inferir_direcao(entidades: EntidadesExtraidas) -> DirecaoImpacto:
    """
    Deriva a direcao do impacto a partir do TIPO DE EVENTO detectado,
    e nao de analise de sentimento generica.

    Vantagem sobre sentimento: "ANVISA aprova generico de dapagliflozina"
    tem sentimento textual positivo, mas e NEGATIVO para o faturamento
    do detentor da marca de referencia. Ancorar a direcao no tipo de
    evento captura essa inversao; um analisador de sentimento nao.

    Contra: depende de o evento ter sido detectado pelo gazetteer. Sem
    evento reconhecido, retorna NEUTRA com confianca baixa - o que
    sinaliza corretamente ao Agente 2 que aquele item precisa de
    validacao adicional, em vez de fabricar uma direcao falsa.
    """
    if not entidades.tipos_evento:
        return DirecaoImpacto(
            polaridade=Polaridade.NEUTRA.value,
            confianca=0.20,
            evento_base=None,
            justificativa="Nenhum tipo de evento de mercado reconhecido no texto.",
        )

    codigo = entidades.evento_principal
    evento = EVENTOS_POR_CODIGO[codigo]

    # Concordancia entre multiplos eventos detectados aumenta a
    # confianca; sinais contraditorios a reduzem.
    polaridades = {EVENTOS_POR_CODIGO[c].polaridade for c in entidades.tipos_evento}
    conflitantes = {Polaridade.POSITIVA, Polaridade.NEGATIVA}.issubset(polaridades)

    confianca = evento.peso_impacto
    justificativa = f"Evento '{evento.codigo}' ({evento.descricao}) -> {evento.polaridade.value}."

    if conflitantes:
        confianca *= 0.65
        justificativa += " Sinais de polaridade oposta no mesmo texto reduzem a confianca."
    if not entidades.tem_ancora_de_mercado:
        confianca *= 0.70
        justificativa += " Sem produto/laboratorio identificado, o efeito nao e atribuivel."
    if evento.inverte_para_concorrente:
        justificativa += " Atencao: a polaridade se inverte para o concorrente direto."

    return DirecaoImpacto(
        polaridade=evento.polaridade.value,
        confianca=min(1.0, confianca),
        evento_base=evento.codigo,
        justificativa=justificativa,
    )
