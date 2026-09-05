import os

# ---------------------------------------------------------------------------
# Diretorios (arquitetura Medalhao: Bronze -> Prata -> Ouro)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
BRONZE_DIR = os.path.join(DATA_DIR, 'bronze')
PRATA_DIR = os.path.join(DATA_DIR, 'prata')
OURO_DIR = os.path.join(DATA_DIR, 'ouro')

# Tabela de referencia municipio -> regiao intermediaria (IBGE 2017). Essa
# classificacao e fixa no tempo, mas nem todo ano do Censo Escolar traz as
# colunas CO_REGIAO_GEOG_INTERM/NO_REGIAO_GEOG_INTERM (confirmado: ausentes
# em 2017-2022, presentes apenas em 2023/2024). Guardamos a correspondencia
# aqui na primeira vez que aparece num ano que tem essas colunas, e
# reaproveitamos para os anos que nao tem.
REGIAO_INTERM_REF_PATH = os.path.join(DATA_DIR, 'referencia_regiao_intermediaria.csv')

# ---------------------------------------------------------------------------
# Recorte temporal (2017 a 2024, inclusive)
# ---------------------------------------------------------------------------
ANOS = range(2017, 2025)

# Ano de referencia de cada fase, usado no calculo dos indicadores
ANO_PRE_PANDEMIA = 2019
ANO_PANDEMIA = 2021
ANO_POS_PANDEMIA = 2024

# UF de interesse (Minas Gerais)
CO_UF_MG = 31

# ---------------------------------------------------------------------------
# Selecao de variaveis por grupo funcional (Secao 3.6.2.1 do TCC)
# Usadas em conjunto como parametro `usecols` do pd.read_csv().
# ---------------------------------------------------------------------------

COLUNAS_IDENTIFICACAO = [
    'CO_ENTIDADE', 'CO_UF', 'CO_MUNICIPIO', 'NO_MUNICIPIO',
    'CO_REGIAO_GEOG_INTERM', 'NO_REGIAO_GEOG_INTERM',
    'TP_LOCALIZACAO_DIFERENCIADA',
]

COLUNAS_MATRICULAS = [
    'QT_MAT_BAS', 'QT_MAT_INF', 'QT_MAT_FUND', 'QT_MAT_MED',
    'QT_MAT_EJA', 'QT_MAT_BAS_EAD',
]

COLUNAS_MODALIDADE = [
    'IN_MEDIACAO_PRESENCIAL', 'IN_MEDIACAO_SEMIPRESENCIAL', 'IN_MEDIACAO_EAD',
]

COLUNAS_ESTRUTURAIS = [
    'TP_DEPENDENCIA', 'TP_LOCALIZACAO', 'IN_INTERNET',
    'IN_INTERNET_APRENDIZAGEM', 'IN_INTERNET_ALUNOS', 'IN_BANDA_LARGA',
    'IN_LABORATORIO_INFORMATICA', 'IN_ENERGIA_REDE_PUBLICA', 'IN_AGUA_POTAVEL',
]

COLUNAS_EQUIPAMENTOS = [
    'IN_DESKTOP_ALUNO', 'QT_DESKTOP_ALUNO',
    'IN_COMP_PORTATIL_ALUNO', 'QT_COMP_PORTATIL_ALUNO',
    'IN_TABLET_ALUNO', 'QT_TABLET_ALUNO',
]

COLUNAS_SITUACAO = ['TP_SITUACAO_FUNCIONAMENTO']

COLUNAS_ESSENCIAIS = (
    COLUNAS_IDENTIFICACAO
    + COLUNAS_MATRICULAS
    + COLUNAS_MODALIDADE
    + COLUNAS_ESTRUTURAIS
    + COLUNAS_EQUIPAMENTOS
    + COLUNAS_SITUACAO
)

# ---------------------------------------------------------------------------
# Mapeamento de nomes de colunas (Tabela 2 do TCC).
# ---------------------------------------------------------------------------
MAPEAMENTO_COLUNAS = {
    'CO_ENTIDADE': 'id_escola',
    'CO_MUNICIPIO': 'cod_municipio',
    'NO_MUNICIPIO': 'nome_municipio',
    'CO_UF': 'id_uf',
    'CO_REGIAO_GEOG_INTERM': 'cod_regiao_interm',
    'NO_REGIAO_GEOG_INTERM': 'nome_regiao_interm',
    'QT_MAT_BAS': 'total_matriculas',
    'QT_MAT_BAS_EAD': 'matriculas_ead',
    'IN_INTERNET': 'internet',
    'IN_INTERNET_APRENDIZAGEM': 'internet_aprendizagem',
    'IN_BANDA_LARGA': 'banda_larga',
    'TP_LOCALIZACAO': 'localizacao',
    'TP_DEPENDENCIA': 'dependencia_administrativa',
}

# Excecoes de nome de coluna por ano (grafia diferente do padrao atual)
EXCECOES_POR_ANO = {}

# ---------------------------------------------------------------------------
# Parametros de processamento (Secao 3.7)
# ---------------------------------------------------------------------------
CHUNK_SIZE = 50_000
LIMIAR_CARDINALIDADE_CATEGORICA = 50

# ---------------------------------------------------------------------------
# Download dos microdados
# ---------------------------------------------------------------------------
URLS_POR_ANO = {
    2017: "https://download.inep.gov.br/dados_abertos/microdados_censo_escolar_2017.zip",
    2018: "https://download.inep.gov.br/dados_abertos/microdados_censo_escolar_2018.zip",
    2019: "https://download.inep.gov.br/dados_abertos/microdados_censo_escolar_2019.zip",
    2020: "https://download.inep.gov.br/dados_abertos/microdados_censo_escolar_2020.zip",
    2021: "https://download.inep.gov.br/dados_abertos/microdados_censo_escolar_2021.zip",
    2022: "https://download.inep.gov.br/dados_abertos/microdados_censo_escolar_2022.zip",
    2023: "https://download.inep.gov.br/dados_abertos/microdados_censo_escolar_2023.zip",
    2024: "https://download.inep.gov.br/dados_abertos/microdados_censo_escolar_2024.zip",
}

HEADERS_DOWNLOAD = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-escolar",
    "Connection": "keep-alive",
}

# Padrao de nome de arquivo esperado para o CSV de microdados de escolas
# dentro do zip do INEP (case-insensitive).
PADRAO_NOME_CSV_MICRODADOS = "microdados_ed_basica"

# Nome padronizado do CSV apos extracao e renomeacao (Secao 3.6.1)
NOME_ARQUIVO_PADRAO = "censo_{ano}.csv"

# Numero maximo de tentativas de download e tempo de espera (segundos) entre
# elas. Como o download agora suporta retomada via HTTP Range (nao perde o
# progresso ja baixado em caso de queda de conexao), pode-se usar mais
# tentativas com espera menor sem custo de tempo adicional relevante.
DOWNLOAD_MAX_TENTATIVAS = 6
DOWNLOAD_ESPERA_SEGUNDOS = 30

# ---------------------------------------------------------------------------
# Pesquisa Resposta Educacional a Pandemia de Covid-19 (INEP), 2a edicao
# (ano letivo 2021), fonte do I_ERE (Indicador de Intensidade de Mediacao
# Remota). Levantamento pontual (nao ha edicao anual continua), por isso o
# arquivo e fornecido manualmente, nao baixado pelo extract.py.
# ---------------------------------------------------------------------------
PESQUISA_PANDEMIA_ARQUIVO = os.path.join(
    DATA_DIR, 'externos', 'Questionario_Resposta_Educacional_a_Pandemia_de_Covid_19_2ed.xlsx'
)
PESQUISA_PANDEMIA_ABA = '2'
PESQUISA_PANDEMIA_LINHA_CABECALHO = 10  # linha (1-indexed) com os codigos internos das colunas
PESQUISA_PANDEMIA_LINHA_DADOS_INICIO = 11

# ---------------------------------------------------------------------------
# Shapefile das Regioes Geograficas Intermediarias de MG (IBGE), usado para
# os mapas coropleticos (Secao 3.10/4.8). Baixado manualmente do portal do
# IBGE (mesma logica de arquivo externo do PESQUISA_PANDEMIA_ARQUIVO).
# ---------------------------------------------------------------------------
SHAPEFILE_REGIOES_PATH = os.path.join(
    DATA_DIR, 'externos', 'shapefile_regioes_intermediarias', 'MG_RG_Intermediarias_2025.shp'
)