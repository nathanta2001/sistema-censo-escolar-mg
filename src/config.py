import os

# Configurações de diretórios e parâmetros para o sistema de dados
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
BRONZE_DIR = os.path.join(DATA_DIR, 'bronze')
PRATA_DIR = os.path.join(DATA_DIR, 'prata')
OURO_DIR = os.path.join(DATA_DIR, 'ouro')

# Range de anos para processamento
ANOS = range(2017, 2024)

# Colunas usadas para o processamento dos dados
COLUNAS_ESSENCIAIS = [
    'CO_ENTIDADE', 'CO_UF', 'CO_MUNICIPIO', 'NO_MUNICIPIO',
    'QT_MAT_BAS', 'QT_MAT_INF', 'QT_MAT_FUND', 'QT_MAT_MED',
    'TP_SITUACAO_FUNCIONAMENTO', 'IN_INTERNET', 'IN_BANDA_LARGA'
]

# configurações de download dos dados
URL_BASE_INEP="https://download.inep.gov.br/dados_abertos/microdados_censo_escolar_{}.zip"
