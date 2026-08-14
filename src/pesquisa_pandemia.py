"""
Leitura da Pesquisa "Resposta Educacional a Pandemia de Covid-19 no Brasil"
(2a edicao, ano letivo de 2021), aplicada pelo INEP como questionario
suplementar ao Censo Escolar.

Diferente do Censo Escolar principal, essa pesquisa e um levantamento
pontual (edicoes de 2020 e 2021 apenas, sem continuidade anual) e por isso
NAO faz parte do loop de download automatico do pipeline (extract.py). O
arquivo XLSX deve ser baixado manualmente do portal do INEP e colocado no
caminho definido em PESQUISA_PANDEMIA_ARQUIVO (config.py).

Fonte: INEP - Sinopse Estatistica do Questionario Resposta Educacional a
Pandemia de Covid-19 no Brasil - Educacao Basica, 2a edicao (ano letivo 2021).
"""

import openpyxl
import pandas as pd

from src.config import (
    CO_UF_MG,
    PESQUISA_PANDEMIA_ABA,
    PESQUISA_PANDEMIA_ARQUIVO,
    PESQUISA_PANDEMIA_LINHA_CABECALHO,
    PESQUISA_PANDEMIA_LINHA_DADOS_INICIO,
)

NOME_UF_MG = "Minas Gerais"

# Colunas (por nome do codigo interno, linha de cabecalho da planilha) que
# precisamos: contexto geografico/desagregacao + a variavel de interesse.
COLUNAS_NECESSARIAS = [
    'NO_UF', 'NO_MUNICIPIO', 'CO_MUNICIPIO', 'TP_LOCALIZACAO', 'TP_DEPENDENCIA',
    'ESC_APLIC_Q2', 'MAT_FUND_AI', 'MEDIACAO_FUND_AI',
    'Q2_AIREM_MEDIA',   # dias medios de mediacao REMOTA, anos iniciais do EF
    'PERC_AIQ2_REM',    # % desses dias em relacao ao total (presencial+hibrido+remoto)
]


def _construir_indice_colunas(linha_cabecalho: tuple) -> dict:
    """Mapeia nome do codigo interno -> indice da coluna, a partir da linha
    de cabecalho tecnico da planilha (nao usa posicao fixa)."""
    indice = {valor: pos for pos, valor in enumerate(linha_cabecalho) if valor is not None}

    faltando = [c for c in COLUNAS_NECESSARIAS if c not in indice]
    if faltando:
        raise KeyError(
            f"As colunas {faltando} nao foram encontradas na planilha "
            f"'{PESQUISA_PANDEMIA_ABA}'. O layout do arquivo do INEP pode ter mudado -- "
            f"confira o cabecalho manualmente antes de prosseguir."
        )
    return indice


def carregar_indicador_ere() -> pd.DataFrame:
    """Le a planilha de estrategias de mediacao de ensino (2021) e retorna,
    por municipio de MG, os dias medios de mediacao remota nos anos iniciais
    do ensino fundamental, usado para construir o I_ERE.

    Filtra apenas as linhas agregadas por municipio (TP_LOCALIZACAO='Total'
    e TP_DEPENDENCIA='Total'), equivalentes ao total do municipio sem
    quebra por zona urbana/rural ou rede de ensino.
    """
    wb = openpyxl.load_workbook(PESQUISA_PANDEMIA_ARQUIVO, read_only=True, data_only=True)
    ws = wb[PESQUISA_PANDEMIA_ABA]

    linhas = ws.iter_rows(values_only=True, min_row=PESQUISA_PANDEMIA_LINHA_CABECALHO)
    linha_cabecalho = next(linhas)
    idx = _construir_indice_colunas(linha_cabecalho)

    registros = []
    for linha in linhas:
        if linha[idx['NO_UF']] != NOME_UF_MG:
            continue
        if linha[idx['NO_MUNICIPIO']] is None:
            continue
        if linha[idx['TP_LOCALIZACAO']] != 'Total' or linha[idx['TP_DEPENDENCIA']] != 'Total':
            continue

        registros.append({
            'cod_municipio': linha[idx['CO_MUNICIPIO']],
            'nome_municipio_pesquisa': linha[idx['NO_MUNICIPIO']],
            'escolas_respondentes_q2': linha[idx['ESC_APLIC_Q2']],
            'escolas_anos_iniciais': linha[idx['MAT_FUND_AI']],
            'escolas_anos_iniciais_com_mediacao': linha[idx['MEDIACAO_FUND_AI']],
            'dias_mediacao_remota_ai': linha[idx['Q2_AIREM_MEDIA']],
            'pct_dias_remoto_ai': linha[idx['PERC_AIQ2_REM']],
        })

    wb.close()

    df = pd.DataFrame(registros)
    df['cod_municipio'] = df['cod_municipio'].astype('int64')
    return df


def calcular_indicador_ere() -> pd.DataFrame:
    """I_ERE = dias medios de mediacao remota nos anos iniciais do ensino
    fundamental, por municipio, no ano letivo de 2021.

    Fonte: Pesquisa Resposta Educacional a Pandemia de Covid-19 no Brasil,
    2a edicao (INEP). Diferente do I_EAD (que mede matriculas formalmente
    classificadas como EAD), o I_ERE mede diretamente o que as escolas
    declararam ter feito durante a suspensao das atividades presenciais,
    sem a ambiguidade de reclassificacao de matricula discutida na Secao
    2.4.2/3.8.3 do TCC.
    """
    df = carregar_indicador_ere()
    return df[['cod_municipio', 'dias_mediacao_remota_ai']].rename(
        columns={'dias_mediacao_remota_ai': 'I_ERE'}
    )