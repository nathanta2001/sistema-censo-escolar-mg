import os

import numpy as np
import pandas as pd

from src.config import (
    BRONZE_DIR,
    CHUNK_SIZE,
    CO_UF_MG,
    COLUNAS_ESSENCIAIS,
    COLUNAS_EQUIPAMENTOS,
    COLUNAS_ESTRUTURAIS,
    COLUNAS_MATRICULAS,
    COLUNAS_MODALIDADE,
    EXCECOES_POR_ANO,
    LIMIAR_CARDINALIDADE_CATEGORICA,
    MAPEAMENTO_COLUNAS,
    NOME_ARQUIVO_PADRAO,
    REGIAO_INTERM_REF_PATH,
)

# Colunas numericas de matricula: ausencia = zero matriculas registradas
COLUNAS_MATRICULA_PREENCHER_ZERO = COLUNAS_MATRICULAS

# Colunas binarias (IN_*) de infraestrutura/modalidade: mantidas como NaN
# quando ausentes, pois zero indevido distorceria os indicadores (Secao 3.6.2)
COLUNAS_BINARIAS = [
    c for c in (COLUNAS_ESTRUTURAIS + COLUNAS_MODALIDADE + COLUNAS_EQUIPAMENTOS)
    if c.startswith('IN_')
]

# Colunas cuja ausencia no arquivo de um ano especifico e "esperada" (nao
# gera aviso de coluna faltando) porque ha um mecanismo de recuperacao
# proprio -- ver _aplicar_referencia_regiao().
COLUNAS_COM_FALLBACK = {'CO_REGIAO_GEOG_INTERM', 'NO_REGIAO_GEOG_INTERM'}


def _colunas_disponiveis(path: str) -> set:
    """Le apenas o cabecalho do CSV para descobrir quais colunas o arquivo
    realmente possui, sem carregar os dados."""
    cabecalho = pd.read_csv(path, sep=';', encoding='latin1', nrows=0)
    return set(cabecalho.columns)


def _aplicar_excecoes_de_nome(chunk: pd.DataFrame, ano: int) -> pd.DataFrame:
    """Renomeia colunas que, em um ano especifico, vieram com grafia
    diferente do padrao INEP atual, antes de aplicar o MAPEAMENTO_COLUNAS."""
    excecoes = EXCECOES_POR_ANO.get(ano)
    if excecoes:
        chunk = chunk.rename(columns=excecoes)
    return chunk


def _downcast_numericos(df: pd.DataFrame) -> pd.DataFrame:
    """Reduz o tipo de todas as colunas numericas para o menor tipo possivel
    (Secao 3.7)."""
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
    return df


def _categorizar_baixa_cardinalidade(df: pd.DataFrame) -> pd.DataFrame:
    """Converte colunas textuais/categoricas com poucos valores distintos
    para o tipo `category` (Secao 3.7)."""
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique(dropna=True) < LIMIAR_CARDINALIDADE_CATEGORICA:
            df[col] = df[col].astype('category')
    return df


# ---------------------------------------------------------------------------
# Tabela de referencia municipio -> regiao intermediaria
#
# A classificacao de Regioes Geograficas Intermediarias (IBGE, 2017) e fixa
# no tempo, mas nem todo ano do Censo Escolar traz as colunas
# CO_REGIAO_GEOG_INTERM/NO_REGIAO_GEOG_INTERM (ex.: ausentes em 2018). Em vez
# de deixar essas colunas como NaN nos anos em que faltam, mantemos uma
# tabela de referencia local: na primeira vez que um ano com essas colunas e
# processado, a correspondencia municipio -> regiao e salva; nos anos sem as
# colunas, ela e recuperada dessa tabela via CO_MUNICIPIO.
# ---------------------------------------------------------------------------
def carregar_referencia_regiao() -> pd.DataFrame:
    if os.path.exists(REGIAO_INTERM_REF_PATH):
        return pd.read_csv(REGIAO_INTERM_REF_PATH)
    return pd.DataFrame(columns=['CO_MUNICIPIO', 'CO_REGIAO_GEOG_INTERM', 'NO_REGIAO_GEOG_INTERM'])


def _atualizar_referencia_regiao(df: pd.DataFrame) -> None:
    """Atualiza a tabela de referencia com os pares municipio->regiao
    observados neste ano (chamado quando o ano TEM as colunas)."""
    referencia_atual = carregar_referencia_regiao()
    novos = df[['CO_MUNICIPIO', 'CO_REGIAO_GEOG_INTERM', 'NO_REGIAO_GEOG_INTERM']].drop_duplicates()

    combinado = (
        pd.concat([referencia_atual, novos], ignore_index=True)
        .drop_duplicates(subset=['CO_MUNICIPIO'])
    )

    os.makedirs(os.path.dirname(REGIAO_INTERM_REF_PATH), exist_ok=True)
    combinado.to_csv(REGIAO_INTERM_REF_PATH, index=False)


def _aplicar_referencia_regiao(df: pd.DataFrame, colunas_no_arquivo: set) -> pd.DataFrame:
    """Garante que CO_REGIAO_GEOG_INTERM/NO_REGIAO_GEOG_INTERM existam no
    DataFrame, usando a tabela de referencia quando o ano nao traz essas
    colunas no CSV original."""
    tem_colunas_regiao = {'CO_REGIAO_GEOG_INTERM', 'NO_REGIAO_GEOG_INTERM'}.issubset(colunas_no_arquivo)

    if tem_colunas_regiao:
        _atualizar_referencia_regiao(df)
        return df

    referencia = carregar_referencia_regiao()
    if referencia.empty:
        print(
            "AVISO: este ano nao traz CO_REGIAO_GEOG_INTERM/NO_REGIAO_GEOG_INTERM "
            "e ainda nao ha nenhum ano com essas colunas processado para "
            "preencher a partir da tabela de referencia. Os municipios deste "
            "ano ficarao sem regiao intermediaria ate que um ano com as "
            "colunas seja processado (rode-o e depois reprocesse este ano)."
        )
        df['CO_REGIAO_GEOG_INTERM'] = pd.NA
        df['NO_REGIAO_GEOG_INTERM'] = pd.NA
        return df

    df = df.merge(referencia, on='CO_MUNICIPIO', how='left')
    faltando = df['CO_REGIAO_GEOG_INTERM'].isna().sum()
    if faltando:
        print(
            f"AVISO: {faltando} registro(s) deste ano com municipio nao "
            f"encontrado na tabela de referencia de regiao intermediaria."
        )
    return df


def transformar_dados(ano: int) -> pd.DataFrame:
    """Le o CSV bruto (camada Bronze) de um ano, aplica filtragem geografica,
    filtro de escolas ativas, padronizacao de nomes e otimizacao de tipos,
    retornando o DataFrame pronto para a camada Prata.

    Robusto a colunas ausentes: nem todo ano do Censo Escolar traz todas as
    colunas de COLUNAS_ESSENCIAIS (Secao 3.6.2 - "variacoes significativas
    entre os anos"). As colunas realmente disponiveis no arquivo sao
    detectadas antes da leitura; as que faltam sao avisadas no console
    (exceto as que tem fallback proprio, ver COLUNAS_COM_FALLBACK).
    """

    path = f'{BRONZE_DIR}/{NOME_ARQUIVO_PADRAO.format(ano=ano)}'

    colunas_disponiveis = _colunas_disponiveis(path)
    colunas_usar = [c for c in COLUNAS_ESSENCIAIS if c in colunas_disponiveis]
    colunas_faltando = set(COLUNAS_ESSENCIAIS) - set(colunas_usar)

    avisar = colunas_faltando - COLUNAS_COM_FALLBACK
    if avisar:
        print(f"AVISO: ano {ano} nao possui as colunas {sorted(avisar)}. Prosseguindo sem elas.")

    chunks = pd.read_csv(
        path,
        sep=';',
        encoding='latin1',
        chunksize=CHUNK_SIZE,
        usecols=colunas_usar,
    )

    lista_processada = []
    for chunk in chunks:
        chunk = _aplicar_excecoes_de_nome(chunk, ano)

        # Filtragem geografica: apenas Minas Gerais
        df_mg = chunk[chunk['CO_UF'] == CO_UF_MG].copy()

        # Filtro: apenas escolas em atividade
        df_mg = df_mg[df_mg['TP_SITUACAO_FUNCIONAMENTO'] == 1]

        lista_processada.append(df_mg)

    df_final = pd.concat(lista_processada, ignore_index=True)

    # Remocao de registros duplicados (mesma escola no mesmo ano)
    df_final = df_final.drop_duplicates(subset=['CO_ENTIDADE'])

    # Recupera regiao intermediaria via tabela de referencia quando o ano
    # nao trouxe essas colunas (ver secao acima)
    df_final = _aplicar_referencia_regiao(df_final, colunas_disponiveis)

    # Tratamento de nulos: matriculas ausentes -> 0
    for col in COLUNAS_MATRICULA_PREENCHER_ZERO:
        if col in df_final.columns:
            df_final[col] = df_final[col].fillna(0)

    # Colunas binarias analiticas: mantidas como NaN se ausentes (nao sao
    # convertidas para 0/1 arbitrariamente)

    # Padronizacao de nomes de colunas (Tabela 2)
    df_final = df_final.rename(columns=MAPEAMENTO_COLUNAS)

    # Otimizacao de tipos (Secao 3.7)
    df_final = _downcast_numericos(df_final)
    df_final = _categorizar_baixa_cardinalidade(df_final)

    # Coluna auxiliar com o ano, util para concatenar anos na camada Ouro
    df_final['ano'] = np.int16(ano)

    return df_final