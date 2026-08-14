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

COLUNAS_MATRICULA_PREENCHER_ZERO = COLUNAS_MATRICULAS

COLUNAS_BINARIAS = [
    c for c in (COLUNAS_ESTRUTURAIS + COLUNAS_MODALIDADE + COLUNAS_EQUIPAMENTOS)
    if c.startswith('IN_')
]

COLUNAS_COM_FALLBACK = {'CO_REGIAO_GEOG_INTERM', 'NO_REGIAO_GEOG_INTERM'}


def _colunas_disponiveis(path: str) -> set:
    cabecalho = pd.read_csv(path, sep=';', encoding='latin1', nrows=0)
    return set(cabecalho.columns)


def _aplicar_excecoes_de_nome(chunk: pd.DataFrame, ano: int) -> pd.DataFrame:
    excecoes = EXCECOES_POR_ANO.get(ano)
    if excecoes:
        chunk = chunk.rename(columns=excecoes)
    return chunk


def _downcast_numericos(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
    return df


def _categorizar_baixa_cardinalidade(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique(dropna=True) < LIMIAR_CARDINALIDADE_CATEGORICA:
            df[col] = df[col].astype('category')
    return df


def carregar_referencia_regiao() -> pd.DataFrame:
    if os.path.exists(REGIAO_INTERM_REF_PATH):
        return pd.read_csv(REGIAO_INTERM_REF_PATH)
    return pd.DataFrame(columns=['CO_MUNICIPIO', 'CO_REGIAO_GEOG_INTERM', 'NO_REGIAO_GEOG_INTERM'])


def _atualizar_referencia_regiao(df: pd.DataFrame) -> None:
    referencia_atual = carregar_referencia_regiao()
    novos = df[['CO_MUNICIPIO', 'CO_REGIAO_GEOG_INTERM', 'NO_REGIAO_GEOG_INTERM']].drop_duplicates()
    combinado = (
        pd.concat([referencia_atual, novos], ignore_index=True)
        .drop_duplicates(subset=['CO_MUNICIPIO'])
    )
    os.makedirs(os.path.dirname(REGIAO_INTERM_REF_PATH), exist_ok=True)
    combinado.to_csv(REGIAO_INTERM_REF_PATH, index=False)


def _aplicar_referencia_regiao(df: pd.DataFrame, colunas_no_arquivo: set) -> pd.DataFrame:
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
        print(f"AVISO: {faltando} registro(s) deste ano com municipio nao encontrado na tabela de referencia.")
    return df


def transformar_dados(ano: int) -> pd.DataFrame:
    path = f'{BRONZE_DIR}/{NOME_ARQUIVO_PADRAO.format(ano=ano)}'

    colunas_disponiveis = _colunas_disponiveis(path)
    colunas_usar = [c for c in COLUNAS_ESSENCIAIS if c in colunas_disponiveis]
    colunas_faltando = set(COLUNAS_ESSENCIAIS) - set(colunas_usar)

    avisar = colunas_faltando - COLUNAS_COM_FALLBACK
    if avisar:
        print(f"AVISO: ano {ano} nao possui as colunas {sorted(avisar)}. Prosseguindo sem elas.")

    chunks = pd.read_csv(
        path, sep=';', encoding='latin1', chunksize=CHUNK_SIZE, usecols=colunas_usar,
    )

    lista_processada = []
    for chunk in chunks:
        chunk = _aplicar_excecoes_de_nome(chunk, ano)
        df_mg = chunk[chunk['CO_UF'] == CO_UF_MG].copy()
        df_mg = df_mg[df_mg['TP_SITUACAO_FUNCIONAMENTO'] == 1]
        lista_processada.append(df_mg)

    df_final = pd.concat(lista_processada, ignore_index=True)
    df_final = df_final.drop_duplicates(subset=['CO_ENTIDADE'])
    df_final = _aplicar_referencia_regiao(df_final, colunas_disponiveis)

    for col in COLUNAS_MATRICULA_PREENCHER_ZERO:
        if col in df_final.columns:
            df_final[col] = df_final[col].fillna(0)

    df_final = df_final.rename(columns=MAPEAMENTO_COLUNAS)
    df_final = _downcast_numericos(df_final)
    df_final = _categorizar_baixa_cardinalidade(df_final)
    df_final['ano'] = np.int16(ano)

    return df_final