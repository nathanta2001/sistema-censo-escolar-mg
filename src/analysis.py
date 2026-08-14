import os

import numpy as np
import pandas as pd

from src.config import (
    ANO_PANDEMIA,
    ANO_POS_PANDEMIA,
    ANO_PRE_PANDEMIA,
    ANOS,
    PRATA_DIR,
)
from src.load import salvar_camada_ouro
from src.transform import carregar_referencia_regiao

COLUNAS_MUNICIPIO = ['cod_municipio', 'nome_municipio']


def _carregar_prata(ano: int) -> pd.DataFrame:
    caminho = os.path.join(PRATA_DIR, f'censo_{ano}.parquet')
    return pd.read_parquet(caminho)


def _agregar_matriculas_municipio(df_escolas: pd.DataFrame) -> pd.DataFrame:
    colunas_soma = [c for c in ['total_matriculas', 'matriculas_ead'] if c in df_escolas.columns]
    return (
        df_escolas
        .groupby(COLUNAS_MUNICIPIO, observed=True)[colunas_soma]
        .sum()
        .reset_index()
    )


def anexar_regiao_intermediaria(df: pd.DataFrame) -> pd.DataFrame:
    referencia = carregar_referencia_regiao().rename(columns={'CO_MUNICIPIO': 'cod_municipio'})
    if referencia.empty:
        df['cod_regiao_interm'] = pd.NA
        df['nome_regiao_interm'] = pd.NA
        return df

    referencia = referencia.rename(columns={
        'CO_REGIAO_GEOG_INTERM': 'cod_regiao_interm',
        'NO_REGIAO_GEOG_INTERM': 'nome_regiao_interm',
    })
    return df.merge(referencia, on='cod_municipio', how='left')


def calcular_indicador_impacto() -> pd.DataFrame:
    df_2019 = _agregar_matriculas_municipio(_carregar_prata(ANO_PRE_PANDEMIA))
    df_2019 = df_2019.rename(columns={'total_matriculas': 'M2019'})[COLUNAS_MUNICIPIO + ['M2019']]

    df_2021 = _agregar_matriculas_municipio(_carregar_prata(ANO_PANDEMIA))
    df_2021 = df_2021.rename(columns={'total_matriculas': 'M2021'})[['cod_municipio', 'M2021']]

    base = df_2019.merge(df_2021, on='cod_municipio', how='inner')
    base['I_imp'] = ((base['M2021'] - base['M2019']) / base['M2019']) * 100
    return base


def calcular_indicador_recuperacao() -> pd.DataFrame:
    df_2019 = _agregar_matriculas_municipio(_carregar_prata(ANO_PRE_PANDEMIA))
    df_2019 = df_2019.rename(columns={'total_matriculas': 'M2019'})[COLUNAS_MUNICIPIO + ['M2019']]

    df_2021 = _agregar_matriculas_municipio(_carregar_prata(ANO_PANDEMIA))
    df_2021 = df_2021.rename(columns={'total_matriculas': 'M2021'})[['cod_municipio', 'M2021']]

    df_2024 = _agregar_matriculas_municipio(_carregar_prata(ANO_POS_PANDEMIA))
    df_2024 = df_2024.rename(columns={'total_matriculas': 'M2024'})[['cod_municipio', 'M2024']]

    base = df_2019.merge(df_2021, on='cod_municipio', how='inner')
    base = base.merge(df_2024, on='cod_municipio', how='inner')

    denominador = base['M2019'] - base['M2021']
    with np.errstate(divide='ignore', invalid='ignore'):
        i_rec_bruto = ((base['M2024'] - base['M2021']) / denominador) * 100

    base['I_rec'] = np.where(denominador <= 0, 100.0, i_rec_bruto)
    return base


def calcular_indicador_ead(ano: int = ANO_PANDEMIA) -> pd.DataFrame:
    agregado = _agregar_matriculas_municipio(_carregar_prata(ano))
    with np.errstate(divide='ignore', invalid='ignore'):
        i_ead = (agregado['matriculas_ead'] / agregado['total_matriculas']) * 100
    agregado['I_EAD'] = np.where(agregado['total_matriculas'] == 0, np.nan, i_ead)
    agregado['ano_referencia_ead'] = ano
    return agregado[COLUNAS_MUNICIPIO + ['I_EAD', 'ano_referencia_ead']]


def calcular_ict(ano: int) -> pd.DataFrame:
    df = _carregar_prata(ano)

    colunas_binarias = [
        'internet_aprendizagem', 'banda_larga',
        'IN_DESKTOP_ALUNO', 'IN_COMP_PORTATIL_ALUNO', 'IN_TABLET_ALUNO',
    ]
    df = df.copy()
    for col in colunas_binarias:
        df[col] = df[col].fillna(0)

    df['equip'] = (
        (df['IN_DESKTOP_ALUNO'] == 1)
        | (df['IN_COMP_PORTATIL_ALUNO'] == 1)
        | (df['IN_TABLET_ALUNO'] == 1)
    ).astype(float)

    df['urbana'] = (df['localizacao'] == 1).astype(float)
    df['estadual'] = (df['dependencia_administrativa'] == 2).astype(float)

    agregado = (
        df.groupby(COLUNAS_MUNICIPIO, observed=True)
        .agg(
            p_internet=('internet_aprendizagem', 'mean'),
            p_banda=('banda_larga', 'mean'),
            p_equip=('equip', 'mean'),
            p_urbana=('urbana', 'mean'),
            p_estadual=('estadual', 'mean'),
        )
        .reset_index()
    )

    agregado['ICT'] = agregado[['p_internet', 'p_banda', 'p_equip']].mean(axis=1)
    agregado['ano'] = ano
    return agregado


def calcular_ict_serie(anos=ANOS) -> pd.DataFrame:
    return pd.concat([calcular_ict(ano) for ano in anos], ignore_index=True)


def calcular_serie_matriculas_regiao(anos=ANOS) -> pd.DataFrame:
    partes = []
    for ano in anos:
        agregado_municipio = _agregar_matriculas_municipio(_carregar_prata(ano))
        agregado_municipio = anexar_regiao_intermediaria(agregado_municipio)
        agregado_regiao = (
            agregado_municipio
            .groupby(['cod_regiao_interm', 'nome_regiao_interm'], observed=True, dropna=False)['total_matriculas']
            .sum()
            .reset_index()
        )
        agregado_regiao['ano'] = ano
        partes.append(agregado_regiao)
    return pd.concat(partes, ignore_index=True)


def _anexar_indicador_ere(base: pd.DataFrame) -> pd.DataFrame:
    """Anexa o I_ERE (dias medios de mediacao remota, anos iniciais, 2021),
    construido a partir da Pesquisa Resposta Educacional a Pandemia
    (src/pesquisa_pandemia.py). Import feito aqui dentro (nao no topo do
    arquivo) para que o restante do pipeline continue funcionando mesmo se
    o arquivo externo da pesquisa ainda nao tiver sido colocado em
    data/externos/. Nesse caso, I_ERE fica ausente com um aviso, em vez de
    quebrar o calculo dos demais indicadores.
    """
    try:
        from src.pesquisa_pandemia import calcular_indicador_ere
        i_ere = calcular_indicador_ere()
        return base.merge(i_ere, on='cod_municipio', how='left')
    except FileNotFoundError:
        print(
            "AVISO: arquivo da Pesquisa Resposta Educacional a Pandemia nao encontrado "
            "em data/externos/ -- I_ERE nao sera calculado nesta execucao."
        )
        base['I_ERE'] = pd.NA
        return base


def gerar_indicadores_ouro() -> pd.DataFrame:
    impacto = calcular_indicador_impacto()
    recuperacao = calcular_indicador_recuperacao()[['cod_municipio', 'M2024', 'I_rec']]
    ead = calcular_indicador_ead(ANO_PANDEMIA)[['cod_municipio', 'I_EAD']]

    infra_2019 = calcular_ict(ANO_PRE_PANDEMIA)[
        ['cod_municipio', 'p_internet', 'p_banda', 'p_urbana', 'ICT']
    ].rename(columns={
        'p_internet': 'pct_internet_2019', 'p_banda': 'pct_banda_2019',
        'p_urbana': 'pct_urbana_2019', 'ICT': 'ICT_2019',
    })

    infra_2024 = calcular_ict(ANO_POS_PANDEMIA)[
        ['cod_municipio', 'p_banda', 'p_urbana', 'p_estadual', 'ICT']
    ].rename(columns={
        'p_banda': 'pct_banda_2024', 'p_urbana': 'pct_urbana_2024',
        'p_estadual': 'pct_estadual_2024', 'ICT': 'ICT_2024',
    })

    base = impacto.merge(recuperacao, on='cod_municipio', how='left')
    base = base.merge(ead, on='cod_municipio', how='left')
    base = base.merge(infra_2019, on='cod_municipio', how='left')
    base = base.merge(infra_2024, on='cod_municipio', how='left')

    base = _anexar_indicador_ere(base)
    base = anexar_regiao_intermediaria(base)

    salvar_camada_ouro(base, 'indicadores_municipios')

    return base