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

# Chave de agrupamento municipal. IMPORTANTE: nao inclui cod_regiao_interm/
# nome_regiao_interm aqui -- essas colunas podem vir NaN em anos que o INEP
# nao disponibiliza a classificacao (ex.: 2018-2022 no Censo Escolar real),
# e o groupby() do pandas DESCARTA por padrao qualquer linha cuja chave de
# agrupamento seja NaN. Se a regiao entrasse na chave, um ano sem essa coluna
# perderia TODOS os seus municipios na agregacao -- foi exatamente o bug que
# gerava I_imp/I_rec vazios. A regiao intermediaria e anexada separadamente,
# a partir da tabela de referencia (sempre completa apos um ano com a coluna
# ser processado), soh no resultado final.
COLUNAS_MUNICIPIO = ['cod_municipio', 'nome_municipio']


def _carregar_prata(ano: int) -> pd.DataFrame:
    """Le o parquet de um ano da camada Prata."""
    caminho = os.path.join(PRATA_DIR, f'censo_{ano}.parquet')
    return pd.read_parquet(caminho)


def _agregar_matriculas_municipio(df_escolas: pd.DataFrame) -> pd.DataFrame:
    """Agrega, a partir dos dados de escolas de UM ano, o total de
    matriculas por municipio (soma). Agrupa apenas por cod_municipio/
    nome_municipio -- ver comentario de COLUNAS_MUNICIPIO acima."""
    colunas_soma = [c for c in ['total_matriculas', 'matriculas_ead'] if c in df_escolas.columns]
    return (
        df_escolas
        .groupby(COLUNAS_MUNICIPIO, observed=True)[colunas_soma]
        .sum()
        .reset_index()
    )


def anexar_regiao_intermediaria(df: pd.DataFrame) -> pd.DataFrame:
    """Anexa cod_regiao_interm/nome_regiao_interm a um DataFrame municipal,
    a partir da tabela de referencia persistente (ver transform.py). Usar
    isso em vez de carregar a regiao a partir dos dados de um ano especifico,
    que pode nao ter essa coluna."""
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


# ---------------------------------------------------------------------------
# I_imp - Indicador de Impacto (Eq. 3.1)
# ---------------------------------------------------------------------------
def calcular_indicador_impacto() -> pd.DataFrame:
    """I_imp = ((M2021 - M2019) / M2019) * 100, por municipio."""
    df_2019 = _agregar_matriculas_municipio(_carregar_prata(ANO_PRE_PANDEMIA))
    df_2019 = df_2019.rename(columns={'total_matriculas': 'M2019'})[COLUNAS_MUNICIPIO + ['M2019']]

    df_2021 = _agregar_matriculas_municipio(_carregar_prata(ANO_PANDEMIA))
    df_2021 = df_2021.rename(columns={'total_matriculas': 'M2021'})[['cod_municipio', 'M2021']]

    base = df_2019.merge(df_2021, on='cod_municipio', how='inner')
    base['I_imp'] = ((base['M2021'] - base['M2019']) / base['M2019']) * 100
    return base


# ---------------------------------------------------------------------------
# I_rec - Indicador de Recuperacao (Eq. 3.2)
# ---------------------------------------------------------------------------
def calcular_indicador_recuperacao() -> pd.DataFrame:
    """I_rec = ((M2024 - M2021) / (M2019 - M2021)) * 100, por municipio.

    Regra de excecao (Secao 3.5): quando M2021 >= M2019 o denominador fica
    <= 0 (nao houve perda a recuperar). Nesses casos I_rec e definido
    deterministicamente como 100% via np.where(), em vez de gerar divisao
    por zero/NaN, e o municipio e mantido na analise como grupo de controle.
    """
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


# ---------------------------------------------------------------------------
# I_EAD - Indicador de Intensidade de Mediacao Pedagogica Remota (Eq. 3.3)
# ---------------------------------------------------------------------------
def calcular_indicador_ead(ano: int = ANO_PANDEMIA) -> pd.DataFrame:
    """I_EAD = (QT_MAT_BAS_EAD / QT_MAT_BAS) * 100, por municipio, para um
    ano especifico. Default = ano de maior restricao sanitaria (2021),
    usado como referencia na Tabela 5 do TCC.

    Municipios sem nenhuma matricula no ano (total_matriculas == 0) recebem
    I_EAD = NaN, pois a proporcao nao e definida nesse caso.
    """
    agregado = _agregar_matriculas_municipio(_carregar_prata(ano))

    with np.errstate(divide='ignore', invalid='ignore'):
        i_ead = (agregado['matriculas_ead'] / agregado['total_matriculas']) * 100

    agregado['I_EAD'] = np.where(agregado['total_matriculas'] == 0, np.nan, i_ead)
    agregado['ano_referencia_ead'] = ano
    return agregado[COLUNAS_MUNICIPIO + ['I_EAD', 'ano_referencia_ead']]


# ---------------------------------------------------------------------------
# ICT - Indice de Capacidade Tecnologica (Eq. 3.4 e 3.5)
# ---------------------------------------------------------------------------
def calcular_ict(ano: int) -> pd.DataFrame:
    """ICT_m = (p_internet_m + p_banda_m + p_equip_m) / 3, por municipio,
    para um ano especifico.

    equip_e = 1 se a escola tem desktop, notebook OU tablet para alunos.

    Tambem calcula, no mesmo agrupamento por municipio, dois outros
    preditores estruturais usados na Tabela 5 do TCC (correlacao/regressao):
    pct_urbana (% de escolas localizadas em area urbana, TP_LOCALIZACAO==1)
    e pct_estadual (% de escolas da rede estadual, TP_DEPENDENCIA==2).

    DECISAO DE PROJETO: a formula do TCC define p_x_m como
    soma(variavel binaria) / N_m, onde N_m e o total de escolas ativas do
    municipio -- ou seja, o denominador e sempre N_m, independente de haver
    valores ausentes na variavel binaria. Por isso, valores ausentes (NaN)
    nas colunas binarias de infraestrutura sao tratados aqui como 0 (escola
    sem a caracteristica) antes da agregacao, e nao excluidos do
    denominador. Se preferir tratar ausencia de dado como informacao
    desconhecida (excluir do denominador em vez de contar como 0), avalie
    com o orientador antes de usar este resultado nas correlacoes.
    """
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

    # TP_LOCALIZACAO: 1 = Urbana, 2 = Rural (dicionario de dados INEP)
    df['urbana'] = (df['localizacao'] == 1).astype(float)
    # TP_DEPENDENCIA: 1 = Federal, 2 = Estadual, 3 = Municipal, 4 = Privada
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
    """Calcula o ICT municipal para todos os anos da serie (2017-2024),
    permitindo avaliar a evolucao da infraestrutura tecnologica municipal
    ao longo do periodo (Secao 3.5)."""
    return pd.concat([calcular_ict(ano) for ano in anos], ignore_index=True)


# ---------------------------------------------------------------------------
# Serie de matriculas por regiao intermediaria (apoio a Secao 3.10 - graficos
# de linha da evolucao temporal por regiao, 2017-2024)
# ---------------------------------------------------------------------------
def calcular_serie_matriculas_regiao(anos=ANOS) -> pd.DataFrame:
    """Retorna total de matriculas por regiao intermediaria, para cada ano
    da serie temporal. A regiao vem sempre da tabela de referencia (nao do
    ano em si), entao funciona mesmo para anos cujo CSV nao traz a coluna."""
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


# ---------------------------------------------------------------------------
# Orquestracao: gera a tabela consolidada de indicadores municipais e salva
# na camada Ouro
# ---------------------------------------------------------------------------
def gerar_indicadores_ouro() -> pd.DataFrame:
    """Monta a tabela municipal consolidada com os 4 indicadores e os 9
    preditores estruturais da Tabela 5 do TCC, pronta para o modulo de
    estatistica (src/stats.py).

    Referencia temporal dos preditores: os preditores do impacto (I_imp) usam
    a infraestrutura de 2019 (baseline pre-pandemia, testando se amorteceu a
    queda); os preditores da recuperacao (I_rec) usam a infraestrutura de
    2024 (mais proxima do periodo de retomada que se busca explicar).
    """
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

    # Regiao intermediaria anexada por ultimo, a partir da tabela de
    # referencia -- util para mapas/agregacao regional (Secao 3.10)
    base = anexar_regiao_intermediaria(base)

    salvar_camada_ouro(base, 'indicadores_municipios')

    return base