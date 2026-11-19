"""
Analise de sensibilidade do I_rec a perdas pequenas entre 2019 e 2021.

Motivacao: o Indicador de Recuperacao tem M2019 - M2021 no denominador.
Quando a perda e pequena em numero absoluto de alunos, qualquer variacao em
2024 e amplificada (ex.: perda de 5 alunos -> denominador = 5; cada aluno
recuperado vale 20 pontos percentuais), inflando o desvio-padrao do
indicador e potencialmente distorcendo correlacoes.

O I_rec so e definido para municipios com perda (teve_perda == True; ver
src/analysis.py: calcular_i_rec). Todas as analises abaixo partem desse
grupo. O criterio principal filtra pela perda real:

    (M2019 - M2021) >= corte, com cortes de 20, 50, 100 e 200 alunos.

Nao se usa o valor absoluto |M2019 - M2021|, que manteria municipios sem
perda (M2021 > M2019) no recorte. Um criterio alternativo, por matricula de
2019 (M2019, porte do municipio), e mantido para comparacao: filtrar so
pelo porte nao exclui os casos de maior amplificacao, porque um municipio
grande pode ter perda absoluta pequena.

Esta analise e um teste de robustez. Ela nao substitui o calculo principal
do I_rec sobre todos os municipios com perda.
"""

import pandas as pd

from src.stats import PARES_CORRELACAO, calcular_matriz_correlacoes

# Limiares de matricula em 2019 (porte do municipio). 0 = todos com perda.
LIMIARES_M2019 = [0, 200, 500, 1000, 2000]

# Cortes da perda real M2019 - M2021, em numero de alunos. 0 = todos com
# perda (linha de base, sem corte adicional).
LIMIARES_PERDA = [0, 20, 50, 100, 200]

PARES_I_REC = [p for p in PARES_CORRELACAO if p[2] == 'I_rec']


def perda_real(df: pd.DataFrame) -> pd.Series:
    """M2019 - M2021, com sinal (positivo = perda)."""
    return df['M2019'] - df['M2021']


def filtrar_por_perda_real(df: pd.DataFrame, corte: float) -> pd.DataFrame:
    """Municipios com perda e (M2019 - M2021) >= corte. Com corte = 0,
    devolve todos os municipios com perda (perda > 0)."""
    perda = perda_real(df)
    return df[(perda > 0) & (perda >= corte)]


def _municipios_com_perda(df: pd.DataFrame) -> pd.DataFrame:
    return filtrar_por_perda_real(df, 0)


def _descritiva(subconjunto: pd.Series, n_base: int) -> dict:
    return {
        'n_municipios': len(subconjunto),
        'municipios_excluidos': n_base - len(subconjunto),
        'media_I_rec': subconjunto.mean(),
        'desvio_padrao_I_rec': subconjunto.std(),
        'min_I_rec': subconjunto.min(),
        'max_I_rec': subconjunto.max(),
        'mediana_I_rec': subconjunto.median(),
    }


def estatistica_descritiva_por_limiar(df: pd.DataFrame, limiares=LIMIARES_M2019) -> pd.DataFrame:
    """Para cada limiar de M2019, entre os municipios com perda, reporta
    quantos restam e como a distribuicao de I_rec muda."""
    base = _municipios_com_perda(df)
    linhas = []
    for limiar in limiares:
        subconjunto = base[base['M2019'] >= limiar]['I_rec'].dropna()
        linhas.append({'limiar_M2019': limiar, **_descritiva(subconjunto, len(base))})
    return pd.DataFrame(linhas)


def estatistica_descritiva_por_perda(df: pd.DataFrame, limiares=LIMIARES_PERDA) -> pd.DataFrame:
    """Para cada corte da perda real (M2019 - M2021) >= corte, reporta
    quantos municipios restam e como a distribuicao de I_rec muda."""
    n_base = len(_municipios_com_perda(df))
    linhas = []
    for corte in limiares:
        subconjunto = filtrar_por_perda_real(df, corte)['I_rec'].dropna()
        linhas.append({'limiar_perda': corte, **_descritiva(subconjunto, n_base)})
    return pd.DataFrame(linhas)


def correlacoes_por_limiar(df: pd.DataFrame, limiares=LIMIARES_M2019) -> pd.DataFrame:
    """Para cada limiar de M2019 (entre os municipios com perda), recalcula
    as correlacoes dos preditores de I_rec: n, r, IC 95%, p e significancia."""
    base = _municipios_com_perda(df)
    linhas = []
    for limiar in limiares:
        matriz = calcular_matriz_correlacoes(base[base['M2019'] >= limiar], pares=PARES_I_REC)
        matriz.insert(0, 'limiar_M2019', limiar)
        linhas.append(matriz)
    return pd.concat(linhas, ignore_index=True)


def correlacoes_por_perda(df: pd.DataFrame, limiares=LIMIARES_PERDA) -> pd.DataFrame:
    """Para cada corte da perda real, recalcula as correlacoes dos
    preditores de I_rec (ICT, banda larga, % estadual e % urbana, 2024):
    n, r, IC 95% (bootstrap), p e significancia com Bonferroni."""
    linhas = []
    for corte in limiares:
        matriz = calcular_matriz_correlacoes(filtrar_por_perda_real(df, corte), pares=PARES_I_REC)
        matriz.insert(0, 'limiar_perda', corte)
        linhas.append(matriz)
    return pd.concat(linhas, ignore_index=True)


def executar_analise_sensibilidade(df: pd.DataFrame) -> dict:
    """Executa a analise de sensibilidade (perda real e porte). A robustez
    do eps do logit, a regressao logistica e o controle por porte ficam em
    src/stats.py."""
    return {
        'descritiva_m2019': estatistica_descritiva_por_limiar(df),
        'correlacoes_m2019': correlacoes_por_limiar(df),
        'descritiva_perda': estatistica_descritiva_por_perda(df),
        'correlacoes_perda': correlacoes_por_perda(df),
    }


def executar_e_salvar_analise_sensibilidade(df: pd.DataFrame) -> dict:
    from src.load import salvar_camada_ouro
    resultado = executar_analise_sensibilidade(df)
    for nome, tabela in resultado.items():
        salvar_camada_ouro(tabela, f'sensibilidade_i_rec_{nome}')
    return resultado
