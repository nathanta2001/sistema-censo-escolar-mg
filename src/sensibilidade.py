"""
Analise de sensibilidade do I_rec a municipios de pequeno porte (Secao 3.8.3
/ 4.6 do TCC).

Motivacao: o Indicador de Recuperacao (Eq. 3.2) tem M2019-M2021 no
denominador. Em municipios com poucos alunos, uma pequena variacao absoluta
de matricula produz uma variacao percentual desproporcional (ex.: queda de
apenas 5 alunos gera denominador=5; qualquer recuperacao em 2024 e
multiplicada por 100/5=20), inflando o desvio-padrao do indicador e
potencialmente distorcendo correlacoes.

Este modulo recalcula a estatistica descritiva de I_rec e a matriz de
correlacoes dos preditores de I_rec, aplicando um limiar minimo de matricula
em 2019 (M2019) para exclusao de municipios pequenos, permitindo verificar
se os pares estatisticamente significativos permanecem significativos apos
essa exclusao -- ou seja, se o resultado e robusto a esse ruido, e nao um
artefato dos municipios menores.

IMPORTANTE: esta analise e um teste de ROBUSTEZ, reportado como evidencia
adicional (Secao 3.8.3/4.6). Ela NAO substitui o calculo principal do I_rec
sobre os 853 municipios, que permanece como definido na Secao 3.5.
"""

import pandas as pd

from src.stats import PARES_CORRELACAO, calcular_matriz_correlacoes

# Limiares de matricula em 2019 testados (proxy de "tamanho do municipio"),
# em ordem crescente de rigor. 0 = base completa (nenhuma exclusao).
LIMIARES_M2019 = [0, 200, 500, 1000, 2000]

# Limiares do DENOMINADOR da Eq. 3.2 (|M2019 - M2021|, em numero absoluto de
# alunos), testados separadamente. Ver nota abaixo sobre por que os dois
# criterios sao necessarios.
LIMIARES_DENOMINADOR = [0, 20, 50, 100, 200]

# Somente os preditores de I_rec sao afetados por esse ruido (I_imp nao usa
# o denominador problematico da Eq. 3.2)
PARES_I_REC = [p for p in PARES_CORRELACAO if p[2] == 'I_rec']


def _denominador_absoluto(df: pd.DataFrame) -> pd.Series:
    return (df['M2019'] - df['M2021']).abs()


def estatistica_descritiva_por_limiar(df: pd.DataFrame, limiares=LIMIARES_M2019) -> pd.DataFrame:
    """Para cada limiar de M2019 (tamanho do municipio), reporta quantos
    municipios restam e como a distribuicao de I_rec muda."""
    linhas = []
    for limiar in limiares:
        subconjunto = df[df['M2019'] >= limiar]['I_rec'].dropna()
        linhas.append({
            'limiar_M2019': limiar,
            'n_municipios': len(subconjunto),
            'municipios_excluidos': len(df) - len(subconjunto),
            'media_I_rec': subconjunto.mean(),
            'desvio_padrao_I_rec': subconjunto.std(),
            'min_I_rec': subconjunto.min(),
            'max_I_rec': subconjunto.max(),
            'mediana_I_rec': subconjunto.median(),
        })
    return pd.DataFrame(linhas)


def estatistica_descritiva_por_denominador(df: pd.DataFrame, limiares=LIMIARES_DENOMINADOR) -> pd.DataFrame:
    """Para cada limiar do DENOMINADOR |M2019-M2021| (o mecanismo real de
    amplificacao da Eq. 3.2), reporta quantos municipios restam e como a
    distribuicao de I_rec muda.

    Este e o criterio mecanisticamente correto: filtrar so por M2019 (tamanho
    do municipio) NAO garante excluir os casos de maior amplificacao, pois um
    municipio grande pode ter, por coincidencia, uma variacao 2019-2021 muito
    pequena em termos absolutos -- e o denominador pequeno resultante infla
    o indicador da mesma forma que em um municipio pequeno.
    """
    denom = _denominador_absoluto(df)
    linhas = []
    for limiar in limiares:
        subconjunto = df[denom >= limiar]['I_rec'].dropna()
        linhas.append({
            'limiar_denominador_abs': limiar,
            'n_municipios': len(subconjunto),
            'municipios_excluidos': len(df) - len(subconjunto),
            'media_I_rec': subconjunto.mean(),
            'desvio_padrao_I_rec': subconjunto.std(),
            'min_I_rec': subconjunto.min(),
            'max_I_rec': subconjunto.max(),
            'mediana_I_rec': subconjunto.median(),
        })
    return pd.DataFrame(linhas)


def correlacoes_por_limiar(df: pd.DataFrame, limiares=LIMIARES_M2019) -> pd.DataFrame:
    """Para cada limiar de M2019, recalcula a matriz de correlacao dos
    preditores de I_rec e reporta r, p-valor e significancia."""
    linhas = []
    for limiar in limiares:
        subconjunto = df[df['M2019'] >= limiar]
        matriz = calcular_matriz_correlacoes(subconjunto, pares=PARES_I_REC)
        matriz.insert(0, 'limiar_M2019', limiar)
        linhas.append(matriz)
    return pd.concat(linhas, ignore_index=True)


def correlacoes_por_denominador(df: pd.DataFrame, limiares=LIMIARES_DENOMINADOR) -> pd.DataFrame:
    """Para cada limiar do denominador |M2019-M2021|, recalcula a matriz de
    correlacao dos preditores de I_rec e reporta r, p-valor e significancia."""
    denom = _denominador_absoluto(df)
    linhas = []
    for limiar in limiares:
        subconjunto = df[denom >= limiar]
        matriz = calcular_matriz_correlacoes(subconjunto, pares=PARES_I_REC)
        matriz.insert(0, 'limiar_denominador_abs', limiar)
        linhas.append(matriz)
    return pd.concat(linhas, ignore_index=True)


def executar_analise_sensibilidade(df: pd.DataFrame) -> dict:
    """Executa a analise de sensibilidade completa (ambos os criterios) e
    retorna os quatro resultados prontos para exportar."""
    return {
        'descritiva_m2019': estatistica_descritiva_por_limiar(df),
        'correlacoes_m2019': correlacoes_por_limiar(df),
        'descritiva_denominador': estatistica_descritiva_por_denominador(df),
        'correlacoes_denominador': correlacoes_por_denominador(df),
    }


def executar_e_salvar_analise_sensibilidade(df: pd.DataFrame) -> dict:
    from src.load import salvar_camada_ouro
    resultado = executar_analise_sensibilidade(df)
    for nome, tabela in resultado.items():
        salvar_camada_ouro(tabela, f'sensibilidade_i_rec_{nome}')
    return resultado