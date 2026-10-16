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

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

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


# ---------------------------------------------------------------------------
# Robustez da especificacao da regressao de I_rec (Secao 3.8.2)
#
# I_rec nao e limitado a [0, 100]: pode ser negativo ou superior a 100%. O
# logit so existe em (0, 1); por isso a transformacao e aplicada apos truncar
# I_rec/100 em [eps, 1 - eps]. Valores <= 0% ou >= 100% passam a ocupar os
# extremos +/- ln((1 - eps)/eps) (cerca de 9,21 para eps = 1e-4), e a
# magnitude de beta1 depende da escolha de eps. As duas analises abaixo
# verificam (i) se sinal e significancia sobrevivem a outros valores de eps e
# (ii) o que ocorre num desfecho binario, de interpretacao direta.
# ---------------------------------------------------------------------------
EPSILONS_LOGIT = [1e-2, 1e-3, 1e-4]
PREDITORES_I_REC = ['ICT_2024', 'pct_estadual_2024', 'pct_urbana_2024']


def robustez_epsilon_logit(df: pd.DataFrame, preditores=PREDITORES_I_REC, epsilons=EPSILONS_LOGIT) -> pd.DataFrame:
    """Reajusta a regressao OLS de logit(I_rec) para varios valores do
    recorte eps e reporta beta1, valor-p e R2 de cada um."""
    linhas = []
    for x in preditores:
        if x not in df.columns:
            continue
        d = df[[x, 'I_rec']].dropna()
        for eps in epsilons:
            p = (d['I_rec'] / 100.0).clip(lower=eps, upper=1 - eps)
            dados = pd.DataFrame({'x': d[x].to_numpy(), 'y': np.log(p / (1 - p)).to_numpy()})
            m = smf.ols('y ~ x', data=dados).fit()
            linhas.append({
                'preditor': x, 'epsilon': eps, 'n': int(m.nobs),
                'beta1': float(m.params['x']), 'p_valor': float(m.pvalues['x']), 'r2': float(m.rsquared),
            })
    return pd.DataFrame(linhas)


def regressao_logistica_recuperacao_plena(df: pd.DataFrame, preditores=PREDITORES_I_REC) -> pd.DataFrame:
    """Regressao logistica com desfecho binario: recuperacao plena ou
    superacao (I_rec >= 100%) contra as demais situacoes. Dispensa a
    transformacao de I_rec e tem interpretacao direta: a razao de chances
    (OR) e reportada por aumento de 1 desvio-padrao do preditor."""
    linhas = []
    for x in preditores:
        if x not in df.columns:
            continue
        d = df[[x, 'I_rec']].dropna()
        if d[x].std() == 0:
            continue
        dados = pd.DataFrame({
            'x': ((d[x] - d[x].mean()) / d[x].std()).to_numpy(),
            'y': (d['I_rec'] >= 100).astype(int).to_numpy(),
        })
        m = smf.logit('y ~ x', data=dados).fit(disp=0)
        ic = m.conf_int().loc['x']
        linhas.append({
            'preditor': x, 'n': int(m.nobs), 'pct_recuperacao_plena': float(dados['y'].mean() * 100),
            'odds_ratio_por_dp': float(np.exp(m.params['x'])),
            'or_ic95_inf': float(np.exp(ic[0])), 'or_ic95_sup': float(np.exp(ic[1])),
            'p_valor': float(m.pvalues['x']),
        })
    return pd.DataFrame(linhas)


def executar_analise_sensibilidade(df: pd.DataFrame) -> dict:
    """Executa a analise de sensibilidade completa (ambos os criterios) e
    retorna os resultados prontos para exportar (inclui a robustez da
    especificacao logit e a regressao logistica de recuperacao plena)."""
    return {
        'descritiva_m2019': estatistica_descritiva_por_limiar(df),
        'correlacoes_m2019': correlacoes_por_limiar(df),
        'descritiva_denominador': estatistica_descritiva_por_denominador(df),
        'correlacoes_denominador': correlacoes_por_denominador(df),
        'robustez_epsilon_logit': robustez_epsilon_logit(df),
        'logistica_recuperacao_plena': regressao_logistica_recuperacao_plena(df),
    }


def executar_e_salvar_analise_sensibilidade(df: pd.DataFrame) -> dict:
    from src.load import salvar_camada_ouro
    resultado = executar_analise_sensibilidade(df)
    for nome, tabela in resultado.items():
        salvar_camada_ouro(tabela, f'sensibilidade_i_rec_{nome}')
    return resultado