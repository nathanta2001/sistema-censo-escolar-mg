"""
Modulo de analise estatistica (Secao 3.8 do TCC).

Consome a tabela municipal consolidada gerada por
`src.analysis.gerar_indicadores_ouro()` e produz:

1. Teste de normalidade (assimetria/curtose + Kolmogorov-Smirnov) para
   decidir entre Pearson e Spearman, por variavel (Secao 3.8.1).
2. Matriz de correlacao dos 9 pares da Tabela 5, com IC 95% via bootstrap
   (1000 reamostras) e correcao de Bonferroni (Secao 3.8.1).
3. Regressao linear simples de cada preditor estatisticamente significativo
   sobre I_rec, com checagem de necessidade de transformacao logit
   (Secao 3.8.2).
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Os 9 pares de variaveis da Tabela 5 do TCC
# ---------------------------------------------------------------------------
PARES_CORRELACAO = [
    # (nome_exibicao, coluna_x, coluna_y, hipotese)
    ('% escolas com internet p/ aprendizagem (2019)', 'pct_internet_2019', 'I_imp',
     'Conectividade voltada ao ensino reduziu a queda de matriculas.'),
    ('% escolas com banda larga (2019)', 'pct_banda_2019', 'I_imp',
     'Melhor qualidade de conexao reduziu o impacto.'),
    ('ICT municipal (2019)', 'ICT_2019', 'I_imp',
     'Maior capacidade tecnologica agregada reduziu o impacto.'),
    ('I_EAD (2021)', 'I_EAD', 'I_imp',
     'Municipios com maior adocao de EAD perderam menos matriculas.'),
    ('% escolas urbanas (2019)', 'pct_urbana_2019', 'I_imp',
     'Localizacao urbana funcionou como fator de protecao.'),
    ('ICT municipal (2024)', 'ICT_2024', 'I_rec',
     'Maior infraestrutura tecnologica acelerou a recuperacao.'),
    ('% escolas com banda larga (2024)', 'pct_banda_2024', 'I_rec',
     'Conectividade de qualidade favoreceu a recuperacao.'),
    ('% escolas estaduais (2024)', 'pct_estadual_2024', 'I_rec',
     'Dependencia administrativa influenciou o ritmo de recuperacao.'),
    ('% escolas urbanas (2024)', 'pct_urbana_2024', 'I_rec',
     'Localizacao urbana favoreceu a retomada mais rapida.'),
]

N_TESTES = len(PARES_CORRELACAO)  # 9
ALPHA = 0.05
ALPHA_BONFERRONI = ALPHA / N_TESTES  # ~0.006

N_REAMOSTRAS_BOOTSTRAP = 1000
SEED_BOOTSTRAP = 42  # fixa a semente para reprodutibilidade (Secao 3.11)


# ---------------------------------------------------------------------------
# 3.8.1 - Teste de normalidade e escolha do metodo de correlacao
# ---------------------------------------------------------------------------
def testar_normalidade(serie: pd.Series) -> dict:
    """Avalia a normalidade de uma variavel combinando tres criterios
    (Secao 3.8.1): assimetria, curtose e o teste de Kolmogorov-Smirnov
    (usado apenas como referencia complementar, sem peso decisorio isolado).

    Considera-se aceitavel para uso do coeficiente de Pearson quando a
    assimetria esta entre -1 e 1 E a curtose (excesso) esta entre -2 e 2.
    """
    valores = serie.dropna().astype(float)

    assimetria = float(scipy_stats.skew(valores))
    curtose = float(scipy_stats.kurtosis(valores))  # curtose de Fisher (excesso)

    valores_padronizados = (valores - valores.mean()) / valores.std(ddof=0)
    ks_estat, ks_p = scipy_stats.kstest(valores_padronizados, 'norm')

    normal_por_forma = (-1 <= assimetria <= 1) and (-2 <= curtose <= 2)

    return {
        'n': int(valores.shape[0]),
        'assimetria': assimetria,
        'curtose': curtose,
        'ks_estatistica': float(ks_estat),
        'ks_p_valor': float(ks_p),
        'normal': normal_por_forma,
    }


def escolher_metodo_correlacao(x: pd.Series, y: pd.Series) -> str:
    """Retorna 'pearson' se AMBAS as variaveis atendem ao criterio de
    normalidade (Secao 3.8.1); caso contrario, 'spearman'."""
    normal_x = testar_normalidade(x)['normal']
    normal_y = testar_normalidade(y)['normal']
    return 'pearson' if (normal_x and normal_y) else 'spearman'


# ---------------------------------------------------------------------------
# Bootstrap do intervalo de confianca de 95% do coeficiente de correlacao
# ---------------------------------------------------------------------------
def bootstrap_ic_correlacao(
    x: np.ndarray,
    y: np.ndarray,
    metodo: str,
    n_reamostras: int = N_REAMOSTRAS_BOOTSTRAP,
    ic: float = 95.0,
    seed: int = SEED_BOOTSTRAP,
) -> tuple:
    """Calcula o IC do coeficiente de correlacao via bootstrap (reamostragem
    com reposicao de pares (x,y), Secao 3.8.1)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    coef_fn = scipy_stats.pearsonr if metodo == 'pearson' else scipy_stats.spearmanr

    coeficientes_bootstrap = np.empty(n_reamostras)
    for i in range(n_reamostras):
        indices = rng.integers(0, n, size=n)
        coeficientes_bootstrap[i] = coef_fn(x[indices], y[indices])[0]

    cauda = (100 - ic) / 2
    ic_inferior = np.percentile(coeficientes_bootstrap, cauda)
    ic_superior = np.percentile(coeficientes_bootstrap, 100 - cauda)
    return float(ic_inferior), float(ic_superior)


def interpretar_forca_pearson(r: float) -> str:
    """Escala de Dancey & Reidy (2006), Tabela 1 do TCC."""
    r_abs = abs(r)
    if r_abs >= 0.90:
        return 'Correlacao muito forte'
    if r_abs >= 0.70:
        return 'Correlacao forte'
    if r_abs >= 0.40:
        return 'Correlacao moderada'
    if r_abs >= 0.10:
        return 'Correlacao fraca'
    return 'Correlacao desprezivel'


# ---------------------------------------------------------------------------
# Matriz de correlacoes dos 9 pares (Tabela 5), com Bonferroni
# ---------------------------------------------------------------------------
def calcular_matriz_correlacoes(df: pd.DataFrame, pares=PARES_CORRELACAO) -> pd.DataFrame:
    resultados = []

    for nome_x, col_x, col_y, hipotese in pares:
        subconjunto = df[[col_x, col_y]].dropna()
        x = subconjunto[col_x].to_numpy(dtype=float)
        y = subconjunto[col_y].to_numpy(dtype=float)

        metodo = escolher_metodo_correlacao(subconjunto[col_x], subconjunto[col_y])
        coef_fn = scipy_stats.pearsonr if metodo == 'pearson' else scipy_stats.spearmanr
        r, p_valor = coef_fn(x, y)

        ic_inf, ic_sup = bootstrap_ic_correlacao(x, y, metodo)

        significativo_bonferroni = p_valor < ALPHA_BONFERRONI
        tendencia_marginal = (not significativo_bonferroni) and (p_valor < ALPHA)

        resultados.append({
            'preditor': nome_x,
            'variavel_x': col_x,
            'variavel_y': col_y,
            'hipotese': hipotese,
            'metodo': metodo,
            'n': len(x),
            'r': round(float(r), 4),
            'interpretacao_forca': interpretar_forca_pearson(r),
            'ic_95_inferior': round(ic_inf, 4),
            'ic_95_superior': round(ic_sup, 4),
            'p_valor': p_valor,
            'significativo_bonferroni': significativo_bonferroni,
            'tendencia_marginal': tendencia_marginal,
        })

    return pd.DataFrame(resultados)


# ---------------------------------------------------------------------------
# 3.8.2 - Regressao linear simples (Y = I_rec)
# ---------------------------------------------------------------------------
def _distribuicao_concentrada_nos_extremos(serie: pd.Series, limiar: float = 0.05) -> bool:
    """Heuristica para decidir se vale considerar a transformacao logit do
    I_rec (Secao 3.8.2): fracao de observacoes a menos de `limiar` (em escala
    0-1) dos extremos 0 ou 1."""
    proporcao = serie.dropna() / 100.0  # I_rec esta em escala percentual
    perto_do_zero = (proporcao <= limiar).mean()
    perto_do_um = (proporcao >= 1 - limiar).mean()
    return (perto_do_zero + perto_do_um) > 0.10  # >10% das observacoes nos extremos


def ajustar_regressao_linear(df: pd.DataFrame, coluna_x: str, coluna_y: str = 'I_rec') -> dict:
    """Ajusta Y = beta0 + beta1*X + erro via OLS (statsmodels), Eq. 3.6.

    Se a distribuicao de Y estiver concentrada perto dos extremos [0, 100],
    tambem ajusta o modelo com a transformacao logit de Y e reporta ambos,
    para decisao do orientador sobre qual reportar no texto final.
    """
    dados = df[[coluna_x, coluna_y]].dropna().rename(columns={coluna_x: 'x', coluna_y: 'y'})

    modelo = smf.ols('y ~ x', data=dados).fit()

    resultado = {
        'coluna_x': coluna_x,
        'coluna_y': coluna_y,
        'n': int(modelo.nobs),
        'beta0': float(modelo.params['Intercept']),
        'beta1': float(modelo.params['x']),
        'p_valor_beta1': float(modelo.pvalues['x']),
        'r2': float(modelo.rsquared),
        'r2_ajustado': float(modelo.rsquared_adj),
        'significativo': float(modelo.pvalues['x']) < ALPHA,
        'concentracao_nos_extremos': _distribuicao_concentrada_nos_extremos(dados['y']),
        'modelo_logit': None,
    }

    if resultado['concentracao_nos_extremos']:
        # Recorte de seguranca para evitar logit(0) e logit(1) exatos
        p = (dados['y'] / 100.0).clip(lower=1e-4, upper=1 - 1e-4)
        dados_logit = dados.assign(y_logit=np.log(p / (1 - p)))
        modelo_logit = smf.ols('y_logit ~ x', data=dados_logit).fit()

        resultado['modelo_logit'] = {
            'beta0': float(modelo_logit.params['Intercept']),
            'beta1': float(modelo_logit.params['x']),
            'p_valor_beta1': float(modelo_logit.pvalues['x']),
            'r2': float(modelo_logit.rsquared),
        }

    return resultado


def ajustar_regressoes_significativas(df: pd.DataFrame, matriz_correlacoes: pd.DataFrame) -> pd.DataFrame:
    """Ajusta a regressao linear simples apenas para os preditores de I_rec
    que apresentaram correlacao estatisticamente significativa (Bonferroni),
    conforme a Secao 3.8.2."""
    preditores_rec_significativos = matriz_correlacoes[
        (matriz_correlacoes['variavel_y'] == 'I_rec')
        & (matriz_correlacoes['significativo_bonferroni'])
    ]

    resultados = [
        ajustar_regressao_linear(df, row['variavel_x'])
        for _, row in preditores_rec_significativos.iterrows()
    ]

    return pd.DataFrame(resultados)


# ---------------------------------------------------------------------------
# Orquestracao
# ---------------------------------------------------------------------------
def executar_analise_estatistica(df_indicadores: pd.DataFrame) -> dict:
    """Executa a analise estatistica completa (Secao 3.8) sobre a tabela
    municipal consolidada e retorna a matriz de correlacoes e as regressoes
    dos preditores significativos de I_rec."""
    matriz = calcular_matriz_correlacoes(df_indicadores)
    regressoes = ajustar_regressoes_significativas(df_indicadores, matriz)

    return {'correlacoes': matriz, 'regressoes': regressoes}


# ---------------------------------------------------------------------------
# Exportacao dos resultados para a camada Ouro (para consulta e redacao do
# TCC -- evita depender de numeros truncados no console)
# ---------------------------------------------------------------------------
def executar_e_salvar_analise_estatistica(df_indicadores):
    from src.load import salvar_camada_ouro
 
    resultado = executar_analise_estatistica(df_indicadores)
 
    salvar_camada_ouro(resultado['correlacoes'], 'matriz_correlacoes')
    salvar_camada_ouro(resultado['regressoes'], 'regressoes_i_rec')
 
    return resultado