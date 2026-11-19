"""
Modulo de analise estatistica (Secao 3.8 do TCC).
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats

PARES_CORRELACAO = [
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
    ('I_ERE - dias de mediacao remota, anos iniciais (2021)', 'I_ERE', 'I_imp',
     'Municipios com maior intensidade de mediacao remota apresentaram menor '
     'impacto negativo nas matriculas durante a pandemia.'),
    ('ICT municipal (2024)', 'ICT_2024', 'I_rec',
     'Maior infraestrutura tecnologica acelerou a recuperacao.'),
    ('% escolas com banda larga (2024)', 'pct_banda_2024', 'I_rec',
     'Conectividade de qualidade favoreceu a recuperacao.'),
    ('% escolas estaduais (2024)', 'pct_estadual_2024', 'I_rec',
     'Dependencia administrativa influenciou o ritmo de recuperacao.'),
    ('% escolas urbanas (2024)', 'pct_urbana_2024', 'I_rec',
     'Localizacao urbana favoreceu a retomada mais rapida.'),
]

N_TESTES = len(PARES_CORRELACAO)  # 10 (era 9; +1 com o I_ERE)
ALPHA = 0.05
ALPHA_BONFERRONI = ALPHA / N_TESTES

N_REAMOSTRAS_BOOTSTRAP = 1000
SEED_BOOTSTRAP = 42

# Constante de truncamento do logit de I_rec (modelo principal) e valores
# testados na robustez da especificacao.
EPSILON_LOGIT = 1e-4
EPSILONS_LOGIT = [1e-2, 1e-3, 1e-4]
PREDITORES_I_REC = ['ICT_2024', 'pct_estadual_2024', 'pct_urbana_2024']


def restringir_a_municipios_com_perda(df: pd.DataFrame, coluna_y: str = 'I_rec') -> pd.DataFrame:
    """Toda analise que envolve o I_rec usa apenas os municipios com perda
    entre 2019 e 2021 (teve_perda == True). Os demais tem I_rec = NaN por
    definicao (src/analysis.py: calcular_i_rec) e sao descritos a parte.
    O filtro e explicito para nao depender apenas do dropna()."""
    if coluna_y == 'I_rec' and 'teve_perda' in df.columns:
        return df[df['teve_perda'].astype(bool)]
    return df


def logit_truncado(i_rec: pd.Series, eps: float = EPSILON_LOGIT) -> pd.Series:
    """logit(I_rec/100) com I_rec/100 truncado em [eps, 1 - eps]."""
    p = (i_rec / 100.0).clip(lower=eps, upper=1 - eps)
    return np.log(p / (1 - p))


def testar_normalidade(serie: pd.Series) -> dict:
    valores = serie.dropna().astype(float)
    assimetria = float(scipy_stats.skew(valores))
    curtose = float(scipy_stats.kurtosis(valores))
    valores_padronizados = (valores - valores.mean()) / valores.std(ddof=0)
    ks_estat, ks_p = scipy_stats.kstest(valores_padronizados, 'norm')
    # Criterio estrito (|assimetria| < 1 e |curtose| < 2), igual ao descrito na
    # metodologia. Nenhuma variavel analisada cai exatamente no limite.
    normal_por_forma = (abs(assimetria) < 1) and (abs(curtose) < 2)
    return {
        'n': int(valores.shape[0]), 'assimetria': assimetria, 'curtose': curtose,
        'ks_estatistica': float(ks_estat), 'ks_p_valor': float(ks_p), 'normal': normal_por_forma,
    }


def escolher_metodo_correlacao(x: pd.Series, y: pd.Series) -> str:
    normal_x = testar_normalidade(x)['normal']
    normal_y = testar_normalidade(y)['normal']
    return 'pearson' if (normal_x and normal_y) else 'spearman'


def bootstrap_ic_correlacao(x, y, metodo, n_reamostras=N_REAMOSTRAS_BOOTSTRAP, ic=95.0, seed=SEED_BOOTSTRAP):
    rng = np.random.default_rng(seed)
    n = len(x)
    coef_fn = scipy_stats.pearsonr if metodo == 'pearson' else scipy_stats.spearmanr
    coeficientes_bootstrap = np.empty(n_reamostras)
    for i in range(n_reamostras):
        indices = rng.integers(0, n, size=n)
        coeficientes_bootstrap[i] = coef_fn(x[indices], y[indices])[0]
    cauda = (100 - ic) / 2
    return float(np.percentile(coeficientes_bootstrap, cauda)), float(np.percentile(coeficientes_bootstrap, 100 - cauda))


def interpretar_forca_pearson(r: float) -> str:
    r_abs = abs(r)
    if r_abs >= 0.90: return 'Correlacao muito forte'
    if r_abs >= 0.70: return 'Correlacao forte'
    if r_abs >= 0.40: return 'Correlacao moderada'
    if r_abs >= 0.10: return 'Correlacao fraca'
    return 'Correlacao desprezivel'


def calcular_matriz_correlacoes(df: pd.DataFrame, pares=PARES_CORRELACAO) -> pd.DataFrame:
    resultados = []
    for nome_x, col_x, col_y, hipotese in pares:
        if col_x not in df.columns:
            print(f"AVISO: coluna '{col_x}' ausente na tabela de indicadores -- par '{nome_x}' pulado.")
            continue

        subconjunto = restringir_a_municipios_com_perda(df, col_y)[[col_x, col_y]].dropna()
        if len(subconjunto) < 8:
            print(f"AVISO: apenas {len(subconjunto)} observacoes validas para '{nome_x}' -- par pulado (amostra insuficiente).")
            continue

        x = subconjunto[col_x].to_numpy(dtype=float)
        y = subconjunto[col_y].to_numpy(dtype=float)

        metodo = escolher_metodo_correlacao(subconjunto[col_x], subconjunto[col_y])
        coef_fn = scipy_stats.pearsonr if metodo == 'pearson' else scipy_stats.spearmanr
        r, p_valor = coef_fn(x, y)
        ic_inf, ic_sup = bootstrap_ic_correlacao(x, y, metodo)
        significativo_bonferroni = p_valor < ALPHA_BONFERRONI
        tendencia_marginal = (not significativo_bonferroni) and (p_valor < ALPHA)

        resultados.append({
            'preditor': nome_x, 'variavel_x': col_x, 'variavel_y': col_y, 'hipotese': hipotese,
            'metodo': metodo, 'n': len(x), 'r': round(float(r), 4),
            'interpretacao_forca': interpretar_forca_pearson(r),
            'ic_95_inferior': round(ic_inf, 4), 'ic_95_superior': round(ic_sup, 4),
            'p_valor': p_valor, 'significativo_bonferroni': significativo_bonferroni,
            'tendencia_marginal': tendencia_marginal,
        })

    return pd.DataFrame(resultados)


def _distribuicao_concentrada_nos_extremos(serie: pd.Series, limiar: float = 0.05) -> bool:
    proporcao = serie.dropna() / 100.0
    perto_do_zero = (proporcao <= limiar).mean()
    perto_do_um = (proporcao >= 1 - limiar).mean()
    return (perto_do_zero + perto_do_um) > 0.10


def ajustar_regressao_linear(df: pd.DataFrame, coluna_x: str, coluna_y: str = 'I_rec') -> dict:
    dados = (
        restringir_a_municipios_com_perda(df, coluna_y)[[coluna_x, coluna_y]]
        .dropna().rename(columns={coluna_x: 'x', coluna_y: 'y'})
    )
    modelo = smf.ols('y ~ x', data=dados).fit()

    resultado = {
        'coluna_x': coluna_x, 'coluna_y': coluna_y, 'n': int(modelo.nobs),
        'beta0': float(modelo.params['Intercept']), 'beta1': float(modelo.params['x']),
        'p_valor_beta1': float(modelo.pvalues['x']), 'r2': float(modelo.rsquared),
        'r2_ajustado': float(modelo.rsquared_adj),
        'significativo': float(modelo.pvalues['x']) < ALPHA,
        'concentracao_nos_extremos': _distribuicao_concentrada_nos_extremos(dados['y']),
        'modelo_logit': None,
    }

    if resultado['concentracao_nos_extremos']:
        dados_logit = dados.assign(y_logit=logit_truncado(dados['y']))
        modelo_logit = smf.ols('y_logit ~ x', data=dados_logit).fit()
        resultado['modelo_logit'] = {
            'beta0': float(modelo_logit.params['Intercept']), 'beta1': float(modelo_logit.params['x']),
            'p_valor_beta1': float(modelo_logit.pvalues['x']), 'r2': float(modelo_logit.rsquared),
        }

    return resultado


def ajustar_regressoes_significativas(df: pd.DataFrame, matriz_correlacoes: pd.DataFrame) -> pd.DataFrame:
    preditores_rec_significativos = matriz_correlacoes[
        (matriz_correlacoes['variavel_y'] == 'I_rec') & (matriz_correlacoes['significativo_bonferroni'])
    ]
    resultados = [
        ajustar_regressao_linear(df, row['variavel_x'])
        for _, row in preditores_rec_significativos.iterrows()
    ]
    return pd.DataFrame(resultados)


def robustez_epsilon_logit(df: pd.DataFrame, preditores=PREDITORES_I_REC, epsilons=EPSILONS_LOGIT) -> pd.DataFrame:
    """Reajusta a regressao OLS de logit(I_rec) para varios valores do
    recorte eps (apenas municipios com perda) e reporta beta1, valor-p e R2.

    I_rec nao e limitado a [0, 100]: valores <= 0% ou >= 100% ocupam os
    extremos +/- ln((1 - eps)/eps), e a magnitude de beta1 depende de eps.
    Interessa verificar se sinal e significancia se mantem."""
    base = restringir_a_municipios_com_perda(df)
    linhas = []
    for x in preditores:
        if x not in base.columns:
            continue
        d = base[[x, 'I_rec']].dropna()
        for eps in epsilons:
            dados = pd.DataFrame({'x': d[x].to_numpy(), 'y': logit_truncado(d['I_rec'], eps).to_numpy()})
            m = smf.ols('y ~ x', data=dados).fit()
            linhas.append({
                'preditor': x, 'epsilon': eps, 'n': int(m.nobs),
                'beta1': float(m.params['x']), 'p_valor': float(m.pvalues['x']), 'r2': float(m.rsquared),
            })
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# Recuperacao plena e controle pelo porte do municipio (apenas com perda)
# ---------------------------------------------------------------------------
PREDITORES_I_REC_TODOS = [p[1] for p in PARES_CORRELACAO if p[2] == 'I_rec']


def _base_porte(df: pd.DataFrame) -> pd.DataFrame:
    base = restringir_a_municipios_com_perda(df).copy()
    base['log_m2019'] = np.log(base['M2019'])
    return base


def regressao_logistica_recuperacao_plena(df: pd.DataFrame, preditores=PREDITORES_I_REC_TODOS) -> pd.DataFrame:
    """Regressao logistica de y = (I_rec >= 100), apenas nos municipios com
    perda, sem e com controle de log(M2019). O preditor e padronizado (z) na
    propria amostra, de modo que a razao de chances (OR) e por aumento de
    1 desvio-padrao do preditor."""
    base = _base_porte(df)
    linhas = []
    for x in preditores:
        if x not in base.columns:
            continue
        d = base[[x, 'I_rec', 'log_m2019']].dropna()
        if d[x].std() == 0:
            continue
        dados = pd.DataFrame({
            'z': ((d[x] - d[x].mean()) / d[x].std()).to_numpy(),
            'log_m2019': d['log_m2019'].to_numpy(),
            'y': (d['I_rec'] >= 100).astype(int).to_numpy(),
        })
        for modelo, formula in (('sem_controle', 'y ~ z'), ('com_controle_porte', 'y ~ z + log_m2019')):
            m = smf.logit(formula, data=dados).fit(disp=0)
            ic = m.conf_int().loc['z']
            linhas.append({
                'preditor': x, 'modelo': modelo, 'n': int(m.nobs),
                'n_recuperacao_plena': int(dados['y'].sum()),
                'pct_recuperacao_plena': float(dados['y'].mean() * 100),
                'odds_ratio_por_dp': float(np.exp(m.params['z'])),
                'or_ic95_inf': float(np.exp(ic.iloc[0])), 'or_ic95_sup': float(np.exp(ic.iloc[1])),
                'p_valor': float(m.pvalues['z']),
            })
    return pd.DataFrame(linhas)


def controle_porte(df: pd.DataFrame, preditores=PREDITORES_I_REC_TODOS) -> pd.DataFrame:
    """Verifica se as associacoes se mantem com o porte do municipio
    (log das matriculas de 2019) como controle.

    - I_rec (apenas com perda): OLS de logit(I_rec), eps = 1e-4, sem e com
      log(M2019), para cada preditor de I_rec.
    - I_imp (853 municipios): OLS de I_imp sobre a presenca de matricula EAD
      (I_EAD > 0), sem e com log(M2019).
    Reporta tambem o Spearman entre o preditor e log(M2019)."""
    linhas = []

    def _linha(desfecho, preditor, dados, y, x):
        m0 = smf.ols(f'{y} ~ {x}', data=dados).fit()
        m1 = smf.ols(f'{y} ~ {x} + log_m2019', data=dados).fit()
        ic = m1.conf_int().loc[x]
        rho, p_rho = scipy_stats.spearmanr(dados[x], dados['log_m2019'])
        return {
            'desfecho': desfecho, 'preditor': preditor, 'n': int(m1.nobs),
            'beta1_sem_controle': float(m0.params[x]), 'p_sem_controle': float(m0.pvalues[x]),
            'beta1_com_controle': float(m1.params[x]), 'p_com_controle': float(m1.pvalues[x]),
            'ic95_inf_com_controle': float(ic.iloc[0]), 'ic95_sup_com_controle': float(ic.iloc[1]),
            'beta_log_m2019': float(m1.params['log_m2019']), 'p_log_m2019': float(m1.pvalues['log_m2019']),
            'spearman_preditor_log_m2019': float(rho), 'p_spearman_preditor_log_m2019': float(p_rho),
        }

    base = _base_porte(df)
    for x in preditores:
        if x not in base.columns:
            continue
        d = base[[x, 'I_rec', 'log_m2019']].dropna().copy()
        d['logit_i_rec'] = logit_truncado(d['I_rec'])
        linhas.append(_linha('logit(I_rec)', x, d, 'logit_i_rec', x))

    if {'I_EAD', 'I_imp', 'M2019'}.issubset(df.columns):
        d = df[['I_EAD', 'I_imp', 'M2019']].dropna().copy()
        d['ead_presente'] = (d['I_EAD'] > 0).astype(int)
        d['log_m2019'] = np.log(d['M2019'])
        linhas.append(_linha('I_imp', 'I_EAD > 0 (presenca de EAD)', d, 'I_imp', 'ead_presente'))

    return pd.DataFrame(linhas)


def executar_analise_estatistica(df_indicadores: pd.DataFrame) -> dict:
    matriz = calcular_matriz_correlacoes(df_indicadores)
    regressoes = ajustar_regressoes_significativas(df_indicadores, matriz)
    return {
        'correlacoes': matriz,
        'regressoes': regressoes,
        'robustez_epsilon': robustez_epsilon_logit(df_indicadores),
        'logistica': regressao_logistica_recuperacao_plena(df_indicadores),
        'controle_porte': controle_porte(df_indicadores),
    }


def executar_e_salvar_analise_estatistica(df_indicadores: pd.DataFrame) -> dict:
    from src.load import salvar_camada_ouro
    resultado = executar_analise_estatistica(df_indicadores)
    salvar_camada_ouro(resultado['correlacoes'], 'matriz_correlacoes')
    salvar_camada_ouro(resultado['regressoes'], 'regressoes_i_rec')
    salvar_camada_ouro(resultado['robustez_epsilon'], 'i_rec_robustez_epsilon')
    salvar_camada_ouro(resultado['logistica'], 'i_rec_logistica_recuperacao_plena')
    salvar_camada_ouro(resultado['controle_porte'], 'controle_porte')
    return resultado