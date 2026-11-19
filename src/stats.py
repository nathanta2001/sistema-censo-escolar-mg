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


# ---------------------------------------------------------------------------
# Descricao dos grupos com e sem perda (I_rec)
# ---------------------------------------------------------------------------
def _grupo_perda(df: pd.DataFrame) -> pd.Series:
    return df['teve_perda'].astype(bool).map({True: 'com_perda', False: 'sem_perda'})


def descritiva_i_rec_por_grupo(df: pd.DataFrame) -> pd.DataFrame:
    """Estatistica descritiva por grupo (com e sem perda entre 2019 e 2021).
    Para o grupo sem perda o I_rec e NaN por definicao; o grupo e descrito
    pelo porte (M2019), pelo I_imp e pela variacao 2021-2024 em %."""
    d = df.copy()
    d['grupo'] = _grupo_perda(d)
    d['var_2021_2024_pct'] = (d['M2024'] - d['M2021']) / d['M2021'] * 100
    linhas = []
    for grupo, g in d.groupby('grupo'):
        i_rec = g['I_rec'].dropna()
        linhas.append({
            'grupo': grupo, 'n': len(g), 'pct_municipios': len(g) / len(d) * 100,
            'n_I_rec_valido': len(i_rec),
            'media_I_rec': i_rec.mean(), 'desvio_padrao_I_rec': i_rec.std(),
            'min_I_rec': i_rec.min(), 'q1_I_rec': i_rec.quantile(0.25), 'mediana_I_rec': i_rec.median(),
            'q3_I_rec': i_rec.quantile(0.75), 'max_I_rec': i_rec.max(),
            'mediana_M2019': g['M2019'].median(), 'media_I_imp': g['I_imp'].mean(),
            'mediana_I_imp': g['I_imp'].median(), 'mediana_var_2021_2024_pct': g['var_2021_2024_pct'].median(),
        })
    return pd.DataFrame(linhas)


def trajetoria_municipios_com_perda(df: pd.DataFrame) -> pd.DataFrame:
    """Classifica os municipios com perda pela trajetoria 2021-2024:
    continuou caindo (I_rec < 0), sem variacao (I_rec = 0, M2024 = M2021),
    recuperacao parcial (0 < I_rec < 100) e recuperacao plena (I_rec >= 100,
    M2024 >= M2019)."""
    i_rec = restringir_a_municipios_com_perda(df)['I_rec'].dropna()
    categorias = [
        ('continuou_caindo', 'I_rec < 0', i_rec < 0),
        ('sem_variacao', 'I_rec = 0', i_rec == 0),
        ('recuperacao_parcial', '0 < I_rec < 100', (i_rec > 0) & (i_rec < 100)),
        ('recuperacao_plena', 'I_rec >= 100', i_rec >= 100),
    ]
    return pd.DataFrame([
        {'trajetoria': nome, 'criterio': criterio, 'n': int(mascara.sum()),
         'pct_com_perda': float(mascara.mean() * 100)}
        for nome, criterio, mascara in categorias
    ])


def descrever_municipios_sem_perda(df: pd.DataFrame) -> dict:
    """Contagem e proporcao dos municipios sem perda (M2021 >= M2019) e sua
    distribuicao por porte (quartis de M2019) e por regiao intermediaria.
    Inclui testes de diferenca de porte (Mann-Whitney) e de regiao
    (qui-quadrado) entre os grupos com e sem perda."""
    d = df.copy()
    sem = ~d['teve_perda'].astype(bool)
    com = ~sem
    mw = scipy_stats.mannwhitneyu(d.loc[sem, 'M2019'], d.loc[com, 'M2019'])
    tem_regiao = 'nome_regiao_interm' in d.columns
    chi2_p = (float(scipy_stats.chi2_contingency(pd.crosstab(d['nome_regiao_interm'], sem))[1])
              if tem_regiao else np.nan)
    resumo = pd.DataFrame([{
        'n_total': len(d), 'n_sem_perda': int(sem.sum()), 'pct_sem_perda': float(sem.mean() * 100),
        'n_M2021_igual_M2019': int((d['M2021'] == d['M2019']).sum()),
        'n_M2021_maior_M2019': int((d['M2021'] > d['M2019']).sum()),
        'n_com_perda': int(com.sum()), 'pct_com_perda': float(com.mean() * 100),
        'mediana_M2019_sem_perda': float(d.loc[sem, 'M2019'].median()),
        'mediana_M2019_com_perda': float(d.loc[com, 'M2019'].median()),
        'p_mann_whitney_M2019': float(mw.pvalue),
        'p_qui_quadrado_regiao': chi2_p,
    }])

    d['sem_perda'] = sem
    d['quartil_M2019'] = pd.qcut(d['M2019'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    por_porte = (
        d.groupby('quartil_M2019', observed=True)
        .agg(M2019_min=('M2019', 'min'), M2019_max=('M2019', 'max'),
             n_municipios=('M2019', 'size'), n_sem_perda=('sem_perda', 'sum'))
        .reset_index()
    )
    por_porte['pct_sem_perda'] = por_porte['n_sem_perda'] / por_porte['n_municipios'] * 100
    por_porte['quartil_M2019'] = por_porte['quartil_M2019'].astype(str)

    por_regiao = pd.DataFrame()
    if tem_regiao:
        por_regiao = (
            d.groupby(['cod_regiao_interm', 'nome_regiao_interm'], observed=True)
            .agg(n_municipios=('sem_perda', 'size'), n_sem_perda=('sem_perda', 'sum'))
            .reset_index()
        )
        por_regiao['pct_sem_perda'] = por_regiao['n_sem_perda'] / por_regiao['n_municipios'] * 100
        por_regiao = por_regiao.sort_values('pct_sem_perda').reset_index(drop=True)
    return {'resumo': resumo, 'por_porte': por_porte, 'por_regiao': por_regiao}


# ---------------------------------------------------------------------------
# Variantes do I_EAD e sensibilidade do I_ERE
# ---------------------------------------------------------------------------
def _spearman_com_ic(x: pd.Series, y: pd.Series) -> dict:
    xa, ya = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
    r, p = scipy_stats.spearmanr(xa, ya)
    ic_inf, ic_sup = bootstrap_ic_correlacao(xa, ya, 'spearman')
    return {'n': len(xa), 'r': float(r), 'ic_95_inferior': ic_inf, 'ic_95_superior': ic_sup, 'p_valor': float(p)}


def spearman_i_ead_variantes(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman do I_EAD com o I_imp em tres versoes: I_EAD continuo (todos),
    I_EAD binario (presenca de EAD, todos) e I_EAD continuo apenas entre os
    municipios com I_EAD > 0. Mostra quanto da correlacao vem do contraste
    entre zeros e nao zeros (a maioria dos municipios tem I_EAD = 0)."""
    d = df[['I_EAD', 'I_imp']].dropna()
    pos = d['I_EAD'] > 0
    variantes = [
        ('continuo_todos', d['I_EAD'], d['I_imp']),
        ('binario_presenca_ead', pos.astype(int), d['I_imp']),
        ('continuo_apenas_I_EAD_maior_0', d.loc[pos, 'I_EAD'], d.loc[pos, 'I_imp']),
    ]
    return pd.DataFrame([{'variante': nome, **_spearman_com_ic(x, y)} for nome, x, y in variantes])


def sensibilidade_i_ere_escolas(df: pd.DataFrame, min_escolas: int = 3) -> pd.DataFrame:
    """Spearman do I_ERE com o I_imp com todos os municipios com I_ERE e
    excluindo aqueles cuja media se baseia em 2 escolas ou menos
    (n_escolas_ere < 3)."""
    if not {'I_ERE', 'I_imp', 'n_escolas_ere'}.issubset(df.columns):
        return pd.DataFrame()
    d = df[['I_ERE', 'I_imp', 'n_escolas_ere']].dropna()
    filtrado = d[d['n_escolas_ere'] >= min_escolas]
    return pd.DataFrame([
        {'recorte': 'todos_com_I_ERE', 'municipios_excluidos': 0, **_spearman_com_ic(d['I_ERE'], d['I_imp'])},
        {'recorte': f'n_escolas_ere >= {min_escolas}', 'municipios_excluidos': len(d) - len(filtrado),
         **_spearman_com_ic(filtrado['I_ERE'], filtrado['I_imp'])},
    ])


def executar_analise_estatistica(df_indicadores: pd.DataFrame) -> dict:
    matriz = calcular_matriz_correlacoes(df_indicadores)
    regressoes = ajustar_regressoes_significativas(df_indicadores, matriz)
    return {
        'correlacoes': matriz,
        'regressoes': regressoes,
        'robustez_epsilon': robustez_epsilon_logit(df_indicadores),
        'logistica': regressao_logistica_recuperacao_plena(df_indicadores),
        'controle_porte': controle_porte(df_indicadores),
        'descritiva_por_grupo': descritiva_i_rec_por_grupo(df_indicadores),
        'trajetoria': trajetoria_municipios_com_perda(df_indicadores),
        'sem_perda': descrever_municipios_sem_perda(df_indicadores),
        'i_ead_variantes': spearman_i_ead_variantes(df_indicadores),
        'i_ere_escolas': sensibilidade_i_ere_escolas(df_indicadores),
    }


def executar_e_salvar_analise_estatistica(df_indicadores: pd.DataFrame) -> dict:
    from src.load import salvar_camada_ouro
    resultado = executar_analise_estatistica(df_indicadores)
    salvar_camada_ouro(resultado['correlacoes'], 'matriz_correlacoes')
    salvar_camada_ouro(resultado['regressoes'], 'regressoes_i_rec')
    salvar_camada_ouro(resultado['robustez_epsilon'], 'i_rec_robustez_epsilon')
    salvar_camada_ouro(resultado['logistica'], 'i_rec_logistica_recuperacao_plena')
    salvar_camada_ouro(resultado['controle_porte'], 'controle_porte')
    salvar_camada_ouro(resultado['descritiva_por_grupo'], 'i_rec_descritiva_por_grupo')
    salvar_camada_ouro(resultado['trajetoria'], 'i_rec_trajetoria_com_perda')
    salvar_camada_ouro(resultado['sem_perda']['resumo'], 'i_rec_sem_perda_resumo')
    salvar_camada_ouro(resultado['sem_perda']['por_porte'], 'i_rec_sem_perda_por_porte')
    salvar_camada_ouro(resultado['sem_perda']['por_regiao'], 'i_rec_sem_perda_por_regiao')
    salvar_camada_ouro(resultado['i_ead_variantes'], 'i_ead_spearman_variantes')
    if not resultado['i_ere_escolas'].empty:
        salvar_camada_ouro(resultado['i_ere_escolas'], 'sensibilidade_i_ere_escolas')
    return resultado