import numpy as np, pandas as pd, statsmodels.formula.api as smf
df = pd.read_parquet('data/ouro/indicadores_municipios.parquet')
print(df.groupby(df['I_EAD'] > 0)['M2019'].median())

print("divisoria aq")

df = pd.read_parquet('data/ouro/indicadores_municipios.parquet')
d = df[['I_imp', 'I_EAD', 'M2019']].dropna().copy()
d['ead_pos'] = (d['I_EAD'] > 0).astype(int)
d['log_m2019'] = np.log(d['M2019'])
print(smf.ols('I_imp ~ ead_pos + log_m2019', d).fit().summary().tables[1])
print(d[d['M2019'] >= 5000].groupby('ead_pos')['I_imp'].agg(['count', 'median']))


print("divisoria aq")


df = pd.read_parquet('data/ouro/indicadores_municipios.parquet')
d = df.dropna(subset=['I_rec', 'M2019']).copy()
p = (d['I_rec'] / 100).clip(1e-4, 1 - 1e-4)
d['logit_rec'] = np.log(p / (1 - p))
d['log_m2019'] = np.log(d['M2019'])
d['rec_plena'] = (d['I_rec'] >= 100).astype(int)
for x in ['ICT_2024', 'pct_estadual_2024', 'pct_urbana_2024']:
    m0 = smf.ols(f'logit_rec ~ {x}', d).fit()
    m1 = smf.ols(f'logit_rec ~ {x} + log_m2019', d).fit()
    l1 = smf.logit(f'rec_plena ~ {x} + log_m2019', d).fit(disp=0)
    print(f"{x}: sem controle b={m0.params[x]:.3f} p={m0.pvalues[x]:.4f} | "
          f"com controle b={m1.params[x]:.3f} p={m1.pvalues[x]:.4f} | logística com controle p={l1.pvalues[x]:.4f}")
print(d[['ICT_2024', 'pct_estadual_2024', 'pct_urbana_2024', 'log_m2019']].corr(method='spearman')['log_m2019'].round(2))