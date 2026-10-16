"""
Modulo de visualizacao estatica (Secao 3.10 do TCC).
"""

import os
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

from src.config import OURO_DIR, SHAPEFILE_REGIOES_PATH

sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)

DIR_GRAFICOS = os.path.join(OURO_DIR, 'graficos')

PARES_IMP_PARA_PLOTAR = [
    ('I_EAD', 'I_imp', 'I_EAD (%)', 'I_imp (%)', 'Intensidade de matrícula EAD (2021) x Impacto'),
    ('I_ERE', 'I_imp', 'I_ERE (dias)', 'I_imp (%)', 'Dias de mediação remota (2021) x Impacto'),
]

PARES_REC_PARA_PLOTAR = [
    ('ICT_2024', 'I_rec', 'ICT municipal (2024)', 'logit(I_rec)', 'Capacidade tecnológica (2024) x Recuperação'),
    ('pct_estadual_2024', 'I_rec', '% escolas estaduais (2024)', 'logit(I_rec)', 'Dependência estadual (2024) x Recuperação'),
    ('pct_urbana_2024', 'I_rec', '% escolas urbanas (2024)', 'logit(I_rec)', 'Localização urbana (2024) x Recuperação'),
]


def _logit(serie_percentual: pd.Series) -> pd.Series:
    p = (serie_percentual / 100.0).clip(lower=1e-4, upper=1 - 1e-4)
    return np.log(p / (1 - p))


def plot_dispersao_imp(df, coluna_x, coluna_y, rotulo_x, rotulo_y, titulo, caminho_saida):
    """Dispersao + tendencia LOWESS (preditor continuo, ex.: I_ERE)."""
    dados = df[[coluna_x, coluna_y]].dropna()
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        sns.regplot(
            data=dados, x=coluna_x, y=coluna_y, ax=ax, lowess=True,
            scatter_kws={'alpha': 0.35, 's': 18, 'color': '#2c6e91'},
            line_kws={'color': '#c0392b', 'linewidth': 2},
        )
    ax.set_xlabel(rotulo_x)
    ax.set_ylabel(rotulo_y)
    ax.set_title(titulo, fontsize=11)
    fig.tight_layout()
    fig.savefig(caminho_saida, bbox_inches='tight')
    plt.close(fig)


def plot_dispersao_zero_inflada(df, coluna_x, rotulo_x, titulo, caminho_saida,
                                 coluna_y='I_imp', rotulo_y='I_imp (%)') -> dict:
    """Figura para preditores com muitos zeros (ex.: I_EAD, em que a maioria
    dos municipios nao registra matricula EAD).

    Com a massa de zeros, um unico LOWESS sobre todos os pontos e instavel e
    pode sugerir uma tendencia oposta a correlacao de Spearman reportada
    (que e dominada pelo contraste entre os grupos x = 0 e x > 0). Por isso a
    figura tem dois paineis: (a) comparacao de y entre os dois grupos e
    (b) dispersao apenas dos municipios com x > 0, com LOWESS.
    Retorna as medianas dos grupos, para citacao no texto.
    """
    dados = df[[coluna_x, coluna_y]].dropna()
    zero = dados[dados[coluna_x] == 0]
    pos = dados[dados[coluna_x] > 0]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9.5, 4.5), dpi=300, gridspec_kw={'width_ratios': [1, 1.7]}
    )

    rotulos = [f"{coluna_x} = 0\n(n = {len(zero)})", f"{coluna_x} > 0\n(n = {len(pos)})"]
    ax1.boxplot([zero[coluna_y], pos[coluna_y]], tick_labels=rotulos, widths=0.55,
                patch_artist=True, medianprops={'color': '#c0392b', 'linewidth': 2},
                boxprops={'facecolor': '#cfe3ee'}, flierprops={'markersize': 3, 'alpha': 0.4})
    ax1.set_ylabel(rotulo_y)
    ax1.set_title('(a) Comparação entre grupos', fontsize=10)

    ax2.scatter(pos[coluna_x], pos[coluna_y], alpha=0.45, s=20, color='#2c6e91')
    if len(pos) >= 15:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            sns.regplot(data=pos, x=coluna_x, y=coluna_y, ax=ax2, lowess=True,
                        scatter=False, line_kws={'color': '#c0392b', 'linewidth': 2})
    ax2.set_xlabel(rotulo_x)
    ax2.set_ylabel(rotulo_y)
    ax2.set_title(f'(b) Apenas municípios com {coluna_x} > 0', fontsize=10)

    fig.suptitle(titulo, fontsize=11)
    fig.tight_layout()
    fig.savefig(caminho_saida, bbox_inches='tight')
    plt.close(fig)

    return {'mediana_x0': float(zero[coluna_y].median()), 'mediana_xpos': float(pos[coluna_y].median()),
            'n_x0': len(zero), 'n_xpos': len(pos)}


def plot_dispersao_rec_logit(df, coluna_x, rotulo_x, titulo, caminho_saida, eps=1e-4) -> dict:
    """Dispersao de logit(I_rec) com a reta OLS ajustada (modelo da Secao 4.6).

    TODOS os municipios sao exibidos. Como I_rec nao e limitado a [0, 100]
    (pode ser negativo ou superior a 100%), a transformacao logit so e
    aplicada apos truncar I_rec/100 em [eps, 1-eps]. Os municipios com
    I_rec <= 0% e >= 100% ficam, portanto, nas faixas extremas (+/- ln((1-eps)/eps),
    cerca de 9,21 para eps = 1e-4); eles sao desenhados com cores proprias e
    pequena dispersao vertical, em vez de omitidos, para que a reta ajustada
    (que usa todos os pontos) seja compativel com o que se ve no grafico.
    """
    dados = df[[coluna_x, 'I_rec']].dropna().copy()
    dados['logit_I_rec'] = _logit(dados['I_rec'])

    modelo = smf.ols(f'logit_I_rec ~ {coluna_x}', data=dados).fit()
    beta0, beta1 = modelo.params['Intercept'], modelo.params[coluna_x]
    r2, p_valor = modelo.rsquared, modelo.pvalues[coluna_x]

    pct = dados['I_rec'] / 100.0
    topo = pct >= 1 - eps
    base = pct <= eps
    meio = ~(topo | base)
    limite = np.log((1 - eps) / eps)
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.35, 0.35, len(dados))

    fig, ax = plt.subplots(figsize=(6.6, 4.8), dpi=300)
    ax.scatter(dados.loc[meio, coluna_x], dados.loc[meio, 'logit_I_rec'], s=16, alpha=0.5,
               color='#2c6e91', label=f'0% < I_rec < 100% (n = {int(meio.sum())})')
    ax.scatter(dados.loc[topo, coluna_x], dados.loc[topo, 'logit_I_rec'] + jitter[topo.values],
               s=12, alpha=0.4, color='#2e8b57', label=f'I_rec ≥ 100% (n = {int(topo.sum())})')
    ax.scatter(dados.loc[base, coluna_x], dados.loc[base, 'logit_I_rec'] + jitter[base.values],
               s=12, alpha=0.4, color='#d9822b', label=f'I_rec ≤ 0% (n = {int(base.sum())})')

    x_linha = np.linspace(dados[coluna_x].min(), dados[coluna_x].max(), 100)
    ax.plot(x_linha, beta0 + beta1 * x_linha, color='#c0392b', linewidth=2, label='Reta ajustada')
    ax.set_ylim(-limite - 1, limite + 1)
    ax.set_xlabel(rotulo_x + "\n" + rf"$\beta_1$ = {beta1:.3f}   $R^2$ = {r2:.4f}   p = {p_valor:.1e}")
    ax.set_ylabel('logit(I_rec)')
    ax.set_title(titulo, fontsize=11)
    ax.legend(fontsize=7, loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(caminho_saida, bbox_inches='tight')
    plt.close(fig)

    return {'n_meio': int(meio.sum()), 'n_topo': int(topo.sum()), 'n_base': int(base.sum())}


def gerar_todas_as_dispersoes(df_indicadores, diretorio_saida=DIR_GRAFICOS):
    os.makedirs(diretorio_saida, exist_ok=True)
    caminhos = []
    for coluna_x, coluna_y, rotulo_x, rotulo_y, titulo in PARES_IMP_PARA_PLOTAR:
        if coluna_x not in df_indicadores.columns:
            continue
        caminho = os.path.join(diretorio_saida, f'dispersao_{coluna_x}_{coluna_y}.png')
        if coluna_x == 'I_EAD':
            info = plot_dispersao_zero_inflada(df_indicadores, coluna_x, rotulo_x, titulo, caminho)
            print(f"I_EAD: mediana de I_imp = {info['mediana_x0']:.2f}% (I_EAD = 0, n = {info['n_x0']}) "
                  f"vs {info['mediana_xpos']:.2f}% (I_EAD > 0, n = {info['n_xpos']})")
        else:
            plot_dispersao_imp(df_indicadores, coluna_x, coluna_y, rotulo_x, rotulo_y, titulo, caminho)
        caminhos.append(caminho)
    for coluna_x, coluna_y, rotulo_x, rotulo_y, titulo in PARES_REC_PARA_PLOTAR:
        if coluna_x not in df_indicadores.columns:
            continue
        caminho = os.path.join(diretorio_saida, f'dispersao_{coluna_x}_logit_I_rec.png')
        info = plot_dispersao_rec_logit(df_indicadores, coluna_x, rotulo_x, titulo, caminho)
        print(f"{coluna_x}: {info['n_topo']} municípios com I_rec ≥ 100%, "
              f"{info['n_base']} com I_rec ≤ 0%, {info['n_meio']} intermediários")
        caminhos.append(caminho)
    return caminhos


def plot_evolucao_ead(serie_dependencia, caminho_saida):
    """Evolucao da proporcao de matriculas em EAD (% do total de matriculas
    da educacao basica), por regiao intermediaria e para o estado (Objetivo
    Especifico 2 -- evolucao temporal da modalidade de ensino).

    serie_dependencia: tabela de matriculas por municipio/ano/dependencia
    com as colunas total_matriculas e matriculas_ead.
    """
    if 'matriculas_ead' not in serie_dependencia.columns:
        raise ValueError("A serie nao tem a coluna 'matriculas_ead'. Regenere a "
                         "serie de matriculas por dependencia (analysis.py).")
    dados = serie_dependencia.dropna(subset=['nome_regiao_interm']).copy()

    por_regiao = (dados.groupby(['ano', 'nome_regiao_interm'], observed=True)[['total_matriculas', 'matriculas_ead']]
                  .sum().reset_index())
    por_regiao['pct_ead'] = por_regiao['matriculas_ead'] / por_regiao['total_matriculas'] * 100
    estado = dados.groupby('ano')[['total_matriculas', 'matriculas_ead']].sum().reset_index()
    estado['pct_ead'] = estado['matriculas_ead'] / estado['total_matriculas'] * 100

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    paleta = sns.color_palette("husl", n_colors=por_regiao['nome_regiao_interm'].nunique())
    sns.lineplot(data=por_regiao, x='ano', y='pct_ead', hue='nome_regiao_interm',
                 palette=paleta, linewidth=1.1, marker='o', markersize=3, ax=ax, alpha=0.8)
    ax.plot(estado['ano'], estado['pct_ead'], color='black', linewidth=2.6, marker='o',
            markersize=5, label='Minas Gerais')
    ax.axvspan(2020, 2021, color='gray', alpha=0.08, label='Período pandêmico')
    ax.set_xlabel('Ano')
    ax.set_ylabel('Matrículas em EAD (% do total)')
    ax.set_title('Evolução da proporção de matrículas em EAD (2017-2024)', fontsize=11)
    ax.legend(title='Região Intermediária', bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=7.5, title_fontsize=8)
    fig.tight_layout()
    fig.savefig(caminho_saida, bbox_inches='tight')
    plt.close(fig)
    print(f"Salvo: {caminho_saida}")


def plot_evolucao_matriculas_regiao(serie_regiao, caminho_saida):
    dados = serie_regiao.dropna(subset=['nome_regiao_interm']).copy()
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    paleta = sns.color_palette("husl", n_colors=dados['nome_regiao_interm'].nunique())
    sns.lineplot(data=dados, x='ano', y='total_matriculas', hue='nome_regiao_interm',
                 marker='o', markersize=4, linewidth=1.4, palette=paleta, ax=ax)
    ax.set_xlabel('Ano')
    ax.set_ylabel('Total de matrículas')
    ax.set_title('Evolução das matrículas por Região Intermediária de MG (2017-2024)', fontsize=11)
    ax.axvspan(2020, 2021, color='gray', alpha=0.08, label='Período pandêmico')
    ax.legend(title='Região Intermediária', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7.5, title_fontsize=8)
    fig.tight_layout()
    fig.savefig(caminho_saida, bbox_inches='tight')
    plt.close(fig)
    print(f"Salvo: {caminho_saida}")


# ---------------------------------------------------------------------------
# Mapas coropleticos (Secao 3.10) - Regioes Geograficas Intermediarias de MG
# ---------------------------------------------------------------------------
def _carregar_shapefile_regioes() -> gpd.GeoDataFrame:
    """Carrega o shapefile do IBGE e padroniza o codigo da regiao para
    inteiro, compativel com cod_regiao_interm usado no resto do pipeline."""
    gdf = gpd.read_file(SHAPEFILE_REGIOES_PATH)
    gdf['cod_regiao_interm'] = gdf['CD_RGINT'].astype('int64')
    return gdf


def _agregar_indicador_por_regiao(df_municipios: pd.DataFrame, coluna_indicador: str,
                                   ponderar_por_matricula: bool = False,
                                   agregacao: str = 'mean') -> pd.DataFrame:
    """Agrega um indicador municipal para o nivel de regiao intermediaria.

    `agregacao`: 'mean' ou 'median'. Para I_rec, usar 'median' -- a mesma
    razao ja discutida na Secao 4.5: a media bruta de I_rec e dominada por
    outliers extremos de municipios pequenos (Secao 3.8.4), tornando a
    mediana a estatistica mais representativa para fins de visualizacao
    regional. Para I_imp (bem-comportado, sem outliers extremos), 'mean' e
    adequado.

    Por padrao (ponderar_por_matricula=False), cada municipio pesa
    igualmente na agregacao regional, mesma convencao das estatisticas
    descritivas da Secao 4.5. Se `ponderar_por_matricula=True`, pondera pela
    matricula de 2019 (M2019) -- so valido com agregacao='mean', pois media
    ponderada e mediana ponderada nao sao diretamente comparaveis aqui.
    """
    dados = df_municipios[['cod_regiao_interm', coluna_indicador]].dropna().copy()

    if ponderar_por_matricula:
        if agregacao != 'mean':
            raise ValueError("ponderar_por_matricula=True requer agregacao='mean'.")
        if 'M2019' not in df_municipios.columns:
            raise ValueError("Ponderacao por matricula requer a coluna M2019 no DataFrame de indicadores.")
        dados['peso'] = df_municipios.loc[dados.index, 'M2019']
        agregado = (
            dados.groupby('cod_regiao_interm', observed=True)
            .apply(lambda g: np.average(g[coluna_indicador], weights=g['peso']), include_groups=False)
            .reset_index(name=coluna_indicador)
        )
        return agregado

    funcao = 'mean' if agregacao == 'mean' else 'median'
    return dados.groupby('cod_regiao_interm', observed=True)[coluna_indicador].agg(funcao).reset_index()


def plot_mapa_coropletico(df_municipios: pd.DataFrame, coluna_indicador: str, titulo: str,
                           caminho_saida: str, cmap: str = 'RdYlGn',
                           ponderar_por_matricula: bool = False,
                           agregacao: str = 'mean',
                           centro_zero: bool = True) -> None:
    """Gera um mapa coropletico das Regioes Geograficas Intermediarias de
    MG, coloridas de acordo com a agregacao do indicador informado (ver
    `_agregar_indicador_por_regiao` para o significado de `agregacao`).

    centro_zero=True usa uma paleta divergente centrada em 0 (util para
    I_imp, onde 0 e um valor de referencia natural -- sem variacao). Para
    I_rec, onde 100 e o valor de referencia (recuperacao plena), ou para
    variaveis sem um zero natural (ICT, percentuais), usar centro_zero=False.
    """
    gdf_regioes = _carregar_shapefile_regioes()
    indicador_regional = _agregar_indicador_por_regiao(
        df_municipios, coluna_indicador, ponderar_por_matricula, agregacao
    )

    gdf = gdf_regioes.merge(indicador_regional, on='cod_regiao_interm', how='left')

    if gdf[coluna_indicador].isna().any():
        faltando = gdf.loc[gdf[coluna_indicador].isna(), 'NM_RGINT'].tolist()
        print(f"AVISO: regioes sem dado para '{coluna_indicador}': {faltando}")

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    vmin, vmax = gdf[coluna_indicador].min(), gdf[coluna_indicador].max()
    if centro_zero:
        limite = max(abs(vmin), abs(vmax))
        vmin, vmax = -limite, limite

    gdf.plot(
        column=coluna_indicador, cmap=cmap, linewidth=0.6, edgecolor='#555555',
        ax=ax, legend=True, vmin=vmin, vmax=vmax,
        missing_kwds={'color': '#dddddd', 'label': 'Sem dado'},
        legend_kwds={'shrink': 0.6, 'label': coluna_indicador},
    )

    for _, linha in gdf.iterrows():
        centroide = linha.geometry.centroid
        ax.annotate(
            linha['NM_RGINT'], xy=(centroide.x, centroide.y), ha='center', fontsize=6,
            color='#222222',
        )

    ax.set_title(titulo, fontsize=12)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(caminho_saida, bbox_inches='tight')
    plt.close(fig)
    print(f"Salvo: {caminho_saida}")


def gerar_mapas_coropleticos(df_indicadores: pd.DataFrame, diretorio_saida: str = DIR_GRAFICOS) -> list:
    """Gera os mapas de Iimp e Irec exigidos pela Secao 3.10."""
    os.makedirs(diretorio_saida, exist_ok=True)
    caminhos = []

    caminho_imp = os.path.join(diretorio_saida, 'mapa_impacto_regiao.png')
    plot_mapa_coropletico(
        df_indicadores, 'I_imp',
        'Indicador de Impacto (I_imp) médio por Região Intermediária',
        caminho_imp, cmap='RdYlGn', centro_zero=True,
    )
    caminhos.append(caminho_imp)

    caminho_rec = os.path.join(diretorio_saida, 'mapa_recuperacao_regiao.png')
    plot_mapa_coropletico(
        df_indicadores, 'I_rec',
        'Indicador de Recuperação (I_rec, mediana) por Região Intermediária',
        caminho_rec, cmap='RdYlGn', agregacao='median', centro_zero=False,
    )
    caminhos.append(caminho_rec)

    return caminhos