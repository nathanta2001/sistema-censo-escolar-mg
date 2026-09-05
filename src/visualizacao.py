"""
Modulo de visualizacao estatica (Secao 3.10 do TCC).
"""

import os

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
    dados = df[[coluna_x, coluna_y]].dropna()
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
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


def plot_dispersao_rec_logit(df, coluna_x, rotulo_x, titulo, caminho_saida, limite_logit_exibicao=6.0):
    dados = df[[coluna_x, 'I_rec']].dropna().copy()
    dados['logit_I_rec'] = _logit(dados['I_rec'])
    modelo = smf.ols(f'logit_I_rec ~ {coluna_x}', data=dados).fit()
    beta0, beta1 = modelo.params['Intercept'], modelo.params[coluna_x]
    r2 = modelo.rsquared
    p_valor = modelo.pvalues[coluna_x]
    dentro_limite = dados['logit_I_rec'].abs() <= limite_logit_exibicao
    dados_visiveis = dados[dentro_limite]
    n_omitidos = (~dentro_limite).sum()

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    ax.scatter(dados_visiveis[coluna_x], dados_visiveis['logit_I_rec'], alpha=0.35, s=18, color='#2c6e91')
    x_linha = np.linspace(dados[coluna_x].min(), dados[coluna_x].max(), 100)
    ax.plot(x_linha, beta0 + beta1 * x_linha, color='#c0392b', linewidth=2)
    ax.set_xlabel(rotulo_x)
    ax.set_ylabel('logit(I_rec)')
    ax.set_ylim(-limite_logit_exibicao - 0.5, limite_logit_exibicao + 0.5)
    ax.set_title(titulo, fontsize=11)
    nota = f"  |  {n_omitidos} município(s) com I_rec muito fora de [0%, 100%] omitidos do gráfico" if n_omitidos > 0 else ""
    ax.text(0.03, 0.03, f"β₁ = {beta1:.3f}   R² = {r2:.4f}   p = {p_valor:.1e}{nota}",
            transform=ax.transAxes, fontsize=7, color='#444444', va='bottom', ha='left', wrap=True)
    fig.tight_layout()
    fig.savefig(caminho_saida, bbox_inches='tight')
    plt.close(fig)


def gerar_todas_as_dispersoes(df_indicadores, diretorio_saida=DIR_GRAFICOS):
    os.makedirs(diretorio_saida, exist_ok=True)
    caminhos = []
    for coluna_x, coluna_y, rotulo_x, rotulo_y, titulo in PARES_IMP_PARA_PLOTAR:
        if coluna_x not in df_indicadores.columns:
            continue
        caminho = os.path.join(diretorio_saida, f'dispersao_{coluna_x}_{coluna_y}.png')
        plot_dispersao_imp(df_indicadores, coluna_x, coluna_y, rotulo_x, rotulo_y, titulo, caminho)
        caminhos.append(caminho)
    for coluna_x, coluna_y, rotulo_x, rotulo_y, titulo in PARES_REC_PARA_PLOTAR:
        if coluna_x not in df_indicadores.columns:
            continue
        caminho = os.path.join(diretorio_saida, f'dispersao_{coluna_x}_logit_I_rec.png')
        plot_dispersao_rec_logit(df_indicadores, coluna_x, rotulo_x, titulo, caminho)
        caminhos.append(caminho)
    return caminhos


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