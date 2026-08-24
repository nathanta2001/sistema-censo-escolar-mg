"""
Modulo de visualizacao estatica (Secao 3.10 do TCC).

Gera os graficos de dispersao com reta/tendencia e a serie temporal de
matriculas por regiao intermediaria, a partir da tabela consolidada de
indicadores (camada Ouro) e da matriz de correlacoes (src/stats.py).

Os mapas coropleticos (que exigem shapefile do IBGE) e o dashboard Power BI
sao tratados em modulos/etapas separadas.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

from src.config import OURO_DIR

sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)

DIR_GRAFICOS = os.path.join(OURO_DIR, 'graficos')

# Pares de I_imp (metodo Spearman -- sem modelo de regressao ajustado; a
# linha exibida e apenas uma tendencia visual suavizada, nao uma reta OLS)
PARES_IMP_PARA_PLOTAR = [
    ('I_EAD', 'I_imp', 'I_EAD (%)', 'I_imp (%)', 'Intensidade de matrícula EAD (2021) x Impacto'),
    ('I_ERE', 'I_imp', 'I_ERE (dias)', 'I_imp (%)', 'Dias de mediação remota (2021) x Impacto'),
]

# Pares de I_rec (tem regressao ajustada -- Secao 3.8.2 -- plotados na
# escala logit, que foi a especificacao recomendada no texto)
PARES_REC_PARA_PLOTAR = [
    ('ICT_2024', 'I_rec', 'ICT municipal (2024)', 'logit(I_rec)', 'Capacidade tecnológica (2024) x Recuperação'),
    ('pct_estadual_2024', 'I_rec', '% escolas estaduais (2024)', 'logit(I_rec)', 'Dependência estadual (2024) x Recuperação'),
    ('pct_urbana_2024', 'I_rec', '% escolas urbanas (2024)', 'logit(I_rec)', 'Localização urbana (2024) x Recuperação'),
]


def _logit(serie_percentual: pd.Series) -> pd.Series:
    """Aplica a mesma transformacao logit usada em src/stats.py (Eq. 3.8.2),
    com o mesmo recorte de seguranca para evitar logit(0)/logit(1)."""
    p = (serie_percentual / 100.0).clip(lower=1e-4, upper=1 - 1e-4)
    return np.log(p / (1 - p))


def plot_dispersao_imp(df: pd.DataFrame, coluna_x: str, coluna_y: str,
                        rotulo_x: str, rotulo_y: str, titulo: str,
                        caminho_saida: str) -> None:
    """Dispersao + tendencia LOWESS para os preditores de I_imp (metodo
    Spearman -- nao ha reta de regressao ajustada para esses pares, apenas
    uma visualizacao da tendencia monotonica)."""
    dados = df[[coluna_x, coluna_y]].dropna()

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    sns.regplot(
        data=dados, x=coluna_x, y=coluna_y, ax=ax,
        lowess=True, scatter_kws={'alpha': 0.35, 's': 18, 'color': '#2c6e91'},
        line_kws={'color': '#c0392b', 'linewidth': 2},
    )
    ax.set_xlabel(rotulo_x)
    ax.set_ylabel(rotulo_y)
    ax.set_title(titulo, fontsize=11)
    fig.tight_layout()
    fig.savefig(caminho_saida, bbox_inches='tight')
    plt.close(fig)


def plot_dispersao_rec_logit(df: pd.DataFrame, coluna_x: str,
                              rotulo_x: str, titulo: str,
                              caminho_saida: str,
                              limite_logit_exibicao: float = 6.0) -> None:
    """Dispersao + reta OLS ajustada na escala logit(I_rec), reproduzindo o
    modelo de regressao efetivamente reportado na Secao 4.6 para esse
    preditor (Secao 3.8.2).

    A reta e ajustada com TODOS os municipios (mesmo modelo do stats.py,
    sem alteracao metodologica). Para a exibicao dos PONTOS, municipios cujo
    logit(I_rec) calculado exceda `limite_logit_exibicao` em modulo sao
    omitidos do grafico -- tipicamente municipios pequenos cujo denominador
    reduzido (Eq. 3.2) leva I_rec a valores muito fora de [0, 100]%, que o
    recorte de seguranca da transformacao logit (necessario para evitar
    log(0)) colapsa nos mesmos pontos extremos, sem variacao visual
    informativa. Filtrar pelo proprio valor de logit (em vez de pela escala
    percentual original de I_rec) e o que efetivamente evita esse
    achatamento, pois o problema ocorre no espaco transformado. O numero de
    pontos omitidos e reportado na propria figura para transparencia.
    """
    dados = df[[coluna_x, 'I_rec']].dropna().copy()
    dados['logit_I_rec'] = _logit(dados['I_rec'])

    # Reta ajustada com a base completa (nao filtrada)
    modelo = smf.ols(f'logit_I_rec ~ {coluna_x}', data=dados).fit()
    beta0, beta1 = modelo.params['Intercept'], modelo.params[coluna_x]
    r2 = modelo.rsquared
    p_valor = modelo.pvalues[coluna_x]

    # Filtro apenas para os pontos exibidos, no espaco logit
    dentro_limite = dados['logit_I_rec'].abs() <= limite_logit_exibicao
    dados_visiveis = dados[dentro_limite]
    n_omitidos = (~dentro_limite).sum()

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    ax.scatter(dados_visiveis[coluna_x], dados_visiveis['logit_I_rec'], alpha=0.35, s=18, color='#2c6e91')

    x_linha = np.linspace(dados[coluna_x].min(), dados[coluna_x].max(), 100)
    y_linha = beta0 + beta1 * x_linha
    ax.plot(x_linha, y_linha, color='#c0392b', linewidth=2)

    ax.set_xlabel(rotulo_x)
    ax.set_ylabel('logit(I_rec)')
    ax.set_ylim(-limite_logit_exibicao - 0.5, limite_logit_exibicao + 0.5)
    ax.set_title(titulo, fontsize=11)
    nota_omissao = (
        f"  |  {n_omitidos} município(s) com I_rec muito fora de [0%, 100%] omitidos do gráfico"
        if n_omitidos > 0 else ""
    )
    ax.text(
        0.03, 0.03,
        f"β₁ = {beta1:.3f}   R² = {r2:.4f}   p = {p_valor:.1e}{nota_omissao}",
        transform=ax.transAxes, fontsize=7, color='#444444',
        va='bottom', ha='left', wrap=True,
    )
    fig.tight_layout()
    fig.savefig(caminho_saida, bbox_inches='tight')
    plt.close(fig)


def gerar_todas_as_dispersoes(df_indicadores: pd.DataFrame, diretorio_saida: str = DIR_GRAFICOS) -> list:
    """Gera todos os graficos de dispersao (I_imp com LOWESS, I_rec com reta
    OLS em escala logit) e retorna a lista de caminhos salvos."""
    os.makedirs(diretorio_saida, exist_ok=True)
    caminhos = []

    for coluna_x, coluna_y, rotulo_x, rotulo_y, titulo in PARES_IMP_PARA_PLOTAR:
        if coluna_x not in df_indicadores.columns:
            print(f"AVISO: coluna '{coluna_x}' ausente -- grafico '{titulo}' pulado.")
            continue
        caminho = os.path.join(diretorio_saida, f'dispersao_{coluna_x}_{coluna_y}.png')
        plot_dispersao_imp(df_indicadores, coluna_x, coluna_y, rotulo_x, rotulo_y, titulo, caminho)
        caminhos.append(caminho)
        print(f"Salvo: {caminho}")

    for coluna_x, coluna_y, rotulo_x, rotulo_y, titulo in PARES_REC_PARA_PLOTAR:
        if coluna_x not in df_indicadores.columns:
            print(f"AVISO: coluna '{coluna_x}' ausente -- grafico '{titulo}' pulado.")
            continue
        caminho = os.path.join(diretorio_saida, f'dispersao_{coluna_x}_logit_I_rec.png')
        plot_dispersao_rec_logit(df_indicadores, coluna_x, rotulo_x, titulo, caminho)
        caminhos.append(caminho)
        print(f"Salvo: {caminho}")

    return caminhos


def plot_evolucao_matriculas_regiao(serie_regiao: pd.DataFrame, caminho_saida: str) -> None:
    """Grafico de linhas da evolucao do total de matriculas por regiao
    intermediaria, 2017-2024 (Secao 3.10).

    serie_regiao: saida de src.analysis.calcular_serie_matriculas_regiao()
    (colunas: cod_regiao_interm, nome_regiao_interm, total_matriculas, ano).
    """
    dados = serie_regiao.dropna(subset=['nome_regiao_interm']).copy()

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    paleta = sns.color_palette("husl", n_colors=dados['nome_regiao_interm'].nunique())

    sns.lineplot(
        data=dados, x='ano', y='total_matriculas', hue='nome_regiao_interm',
        marker='o', markersize=4, linewidth=1.4, palette=paleta, ax=ax,
    )

    ax.set_xlabel('Ano')
    ax.set_ylabel('Total de matrículas')
    ax.set_title('Evolução das matrículas por Região Intermediária de MG (2017-2024)', fontsize=11)
    ax.axvspan(2020, 2021, color='gray', alpha=0.08, label='Período pandêmico')
    ax.legend(title='Região Intermediária', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7.5, title_fontsize=8)
    fig.tight_layout()
    fig.savefig(caminho_saida, bbox_inches='tight')
    plt.close(fig)
    print(f"Salvo: {caminho_saida}")
