"""
Painel interativo (Streamlit) — Censo Escolar MG (Secao 3.10/4.8 do TCC).

Le exclusivamente da camada Ouro (data/ouro/), sem depender de Bronze/Prata
em tempo de execucao -- pensado para ser leve e facil de rodar/compartilhar
com gestores, conforme o objetivo de acessibilidade do trabalho.

Uso:
    streamlit run dashboard.py

Pre-requisitos: rodar main.py, gerar_graficos.py (ou pelo menos as funcoes
gerar_indicadores_ouro, gerar_serie_regiao_ouro, gerar_serie_dependencia_ouro
e executar_e_salvar_analise_estatistica) ao menos uma vez antes, para que os
arquivos parquet existam em data/ouro/.
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.formula.api as smf
import streamlit as st

try:
    import geopandas as gpd
except ImportError:
    gpd = None

from src.config import OURO_DIR, SHAPEFILE_REGIOES_PATH

st.set_page_config(page_title="Painel Censo Escolar MG", layout="wide")


# ---------------------------------------------------------------------------
# Carregamento de dados (cacheado -- so recarrega se os arquivos mudarem)
# ---------------------------------------------------------------------------
ARQUIVOS_OURO = {
    'indicadores': 'indicadores_municipios.parquet',
    'serie_regiao': 'serie_matriculas_regiao.parquet',
    'serie_dependencia': 'serie_matriculas_dependencia.parquet',
    'correlacoes': 'matriz_correlacoes.parquet',
    'regressoes': 'regressoes_i_rec.parquet',
}


def assinatura_ouro():
    """Datas de modificacao dos arquivos da camada Ouro. Usada como argumento
    do cache: se o pipeline for reexecutado e os arquivos mudarem, o painel
    recarrega os dados sozinho, sem precisar reiniciar o Streamlit."""
    return tuple(
        os.path.getmtime(os.path.join(OURO_DIR, arq)) if os.path.exists(os.path.join(OURO_DIR, arq)) else 0.0
        for arq in ARQUIVOS_OURO.values()
    )


@st.cache_data
def carregar_dados(assinatura):
    faltando = [
        nome for nome, arq in ARQUIVOS_OURO.items()
        if not os.path.exists(os.path.join(OURO_DIR, arq))
    ]
    if faltando:
        return None, faltando

    dados = {
        nome: pd.read_parquet(os.path.join(OURO_DIR, arq))
        for nome, arq in ARQUIVOS_OURO.items()
    }
    return dados, []


@st.cache_data
def carregar_geojson(tolerancia_simplificacao: float = 0.005):
    """Carrega o shapefile e converte para GeoJSON, simplificando a
    geometria antes.

    O shapefile do IBGE vem em resolucao de precisao cartografica -- muito
    mais detalhe de contorno (milhares de vertices por poligono) do que o
    necessario para um mapa coropletico em tela. Isso deixa o carregamento
    e a renderizacao no navegador visivelmente mais lentos sem ganho visual
    perceptivel. `simplify()` (algoritmo Douglas-Peucker) reduz o numero de
    vertices preservando a forma geral dos poligonos; `preserve_topology`
    evita que a simplificacao quebre a geometria (auto-intersecoes etc).
    A tolerancia esta em graus (CRS geografico EPSG:4674) -- 0.005 graus
    equivale a aproximadamente 500m, adequado para visualizacao em nivel
    estadual sem distorcer perceptivelmente os contornos.
    """
    if gpd is None or not os.path.exists(SHAPEFILE_REGIOES_PATH):
        return None
    gdf = gpd.read_file(SHAPEFILE_REGIOES_PATH)
    gdf['cod_regiao_interm'] = gdf['CD_RGINT'].astype('int64')
    gdf['geometry'] = gdf['geometry'].simplify(tolerancia_simplificacao, preserve_topology=True)
    gdf = gdf[['cod_regiao_interm', 'NM_RGINT', 'geometry']]
    return json.loads(gdf.to_json())


def logit(serie_percentual: pd.Series) -> pd.Series:
    p = (serie_percentual / 100.0).clip(lower=1e-4, upper=1 - 1e-4)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------
dados, arquivos_faltando = carregar_dados(assinatura_ouro())

st.title("Impacto e Recuperação Educacional em Minas Gerais (2017-2024)")

if dados is None:
    st.error(
        "Não foi possível carregar os dados. Os seguintes arquivos não foram "
        f"encontrados em `{OURO_DIR}`: {', '.join(arquivos_faltando)}.\n\n"
        "Rode `python main.py` e `python gerar_graficos.py` antes de abrir o painel."
    )
    st.stop()

indicadores = dados['indicadores']
serie_regiao = dados['serie_regiao']
serie_dependencia = dados['serie_dependencia']
correlacoes = dados['correlacoes']
regressoes = dados['regressoes']

geojson = carregar_geojson()

regioes_disponiveis = sorted(indicadores['nome_regiao_interm'].dropna().unique())

aba1, aba2, aba3, aba4 = st.tabs(
    ["Visão Geral Municipal", "Evolução Temporal", "Achados Estatísticos", "Mapas"]
)

# ---------------------------------------------------------------------------
# Aba 1 — Visão Geral Municipal
# ---------------------------------------------------------------------------
with aba1:
    regiao_sel = st.multiselect(
        "Região Intermediária", regioes_disponiveis, default=regioes_disponiveis, key="regiao_aba1"
    )
    filtrado = indicadores[indicadores['nome_regiao_interm'].isin(regiao_sel)]

    col1, col2, col3 = st.columns(3)
    col1.metric("Municípios no filtro", len(filtrado))
    col2.metric("Média de I_imp", f"{filtrado['I_imp'].mean():.2f}%")
    col3.metric("Mediana de I_rec", f"{filtrado['I_rec'].median():.2f}%")

    colunas_tabela = [
        c for c in ['nome_municipio', 'nome_regiao_interm', 'I_imp', 'I_rec',
                     'I_EAD', 'I_ERE', 'ICT_2019', 'ICT_2024']
        if c in filtrado.columns
    ]
    st.dataframe(filtrado[colunas_tabela], width='stretch', hide_index=True)

# ---------------------------------------------------------------------------
# Aba 2 — Evolução Temporal
# ---------------------------------------------------------------------------
with aba2:
    dependencias_disponiveis = sorted(serie_dependencia['dependencia_administrativa'].dropna().unique())
    ano_min, ano_max = int(serie_dependencia['ano'].min()), int(serie_dependencia['ano'].max())

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        regiao_sel_2 = st.multiselect(
            "Região Intermediária", regioes_disponiveis, default=regioes_disponiveis, key="regiao_aba2"
        )
    with col_f2:
        dep_sel = st.multiselect(
            "Dependência Administrativa (tipo de escola)", dependencias_disponiveis,
            default=dependencias_disponiveis, key="dep_aba2"
        )
    periodo = st.slider("Período (ano)", ano_min, ano_max, (ano_min, ano_max), key="periodo_aba2")

    filtrado2 = serie_dependencia[
        serie_dependencia['nome_regiao_interm'].isin(regiao_sel_2)
        & serie_dependencia['dependencia_administrativa'].isin(dep_sel)
        & serie_dependencia['ano'].between(*periodo)
    ]

    st.subheader("Total de matrículas")
    agregado = (
        filtrado2.groupby(['ano', 'nome_regiao_interm'], observed=True)['total_matriculas']
        .sum()
        .reset_index()
    )
    fig_evolucao = px.line(
        agregado, x='ano', y='total_matriculas', color='nome_regiao_interm', markers=True,
        labels={'ano': 'Ano', 'total_matriculas': 'Total de matrículas', 'nome_regiao_interm': 'Região'},
    )
    fig_evolucao.add_vrect(x0=2019.5, x1=2021.5, fillcolor="gray", opacity=0.12, line_width=0)
    st.plotly_chart(fig_evolucao, width='stretch')

    st.subheader("Matrículas em EAD (% do total)")
    if 'matriculas_ead' not in filtrado2.columns:
        st.info(
            "A série carregada não tem a coluna `matriculas_ead`. Regenere a série de "
            "matrículas por dependência (rode `python gerar_graficos.py`) para exibir este gráfico."
        )
    else:
        ead = (
            filtrado2.groupby(['ano', 'nome_regiao_interm'], observed=True)[['total_matriculas', 'matriculas_ead']]
            .sum(min_count=1)
            .reset_index()
        )
        ead['pct_ead'] = ead['matriculas_ead'] / ead['total_matriculas'] * 100
        fig_ead = px.line(
            ead, x='ano', y='pct_ead', color='nome_regiao_interm', markers=True,
            labels={'ano': 'Ano', 'pct_ead': 'Matrículas em EAD (%)', 'nome_regiao_interm': 'Região'},
        )
        fig_ead.add_vrect(x0=2019.5, x1=2021.5, fillcolor="gray", opacity=0.12, line_width=0)
        st.plotly_chart(fig_ead, width='stretch')
        st.caption(
            "Proporção calculada sobre o total das regiões e redes selecionadas. "
            "Anos sem informação de matrícula EAD aparecem como lacuna."
        )

# ---------------------------------------------------------------------------
# Aba 3 — Achados Estatísticos
# ---------------------------------------------------------------------------
with aba3:
    st.subheader("Matriz de correlações")
    st.dataframe(correlacoes, width='stretch', hide_index=True)

    st.subheader("Regressões (preditores significativos de I_rec)")
    st.dataframe(regressoes, width='stretch', hide_index=True)

    st.subheader("Dispersão por preditor")
    preditor_sel = st.selectbox("Escolha um preditor", correlacoes['preditor'].tolist())
    linha = correlacoes[correlacoes['preditor'] == preditor_sel].iloc[0]
    coluna_x, coluna_y = linha['variavel_x'], linha['variavel_y']

    if coluna_x not in indicadores.columns:
        st.warning(f"Coluna '{coluna_x}' não disponível nos dados carregados.")
    elif coluna_x == 'I_EAD' and coluna_y == 'I_imp':
        # Variavel com muitos zeros: um unico LOWESS e instavel; compara-se os
        # grupos (x = 0 e x > 0) e mostra-se a dispersao apenas de x > 0.
        dados_ead = indicadores[['I_EAD', 'I_imp']].dropna()
        zero = dados_ead[dados_ead['I_EAD'] == 0]
        pos = dados_ead[dados_ead['I_EAD'] > 0]
        grupos = pd.concat([
            pd.DataFrame({'grupo': f'I_EAD = 0 (n = {len(zero)})', 'I_imp': zero['I_imp']}),
            pd.DataFrame({'grupo': f'I_EAD > 0 (n = {len(pos)})', 'I_imp': pos['I_imp']}),
        ])
        col_a, col_b = st.columns([1, 1.6])
        with col_a:
            st.plotly_chart(
                px.box(grupos, x='grupo', y='I_imp', labels={'grupo': '', 'I_imp': 'I_imp (%)'},
                       title='(a) Comparação entre grupos'),
                width='stretch',
            )
        with col_b:
            fig_pos = px.scatter(
                pos, x='I_EAD', y='I_imp', opacity=0.5,
                trendline='lowess' if len(pos) >= 15 else None,
                labels={'I_EAD': 'I_EAD (%)', 'I_imp': 'I_imp (%)'},
                title='(b) Apenas municípios com I_EAD > 0',
            )
            st.plotly_chart(fig_pos, width='stretch')
        st.caption(
            f"Mediana de I_imp: {zero['I_imp'].median():.2f}% (I_EAD = 0, n = {len(zero)}) "
            f"contra {pos['I_imp'].median():.2f}% (I_EAD > 0, n = {len(pos)})."
        )
    elif coluna_y == 'I_imp':
        fig_disp = px.scatter(
            indicadores, x=coluna_x, y='I_imp', trendline='lowess',
            opacity=0.4, labels={coluna_x: coluna_x, 'I_imp': 'I_imp (%)'},
        )
        st.plotly_chart(fig_disp, width='stretch')
    else:
        # I_rec nao e limitado a [0, 100]: o logit usa I_rec/100 truncado em
        # [eps, 1 - eps]. Todos os municipios sao exibidos; os que ficam nas
        # faixas extremas (I_rec <= 0% ou >= 100%) ganham cor propria.
        eps = 1e-4
        dados_rec = indicadores[[coluna_x, 'I_rec']].dropna().copy()
        dados_rec['logit_I_rec'] = logit(dados_rec['I_rec'])
        modelo = smf.ols(f'logit_I_rec ~ {coluna_x}', data=dados_rec).fit()

        pct = dados_rec['I_rec'] / 100.0
        dados_rec['faixa'] = np.where(
            pct >= 1 - eps, 'I_rec ≥ 100%', np.where(pct <= eps, 'I_rec ≤ 0%', '0% < I_rec < 100%')
        )
        rng = np.random.default_rng(42)
        extremo = dados_rec['faixa'] != '0% < I_rec < 100%'
        dados_rec['logit_exibido'] = dados_rec['logit_I_rec'] + np.where(
            extremo, rng.uniform(-0.35, 0.35, len(dados_rec)), 0.0
        )

        fig_disp = px.scatter(
            dados_rec, x=coluna_x, y='logit_exibido', color='faixa', opacity=0.5,
            color_discrete_map={'0% < I_rec < 100%': '#2c6e91', 'I_rec ≥ 100%': '#2e8b57', 'I_rec ≤ 0%': '#d9822b'},
            labels={coluna_x: coluna_x, 'logit_exibido': 'logit(I_rec)', 'faixa': ''},
        )
        x_linha = np.linspace(dados_rec[coluna_x].min(), dados_rec[coluna_x].max(), 100)
        y_linha = modelo.params['Intercept'] + modelo.params[coluna_x] * x_linha
        fig_disp.add_scatter(x=x_linha, y=y_linha, mode='lines', name='Reta ajustada',
                             line={'color': '#c0392b', 'width': 3})
        st.plotly_chart(fig_disp, width='stretch')

        contagem = dados_rec['faixa'].value_counts()
        st.caption(
            f"β₁ = {modelo.params[coluna_x]:.3f}  |  R² = {modelo.rsquared:.4f}  |  "
            f"p = {modelo.pvalues[coluna_x]:.2e}  |  "
            f"{int(contagem.get('I_rec ≥ 100%', 0))} municípios com I_rec ≥ 100%, "
            f"{int(contagem.get('I_rec ≤ 0%', 0))} com I_rec ≤ 0% e "
            f"{int(contagem.get('0% < I_rec < 100%', 0))} intermediários. "
            f"Valores fora de (0%, 100%) foram truncados em ±{np.log((1 - eps) / eps):.2f} (ε = 10⁻⁴); "
            f"a magnitude de β₁ depende de ε (ver Seção 3.8.2 do TCC)."
        )

# ---------------------------------------------------------------------------
# Aba 4 — Mapas
# ---------------------------------------------------------------------------
@st.cache_data
def construir_mapa(indicador_mapa: str, assinatura):
    agregacao = 'mean' if indicador_mapa == 'I_imp' else 'median'

    regional = (
        indicadores.groupby('cod_regiao_interm', observed=True)[indicador_mapa]
        .agg(agregacao)
        .reset_index()
    )
    nomes_regiao = indicadores[['cod_regiao_interm', 'nome_regiao_interm']].drop_duplicates()
    regional = regional.merge(nomes_regiao, on='cod_regiao_interm', how='left')

    fig_mapa = px.choropleth(
        regional, geojson=geojson, locations='cod_regiao_interm',
        featureidkey='properties.cod_regiao_interm',
        color=indicador_mapa, color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0 if indicador_mapa == 'I_imp' else None,
        hover_name='nome_regiao_interm',
        labels={indicador_mapa: f"{indicador_mapa} ({'média' if agregacao == 'mean' else 'mediana'})"},
    )
    fig_mapa.update_geos(fitbounds="locations", visible=False)
    fig_mapa.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=600)
    return fig_mapa


with aba4:
    if geojson is None:
        st.warning(
            "Shapefile das Regiões Intermediárias não encontrado em "
            f"`{SHAPEFILE_REGIOES_PATH}`. Verifique se o arquivo foi baixado "
            "e colocado no caminho esperado (ver README)."
        )
    else:
        indicador_mapa = st.radio("Indicador", ["I_imp", "I_rec"], horizontal=True)
        st.plotly_chart(construir_mapa(indicador_mapa, assinatura_ouro()), width='stretch')