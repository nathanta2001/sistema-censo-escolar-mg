"""Testes do I_rec, de teve_perda e do filtro de sensibilidade por perda
real. Usam apenas dados sinteticos (nao dependem da camada Prata/Ouro).

Uso: python -m pytest
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis import calcular_i_rec
from src.sensibilidade import (
    correlacoes_por_perda,
    estatistica_descritiva_por_perda,
    filtrar_por_perda_real,
)
from src.stats import (
    calcular_matriz_correlacoes,
    regressao_logistica_recuperacao_plena,
    restringir_a_municipios_com_perda,
    trajetoria_municipios_com_perda,
)


@pytest.fixture
def municipios():
    # perda real (M2019 - M2021):   20   0  -10   50    5   100
    return pd.DataFrame({
        'cod_municipio': [1, 2, 3, 4, 5, 6],
        'M2019': [100, 100, 100, 500, 50, 1000],
        'M2021': [80, 100, 110, 450, 45, 900],
        'M2024': [90, 120, 90, 500, 40, 1100],
    })


# --- I_rec e teve_perda ------------------------------------------------------

def test_i_rec_com_perda(municipios):
    r = calcular_i_rec(municipios).set_index('cod_municipio')
    assert r.loc[1, 'I_rec'] == pytest.approx(50.0)     # recuperou 10 de 20
    assert r.loc[4, 'I_rec'] == pytest.approx(100.0)    # voltou exatamente a M2019
    assert r.loc[5, 'I_rec'] == pytest.approx(-100.0)   # continuou caindo
    assert r.loc[6, 'I_rec'] == pytest.approx(200.0)    # superou M2019


def test_i_rec_nan_quando_m2021_igual_m2019(municipios):
    r = calcular_i_rec(municipios).set_index('cod_municipio')
    assert np.isnan(r.loc[2, 'I_rec'])
    assert not r.loc[2, 'teve_perda']


def test_i_rec_nan_quando_m2021_maior_m2019(municipios):
    r = calcular_i_rec(municipios).set_index('cod_municipio')
    assert np.isnan(r.loc[3, 'I_rec'])
    assert not r.loc[3, 'teve_perda']


def test_i_rec_nunca_100_artificial(municipios):
    r = calcular_i_rec(municipios)
    assert r.loc[~r['teve_perda'], 'I_rec'].isna().all()
    assert r.loc[r['teve_perda'], 'I_rec'].notna().all()


def test_teve_perda(municipios):
    r = calcular_i_rec(municipios)
    assert r['teve_perda'].dtype == bool
    assert r['teve_perda'].tolist() == [True, False, False, True, True, True]
    assert r['teve_perda'].tolist() == (r['M2021'] < r['M2019']).tolist()


def test_municipios_sem_perda_permanecem_na_tabela(municipios):
    r = calcular_i_rec(municipios)
    assert len(r) == len(municipios)
    assert set(r['cod_municipio']) == set(municipios['cod_municipio'])


# --- Filtro de sensibilidade por perda real ----------------------------------

def test_filtro_perda_real_exclui_ganho_grande():
    # Municipio 2 ganhou 300 alunos: |M2019 - M2021| = 300 >= 200, mas nao
    # teve perda. O filtro antigo (valor absoluto) o manteria.
    df = calcular_i_rec(pd.DataFrame({
        'cod_municipio': [1, 2],
        'M2019': [1000, 1000], 'M2021': [700, 1300], 'M2024': [800, 1200],
    }))
    assert filtrar_por_perda_real(df, 200)['cod_municipio'].tolist() == [1]
    assert (df['M2019'] - df['M2021']).abs().ge(200).sum() == 2


@pytest.mark.parametrize('corte, esperados', [
    (0, [1, 4, 5, 6]),   # todos com perda
    (5, [1, 4, 5, 6]),   # corte inclusivo: perda = 5 entra
    (20, [1, 4, 6]),
    (50, [4, 6]),
    (100, [6]),
    (200, []),
])
def test_filtro_perda_real_cortes(municipios, corte, esperados):
    df = calcular_i_rec(municipios)
    assert filtrar_por_perda_real(df, corte)['cod_municipio'].tolist() == esperados


def test_filtro_perda_real_nunca_inclui_sem_perda(municipios):
    df = calcular_i_rec(municipios)
    for corte in (0, 1, 20, 50, 100, 200):
        assert filtrar_por_perda_real(df, corte)['teve_perda'].all()


def test_descritiva_por_perda_conta_a_partir_dos_com_perda(municipios):
    df = calcular_i_rec(municipios)
    desc = estatistica_descritiva_por_perda(df, limiares=[0, 20]).set_index('limiar_perda')
    assert desc.loc[0, 'n_municipios'] == 4
    assert desc.loc[0, 'municipios_excluidos'] == 0
    assert desc.loc[20, 'n_municipios'] == 3
    assert desc.loc[20, 'municipios_excluidos'] == 1


# --- Analises do I_rec restritas aos municipios com perda --------------------

@pytest.fixture
def base_grande():
    rng = np.random.default_rng(0)
    n = 300
    m2019 = rng.integers(500, 5000, n)
    m2021 = m2019 + rng.integers(-300, 150, n)
    m2024 = m2021 + rng.integers(-200, 300, n)
    df = pd.DataFrame({'cod_municipio': range(n), 'M2019': m2019, 'M2021': m2021, 'M2024': m2024})
    df = calcular_i_rec(df)
    for col in ('ICT_2024', 'pct_banda_2024', 'pct_estadual_2024', 'pct_urbana_2024'):
        df[col] = rng.uniform(0, 1, n)
    df['I_imp'] = (df['M2021'] - df['M2019']) / df['M2019'] * 100
    return df


def test_restringir_a_municipios_com_perda(base_grande):
    r = restringir_a_municipios_com_perda(base_grande)
    assert len(r) == base_grande['teve_perda'].sum()
    assert r['teve_perda'].all()
    # Para outros desfechos o filtro nao se aplica
    assert len(restringir_a_municipios_com_perda(base_grande, 'I_imp')) == len(base_grande)


def test_correlacoes_i_rec_usam_apenas_com_perda(base_grande):
    pares = [('ICT', 'ICT_2024', 'I_rec', ''), ('ICT imp', 'ICT_2024', 'I_imp', '')]
    m = calcular_matriz_correlacoes(base_grande, pares=pares).set_index('variavel_y')
    assert m.loc['I_rec', 'n'] == base_grande['teve_perda'].sum()
    assert m.loc['I_imp', 'n'] == len(base_grande)


def test_correlacoes_por_perda_n_decrescente(base_grande):
    c = correlacoes_por_perda(base_grande)
    n_por_corte = c[c['variavel_x'] == 'ICT_2024'].set_index('limiar_perda')['n']
    perda = base_grande['M2019'] - base_grande['M2021']
    for corte, n in n_por_corte.items():
        assert n == ((perda > 0) & (perda >= corte)).sum()


def test_logistica_apenas_com_perda(base_grande):
    lg = regressao_logistica_recuperacao_plena(base_grande, preditores=['ICT_2024'])
    assert set(lg['modelo']) == {'sem_controle', 'com_controle_porte'}
    assert (lg['n'] == base_grande['teve_perda'].sum()).all()


def test_trajetoria_soma_os_com_perda(base_grande):
    t = trajetoria_municipios_com_perda(base_grande)
    assert t['n'].sum() == base_grande['teve_perda'].sum()
    assert t['pct_com_perda'].sum() == pytest.approx(100.0)
