"""
Gera todos os graficos estaticos da Secao 3.10/4.8 do TCC a partir dos dados
ja processados na camada Ouro. Requer que main.py ja tenha sido executado
com sucesso pelo menos uma vez.

Uso: python gerar_graficos.py
"""

import os

import pandas as pd

from src.analysis import calcular_serie_matriculas_regiao
from src.config import OURO_DIR
from src.visualizacao import (
    DIR_GRAFICOS,
    gerar_mapas_coropleticos,
    gerar_todas_as_dispersoes,
    plot_evolucao_matriculas_regiao,
)


def main():
    caminho_indicadores = os.path.join(OURO_DIR, 'indicadores_municipios.parquet')
    if not os.path.exists(caminho_indicadores):
        print(
            f"Arquivo {caminho_indicadores} nao encontrado. "
            f"Rode 'python main.py' antes de gerar os graficos."
        )
        return

    df_indicadores = pd.read_parquet(caminho_indicadores)

    print("Gerando gráficos de dispersão...")
    gerar_todas_as_dispersoes(df_indicadores)

    print("\nGerando gráfico de evolução de matrículas por região intermediária...")
    serie_regiao = calcular_serie_matriculas_regiao()
    caminho_evolucao = os.path.join(DIR_GRAFICOS, 'evolucao_matriculas_regiao.png')
    plot_evolucao_matriculas_regiao(serie_regiao, caminho_evolucao)

    print("\nGerando mapas coropléticos...")
    gerar_mapas_coropleticos(df_indicadores)

    print(f"\nTodos os gráficos foram salvos em: {DIR_GRAFICOS}")


if __name__ == "__main__":
    main()