"""
Roda a analise de sensibilidade do I_rec (Secao 3.8.3/4.6 do TCC) sobre os
dados reais ja processados na camada Ouro.

Uso: python analise_sensibilidade.py
"""

import os

import pandas as pd

from src.config import OURO_DIR
from src.sensibilidade import executar_e_salvar_analise_sensibilidade


def main():
    caminho = os.path.join(OURO_DIR, 'indicadores_municipios.parquet')
    if not os.path.exists(caminho):
        print(f"Arquivo {caminho} nao encontrado. Rode 'python main.py' antes.")
        return

    df = pd.read_parquet(caminho)

    if not {'M2019', 'M2021'}.issubset(df.columns):
        print(
            "AVISO: as colunas M2019/M2021 nao estao na tabela de indicadores salva. "
            "Verifique se gerar_indicadores_ouro() em src/analysis.py inclui essas "
            "colunas no merge final (necessarias para a analise de sensibilidade)."
        )
        return

    resultado = executar_e_salvar_analise_sensibilidade(df)

    print("\n=== Estatística descritiva de I_rec por limiar de M2019 ===")
    print(resultado['descritiva_m2019'].to_string(index=False))

    print("\n=== Estatística descritiva de I_rec por limiar do denominador |M2019-M2021| ===")
    print(resultado['descritiva_denominador'].to_string(index=False))

    print("\n=== Correlações por limiar do denominador (preditores de I_rec) ===")
    cols = ['limiar_denominador_abs', 'preditor', 'n', 'r', 'p_valor', 'significativo_bonferroni']
    print(resultado['correlacoes_denominador'][cols].to_string(index=False))


if __name__ == "__main__":
    main()