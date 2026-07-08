import pandas as pd

# Função para gerar indicadores a partir dos dados processados na camada Prata e salvar na camada Ouro
def gerar_indicadores_ouro():

    # Leitura dos dados processados na camada Prata
    df_2019 = pd.read_parquet('data/prata/censo_2019.parquet')
    df_2021 = pd.read_parquet('data/prata/censo_2021.parquet')
    
    # Cálculo do indicador de impacto: I_imp = ((M2021 - M2019) / M2019) * 100
    m2019 = df_2019.groupby('cod_municipio')['total_matriculas'].sum()
    m2021 = df_2021.groupby('cod_municipio')['total_matriculas'].sum()
    
    # Cálculo do I_imp = ((M2021 - M2019) / M2019) * 100 
    impacto = ((m2021 - m2019) / m2019) * 100
    
    # Salva na camada Ouro 
    impacto.to_csv('data/ouro/indicador_impacto_municipios.csv')

