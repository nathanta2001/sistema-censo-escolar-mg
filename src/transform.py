import pandas as pd
from src.config import COLUNAS_ESSENCIAIS

# Função para transformar os dados da camada Bronze para a camada Prata
def transformar_dados(ano):
    path = f'data/bronze/microdados_ed_basica_{ano}.csv'
    
    # chunking limitando a leitura para 50k linhas por vez
    chunks = pd.read_csv(
        path, 
        sep=';', 
        encoding='latin1', 
        chunksize=50000, # Tamanho do chunk
        usecols=COLUNAS_ESSENCIAIS # Seleção de colunas
    )
    
    lista_processada = []
    for chunk in chunks:
        # Filtragem geográfica: Minas Gerais = 31 
        df_mg = chunk[chunk['CO_UF'] == 31].copy()
        
        # Filtro: Apenas escolas em atividade
        df_mg = df_mg[df_mg['TP_SITUACAO_FUNCIONAMENTO'] == 1]
        
        # Adicionando à lista de DataFrames processados
        lista_processada.append(df_mg)
    
    # Concatenando os chunks processados
    df_final = pd.concat(lista_processada)
    
    # Padronização e Limpeza 
    df_final.rename(columns=COLUNAS_ESSENCIAIS, inplace=True)
    df_final['total_matriculas'] = df_final['total_matriculas'].fillna(0).astype('int32') # Downcasting
    
    return df_final