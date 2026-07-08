
import os

# Função para salvar a camada Prata
def salvar_camada_prata(df, ano):

    # Definindo o caminho de saída para a camada Prata
    caminho_saida = f'data/prata/censo_{ano}.parquet'
    os.makedirs('data/prata', exist_ok=True)
    
    # Salvando em Parquet 
    df.to_parquet(caminho_saida, index=False, compression='snappy')
    print(f"Ano {ano} salvo na camada Prata.")