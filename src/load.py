import os

from src.config import OURO_DIR, PRATA_DIR


def salvar_camada_prata(df, ano: int) -> None:
    os.makedirs(PRATA_DIR, exist_ok=True)
    caminho_saida = os.path.join(PRATA_DIR, f'censo_{ano}.parquet')
    df.to_parquet(caminho_saida, index=False, compression='snappy')
    print(f"Ano {ano} salvo na camada Prata em {caminho_saida}.")


def salvar_camada_ouro(df, nome_arquivo: str) -> str:
    os.makedirs(OURO_DIR, exist_ok=True)
    caminho_parquet = os.path.join(OURO_DIR, f'{nome_arquivo}.parquet')
    caminho_csv = os.path.join(OURO_DIR, f'{nome_arquivo}.csv')
    df.to_parquet(caminho_parquet, index=False, compression='snappy')
    df.to_csv(caminho_csv, index=False)
    print(f"Indicadores salvos na camada Ouro: {caminho_parquet}")
    return caminho_parquet