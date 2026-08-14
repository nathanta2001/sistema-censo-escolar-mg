import os

from src.config import OURO_DIR, PRATA_DIR


def salvar_camada_prata(df, ano: int) -> None:
    """Salva o DataFrame tratado de um ano na camada Prata (Parquet)."""
    os.makedirs(PRATA_DIR, exist_ok=True)
    caminho_saida = os.path.join(PRATA_DIR, f'censo_{ano}.parquet')
    df.to_parquet(caminho_saida, index=False, compression='snappy')
    print(f"Ano {ano} salvo na camada Prata em {caminho_saida}.")


def salvar_camada_ouro(df, nome_arquivo: str) -> str:
    """Salva um DataFrame de indicadores agregados na camada Ouro, em
    Parquet (para o dashboard/Power BI) e em CSV (para inspecao rapida).

    Retorna o caminho do arquivo Parquet salvo.
    """
    os.makedirs(OURO_DIR, exist_ok=True)
    caminho_parquet = os.path.join(OURO_DIR, f'{nome_arquivo}.parquet')
    caminho_csv = os.path.join(OURO_DIR, f'{nome_arquivo}.csv')

    df.to_parquet(caminho_parquet, index=False, compression='snappy')
    df.to_csv(caminho_csv, index=False)

    print(f"Indicadores salvos na camada Ouro: {caminho_parquet}")
    return caminho_parquet