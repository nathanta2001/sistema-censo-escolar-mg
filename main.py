from src.analysis import gerar_indicadores_ouro
#from src.extract import baixar_e_extrair_dados
from src.load import salvar_camada_prata
from src.transform import transformar_dados
from src.config import ANOS

# Função principal para executar o sistema de dados completo: extração, transformação e análise
def executar_sistema():
    
    # baixar os dados para a camada Bronze
    for ano in ANOS:
        
        #baixar_e_extrair_dados(ano)

        print(f"Dados do ano {ano} baixados e extraídos para a camada Bronze.")
        
        # Transformação dos dados da camada Bronze para a camada Prata
        df_limpo = transformar_dados(ano)

        # Salvamento dos dados transformados na camada Prata
        salvar_camada_prata(df_limpo, ano)
    
    # Análise dos dados e geração de indicadores na camada Ouro
    gerar_indicadores_ouro()

    print("Processamento concluído. Indicadores salvos na camada Ouro.")

if __name__ == "__main__":
    executar_sistema()
        