from src.analysis import gerar_indicadores_ouro
from src.config import ANOS
from src.extract import baixar_e_extrair_dados
from src.load import salvar_camada_prata
from src.stats import executar_e_salvar_analise_estatistica
from src.transform import transformar_dados


# Função principal para executar o sistema de dados completo: extração, transformação e análise
def executar_sistema():

    # baixar os dados para a camada Bronze
    for ano in ANOS:

        sucesso = baixar_e_extrair_dados(ano)
        if not sucesso:
            print(f"Falha ao obter dados do ano {ano} após todas as tentativas. Pulando este ano.")
            continue

        print(f"Dados do ano {ano} baixados e extraídos para a camada Bronze.")

        # Transformação dos dados da camada Bronze para a camada Prata
        df_limpo = transformar_dados(ano)

        # Salvamento dos dados transformados na camada Prata
        salvar_camada_prata(df_limpo, ano)

    # Análise dos dados e geração de indicadores na camada Ouro
    df_indicadores = gerar_indicadores_ouro()

    print("Processamento concluído. Indicadores salvos na camada Ouro.")

    # Análise estatística (correlação/regressão) sobre os indicadores gerados
    resultado_estatistica = executar_e_salvar_analise_estatistica(df_indicadores)
    print("\n=== Matriz de correlações ===")
    print(resultado_estatistica['correlacoes'])
    print("\n=== Regressões significativas para I_rec ===")
    print(resultado_estatistica['regressoes'])


if __name__ == "__main__":
    executar_sistema()