from src.analysis import gerar_indicadores_ouro
from src.config import ANO_PANDEMIA, ANO_POS_PANDEMIA, ANO_PRE_PANDEMIA, ANOS
from src.extract import baixar_e_extrair_dados
from src.load import salvar_camada_prata
from src.stats import executar_e_salvar_analise_estatistica
from src.transform import transformar_dados

ANOS_ESSENCIAIS = {ANO_PRE_PANDEMIA, ANO_PANDEMIA, ANO_POS_PANDEMIA}


def executar_sistema():
    anos_com_sucesso = set()

    for ano in ANOS:
        sucesso = baixar_e_extrair_dados(ano)
        if not sucesso:
            print(f"Falha ao obter dados do ano {ano} após todas as tentativas. Pulando este ano.")
            continue

        print(f"Dados do ano {ano} baixados e extraídos para a camada Bronze.")

        df_limpo = transformar_dados(ano)
        salvar_camada_prata(df_limpo, ano)
        anos_com_sucesso.add(ano)

    anos_essenciais_faltando = ANOS_ESSENCIAIS - anos_com_sucesso
    if anos_essenciais_faltando:
        print(
            f"\nProcessamento interrompido: os anos {sorted(anos_essenciais_faltando)} "
            f"são necessários para calcular os indicadores e não foram baixados com "
            f"sucesso. Corrija o problema de download acima e rode novamente."
        )
        return

    df_indicadores = gerar_indicadores_ouro()
    print("Processamento concluído. Indicadores salvos na camada Ouro.")

    resultado_estatistica = executar_e_salvar_analise_estatistica(df_indicadores)
    print("\n=== Matriz de correlações ===")
    print(resultado_estatistica['correlacoes'])
    print("\n=== Regressões significativas para I_rec ===")
    print(resultado_estatistica['regressoes'])


if __name__ == "__main__":
    executar_sistema()