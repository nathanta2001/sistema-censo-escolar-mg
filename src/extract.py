import requests
import zipfile
import os
import time
from config import URL_BASE_INEP, BRONZE_DIR, ANOS


def baixar_e_extrair_dados(ano):
    
    
    """
    Baixa e extrai os dados do Censo Escolar para um ano específico.
    """

    # Monta a URL de download e o caminho do arquivo zip
    url = URL_BASE_INEP.format(ano)
    zip_path = os.path.join(BRONZE_DIR, f'censo_{ano}.zip')

    # Garante que o diretório Bronze exista
    os.makedirs(BRONZE_DIR, exist_ok=True)

    print(f"Baixando dados do ano {ano}...")
    
    # Implementação de tentativas com tratamento de erros para download e extração
    tentativas = 3

    # Loop de tentativas para download e extração
    for tentativa in range(tentativas):
        try:

            # Download do arquivo zip 
            reponse = requests.get(url, stream=True, timeout=10)

            # Verifica se o download foi bem-sucedido
            if reponse.status_code == 200:

                # Salva o arquivo zip localmente
                with open(zip_path, 'wb') as f:
                    # Escreve o conteúdo do arquivo em blocos para evitar sobrecarga de memória
                    for chunk in reponse.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Extrai o conteúdo do arquivo zip para o diretório Bronze
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    print(f"Extraindo dados do ano {ano}...")
                    zip_ref.extractall(BRONZE_DIR)
                
                # Remove o arquivo zip após a extração para economizar espaço
                os.remove(zip_path)
                print(f"Dados do ano {ano} processados com sucesso.")
                return True
            
            else:

                print(f"Falha ao baixar dados do ano {ano}. Status code: {reponse.status_code}")

        # Tratamento de exceções para erros de rede e arquivos zip corrompidos
        except (requests.exceptions.RequestException, zipfile.BadZipFile) as e:

            print(f"Erro na tentativa {tentativa + 1} para o ano {ano}: {e}")

            # Se não for a última tentativa, aguarda antes de tentar novamente
            if tentativa < tentativas - 1:
                print("Nova tentativa em 100 segundos...")
                time.sleep(100)

    return False

if __name__ == "__main__":
    for ano in ANOS:
        baixar_e_extrair_dados(ano)