import glob
import os
import time
import zipfile

import certifi
import requests

# Alguns antivirus (Kaspersky, Avast, AVG, ESET, Bitdefender etc.) fazem
# inspecao de HTTPS: interceptam a conexao e apresentam um certificado
# proprio. O navegador confia nele porque usa o repositorio de certificados
# do Windows (onde o antivirus se registra); o certifi do Python usa uma
# lista propria que NAO inclui esse certificado, e a verificacao SSL falha
# mesmo com o site funcionando perfeitamente no navegador.
#
# Se o pacote `truststore` estiver instalado, usamos o mesmo repositorio de
# certificados do sistema operacional que o navegador usa. Se nao estiver
# instalado, cai para o bundle do certifi.
try:
    import truststore
    truststore.inject_into_ssl()
    VERIFY_SSL = True
except ImportError:
    VERIFY_SSL = certifi.where()

from src.config import (
    ANOS,
    BRONZE_DIR,
    DOWNLOAD_ESPERA_SEGUNDOS,
    DOWNLOAD_MAX_TENTATIVAS,
    HEADERS_DOWNLOAD,
    NOME_ARQUIVO_PADRAO,
    PADRAO_NOME_CSV_MICRODADOS,
    URLS_POR_ANO,
)


def _localizar_csv_principal(diretorio_extraido: str) -> str:
    """Localiza, dentro do diretorio onde o zip do INEP foi extraido, o CSV
    de microdados da educacao basica (nivel escola).

    O pacote do INEP pode conter varios CSVs (escolas, turmas, docentes,
    matriculas, gestor escolar, dicionario de dados etc.), entao a escolha
    NAO pode ser feita apenas pelo tamanho do arquivo. Estrategia:

    1. Procurar um arquivo cujo nome contenha o padrao conhecido
       (PADRAO_NOME_CSV_MICRODADOS, ex. "microdados_ed_basica").
    2. Se nenhum arquivo bater com o padrao, cai para o maior .csv
       encontrado como ultimo recurso -- com aviso para conferencia manual.
    """
    candidatos = glob.glob(os.path.join(diretorio_extraido, '**', '*.csv'), recursive=True)

    if not candidatos:
        raise FileNotFoundError(
            f"Nenhum arquivo .csv encontrado em {diretorio_extraido} apos a extracao."
        )

    por_nome = [
        c for c in candidatos
        if PADRAO_NOME_CSV_MICRODADOS.lower() in os.path.basename(c).lower()
    ]

    if por_nome:
        if len(por_nome) > 1:
            print(
                f"AVISO: mais de um CSV bateu com o padrao "
                f"'{PADRAO_NOME_CSV_MICRODADOS}': {por_nome}. Usando o maior."
            )
        return max(por_nome, key=os.path.getsize)

    print(
        f"AVISO: nenhum CSV com o padrao '{PADRAO_NOME_CSV_MICRODADOS}' foi "
        f"encontrado. Usando o maior .csv extraido como fallback -- "
        f"confira manualmente se e o arquivo correto: {candidatos}"
    )
    return max(candidatos, key=os.path.getsize)


def _limpar_diretorio_extracao(dir_extracao: str) -> None:
    """Remove o diretorio temporario de extracao (dicionario de dados,
    anexos e demais arquivos que nao sao o CSV de microdados escolhido)."""
    import shutil
    if os.path.isdir(dir_extracao):
        shutil.rmtree(dir_extracao, ignore_errors=True)


def _baixar_com_retomada(url: str, zip_path: str) -> None:
    """Baixa um arquivo via streaming, retomando de onde parou se ja existir
    um download parcial (HTTP Range request).

    Isso e importante especificamente para arquivos grandes (~200MB) em
    conexoes instaveis: sem retomada, uma queda de conexao aos 190MB jogaria
    fora todo o progresso e reiniciaria do zero a cada tentativa -- o que
    pode nunca completar se a conexao cair sempre antes de terminar. Com
    retomada, cada tentativa aproveita o que ja foi baixado.

    Levanta requests.exceptions.RequestException em caso de falha (deixando
    o .zip parcial no disco, de proposito, para a proxima tentativa retomar).
    """
    tamanho_existente = os.path.getsize(zip_path) if os.path.exists(zip_path) else 0

    headers = dict(HEADERS_DOWNLOAD)
    modo_arquivo = 'wb'
    if tamanho_existente > 0:
        headers['Range'] = f'bytes={tamanho_existente}-'
        modo_arquivo = 'ab'
        print(f"Retomando download a partir de {tamanho_existente / 1_000_000:.1f} MB ja baixados...")

    response = requests.get(
        url, stream=True, timeout=(10, 60), headers=headers, verify=VERIFY_SSL
    )

    # Se pedimos Range mas o servidor nao suporta retomada (responde 200
    # inteiro em vez de 206 parcial), recomeça do zero para nao corromper o
    # arquivo (dado novo completo seria anexado a um arquivo ja completo).
    if tamanho_existente > 0 and response.status_code == 200:
        print("Servidor nao suporta retomada parcial para esta URL. Reiniciando do zero.")
        tamanho_existente = 0
        modo_arquivo = 'wb'

    response.raise_for_status()

    with open(zip_path, modo_arquivo) as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def baixar_e_extrair_dados(ano: int, forcar_download: bool = False) -> bool:
    """Baixa e extrai os dados do Censo Escolar para um ano especifico,
    deixando o CSV de microdados ja renomeado para o padrao unificado
    (Secao 3.6.1: censo_<ANO>.csv) diretamente na camada Bronze.

    Se o CSV final ja existir na Bronze, o download e pulado (a menos que
    forcar_download=True), evita reprocessar anos ja obtidos com sucesso
    em execucoes anteriores.

    Retorna True em caso de sucesso, False se todas as tentativas falharem.
    """

    zip_path = os.path.join(BRONZE_DIR, f'censo_{ano}.zip')
    dir_extracao = os.path.join(BRONZE_DIR, f'_extraido_{ano}')
    caminho_final = os.path.join(BRONZE_DIR, NOME_ARQUIVO_PADRAO.format(ano=ano))

    os.makedirs(BRONZE_DIR, exist_ok=True)

    if os.path.exists(caminho_final) and not forcar_download:
        print(f"Ano {ano}: {caminho_final} ja existe na camada Bronze. Pulando download.")
        return True

    url = URLS_POR_ANO.get(ano)
    if url is None:
        print(f"Nao ha URL cadastrada para o ano {ano} em URLS_POR_ANO (config.py). Pulando.")
        return False

    print(f"Baixando dados do ano {ano}...")

    tamanho_apos_falha_anterior = None

    for tentativa in range(DOWNLOAD_MAX_TENTATIVAS):
        try:
            _baixar_com_retomada(url, zip_path)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                print(f"Extraindo dados do ano {ano}...")
                zip_ref.extractall(dir_extracao)

            os.remove(zip_path)

            csv_origem = _localizar_csv_principal(dir_extracao)
            os.replace(csv_origem, caminho_final)

            _limpar_diretorio_extracao(dir_extracao)

            print(f"Dados do ano {ano} processados com sucesso -> {caminho_final}")
            return True

        except requests.exceptions.HTTPError as e:
            print(f"Erro HTTP na tentativa {tentativa + 1}/{DOWNLOAD_MAX_TENTATIVAS} para o ano {ano}: {e}")
            # Erro HTTP (404, 403 etc.) nao se beneficia de retomada -- o
            # .zip parcial (se houver) e removido para nao ficar obsoleto.
            if os.path.exists(zip_path):
                os.remove(zip_path)
            _limpar_diretorio_extracao(dir_extracao)

        except zipfile.BadZipFile:
            # O .zip baixado ate agora esta corrompido/incompleto. Se o
            # tamanho cresceu desde a ultima falha deste tipo, mantemos o
            # arquivo para a proxima tentativa retomar via Range. Se o
            # tamanho NAO mudou (o servidor considera a "retomada" completa,
            # mas o conteudo continua invalido), o arquivo esta genuinamente
            # corrompido -- descartamos e recomecamos do zero, para nao
            # ficar preso pedindo bytes que o servidor ja julga entregues.
            tamanho_atual = os.path.getsize(zip_path) if os.path.exists(zip_path) else 0
            if tamanho_atual == tamanho_apos_falha_anterior:
                print(
                    f"Erro na tentativa {tentativa + 1}/{DOWNLOAD_MAX_TENTATIVAS} para o ano {ano}: "
                    f"arquivo zip corrompido e sem progresso na retomada. Reiniciando do zero."
                )
                os.remove(zip_path)
                tamanho_apos_falha_anterior = None
            else:
                print(f"Erro na tentativa {tentativa + 1}/{DOWNLOAD_MAX_TENTATIVAS} para o ano {ano}: arquivo zip invalido/incompleto.")
                tamanho_apos_falha_anterior = tamanho_atual
            _limpar_diretorio_extracao(dir_extracao)

        except requests.exceptions.RequestException as e:
            print(f"Erro na tentativa {tentativa + 1}/{DOWNLOAD_MAX_TENTATIVAS} para o ano {ano}: {e}")
            _limpar_diretorio_extracao(dir_extracao)

        if tentativa < DOWNLOAD_MAX_TENTATIVAS - 1:
            print(f"Nova tentativa em {DOWNLOAD_ESPERA_SEGUNDOS} segundos...")
            time.sleep(DOWNLOAD_ESPERA_SEGUNDOS)

    # Todas as tentativas falharam: limpa o zip parcial para nao deixar lixo
    # inconsistente na Bronze, e para a proxima execucao comecar do zero.
    if os.path.exists(zip_path):
        os.remove(zip_path)
    _limpar_diretorio_extracao(dir_extracao)

    print(f"Falha ao baixar/extrair os dados do ano {ano} apos {DOWNLOAD_MAX_TENTATIVAS} tentativas.")
    return False


if __name__ == "__main__":
    for ano in ANOS:
        baixar_e_extrair_dados(ano)