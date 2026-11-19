"""
Gera docs/contagem_base_por_ano.csv: numero de registros e colunas dos
microdados originais (camada Bronze) e da camada Prata, por ano, e quantos
registros cada filtro de src/transform.py removeu.

Os filtros sao reaplicados na mesma ordem do transform.py, lendo da Bronze
apenas CO_UF, TP_SITUACAO_FUNCIONAMENTO e CO_ENTIDADE:
    1. CO_UF == 31 (Minas Gerais)
    2. TP_SITUACAO_FUNCIONAMENTO == 1 (em atividade)
    3. remocao de duplicatas de CO_ENTIDADE
A selecao das variaveis de config.py reduz colunas, nao registros.

Opcionalmente (--microdados-inep), confere o md5 de cada CSV da Bronze com o
arquivo md5_microdados_ed_basica_<ano>.txt distribuido pelo INEP.

Uso:
    python scripts/contar_base_por_ano.py [--microdados-inep <dir>]
"""

import argparse
import hashlib
import os
import sys

import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from src import config  # noqa: E402

COLUNAS_FILTRO = ['CO_UF', 'CO_ENTIDADE', 'TP_SITUACAO_FUNCIONAMENTO']
SITUACAO = {1: 'ativas', 2: 'paralisadas', 3: 'extintas_no_ano', 4: 'extintas_anos_anteriores'}


def md5_confere(caminho_csv, raiz_inep, ano):
    if not raiz_inep:
        return ''
    esperado = ''
    for raiz, _, arquivos in os.walk(os.path.join(raiz_inep, f'microdados_censo_escolar_{ano}')):
        for a in arquivos:
            if a.lower().startswith('md5_microdados_ed_basica'):
                esperado = open(os.path.join(raiz, a), encoding='latin-1').read().lower()
    if not esperado:
        return 'md5 do INEP nao encontrado'
    h = hashlib.md5()
    with open(caminho_csv, 'rb') as f:
        for bloco in iter(lambda: f.read(1 << 22), b''):
            h.update(bloco)
    return h.hexdigest().lower() in esperado


def contar_ano(ano, raiz_inep):
    caminho = os.path.join(config.BRONZE_DIR, config.NOME_ARQUIVO_PADRAO.format(ano=ano))
    cabecalho = pd.read_csv(caminho, sep=';', encoding='latin1', nrows=0).columns
    selecionadas = [c for c in config.COLUNAS_ESSENCIAIS if c in cabecalho]
    ausentes = [c for c in config.COLUNAS_ESSENCIAIS if c not in cabecalho]

    n_total = n_mg = 0
    situacao = {}
    entidades_ativas = []
    for chunk in pd.read_csv(caminho, sep=';', encoding='latin1', usecols=COLUNAS_FILTRO,
                             chunksize=config.CHUNK_SIZE):
        n_total += len(chunk)
        mg = chunk[chunk['CO_UF'] == config.CO_UF_MG]
        n_mg += len(mg)
        for k, v in mg['TP_SITUACAO_FUNCIONAMENTO'].value_counts(dropna=False).items():
            situacao[k] = situacao.get(k, 0) + int(v)
        entidades_ativas.append(mg.loc[mg['TP_SITUACAO_FUNCIONAMENTO'] == 1, 'CO_ENTIDADE'])
    ativas = pd.concat(entidades_ativas)
    n_ativas, n_final = len(ativas), ativas.nunique()

    prata = pd.read_parquet(os.path.join(config.PRATA_DIR, f'censo_{ano}.parquet'))
    linha = {
        'ano': ano,
        'bronze_registros': n_total,
        'bronze_colunas': len(cabecalho),
        'removidos_filtro_uf_mg': n_total - n_mg,
        'registros_mg': n_mg,
        'removidos_filtro_escolas_ativas': n_mg - n_ativas,
    }
    for codigo, nome in SITUACAO.items():
        linha[f'mg_situacao_{codigo}_{nome}'] = situacao.get(codigo, 0)
    linha['mg_situacao_ausente'] = sum(v for k, v in situacao.items() if pd.isna(k))
    linha.update({
        'registros_mg_ativas': n_ativas,
        'removidos_duplicatas_co_entidade': n_ativas - n_final,
        'registros_apos_filtros': n_final,
        'prata_registros': len(prata),
        'prata_colunas': prata.shape[1],
        'variaveis_selecionadas_na_bronze': len(selecionadas),
        'variaveis_ausentes_na_bronze': ';'.join(ausentes),
        'md5_confere_inep': md5_confere(caminho, raiz_inep, ano),
    })
    if n_final != len(prata):
        print(f'AVISO: {ano}: {n_final} registros apos filtros, mas a Prata tem {len(prata)}.')
    return linha


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--microdados-inep', default='',
                        help='Diretorio com as pastas microdados_censo_escolar_<ano> (para conferir o md5)')
    parser.add_argument('--saida', default=os.path.join(RAIZ, 'docs', 'contagem_base_por_ano.csv'))
    args = parser.parse_args()

    linhas = []
    for ano in config.ANOS:
        linhas.append(contar_ano(ano, args.microdados_inep))
        print(f'{ano}: {linhas[-1]["bronze_registros"]} registros na Bronze, '
              f'{linhas[-1]["registros_apos_filtros"]} apos filtros', flush=True)
    df = pd.DataFrame(linhas)
    os.makedirs(os.path.dirname(args.saida), exist_ok=True)
    df.to_csv(args.saida, index=False, encoding='utf-8')
    print(f'Gravado: {args.saida}')


if __name__ == '__main__':
    main()
