"""
Gera docs/variaveis_utilizadas.csv: uma linha por variavel selecionada em
src/config.py (COLUNAS_ESSENCIAIS), com nome original do INEP, nome
padronizado, grupo funcional, tipo, anos em que existe e uso no pipeline.

A descricao vem do dicionario de dados do INEP (Anexo I dos microdados),
que nao acompanha o repositorio. O diretorio com os pacotes de microdados
(uma pasta microdados_censo_escolar_<ano> por ano) e passado como argumento.
Usa-se a descricao do dicionario de 2024 e indica-se em que anos ela difere.

Uso:
    python scripts/listar_variaveis.py --dicionarios <dir_microdados_inep>
"""

import argparse
import glob
import os
import re
import sys

import openpyxl
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from src import config  # noqa: E402

ANOS = list(config.ANOS)
ANO_DESCRICAO = 2024

GRUPOS = {
    'COLUNAS_IDENTIFICACAO': config.COLUNAS_IDENTIFICACAO,
    'COLUNAS_MATRICULAS': config.COLUNAS_MATRICULAS,
    'COLUNAS_MODALIDADE': config.COLUNAS_MODALIDADE,
    'COLUNAS_ESTRUTURAIS': config.COLUNAS_ESTRUTURAIS,
    'COLUNAS_EQUIPAMENTOS': config.COLUNAS_EQUIPAMENTOS,
    'COLUNAS_SITUACAO': config.COLUNAS_SITUACAO,
}

# Uso de cada variavel, conforme src/transform.py, src/analysis.py,
# src/stats.py, Dashboard.py e as saidas de data/ouro/. O I_ERE nao usa
# nenhuma destas variaveis (vem da Pesquisa Resposta Educacional).
SEM_USO = ('Selecionada em config.py e mantida na camada Prata; nao e usada em nenhum '
           'indicador, modulo de analise, dashboard ou saida de data/ouro/')
USO = {
    'CO_ENTIDADE': ('identificacao', 'Remocao de duplicatas na Prata (transform.py)'),
    'CO_UF': ('filtro', 'Filtro territorial CO_UF == 31 (transform.py)'),
    'CO_MUNICIPIO': ('agregacao', 'Chave de agregacao municipal de todos os indicadores'),
    'NO_MUNICIPIO': ('agregacao', 'Nome do municipio nas saidas Ouro e no dashboard'),
    'CO_REGIAO_GEOG_INTERM': ('agregacao', 'Agregacao regional (mapas, series por regiao, descricao dos municipios sem perda); '
                              'preenchida pela tabela de referencia nos anos sem a coluna'),
    'NO_REGIAO_GEOG_INTERM': ('agregacao', 'Nome da regiao nas saidas Ouro, mapas e dashboard; '
                              'preenchida pela tabela de referencia nos anos sem a coluna'),
    'TP_LOCALIZACAO_DIFERENCIADA': ('sem uso', SEM_USO),
    'QT_MAT_BAS': ('I_imp; I_rec; I_EAD', 'M2019, M2021, M2024 (I_imp, I_rec, teve_perda); denominador do I_EAD; '
                   'series de matriculas por regiao e dependencia'),
    'QT_MAT_INF': ('sem uso', SEM_USO + '; ausentes preenchidos com 0 na Prata'),
    'QT_MAT_FUND': ('sem uso', SEM_USO + '; ausentes preenchidos com 0 na Prata'),
    'QT_MAT_MED': ('sem uso', SEM_USO + '; ausentes preenchidos com 0 na Prata'),
    'QT_MAT_EJA': ('sem uso', SEM_USO + '; ausentes preenchidos com 0 na Prata'),
    'QT_MAT_BAS_EAD': ('I_EAD', 'Numerador do I_EAD (2021); serie de matriculas EAD por dependencia'),
    'IN_MEDIACAO_PRESENCIAL': ('sem uso', SEM_USO),
    'IN_MEDIACAO_SEMIPRESENCIAL': ('sem uso', SEM_USO),
    'IN_MEDIACAO_EAD': ('sem uso', SEM_USO),
    'TP_DEPENDENCIA': ('preditor do I_rec', 'pct_estadual_2024 (preditor do I_rec); serie de matriculas por dependencia'),
    'TP_LOCALIZACAO': ('preditor do I_imp e do I_rec', 'pct_urbana_2019 (I_imp) e pct_urbana_2024 (I_rec)'),
    'IN_INTERNET': ('sem uso', SEM_USO),
    'IN_INTERNET_APRENDIZAGEM': ('ICT', 'p_internet do ICT; pct_internet_2019 (preditor do I_imp)'),
    'IN_INTERNET_ALUNOS': ('sem uso', SEM_USO),
    'IN_BANDA_LARGA': ('ICT', 'p_banda do ICT; pct_banda_2019 (I_imp) e pct_banda_2024 (I_rec)'),
    'IN_LABORATORIO_INFORMATICA': ('sem uso', SEM_USO),
    'IN_ENERGIA_REDE_PUBLICA': ('sem uso', SEM_USO),
    'IN_AGUA_POTAVEL': ('sem uso', SEM_USO),
    'IN_DESKTOP_ALUNO': ('ICT', 'Componente de equip (p_equip do ICT)'),
    'QT_DESKTOP_ALUNO': ('sem uso', SEM_USO),
    'IN_COMP_PORTATIL_ALUNO': ('ICT', 'Componente de equip (p_equip do ICT)'),
    'QT_COMP_PORTATIL_ALUNO': ('sem uso', SEM_USO),
    'IN_TABLET_ALUNO': ('ICT', 'Componente de equip (p_equip do ICT)'),
    'QT_TABLET_ALUNO': ('sem uso', SEM_USO),
    'TP_SITUACAO_FUNCIONAMENTO': ('filtro', 'Filtro de escolas em atividade (valor 1) (transform.py)'),
}


def _normalizar(texto):
    return re.sub(r'\s+', ' ', str(texto)).strip() if texto is not None else ''


def ler_dicionario(raiz_microdados, ano):
    arqs = [p for p in glob.glob(os.path.join(raiz_microdados, f'microdados_censo_escolar_{ano}', '**', '*.xlsx'),
                                 recursive=True)
            if 'Anexos' in p and not os.path.basename(p).startswith('~$')
            and 'dicion' in os.path.basename(p).lower()]
    if len(arqs) != 1:
        raise FileNotFoundError(f'Dicionario de {ano}: esperado 1 arquivo, encontrados {arqs}')
    wb = openpyxl.load_workbook(arqs[0], read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]  # aba da tabela de escolas
    registros = {}
    for r in ws.iter_rows(values_only=True):
        if r and len(r) > 5 and isinstance(r[1], str) and re.fullmatch(r'[A-Z0-9_]+', r[1]):
            registros[r[1]] = {'descricao': _normalizar(r[2]), 'tipo': _normalizar(r[3]),
                               'tamanho': _normalizar(r[4]), 'categorias': _normalizar(r[5])}
    wb.close()
    return registros


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dicionarios', required=True,
                        help='Diretorio com as pastas microdados_censo_escolar_<ano> do INEP')
    parser.add_argument('--saida', default=os.path.join(RAIZ, 'docs', 'variaveis_utilizadas.csv'))
    args = parser.parse_args()

    dicionarios = {ano: ler_dicionario(args.dicionarios, ano) for ano in ANOS}
    cabecalhos = {
        ano: set(pd.read_csv(os.path.join(config.BRONZE_DIR, config.NOME_ARQUIVO_PADRAO.format(ano=ano)),
                             sep=';', encoding='latin1', nrows=0).columns)
        for ano in ANOS
    }
    prata = {ano: pd.read_parquet(os.path.join(config.PRATA_DIR, f'censo_{ano}.parquet')) for ano in ANOS}
    colunas_ouro = {os.path.basename(p): set(pd.read_csv(p, nrows=0).columns)
                    for p in glob.glob(os.path.join(config.OURO_DIR, '*.csv'))}

    linhas = []
    for grupo, variaveis in GRUPOS.items():
        for var in variaveis:
            nome_pad = config.MAPEAMENTO_COLUNAS.get(var, var)
            ref = dicionarios[ANO_DESCRICAO].get(var, {})
            difere = [str(a) for a in ANOS
                      if a != ANO_DESCRICAO and var in dicionarios[a]
                      and dicionarios[a][var]['descricao'] != ref.get('descricao', '')]
            ausente_dic = [str(a) for a in ANOS if var not in dicionarios[a]]
            tipos_prata = sorted({str(prata[a][nome_pad].dtype) for a in ANOS if nome_pad in prata[a].columns})
            saidas = sorted(f for f, cols in colunas_ouro.items() if nome_pad in cols or var in cols)
            indicador, uso = USO[var]
            linhas.append({
                'nome_inep': var,
                'nome_padronizado': nome_pad,
                'grupo_funcional': grupo,
                'tipo_inep_2024': ref.get('tipo', ''),
                'tamanho_inep_2024': ref.get('tamanho', ''),
                'tipo_prata': ';'.join(tipos_prata),
                'anos_na_bronze': ';'.join(str(a) for a in ANOS if var in cabecalhos[a]),
                'anos_na_prata': ';'.join(str(a) for a in ANOS if nome_pad in prata[a].columns),
                'indicador_ou_papel': indicador,
                'uso_no_pipeline': uso,
                'colunas_em_saidas_ouro': ';'.join(saidas),
                'descricao_inep_2024': ref.get('descricao', ''),
                'categorias_inep_2024': ref.get('categorias', ''),
                'descricao_difere_nos_anos': ';'.join(difere),
                'ausente_no_dicionario_dos_anos': ';'.join(ausente_dic),
            })

    df = pd.DataFrame(linhas)
    os.makedirs(os.path.dirname(args.saida), exist_ok=True)
    df.to_csv(args.saida, index=False, encoding='utf-8')
    sem_desc = df.loc[df['descricao_inep_2024'] == '', 'nome_inep'].tolist()
    print(f'{len(df)} variaveis gravadas em {args.saida}')
    print('Sem descricao no dicionario de 2024:', sem_desc or 'nenhuma')
    print('Descricao difere de 2024 em outros anos:', df.loc[df['descricao_difere_nos_anos'] != '', 'nome_inep'].tolist())


if __name__ == '__main__':
    main()
