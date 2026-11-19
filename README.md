# Sistema de Engenharia e Análise de Dados — Censo Escolar MG (2017–2024)

Repositório oficial do Trabalho de Conclusão de Curso em Engenharia de Computação pelo **Centro Federal de Educação Tecnológica de Minas Gerais (CEFET-MG)**.

Este projeto implementa um pipeline de dados ponta a ponta na **Arquitetura Medalhão (Bronze, Prata e Ouro)** para investigar os impactos da pandemia de COVID-19 nas matrículas da Educação Básica dos 853 municípios de Minas Gerais, integrando modelagem econométrica (OLS, Logit fracionário, regressão logística), análise espacial e um dashboard analítico interativo.

---

## 🌐 Acesso Rápido ao Dashboard Online

O painel interativo está publicado e acessível publicamente via Streamlit Community Cloud:

🔗 **[Acessar Dashboard Interativo](https://sistema-censo-escolar-mg-4yzptpydm4p52j9tnycspm.streamlit.app/)**

---

## 📊 Principais Indicadores Desenvolvidos

* **I<sub>imp</sub> (Indicador de Impacto):** Mede a variação percentual de matrículas entre o período pré-pandêmico (2019) e o pico da pandemia (2021).
* **I<sub>rec</sub> (Indicador de Recuperação):** Avalia a taxa de recomposição das matrículas em 2024 em relação à perda sofrida entre 2019 e 2021: I<sub>rec</sub> = (M2024 − M2021) / (M2019 − M2021) × 100. É definido **apenas para municípios com perda** (M2021 < M2019, coluna `teve_perda = True`). Os municípios sem perda (M2021 ≥ M2019) permanecem na tabela Ouro com `teve_perda = False` e `I_rec` vazio (NaN), ficam fora das correlações, regressões e da análise de sensibilidade do I<sub>rec</sub> e são descritos à parte (`i_rec_sem_perda_*`).
* **I<sub>EAD</sub> (Indicador de EAD):** Proporção de matrículas formalmente registradas na modalidade a distância no Censo Escolar 2021.
* **I<sub>ERE</sub> (Mediação Remota de Emergência):** Média de dias de ensino remoto declarados pelas escolas nos anos iniciais do Ensino Fundamental (pesquisa suplementar COVID-19/Inep).
* **ICT (Índice de Capacidade Tecnológica):** Métrica agregada da infraestrutura das escolas (computadores, internet para aprendizagem e banda larga).

---

## 🏗️ Arquitetura do Pipeline de Dados

O projeto segue a separação em camadas Medalhão:

```text
├── data/
│   ├── bronze/     # Microdados brutos baixados do Inep (CSVs compactados)
│   ├── prata/      # Dados filtrados para MG, limpos, tratados e convertidos para Parquet
│   ├── ouro/       # Bases analíticas consolidadas por município/região e matrizes estatísticas
│   └── externos/   # Questionário COVID-19 (XLSX) e Shapefile IBGE das Regiões Intermediárias
```

* **`src/extract.py`**: Download automatizado com suporte a retomada (`HTTP Range`) dos microdados do Inep (2017–2024).
* **`src/transform.py`**: Leitura em chunks, filtro territorial (MG / código `31`), tipagem de baixo consumo de memória e geração da camada Prata (`.parquet`).
* **`src/analysis.py`**: Cálculo de indicadores municipais (I_imp, I_rec, I_EAD, ICT) e consolidação da camada Ouro.
* **`src/pesquisa_pandemia.py`**: Ingestão da Pesquisa de Resposta Educacional à Pandemia do Inep para cálculo do I_ERE.
* **`src/stats.py`**: Testes de normalidade, correlações (Pearson/Spearman com bootstrap de 1.000 reamostras, seed 42, e correção de Bonferroni, α* = 0,005), regressões OLS e logit truncado (ε = 10⁻⁴) do I<sub>rec</sub>, robustez de ε, regressão logística de recuperação plena (I<sub>rec</sub> ≥ 100), controle por porte (log M2019) e descrição dos grupos com e sem perda. Toda análise do I<sub>rec</sub> usa apenas `teve_perda == True`.
* **`src/sensibilidade.py`**: Robustez do I<sub>rec</sub> a perdas pequenas: filtro pela perda real (M2019 − M2021) ≥ corte, com cortes de 20, 50, 100 e 200 matrículas, e critério alternativo por M2019.
* **`src/visualizacao.py`**: Geração de mapas coropléticos regionais (I<sub>rec</sub>: mediana dos municípios com perda; mapa próprio, em cinza, da proporção de municípios sem perda) e gráficos de dispersão/LOWESS.
* **`Dashboard.py`**: Interface web analítica construída em Streamlit.
* **`tests/`**: Testes automatizados (pytest) do I<sub>rec</sub>, de `teve_perda` e do filtro de sensibilidade.
* **`scripts/`**: Scripts de documentação da base: `listar_variaveis.py` (tabela das variáveis selecionadas) e `contar_base_por_ano.py` (registros e colunas por ano, da Bronze à Prata).
* **`docs/`**: Saídas desses scripts: `variaveis_utilizadas.csv` e `contagem_base_por_ano.csv`.

### Principais saídas da camada Ouro (`data/ouro/`, em `.csv` e `.parquet`)

| Arquivo | Conteúdo |
|---|---|
| `indicadores_municipios` | Indicadores por município (inclui `teve_perda` e `n_escolas_ere`) |
| `matriz_correlacoes`, `regressoes_i_rec` | Correlações dos 10 pares e regressões dos preditores significativos de I<sub>rec</sub> |
| `i_rec_robustez_epsilon`, `i_rec_logistica_recuperacao_plena`, `controle_porte` | Robustez de ε, regressão logística (com e sem log M2019) e controle por porte |
| `i_rec_descritiva_por_grupo`, `i_rec_trajetoria_com_perda` | I<sub>rec</sub> por grupo e trajetória dos municípios com perda |
| `i_rec_sem_perda_resumo`, `i_rec_sem_perda_por_porte`, `i_rec_sem_perda_por_regiao` | Municípios sem perda: contagem, porte e região |
| `sensibilidade_i_rec_{descritiva,correlacoes}_{perda,m2019}` | Análise de sensibilidade do I<sub>rec</sub> |
| `i_ead_spearman_variantes`, `sensibilidade_i_ere_escolas` | I<sub>EAD</sub> contínuo, binário e só entre I<sub>EAD</sub> > 0; I<sub>ERE</sub> excluindo municípios com ≤ 2 escolas |
| `serie_matriculas_regiao`, `serie_matriculas_dependencia` | Séries anuais de matrículas (2017–2024) por região intermediária e por município e dependência administrativa |

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.13
* **Manipulação e Engenharia de Dados:** `pandas`, `pyarrow`, `openpyxl`
* **Computação Estatística e Modelagem:** `scipy`, `statsmodels`, `numpy`
* **Análise Espacial e Visualização:** `geopandas`, `plotly`, `matplotlib`, `seaborn`
* **Aplicação Web:** `streamlit` (hospedagem no Streamlit Community Cloud)
* **Rede e Extração:** `requests`, `certifi`, `truststore`

---

## 🚀 Como Executar Localmente

### 1. Pré-requisitos
* Git instalado
* Python 3.11 ou superior (testado com Python 3.13.7)

### 2. Clonar o Repositório e Configurar Ambiente
```bash
# Clonar o repositório
git clone https://github.com/nathanta2001/sistema-censo-escolar-mg.git
cd sistema-censo-escolar-mg

# Criar e ativar o ambiente virtual
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Instalar dependências (diretas, com versões fixadas)
pip install -r requirements.txt

# Ou, para recriar exatamente o ambiente que gerou as saídas:
# pip install -r requirements-lock.txt
```

### 3. Executar o Dashboard
Como a camada Ouro (`data/ouro/*.parquet`) e os dados geográficos já acompanham o repositório, o dashboard pode ser iniciado imediatamente:
```bash
streamlit run Dashboard.py
```

### 4. (Opcional) Executar o Pipeline Completo
Para reproduzir a extração dos microdados brutos e o recálculo dos indicadores:
1. Certifique-se de que o arquivo da pesquisa suplementar esteja em `data/externos/Questionario_Resposta_Educacional_a_Pandemia_de_Covid_19_2ed.xlsx`.
2. Execute o pipeline:
```bash
# Baixa os microdados (Bronze), gera a Prata, os indicadores e a análise
# estatística na camada Ouro
python main.py

# Análise de sensibilidade do I_rec
python analise_sensibilidade.py

# Gráficos e mapas (data/ouro/graficos/)
python gerar_graficos.py
```

### 5. Testes
```bash
pip install -r requirements-dev.txt
python -m pytest
```

### 6. (Opcional) Documentação da base
Requer a camada Bronze em `data/bronze/`. O dicionário de dados do INEP não acompanha o repositório; informe o diretório com as pastas `microdados_censo_escolar_<ano>` baixadas do INEP:
```bash
python scripts/listar_variaveis.py --dicionarios <dir_microdados_inep>
python scripts/contar_base_por_ano.py --microdados-inep <dir_microdados_inep>   # o argumento é opcional (confere o md5)
```

---

## 📌 Reprodutibilidade e Versionamento

O código e os dados derivados deste trabalho seguem princípios de ciência aberta:
* As versões estáveis do pipeline estão demarcadas com tags Git; a versão atual é a `v1.1.0`.
* As dependências diretas, com versões exatas, estão em `requirements.txt`; o ambiente completo (`pip freeze`), em `requirements-lock.txt`; as dependências de teste, em `requirements-dev.txt`.
