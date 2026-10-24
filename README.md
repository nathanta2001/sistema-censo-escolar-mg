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
* **I<sub>rec</sub> (Indicador de Recuperação):** Avalia a taxa de recomposição das matrículas em 2024 em relação à perda sofrida entre 2019 e 2021.
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
* **`src/stats.py`**: Testes de normalidade, correlações (Pearson/Spearman com bootstrap e correção de Bonferroni) e modelos de regressão.
* **`src/sensibilidade.py`**: Análise de robustez do indicador de recuperação e controle por porte municipal.
* **`src/visualizacao.py`**: Geração de mapas coropléticos regionais e gráficos de dispersão/LOWESS.
* **`Dashboard.py`**: Interface web analítica construída em Streamlit.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.13
* **Manipulação e Engenharia de Dados:** `pandas`, `pyarrow`, `fastparquet`, `openpyxl`
* **Computação Estatística e Modelagem:** `scipy`, `statsmodels`, `numpy`
* **Análise Espacial e Visualização:** `geopandas`, `plotly`, `matplotlib`, `seaborn`
* **Deploy e Aplicação Web:** `streamlit`, `streamlit-cloud`

---

## 🚀 Como Executar Localmente

### 1. Pré-requisitos
* Git instalado
* Python 3.10 ou superior

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

# Instalar dependências
pip install -r requirements.txt
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
# Baixar microdados do Inep (camada Bronze)
python -m src.extract

# Processar camada Prata e gerar camada Ouro com os indicadores
python -m src.analysis

# Executar análises estatísticas e gráficos
python -m src.stats
python -m src.visualizacao
```

---

## 📌 Reprodutibilidade e Versionamento

O código e os dados derivados deste trabalho seguem princípios de ciência aberta:
* As versões estáveis do pipeline e o estado da entrega acadêmica estão demarcados com tags Git (ex.: `v1.0.0`).
* As dependências e requisitos de compilação estão descritos em `requirements.txt`.
