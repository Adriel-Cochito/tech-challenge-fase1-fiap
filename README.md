
# 🏥 Tech Challenge - Fase 1: Inteligência Artificial na Saúde

Sistema inteligente de suporte ao diagnóstico clínico para análise de exames e dados médicos (Foco: Câncer de Mama - Maligno vs. Benigno)[cite: 13].

---

## 🎯 Sobre o Desafio
Este projeto faz parte do **Tech Challenge** da Pós-Tech[cite: 13]. O objetivo é desenvolver uma solução prática baseada em Machine Learning para auxiliar equipes médicas de um grande hospital universitário na classificação automatizada de lesões mamárias a partir de dados estruturados[cite: 13].

---

## 📂 Estrutura do Repositório

```text
tech-challenge-fase1/
├── data/
│   └── README.md              # Documentação e origem do dataset (Wisconsin Breast Cancer)
├── notebooks/
│   └── tech_challenge_fase1.ipynb  # Jupyter Notebook completo com código, gráficos e análises
├── src/
│   ├── pipeline.py            # Script Python automatizado de treino, validação e persistência
│   └── app.py                 # API REST (FastAPI) para inferência do modelo em produção
├── models/
│   └── best_model.pkl         # Artefato gerado do modelo treinado (SVM)
├── Dockerfile                 # Configuração de containerização da aplicação
├── requirements.txt           # Dependências e bibliotecas do projeto
└── README.md                  # Documentação principal do repositório

```

---

## 📊 Dataset Utilizado

* **Nome:** Breast Cancer Wisconsin (Diagnostic) Database


* **Fonte:** Repositório UCI / [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data) (também carregável diretamente via `scikit-learn`).


* **Instâncias:** 569 amostras de características nucleares extraídas de imagens de punção aspirativa.


* **Classes:**
* `0` = Maligno


* `1` = Benigno





---

## 🚀 Como Executar o Projeto

### Pré-requisitos

* Python 3.10 ou superior instalado.
* Git instalado.

### 1. Instalação Local e Ambiente Virtual

Abra o terminal na pasta raiz do projeto e execute:

```bash
python -m venv venv
# No Windows PowerShell:
.\venv\Scripts\Activate
# Instale as dependências:
pip install --upgrade pip
pip install -r requirements.txt

```

### 2. Executando o Notebook Jupyter

Para visualizar o passo a passo interativo, gráficos de matriz de confusão, curva ROC e PCA:

```bash
python -m jupyter notebook notebooks/tech_challenge_fase1.ipynb

```

### 3. Executando via API (FastAPI)

Para testar o modelo em um ambiente simulado de produção:

```bash
python src/app.py

```

---

## 🐳 Executando com Docker

Se preferir executar o projeto isolado em um container Docker:

1. Construa a imagem Docker:
```bash
docker build -t tech-challenge-fase1 .

```


2. Execute o container:
```bash
docker run -p 8000:8000 tech-challenge-fase1

```



---

## 📈 Metodologia e Resultados Principais

1. **Pré-processamento e Feature Scaling:** Uso de `StandardScaler` embutido em `Pipeline` do Scikit-Learn para evitar vazamento de dados (*data leakage*).


2. **Validação Cruzada ($CV = 5$):** Comparação entre Regressão Logística, KNN, Random Forest e SVM. O **SVM (Support Vector Machine)** e a Regressão Logística destacaram-se com acurácias superiores a $97\%$.


3. **Métricas Clínicas:** Foco especial na métrica de **Recall (Sensibilidade)** para minimizar o risco de falsos negativos em casos malignos, atingindo excelente performance operacional.


4. **Redução de Dimensionalidade:** Aplicação de PCA (Análise de Componentes Principais) para visualização espacial da separabilidade das classes.



---

## 👥 Integrantes do Grupo

* Adriel Cochito
