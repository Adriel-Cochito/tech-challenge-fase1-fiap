# Dataset: Câncer de Mama (Wisconsin Breast Cancer)

## 1. Origem dos Dados
Este conjunto de dados é o **Breast Cancer Wisconsin (Diagnostic) Database**, amplamente utilizado na literatura de Machine Learning para tarefas de classificação binária médica. 
- **Fonte Oficial / Referência:** UCI Machine Learning Repository / [Kaggle - Breast Cancer Wisconsin Data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)[cite: 13].
- **Disponibilidade Programática:** Pode ser carregado diretamente em Python utilizando a biblioteca `scikit-learn`:
  ```python
  from sklearn.datasets import load_breast_cancer
  data = load_breast_cancer()
---

## 2. Descrição e Características do Dataset

O dataset contém características numéricas computadas a partir de imagens digitalizadas de punções aspirativas por agulha fina (PAAF) de massas mamárias. Elas descrevem características das células nucleares presentes na imagem.

* **Número de Amostras (Instâncias):** 569 registros


* **Número de Atributos (Features):** 30 variáveis preditoras numéricas contínuas


* **Variável Alvo (Target / Rótulo):**
* `0` = Maligno (Tumor cancerígeno)


* `1` = Benigno (Tumor não cancerígeno)





---

## 3. Principais Atributos (Features)

Para cada imagem, são calculados 10 atributos fundamentais em três dimensões (média, erro padrão e o "pior" ou maior valor medido):

1. **Radius** (média das distâncias do centro aos pontos do perímetro)
2. **Texture** (desvio padrão dos valores da escala de cinza)
3. **Perimeter** (perímetro do núcleo celular)
4. **Area** (área do núcleo celular)
5. **Smoothness** (variação local nos comprimentos dos raios)
6. **Compactness** (perímetro² / área - 1.0)
7. **Concavity** (severidade das porções côncavas do contorno)
8. **Concave Points** (número de porções côncavas do contorno)
9. **Symmetry** (simetria celular)
10. **Fractal Dimension** ("aproximação da costa" - 1)

---

## 4. Tratamento e Pré-processamento Aplicados no Projeto

* **Limpeza:** A base possui excelente integridade, não apresentando valores nulos ou ausentes (*missing values*).
* **Padronização (Feature Scaling):** Aplicou-se o `StandardScaler` para centralizar as variáveis em média zero e desvio padrão unitário, garantindo estabilidade e convergência otimizada para modelos baseados em distâncias e hiperplanos (como SVM e KNN).
* **Divisão Estratificada:** O conjunto foi dividido em treino (80%) e teste (20%) utilizando amostragem estratificada (`stratify=y`) para preservar a proporção real entre casos benignos e malignos.
