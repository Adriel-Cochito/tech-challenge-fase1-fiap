Aqui está o **Roteiro Completo para o Vídeo de Demonstração**, dividido passo a passo por cada fase do código e de suas respectivas saídas. O roteiro foi redigido com uma linguagem clara e natural para que você possa ler, explicar o código, detalhar as saídas e contextualizar a importância clínica de cada etapa durante a gravação da sua tela.

---

# 🎬 Roteiro do Vídeo Explicativo - Tech Challenge Fase 1

**Tempo Estimado:** 8 a 10 minutos

**Dica de Gravação:** Deixe a tela do seu Jupyter Notebook aberta no VS Code ou navegador e siga o fluxo abaixo.

---

### Bloco 1: Introdução e Contextualização do Projeto

* **O que mostrar na tela:** A estrutura inicial do repositório ou o topo do Jupyter Notebook.
* **O que falar:**
> "Olá! Sejam bem-vindos à apresentação do projeto da Fase 1 do Tech Challenge da Pós-Tech de Inteligência Artificial da FIAP. O nosso desafio consiste em desenvolver um sistema inteligente de suporte ao diagnóstico clínico para um grande hospital universitário, com foco na classificação automatizada de lesões mamárias em malignas ou benignas a partir de dados numéricos extraídos de exames de punção aspirativa. O objetivo aqui não é substituir o médico, mas fornecer uma ferramenta preditiva de alta precisão para auxiliar na tomada de decisão rápida e segura."
> 
> 



---

### Bloco 2: Fase de Ingestão e Exploração dos Dados

* **O que mostrar na tela:** Célula de código `[2]` e a saída tabular do DataFrame com as 5 linhas e as métricas de contagem das classes.
* **O que falar:**
> "Na primeira fase prática do nosso código, realizamos a importação das bibliotecas fundamentais e o carregamento da base de dados *Breast Cancer Wisconsin* diretamente pelo `scikit-learn`. Como vocês podem observar na saída da célula, o dataset possui 569 amostras e 30 atributos preditivos numéricos, que descrevem características como raio, textura, perímetro e concavidade dos núcleos celulares. Temos também a distribuição das classes na variável alvo: 357 casos benignos e 212 malignos. Esse leve desbalanceamento é tratado logo em seguida na divisão dos dados."
> 
> 



---

### Bloco 3: Fase de Pré-processamento e Pipelines

* **O que mostrar na tela:** A célula de código `[4]` que define os modelos e as instâncias de `StandardScaler` dentro do `Pipeline`.
* **O que falar:**
> "Avançando para Machine Learning Avançado, um dos pontos mais críticos do projeto foi evitar o vazamento de dados (*data leakage*). Para isso, estruturamos pipelines que unem o `StandardScaler` diretamente aos estimadores. O scaler é ajustado estritamente dentro do conjunto de treino e aplicado no teste. Criamos quatro abordagens para comparação: Regressão Logística, K-Nearest Neighbors (KNN), Support Vector Machine (SVM) e Random Forest."



---

### Bloco 4: Fase de Treinamento e Validação Cruzada ($CV = 5$)

* **O que mostrar na tela:** A célula de código `[5]` e os valores impressos no console com as acurácias médias e desvios padrão.
* **O que falar:**
> "Executando o treinamento e a Validação Cruzada com 5 folds, conseguimos testar a robustez de cada algoritmo de forma isolada de divisões aleatórias enviesadas. Olhando para as saídas na tela, a Regressão Logística obteve uma acurácia média de 98,02%, seguida de perto pelo SVM com 97,14%, o KNN com 96,70% e Random Forest com 95,38%. Esses resultados iniciais demonstram que os dados possuem um forte padrão de separabilidade linear."



---

### Bloco 5: Fase de Avaliação no Conjunto de Teste e Métricas Clínicas (SVM)

* **O que mostrar na tela:** As saídas da célula `[6]`, exibindo o *Classification Report*, a Matriz de Confusão e o gráfico da Curva ROC.
* **O que falar:**
> "Escolhemos o modelo SVM para aprofundar a avaliação no conjunto de teste independente com 114 amostras. No contexto de saúde, a métrica mais importante é o **Recall ou Sensibilidade**, pois um falso negativo — ou seja, classificar um tumor maligno como benigno — pode ser fatal para o paciente. Analisando a matriz de confusão, o modelo acertou 41 dos 42 casos malignos reais, registrando apenas **1 único Falso Negativo**. Isso nos garantiu um Recall de 98% para malignidade e uma acurácia global de 98%. Além disso, a Curva ROC exibida na tela atingiu um AUC Score espetacular de 0.9950, provando a excelente capacidade discriminatória do modelo."
> 
> 



---

### Bloco 6: Fase de Redução de Dimensionalidade com PCA

* **O que mostrar na tela:** O gráfico de dispersão gerado pelo PCA (célula `[7]`), destacando os pontos vermelhos (malignos) e verdes (benignos).
* **O que falar:**
> "Para finalizar a exploração visual do nosso modelo, aplicamos a Análise de Componentes Principais (PCA) para reduzir a complexidade de 30 dimensões para um plano cartesiano 2D. Como podemos ver no gráfico de dispersão, há uma separação espacial muito clara entre os tumores benignos (em verde, concentrados à esquerda) e os malignos (em vermelho, à direita). Essa clara separação visual justifica o porquê de algoritmos baseados em hiperplanos, como o SVM, terem performado tão bem na tarefa de classificação."



---

### Bloco 7: Considerações Finais e Deploy

* **O que mostrar na tela:** O arquivo `Dockerfile` ou a estrutura de pastas do projeto no VS Code.
* **O que falar:**
> "Para garantir que esta solução saia do ambiente de laboratório e possa ser integrada de fato na rotina de um hospital universitário, estruturamos o código de forma modular na pasta `src`, com pipelines de salvamento de artefatos `.pkl` e suporte completo a containerização via Docker. Com isso, a API pode ser facilmente implantada em servidores seguros da instituição. Agradeço a atenção de todos e encerro aqui a demonstração do Tech Challenge da Fase 1!"



---