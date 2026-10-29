import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def run_pipeline():
    print("--- 1. CARREGANDO DADOS ---")
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target  # 0: Maligno, 1: Benigno

    X = df.drop(columns=['target'])
    y = df['target']

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n--- 2. AVALIANDO MODELOS VIA VALIDAÇÃO CRUZADA ---")
    pipelines = {
        'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())]),
        'KNN': Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier(n_neighbors=5))]),
        'SVM': Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True, random_state=42))]),
        'Random Forest': Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(random_state=42))])
    }

    best_score = 0
    best_pipeline_name = None
    best_pipeline = None

    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
        mean_score = scores.mean()
        print(f"[{name}] Acurácia Média na Validação Cruzada: {mean_score:.4f} (+/- {scores.std():.4f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_pipeline_name = name
            best_pipeline = pipe

    print(f"\nModelo Selecionado: {best_pipeline_name}")
    
    print("\n--- 3. TREINANDO E AVALIANDO O MELHOR MODELO ---")
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]

    print("\nRelatório de Classificação (Teste):")
    print(classification_report(y_test, y_pred, target_names=['Maligno', 'Benigno']))

    auc = roc_auc_score(y_test, y_prob)
    print(f"AUC-ROC Score: {auc:.4f}")

    # Criando pasta para salvar artefatos
    os.makedirs('models', exist_ok=True)
    model_path = 'models/breast_cancer_model.pkl'
    
    # Salvando o pipeline completo (com o scaler ajustado)
    joblib.dump(best_pipeline, model_path)
    print(f"\nModelo serializado e salvo com sucesso em: {model_path}")

if __name__ == "__main__":
    run_pipeline()