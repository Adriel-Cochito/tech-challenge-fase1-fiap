from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Inicializa o app FastAPI
app = FastAPI(
    title="API de Suporte ao Diagnóstico de Câncer de Mama",
    description="Interface para inferência de lesões mamárias (Maligno vs Benigno)",
    version="1.0.0"
)

MODEL_PATH = "models/breast_cancer_model.pkl"

# Estrutura para os dados de entrada (30 características extraídas do exame)
class PatientFeatures(BaseModel):
    features: list[float]

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Modelo recarregado da memória com sucesso.")
    else:
        raise FileNotFoundError(f"Arquivo de modelo não encontrado em {MODEL_PATH}. Execute o pipeline primeiro.")

@app.get("/")
def home():
    return {"message": "API de Predição de Câncer de Mama operacional."}

@app.post("/predict")
def predict(data: PatientFeatures):
    if len(data.features) != 30:
        raise HTTPException(
            status_code=400, 
            detail="É necessário fornecer exatamente 30 valores numéricos das características do exame."
        )
    
    # Preparar array para o modelo
    input_data = np.array(data.features).reshape(1, -1)
    
    # Realizar predição e probabilidade
    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]
    
    label_map = {0: "Maligno", 1: "Benigno"}
    
    return {
        "diagnostico_previsto": label_map[prediction],
        "classe": prediction,
        "probabilidades": {
            "Maligno": round(float(probabilities[0]), 4),
            "Benigno": round(float(probabilities[1]), 4)
        }
    }