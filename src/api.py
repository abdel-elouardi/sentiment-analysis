# ============================================
# FICHIER : api.py
# RÔLE : API FastAPI pour la prédiction
# À MODIFIER : les champs du modèle Pydantic
# UTILISÉ PAR : Railway (déploiement)
# ============================================

import torch
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.cleaning import clean_text

app = FastAPI(title="API Analyse de Sentiment")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement du modèle et du vectorizer
model = torch.load('models/model.pt', weights_only=False)
vectorizer = torch.load('models/vectorizer.pt', weights_only=False)
model.eval()

class Review(BaseModel):
    # ⚠️ À MODIFIER : selon ton projet
    text: str

@app.get("/")
def home():
    return {"message": "API Analyse de Sentiment - Bienvenue !"}

@app.get("/health")
def health():
    return {"status": "API en ligne"}

@app.post("/predict")
def predict(review: Review):
    # Nettoyer le texte
    clean = clean_text(review.text)

    # Vectoriser
    X = vectorizer.transform([clean]).toarray()
    tensor = torch.FloatTensor(X)

    # Prédire
    with torch.no_grad():
        output = model(tensor).item()

    prediction = 1 if output >= 0.5 else 0
    label = "Positif 😊" if prediction == 1 else "Négatif 😞"

    return {
        "prediction": prediction,
        "label": label,
        "probabilité": round(output * 100, 2)
    }