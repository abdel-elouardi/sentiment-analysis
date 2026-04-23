# ============================================
# FICHIER : model.py
# RÔLE : Construire et entraîner le modèle NLP
# À MODIFIER : epochs, hidden_size si besoin
# UTILISÉ PAR : main.py et api.py
# ============================================

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def prepare_data(df, target_col='sentiment', text_col='clean_text'):
    # Vectorisation TF-IDF
    # ⚠️ À MODIFIER : max_features selon la taille du dataset
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df[text_col]).toarray()
    y = df[target_col].values

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Conversion en tenseurs PyTorch
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)

    print(f"✅ Train : {X_train.shape[0]} lignes")
    print(f"✅ Test  : {X_test.shape[0]} lignes")

    return X_train, X_test, y_train, y_test, vectorizer

class SentimentModel(nn.Module):
    # ⚠️ À MODIFIER : input_size = max_features du TfidfVectorizer
    def __init__(self, input_size=5000):
        super(SentimentModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

def train_model(X_train, y_train, epochs=10):
    model = SentimentModel()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\n=== Entraînement du modèle ===")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train).squeeze()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    return model

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test).squeeze()
        predicted = (outputs >= 0.5).float()
        accuracy = (predicted == y_test).float().mean()
        print(f"\n=== Résultats ===")
        print(f"✅ Précision : {accuracy.item() * 100:.2f}%")

def save_model(model, vectorizer, 
               model_path="models/model.pt",
               vectorizer_path="models/vectorizer.pt"):
    torch.save(model, model_path)
    torch.save(vectorizer, vectorizer_path)
    print(f"✅ Modèle sauvegardé dans {model_path}")
    print(f"✅ Vectorizer sauvegardé dans {vectorizer_path}")