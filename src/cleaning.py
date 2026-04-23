# ============================================
# FICHIER : cleaning.py
# RÔLE : Charger et nettoyer les données texte
# À MODIFIER : le nom du fichier CSV et les colonnes
# UTILISÉ PAR : main.py et api.py
# ============================================

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

# Télécharger les stopwords NLTK
nltk.download('stopwords', quiet=True)

def load_data(path, nrows=10000):
    # ⚠️ À MODIFIER : changer les colonnes selon ton dataset
    df = pd.read_csv(path, nrows=nrows)
    print(f"✅ Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df

def get_info(df):
    print("\n=== .info() ===")
    print(df.info())
    print("\n=== .describe() ===")
    print(df.describe())
    print("\n=== Valeurs manquantes ===")
    print(df.isna().sum())

def clean_text(text):
    # Convertir en minuscules
    text = str(text).lower()
    # Supprimer les caractères spéciaux
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    # Supprimer les stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

def prepare_sentiment(df):
    # ⚠️ À MODIFIER : changer 'Score' et 'Text' selon ton dataset
    # Garder uniquement les colonnes utiles
    df = df[['Score', 'Text']].dropna()
    
    # Créer la colonne sentiment : 1 = positif, 0 = négatif
    # Score 4-5 = positif, Score 1-2 = négatif, Score 3 = neutre (ignoré)
    df = df[df['Score'] != 3]
    df['sentiment'] = (df['Score'] >= 4).astype(int)
    
    # Nettoyer le texte
    print("🧹 Nettoyage du texte...")
    df['clean_text'] = df['Text'].apply(clean_text)
    
    print(f"✅ Positifs : {df['sentiment'].sum()}")
    print(f"✅ Négatifs : {len(df) - df['sentiment'].sum()}")
    
    return df

def clean_data(df):
    print("🧹 Début du nettoyage...\n")
    df = prepare_sentiment(df)
    print("\n✅ Nettoyage terminé !")
    return df