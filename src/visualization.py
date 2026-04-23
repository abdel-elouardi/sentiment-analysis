# ============================================
# FICHIER : visualization.py
# RÔLE : Visualiser les données texte
# À MODIFIER : selon tes colonnes
# UTILISÉ PAR : main.py
# ============================================

import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

def plot_sentiment_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='sentiment', data=df,
                  hue='sentiment', palette='Set2', legend=False)
    plt.title("Distribution des sentiments")
    plt.xticks([0, 1], ['Négatif', 'Positif'])
    plt.tight_layout()
    plt.savefig("sentiment_distribution.png")
    plt.show()
    print("✅ sentiment_distribution.png sauvegardé")

def plot_text_length(df):
    df['text_length'] = df['clean_text'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(10, 4))
    sns.histplot(data=df, x='text_length', hue='sentiment',
                 kde=True, palette='Set2')
    plt.title("Longueur des textes par sentiment")
    plt.tight_layout()
    plt.savefig("text_length.png")
    plt.show()
    print("✅ text_length.png sauvegardé")

def plot_top_words(df):
    pos_words = ' '.join(df[df['sentiment']==1]['clean_text']).split()
    neg_words = ' '.join(df[df['sentiment']==0]['clean_text']).split()

    pos_common = Counter(pos_words).most_common(10)
    neg_common = Counter(neg_words).most_common(10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh([w[0] for w in pos_common], [w[1] for w in pos_common], color='#00b894')
    axes[0].set_title("Top 10 mots positifs")
    axes[0].invert_yaxis()

    axes[1].barh([w[0] for w in neg_common], [w[1] for w in neg_common], color='#d63031')
    axes[1].set_title("Top 10 mots négatifs")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig("top_words.png")
    plt.show()
    print("✅ top_words.png sauvegardé")

def visualize_all(df):
    print("🎨 Génération des graphiques...\n")
    plot_sentiment_distribution(df)
    plot_text_length(df)
    plot_top_words(df)
    print("\n✅ Tous les graphiques générés !")