# ============================================================
# src/viz.py
# Fonctions de visualisation des discours présidentiels
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Sentiment dans le temps ────────────────────────────
def plot_sentiment_over_time(df):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Extraire l'année
    years = df["date"].str.extract(r"(\d{4})")[0].astype(int)

    # TextBlob polarité
    axes[0].plot(years, df["polarity"], marker="o", color="steelblue",
                 linewidth=2, markersize=4)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].fill_between(years, df["polarity"], 0,
                         where=df["polarity"] >= 0, alpha=0.2, color="green")
    axes[0].fill_between(years, df["polarity"], 0,
                         where=df["polarity"] < 0,  alpha=0.2, color="red")
    axes[0].set_title("Polarité TextBlob au fil du temps", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Année")
    axes[0].set_ylabel("Polarité")
    axes[0].grid(True, alpha=0.3)

    # VADER compound
    colors = ["green" if c >= 0.05 else "red" if c <= -0.05 else "gray"
              for c in df["vader_compound"]]
    axes[1].bar(years, df["vader_compound"], color=colors, alpha=0.7, width=2)
    axes[1].axhline(0, color="black", linestyle="-", linewidth=0.8)
    axes[1].set_title("Score VADER Compound au fil du temps", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Année")
    axes[1].set_ylabel("Score Compound")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/01_sentiment_over_time.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f" Sauvegardé : {path}")


# ── 2. Distribution des tonalités ─────────────────────────
def plot_tone_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    tone_counts = df["tone"].value_counts()
    colors = {"Positif": "#2ecc71", "Négatif": "#e74c3c", "Neutre": "#95a5a6"}
    axes[0].pie(tone_counts.values,
                labels=tone_counts.index,
                colors=[colors[t] for t in tone_counts.index],
                autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 12})
    axes[0].set_title("Répartition des tonalités (VADER)", fontsize=14, fontweight="bold")

    # Subjectivité par président (top 15)
    top15 = df.nlargest(15, "subjectivity")[["president", "subjectivity"]]
    axes[1].barh(top15["president"], top15["subjectivity"],
                 color="coral", alpha=0.8)
    axes[1].set_title("Top 15 discours les plus subjectifs (TextBlob)",
                      fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Subjectivité")
    axes[1].grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/02_tone_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f" Sauvegardé : {path}")


# ── 3. Wordcloud global ───────────────────────────────────
def plot_wordcloud_global(df):
    all_words = " ".join(
        [" ".join(tokens) for tokens in df["lemmas"]]
    )
    wc = WordCloud(width=1200, height=600, background_color="white",
                   colormap="Blues", max_words=150,
                   collocations=False).generate(all_words)

    plt.figure(figsize=(16, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Mots les plus fréquents — Tous les discours",
              fontsize=16, fontweight="bold", pad=20)
    path = f"{OUTPUT_DIR}/03_wordcloud_global.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f" Sauvegardé : {path}")


# ── 4. Wordclouds par époque ──────────────────────────────
def plot_wordcloud_by_era(df):
    df = df.copy()
    df["year"] = df["date"].str.extract(r"(\d{4})")[0].astype(int)

    eras = {
        "Fondateurs (1789-1849)":  df[df["year"] <= 1849],
        "Guerre civile (1850-1900)": df[(df["year"] > 1849) & (df["year"] <= 1900)],
        "Ère moderne (1901-1960)": df[(df["year"] > 1900) & (df["year"] <= 1960)],
        "Ère contemporaine (1961-2017)": df[df["year"] > 1960],
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (era_name, era_df) in enumerate(eras.items()):
        words = " ".join([" ".join(t) for t in era_df["lemmas"]])
        wc = WordCloud(width=600, height=350, background_color="white",
                       colormap="viridis", max_words=80,
                       collocations=False).generate(words)
        axes[idx].imshow(wc, interpolation="bilinear")
        axes[idx].axis("off")
        axes[idx].set_title(era_name, fontsize=13, fontweight="bold")

    plt.suptitle("Mots clés par époque historique", fontsize=16,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/04_wordcloud_by_era.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f" Sauvegardé : {path}")


# ── 5. TF-IDF — graphique en barres opposition ────────────
def plot_tfidf_comparison(df_tfidf, president1, president2):
    def get_scores(president):
        rows = df_tfidf[df_tfidf["president"] == president]
        if rows.empty:
            return {}
        return dict(rows.iloc[-1]["top_tfidf"])

    scores1 = get_scores(president1)
    scores2 = get_scores(president2)

    # Mots communs + spécifiques
    all_words = list(set(list(scores1.keys())[:8] + list(scores2.keys())[:8]))

    v1 = [scores1.get(w, 0) for w in all_words]
    v2 = [-scores2.get(w, 0) for w in all_words]

    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(all_words))
    ax.barh(y, v1, align="center", color="steelblue",  alpha=0.8, label=president1)
    ax.barh(y, v2, align="center", color="coral",      alpha=0.8, label=president2)
    ax.set_yticks(y)
    ax.set_yticklabels(all_words, fontsize=11)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(f"TF-IDF : {president1} vs {president2}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Score TF-IDF")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/05_tfidf_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f" Sauvegardé : {path}")


# ── 6. NER — entités les plus citées ─────────────────────
def plot_top_entities(df, entity_type="GPE", top_n=15):
    all_ents = [
        ent for ents in df["entities"]
        for ent, label in ents if label == entity_type
    ]
    counts = Counter(all_ents).most_common(top_n)
    words, freqs = zip(*counts)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(words, freqs, color=cm.Blues_r(np.linspace(0.3, 0.9, len(words))))
    plt.title(f"Top {top_n} entités de type '{entity_type}' — tous discours",
              fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.ylabel("Fréquence")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/06_top_entities.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f" Sauvegardé : {path}")