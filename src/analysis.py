# ============================================================
# src/analysis.py
# Fonctions d'analyse des discours
# ============================================================

import pandas as pd
from collections import Counter
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

vader = SentimentIntensityAnalyzer()


# ── 1. Analyse de sentiment TextBlob ─────────────────────
def textblob_sentiment(text: str) -> dict:
    """Retourne polarité et subjectivité via TextBlob."""
    blob = TextBlob(text)
    return {
        "polarity":     blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity
    }


# ── 2. Analyse de sentiment VADER ────────────────────────
def vader_sentiment(text: str) -> dict:
    """Retourne les scores VADER (pos, neg, neu, compound)."""
    return vader.polarity_scores(text)


# ── 3. Tonalité globale ───────────────────────────────────
def get_tone(compound: float) -> str:
    """Classe le discours selon son score compound VADER."""
    if compound >= 0.05:
        return "Positif"
    elif compound <= -0.05:
        return "Négatif"
    else:
        return "Neutre"


# ── 4. Fréquence des mots ────────────────────────────────
def word_frequency(tokens: list, top_n: int = 20) -> dict:
    """Retourne les top_n mots les plus fréquents."""
    return dict(Counter(tokens).most_common(top_n))


# ── Application sur le DataFrame ─────────────────────────
def analyze_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    print(" TextBlob sentiment...")
    tb = df["text"].apply(textblob_sentiment)
    df["polarity"]     = tb.apply(lambda x: x["polarity"])
    df["subjectivity"] = tb.apply(lambda x: x["subjectivity"])

    print(" VADER sentiment...")
    vd = df["text"].apply(vader_sentiment)
    df["vader_pos"]      = vd.apply(lambda x: x["pos"])
    df["vader_neg"]      = vd.apply(lambda x: x["neg"])
    df["vader_neu"]      = vd.apply(lambda x: x["neu"])
    df["vader_compound"] = vd.apply(lambda x: x["compound"])
    df["tone"]           = df["vader_compound"].apply(get_tone)

    print(" Fréquence des mots...")
    df["word_freq"] = df["lemmas"].apply(word_frequency)

    print(" Analyses terminées !")
    return df


# ── 5. TF-IDF ────────────────────────────────────────────
def compute_tfidf(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Calcule le TF-IDF sur les lemmes de chaque discours."""
    corpus = df["lemmas"].apply(lambda tokens: " ".join(tokens))
    vectorizer = TfidfVectorizer(max_features=500)
    matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    # Top N mots par discours
    results = []
    for i, row in enumerate(matrix):
        scores = zip(feature_names, row.toarray()[0])
        top = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
        results.append({
            "president": df["president"].iloc[i],
            "date":      df["date"].iloc[i],
            "top_tfidf": top
        })

    return pd.DataFrame(results), vectorizer, matrix