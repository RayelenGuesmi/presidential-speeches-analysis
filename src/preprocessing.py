# src/preprocessing.py
# ============================================================
# Module de prétraitement des discours présidentiels
# ============================================================

import re
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Chargement des modèles
nlp          = spacy.load("en_core_web_sm")
stemmer      = PorterStemmer()
lemmatizer   = WordNetLemmatizer()
stop_words   = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Nettoie le texte : minuscules, suppression ponctuation et chiffres."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list:
    """Tokenise et supprime les stop words et tokens courts."""
    tokens = word_tokenize(text)
    return [t for t in tokens if t not in stop_words and len(t) > 2]


def apply_stemming(tokens: list) -> list:
    """Applique le stemming PorterStemmer."""
    return [stemmer.stem(t) for t in tokens]


def apply_lemmatization(tokens: list) -> list:
    """Applique la lemmatisation WordNetLemmatizer."""
    return [lemmatizer.lemmatize(t) for t in tokens]


def apply_pos_tagging(text: str) -> list:
    """POS tagging via spaCy. Retourne liste de tuples (mot, POS)."""
    doc = nlp(text)
    return [
        (token.text, token.pos_)
        for token in doc
        if not token.is_stop and not token.is_punct
    ]


def apply_ner(text: str) -> list:
    """Named Entity Recognition via spaCy. Retourne liste de tuples (entité, label)."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def preprocess_dataframe(df):
    """
    Applique tout le pipeline de prétraitement sur le DataFrame.
    Retourne le DataFrame enrichi.
    """
    print(" Nettoyage...")
    df["clean_text"] = df["text"].apply(clean_text)

    print(" Tokenisation...")
    df["tokens"] = df["clean_text"].apply(tokenize)

    print(" Stemming...")
    df["stems"] = df["tokens"].apply(apply_stemming)

    print(" Lemmatisation...")
    df["lemmas"] = df["tokens"].apply(apply_lemmatization)

    print(" POS Tagging...")
    df["pos_tags"] = df["clean_text"].apply(apply_pos_tagging)

    print(" NER...")
    df["entities"] = df["text"].apply(apply_ner)

    print(" Prétraitement terminé !")
    return df