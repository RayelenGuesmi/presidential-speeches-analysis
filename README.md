# 🇺🇸 Analyse NLP des Discours d'Investiture Présidentiels Américains

##  Objectif

Analyser les 58 discours d'investiture des présidents américains (1789–2017),
de George Washington à Donald J. Trump, afin de comprendre :

- Les **thèmes récurrents** abordés au fil des siècles
- L'**évolution de la tonalité** et des sentiments exprimés
- Les **mots et entités clés** qui caractérisent chaque époque
- Les **ruptures rhétoriques** entre présidents

---

##  Structure du projet

presidential-speeches-analysis/
├── data/
│   └── inaug_speeches.csv        # Dataset Kaggle (58 discours, 1789-2017)
├── notebooks/
│   └── analysis.ipynb            # Notebook principal — analyse complète
├── src/
│   ├── preprocessing.py          # Nettoyage, tokenisation, stemming, lemmatisation, NER, POS
│   ├── analysis.py               # Sentiment (TextBlob, VADER), fréquence, TF-IDF
│   └── viz.py                    # Toutes les visualisations (matplotlib, seaborn, wordcloud)
├── output/
│   ├── 01_sentiment_over_time.png
│   ├── 02_tone_distribution.png
│   ├── 03_wordcloud_global.png
│   ├── 04_wordcloud_by_era.png
│   ├── 05_tfidf_comparison.png
│   ├── 06_top_entities.png
│   └── rapport_presidential_speeches.pdf
├── requirements.txt
└── README.md

---

##  Dataset

- **Source** : [Kaggle — Presidential Inaugural Addresses](https://www.kaggle.com/datasets/adhok93/presidentialaddress)
- **Contenu** : 58 discours couvrant 228 ans d'histoire américaine
- **Colonnes** : `president`, `title`, `date`, `text`
- **Encodage** : latin-1
- **Valeurs manquantes** : aucune

---

## 🔧 Installation

### Prérequis
- Python 3.10+
- pip

### Étapes

```bash
# 1. Cloner le repo
git clone https://github.com/RayelenGuesmi/presidential-speeches-analysis.git
cd presidential-speeches-analysis

# 2. Créer et activer l'environnement virtuel
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Télécharger le modèle spaCy
python -m spacy download en_core_web_sm

# 5. Télécharger les ressources NLTK
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); \
nltk.download('wordnet'); nltk.download('sentiwordnet'); nltk.download('punkt_tab')"

# 6. Lancer le notebook
code .   # Ouvrir VS Code, puis notebooks/analysis.ipynb
```

---

##  Pipeline d'analyse

### 1. Acquisition des données
Chargement du dataset CSV avec pandas, renommage des colonnes,
vérification des valeurs manquantes.

### 2. Prétraitement (`src/preprocessing.py`)
| Étape | Outil | Description |
|-------|-------|-------------|
| Nettoyage | `re` | Suppression ponctuation, chiffres, mise en minuscules |
| Tokenisation | `nltk` | Découpage en tokens + suppression stop words |
| Stemming | `nltk` PorterStemmer | Réduction à la racine (`running` → `run`) |
| Lemmatisation | `nltk` WordNetLemmatizer | Forme canonique (`better` → `good`) |
| POS Tagging | `spaCy` | Étiquetage grammatical (NOUN, VERB, PROPN...) |
| NER | `spaCy` | Reconnaissance d'entités (ORG, GPE, DATE, LAW...) |

### 3. Analyses de base (`src/analysis.py`)
| Analyse | Outil | Résultat |
|---------|-------|----------|
| Polarité | `TextBlob` | Score entre -1 (négatif) et +1 (positif) |
| Subjectivité | `TextBlob` | Score entre 0 (objectif) et 1 (subjectif) |
| Tonalité | `VADER` | Scores pos/neg/neu + compound |
| Fréquence | `collections.Counter` | Top 20 mots par discours |

### 4. Analyses avancées (`src/analysis.py`)
| Analyse | Outil | Résultat |
|---------|-------|----------|
| TF-IDF | `scikit-learn` TfidfVectorizer | Mots les plus distinctifs par discours |

### 5. Visualisations (`src/viz.py`)
| Visualisation | Description |
|---------------|-------------|
| `01_sentiment_over_time` | Polarité TextBlob + VADER compound (1789→2017) |
| `02_tone_distribution` | Pie chart des tonalités + top 15 discours subjectifs |
| `03_wordcloud_global` | Nuage de mots — tous les discours confondus |
| `04_wordcloud_by_era` | Nuages de mots par époque historique (4 périodes) |
| `05_tfidf_comparison` | Graphique en barres opposition Lincoln vs Trump |
| `06_top_entities` | Top 15 entités géographiques (NER) |

---

##  Résultats clés

### Sentiment
- **98.3%** des discours ont une tonalité **positive** — cohérent avec la
  rhétorique d'espoir propre aux investitures
- **Lincoln (1865)** est le seul discours **négatif** (VADER : -0.95) —
  contexte de fin de guerre civile, discours de réconciliation chargé de gravité
- La polarité TextBlob oscille entre **0.05 et 0.25** sur 228 ans

### Évolution historique
| Époque | Mots dominants | Contexte |
|--------|---------------|---------|
| Fondateurs (1789-1849) | government, state, constitution | Construction des institutions |
| Guerre civile (1850-1900) | war, law, union, territory | Conflits internes et reconstruction |
| Ère moderne (1901-1960) | world, peace, nation, democracy | Montée en puissance internationale |
| Contemporain (1961-2017) | america, freedom, new, world | Projection mondiale et idéaux démocratiques |

### Entités géographiques (NER)
- **America** (203x) est l'entité la plus citée, loin devant
  **the United States** (118x) et **States** (94x)
- Présence de Spain, Philippines, Cuba → reflet des guerres coloniales
  de la fin du XIXe siècle

### Insights notables
- **Trump (2017)** se distingue nettement avec `back`, `dream`, `america`,
  `american` → empreinte du slogan "Make America Great Again" clairement
  visible dans le TF-IDF, rupture rhétorique par rapport aux prédécesseurs
- **Kennedy (1961)** se distingue par `let`, `pledge`, `side` →
  écho de son célèbre "Ask not what your country can do for you"
- **Roosevelt (1933)** met en avant `money`, `action` →
  réponse directe à la Grande Dépression
- Le vocabulaire religieux (`god`, `divine`, `providence`) diminue
  progressivement du XVIIIe au XXe siècle

---

##  Dépendances principales

| Librairie | Version | Usage |
|-----------|---------|-------|
| pandas | 3.0+ | Manipulation des données |
| nltk | 3.9+ | Tokenisation, stemming, lemmatisation |
| spacy | 3.8+ | NER, POS tagging |
| textblob | 0.20+ | Analyse de sentiment |
| vaderSentiment | 3.3+ | Analyse de sentiment et tonalité |
| scikit-learn | 1.8+ | TF-IDF |
| matplotlib | 3.10+ | Visualisations |
| seaborn | 0.13+ | Visualisations avancées |
| wordcloud | 1.9+ | Nuages de mots |

---

##  Licence
Projet réalisé dans un cadre académique — données issues de Kaggle (domaine public).
> **Auteur** : Rayelen Guesmi 