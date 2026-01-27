import json
import os 
import string
from nltk.stem import PorterStemmer

DEFAULT_SEARCH_LIMIT = 5 

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

BM25_K1 = 1.5
BM25_B = 0.75

DEFAULT_CHUNK_SIZE = 200

def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def read_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()

def preprocessed_text(text: str) -> str:
    table = str.maketrans('', '', string.punctuation)
    text = text.translate(table)
    return text.lower()

def tokenize_text(text: str) -> list[str]:
    text = preprocessed_text(text)
    stopwords = read_stopwords()
    stemmer = PorterStemmer()

    valid_tokens = []
    stopwords_removed_tokens = []
    stemmed_tokens = []

    for t in text.split():
        if t:
            valid_tokens.append(t)

    for token in valid_tokens:
        if token not in stopwords:
            stopwords_removed_tokens.append(token)
    
    for token in stopwords_removed_tokens:
        stemmed_tokens.append(stemmer.stem(token))

    return stemmed_tokens