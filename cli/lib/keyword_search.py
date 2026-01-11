from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, read_stopwords
import string

def search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    
    results = []

    for movie in movies:
        preprocessed_query = tokenize_text(query)
        preprocessed_title = tokenize_text(movie['title'])
        if has_matching_token(preprocessed_query, preprocessed_title):
            results.append(movie)
            if len(results) > 5: 
                break
    return results

def has_matching_token(query_token: list[str], title_token: list[str]) -> bool:
    for q in query_token:
        for t in title_token:
            if q in t:
                return True
    return False

def preprocessed_text(text: str) -> str:
    table = str.maketrans('', '', string.punctuation)
    text = text.translate(table)
    return text.lower()

def tokenize_text(text: str) -> list[str]:
    text = preprocessed_text(text)
    stopwords = read_stopwords()
    stopwords_set = set(stopwords)

    valid_tokens = []
    stopwords_removed_tokens = []

    for t in text.split():
        if t:
            valid_tokens.append(t)

    for token in valid_tokens:
        if token not in stopwords_set:
            stopwords_removed_tokens.append(token)
    return stopwords_removed_tokens