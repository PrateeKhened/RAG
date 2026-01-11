from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies
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
    valid_tokens = []
    for t in text.split():
        if t:
            valid_tokens.append(t)
    return valid_tokens