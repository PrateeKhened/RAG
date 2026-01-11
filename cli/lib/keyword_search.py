from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies

def search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    
    results = []

    for movie in movies:
        preprocessed_query = preprocessed_text(query)
        preprocessed_title = preprocessed_text(movie['title'])
        if preprocessed_query in preprocessed_title:
            results.append(movie)
            if len(results) > 5: 
                break

    return results

def preprocessed_text(text: str) -> str:
    return text.lower()