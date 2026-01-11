from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies

def search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    
    results = []

    for movie in movies:
        if query in movie['title']:
            results.append(movie)
            if len(results) > 5: 
                break

    return results