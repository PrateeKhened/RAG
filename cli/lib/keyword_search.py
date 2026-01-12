from .search_utils import DEFAULT_SEARCH_LIMIT, tokenize_text
from .inverted_index import InvertedIndex

def search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    indx = InvertedIndex() 
    indx.load()
    preprocessed_query = tokenize_text(query)
    return all_matching_token(preprocessed_query, indx, limit)

def all_matching_token(query_token: list[str], indx: InvertedIndex, limit: int = DEFAULT_SEARCH_LIMIT) -> list:
    result = [] 
    for q in query_token:
        if q in indx.index:
            docs = indx.get_documents(q)
            for doc in docs:
                result.append(indx.docmap[doc])
                if len(result) >= limit:
                    return result
    return result
