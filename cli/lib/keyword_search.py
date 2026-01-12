from .search_utils import DEFAULT_SEARCH_LIMIT, tokenize_text
from .inverted_index import InvertedIndex

def search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    indx = InvertedIndex() 
    indx.load()
    preprocessed_query = tokenize_text(query)
    return all_matching_token(preprocessed_query, indx, limit)

def all_matching_token(query_token: list[str], indx: InvertedIndex, limit: int = DEFAULT_SEARCH_LIMIT) -> list:
    results = [] 
    seen = set() 
    for q in query_token:
        if q in indx.index:
            docs = indx.get_documents(q)
            for doc_id in docs:
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                results.append(indx.docmap[doc_id])
                if len(results) >= limit:
                    return results
    return results
