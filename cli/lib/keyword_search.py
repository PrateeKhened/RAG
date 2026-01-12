from .search_utils import DEFAULT_SEARCH_LIMIT, read_stopwords
import string
from nltk.stem import PorterStemmer
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
            for d in indx.index[q]:
                result.append(indx.docmap[d])
                if len(result) >= limit:
                    return result
    return result

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