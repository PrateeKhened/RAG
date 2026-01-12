import pickle
from .search_utils import load_movies, CACHE_DIR
from .keyword_search import tokenize_text
from typing import Dict, Set
import os 

class InvertedIndex():
    def __init__(self):
        self.index: Dict[str, Set[int]] = {}
        self.docmap: Dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")

    
    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id) 
    
    def get_documents(self, term) ->  list[int]:
        if term not in self.index:
            return []
        return sorted(list(self.index[term]))

    def build(self):
        movies = load_movies()
        for m in movies:
            custom_str = f"{m['title']} {m['description']}"
            self.__add_document(m['id'], custom_str)
            self.docmap[m['id']] = m
    
    def save(self, cache_dir: str = CACHE_DIR):
        
        os.makedirs(cache_dir, exist_ok=True)

        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

def build_comand() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")
