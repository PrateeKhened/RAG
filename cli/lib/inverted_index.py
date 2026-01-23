import pickle
from .search_utils import load_movies, CACHE_DIR, tokenize_text, BM25_K1
from typing import Dict, Set
import os 
from collections import Counter, defaultdict
import math

class InvertedIndex():
    def __init__(self):
        self.index: Dict[str, Set[int]] = {}
        self.docmap: Dict[int, dict] = {}
        self.term_frequencies: Dict[int, Counter] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
    
    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id) 
        self.term_frequencies[doc_id] = Counter(tokens)
    
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
        
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self, cache_dir: str = CACHE_DIR):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Missing file: {self.index_path}")
        if not os.path.exists(self.docmap_path):
            raise FileNotFoundError(f"Missing file: {self.docmap_path}")
        if not os.path.exists(self.term_frequencies_path):
            raise FileNotFoundError(f"Missing file: {self.term_frequencies_path}")

        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
    
    def get_tf(self, doc_id, term):
        token = tokenize_text(term)
        if len(token) != 1:
            raise Exception("term has more than one token")
        
        token = token[0]
        if doc_id not in self.term_frequencies:
            return 0 
        if token not in self.term_frequencies[doc_id]:
            return 0
        return self.term_frequencies[doc_id][token]
    
    def get_idf(self, term):
        token = tokenize_text(term)
        if len(token) != 1:
            raise Exception("term has more then one token")
        
        token = token[0]
        term_match_doc_count = len(self.get_documents(token))
        total_doc_count = len(self.docmap)
        idf_val = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
        return idf_val

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
    
    def get_bm25_idf(self, term: str) -> float:
        token = tokenize_text(term)
        if len(token) != 1:
            raise Exception("term has more then one token")
        
        token = token[0]
        term_match_doc_count = len(self.get_documents(token))
        total_doc_count = len(self.docmap)
        idf_val = math.log((total_doc_count - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5) + 1)
        return idf_val
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1):
        tf = self.get_tf(doc_id, term)
        bm25_tf = (tf * (k1 + 1)) / (tf + k1)
        return bm25_tf

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()

def tf_command(doc_id: int, term: str) -> int:
    indx = InvertedIndex()
    indx.load()
    num = indx.get_tf(doc_id, term)
    return num

def idf_command(term: str) -> int:
    indx = InvertedIndex()
    indx.load()
    return indx.get_idf(term)

def tfidf_command(doc_id: int, term: str) -> float:
    indx = InvertedIndex()
    indx.load()
    return indx.get_tf_idf(doc_id, term)

def bm25_idf_command(term: str):
    indx = InvertedIndex() 
    indx.load() 
    return indx.get_bm25_idf(term)

def bm25_tf_command(doc_id, term, k1=BM25_K1):
    indx = InvertedIndex() 
    indx.load() 
    return indx.get_bm25_tf(doc_id, term, k1)
    