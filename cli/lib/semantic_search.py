from sentence_transformers import SentenceTransformer
import numpy as np
import os
from lib.search_utils import CACHE_DIR, DATA_PATH
import json

class SemanticSearch(): 
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.movie_embedding_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def generate_embedding(self, text):
        if not text or text.isspace():
            raise ValueError("input must not be empty")
        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents):
        movie_list = [] 
        self.documents = documents
        for document in self.documents:
            self.document_map[document["id"]] = document
            movie_list.append(f"{document['title']}: {document['description']}")
        self.embeddings = self.model.encode(movie_list, show_progress_bar=True)

        np.save(self.movie_embedding_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for document in self.documents:
            self.document_map[document["id"]] = document
        if os.path.exists(self.movie_embedding_path):
            self.embeddings = np.load(self.movie_embedding_path)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        return self.build_embeddings(documents)

def verify_model(): 
    ss = SemanticSearch() 
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")       

def embed_text(text):
    ss = SemanticSearch() 
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    ss = SemanticSearch()  
    with open(DATA_PATH, "r") as f:
        movies = json.load(f)
        documents = movies["movies"]
    ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {ss.embeddings.shape[0]} vectors in {ss.embeddings.shape[1]} dimensions")