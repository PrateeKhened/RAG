from sentence_transformers import SentenceTransformer
import numpy as np
import os
from lib.search_utils import CACHE_DIR, DATA_PATH, DEFAULT_CHUNK_SIZE
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

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        cosine_list = []
        q_embedding = self.generate_embedding(query)
        for i in range(len(self.documents)):
            cos_sim = cosine_similarity(q_embedding, self.embeddings[i])
            cosine_list.append((cos_sim, self.documents[i]))
        sorted_cosine_list = sorted(cosine_list, key=lambda x: x[0], reverse=True)[:limit]
        result = [] 
        for i in sorted_cosine_list:
            result.append({"score":i[0], "title":i[1]["title"], "description":i[1]["description"]})
        return result

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

def embed_query_text(query):
    ss = SemanticSearch() 
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def fixed_size_chunking(text: str, chunk_size: int=DEFAULT_CHUNK_SIZE) -> list[list]:
    words = text.split() 
    chunks = [] 

    n_words = len(words)
    i = 0 
    while i < n_words:
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size
    return chunks

def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
    chunks = fixed_size_chunking(text, chunk_size)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")
