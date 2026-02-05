import os 
from .semantic_search import SemanticSearch, semantic_chunk
from .search_utils import CACHE_DIR
import numpy as np 
import json

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embeddings_path = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
        self.chunk_metadata_path = os.path.join(CACHE_DIR, "chunk_metadata.json")

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        for document in self.documents:
            self.document_map[document["id"]] = document
        
        chunk_list = []
        metadata_chunk_list = []
        for movie_idx, document in enumerate(self.documents):
            if not document["description"]:
                continue
            chunks = semantic_chunk(document["description"], 1, 4)
            chunk_list.extend(chunks)
            for chunk_idx, chunk in enumerate(chunks):
                metadata_chunk_list.append({
                    "movie_idx": movie_idx, 
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(chunks)
                })
        self.chunk_embeddings = self.model.encode(chunk_list)
        self.chunk_metadata = metadata_chunk_list

        os.makedirs(CACHE_DIR, exist_ok=True)

        np.save(self.chunk_embeddings_path, self.chunk_embeddings)

        with open(self.chunk_metadata_path, "w") as f:
            json.dump(
                {"chunks": metadata_chunk_list, "total_chunks": len(chunk_list)},
                    f,
                    indent=2,
            )

        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for document in self.documents:
            self.document_map[document["id"]] = document

        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(self.chunk_metadata_path):
            with open(self.chunk_metadata_path, "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)