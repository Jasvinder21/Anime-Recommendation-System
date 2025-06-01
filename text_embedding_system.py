import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import re
from typing import List, Tuple

class TextEmbeddingSystem:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', embedding_dir: str = './embeddings'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.texts = []
        self.titles = []  # Store anime titles separately
        self.embedding_dir = embedding_dir
        self.index_file = os.path.join(self.embedding_dir, 'faiss_index.index')
        self.embeddings_file = os.path.join(self.embedding_dir, 'embeddings.pkl')
    
    def parse_combined_info(self, text: str) -> dict:
        """Extract title and overview from a text string."""
        title_match = re.search(r'Title: (.*?)\.', text)
        overview_match = re.search(r'Overview: (.*)', text)
        
        return {
            'title': title_match.group(1) if title_match else '',
            'overview': overview_match.group(1) if overview_match else ''
        }
    
    def preprocess_data(self, df: pd.DataFrame) -> List[str]:
        """Preprocess data to extract text embeddings and store titles."""
        processed_texts = []
        
        for text in df['combined_info']:
            parsed = self.parse_combined_info(text)
            combined = f"{parsed['title']} {parsed['overview']}"
            processed_texts.append(combined)
            self.texts.append(combined)
            self.titles.append(parsed['title'])  # Store extracted title separately
            
        return processed_texts
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def build_index(self, embeddings: np.ndarray):
        """Build FAISS index for fast similarity search."""
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, str]]:
        """Search for similar items based on query."""
        if self.index is None:
            raise ValueError("FAISS index not loaded. Please load or build the index first.")
        
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            # Check if idx is within bounds to prevent IndexError
            if idx < len(self.titles) and idx < len(self.texts):
                results.append((self.titles[idx], distance, self.texts[idx]))
            else:
                print(f"Warning: index {idx} is out of bounds for titles or texts.")
                
        return results
    
    def save_embeddings_and_index(self):
        """Save embeddings and FAISS index to disk."""
        if not os.path.exists(self.embedding_dir):
            os.makedirs(self.embedding_dir)
        
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump((self.texts, self.titles), f)
        
        faiss.write_index(self.index, self.index_file)
        print("Embeddings and FAISS index saved successfully.")
    
    def load_embeddings_and_index(self):
        """Load embeddings and FAISS index from disk."""
        if os.path.exists(self.embeddings_file) and os.path.exists(self.index_file):
            with open(self.embeddings_file, 'rb') as f:
                self.texts, self.titles = pickle.load(f)
            
            self.index = faiss.read_index(self.index_file)
            print("Embeddings and FAISS index loaded successfully.")
        else:
            print("No saved embeddings or index found. Please process and index the data first.")
    
    def process_and_index(self, df: pd.DataFrame):
        """Process data, create embeddings, and build FAISS index."""
        processed_texts = self.preprocess_data(df)
        embeddings = self.create_embeddings(processed_texts)
        self.build_index(embeddings)
        self.save_embeddings_and_index()
