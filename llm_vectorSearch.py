import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st
from typing import List, Tuple
import re
import pickle

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
        title_match = re.search(r'Title: (.*?)\.', text)
        overview_match = re.search(r'Overview: (.*)', text)
        
        return {
            'title': title_match.group(1) if title_match else '',
            'overview': overview_match.group(1) if overview_match else ''
        }
    
    def preprocess_data(self, df: pd.DataFrame) -> List[str]:
        processed_texts = []
        
        for text in df['combined_info']:
            parsed = self.parse_combined_info(text)
            combined = f"{parsed['title']} {parsed['overview']}"
            processed_texts.append(combined)
            self.texts.append(combined)
            self.titles.append(parsed['title'])  # Store extracted title separately
            
        return processed_texts
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def build_index(self, embeddings: np.ndarray):
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
    
    def search(self, query: str, k: int = 5) -> List[Tuple[int, float, str]]:
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
        if not os.path.exists(self.embedding_dir):
            os.makedirs(self.embedding_dir)
        
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump((self.texts, self.titles), f)
        
        faiss.write_index(self.index, self.index_file)
        print("Embeddings and FAISS index saved successfully.")
    
    def load_embeddings_and_index(self):
        if os.path.exists(self.embeddings_file) and os.path.exists(self.index_file):
            with open(self.embeddings_file, 'rb') as f:
                self.texts, self.titles = pickle.load(f)
            
            self.index = faiss.read_index(self.index_file)
            print("Embeddings and FAISS index loaded successfully.")
        else:
            print("No saved embeddings or index found. Please process and index the data first.")
    
    def process_and_index(self, df: pd.DataFrame):
        processed_texts = self.preprocess_data(df)
        embeddings = self.create_embeddings(processed_texts)
        self.build_index(embeddings)
        self.save_embeddings_and_index()

# Streamlit app
def main():
    
    """Main application function."""
    st.set_page_config(page_title="AI Anime Recommender", page_icon="🤖", layout="wide")
    st.title("🤖 Chat with AnimeLLM GURU")

    embedding_system = TextEmbeddingSystem()
    


    if os.path.exists(embedding_system.embeddings_file) and os.path.exists(embedding_system.index_file):
        embedding_system.load_embeddings_and_index()
    
    #if os.path.exists(embedding_system.embeddings_file):
     #   os.remove(embedding_system.embeddings_file)

    #if os.path.exists(embedding_system.index_file):
    #    os.remove(embedding_system.index_file)

    #df = pd.read_csv(r"E:\Flask\anime_data3.csv")
    #embedding_system.process_and_index(df)
    
    query = st.text_input(
        "🔍 Recommend Anime:", 
        placeholder="E.g., Ask about animes?"
    )
    
    if query:
        results = embedding_system.search(query, k=3)

        if results:
            for title, distance, text in results:
                st.write(f"Anime Title: {title}")
                st.write(f"Distance: {distance:.2f}")
                st.write(f"Text Preview: {text[:500]}...")
                st.write("-" * 50)
        else:
            st.write("No results found.")
        
    #embedding_system.load_embeddings_and_index()

if __name__ == "__main__":
    main()
