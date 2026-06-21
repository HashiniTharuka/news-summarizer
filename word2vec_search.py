"""
Word2Vec semantic search for news articles
"""

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import os

class SemanticSearch:
    def __init__(self, data_path=None):
        self.stop_words = set(stopwords.words('english'))
        self.word2vec_model = None
        self.article_vectors = None
        self.articles = []
        self.embeddings_path = "./models/embeddings.pkl"
        
    def preprocess(self, text):
        """Preprocess text for Word2Vec"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w not in self.stop_words]
        return tokens
    
    def train_model(self, df, vector_size=100, window=5, min_count=1):
        """Train Word2Vec model on articles"""
        print("📚 Tokenizing articles...")
        tokenized = df["Review"].apply(self.preprocess)
        sentences = tokenized.tolist()
        
        print("🔄 Training Word2Vec...")
        self.word2vec_model = Word2Vec(
            sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            seed=42
        )
        
        print(f"✅ Word2Vec trained! Vocabulary: {len(self.word2vec_model.wv)}")
        return self.word2vec_model
    
    def build_embeddings(self, df):
        """Build article embeddings"""
        self.articles = df["Review"].tolist()
        
        def get_vector(text):
            tokens = self.preprocess(text)
            vecs = [self.word2vec_model.wv[w] for w in tokens 
                   if w in self.word2vec_model.wv]
            return np.mean(vecs, axis=0) if vecs else np.zeros(100)
        
        print("📊 Building article embeddings...")
        self.article_vectors = np.array([get_vector(a) for a in self.articles])
        print(f"✅ Embeddings shape: {self.article_vectors.shape}")
        
        # Save embeddings
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump({
                'vectors': self.article_vectors,
                'articles': self.articles,
                'model': self.word2vec_model
            }, f)
        
        return self.article_vectors
    
    def load_embeddings(self):
        """Load pre-computed embeddings"""
        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, 'rb') as f:
                data = pickle.load(f)
                self.article_vectors = data['vectors']
                self.articles = data['articles']
                self.word2vec_model = data['model']
            print("✅ Embeddings loaded!")
            return True
        return False
    
    def search(self, query, top_n=5):
        """Find similar articles"""
        if self.word2vec_model is None:
            return [{"error": "Model not trained"}]
        
        # Get query vector
        query_vec = self._get_query_vector(query)
        if query_vec is None:
            return []
        
        # Compute similarities
        similarities = cosine_similarity(query_vec, self.article_vectors)[0]
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        results = []
        for i, idx in enumerate(top_indices):
            results.append({
                "rank": i + 1,
                "similarity": round(float(similarities[idx]), 4),
                "article": self.articles[idx][:300] + "..."
            })
        
        return results
    
    def _get_query_vector(self, query):
        """Get vector for query"""
        tokens = self.preprocess(query)
        vecs = [self.word2vec_model.wv[w] for w in tokens 
               if w in self.word2vec_model.wv]
        
        if not vecs:
            return None
        
        return np.mean(vecs, axis=0).reshape(1, -1)
    
    def get_similar_words(self, word, top_n=5):
        """Get similar words from Word2Vec"""
        if word in self.word2vec_model.wv:
            return self.word2vec_model.wv.most_similar(word, topn=top_n)
        return []