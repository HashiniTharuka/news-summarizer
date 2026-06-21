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
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)

class SemanticSearch:
    def __init__(self, data_path=None):
        """Initialize Semantic Search with Word2Vec"""
        self.stop_words = set(stopwords.words('english'))
        self.word2vec_model = None
        self.article_vectors = None
        self.articles = []
        self.embeddings_path = "./models/embeddings.pkl"
        self.vector_size = 100
        
    def preprocess(self, text):
        """Preprocess text for Word2Vec"""
        if not text or not isinstance(text, str):
            return []
        
        # Clean text
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Remove stopwords and short words
        tokens = [w for w in tokens if w not in self.stop_words and len(w) > 1]
        return tokens
    
    def train_model(self, df, vector_size=100, window=5, min_count=1):
        """Train Word2Vec model on articles"""
        logger.info("📚 Tokenizing articles...")
        
        # Tokenize all articles
        tokenized = df["Review"].apply(self.preprocess)
        sentences = tokenized.tolist()
        sentences = [s for s in sentences if len(s) > 0]
        
        if not sentences:
            logger.error("❌ No valid sentences found!")
            return None
        
        logger.info(f"🔄 Training Word2Vec on {len(sentences)} articles...")
        
        # Train Word2Vec
        self.word2vec_model = Word2Vec(
            sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            seed=42,
            epochs=10
        )
        
        self.vector_size = vector_size
        logger.info(f"✅ Word2Vec trained! Vocabulary: {len(self.word2vec_model.wv)}")
        return self.word2vec_model
    
    def build_embeddings(self, df):
        """Build article embeddings"""
        if self.word2vec_model is None:
            logger.error("❌ Train Word2Vec first!")
            return None
        
        self.articles = df["Review"].tolist()
        
        def get_vector(text):
            """Get vector representation for a text"""
            tokens = self.preprocess(text)
            if not tokens:
                return np.zeros(self.vector_size)
            
            vecs = [self.word2vec_model.wv[w] for w in tokens 
                   if w in self.word2vec_model.wv]
            
            return np.mean(vecs, axis=0) if vecs else np.zeros(self.vector_size)
        
        logger.info("📊 Building article embeddings...")
        self.article_vectors = np.array([get_vector(a) for a in self.articles])
        logger.info(f"✅ Embeddings shape: {self.article_vectors.shape}")
        
        # Save embeddings
        try:
            os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump({
                    'vectors': self.article_vectors,
                    'articles': self.articles,
                    'model': self.word2vec_model,
                    'vector_size': self.vector_size
                }, f)
            logger.info(f"✅ Saved embeddings to: {self.embeddings_path}")
        except Exception as e:
            logger.error(f"❌ Error saving embeddings: {e}")
            return None
        
        return self.article_vectors
    
    def load_embeddings(self):
        """Load pre-computed embeddings"""
        if not os.path.exists(self.embeddings_path):
            logger.warning("⚠️ No embeddings found at: " + self.embeddings_path)
            return False
        
        try:
            with open(self.embeddings_path, 'rb') as f:
                data = pickle.load(f)
                
            self.article_vectors = data['vectors']
            self.articles = data['articles']
            self.word2vec_model = data['model']
            self.vector_size = data.get('vector_size', 100)
            
            logger.info(f"✅ Embeddings loaded! Articles: {len(self.articles)}, Vector size: {self.vector_size}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading embeddings: {e}")
            return False
    
    def search(self, query, top_n=5):
        """Find similar articles"""
        # Validate
        if self.word2vec_model is None or self.article_vectors is None:
            return [{"error": "Model not trained or embeddings not loaded"}]
        
        if not query or not isinstance(query, str):
            return []
        
        # Get query vector
        query_vec = self._get_query_vector(query)
        if query_vec is None:
            return []
        
        # Compute similarities
        try:
            similarities = cosine_similarity(query_vec, self.article_vectors)[0]
        except Exception as e:
            logger.error(f"Error computing similarities: {e}")
            return []
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        results = []
        for i, idx in enumerate(top_indices):
            results.append({
                "rank": i + 1,
                "similarity": round(float(similarities[idx]), 4),
                "article": self.articles[idx][:300] + "..." if len(self.articles[idx]) > 300 else self.articles[idx]
            })
        
        return results
    
    def _get_query_vector(self, query):
        """Get vector for query"""
        tokens = self.preprocess(query)
        if not tokens:
            return None
        
        vecs = [self.word2vec_model.wv[w] for w in tokens 
               if w in self.word2vec_model.wv]
        
        if not vecs:
            return None
        
        return np.mean(vecs, axis=0).reshape(1, -1)
    
    def get_similar_words(self, word, top_n=5):
        """Get similar words from Word2Vec"""
        if self.word2vec_model is None:
            return []
        
        if word in self.word2vec_model.wv:
            return self.word2vec_model.wv.most_similar(word, topn=top_n)
        return []
    
    def get_word_vectors(self, words):
        """Get vectors for multiple words"""
        if self.word2vec_model is None:
            return None
        
        vectors = []
        for word in words:
            if word in self.word2vec_model.wv:
                vectors.append(self.word2vec_model.wv[word])
        return np.array(vectors) if vectors else None
    
    def get_vocabulary_size(self):
        """Get vocabulary size"""
        if self.word2vec_model is None:
            return 0
        return len(self.word2vec_model.wv)
    
    def add_articles(self, new_articles):
        """Add new articles and rebuild embeddings"""
        if self.word2vec_model is None:
            logger.error("❌ Train Word2Vec first!")
            return False
        
        # Update articles list
        if isinstance(new_articles, list):
            self.articles.extend(new_articles)
        elif isinstance(new_articles, pd.DataFrame):
            self.articles.extend(new_articles["Review"].tolist())
        else:
            logger.error("❌ Invalid input type")
            return False
        
        # Rebuild embeddings
        self.build_embeddings(pd.DataFrame({"Review": self.articles}))
        return True
    
    def reset(self):
        """Reset all data"""
        self.word2vec_model = None
        self.article_vectors = None
        self.articles = []
        logger.info("🔄 Reset complete")

# Create singleton instance
semantic_search = SemanticSearch()