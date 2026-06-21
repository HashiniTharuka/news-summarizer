"""
Script to create Word2Vec embeddings
"""

import pandas as pd
from word2vec_search import SemanticSearch
import os

def main():
    # Sample news articles
    articles = [
        "The UK economy showed strong growth in Q3 as businesses reopened following lockdown",
        "Scientists discovered a new species of dinosaur in Argentina that could be largest ever",
        "New superhero movie broke box office records over the weekend earning 250 million",
        "Manchester City won their fourth consecutive Premier League title after victory",
        "NASA's Perseverance rover discovered organic compounds on Mars indicating past life",
        "Apple announced record quarterly earnings of 120 billion driven by iPhone sales",
        "Stock markets rallied as inflation fears eased and investors regained confidence",
        "New AI model can predict weather patterns with 90% accuracy using deep learning",
        "Government announced new economic stimulus package to support small businesses",
        "Tech company launched revolutionary smartphone with AI-powered camera features",
        "Researchers developed vaccine that shows promising results against new virus strain",
        "Olympic committee announced new host city for 2028 summer games",
        "Major bank reported increased profits despite economic uncertainty this quarter",
        "SpaceX successfully launched new satellite into orbit for global internet coverage",
        "Automaker unveiled electric vehicle with 500-mile range and fast charging",
    ]
    
    print(f"📚 Using {len(articles)} sample articles")
    
    # Create DataFrame
    df = pd.DataFrame({"Review": articles})
    
    # Initialize search
    search = SemanticSearch()
    
    # Train Word2Vec
    search.train_model(df)
    
    # Build embeddings
    search.build_embeddings(df)
    
    # Test search
    print("\n🔍 Testing search...")
    results = search.search("economy growth")
    if results:
        for r in results:
            print(f"  Rank {r['rank']}: Similarity {r['similarity']}")
            print(f"  {r['article'][:100]}...")
    
    print("\n✅ Embeddings created successfully!")
    print(f"📁 Location: {search.embeddings_path}")
    print(f"📊 Vocabulary size: {search.get_vocabulary_size()}")

if __name__ == "__main__":
    main()