#!/usr/bin/env python3
"""Test self-defense query in enhanced system"""

import json
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def test_self_defense_query():
    """Test the self-defense query"""
    
    # Load models
    with open('models/enhanced_tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    tfidf_matrix = np.load('data/enhanced/enhanced_tfidf_matrix.npy')
    
    # Load data
    with open('data/enhanced/enhanced_legal_documents.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Test query
    query = 'can i kill someone in self defense'
    print(f"🔍 Query: {query}")
    
    # Search
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-5:][::-1]
    
    print(f"\n📊 Top 5 Results:")
    for i, idx in enumerate(top_indices, 1):
        if similarities[idx] > 0:
            doc = data[idx]
            print(f"\n{i}. {doc.get('title', 'Unknown')} (Score: {similarities[idx]:.3f})")
            print(f"   Section: {doc.get('section', 'N/A')}")
            print(f"   Summary: {doc.get('simplified_summary', 'N/A')[:100]}...")
            print(f"   Keywords: {', '.join(doc.get('keywords', [])[:5])}")
    
    return top_indices, similarities

if __name__ == "__main__":
    test_self_defense_query()
