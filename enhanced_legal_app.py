#!/usr/bin/env python3
"""
A-Qlegal 2.0 - Enhanced Version with All Data
Advanced legal AI assistant with comprehensive training
"""

import json
import streamlit as st
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import re

# Load trained models
@st.cache_resource
def load_models():
    """Load all trained models"""
    try:
        # Load TF-IDF vectorizer
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        
        # Load TF-IDF matrix
        tfidf_matrix = np.load('data/embeddings/tfidf_matrix.npy')
        
        return tfidf_vectorizer, tfidf_matrix
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None

# Load legal data
@st.cache_data
def load_legal_data():
    """Load processed legal data"""
    all_data = []
    
    # Load processed data
    try:
        with open("data/processed/all_legal_documents.json", "r", encoding="utf-8") as f:
            processed_data = json.load(f)
            all_data.extend(processed_data)
    except FileNotFoundError:
        pass
    
    # Load new enhanced dataset
    try:
        with open("data/enhanced_legal_documents_v2.json", "r", encoding="utf-8") as f:
            enhanced_data = json.load(f)
            # Convert to the expected format
            for item in enhanced_data:
                formatted_item = {
                    "id": item.get("id", ""),
                    "title": item.get("title", ""),
                    "content": f"{item.get('text', '')} {item.get('simplified_summary', '')} {item.get('real_life_example', '')}",
                    "category": item.get("category", "").lower().replace(" ", "_"),
                    "section": item.get("section", ""),
                    "punishment": item.get("punishment", ""),
                    "citations": [],
                    "source": item.get("source", ""),
                    "keywords": item.get("keywords", [])
                }
                all_data.append(formatted_item)
    except FileNotFoundError:
        pass
    
    return all_data

def keyword_search(query, tfidf_vectorizer, tfidf_matrix, data, top_k=5):
    """Perform keyword search using TF-IDF"""
    try:
        # Transform query
        query_vector = tfidf_vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Get top results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                doc = data[idx]
                doc['similarity_score'] = float(similarities[idx])
                results.append(doc)
        
        return results
    except Exception as e:
        st.error(f"Keyword search failed: {e}")
        return []

def enhanced_search(query, data, top_k=5):
    """Enhanced search with keyword matching and self-defense specific handling"""
    query_lower = query.lower()
    results = []
    
    # Self-defense specific keywords
    self_defense_keywords = [
        'self defense', 'self-defence', 'private defence', 'right to defend',
        'defend yourself', 'kill in self defense', 'self protection',
        'section 96', 'section 97', 'section 99', 'section 100'
    ]
    
    # Check if query is about self-defense
    is_self_defense = any(keyword in query_lower for keyword in self_defense_keywords)
    
    for item in data:
        score = 0
        content_lower = (item.get('content', '') + ' ' + item.get('title', '')).lower()
        
        # Direct keyword matching
        for keyword in self_defense_keywords:
            if keyword in content_lower:
                score += 2
        
        # Title matching
        if any(word in item.get('title', '').lower() for word in query_lower.split()):
            score += 1.5
        
        # Content matching
        if any(word in content_lower for word in query_lower.split()):
            score += 1
        
        # Section matching for self-defense
        if is_self_defense and 'section' in item.get('section', '').lower():
            if any(section in item.get('section', '') for section in ['96', '97', '99', '100', '101', '102', '103', '104', '105', '106']):
                score += 3
        
        if score > 0:
            item['similarity_score'] = float(score)
            results.append(item)
    
    # Sort by score and return top results
    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    return results[:top_k]

def main():
    st.set_page_config(
        page_title="A-Qlegal 2.0 - Enhanced",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ A-Qlegal 2.0 - Enhanced Legal AI Assistant")
    st.markdown("**Your comprehensive AI-powered legal assistant trained on extensive Indian law data**")
    
    # Load models and data
    with st.spinner("Loading models and data..."):
        tfidf_vectorizer, tfidf_matrix = load_models()
        data = load_legal_data()
    
    if not data:
        st.error("No legal data found. Please run the training script first.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("📚 Legal Database")
        st.metric("Total Documents", len(data))
        
        # Category breakdown
        categories = {}
        for doc in data:
            cat = doc.get('category', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        st.header("📊 Categories")
        for cat, count in sorted(categories.items()):
            st.write(f"• {cat}: {count}")
        
        st.header("🔍 Search Options")
        top_k = st.slider("Number of Results", 1, 20, 5)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask a Legal Question")
        query = st.text_input(
            "Enter your legal question:",
            placeholder="e.g., What is the punishment for fraud?",
            key="query_input"
        )
        
        if st.button("🔍 Search", type="primary"):
            if query:
                with st.spinner("Searching..."):
                    # Try enhanced search first
                    results = enhanced_search(query, data, top_k)
                    
                    # If no results, try TF-IDF search
                    if not results:
                        results = keyword_search(query, tfidf_vectorizer, tfidf_matrix, data, top_k)
                
                if results:
                    st.success(f"Found {len(results)} relevant legal document(s)")
                    
                    for i, doc in enumerate(results, 1):
                        with st.expander(f"{i}. {doc.get('title', 'Unknown')} - {doc.get('category', 'Unknown')} (Score: {doc.get('similarity_score', 0):.3f})", expanded=i==1):
                            if doc.get('section'):
                                st.subheader(f"📖 {doc['section']}")
                            
                            if doc.get('content'):
                                st.subheader("📄 Legal Text")
                                st.write(doc['content'])
                            
                            if doc.get('simplified_summary'):
                                st.subheader("📝 Simplified Summary")
                                st.write(doc['simplified_summary'])
                            
                            if doc.get('real_life_example'):
                                st.subheader("🏠 Real-Life Example")
                                st.write(doc['real_life_example'])
                            
                            if doc.get('punishment'):
                                st.subheader("⚖️ Punishment")
                                st.write(f"**{doc['punishment']}**")
                            
                            if doc.get('keywords'):
                                st.subheader("🏷️ Keywords")
                                st.write(", ".join(doc['keywords']))
                            
                            if doc.get('citations'):
                                st.subheader("📚 Citations")
                                st.write(", ".join(doc['citations']))
                else:
                    st.warning("No relevant legal documents found. Try different keywords.")
            else:
                st.warning("Please enter a question.")
    
    with col2:
        st.header("📊 Statistics")
        st.metric("Total Legal Documents", len(data))
        st.metric("Categories", len(categories))
        
        st.header("💡 Search Tips")
        st.info("""
        **Enhanced Search Features:**
        
        🔍 **Smart Keyword Search**: Advanced TF-IDF based search
        
        📚 **Comprehensive Database**: Trained on extensive Indian legal data
        
        🎯 **Smart Ranking**: Results ranked by relevance and similarity
        
        📖 **Rich Context**: Legal text, simplified summaries, and examples
        
        ⚖️ **Complete Information**: Punishments, citations, and keywords
        """)
        
        st.header("🚀 Advanced Features")
        st.success("""
        ✅ **Smart Search**: AI-powered keyword matching
        
        ✅ **Multi-Source Data**: IPC, CrPC, Constitution, Court judgments
        
        ✅ **Real-Time Search**: Fast retrieval from large legal database
        
        ✅ **Contextual Answers**: Comprehensive legal information
        
        ✅ **User-Friendly**: Simple interface for complex legal queries
        """)

if __name__ == "__main__":
    main()
