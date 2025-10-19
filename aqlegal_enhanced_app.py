#!/usr/bin/env python3
"""
A-Qlegal 3.0 Enhanced - With 10,000+ Additional Law Documents
Advanced legal intelligence with comprehensive Indian law coverage
"""

import json
import streamlit as st
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import re
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="A-Qlegal 3.0 Enhanced",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E3A8A;
        font-weight: bold;
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #64748B;
        font-size: 1.4rem;
        margin-bottom: 2rem;
    }
    .stats-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-badge {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Load enhanced models
@st.cache_resource
def load_enhanced_models():
    """Load enhanced models"""
    try:
        # Try enhanced models first
        try:
            with open('models/enhanced_tfidf_vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            tfidf_matrix = np.load('data/enhanced/enhanced_tfidf_matrix.npy')
            logger.info("Using enhanced models")
        except:
            # Fallback to original models
            with open('models/tfidf_vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            tfidf_matrix = np.load('data/embeddings/tfidf_matrix.npy')
            logger.info("Using original models")
        
        return vectorizer, tfidf_matrix
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None

@st.cache_data
def load_enhanced_data():
    """Load enhanced legal data"""
    try:
        # Try enhanced data first
        try:
            with open("data/enhanced/enhanced_legal_documents.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            # Fallback to original data
            with open("data/processed/all_legal_documents.json", "r", encoding="utf-8") as f:
                return json.load(f)
    except:
        return []

def search_enhanced(query, vectorizer, tfidf_matrix, data, top_k=5):
    """Enhanced search with larger dataset"""
    try:
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                doc = data[idx].copy()
                doc['similarity_score'] = float(similarities[idx])
                results.append(doc)
        
        return results
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []

def main():
    # Header
    st.markdown('<h1 class="main-header">⚖️ A-Qlegal 3.0 Enhanced</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Legal Intelligence with 10,000+ Indian Law Documents</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("🔄 Loading enhanced AI system..."):
        vectorizer, tfidf_matrix = load_enhanced_models()
        data = load_enhanced_data()
    
    if not data:
        st.error("❌ No legal data found.")
        return
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
        st.metric("📚 Total Documents", f"{len(data):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
        categories = len(set(doc.get('category', '') for doc in data))
        st.metric("📁 Categories", f"{categories}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
        sections = len([doc for doc in data if doc.get('section')])
        st.metric("📖 Legal Sections", f"{sections:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
        st.metric("🎯 Intelligence", "Enhanced")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Enhanced Settings")
        
        # Persona selection
        st.subheader("👤 User Persona")
        persona = st.selectbox(
            "Select your role:",
            ["Citizen", "Student", "Business", "Lawyer"]
        )
        
        # Search settings
        st.subheader("🔍 Search Settings")
        top_k = st.slider("Results to show:", 1, 20, 5)
        similarity_threshold = st.slider("Min similarity:", 0.0, 1.0, 0.0, 0.1)
        
        # Language
        st.subheader("🌍 Language")
        language = st.selectbox(
            "Select language:",
            ["English", "हिन्दी (Hindi)", "தமிழ் (Tamil)", "বাংলা (Bengali)", "తెలుగు (Telugu)"]
        )
        
        # Category filter
        st.subheader("📁 Filter by Category")
        categories = sorted(set(doc.get('category', '') for doc in data))
        selected_category = st.selectbox(
            "Choose category:",
            ["All"] + categories
        )
    
    # Main content
    st.header("💬 Ask Your Legal Question")
    
    # Query input
    query = st.text_area(
        "Enter your question:",
        placeholder="e.g., What is the punishment for fraud? Explain my rights under Article 21",
        height=100,
        key="query_input"
    )
    
    # Example questions
    with st.expander("💡 Example Questions"):
        examples = [
            "What is the punishment for fraud?",
            "Explain Section 420 IPC",
            "Tell me about fundamental rights",
            "What is culpable homicide?",
            "Explain right to freedom of speech",
            "What should I do if I'm arrested?",
            "How to file an FIR?",
            "What are my consumer rights?",
            "Tell me about property laws",
            "Explain labor law rights"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            col = cols[i % 2]
            if col.button(f"📌 {example}", key=f"ex_{i}"):
                query = example
    
    # Search
    if st.button("🔍 Search Enhanced Database", type="primary") or query:
        if query:
            with st.spinner("🤖 AI is analyzing your question..."):
                # Filter by category if selected
                if selected_category != "All":
                    filtered_data = [doc for doc in data if doc.get('category') == selected_category]
                else:
                    filtered_data = data
                
                # Search
                results = search_enhanced(query, vectorizer, tfidf_matrix, filtered_data, top_k)
                
                # Filter by threshold
                results = [r for r in results if r['similarity_score'] >= similarity_threshold]
                
                if results:
                    st.success(f"✅ Found {len(results)} relevant document(s) in enhanced database")
                    
                    for i, doc in enumerate(results, 1):
                        with st.expander(
                            f"📖 {i}. {doc.get('title', 'Unknown')} | "
                            f"{doc.get('category', 'Unknown')} | "
                            f"Match: {doc.get('similarity_score', 0):.1%}",
                            expanded=i==1
                        ):
                            # Confidence score
                            confidence = min(doc.get('similarity_score', 0) * 100, 100)
                            st.markdown(
                                f'<span class="confidence-badge">Confidence: {confidence:.0f}%</span>',
                                unsafe_allow_html=True
                            )
                            
                            # Main content
                            if doc.get('section'):
                                st.subheader(f"📖 {doc['section']}")
                            
                            st.subheader("📝 Simplified Summary")
                            st.write(doc.get('simplified_summary', 'No summary available'))
                            
                            if doc.get('real_life_example'):
                                st.subheader("🏠 Real-Life Example")
                                st.write(doc['real_life_example'])
                            
                            if doc.get('punishment'):
                                st.subheader("⚖️ Punishment")
                                st.write(f"**{doc['punishment']}**")
                            
                            if doc.get('keywords'):
                                st.subheader("🏷️ Keywords")
                                st.write(", ".join(doc['keywords'][:10]))
                            
                            # Feedback
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.button(f"👍 Helpful", key=f"helpful_{i}")
                            with col_b:
                                st.button(f"👎 Not helpful", key=f"not_helpful_{i}")
                else:
                    st.warning("⚠️ No relevant documents found. Try different keywords or adjust similarity threshold.")
        else:
            st.info("💡 Please enter a question above")
    
    # Footer
    st.markdown("---")
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    
    with col_f1:
        st.metric("Version", "3.0 Enhanced")
    with col_f2:
        st.metric("Documents", f"{len(data):,}")
    with col_f3:
        st.metric("Intelligence", "Advanced")
    with col_f4:
        st.metric("Status", "🟢 Online")
    
    st.markdown(
        '<p style="text-align: center; color: #64748B;">© 2025 A-Qlegal 3.0 Enhanced - Comprehensive Legal Intelligence for India</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
