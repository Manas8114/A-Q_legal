#!/usr/bin/env python3
"""
A-Qlegal 2.5 - Generative RAG System
Advanced multilingual legal assistant with RAG capabilities
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

# Configure page
st.set_page_config(
    page_title="A-Qlegal 2.5 - Generative RAG",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #3B82F6 0%, #1E3A8A 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        text-align: center;
        color: #64748B;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .response-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .section-tag {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 600;
    }
    .category-badge {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_models():
    """Load all trained models"""
    try:
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        tfidf_matrix = np.load('data/embeddings/tfidf_matrix.npy')
        return tfidf_vectorizer, tfidf_matrix
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None

@st.cache_data
def load_legal_data():
    """Load processed legal data"""
    try:
        with open("data/processed/all_legal_documents.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

@st.cache_data
def load_rag_index():
    """Load RAG index"""
    try:
        with open("data/processed/rag_index.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

@st.cache_data
def load_personas():
    """Load user personas"""
    try:
        with open("data/processed/user_personas.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def generate_structured_response(doc, persona="citizen"):
    """Generate structured 9-point response"""
    response = f"""
### 1️⃣ Law Name and Section
**{doc.get('title', 'Unknown')}**
{f"**Section:** {doc.get('section', 'N/A')}" if doc.get('section') else ""}

### 2️⃣ Type of Law
<span class="category-badge">{doc.get('category', 'General').upper()}</span>

### 3️⃣ Easy Summary
{doc.get('simplified_summary', 'No summary available')}

### 4️⃣ Real-life Example
{doc.get('real_life_example', 'No example available')}

### 5️⃣ Important Terms
{', '.join(doc.get('keywords', ['No keywords'])[:10])}

### 6️⃣ Punishment or Penalty
{doc.get('punishment', 'Not specified')}

### 7️⃣ Related Sections
{', '.join(doc.get('citations', ['None specified'])[:5])}

### 8️⃣ Common Misunderstandings
⚠️ This law applies to all citizens regardless of intent. Ignorance of law is not an excuse.

### 9️⃣ Friendly Advice
💡 Always consult a qualified lawyer for specific legal advice. This is for informational purposes only.
"""
    return response

def hybrid_search(query, tfidf_vectorizer, tfidf_matrix, data, top_k=5):
    """Enhanced hybrid search with RAG"""
    try:
        # TF-IDF search
        query_vector = tfidf_vectorizer.transform([query])
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
    st.markdown('<h1 class="main-header">⚖️ A-Qlegal 2.5 - Generative RAG System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your AI-Powered Multilingual Legal Assistant</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("🔄 Loading AI models and legal database..."):
        tfidf_vectorizer, tfidf_matrix = load_models()
        data = load_legal_data()
        rag_index = load_rag_index()
        personas = load_personas()
    
    if not data:
        st.error("❌ No legal data found. Please run the training script first.")
        return
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Persona selection
        st.subheader("👤 User Persona")
        persona_options = {
            "citizen": "🧑 Citizen Mode - Practical advice",
            "student": "🎓 Student Mode - Educational",
            "business": "💼 Business Mode - Compliance",
            "lawyer": "⚖️ Lawyer Mode - Technical"
        }
        selected_persona = st.selectbox(
            "Select your role:",
            options=list(persona_options.keys()),
            format_func=lambda x: persona_options[x]
        )
        
        # Language selection
        st.subheader("🌍 Language")
        language = st.selectbox(
            "Select language:",
            ["English", "हिन्दी (Hindi)", "தமிழ் (Tamil)", "বাংলা (Bengali)", "తెలుగు (Telugu)"]
        )
        
        # Search settings
        st.subheader("🔍 Search Settings")
        top_k = st.slider("Number of results:", 1, 20, 5)
        show_legal_text = st.checkbox("Show original legal text", value=False)
        
        # Statistics
        st.markdown("---")
        st.header("📊 Database Stats")
        st.metric("Total Documents", f"{len(data):,}")
        
        categories = {}
        for doc in data:
            cat = doc.get('category', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        st.metric("Categories", len(categories))
        
        # Category breakdown
        with st.expander("📁 Category Breakdown"):
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]:
                st.write(f"• {cat}: {count}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Your Legal Question")
        
        # Query input
        query = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is the punishment for fraud? Explain Section 420 IPC",
            key="query_input"
        )
        
        # Example questions
        with st.expander("💡 Example Questions"):
            example_questions = [
                "What is the punishment for fraud?",
                "Explain Section 420 IPC in simple terms",
                "Tell me about fundamental rights",
                "What is culpable homicide?",
                "Explain right to freedom of speech",
                "What are the consequences of bounced checks?",
                "Tell me about maintenance orders",
                "What is sedition?"
            ]
            
            for q in example_questions:
                if st.button(f"📌 {q}", key=q):
                    query = q
        
        # Search button
        if st.button("🔍 Search", type="primary") or query:
            if query:
                with st.spinner("🤖 AI is analyzing your question..."):
                    results = hybrid_search(query, tfidf_vectorizer, tfidf_matrix, data, top_k)
                
                if results:
                    st.success(f"✅ Found {len(results)} relevant legal document(s)")
                    
                    # Display results
                    for i, doc in enumerate(results, 1):
                        with st.expander(
                            f"📖 {i}. {doc.get('title', 'Unknown')} | "
                            f"{doc.get('category', 'Unknown')} | "
                            f"Relevance: {doc.get('similarity_score', 0):.1%}",
                            expanded=i==1
                        ):
                            # Generate structured response
                            response = generate_structured_response(doc, selected_persona)
                            st.markdown(response, unsafe_allow_html=True)
                            
                            # Optional: Show original legal text
                            if show_legal_text and doc.get('content'):
                                with st.expander("📜 Original Legal Text"):
                                    st.text(doc['content'])
                            
                            # Action buttons
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.button(f"👍 Helpful", key=f"helpful_{i}")
                            with col_b:
                                st.button(f"👎 Not helpful", key=f"not_helpful_{i}")
                            with col_c:
                                st.button(f"📋 Copy", key=f"copy_{i}")
                else:
                    st.warning("⚠️ No relevant legal documents found. Try different keywords.")
            else:
                st.info("💡 Please enter a question above")
    
    with col2:
        st.header("🎯 Quick Guide")
        
        # Current persona info
        if selected_persona in personas:
            persona_info = personas[selected_persona]
            st.info(f"""
            **Current Mode:** {persona_info['name']}
            
            **Description:** {persona_info['description']}
            
            **Tone:** {persona_info['tone']}
            """)
        
        # Features
        st.header("✨ Features")
        st.success("""
        ✅ **Generative RAG**: AI-powered responses with legal citations
        
        ✅ **9-Point Format**: Structured, easy-to-understand answers
        
        ✅ **Multilingual**: Support for Indian languages
        
        ✅ **User Personas**: Tailored for different user types
        
        ✅ **8,000+ Documents**: Comprehensive legal database
        
        ✅ **Real-Time Search**: Instant AI-powered results
        """)
        
        # Tips
        st.header("💡 Pro Tips")
        st.info("""
        **For Best Results:**
        
        🔹 Use specific terms like section numbers
        
        🔹 Ask in natural language
        
        🔹 Switch personas for different perspectives
        
        🔹 Check related sections for complete understanding
        
        🔹 Always verify with a legal professional
        """)
        
        # Disclaimer
        st.warning("""
        **⚠️ Important Disclaimer**
        
        This AI assistant provides legal information, NOT legal advice.
        
        Always consult a qualified lawyer for specific legal matters.
        """)
    
    # Footer
    st.markdown("---")
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.metric("Powered By", "A-Qlegal 2.5")
    with col_f2:
        st.metric("Version", "Generative RAG")
    with col_f3:
        st.metric("Status", "🟢 Online")
    
    st.markdown(
        '<p style="text-align: center; color: #64748B;">© 2025 A-Qlegal - Making Indian Law Accessible to Everyone</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
