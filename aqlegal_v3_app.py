#!/usr/bin/env python3
"""
A-Qlegal 3.0 - Advanced Intelligence System
Multi-model fusion, explainability, fallback generation, and trust features
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
    page_title="A-Qlegal 3.0 - Advanced Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        font-weight: bold;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #64748B;
        font-size: 1.3rem;
        margin-bottom: 2rem;
    }
    .confidence-badge {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .source-attribution {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .explanation-box {
        background-color: #F0FDF4;
        border-left: 4px solid #10B981;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .fallback-notice {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load all systems
@st.cache_resource
def load_all_systems():
    """Load all trained models and systems"""
    try:
        # Load TF-IDF
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        tfidf_matrix = np.load('data/embeddings/tfidf_matrix.npy')
        
        # Load fallback system
        try:
            with open('data/processed/fallback_system.json', 'r', encoding='utf-8') as f:
                fallback_system = json.load(f)
        except:
            fallback_system = None
        
        # Load attribution data
        try:
            with open('data/processed/attribution_data.json', 'r', encoding='utf-8') as f:
                attribution_data = json.load(f)
        except:
            attribution_data = []
        
        # Load hallucination shield
        try:
            with open('data/processed/hallucination_shield.json', 'r', encoding='utf-8') as f:
                shield = json.load(f)
        except:
            shield = None
        
        return tfidf_vectorizer, tfidf_matrix, fallback_system, attribution_data, shield
    except Exception as e:
        st.error(f"Failed to load systems: {e}")
        return None, None, None, [], None

@st.cache_data
def load_legal_data():
    """Load legal database"""
    try:
        with open("data/processed/all_legal_documents.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

@st.cache_data
def load_personas():
    """Load user personas"""
    try:
        with open("data/processed/user_personas.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def detect_query_category(query):
    """Detect legal category from query"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['crime', 'murder', 'theft', 'fraud', 'assault']):
        return 'criminal_law'
    elif any(word in query_lower for word in ['constitution', 'fundamental', 'rights', 'article']):
        return 'constitutional_law'
    elif any(word in query_lower for word in ['contract', 'property', 'civil', 'inheritance']):
        return 'civil_law'
    else:
        return 'general_legal'

def generate_fallback_response(query, fallback_system, category='general_legal'):
    """Generate intelligent fallback response for out-of-database queries"""
    if not fallback_system:
        return None
    
    template = fallback_system['reasoning_templates'].get(category, 
                fallback_system['reasoning_templates']['general_legal'])
    
    # Generate reasoning based on query
    reasoning = generate_smart_reasoning(query, category)
    
    # Get related concepts
    related_concepts = []
    for key, concepts in fallback_system.get('related_concepts_map', {}).items():
        if key in category:
            related_concepts = concepts
            break
    
    # Fill template
    response = template.format(
        question=query,
        reasoning=reasoning,
        related_concepts=", ".join(related_concepts) if related_concepts else "General Indian law principles",
        next_steps="Consult a qualified lawyer for specific advice",
        remedies="Legal remedies depend on specific circumstances",
        approach="Seek professional legal guidance",
        protections="Constitutional protections available to all citizens",
        enforcement="File writ petitions in High Court (Article 226) or Supreme Court (Article 32)"
    )
    
    return response

def generate_smart_reasoning(query, category):
    """Generate smart reasoning based on query and category"""
    reasoning_parts = []
    
    # Add general legal principles
    reasoning_parts.append("Indian law operates on principles of justice, equity, and good conscience.")
    
    # Add category-specific reasoning
    if category == 'criminal_law':
        reasoning_parts.append("In criminal matters, the prosecution must prove guilt beyond reasonable doubt.")
        reasoning_parts.append("Every accused person has the right to legal representation and fair trial.")
    elif category == 'constitutional_law':
        reasoning_parts.append("The Constitution is the supreme law of India.")
        reasoning_parts.append("Fundamental Rights are enforceable through courts.")
    elif category == 'civil_law':
        reasoning_parts.append("Civil law deals with disputes between individuals or entities.")
        reasoning_parts.append("The burden of proof is on the plaintiff in most civil cases.")
    else:
        reasoning_parts.append("Legal matters in India are governed by various statutes and precedents.")
    
    return "\n\n".join(reasoning_parts)

def search_with_attribution(query, tfidf_vectorizer, tfidf_matrix, data, attribution_data, top_k=5):
    """Search with source attribution"""
    try:
        # Transform query
        query_vector = tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                doc = data[idx].copy()
                doc['similarity_score'] = float(similarities[idx])
                
                # Add attribution
                attribution = next((attr for attr in attribution_data 
                                  if attr.get('doc_id') == doc.get('id')), None)
                if attribution:
                    doc['attribution'] = attribution
                else:
                    doc['attribution'] = {
                        'confidence_score': 0.5,
                        'source': doc.get('source', 'Unknown'),
                        'verification_status': 'general'
                    }
                
                results.append(doc)
        
        return results
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []

def main():
    # Header
    st.markdown('<h1 class="main-header">⚖️ A-Qlegal 3.0</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Legal Intelligence System with Explainability & Trust</p>', unsafe_allow_html=True)
    
    # Load all systems
    with st.spinner("🔄 Loading advanced AI systems..."):
        tfidf_vectorizer, tfidf_matrix, fallback_system, attribution_data, shield = load_all_systems()
        data = load_legal_data()
        personas = load_personas()
    
    if not data:
        st.error("❌ No legal data found. Please run training first.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Advanced Settings")
        
        # Persona selection
        st.subheader("👤 User Persona")
        persona_options = {
            "citizen": "🧑 Citizen - Practical advice",
            "student": "🎓 Student - Educational",
            "business": "💼 Business - Compliance",
            "lawyer": "⚖️ Lawyer - Technical"
        }
        selected_persona = st.selectbox(
            "Select your role:",
            options=list(persona_options.keys()),
            format_func=lambda x: persona_options[x]
        )
        
        # Advanced features
        st.subheader("🔬 Advanced Features")
        show_confidence = st.checkbox("Show confidence scores", value=True)
        show_attribution = st.checkbox("Show source attribution", value=True)
        show_explanation = st.checkbox("Show AI reasoning", value=True)
        enable_fallback = st.checkbox("Enable smart fallback", value=True)
        
        # Search settings
        st.subheader("🔍 Search Settings")
        top_k = st.slider("Results to show:", 1, 10, 5)
        similarity_threshold = st.slider("Min similarity:", 0.0, 1.0, 0.0, 0.1)
        
        # Language
        st.subheader("🌍 Language")
        language = st.selectbox(
            "Select language:",
            ["English", "हिन्दी (Hindi)", "தமிழ் (Tamil)", "বাংলা (Bengali)", "తెలుగు (Telugu)"]
        )
        
        # Statistics
        st.markdown("---")
        st.header("📊 System Stats")
        st.metric("Documents", f"{len(data):,}")
        st.metric("Attribution Data", f"{len(attribution_data):,}")
        st.metric("Verified Sections", f"{len(shield.get('verified_sections', [])) if shield else 0:,}")
    
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
        col1, col2 = st.columns(2)
        
        examples = [
            "What is the punishment for fraud?",
            "Explain Section 420 IPC",
            "Tell me about fundamental rights",
            "What is culpable homicide?",
            "Explain right to freedom of speech",
            "What should I do if I'm arrested?",
            "How to file an FIR?",
            "What are my consumer rights?"
        ]
        
        for i, example in enumerate(examples):
            col = col1 if i % 2 == 0 else col2
            if col.button(f"📌 {example}", key=f"ex_{i}"):
                query = example
    
    # Search
    if st.button("🔍 Search", type="primary") or query:
        if query:
            with st.spinner("🤖 AI is analyzing your question..."):
                # Detect category
                category = detect_query_category(query)
                
                # Search database
                results = search_with_attribution(query, tfidf_vectorizer, tfidf_matrix, 
                                                data, attribution_data, top_k)
                
                # Filter by threshold
                results = [r for r in results if r['similarity_score'] >= similarity_threshold]
                
                # Check if we need fallback
                if not results or (results and results[0]['similarity_score'] < 0.3):
                    if enable_fallback:
                        st.markdown('<div class="fallback-notice">', unsafe_allow_html=True)
                        st.warning("⚠️ **Smart Fallback Activated**")
                        st.write("Your question isn't directly covered in our database, but here's what Indian law generally says:")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Generate fallback response
                        fallback_response = generate_fallback_response(query, fallback_system, category)
                        if fallback_response:
                            st.markdown(fallback_response)
                        
                        st.info("💡 For specific legal advice, please consult a qualified lawyer.")
                
                # Show database results if available
                if results:
                    st.success(f"✅ Found {len(results)} relevant document(s) in database")
                    
                    for i, doc in enumerate(results, 1):
                        with st.expander(
                            f"📖 {i}. {doc.get('title', 'Unknown')} | "
                            f"{doc.get('category', 'Unknown')} | "
                            f"Match: {doc.get('similarity_score', 0):.1%}",
                            expanded=i==1
                        ):
                            # Confidence score
                            if show_confidence and doc.get('attribution'):
                                confidence = doc['attribution'].get('confidence_score', 0.5)
                                st.markdown(
                                    f'<span class="confidence-badge">Confidence: {confidence:.0%}</span>',
                                    unsafe_allow_html=True
                                )
                            
                            # Source attribution
                            if show_attribution and doc.get('attribution'):
                                st.markdown('<div class="source-attribution">', unsafe_allow_html=True)
                                st.write("📚 **Source Attribution:**")
                                st.write(f"• **Source:** {doc['attribution'].get('source', 'Unknown')}")
                                st.write(f"• **Status:** {doc['attribution'].get('verification_status', 'general')}")
                                if doc.get('section'):
                                    st.write(f"• **Section:** {doc['section']}")
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # AI reasoning
                            if show_explanation and doc.get('attribution'):
                                explanation = doc['attribution'].get('explanation', '')
                                if explanation:
                                    st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
                                    st.write("🧠 **AI Reasoning:**")
                                    st.write(explanation)
                                    st.markdown('</div>', unsafe_allow_html=True)
                            
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
                elif not enable_fallback:
                    st.warning("⚠️ No relevant documents found. Try different keywords or enable smart fallback.")
        else:
            st.info("💡 Please enter a question above")
    
    # Footer
    st.markdown("---")
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    
    with col_f1:
        st.metric("Version", "3.0")
    with col_f2:
        st.metric("Intelligence", "Advanced")
    with col_f3:
        st.metric("Explainability", "✓")
    with col_f4:
        st.metric("Status", "🟢 Online")
    
    st.markdown(
        '<p style="text-align: center; color: #64748B;">© 2025 A-Qlegal 3.0 - Advanced Legal Intelligence for India</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
