#!/usr/bin/env python3
"""
A-Qlegal 3.0 - Simplified Generative and Retrieval-Augmented AI Legal Assistant
Trained on Indian law datasets with advanced reasoning capabilities
"""

import json
import streamlit as st
import numpy as np
import pickle
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# Configure page
st.set_page_config(
    page_title="A-Qlegal 3.0 - Advanced Legal AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AQlegalV3:
    def __init__(self):
        self.data_dir = Path("data")
        self.models_dir = Path("models")
        self.legal_data = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.confidence_threshold = 0.65
        
    @st.cache_resource
    def load_models(_self):
        """Load all AI models and data"""
        try:
            # Load TF-IDF components
            with open('models/tfidf_vectorizer.pkl', 'rb') as f:
                _self.tfidf_vectorizer = pickle.load(f)
            _self.tfidf_matrix = np.load('data/embeddings/tfidf_matrix.npy')
            return True
        except Exception as e:
            st.error(f"Model loading failed: {e}")
            return False
    
    @st.cache_data
    def load_legal_data(_self):
        """Load and process all legal datasets"""
        all_data = []
        
        # Load processed data
        try:
            with open("data/processed/all_legal_documents.json", "r", encoding="utf-8") as f:
                processed_data = json.load(f)
                all_data.extend(processed_data)
        except FileNotFoundError:
            pass
        
        # Load enhanced dataset v2
        try:
            with open("data/enhanced_legal_documents_v2.json", "r", encoding="utf-8") as f:
                enhanced_data = json.load(f)
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
                        "keywords": item.get("keywords", []),
                        "simplified_summary": item.get("simplified_summary", ""),
                        "real_life_example": item.get("real_life_example", "")
                    }
                    all_data.append(formatted_item)
        except FileNotFoundError:
            pass
        
        return all_data
    
    def semantic_search(self, query, data, top_k=3):
        """Perform semantic search using TF-IDF"""
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            return []
        
        try:
            # Transform query
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top results
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    doc = data[idx].copy()
                    doc['similarity_score'] = float(similarities[idx])
                    results.append(doc)
            
            return results
        except Exception as e:
            st.error(f"Semantic search failed: {e}")
            return []
    
    def keyword_search(self, query, data, top_k=3):
        """Enhanced keyword search with self-defense specific handling"""
        query_lower = query.lower()
        results = []
        
        # Self-defense and legal keywords
        legal_keywords = [
            'self defense', 'self-defence', 'private defence', 'right to defend',
            'defend yourself', 'kill in self defense', 'self protection',
            'section 96', 'section 97', 'section 99', 'section 100',
            'murder', 'homicide', 'assault', 'battery', 'theft', 'robbery',
            'fraud', 'cheating', 'contract', 'property', 'marriage', 'divorce',
            'criminal force', 'voluntarily causing hurt', 'grievous hurt'
        ]
        
        for item in data:
            score = 0
            content_lower = (item.get('content', '') + ' ' + item.get('title', '')).lower()
            
            # Direct keyword matching
            for keyword in legal_keywords:
                if keyword in content_lower:
                    score += 2
            
            # Title matching
            if any(word in item.get('title', '').lower() for word in query_lower.split()):
                score += 1.5
            
            # Content matching
            if any(word in content_lower for word in query_lower.split()):
                score += 1
            
            # Section matching
            if 'section' in item.get('section', '').lower():
                if any(section in item.get('section', '') for section in re.findall(r'\d+', query)):
                    score += 3
            
            if score > 0:
                item['similarity_score'] = float(score)
                results.append(item)
        
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_k]
    
    def generate_legal_explanation(self, query, context_docs):
        """Generate legal explanation using rule-based reasoning"""
        if not context_docs:
            return self.get_general_legal_advice(query)
        
        # Extract relevant information
        sections = [doc.get('section', '') for doc in context_docs if doc.get('section')]
        punishments = [doc.get('punishment', '') for doc in context_docs if doc.get('punishment')]
        
        # Generate explanation based on context
        explanation = f"Based on the relevant legal provisions"
        if sections:
            explanation += f" (including {', '.join(sections[:2])})"
        explanation += ", here's what you need to know:\n\n"
        
        # Add context-specific information
        if any('self defense' in doc.get('content', '').lower() for doc in context_docs):
            explanation += "• Self-defense is a legal right under Indian law (Sections 96-106 IPC)\n"
            explanation += "• You can use reasonable force to protect yourself or others\n"
            explanation += "• The force used must be proportional to the threat\n"
            explanation += "• You cannot claim self-defense if you initiated the attack\n"
        
        if punishments:
            explanation += f"• Punishment: {punishments[0]}\n"
        
        return explanation
    
    def get_general_legal_advice(self, query):
        """Provide general legal advice when no specific context is found"""
        query_lower = query.lower()
        
        if 'self defense' in query_lower or 'kill' in query_lower:
            return """• Self-defense is a fundamental right under Indian law (Sections 96-106 IPC)
• You can use reasonable force to protect yourself, others, or property
• The force must be proportional to the threat faced
• You cannot claim self-defense if you were the aggressor
• In extreme cases, causing death in self-defense may be justified
• Always report the incident to police immediately
• Consult a lawyer for specific situations"""
        
        elif 'theft' in query_lower:
            return """• Theft is defined under Section 378 IPC
• Taking movable property without consent with dishonest intention
• Punishment: Up to 3 years imprisonment and fine
• Theft becomes robbery if force or threat is used
• Report theft to police immediately
• Keep evidence of the stolen property"""
        
        elif 'fraud' in query_lower:
            return """• Fraud is covered under Section 420 IPC (Cheating)
• Deceiving someone to cause wrongful gain or loss
• Punishment: Up to 7 years imprisonment and fine
• Gather evidence of the fraudulent act
• File a complaint with police or cyber cell
• Consider civil remedies for recovery"""
        
        else:
            return """• This appears to be a legal question
• Indian law covers most situations comprehensively
• Consult a qualified lawyer for specific advice
• Keep all relevant documents and evidence
• Be aware of your rights and obligations
• Consider alternative dispute resolution methods"""
    
    def process_query(self, query):
        """Main query processing pipeline"""
        # Step 1: Semantic search
        semantic_results = self.semantic_search(query, self.legal_data, 3)
        
        # Step 2: Keyword search if semantic search fails
        if not semantic_results:
            semantic_results = self.keyword_search(query, self.legal_data, 3)
        
        # Step 3: Check confidence
        max_confidence = max([doc.get('similarity_score', 0) for doc in semantic_results]) if semantic_results else 0
        
        # Adjust threshold based on search type (semantic vs keyword)
        # Semantic search uses 0-1 scale, keyword search uses 0-20+ scale
        threshold = self.confidence_threshold if max_confidence <= 1.0 else 5.0
        
        if max_confidence >= threshold:
            # High confidence - use retrieved results
            return self.format_retrieved_response(query, semantic_results)
        else:
            # Low confidence - use generative mode
            return self.format_generative_response(query, semantic_results)
    
    def format_retrieved_response(self, query, results):
        """Format response from retrieved documents"""
        response = {
            "type": "retrieved",
            "confidence": "high",
            "query": query,
            "sections": [doc.get('section', 'N/A') for doc in results],
            "explanation": results[0].get('simplified_summary', results[0].get('content', ''))[:300] + "...",
            "example": results[0].get('real_life_example', ''),
            "punishment": results[0].get('punishment', ''),
            "source": results[0].get('source', ''),
            "documents": results
        }
        return response
    
    def format_generative_response(self, query, context_docs):
        """Format response from generative model"""
        generated_text = self.generate_legal_explanation(query, context_docs)
        
        response = {
            "type": "generated",
            "confidence": "ai-inferred",
            "query": query,
            "sections": [doc.get('section', 'N/A') for doc in context_docs] if context_docs else ["No direct match found"],
            "explanation": generated_text,
            "example": "",
            "punishment": "",
            "source": "AI-generated based on general legal principles",
            "documents": context_docs
        }
        return response

def main():
    st.title("⚖️ A-Qlegal 3.0 - Advanced Legal AI Assistant")
    st.markdown("**Generative and Retrieval-Augmented AI for Indian Law**")
    
    # Initialize session state
    if 'selected_query' not in st.session_state:
        st.session_state.selected_query = ""
    
    # Initialize the system
    aqlegal = AQlegalV3()
    
    # Load models and data
    with st.spinner("Loading AI models and legal data..."):
        if not aqlegal.load_models():
            st.error("Failed to load models. Please check the setup.")
            return
        
        aqlegal.legal_data = aqlegal.load_legal_data()
    
    st.success(f"✅ Loaded {len(aqlegal.legal_data)} legal documents")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.65,
            help="Higher values require more confident matches before using generative mode"
        )
        aqlegal.confidence_threshold = confidence_threshold
        
        st.header("📊 Statistics")
        st.metric("Legal Documents", len(aqlegal.legal_data))
        st.metric("Categories", len(set(doc.get('category', '') for doc in aqlegal.legal_data)))
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask a Legal Question")
        
        # Use the selected query if available
        default_query = st.session_state.selected_query if st.session_state.selected_query else ""
        
        query = st.text_area(
            "Enter your legal question:",
            value=default_query,
            placeholder="e.g., Can I kill someone in self-defense? What is the punishment for fraud?",
            height=100,
            key="query_input"
        )
        
        if st.button("🔍 Analyze", type="primary", use_container_width=True):
            if query:
                # Clear the selected query after analysis
                st.session_state.selected_query = ""
                with st.spinner("Analyzing legal query..."):
                    response = aqlegal.process_query(query)
                
                # Display response
                st.markdown("---")
                st.markdown("### ⚖️ Law Query Summary")
                st.markdown(f"**User Question:** {response['query']}")
                
                st.markdown("**Closest Legal Context / Law Section(s):**")
                for section in response['sections']:
                    if section != 'N/A':
                        st.markdown(f"- {section}")
                
                st.markdown("**Simplified Explanation:**")
                st.markdown(f"- {response['explanation']}")
                
                if response['example']:
                    st.markdown("**Example / Use Case:**")
                    st.markdown(f"- {response['example']}")
                
                if response['punishment']:
                    st.markdown("**Punishment:**")
                    st.markdown(f"- {response['punishment']}")
                
                # Confidence indicator
                if response['type'] == 'retrieved':
                    st.success(f"✅ High Confidence Match (Retrieved from legal database)")
                else:
                    st.warning(f"⚠️ AI-Generated Response (Based on general legal principles)")
                
                st.markdown("**Disclaimer:** This is an AI-generated explanation. For verified legal advice, consult a qualified lawyer.")
                
                # Show source documents if available
                if response['documents']:
                    with st.expander("📚 Source Documents", expanded=False):
                        for i, doc in enumerate(response['documents'], 1):
                            st.markdown(f"**{i}. {doc.get('title', 'Unknown')}**")
                            st.markdown(f"Section: {doc.get('section', 'N/A')}")
                            st.markdown(f"Similarity: {doc.get('similarity_score', 0):.3f}")
                            st.markdown("---")
            else:
                st.warning("Please enter a legal question.")
    
    with col2:
        st.header("🎯 Quick Examples")
        example_queries = [
            "Can I kill someone in self-defense?",
            "What is the punishment for theft?",
            "Can a minor enter into a contract?",
            "What are my rights if arrested?",
            "How to file a divorce case?",
            "What is the punishment for fraud?"
        ]
        
        for example in example_queries:
            if st.button(example, use_container_width=True, key=f"example_{example}"):
                # Set the selected query in session state
                st.session_state.selected_query = example
                st.rerun()
        
        st.header("📈 System Status")
        st.metric("AI Confidence", f"{confidence_threshold:.1%}")
        st.metric("Search Mode", "Semantic + Generative")
        st.metric("Model Status", "✅ Active")

if __name__ == "__main__":
    main()
