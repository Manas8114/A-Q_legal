#!/usr/bin/env python3
"""
A-Qlegal 3.0 - Production-Ready Legal AI Assistant
Generative and Retrieval-Augmented AI for Indian Law

Features:
- Dual search system (Semantic + Keyword)
- Confidence-based response generation
- 8,369+ legal documents coverage
- Real-time legal query analysis

Author: A-Qlegal Team
Version: 3.0.0
"""

import json
import streamlit as st
import numpy as np
import pickle
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings("ignore")

# Configure Streamlit page
st.set_page_config(
    page_title="A-Qlegal 3.0 - Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


class AQlegalV3:
    """
    A-Qlegal 3.0 Legal AI Assistant
    
    Combines semantic search, keyword matching, and AI-generated explanations
    to provide comprehensive legal guidance based on Indian law.
    """
    
    # Constants
    SEMANTIC_CONFIDENCE_THRESHOLD = 0.65  # For TF-IDF similarity (0-1 scale)
    KEYWORD_CONFIDENCE_THRESHOLD = 5.0    # For keyword matching (0-20+ scale)
    SEMANTIC_MIN_THRESHOLD = 0.1          # Minimum TF-IDF score to consider
    DEFAULT_TOP_K = 3                     # Number of results to return
    
    def __init__(self):
        """Initialize the A-Qlegal system"""
        self.data_dir = Path("data")
        self.models_dir = Path("models")
        self.legal_data: List[Dict[str, Any]] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        
    @st.cache_resource
    def load_models(_self) -> bool:
        """
        Load TF-IDF vectorizer and matrix for semantic search
        
        Returns:
            bool: True if models loaded successfully, False otherwise
        """
        try:
            vectorizer_path = _self.models_dir / 'tfidf_vectorizer.pkl'
            matrix_path = _self.data_dir / 'embeddings' / 'tfidf_matrix.npy'
            
            with open(vectorizer_path, 'rb') as f:
                _self.tfidf_vectorizer = pickle.load(f)
            
            _self.tfidf_matrix = np.load(matrix_path)
            
            return True
        except FileNotFoundError as e:
            st.error(f"❌ Model files not found: {e}")
            return False
        except Exception as e:
            st.error(f"❌ Model loading failed: {e}")
            return False
    
    @st.cache_data
    def load_legal_data(_self) -> List[Dict[str, Any]]:
        """
        Load and process all legal datasets
        
        Returns:
            List of legal documents with standardized format
        """
        all_data = []
        
        # Load processed legal documents
        processed_path = _self.data_dir / "processed" / "all_legal_documents.json"
        if processed_path.exists():
            try:
                with open(processed_path, "r", encoding="utf-8") as f:
                    processed_data = json.load(f)
                    all_data.extend(processed_data)
            except Exception as e:
                st.warning(f"⚠️ Failed to load processed data: {e}")
        
        # Load enhanced dataset v2
        enhanced_path = _self.data_dir / "enhanced_legal_documents_v2.json"
        if enhanced_path.exists():
            try:
                with open(enhanced_path, "r", encoding="utf-8") as f:
                    enhanced_data = json.load(f)
                    
                    # Standardize format
                    for item in enhanced_data:
                        formatted_item = {
                            "id": item.get("id", ""),
                            "title": item.get("title", ""),
                            "content": " ".join([
                                item.get('text', ''),
                                item.get('simplified_summary', ''),
                                item.get('real_life_example', '')
                            ]).strip(),
                            "category": item.get("category", "").lower().replace(" ", "_"),
                            "section": item.get("section", ""),
                            "punishment": item.get("punishment", ""),
                            "citations": item.get("citations", []),
                            "source": item.get("source", ""),
                            "keywords": item.get("keywords", []),
                            "simplified_summary": item.get("simplified_summary", ""),
                            "real_life_example": item.get("real_life_example", "")
                        }
                        all_data.append(formatted_item)
            except Exception as e:
                st.warning(f"⚠️ Failed to load enhanced data: {e}")
        
        return all_data
    
    def semantic_search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """
        Perform semantic search using TF-IDF vectorization
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of documents with similarity scores
        """
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            return []
        
        if not self.legal_data:
            return []
        
        try:
            # Transform query to TF-IDF vector
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top K indices
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            # Build results
            results = []
            for idx in top_indices:
                if idx < len(self.legal_data) and similarities[idx] > self.SEMANTIC_MIN_THRESHOLD:
                    doc = self.legal_data[idx].copy()
                    doc['similarity_score'] = float(similarities[idx])
                    doc['search_type'] = 'semantic'
                    results.append(doc)
            
            return results
            
        except Exception as e:
            st.error(f"❌ Semantic search failed: {e}")
            return []
    
    def keyword_search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """
        Perform intelligent keyword search with context-aware legal pattern matching
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of documents with relevance scores
        """
        if not self.legal_data:
            return []
        
        query_lower = query.lower()
        query_words = set(w for w in query_lower.split() if len(w) > 2)
        
        # Stop words to ignore (common words that don't add meaning)
        stop_words = {'can', 'what', 'how', 'when', 'where', 'who', 'the', 'is', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or'}
        query_words = query_words - stop_words
        
        # Comprehensive legal domain patterns with context
        legal_domains = {
            # Contract Law
            'contract': {
                'primary_terms': ['contract', 'agreement', 'sign', 'bind', 'execute', 'valid contract'],
                'related_terms': ['minor', 'age', 'capacity', 'competent', 'void', 'voidable', 'consideration', 'offer', 'acceptance'],
                'negative_terms': ['kidnap', 'abduct', 'murder', 'theft'],  # Exclude these
                'sources': ['Indian Contract Act', 'Contract Act'],
                'weight': 25.0
            },
            # Self-Defense
            'self_defense': {
                'primary_terms': ['self defense', 'self-defense', 'self defence', 'private defence', 
                                'right to defend', 'defend myself', 'defend yourself'],
                'related_terms': ['force', 'protect', 'attack', 'threat', 'body', 'property'],
                'negative_terms': ['kidnap', 'abduct', 'extortion'],
                'sections': ['96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106'],
                'weight': 25.0
            },
            # Theft & Property Crimes
            'theft': {
                'primary_terms': ['theft', 'steal', 'stolen', 'robbery', 'dacoity'],
                'related_terms': ['property', 'movable', 'dishonest', 'intention'],
                'negative_terms': ['contract', 'agreement', 'defend'],
                'sections': ['378', '379', '380', '381', '382', '390', '391', '392'],
                'weight': 20.0
            },
            # Fraud & Cheating
            'fraud': {
                'primary_terms': ['fraud', 'cheat', 'deceive', 'dishonest', 'forgery'],
                'related_terms': ['wrongful gain', 'wrongful loss', 'false', 'mislead'],
                'negative_terms': ['theft', 'kidnap', 'murder'],
                'sections': ['415', '416', '417', '418', '419', '420', '463', '464', '465'],
                'weight': 20.0
            },
            # Murder & Homicide
            'murder': {
                'primary_terms': ['murder', 'kill', 'homicide', 'death', 'culpable homicide'],
                'related_terms': ['intention', 'knowledge', 'cause death'],
                'negative_terms': ['contract', 'theft', 'fraud'],
                'sections': ['299', '300', '302', '304', '304A', '307'],
                'weight': 20.0,
                'context_required': ['self', 'defend']  # Only high score if these are NOT present
            },
            # Assault & Hurt
            'assault': {
                'primary_terms': ['assault', 'hurt', 'battery', 'grievous hurt', 'criminal force'],
                'related_terms': ['voluntarily', 'causing', 'injury', 'harm'],
                'negative_terms': ['kidnap', 'theft'],
                'sections': ['319', '320', '321', '322', '323', '324', '325', '350', '351'],
                'weight': 18.0
            },
            # Kidnapping & Abduction
            'kidnapping': {
                'primary_terms': ['kidnap', 'abduct', 'kidnapping', 'abduction'],
                'related_terms': ['child', 'minor', 'lawful guardian', 'begging'],
                'negative_terms': ['contract', 'agreement', 'defend', 'theft', 'fraud'],
                'sections': ['359', '360', '361', '363', '363A', '364', '365', '366', '367'],
                'weight': 15.0
            },
            # Arrest & Criminal Procedure
            'arrest': {
                'primary_terms': ['arrest', 'arrested', 'detention', 'custody'],
                'related_terms': ['rights', 'bail', 'police', 'warrant', 'cognizable'],
                'negative_terms': ['kidnap', 'murder', 'theft'],
                'sources': ['CrPC', 'Criminal Procedure Code'],
                'sections': ['41', '41A', '41B', '41C', '41D', '50', '56', '57'],
                'weight': 22.0
            },
            # Marriage & Family
            'marriage': {
                'primary_terms': ['marriage', 'marry', 'divorce', 'matrimonial', 'spouse'],
                'related_terms': ['husband', 'wife', 'dissolution', 'separation', 'alimony'],
                'negative_terms': ['kidnap', 'murder', 'theft', 'fraud'],
                'weight': 20.0
            }
        }
        
        # Identify query domain
        detected_domains = []
        for domain, config in legal_domains.items():
            # Check if query contains primary terms
            if any(term in query_lower for term in config.get('primary_terms', [])):
                # Check if negative terms are NOT heavily present
                negative_count = sum(1 for term in config.get('negative_terms', []) if term in query_lower)
                if negative_count == 0:
                    detected_domains.append((domain, config))
        
        results = []
        
        for doc in self.legal_data:
            score = 0.0
            content_lower = (doc.get('content', '') + ' ' + doc.get('title', '')).lower()
            title_lower = doc.get('title', '').lower()
            source_lower = doc.get('source', '').lower()
            section_num = re.search(r'(\d+[A-Z]?)', doc.get('section', ''))
            section_number = section_num.group(1) if section_num else ''
            
            # SCORING STRATEGY
            
            # 1. DOMAIN-SPECIFIC SCORING (Highest Priority)
            for domain, config in detected_domains:
                domain_score = 0
                
                # Check primary terms in title (very high weight)
                if any(term in title_lower for term in config.get('primary_terms', [])):
                    domain_score += config['weight'] * 2
                
                # Check primary terms in content
                if any(term in content_lower for term in config.get('primary_terms', [])):
                    domain_score += config['weight']
                
                # Check related terms
                related_matches = sum(1 for term in config.get('related_terms', []) if term in content_lower)
                domain_score += related_matches * 3
                
                # Check source matching (for contract, CrPC, etc.)
                if 'sources' in config:
                    if any(source in source_lower for source in config['sources']):
                        domain_score += 15.0
                
                # Check section matching
                if 'sections' in config and section_number in config['sections']:
                    domain_score += 20.0
                
                # NEGATIVE SCORING - Penalize wrong domains
                if any(neg_term in title_lower for neg_term in config.get('negative_terms', [])):
                    domain_score -= 20.0
                
                score += domain_score
            
            # 2. EXACT PHRASE MATCHING
            # Remove common question words for phrase matching
            clean_query = query_lower
            for word in ['can', 'what', 'how', 'is', 'the', 'a', 'an']:
                clean_query = clean_query.replace(f' {word} ', ' ')
            
            if clean_query.strip() in title_lower:
                score += 30.0
            
            # 3. TITLE WORD MATCHING (High importance)
            title_words = set(w for w in title_lower.split() if len(w) > 2 and w not in stop_words)
            common_title_words = query_words & title_words
            score += len(common_title_words) * 5.0
            
            # 4. SECTION NUMBER EXACT MATCH
            section_numbers = re.findall(r'\d+[A-Z]?', query)
            if section_numbers and section_number in section_numbers:
                score += 25.0
            
            # 5. KEYWORD MATCHING
            if 'keywords' in doc and doc['keywords']:
                keyword_matches = sum(1 for kw in doc['keywords'] if kw.lower() in query_lower)
                score += keyword_matches * 4.0
            
            # 6. CONTENT RELEVANCE (Lower weight)
            content_words = set(w for w in content_lower.split() if len(w) > 3 and w not in stop_words)
            common_content = query_words & content_words
            score += len(common_content) * 0.5
            
            # Only add if score is meaningfully positive
            if score > 2.0:
                doc_copy = doc.copy()
                doc_copy['similarity_score'] = float(score)
                doc_copy['search_type'] = 'keyword'
                results.append(doc_copy)
        
        # Sort by score and return top K
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_k]
    
    def generate_legal_explanation(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Generate AI-powered legal explanation using rule-based reasoning
        
        Args:
            query: User's legal question
            context_docs: Retrieved context documents
            
        Returns:
            Generated explanation text
        """
        if not context_docs:
            return self._get_fallback_advice(query)
        
        # Extract information from context
        sections = [doc.get('section', '') for doc in context_docs if doc.get('section')]
        punishments = [doc.get('punishment', '') for doc in context_docs if doc.get('punishment')]
        summaries = [doc.get('simplified_summary', '') for doc in context_docs if doc.get('simplified_summary')]
        
        # Build explanation
        explanation_parts = []
        
        # Add context-based introduction
        if sections:
            unique_sections = list(dict.fromkeys(sections[:2]))  # Remove duplicates, keep order
            explanation_parts.append(f"Based on {', '.join(unique_sections)}, here's what you need to know:")
        else:
            explanation_parts.append("Based on relevant legal provisions:")
        
        # Add specific guidance based on query type
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['self defense', 'self-defense', 'kill', 'defend']):
            explanation_parts.extend([
                "• Self-defense is a fundamental right under Indian law (Sections 96-106 IPC)",
                "• You can use reasonable force to protect yourself, others, or property",
                "• The force must be proportional to the threat faced",
                "• You cannot claim self-defense if you initiated the confrontation",
                "• In cases of grave and imminent danger, causing death may be justified",
                "• Always report the incident to police immediately after the event",
                "• Seek legal counsel to understand your specific situation"
            ])
        elif summaries:
            explanation_parts.append(f"• {summaries[0]}")
        
        # Add punishment information
        if punishments:
            explanation_parts.append(f"• Punishment: {punishments[0]}")
        
        return "\n".join(explanation_parts)
    
    def _get_fallback_advice(self, query: str) -> str:
        """
        Provide general legal advice when no specific context is found
        
        Args:
            query: User's legal question
            
        Returns:
            General legal guidance text
        """
        query_lower = query.lower()
        
        advice_map = {
            ('self defense', 'self-defense', 'kill', 'defend'): [
                "• Self-defense is a fundamental right under Indian law (Sections 96-106 IPC)",
                "• You can use reasonable force to protect yourself, others, or property",
                "• The force must be proportional to the threat faced",
                "• You cannot claim self-defense if you were the aggressor",
                "• In extreme cases, causing death in self-defense may be justified",
                "• Always report the incident to police immediately",
                "• Consult a lawyer for your specific situation"
            ],
            ('theft', 'steal', 'stolen'): [
                "• Theft is defined under Section 378 IPC",
                "• Involves taking movable property without consent with dishonest intention",
                "• Punishment: Up to 3 years imprisonment and/or fine",
                "• Theft becomes robbery if force or threat is used (Section 390)",
                "• Report to police immediately with evidence",
                "• Keep receipts and proof of ownership"
            ],
            ('fraud', 'cheat', 'deceive'): [
                "• Fraud/Cheating is covered under Section 420 IPC",
                "• Involves deceiving someone to cause wrongful gain or loss",
                "• Punishment: Up to 7 years imprisonment and fine",
                "• Gather all evidence of the fraudulent act",
                "• File complaint with police or cyber cell (for online fraud)",
                "• Consider civil remedies for monetary recovery"
            ],
            ('contract', 'agreement'): [
                "• Contracts are governed by the Indian Contract Act, 1872",
                "• A valid contract requires offer, acceptance, and consideration",
                "• Minors (under 18) cannot enter into valid contracts",
                "• Breach of contract may lead to civil remedies",
                "• Keep written records of all agreements",
                "• Consult a lawyer before signing important contracts"
            ]
        }
        
        for keywords, advice in advice_map.items():
            if any(kw in query_lower for kw in keywords):
                return "\n".join(advice)
        
        # Default advice
        return "\n".join([
            "• This appears to be a legal question requiring specific analysis",
            "• Indian law provides comprehensive coverage for most situations",
            "• Consult a qualified lawyer for advice tailored to your case",
            "• Keep all relevant documents and evidence",
            "• Be aware of your legal rights and obligations",
            "• Consider alternative dispute resolution methods when appropriate"
        ])
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Main query processing pipeline with intelligent search and response generation
        
        Args:
            query: User's legal question
            
        Returns:
            Dictionary containing response type, confidence, explanation, and source documents
        """
        if not query or not query.strip():
            return self._empty_response(query)
        
        # Step 1: Try keyword search FIRST (it's more accurate with our improvements)
        keyword_results = self.keyword_search(query, top_k=self.DEFAULT_TOP_K)
        
        # Step 2: Try semantic search as backup
        semantic_results = self.semantic_search(query, top_k=self.DEFAULT_TOP_K)
        
        # Step 3: Choose the best results
        keyword_score = max([doc.get('similarity_score', 0) for doc in keyword_results]) if keyword_results else 0
        semantic_score = max([doc.get('similarity_score', 0) for doc in semantic_results]) if semantic_results else 0
        
        # Prefer keyword search if it has reasonable results
        if keyword_score >= 10.0:  # Keyword search has meaningful results
            search_results = keyword_results
            search_type = 'keyword'
            max_confidence = keyword_score
        elif semantic_score > 0.3:  # Semantic search has decent results
            search_results = semantic_results
            search_type = 'semantic'
            max_confidence = semantic_score
        elif keyword_results:  # Keyword search has something
            search_results = keyword_results
            search_type = 'keyword'
            max_confidence = keyword_score
        elif semantic_results:  # Semantic search has something
            search_results = semantic_results
            search_type = 'semantic'
            max_confidence = semantic_score
        else:
            return self._format_generative_response(query, [])
        
        # Step 4: Determine response type based on confidence
        if search_type == 'semantic':
            threshold = self.SEMANTIC_CONFIDENCE_THRESHOLD
        else:  # keyword search
            threshold = self.KEYWORD_CONFIDENCE_THRESHOLD
        
        # Step 5: Format response
        if max_confidence >= threshold:
            return self._format_retrieved_response(query, search_results)
        else:
            return self._format_generative_response(query, search_results)
    
    def _format_retrieved_response(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format high-confidence retrieved response"""
        return {
            "type": "retrieved",
            "confidence": "high",
            "query": query,
            "sections": [doc.get('section', 'N/A') for doc in results if doc.get('section')],
            "explanation": results[0].get('simplified_summary') or results[0].get('content', '')[:300] + "...",
            "example": results[0].get('real_life_example', ''),
            "punishment": results[0].get('punishment', ''),
            "source": results[0].get('source', 'Indian Legal Database'),
            "documents": results,
            "max_score": max([doc.get('similarity_score', 0) for doc in results])
        }
    
    def _format_generative_response(self, query: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format AI-generated response"""
        explanation = self.generate_legal_explanation(query, context_docs)
        
        return {
            "type": "generated",
            "confidence": "ai-inferred",
            "query": query,
            "sections": [doc.get('section', 'N/A') for doc in context_docs if doc.get('section')] or ["No direct match found"],
            "explanation": explanation,
            "example": "",
            "punishment": "",
            "source": "AI-generated based on general legal principles",
            "documents": context_docs,
            "max_score": max([doc.get('similarity_score', 0) for doc in context_docs]) if context_docs else 0
        }
    
    def _empty_response(self, query: str) -> Dict[str, Any]:
        """Format response for empty query"""
        return {
            "type": "error",
            "confidence": "none",
            "query": query,
            "sections": [],
            "explanation": "Please enter a valid legal question.",
            "example": "",
            "punishment": "",
            "source": "",
            "documents": [],
            "max_score": 0
        }


def main():
    """Main Streamlit application"""
    
    # Header
    st.title("⚖️ A-Qlegal 3.0 - Legal AI Assistant")
    st.markdown("**Generative and Retrieval-Augmented AI for Indian Law**")
    st.markdown("*Trained on 8,369+ legal documents including IPC, CrPC, and Constitution*")
    
    # Initialize session state
    if 'selected_query' not in st.session_state:
        st.session_state.selected_query = ""
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
    # Initialize A-Qlegal system
    aqlegal = AQlegalV3()
    
    # Load models and data
    with st.spinner("🔄 Loading AI models and legal database..."):
        models_loaded = aqlegal.load_models()
        if not models_loaded:
            st.error("❌ Failed to load models. Please check the setup.")
            st.stop()
        
        aqlegal.legal_data = aqlegal.load_legal_data()
        
        if not aqlegal.legal_data:
            st.error("❌ No legal data loaded. Please check the data files.")
            st.stop()
    
    st.success(f"✅ System ready: {len(aqlegal.legal_data):,} legal documents loaded")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Display current thresholds (read-only for production)
        st.metric("Semantic Threshold", f"{aqlegal.SEMANTIC_CONFIDENCE_THRESHOLD:.2f}")
        st.metric("Keyword Threshold", f"{aqlegal.KEYWORD_CONFIDENCE_THRESHOLD:.1f}")
        
        st.markdown("---")
        st.header("📊 Database Statistics")
        st.metric("Total Documents", f"{len(aqlegal.legal_data):,}")
        
        categories = [doc.get('category', 'unknown') for doc in aqlegal.legal_data]
        unique_categories = len(set(categories))
        st.metric("Legal Categories", unique_categories)
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("""
        **A-Qlegal 3.0** is an advanced legal AI assistant powered by:
        - TF-IDF semantic search
        - Enhanced keyword matching
        - Rule-based explanation generation
        - 8,369+ legal documents from Indian law
        
        **Disclaimer**: This is an AI assistant for informational purposes only. 
        Always consult a qualified lawyer for legal advice.
        """)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask a Legal Question")
        
        # Query input
        default_query = st.session_state.selected_query if st.session_state.selected_query else ""
        
        query = st.text_area(
            "Enter your legal question:",
            value=default_query,
            placeholder="e.g., Can I use force in self-defense? What is the punishment for theft?",
            height=120,
            key="query_input",
            help="Type your legal question in plain English"
        )
        
        # Analyze button
        if st.button("🔍 Analyze Legal Question", type="primary", use_container_width=True):
            if query and query.strip():
                # Clear selected query
                st.session_state.selected_query = ""
                
                # Add to history
                if query not in st.session_state.query_history:
                    st.session_state.query_history.insert(0, query)
                    st.session_state.query_history = st.session_state.query_history[:5]  # Keep last 5
                
                with st.spinner("🔍 Analyzing your legal question..."):
                    response = aqlegal.process_query(query)
                
                # Display results
                st.markdown("---")
                
                # Display the top matching section prominently
                if response['sections'] and response['sections'][0] != 'N/A' and response['sections'][0] != 'No direct match found':
                    st.markdown(f"# {response['sections'][0]}")
                else:
                    st.markdown(f"### ⚖️ Legal Analysis")
                
                # Confidence indicator
                if response['type'] == 'retrieved':
                    st.success(f"✅ **High Confidence Match** (Score: {response['max_score']:.2f})")
                elif response['type'] == 'generated':
                    st.info(f"ℹ️ **AI-Inferred Response** (Best match score: {response['max_score']:.2f})")
                
                # Simplified Summary
                st.markdown("### 📝 Simplified Summary")
                # Get the simplified summary from the document if available
                if response['documents'] and response['documents'][0].get('simplified_summary'):
                    st.write(response['documents'][0]['simplified_summary'])
                else:
                    st.write(response['explanation'])
                
                # Real-Life Example
                if response.get('example') or (response['documents'] and response['documents'][0].get('real_life_example')):
                    st.markdown("### 🏠 Real-Life Example")
                    example_text = response.get('example') or response['documents'][0].get('real_life_example')
                    st.write(example_text)
                
                # Punishment/Legal Consequences
                st.markdown("### ⚖️ Punishment")
                if response.get('punishment'):
                    st.write(f"**{response['punishment']}**")
                elif response['documents'] and response['documents'][0].get('punishment'):
                    st.write(f"**{response['documents'][0]['punishment']}**")
                else:
                    st.write("*Refer to specific legal provisions for punishment details*")
                
                # Keywords
                if response['documents'] and response['documents'][0].get('keywords'):
                    st.markdown("### 🏷️ Keywords")
                    keywords = response['documents'][0]['keywords']
                    st.write(", ".join(keywords))
                elif response['sections']:
                    st.markdown("### 🏷️ Keywords")
                    # Generate keywords from the query and sections
                    keywords_list = []
                    query_words = [w for w in response['query'].lower().split() if len(w) > 3]
                    keywords_list.extend(query_words[:3])
                    if response['sections'][0] not in ['N/A', 'No direct match found']:
                        keywords_list.append(response['sections'][0])
                    st.write(", ".join(keywords_list))
                
                # Disclaimer
                st.markdown("---")
                st.caption("⚠️ **Legal Disclaimer:** This is an AI-generated explanation for informational purposes only. For personalized legal advice, please consult a qualified lawyer.")
                
                # Source documents
                if response.get('documents'):
                    with st.expander(f"📚 View Source Documents ({len(response['documents'])} found)", expanded=False):
                        for i, doc in enumerate(response['documents'], 1):
                            st.markdown(f"**{i}. {doc.get('title', 'Unknown Document')}**")
                            st.markdown(f"- **Section:** {doc.get('section', 'N/A')}")
                            st.markdown(f"- **Category:** {doc.get('category', 'N/A').replace('_', ' ').title()}")
                            st.markdown(f"- **Relevance Score:** {doc.get('similarity_score', 0):.3f}")
                            st.markdown(f"- **Search Type:** {doc.get('search_type', 'N/A').title()}")
                            
                            if doc.get('content'):
                                st.markdown(f"- **Content Preview:** {doc['content'][:200]}...")
                            
                            st.markdown("---")
            else:
                st.warning("⚠️ Please enter a legal question to analyze.")
    
    with col2:
        st.header("🎯 Quick Examples")
        
        example_queries = [
            "Can I kill someone in self-defense?",
            "What is the punishment for theft?",
            "Can a minor sign a contract?",
            "What are my rights if arrested?",
            "How do I file for divorce?",
            "What is Section 420 IPC?"
        ]
        
        st.markdown("*Click any example to try it:*")
        for i, example in enumerate(example_queries):
            if st.button(example, use_container_width=True, key=f"ex_{i}"):
                st.session_state.selected_query = example
                st.rerun()
        
        # Query history
        if st.session_state.query_history:
            st.markdown("---")
            st.header("🕐 Recent Queries")
            for i, hist_query in enumerate(st.session_state.query_history):
                if st.button(f"↻ {hist_query[:40]}...", use_container_width=True, key=f"hist_{i}"):
                    st.session_state.selected_query = hist_query
                    st.rerun()
        
        # System status
        st.markdown("---")
        st.header("📊 System Status")
        st.metric("AI Status", "🟢 Online")
        st.metric("Search Modes", "Semantic + Keyword")
        st.metric("Data Coverage", "Indian Law")


if __name__ == "__main__":
    main()

