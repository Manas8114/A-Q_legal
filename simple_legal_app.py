#!/usr/bin/env python3
"""
A-Qlegal 2.0 - Simple Working Version
Basic legal Q&A without complex ML models
"""

import json
import streamlit as st
from pathlib import Path
import re

# Load sample data
def load_legal_data():
    try:
        with open("data/processed/sample_legal_data.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"legal_documents": []}

def search_legal_documents(query, data):
    """Simple keyword-based search"""
    query_lower = query.lower()
    results = []
    
    for doc in data["legal_documents"]:
        score = 0
        
        # Check in simplified summary
        if query_lower in doc["simplified_summary"].lower():
            score += 3
        
        # Check in keywords
        for keyword in doc["keywords"]:
            if query_lower in keyword.lower():
                score += 2
        
        # Check in legal text
        if query_lower in doc["legal_text"].lower():
            score += 1
        
        if score > 0:
            results.append((score, doc))
    
    # Sort by score
    results.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in results]

def main():
    st.set_page_config(
        page_title="A-Qlegal 2.0",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ A-Qlegal 2.0 - Legal AI Assistant")
    st.markdown("**Your AI-powered legal assistant for Indian law**")
    
    # Load data
    data = load_legal_data()
    
    if not data["legal_documents"]:
        st.error("No legal data found. Please run the setup first.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("📚 Available Legal Sections")
        for doc in data["legal_documents"]:
            st.write(f"• {doc['section']} - {doc['category']}")
    
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
                results = search_legal_documents(query, data)
                
                if results:
                    st.success(f"Found {len(results)} relevant legal section(s)")
                    
                    for i, doc in enumerate(results, 1):
                        with st.expander(f"{i}. {doc['section']} - {doc['category']}", expanded=i==1):
                            st.subheader("📖 Legal Text")
                            st.write(doc['legal_text'])
                            
                            st.subheader("📝 Simplified Summary")
                            st.write(doc['simplified_summary'])
                            
                            st.subheader("🏠 Real-Life Example")
                            st.write(doc['real_life_example'])
                            
                            st.subheader("⚖️ Punishment")
                            st.write(f"**{doc['punishment']}**")
                            
                            st.subheader("🏷️ Keywords")
                            st.write(", ".join(doc['keywords']))
                else:
                    st.warning("No relevant legal sections found. Try different keywords.")
            else:
                st.warning("Please enter a question.")
    
    with col2:
        st.header("📊 Quick Stats")
        st.metric("Total Legal Sections", len(data["legal_documents"]))
        
        categories = {}
        for doc in data["legal_documents"]:
            cat = doc["category"]
            categories[cat] = categories.get(cat, 0) + 1
        
        for cat, count in categories.items():
            st.metric(f"{cat} Cases", count)
        
        st.header("💡 Tips")
        st.info("""
        **How to use:**
        1. Ask questions in simple English
        2. Use keywords like 'fraud', 'murder', 'maintenance'
        3. Click on results to see full details
        4. Check the sidebar for available sections
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("**A-Qlegal 2.0** - Making Indian law accessible to everyone")

if __name__ == "__main__":
    main()
