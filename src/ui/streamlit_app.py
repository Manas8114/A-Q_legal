"""
Streamlit UI for Legal QA System
"""
import streamlit as st
import requests
import json
from typing import Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Legal QA System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .question-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .answer-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def ask_question(question: str, top_k: int = 5) -> Dict[str, Any]:
    """Ask a question to the API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={"question": question, "top_k": top_k},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}

def get_system_status() -> Dict[str, Any]:
    """Get system status from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/cache/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}

def get_models_info() -> Dict[str, Any]:
    """Get models information"""
    try:
        response = requests.get(f"{API_BASE_URL}/models/info", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}

def main():
    # Header
    st.markdown('<h1 class="main-header">⚖️ Legal QA System</h1>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("❌ API is not running. Please start the API server first.")
        st.code("python -m src.api.main")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Controls")
        
        # System status
        if st.button("🔄 Refresh Status"):
            st.rerun()
        
        # Get system status
        status = get_system_status()
        models_info = get_models_info()
        
        if "error" not in status:
            st.success("✅ System Online")
            
            # Display basic stats
            if status.get("dataset_stats"):
                st.metric("Total Q&A Pairs", status["dataset_stats"]["total_qa_pairs"])
                st.metric("Categories", len(status["dataset_stats"]["categories"]))
            
            if status.get("cache_stats"):
                st.metric("Cached Answers", status["cache_stats"]["total_entries"])
            
            # Model status
            if "error" not in models_info:
                st.subheader("🤖 Model Status")
                classifier_trained = models_info.get("classifier", {}).get("is_trained", False)
                extractive_trained = models_info.get("extractive_model", {}).get("is_trained", False)
                generative_trained = models_info.get("generative_model", {}).get("is_trained", False)
                
                st.write(f"**Classifier:** {'✅ Trained' if classifier_trained else '❌ Not Trained'}")
                st.write(f"**Extractive:** {'✅ Trained' if extractive_trained else '❌ Not Trained'}")
                st.write(f"**Generative:** {'✅ Trained' if generative_trained else '❌ Not Trained'}")
                
                if not extractive_trained and not generative_trained:
                    st.info("💡 Using fallback answer generation")
        else:
            st.error("❌ System Offline")
            st.error(status["error"])
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🤔 Ask Question", "📊 Analytics", "🔍 Cache Explorer", "⚙️ Settings"])
    
    with tab1:
        st.header("Ask a Legal Question")
        
        # Question input
        question = st.text_area(
            "Enter your legal question:",
            placeholder="e.g., What is the punishment for theft under IPC?",
            height=100
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            top_k = st.slider("Number of documents to retrieve", 1, 10, 5)
        with col2:
            ask_button = st.button("🔍 Ask Question", type="primary")
        
        if ask_button and question:
            with st.spinner("Processing your question..."):
                result = ask_question(question, top_k)
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                # Display question
                st.markdown('<div class="question-box">', unsafe_allow_html=True)
                st.markdown(f"**Question:** {result['question']}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display answer
                st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                st.markdown(f"**Answer:** {result['answer']}")
                
                # Confidence indicator
                confidence = result['confidence']
                if confidence >= 0.8:
                    confidence_class = "confidence-high"
                    confidence_text = "High Confidence"
                elif confidence >= 0.5:
                    confidence_class = "confidence-medium"
                    confidence_text = "Medium Confidence"
                else:
                    confidence_class = "confidence-low"
                    confidence_text = "Low Confidence"
                
                st.markdown(f'**Confidence:** <span class="{confidence_class}">{confidence_text} ({confidence:.2f})</span>', unsafe_allow_html=True)
                st.markdown(f"**Source:** {result['source']}")
                st.markdown(f"**Documents Retrieved:** {result['retrieved_documents']}")
                
                # Show if using fallback mode
                if 'fallback_mode' in result and result['fallback_mode']:
                    st.info("💡 Answer generated using fallback logic (models not fully trained)")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Classification
                if 'classification' in result:
                    classification = result['classification']
                    st.subheader("📋 Question Classification")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Category", classification['predicted_category'])
                    with col2:
                        st.metric("Confidence", f"{classification['confidence']:.2f}")
                    with col3:
                        st.metric("Source", result['source'])
                
                # Explanation
                if 'explanation' in result:
                    with st.expander("🔍 Answer Explanation"):
                        explanation = result['explanation']
                        st.json(explanation)
                
                # Feedback
                st.subheader("💬 Feedback")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Helpful"):
                        # Submit positive feedback
                        feedback_data = {
                            "question": question,
                            "answer": result['answer'],
                            "is_helpful": True
                        }
                        # In a real implementation, submit this feedback
                        st.success("Thank you for your feedback!")
                
                with col2:
                    if st.button("👎 Not Helpful"):
                        # Submit negative feedback
                        feedback_data = {
                            "question": question,
                            "answer": result['answer'],
                            "is_helpful": False
                        }
                        # In a real implementation, submit this feedback
                        st.error("Thank you for your feedback. We'll improve!")
    
    with tab2:
        st.header("📊 System Analytics")
        
        # Get system status
        status = get_system_status()
        if "error" in status:
            st.error(f"Error loading analytics: {status['error']}")
        else:
            # Dataset statistics
            if status.get("dataset_stats"):
                st.subheader("📚 Dataset Statistics")
                dataset_stats = status["dataset_stats"]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Q&A Pairs", dataset_stats["total_qa_pairs"])
                with col2:
                    st.metric("Categories", len(dataset_stats["categories"]))
                with col3:
                    st.metric("Avg Question Length", f"{dataset_stats['avg_question_length']:.1f}")
                with col4:
                    st.metric("Avg Answer Length", f"{dataset_stats['avg_answer_length']:.1f}")
                
                # Category distribution
                if dataset_stats["categories"]:
                    st.subheader("📊 Category Distribution")
                    categories_df = pd.DataFrame(
                        list(dataset_stats["categories"].items()),
                        columns=["Category", "Count"]
                    )
                    fig = px.pie(categories_df, values="Count", names="Category", title="Q&A Pairs by Category")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Cache statistics
            if status.get("cache_stats"):
                st.subheader("💾 Cache Statistics")
                cache_stats = status["cache_stats"]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Cached Entries", cache_stats["total_entries"])
                with col2:
                    st.metric("Cache Size (MB)", cache_stats["cache_size_mb"])
                with col3:
                    st.metric("Avg Access Count", f"{cache_stats['avg_access_count']:.1f}")
    
    with tab3:
        st.header("🔍 Cache Explorer")
        
        # Search cache
        search_query = st.text_input("Search cached answers:", placeholder="Enter keywords...")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            search_button = st.button("🔍 Search")
        
        if search_button and search_query:
            with st.spinner("Searching cache..."):
                try:
                    response = requests.get(
                        f"{API_BASE_URL}/cache/search",
                        params={"query": search_query, "limit": 10},
                        timeout=10
                    )
                    if response.status_code == 200:
                        search_results = response.json()
                        st.success(f"Found {search_results['count']} cached answers")
                        
                        for i, result in enumerate(search_results['results']):
                            with st.expander(f"Cached Answer {i+1}"):
                                st.write(f"**Question:** {result['question']}")
                                st.write(f"**Answer:** {result['answer']}")
                                st.write(f"**Access Count:** {result['access_count']}")
                                if 'metadata' in result:
                                    st.write(f"**Metadata:** {result['metadata']}")
                    else:
                        st.error(f"Search failed: {response.status_code}")
                except Exception as e:
                    st.error(f"Search error: {str(e)}")
        
        # Cache management
        st.subheader("🗑️ Cache Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear Cache"):
                try:
                    response = requests.delete(f"{API_BASE_URL}/cache", timeout=10)
                    if response.status_code == 200:
                        st.success("Cache cleared successfully!")
                    else:
                        st.error(f"Failed to clear cache: {response.status_code}")
                except Exception as e:
                    st.error(f"Error clearing cache: {str(e)}")
        
        with col2:
            if st.button("Refresh Cache Stats"):
                st.rerun()
    
    with tab4:
        st.header("⚙️ System Settings")
        
        # Model information
        st.subheader("🤖 Model Information")
        try:
            response = requests.get(f"{API_BASE_URL}/models/info", timeout=10)
            if response.status_code == 200:
                models_info = response.json()
                
                # Classifier info
                st.write("**Question Classifier:**")
                st.write(f"- Trained: {models_info['classifier']['is_trained']}")
                st.write(f"- Categories: {', '.join(models_info['classifier']['categories'])}")
                
                # Extractive model info
                st.write("**Extractive Model:**")
                st.write(f"- Trained: {models_info['extractive_model']['is_trained']}")
                st.write(f"- Model: {models_info['extractive_model']['model_name']}")
                
                # Generative model info
                st.write("**Generative Model:**")
                st.write(f"- Trained: {models_info['generative_model']['is_trained']}")
                st.write(f"- Model: {models_info['generative_model']['model_name']}")
                
                # Retriever info
                st.write("**Retrieval System:**")
                st.write(f"- Fitted: {models_info['retriever']['is_fitted']}")
                st.write(f"- Documents: {models_info['retriever']['num_documents']}")
            else:
                st.error(f"Failed to get model info: {response.status_code}")
        except Exception as e:
            st.error(f"Error getting model info: {str(e)}")
        
        # System actions
        st.subheader("🔧 System Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save System"):
                try:
                    response = requests.post(f"{API_BASE_URL}/save", timeout=30)
                    if response.status_code == 200:
                        st.success("System saved successfully!")
                    else:
                        st.error(f"Failed to save system: {response.status_code}")
                except Exception as e:
                    st.error(f"Error saving system: {str(e)}")
        
        with col2:
            if st.button("📁 Load System"):
                try:
                    response = requests.post(f"{API_BASE_URL}/load", timeout=30)
                    if response.status_code == 200:
                        st.success("System loaded successfully!")
                    else:
                        st.error(f"Failed to load system: {response.status_code}")
                except Exception as e:
                    st.error(f"Error loading system: {str(e)}")

if __name__ == "__main__":
    main()