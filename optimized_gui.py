"""
Optimized Legal QA System GUI with Model Selection and Fast Loading
"""
import streamlit as st
import sys
import os
from pathlib import Path
import time
import json
import pickle

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Set page config
st.set_page_config(
    page_title="⚡ Fast Legal QA System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .model-status {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 0.3rem;
        background-color: white;
    }
    .status-good {
        color: #28a745;
        font-weight: bold;
    }
    .status-bad {
        color: #dc3545;
        font-weight: bold;
    }
    .answer-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #28a745;
        background-color: #d4edda;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-weight: bold;
    }
    .confidence-medium {
        color: #856404;
        background-color: #fff3cd;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-weight: bold;
    }
    .confidence-low {
        color: #721c24;
        background-color: #f8d7da;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-weight: bold;
    }
    .fast-loading {
        color: #17a2b8;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_optimized_system():
    """Load the pre-trained optimized system with caching"""
    try:
        from src.main import LegalQASystem
        
        # Set Gemini API key
        os.environ['GEMINI_API_KEY'] = "AIzaSyDLOMncFan_QBHFz0BDYw_gWtEVNTJ3NyE"
        
        # Optimized configuration for speed
        config = {
            'data_dir': 'data/',
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'extractive_model': 'bert-base-uncased',
            'generative_model': 'gemini-1.5-flash',
            'gemini_api_key': os.environ['GEMINI_API_KEY'],
            'question_categories': ['fact', 'procedure', 'interpretive'],
            'bm25_weight': 0.3,
            'dense_weight': 0.7,
            'bayesian_weight': 0.3,
            'similarity_weight': 0.4,
            'confidence_weight': 0.3,
            'cache_file': 'answer_cache.pkl',
            'similarity_threshold': 0.8,
            'confidence_threshold': 0.7,
            'default_top_k': 15,  # Reduced for speed
            'max_context_length': 2000,  # Reduced for speed
            'batch_size': 32,  # Increased for speed
            'max_epochs': 2,  # Reduced for speed
            'learning_rate': 2e-5  # Increased for faster convergence
        }
        
        system = LegalQASystem(config)
        
        # Try to load saved system first
        try:
            if Path("models/optimized_legal_qa").exists():
                system.load_system("models/optimized_legal_qa")
                st.success("✅ Loaded pre-trained system (fast!)")
                return system
        except Exception as e:
            st.warning(f"⚠️ Could not load saved system: {e}")
        
        # Load datasets if no saved system
        dataset_paths = {
            'constitution': 'data/constitution_qa.json',
            'crpc': 'data/crpc_qa.json', 
            'ipc': 'data/ipc_qa.json'
        }
        
        system.initialize_system(dataset_paths)
        st.success("✅ System initialized with fresh data")
        return system
        
    except Exception as e:
        st.error(f"❌ Failed to initialize system: {e}")
        return None

def get_confidence_class(confidence):
    """Get CSS class for confidence level"""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.5:
        return "confidence-medium"
    else:
        return "confidence-low"

def get_confidence_text(confidence):
    """Get confidence text"""
    if confidence >= 0.8:
        return "HIGH CONFIDENCE"
    elif confidence >= 0.5:
        return "MEDIUM CONFIDENCE"
    else:
        return "LOW CONFIDENCE"

def display_model_status(status):
    """Display model status in sidebar"""
    st.sidebar.markdown("### 🤖 Model Status")
    
    # Classifier
    classifier_trained = status['classifier_info']['is_trained']
    classifier_status = "✅ Trained" if classifier_trained else "❌ Not Trained"
    classifier_color = "status-good" if classifier_trained else "status-bad"
    
    st.sidebar.markdown(f"""
    <div class="model-status">
        <span>📋 Classifier</span>
        <span class="{classifier_color}">{classifier_status}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Extractive Model
    extractive_trained = status['extractive_model_info']['is_trained']
    extractive_status = "✅ Trained" if extractive_trained else "❌ Not Trained"
    extractive_color = "status-good" if extractive_trained else "status-bad"
    
    st.sidebar.markdown(f"""
    <div class="model-status">
        <span>📝 Extractive</span>
        <span class="{extractive_color}">{extractive_status}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Generative Model
    generative_info = status['generative_model_info']
    if generative_info.get('use_gemini', False):
        generative_status = f"✅ Gemini AI ({generative_info.get('model_name', 'N/A')})"
        generative_color = "status-good"
    else:
        generative_trained = generative_info.get('is_trained', False)
        generative_status = "✅ Trained" if generative_trained else "❌ Not Trained"
        generative_color = "status-good" if generative_trained else "status-bad"
    
    st.sidebar.markdown(f"""
    <div class="model-status">
        <span>🤖 Generative</span>
        <span class="{generative_color}">{generative_status}</span>
    </div>
    """, unsafe_allow_html=True)

def display_system_stats(status):
    """Display system statistics"""
    st.sidebar.markdown("### 📊 System Statistics")
    
    if status['dataset_stats']:
        st.sidebar.metric("📚 Total Q&A Pairs", status['dataset_stats']['total_qa_pairs'])
        
        if 'category_distribution' in status['dataset_stats']:
            st.sidebar.markdown("#### Category Distribution")
            for category, count in status['dataset_stats']['category_distribution'].items():
                st.sidebar.metric(f"📋 {category.title()}", count)
    
    st.sidebar.metric("💾 Cached Answers", status['cache_stats']['total_entries'])

def main():
    """Main GUI function"""
    # Header
    st.markdown('<h1 class="main-header">⚡ Fast Legal QA System</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by Gemini AI and Advanced Legal Document Processing")
    
    # Initialize system with progress
    with st.spinner("⚡ Loading optimized Legal QA System..."):
        system = load_optimized_system()
    
    if system is None:
        st.error("❌ Failed to initialize system. Please check the logs.")
        return
    
    # Get system status
    status = system.get_system_status()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ System Control")
        
        # Model Selection
        st.markdown("### 🎯 Model Selection")
        model_choice = st.selectbox(
            "Choose Answering Method:",
            ["Both (Hybrid)", "Extractive Only", "Generative Only (Gemini)"],
            help="Select how you want answers to be generated"
        )
        
        # Configuration
        st.markdown("### 🔧 Configuration")
        top_k = st.slider("Documents to Retrieve", 5, 25, 15)
        max_context = st.slider("Max Context Length", 1000, 3000, 2000)
        
        # Update config if changed
        if 'top_k' not in st.session_state or st.session_state.top_k != top_k:
            st.session_state.top_k = top_k
        if 'max_context' not in st.session_state or st.session_state.max_context != max_context:
            st.session_state.max_context = max_context
        
        # Model status
        display_model_status(status)
        
        # System stats
        display_system_stats(status)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 💬 Ask Your Legal Question")
        
        # Question input
        question = st.text_area(
            "Enter your legal question:",
            placeholder="e.g., What is the punishment for theft under IPC?",
            height=100
        )
        
        # Ask button
        if st.button("🔍 Ask Question", type="primary", use_container_width=True):
            if question.strip():
                with st.spinner("🔍 Processing your question..."):
                    try:
                        # Process question based on model choice
                        if model_choice == "Extractive Only":
                            # Force extractive only
                            result = system.ask_question(question, top_k, use_extractive_only=True)
                        elif model_choice == "Generative Only (Gemini)":
                            # Force generative only
                            result = system.ask_question(question, top_k, use_generative_only=True)
                        else:
                            # Use both (default)
                            result = system.ask_question(question, top_k)
                        
                        # Display results
                        st.markdown("## 💡 Answer")
                        
                        # Answer box
                        st.markdown(f"""
                        <div class="answer-box">
                            <h4>Answer:</h4>
                            <p>{result['answer']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Metrics
                        col_a, col_b, col_c, col_d = st.columns(4)
                        
                        with col_a:
                            confidence_class = get_confidence_class(result['confidence'])
                            confidence_text = get_confidence_text(result['confidence'])
                            st.markdown(f"""
                            <div class="{confidence_class}">
                                📊 {confidence_text}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_b:
                            st.metric("Confidence Score", f"{result['confidence']:.3f}")
                        
                        with col_c:
                            st.metric("Source", result['source'])
                        
                        with col_d:
                            st.metric("Documents Used", result['retrieved_documents'])
                        
                        # Additional info
                        st.markdown("### 📋 Additional Information")
                        
                        col_e, col_f = st.columns(2)
                        
                        with col_e:
                            st.markdown(f"**Category:** {result['classification']['predicted_category']}")
                            st.markdown(f"**Classification Confidence:** {result['classification']['confidence']:.3f}")
                        
                        with col_f:
                            st.markdown(f"**Model Used:** {model_choice}")
                            if 'explanation' in result and result['explanation']:
                                st.markdown(f"**Explanation:** {result['explanation']}")
                        
                        # Store result for history
                        if 'question_history' not in st.session_state:
                            st.session_state.question_history = []
                        
                        st.session_state.question_history.insert(0, {
                            'question': question,
                            'answer': result['answer'],
                            'confidence': result['confidence'],
                            'source': result['source'],
                            'model_used': model_choice,
                            'timestamp': time.time()
                        })
                        
                        # Limit history to 10 items
                        if len(st.session_state.question_history) > 10:
                            st.session_state.question_history = st.session_state.question_history[:10]
                        
                    except Exception as e:
                        st.error(f"❌ Error processing question: {e}")
            else:
                st.warning("⚠️ Please enter a question.")
    
    with col2:
        st.markdown("## 📚 Quick Questions")
        
        # Sample questions
        sample_questions = [
            "What is the punishment for theft under IPC?",
            "What are the fundamental rights in the Constitution?",
            "How to file an FIR under CrPC?",
            "What is anticipatory bail under Section 438?",
            "What are the rights of an accused person?",
            "What is the procedure for criminal trial?",
            "What is the punishment for murder under IPC?",
            "What is the right to freedom of speech under Article 19?",
            "What is the procedure for investigation under CrPC?",
            "What is the punishment for rape under IPC?",
            "What is the right to life and personal liberty under Article 21?",
            "What is the procedure for bail under CrPC?",
            "What is the punishment for cheating under IPC?",
            "What is the right to constitutional remedies under Article 32?",
            "What is the procedure for arrest under CrPC?",
            "What is the punishment for defamation under IPC?",
            "What are Directive Principles of State Policy?",
            "What is the procedure for appeal under CrPC?",
            "What is the punishment for criminal breach of trust?",
            "What is the right to freedom of religion under Article 25?"
        ]
        
        for i, sample_q in enumerate(sample_questions):
            if st.button(f"❓ {sample_q[:50]}...", key=f"sample_{i}", use_container_width=True):
                st.session_state.current_question = sample_q
                st.rerun()
        
        # Set question from sample
        if 'current_question' in st.session_state:
            st.session_state.current_question = st.session_state.current_question
            del st.session_state.current_question
    
    # Question history
    if 'question_history' in st.session_state and st.session_state.question_history:
        st.markdown("## 📜 Recent Questions")
        
        for i, item in enumerate(st.session_state.question_history[:5]):
            with st.expander(f"❓ {item['question'][:60]}..."):
                st.markdown(f"**Answer:** {item['answer'][:200]}...")
                st.markdown(f"**Confidence:** {item['confidence']:.3f} | **Source:** {item['source']} | **Model:** {item['model_used']}")
    
    # Footer
    st.markdown("---")
    st.markdown("### ⚡ Fast Legal QA System Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 Model Selection**
        - Choose Extractive Only
        - Choose Generative Only (Gemini)
        - Use Both (Hybrid)
        """)
    
    with col2:
        st.markdown("""
        **⚡ Fast Loading**
        - Pre-trained models
        - Optimized configuration
        - Cached responses
        """)
    
    with col3:
        st.markdown("""
        **📚 Comprehensive Data**
        - 3,042+ legal Q&A pairs
        - Constitution, CrPC, IPC
        - Real-time processing
        """)

if __name__ == "__main__":
    main()
