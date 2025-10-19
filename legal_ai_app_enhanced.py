#!/usr/bin/env python3
"""
🏛️ A-Qlegal AI - Enhanced Version with ALL Features
Complete legal assistant with semantic search, history, export, filters, and more!
"""

# Disable TensorFlow (we only use PyTorch)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['USE_TF'] = 'NO'
os.environ['USE_TORCH'] = 'YES'

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering
import json
from pathlib import Path
import PyPDF2
import docx
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configure page
st.set_page_config(
    page_title="A-Qlegal AI Enhanced",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = Path(r"C:\Users\msgok\Desktop\A-Qlegal-main")
CLASSIFICATION_MODEL_DIR = BASE_DIR / "models" / "legal_model" / "legal_classification_model"
QA_MODEL_DIR = BASE_DIR / "models" / "legal_model" / "legal_qa_model"
CATEGORY_MAPPING_FILE = BASE_DIR / "models" / "legal_model" / "category_mapping.json"
DATASET_FILE = BASE_DIR / "data" / "expanded_legal_dataset.json"
FEEDBACK_FILE = BASE_DIR / "feedback.json"

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'theme' not in st.session_state:
    st.session_state.theme = "Light"
if 'filters' not in st.session_state:
    st.session_state.filters = {'min_confidence': 0.0, 'categories': []}

# Custom CSS with theme support
def apply_theme(theme):
    if theme == "Dark":
        st.markdown("""
        <style>
            .main { background-color: #1E1E1E; color: #FFFFFF; }
            .stTextInput > div > div > input { background-color: #2D2D2D; color: #FFFFFF; }
            .stTextArea > div > div > textarea { background-color: #2D2D2D; color: #FFFFFF; }
            .question-box { background-color: #2D2D2D; }
            .answer-box { background-color: #3D3D3D; }
            .document-box { background-color: #2A3F2A; }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            .main-header {
                font-size: 3rem;
                font-weight: bold;
                background: linear-gradient(90deg, #1f77b4, #ff7f0e);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-align: center;
                margin-bottom: 2rem;
            }
            .question-box {
                background-color: #f0f8ff;
                padding: 1.5rem;
                border-radius: 10px;
                margin: 1rem 0;
                border-left: 5px solid #1f77b4;
            }
            .answer-box {
                background-color: #fff8dc;
                padding: 1.5rem;
                border-radius: 10px;
                margin: 1rem 0;
                border-left: 5px solid #ff7f0e;
            }
            .document-box {
                background-color: #f0fff0;
                padding: 1.5rem;
                border-radius: 10px;
                margin: 1rem 0;
                border-left: 5px solid #28a745;
            }
            .confidence-high {
                color: #28a745;
                font-weight: bold;
                font-size: 1.2rem;
            }
            .confidence-medium {
                color: #ffc107;
                font-weight: bold;
                font-size: 1.2rem;
            }
            .confidence-low {
                color: #dc3545;
                font-weight: bold;
                font-size: 1.2rem;
            }
            .stButton>button {
                width: 100%;
                border-radius: 10px;
                height: 3rem;
                font-weight: bold;
            }
        </style>
        """, unsafe_allow_html=True)

apply_theme(st.session_state.theme)


@st.cache_resource
def load_sentence_model():
    """Load sentence transformer for semantic search"""
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def load_models():
    """Load the trained classification and QA models"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        with st.spinner("🔄 Loading AI models..."):
            # Load classification model
            classification_tokenizer = AutoTokenizer.from_pretrained(CLASSIFICATION_MODEL_DIR)
            classification_model = AutoModelForSequenceClassification.from_pretrained(CLASSIFICATION_MODEL_DIR)
            classification_model.to(device)
            classification_model.eval()
            
            # Load QA model
            qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_DIR)
            qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_DIR)
            qa_model.to(device)
            qa_model.eval()
            
            # Load category mapping
            with open(CATEGORY_MAPPING_FILE, 'r', encoding='utf-8') as f:
                category_mapping = json.load(f)
        
        return {
            'classification_tokenizer': classification_tokenizer,
            'classification_model': classification_model,
            'qa_tokenizer': qa_tokenizer,
            'qa_model': qa_model,
            'category_mapping': category_mapping,
            'device': device
        }
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None


@st.cache_data
def load_dataset():
    """Load the legal dataset"""
    try:
        with open(DATASET_FILE, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        return dataset
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return []


@st.cache_data
def compute_embeddings(_dataset):
    """Compute and cache document embeddings for semantic search"""
    with st.spinner("🔄 Computing document embeddings (first time only)..."):
        model = load_sentence_model()
        texts = [d.get('text', '')[:512] for d in _dataset]  # Limit length
        embeddings = model.encode(texts, show_progress_bar=False)
        return embeddings


def explain_confidence(confidence):
    """Explain confidence level"""
    if confidence > 0.8:
        return "✅ **High Confidence**: Strong keyword matches and clear context."
    elif confidence > 0.5:
        return "⚠️ **Medium Confidence**: Likely correct but some ambiguity. Check alternatives."
    else:
        return "❌ **Low Confidence**: Uncertain. Question may be vague or data unavailable."


def extract_citations(text):
    """Extract legal citations from text"""
    ipc_pattern = r'Section \d+[A-Z]?\s*(?:of\s*)?(?:the\s*)?IPC|IPC\s*Section\s*\d+[A-Z]?'
    article_pattern = r'Article\s*\d+[A-Z]?'
    case_pattern = r'\d{4}\s*\(\d+\)\s*\w+\s*\d+'
    
    citations = {
        'ipc_sections': list(set(re.findall(ipc_pattern, text, re.IGNORECASE))),
        'articles': list(set(re.findall(article_pattern, text, re.IGNORECASE))),
        'cases': list(set(re.findall(case_pattern, text)))
    }
    
    return citations


def classify_text(text, models):
    """Classify legal text"""
    try:
        tokenizer = models['classification_tokenizer']
        model = models['classification_model']
        device = models['device']
        
        inputs = tokenizer(text, max_length=256, padding='max_length', truncation=True, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_class].item()
        
        category_name = models['category_mapping']['id_to_category'][str(predicted_class)]
        
        top5_probs, top5_classes = torch.topk(probs[0], min(5, len(probs[0])))
        top5_predictions = [
            {'category': models['category_mapping']['id_to_category'][str(cls.item())], 'confidence': prob.item()}
            for prob, cls in zip(top5_probs, top5_classes)
        ]
        
        return {'category': category_name, 'confidence': confidence, 'top_predictions': top5_predictions}
    except Exception as e:
        st.error(f"❌ Classification error: {e}")
        return None


def clean_answer_text(text):
    """Clean answer text from Q&A formatting and artifacts"""
    if not text or not text.strip():
        return text
    
    # Remove question prefix patterns (from augmented data)
    patterns_to_remove = [
        r'^.*?\?\s*q\s*:',     # "question? q :"
        r'^.*?\?\s*q:',        # "question? q:"
        r'^.*?\?\s*a\s*:',     # "question? a :"
        r'^.*?\?\s*a:',        # "question? a:"
        r'^q\s*:.*?\?\s*a\s*:', # "q : ...? a :"
        r'^q:.*?\?\s*a:',      # "q: ...? a:"
    ]
    
    cleaned = text.strip()
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove leading "a :" or "a:" or "q :" or "q:"
    cleaned = re.sub(r'^[aq]\s*:\s*', '', cleaned, flags=re.IGNORECASE)
    
    # If first line looks like a question (ends with ?) and is short, remove it
    lines = cleaned.split('\n')
    if len(lines) > 1 and lines[0].strip().endswith('?') and len(lines[0]) < 150:
        cleaned = '\n'.join(lines[1:])
    
    # Clean up multiple spaces and newlines
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()
    
    # Capitalize first letter if not already
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    
    return cleaned


def answer_question(question, context, models):
    """Answer a question based on context"""
    try:
        tokenizer = models['qa_tokenizer']
        model = models['qa_model']
        device = models['device']
        
        # Clean context from Q&A formatting
        context_clean = clean_answer_text(context)
        if not context_clean:
            context_clean = context
        
        inputs = tokenizer(question, context_clean, max_length=256, padding='max_length', truncation=True, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
            
            start_idx = torch.argmax(start_scores)
            end_idx = torch.argmax(end_scores)
            
            answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
            answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
            
            # Clean the extracted answer
            answer = clean_answer_text(answer)
            
            start_prob = torch.softmax(start_scores, dim=-1)[0][start_idx].item()
            end_prob = torch.softmax(end_scores, dim=-1)[0][end_idx].item()
            confidence = (start_prob + end_prob) / 2
        
        return {'answer': answer if answer.strip() else "Unable to extract answer.", 'confidence': confidence}
    except Exception as e:
        st.error(f"❌ QA error: {e}")
        return None


def semantic_search(question, dataset, embeddings, top_k=10):
    """Semantic search using sentence transformers"""
    model = load_sentence_model()
    question_embedding = model.encode([question])
    
    similarities = cosine_similarity(question_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    return [(dataset[i], similarities[i]) for i in top_indices]


def search_relevant_context(question, dataset, embeddings, models, top_k=5):
    """Combined keyword + semantic search with better relevance"""
    question_lower = question.lower()
    
    # Enhanced legal keywords mapping with better coverage
    legal_keywords = {
        'kill': ['section 299', 'section 300', 'section 302', 'section 304', 'culpable homicide', 'murder', 'killing', 'death'],
        'murder': ['section 299', 'section 300', 'section 302', 'murder', 'culpable homicide'],
        'defend': ['section 96', 'section 97', 'section 98', 'section 99', 'section 100', 'section 101', 'section 102', 'section 103', 'section 104', 'section 105', 'section 106', 'private defence', 'self defense', 'self defence', 'right of private defence'],
        'defence': ['section 96', 'section 97', 'section 98', 'section 99', 'section 100', 'section 101', 'section 102', 'section 103', 'private defence', 'self defense', 'right of private defence'],
        'attack': ['section 96', 'section 97', 'section 100', 'section 351', 'section 352', 'assault', 'attack', 'private defence'],
        'myself': ['section 96', 'section 97', 'section 100', 'private defence', 'self defense', 'self defence'],
        'theft': ['section 378', 'section 379', 'section 380', 'theft', 'stealing', 'stolen property'],
        'steal': ['section 378', 'section 379', 'theft', 'stealing'],
        'robbery': ['section 390', 'section 391', 'section 392', 'robbery', 'dacoity'],
        'rape': ['section 375', 'section 376', 'rape', 'sexual assault'],
        'cheating': ['section 415', 'section 420', 'cheating', 'fraud', 'dishonestly'],
        'fraud': ['section 415', 'section 420', 'cheating', 'fraud', 'deception'],
        'assault': ['section 319', 'section 323', 'section 325', 'section 351', 'section 352', 'assault', 'hurt', 'injury'],
        'hurt': ['section 319', 'section 323', 'section 325', 'hurt', 'injury', 'grievous hurt'],
        'article': ['fundamental rights', 'constitution', 'constitutional law'],
        'constitution': ['fundamental rights', 'constitution', 'constitutional law', 'article'],
    }
    
    # Collect all relevant search terms
    search_terms = set()
    question_words = question_lower.split()
    
    for word in question_words:
        search_terms.add(word)
        if word in legal_keywords:
            search_terms.update(legal_keywords[word])
    
    # Get semantic search results
    semantic_results = semantic_search(question, dataset, embeddings, top_k=top_k*4)
    
    # Classify question
    classification = classify_text(question, models)
    
    # Score documents with multiple factors
    scored_docs = []
    seen_titles = set()  # Avoid duplicates
    
    for doc, sim_score in semantic_results:
        title = doc.get('title', '')
        
        # Skip duplicate titles (Q&A variations)
        if title in seen_titles:
            continue
        seen_titles.add(title)
        
        text = doc.get('text', '').lower()
        title_lower = title.lower()
        category = doc.get('category', '').lower()
        
        # Base score from semantic similarity (0-100)
        score = sim_score * 100
        
        # Keyword matching in text (high weight)
        keyword_matches = 0
        for term in search_terms:
            if len(term) > 3:  # Only check meaningful terms
                if term in text:
                    keyword_matches += 1
                    score += 10
                if term in title_lower:
                    keyword_matches += 1
                    score += 15
        
        # Boost if multiple keywords match
        if keyword_matches >= 3:
            score += 20
        
        # Category relevance
        question_has_criminal = any(word in question_lower for word in ['kill', 'murder', 'defend', 'attack', 'theft', 'assault'])
        if question_has_criminal and 'criminal' in category:
            score += 15
        
        question_has_constitutional = any(word in question_lower for word in ['article', 'constitution', 'fundamental', 'right'])
        if question_has_constitutional and 'constitutional' in category:
            score += 15
        
        # Penalize Q&A variations (we want original content)
        if category.endswith('_qa'):
            score -= 10
        
        # Only include documents with reasonable scores
        if score > 30:
            scored_docs.append((score, doc, keyword_matches))
    
    # Sort by score (descending) and keyword matches
    scored_docs.sort(key=lambda x: (x[0], x[2]), reverse=True)
    
    # Get top unique documents
    relevant_docs = [doc for score, doc, _ in scored_docs[:top_k*2]]
    
    return relevant_docs, classification


def save_feedback(question, answer, feedback_type, comment=""):
    """Save user feedback"""
    feedback = {
        'question': question,
        'answer': answer,
        'feedback': feedback_type,
        'comment': comment,
        'timestamp': str(datetime.now())
    }
    
    try:
        with open(FEEDBACK_FILE, 'a', encoding='utf-8') as f:
            json.dump(feedback, f)
            f.write('\n')
    except Exception as e:
        st.error(f"Error saving feedback: {e}")


def extract_text_from_pdf(file):
    """Extract text from PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages])
        return text
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")
        return None


def extract_text_from_docx(file):
    """Extract text from DOCX"""
    try:
        doc = docx.Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"❌ Error reading DOCX: {e}")
        return None


def main():
    # Header
    st.markdown('<h1 class="main-header">⚖️ A-Qlegal AI Enhanced</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #666;'>Your Advanced AI Legal Research Companion</p>", unsafe_allow_html=True)
    
    # Load models and dataset
    models = load_models()
    if not models:
        st.error("❌ Failed to load models. Please check the model files.")
        return
    
    dataset = load_dataset()
    embeddings = compute_embeddings(dataset)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ System Status")
        
        # Model status
        st.success(f"✅ Classification Model")
        st.success(f"✅ QA Model")
        st.success(f"✅ Semantic Search")
        st.info(f"💾 Device: {models['device'].upper()}")
        st.metric("📚 Documents", len(dataset))
        st.metric("🏷️ Categories", len(models['category_mapping']['categories']))
        
        st.markdown("---")
        
        # Theme toggle
        st.subheader("🎨 Theme")
        theme = st.radio("Select Theme", ["Light", "Dark"], index=0 if st.session_state.theme == "Light" else 1)
        if theme != st.session_state.theme:
            st.session_state.theme = theme
            apply_theme(theme)
            st.rerun()
        
        st.markdown("---")
        
        # Filters
        st.subheader("🔧 Filters")
        min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.1)
        st.session_state.filters['min_confidence'] = min_confidence
        
        st.markdown("---")
        
        # Sample questions by category
        st.subheader("💡 Sample Questions")
        sample_categories = {
            "Criminal Law": [
                "What is Section 302 IPC?",
                "Can I kill someone in self defence?",
                "What is the punishment for theft?",
            ],
            "Constitutional Law": [
                "What is Article 21?",
                "What are Fundamental Rights?",
                "What is Right to Equality?",
            ],
            "Civil Law": [
                "What is a civil suit?",
                "How to file a complaint?",
            ]
        }
        
        category = st.selectbox("Category", list(sample_categories.keys()))
        for q in sample_categories[category]:
            if st.button(q, key=f"sample_{q}"):
                st.session_state['sample_question'] = q
        
        st.markdown("---")
        
        # Question History
        if st.session_state.history:
            st.subheader("📜 Recent Questions")
            for i, item in enumerate(reversed(st.session_state.history[-5:])):
                q_short = item['question'][:40] + "..." if len(item['question']) > 40 else item['question']
                if st.button(q_short, key=f"history_{i}"):
                    st.session_state['sample_question'] = item['question']
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Ask Questions", "📄 Analyze Document", "📊 Analytics", "🤖 Chatbot"])
    
    with tab1:
        st.header("Ask Legal Questions")
        
        with st.expander("❓ How to ask good questions"):
            st.markdown("""
            **Good Questions:**
            - ✅ "What is the punishment for theft under IPC Section 378?"
            - ✅ "What does Article 21 of the Constitution state?"
            - ✅ "Can I defend myself if someone attacks me?"
            
            **Avoid:**
            - ❌ "Tell me about law" (too vague)
            - ❌ "Help me" (not specific)
            - ❌ Questions outside Indian law
            """)
        
        # Question input
        default_question = st.session_state.get('sample_question', '')
        question = st.text_area(
            "Enter your legal question:",
            value=default_question,
            placeholder="e.g., What is the punishment for murder under IPC?",
            height=100,
            help="Ask specific questions about Indian laws, constitution, or legal procedures",
            key="question_input"
        )
        
        if 'sample_question' in st.session_state:
            del st.session_state['sample_question']
        
        col1, col2 = st.columns([4, 1])
        with col2:
            ask_button = st.button("🔍 Ask", type="primary", key="ask_btn")
        
        if ask_button and question.strip():
            progress_bar = st.progress(0)
            
            # Step 1: Search
            with st.spinner("🔍 Searching legal database..."):
                progress_bar.progress(25)
                relevant_docs, classification = search_relevant_context(
                    question, dataset, embeddings, models
                )
            
            # Step 2: Answer
            with st.spinner("🤖 AI is analyzing documents..."):
                progress_bar.progress(50)
                
                best_answer = None
                best_confidence = 0
                best_source = None
                all_answers = []
                
                for doc in relevant_docs:
                    context = doc.get('text', '')
                    if context:
                        qa_result = answer_question(question, context, models)
                        if qa_result and qa_result['answer'] and qa_result['answer'] != "Unable to extract answer.":
                            all_answers.append({
                                'answer': qa_result['answer'],
                                'confidence': qa_result['confidence'],
                                'source': doc.get('title', 'Unknown'),
                                'category': doc.get('category', 'Unknown'),
                                'full_text': context
                            })
                            
                            if qa_result['confidence'] > best_confidence:
                                best_confidence = qa_result['confidence']
                                best_answer = qa_result['answer']
                                best_source = doc
            
            # Step 3: Display
            with st.spinner("📊 Preparing results..."):
                progress_bar.progress(75)
                
                # Save to history
                st.session_state.history.append({
                    'question': question,
                    'answer': best_answer or "No answer found",
                    'confidence': best_confidence,
                    'timestamp': datetime.now()
                })
                
                progress_bar.progress(100)
            
            # Display results
            st.markdown('<div class="question-box">', unsafe_allow_html=True)
            st.markdown(f"**📝 Question:** {question}")
            st.markdown(f"**🏷️ Category:** {classification['category']}")
            
            conf = classification['confidence']
            conf_class = "confidence-high" if conf >= 0.8 else "confidence-medium" if conf >= 0.5 else "confidence-low"
            conf_emoji = "🟢" if conf >= 0.8 else "🟡" if conf >= 0.5 else "🔴"
            st.markdown(f'{conf_emoji} <span class="{conf_class}">Confidence: {conf*100:.1f}%</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Confidence explanation
            st.info(explain_confidence(conf))
            
            if best_answer:
                st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                st.markdown(f"### 💡 Answer")
                st.markdown(best_answer)
                
                ans_conf = best_confidence
                ans_conf_class = "confidence-high" if ans_conf >= 0.8 else "confidence-medium" if ans_conf >= 0.5 else "confidence-low"
                ans_emoji = "🟢" if ans_conf >= 0.8 else "🟡" if ans_conf >= 0.5 else "🔴"
                st.markdown(f'{ans_emoji} <span class="{ans_conf_class}">Answer Confidence: {ans_conf*100:.1f}%</span>', unsafe_allow_html=True)
                st.markdown(f"**📚 Source:** {best_source.get('title', 'Unknown')}")
                st.markdown(f"**🏷️ Category:** {best_source.get('category', 'Unknown')}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Citations
                citations = extract_citations(best_answer + " " + best_source.get('text', ''))
                if any(citations.values()):
                    with st.expander("📎 Citations Found"):
                        for cite_type, cites in citations.items():
                            if cites:
                                st.write(f"**{cite_type.replace('_', ' ').title()}:**")
                                for cite in cites:
                                    st.write(f"- {cite}")
                
                # Export options
                col1, col2 = st.columns(2)
                
                with col1:
                    result_json = json.dumps({
                        'question': question,
                        'answer': best_answer,
                        'confidence': best_confidence,
                        'source': best_source.get('title', 'Unknown'),
                        'category': classification['category'],
                        'timestamp': str(datetime.now())
                    }, indent=2)
                    
                    st.download_button(
                        "📥 Download (JSON)",
                        result_json,
                        f"legal_answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        "application/json"
                    )
                
                with col2:
                    result_text = f"""Question: {question}
Answer: {best_answer}
Confidence: {best_confidence*100:.1f}%
Source: {best_source.get('title', 'Unknown')}
Category: {classification['category']}
Date: {datetime.now()}
"""
                    st.download_button(
                        "📄 Download (TXT)",
                        result_text,
                        f"legal_answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        "text/plain"
                    )
                
                # Feedback
                st.subheader("💬 Was this helpful?")
                col1, col2, col3 = st.columns([1, 1, 3])
                
                with col1:
                    if st.button("👍 Helpful"):
                        save_feedback(question, best_answer, "positive")
                        st.success("Thanks for your feedback!")
                
                with col2:
                    if st.button("👎 Not Helpful"):
                        feedback_comment = st.text_input("What was wrong?", key="feedback_comment")
                        if feedback_comment:
                            save_feedback(question, best_answer, "negative", feedback_comment)
                            st.error("Thanks! We'll improve!")
                
                # Source context
                with st.expander("📖 View Full Source"):
                    st.markdown(best_source.get('text', 'No context available'))
                
                # Alternative answers
                if len(all_answers) > 1:
                    with st.expander(f"🔍 View {len(all_answers)-1} Alternative Answers"):
                        for i, ans in enumerate(all_answers[1:6], 1):
                            st.markdown(f"**Answer {i}:** {ans['answer']}")
                            st.markdown(f"*Confidence: {ans['confidence']*100:.1f}% | Source: {ans['source']}*")
                            st.markdown("---")
            else:
                st.warning("⚠️ Could not extract answer. Showing relevant provisions:")
                for i, doc in enumerate(relevant_docs[:3], 1):
                    with st.expander(f"📄 Document {i}: {doc.get('title', 'Unknown')}"):
                        st.markdown(doc.get('text', '')[:500] + "...")
            
            # Top predictions
            with st.expander("📊 Category Predictions"):
                for i, pred in enumerate(classification['top_predictions'], 1):
                    st.progress(pred['confidence'])
                    st.markdown(f"{i}. **{pred['category']}**: {pred['confidence']*100:.1f}%")
    
    with tab2:
        st.header("📄 Analyze Legal Document")
        
        uploaded_file = st.file_uploader(
            "Upload a legal document",
            type=['pdf', 'docx', 'txt'],
            help="Upload PDF, DOCX, or TXT files (max 5MB)"
        )
        
        if uploaded_file:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Extract text
            with st.spinner("📖 Reading document..."):
                if uploaded_file.name.endswith('.pdf'):
                    text = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.name.endswith('.docx'):
                    text = extract_text_from_docx(uploaded_file)
                else:
                    text = uploaded_file.read().decode('utf-8')
            
            if text:
                st.markdown('<div class="document-box">', unsafe_allow_html=True)
                st.markdown(f"**📄 Document:** {uploaded_file.name}")
                st.markdown(f"**📏 Length:** {len(text)} chars, {len(text.split())} words")
                st.markdown('</div>', unsafe_allow_html=True)
                
                with st.expander("👁️ Preview (first 500 chars)"):
                    st.text(text[:500] + "..." if len(text) > 500 else text)
                
                if st.button("🔍 Analyze Document", type="primary"):
                    with st.spinner("🤖 Analyzing..."):
                        classification = classify_text(text[:1000], models)
                        
                        if classification:
                            st.success("✅ Analysis Complete!")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("📂 Primary Category", classification['category'])
                                st.metric("🎯 Confidence", f"{classification['confidence']*100:.1f}%")
                            
                            with col2:
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=classification['confidence']*100,
                                    title={'text': "Confidence"},
                                    gauge={
                                        'axis': {'range': [0, 100]},
                                        'bar': {'color': "darkblue"},
                                        'steps': [
                                            {'range': [0, 50], 'color': "lightgray"},
                                            {'range': [50, 80], 'color': "gray"},
                                            {'range': [80, 100], 'color': "lightblue"}
                                        ]
                                    }
                                ))
                                fig.update_layout(height=250)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Category breakdown
                            st.subheader("📊 Category Breakdown")
                            pred_df = pd.DataFrame(classification['top_predictions'])
                            pred_df['confidence_pct'] = pred_df['confidence'] * 100
                            
                            fig = px.bar(
                                pred_df,
                                x='confidence_pct',
                                y='category',
                                orientation='h',
                                title='Top Category Predictions',
                                labels={'confidence_pct': 'Confidence (%)', 'category': 'Category'},
                                color='confidence_pct',
                                color_continuous_scale='Blues'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Ask questions about document
                            st.subheader("💬 Ask Questions About This Document")
                            doc_question = st.text_input(
                                "Enter your question:",
                                placeholder="e.g., What is the main legal provision?"
                            )
                            
                            if doc_question:
                                with st.spinner("🤔 Finding answer..."):
                                    qa_result = answer_question(doc_question, text, models)
                                    
                                    if qa_result:
                                        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                                        st.markdown(f"**💡 Answer:** {qa_result['answer']}")
                                        st.markdown(f"**🎯 Confidence:** {qa_result['confidence']*100:.1f}%")
                                        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.header("📊 System Analytics")
        
        # Dataset stats
        st.subheader("📚 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Documents", len(dataset))
        with col2:
            st.metric("Categories", len(models['category_mapping']['categories']))
        with col3:
            avg_length = sum(len(d.get('text', '')) for d in dataset) / len(dataset) if dataset else 0
            st.metric("Avg Length", f"{int(avg_length)} chars")
        with col4:
            total_words = sum(len(d.get('text', '').split()) for d in dataset)
            st.metric("Total Words", f"{total_words:,}")
        
        # Category distribution
        st.subheader("📊 Category Distribution")
        
        category_counts = {}
        for doc in dataset:
            cat = doc.get('category', 'Unknown')
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        cat_df = pd.DataFrame(list(category_counts.items()), columns=['Category', 'Count'])
        cat_df = cat_df.sort_values('Count', ascending=False)
        
        fig = px.bar(
            cat_df.head(15),
            x='Count',
            y='Category',
            orientation='h',
            title='Top 15 Legal Categories',
            color='Count',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        fig2 = px.pie(cat_df.head(10), values='Count', names='Category', title='Top 10 Categories')
        st.plotly_chart(fig2, use_container_width=True)
        
        # System info
        st.subheader("🖥️ System Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Device:** {models['device'].upper()}")
            st.info(f"**PyTorch:** {torch.__version__}")
            st.info(f"**CUDA:** {'✅ Available' if torch.cuda.is_available() else '❌ Not Available'}")
        
        with col2:
            st.info(f"**Model:** Legal-BERT")
            st.info(f"**Max Length:** 256 tokens")
            st.info(f"**Semantic Search:** ✅ Enabled")
    
    with tab4:
        st.header("🤖 Chatbot Mode")
        st.info("💡 Conversational interface for continuous legal discussions")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask your legal question..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    relevant_docs, classification = search_relevant_context(
                        prompt, dataset, embeddings, models
                    )
                    
                    best_answer = None
                    best_confidence = 0
                    
                    for doc in relevant_docs[:3]:
                        context = doc.get('text', '')
                        if context:
                            qa_result = answer_question(prompt, context, models)
                            if qa_result and qa_result['confidence'] > best_confidence:
                                best_confidence = qa_result['confidence']
                                best_answer = qa_result['answer']
                    
                    response = best_answer or "I couldn't find a specific answer. Could you rephrase your question?"
                    st.markdown(response)
                    st.markdown(f"*Confidence: {best_confidence*100:.1f}% | Category: {classification['category']}*")
            
            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"""
        ❌ **Oops! Something went wrong.**
        
        **Error**: {str(e)}
        
        **Possible Solutions**:
        - Try rephrasing your question
        - Check if models are loaded properly
        - Restart the app
        
        **Need Help?** Check `APP_GUIDE.md`
        """)

