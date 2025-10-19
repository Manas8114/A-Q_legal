#!/usr/bin/env python3
"""
A-Qlegal 2.0 - Simple Training Script
Trains the system with all available data without complex ML dependencies
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import pickle
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logger.remove()
logger.add("logs/simple_training.log", level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}")
logger.add(lambda msg: print(f"\033[92m{msg}\033[0m"), level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

class SimpleAQlegalTrainer:
    def __init__(self):
        self.data_dir = Path("data")
        self.models_dir = Path("models")
        self.embeddings_dir = Path("data/embeddings")
        self.processed_dir = Path("data/processed")
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.embeddings_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.legal_documents = []
        self.qa_pairs = []
        self.vectorizer = None
        
    def load_all_data(self):
        """Load all available legal data from various sources"""
        logger.info("🔄 Loading all legal data...")
        
        # 1. Load expanded legal dataset
        try:
            with open(self.data_dir / "expanded_legal_dataset.json", "r", encoding="utf-8") as f:
                expanded_data = json.load(f)
            logger.info(f"✅ Loaded {len(expanded_data)} documents from expanded dataset")
            self.legal_documents.extend(expanded_data)
        except Exception as e:
            logger.warning(f"❌ Failed to load expanded dataset: {e}")
        
        # 2. Load enhanced legal documents
        try:
            with open(self.data_dir / "enhanced_legal" / "enhanced_legal_documents.json", "r", encoding="utf-8") as f:
                enhanced_data = json.load(f)
            logger.info(f"✅ Loaded {len(enhanced_data)} documents from enhanced dataset")
            self.legal_documents.extend(enhanced_data)
        except Exception as e:
            logger.warning(f"❌ Failed to load enhanced dataset: {e}")
        
        # 2.1. Load new enhanced legal documents v2
        try:
            with open(self.data_dir / "enhanced_legal_documents_v2.json", "r", encoding="utf-8") as f:
                enhanced_v2_data = json.load(f)
            logger.info(f"✅ Loaded {len(enhanced_v2_data)} documents from enhanced dataset v2")
            self.legal_documents.extend(enhanced_v2_data)
        except Exception as e:
            logger.warning(f"❌ Failed to load enhanced dataset v2: {e}")
        
        # 3. Load world class legal data
        world_class_dirs = [
            "criminal_law", "civil_law", "constitutional_amendments",
            "high_court_decisions", "indian_constitution", "legal_glossary",
            "legal_precedents", "multilingual_legal", "supreme_court_judgments"
        ]
        
        for dir_name in world_class_dirs:
            try:
                file_path = self.data_dir / "world_class_legal" / dir_name / f"{dir_name}.json"
                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    logger.info(f"✅ Loaded {len(data)} documents from {dir_name}")
                    self.legal_documents.extend(data)
            except Exception as e:
                logger.warning(f"❌ Failed to load {dir_name}: {e}")
        
        # 4. Load Indian legal data
        indian_legal_files = [
            "fundamental_rights.json", "indian_constitution.json",
            "indian_legal_qa_pairs.json", "legal_documents.json", "legal_glossary.json"
        ]
        
        for file_name in indian_legal_files:
            try:
                file_path = self.data_dir / "indian_legal" / file_name
                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    logger.info(f"✅ Loaded {len(data)} documents from {file_name}")
                    self.legal_documents.extend(data)
            except Exception as e:
                logger.warning(f"❌ Failed to load {file_name}: {e}")
        
        # 5. Load external datasets
        try:
            # HuggingFace datasets
            hf_path = self.data_dir / "external_datasets" / "huggingface" / "indian_law_dataset"
            if (hf_path / "train.json").exists():
                with open(hf_path / "train.json", "r", encoding="utf-8") as f:
                    hf_data = json.load(f)
                logger.info(f"✅ Loaded {len(hf_data)} documents from HuggingFace dataset")
                self.legal_documents.extend(hf_data)
        except Exception as e:
            logger.warning(f"❌ Failed to load HuggingFace dataset: {e}")
        
        logger.info(f"📊 Total documents loaded: {len(self.legal_documents)}")
        return len(self.legal_documents)
    
    def preprocess_documents(self):
        """Preprocess and standardize all legal documents"""
        logger.info("🔄 Preprocessing legal documents...")
        
        processed_docs = []
        
        for doc in tqdm(self.legal_documents, desc="Processing documents"):
            try:
                # Extract key information
                processed_doc = {
                    "id": doc.get("id", f"doc_{len(processed_docs)}"),
                    "title": doc.get("title", doc.get("section", "Unknown")),
                    "content": doc.get("content", doc.get("text", "")),
                    "category": doc.get("category", "general"),
                    "section": doc.get("section", ""),
                    "punishment": doc.get("punishment", ""),
                    "citations": doc.get("citations", []),
                    "source": doc.get("source", "Unknown")
                }
                
                # Clean and normalize content
                processed_doc["content"] = self.clean_text(processed_doc["content"])
                
                # Generate simplified summary
                processed_doc["simplified_summary"] = self.generate_simplified_summary(
                    processed_doc["content"], processed_doc["title"]
                )
                
                # Extract keywords
                processed_doc["keywords"] = self.extract_keywords(processed_doc["content"])
                
                # Generate real-life example
                processed_doc["real_life_example"] = self.generate_real_life_example(
                    processed_doc["content"], processed_doc["title"]
                )
                
                processed_docs.append(processed_doc)
                
            except Exception as e:
                logger.warning(f"❌ Failed to process document: {e}")
                continue
        
        self.legal_documents = processed_docs
        logger.info(f"✅ Processed {len(processed_docs)} documents")
        
        # Save processed data
        with open(self.processed_dir / "all_legal_documents.json", "w", encoding="utf-8") as f:
            json.dump(processed_docs, f, indent=2, ensure_ascii=False)
        
        return len(processed_docs)
    
    def clean_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep legal citations
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\(\)\[\]\-\'\"\/]', '', text)
        
        # Normalize legal citations
        text = re.sub(r'Section\s+(\d+[A-Z]?)', r'Section \1', text)
        
        return text.strip()
    
    def generate_simplified_summary(self, content, title):
        """Generate simplified summary of legal text"""
        if not content:
            return "No content available"
        
        # Simple rule-based summarization
        sentences = content.split('.')
        
        # Look for key phrases
        key_phrases = [
            "punishment", "penalty", "fine", "imprisonment", "jail",
            "offence", "crime", "violation", "prohibited", "forbidden"
        ]
        
        important_sentences = []
        for sentence in sentences:
            if any(phrase in sentence.lower() for phrase in key_phrases):
                important_sentences.append(sentence.strip())
        
        if important_sentences:
            return '. '.join(important_sentences[:3]) + '.'
        else:
            # Take first few sentences
            return '. '.join(sentences[:2]) + '.'
    
    def extract_keywords(self, content):
        """Extract keywords from legal content"""
        if not content:
            return []
        
        # Legal keywords
        legal_keywords = [
            "punishment", "fine", "imprisonment", "offence", "crime",
            "violation", "prohibited", "forbidden", "penalty", "jail",
            "court", "judge", "law", "legal", "criminal", "civil",
            "constitution", "rights", "duty", "liability", "damages"
        ]
        
        content_lower = content.lower()
        found_keywords = [kw for kw in legal_keywords if kw in content_lower]
        
        # Add section numbers
        section_matches = re.findall(r'Section\s+(\d+[A-Z]?)', content)
        found_keywords.extend([f"Section {s}" for s in section_matches])
        
        return list(set(found_keywords))[:10]  # Limit to 10 keywords
    
    def generate_real_life_example(self, content, title):
        """Generate real-life example for legal concept"""
        if not content:
            return "No example available"
        
        # Simple template-based example generation
        if "fraud" in content.lower() or "cheat" in content.lower():
            return "A person sells fake gold jewelry claiming it's real gold, tricking customers into paying high prices."
        elif "murder" in content.lower() or "homicide" in content.lower():
            return "A person intentionally kills another person with a weapon."
        elif "theft" in content.lower() or "steal" in content.lower():
            return "A person takes someone else's property without permission."
        elif "assault" in content.lower():
            return "A person physically attacks another person causing injury."
        elif "defamation" in content.lower():
            return "A person spreads false rumors about someone to damage their reputation."
        else:
            return f"Example related to {title}: A person violates this law and faces consequences."
    
    def create_qa_pairs(self):
        """Create question-answer pairs from legal documents"""
        logger.info("🔄 Creating Q&A pairs...")
        
        qa_pairs = []
        
        for doc in tqdm(self.legal_documents, desc="Creating Q&A pairs"):
            try:
                # Generate questions based on document content
                questions = self.generate_questions(doc)
                
                for question in questions:
                    qa_pairs.append({
                        "question": question,
                        "answer": doc["simplified_summary"],
                        "context": doc["content"],
                        "section": doc.get("section", ""),
                        "category": doc["category"],
                        "source": doc["source"]
                    })
                    
            except Exception as e:
                logger.warning(f"❌ Failed to create Q&A for document: {e}")
                continue
        
        self.qa_pairs = qa_pairs
        logger.info(f"✅ Created {len(qa_pairs)} Q&A pairs")
        
        # Save Q&A pairs
        with open(self.processed_dir / "legal_qa_pairs.json", "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
        
        return len(qa_pairs)
    
    def generate_questions(self, doc):
        """Generate questions from legal document"""
        questions = []
        
        title = doc["title"]
        section = doc.get("section", "")
        
        # Basic questions
        if section:
            questions.append(f"What is {section}?")
            questions.append(f"Explain {section}")
            questions.append(f"Tell me about {section}")
        
        questions.append(f"What is {title}?")
        questions.append(f"Explain {title}")
        
        # Category-specific questions
        category = doc["category"]
        if "criminal" in category.lower():
            questions.append(f"What is the punishment for {title}?")
            questions.append(f"Is {title} a crime?")
        elif "civil" in category.lower():
            questions.append(f"What are the civil implications of {title}?")
        elif "constitutional" in category.lower():
            questions.append(f"What does the Constitution say about {title}?")
        
        return questions[:3]  # Limit to 3 questions per document
    
    def train_tfidf_vectorizer(self):
        """Train TF-IDF vectorizer for keyword search"""
        logger.info("🔄 Training TF-IDF vectorizer...")
        
        try:
            # Prepare texts
            texts = []
            for doc in self.legal_documents:
                texts.append(doc["content"])
            
            # Train TF-IDF
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Save vectorizer and matrix
            with open(self.models_dir / "tfidf_vectorizer.pkl", "wb") as f:
                pickle.dump(self.vectorizer, f)
            
            np.save(self.embeddings_dir / "tfidf_matrix.npy", tfidf_matrix.toarray())
            
            logger.info("✅ TF-IDF vectorizer trained and saved")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to train TF-IDF vectorizer: {e}")
            return False
    
    def create_enhanced_app(self):
        """Create enhanced app with all trained components"""
        logger.info("🔄 Creating enhanced app...")
        
        app_code = '''#!/usr/bin/env python3
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
    try:
        with open("data/processed/all_legal_documents.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

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
'''
        
        with open("enhanced_legal_app.py", "w", encoding="utf-8") as f:
            f.write(app_code)
        
        logger.info("✅ Enhanced app created")
        return True
    
    def run_training(self):
        """Run complete training pipeline"""
        logger.info("🚀 Starting A-Qlegal 2.0 Simple Training")
        logger.info("=" * 60)
        
        steps = [
            ("Loading all data", self.load_all_data),
            ("Preprocessing documents", self.preprocess_documents),
            ("Creating Q&A pairs", self.create_qa_pairs),
            ("Training TF-IDF vectorizer", self.train_tfidf_vectorizer),
            ("Creating enhanced app", self.create_enhanced_app)
        ]
        
        for i, (description, func) in enumerate(steps, 1):
            logger.info(f"Step {i}/{len(steps)}: {description}")
            try:
                result = func()
                if result:
                    logger.info(f"✅ {description} completed")
                else:
                    logger.error(f"❌ {description} failed")
                    return False
            except Exception as e:
                logger.error(f"❌ {description} failed: {e}")
                return False
            logger.info("")
        
        logger.success("🎉 A-Qlegal 2.0 Training Completed Successfully!")
        logger.info("")
        logger.info("🚀 To run the enhanced app:")
        logger.info("   streamlit run enhanced_legal_app.py")
        logger.info("")
        logger.info("📊 Training Summary:")
        logger.info(f"   • Documents processed: {len(self.legal_documents)}")
        logger.info(f"   • Q&A pairs created: {len(self.qa_pairs)}")
        logger.info("   • Models trained: TF-IDF Vectorizer")
        logger.info("   • Search type: Advanced Keyword Search")
        
        return True

def main():
    """Main function"""
    trainer = SimpleAQlegalTrainer()
    success = trainer.run_training()
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("🚀 Run: streamlit run enhanced_legal_app.py")
    else:
        print("\n❌ Training failed. Check logs for details.")

if __name__ == "__main__":
    main()
