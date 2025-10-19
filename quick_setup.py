#!/usr/bin/env python3
"""
Quick Setup Script for A-Qlegal 2.0
Minimal setup to get the system running
"""

import os
import sys
import json
from pathlib import Path
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

def create_directories():
    """Create necessary directories"""
    dirs = [
        "data/raw", "data/processed", "data/embeddings",
        "models", "models/legal_bert", "models/flan_t5", "models/indic_bert",
        "models/lora", "models/rag", "logs", "cache", "configs"
    ]
    
    for dir_path in dirs:
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        except Exception as e:
            logger.error(f"Failed to create {dir_path}: {e}")
            return False
    
    return True

def create_sample_data():
    """Create sample legal data for testing"""
    logger.info("Creating sample legal data...")
    
    sample_data = {
        "legal_documents": [
            {
                "section": "Section 420 IPC",
                "legal_text": "Whoever cheats and thereby dishonestly induces the person deceived to deliver any property to any person, or to make, alter or destroy the whole or any part of a valuable security, or anything which is signed or sealed, and which is capable of being converted into a valuable security, shall be punished with imprisonment of either description for a term which may extend to seven years, and shall also be liable to fine.",
                "simplified_summary": "If someone tricks another person into giving them money or property by lying, they can be jailed for up to 7 years and fined.",
                "real_life_example": "A person sells fake gold jewelry claiming it's real gold, tricking customers into paying high prices.",
                "punishment": "Imprisonment up to 7 years + Fine",
                "keywords": ["cheating", "fraud", "property", "deception"],
                "category": "Criminal"
            },
            {
                "section": "Section 138 NI Act",
                "legal_text": "Where any cheque drawn by a person on an account maintained by him with a banker for payment of any amount of money to another person from out of that account for the discharge, in whole or in part, of any debt or other liability, is returned by the bank unpaid, either because of the amount of money standing to the credit of that account is insufficient to honour the cheque or that it exceeds the amount arranged to be paid from that account by an agreement made with that bank, such person shall be deemed to have committed an offence.",
                "simplified_summary": "If you write a check but don't have enough money in your account to pay it, you commit a crime.",
                "real_life_example": "A person writes a check for ₹50,000 but only has ₹10,000 in their account.",
                "punishment": "Imprisonment up to 2 years + Fine up to twice the check amount",
                "keywords": ["bounced check", "insufficient funds", "banking", "debt"],
                "category": "Criminal"
            },
            {
                "section": "Section 299 IPC",
                "legal_text": "Whoever causes death by doing an act with the intention of causing death, or with the intention of causing such bodily injury as is likely to cause death, or with the knowledge that he is likely by such act to cause death, commits the offence of culpable homicide.",
                "simplified_summary": "If someone does something that they know could kill another person, and that person dies, it's called culpable homicide.",
                "real_life_example": "A person hits someone on the head with a heavy object, knowing it could kill them, and the person dies.",
                "punishment": "Life imprisonment or up to 10 years + Fine",
                "keywords": ["murder", "homicide", "death", "intention"],
                "category": "Criminal"
            },
            {
                "section": "Section 125 CrPC",
                "legal_text": "If any person having sufficient means neglects or refuses to maintain his wife, legitimate or illegitimate minor child, legitimate or illegitimate child (not being a married daughter) who has attained majority, where such child is, by reason of any physical or mental abnormality or injury unable to maintain itself, or his father or mother, unable to maintain himself or herself, a Magistrate of the first class may, upon proof of such neglect or refusal, order such person to make a monthly allowance for the maintenance of his wife or such child, father or mother, at such monthly rate not exceeding five hundred rupees in the whole, as such Magistrate thinks fit, and to pay the same to such person as the Magistrate from time to time directs.",
                "simplified_summary": "If a person has money but refuses to take care of their family members who can't support themselves, a court can order them to pay monthly support.",
                "real_life_example": "A man earns well but refuses to give money to his elderly parents who can't work anymore.",
                "punishment": "Court order to pay monthly maintenance up to ₹500",
                "keywords": ["maintenance", "family support", "neglect", "court order"],
                "category": "Civil"
            }
        ]
    }
    
    with open("data/processed/sample_legal_data.json", "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    logger.success("✅ Sample data created")
    return True

def create_config():
    """Create configuration file"""
    config = {
        "model_settings": {
            "legal_bert_model": "nlpaueb/legal-bert-base-uncased",
            "flan_t5_model": "google/flan-t5-base",
            "sentence_transformer": "all-MiniLM-L6-v2"
        },
        "data_settings": {
            "max_length": 512,
            "batch_size": 8,
            "learning_rate": 2e-5
        },
        "rag_settings": {
            "top_k": 5,
            "similarity_threshold": 0.7
        },
        "multilingual_settings": {
            "supported_languages": ["en", "hi", "ta", "bn", "te"],
            "default_language": "en"
        }
    }
    
    with open("configs/aqlegal_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.success("✅ Configuration created")
    return True

def create_simple_app():
    """Create a simple working app"""
    app_code = '''#!/usr/bin/env python3
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
'''
    
    with open("simple_legal_app.py", "w", encoding="utf-8") as f:
        f.write(app_code)
    
    logger.success("✅ Simple app created")
    return True

def test_basic_functionality():
    """Test basic functionality"""
    logger.info("Testing basic functionality...")
    
    try:
        # Test data loading
        with open("data/processed/sample_legal_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if len(data["legal_documents"]) > 0:
            logger.success("✅ Data loading works")
        else:
            logger.error("❌ No data found")
            return False
        
        # Test config loading
        with open("configs/aqlegal_config.json", "r") as f:
            config = json.load(f)
        
        if "model_settings" in config:
            logger.success("✅ Configuration loading works")
        else:
            logger.error("❌ Configuration incomplete")
            return False
        
        logger.success("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("🚀 Starting A-Qlegal 2.0 Quick Setup")
    logger.info("=" * 50)
    
    steps = [
        ("Creating directories", create_directories),
        ("Creating sample data", create_sample_data),
        ("Creating configuration", create_config),
        ("Creating simple app", create_simple_app),
        ("Testing functionality", test_basic_functionality)
    ]
    
    for i, (description, func) in enumerate(steps, 1):
        logger.info(f"Step {i}/{len(steps)}: {description}")
        if not func():
            logger.error(f"❌ Setup failed at step {i}: {description}")
            return False
        logger.info("")
    
    logger.success("🎉 A-Qlegal 2.0 Quick Setup Completed!")
    logger.info("")
    logger.info("🚀 To run the app:")
    logger.info("   streamlit run simple_legal_app.py")
    logger.info("")
    logger.info("📚 The app includes sample legal data and basic search functionality.")
    logger.info("   You can ask questions like:")
    logger.info("   - 'What is the punishment for fraud?'")
    logger.info("   - 'Tell me about bounced checks'")
    logger.info("   - 'What is culpable homicide?'")
    
    return True

if __name__ == "__main__":
    main()
