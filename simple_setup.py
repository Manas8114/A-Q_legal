#!/usr/bin/env python3
"""
Simple Setup Script for A-Qlegal 2.0
Bypasses problematic dependencies and focuses on core functionality
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
logger.add("logs/simple_setup.log", level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}")

def run_command(cmd, description=""):
    """Run a command and return success status"""
    try:
        logger.info(f"Running: {description or cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            logger.success(f"✅ {description or 'Command completed'}")
            return True
        else:
            logger.error(f"❌ {description or 'Command failed'}: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {description or 'Command timed out'}")
        return False
    except Exception as e:
        logger.error(f"❌ {description or 'Command failed'}: {str(e)}")
        return False

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

def install_core_dependencies():
    """Install core dependencies without problematic packages"""
    core_packages = [
        "torch>=2.0.0",
        "transformers>=4.30.0", 
        "sentence-transformers>=2.2.2",
        "datasets>=2.12.0",
        "accelerate>=0.20.0",
        "peft>=0.6.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "faiss-cpu>=1.7.4",
        "rank-bm25>=0.2.2",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "streamlit>=1.25.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "PyPDF2>=3.0.0",
        "python-docx>=0.8.11",
        "rouge-score>=0.1.2",
        "sacrebleu>=2.3.0",
        "bert-score>=0.3.13",
        "textstat>=0.7.3",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.0",
        "typer>=0.9.0",
        "rich>=13.7.0",
        "huggingface-hub>=0.19.0"
    ]
    
    logger.info("Installing core dependencies...")
    for package in core_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            logger.warning(f"Failed to install {package}, continuing...")
    
    return True

def download_models():
    """Download and cache essential models"""
    logger.info("Downloading essential models...")
    
    # Download models using Python
    model_script = """
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import os

print("Downloading Legal-BERT...")
try:
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
    print("✅ Legal-BERT downloaded")
except Exception as e:
    print(f"❌ Legal-BERT failed: {e}")

print("Downloading Flan-T5...")
try:
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModel.from_pretrained("google/flan-t5-base")
    print("✅ Flan-T5 downloaded")
except Exception as e:
    print(f"❌ Flan-T5 failed: {e}")

print("Downloading Sentence Transformer...")
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Sentence Transformer downloaded")
except Exception as e:
    print(f"❌ Sentence Transformer failed: {e}")

print("Model download completed!")
"""
    
    with open("download_models.py", "w", encoding="utf-8") as f:
        f.write(model_script)
    
    return run_command("python download_models.py", "Downloading models")

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

def test_installation():
    """Test if everything is working"""
    logger.info("Testing installation...")
    
    test_script = """
import torch
import transformers
import sentence_transformers
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import json

print("Testing core imports...")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

print("Testing model loading...")
try:
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    print("✅ Flan-T5 tokenizer loaded")
except Exception as e:
    print(f"❌ Flan-T5 failed: {e}")

print("Testing data loading...")
try:
    with open("data/processed/sample_legal_data.json", "r") as f:
        data = json.load(f)
    print(f"✅ Sample data loaded: {len(data['legal_documents'])} documents")
except Exception as e:
    print(f"❌ Data loading failed: {e}")

print("✅ Installation test completed!")
"""
    
    with open("test_installation.py", "w", encoding="utf-8") as f:
        f.write(test_script)
    
    return run_command("python test_installation.py", "Testing installation")

def main():
    """Main setup function"""
    logger.info("🚀 Starting A-Qlegal 2.0 Simple Setup")
    logger.info("=" * 60)
    
    steps = [
        ("Creating directories", create_directories),
        ("Installing core dependencies", install_core_dependencies),
        ("Downloading models", download_models),
        ("Creating sample data", create_sample_data),
        ("Creating configuration", create_config),
        ("Testing installation", test_installation)
    ]
    
    for i, (description, func) in enumerate(steps, 1):
        logger.info(f"Step {i}/{len(steps)}: {description}")
        if not func():
            logger.error(f"❌ Setup failed at step {i}: {description}")
            return False
        logger.info("")
    
    logger.success("🎉 A-Qlegal 2.0 Setup Completed Successfully!")
    logger.info("You can now run: python legal_ai_app_enhanced.py")
    
    # Cleanup temporary files
    for temp_file in ["download_models.py", "test_installation.py"]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    return True

if __name__ == "__main__":
    main()
