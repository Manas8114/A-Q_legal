import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from src.main import LegalQASystem

os.environ['GEMINI_API_KEY'] = "AIzaSyDLOMncFan_QBHFz0BDYw_gWtEVNTJ3NyE"

print("Loading saved system...")
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
    'default_top_k': 15,
    'max_context_length': 2000,
    'batch_size': 32,
    'max_epochs': 2,
    'learning_rate': 2e-5
}

system = LegalQASystem(config)
system.load_system("models/optimized_legal_qa")
print("✅ System loaded successfully!")

# Test a question
print("\nTesting with a sample question...")
result = system.ask_question("What are the fundamental rights in the Constitution?", use_generative_only=True)
print(f"Question: {result['question']}")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
print(f"Source: {result['source']}")
print(f"Documents Retrieved: {len(result.get('retrieved_documents', []))}")
