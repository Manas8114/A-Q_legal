
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
