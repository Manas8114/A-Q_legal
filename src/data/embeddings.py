"""
Embedding generation using Sentence-BERT and other models
"""
import numpy as np
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
import torch
from loguru import logger
import pickle
from pathlib import Path


class EmbeddingGenerator:
    """Generates embeddings for legal text using various models"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            # Force GPU usage if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(self.model_name, device=device)
            logger.info(f"Model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if not texts:
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def generate_question_embeddings(self, questions: List[str]) -> np.ndarray:
        """Generate embeddings specifically for questions"""
        return self.generate_embeddings(questions)
    
    def generate_answer_embeddings(self, answers: List[str]) -> np.ndarray:
        """Generate embeddings specifically for answers"""
        return self.generate_embeddings(answers)
    
    def generate_context_embeddings(self, contexts: List[str]) -> np.ndarray:
        """Generate embeddings specifically for contexts"""
        return self.generate_embeddings(contexts)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        if embedding1.shape != embedding2.shape:
            raise ValueError("Embeddings must have the same shape")
        
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                         candidate_embeddings: np.ndarray, 
                         top_k: int = 5) -> List[Dict[str, Any]]:
        """Find most similar embeddings to query"""
        if len(candidate_embeddings) == 0:
            return []
        
        # Compute similarities
        similarities = []
        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate)
            similarities.append({
                'index': i,
                'similarity': similarity
            })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]
    
    def save_embeddings(self, embeddings: np.ndarray, file_path: str):
        """Save embeddings to file"""
        logger.info(f"Saving embeddings to {file_path}")
        np.save(file_path, embeddings)
        logger.info("Embeddings saved successfully")
    
    def load_embeddings(self, file_path: str) -> np.ndarray:
        """Load embeddings from file"""
        logger.info(f"Loading embeddings from {file_path}")
        embeddings = np.load(file_path)
        logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def save_metadata(self, metadata: Dict[str, Any], file_path: str):
        """Save embedding metadata"""
        with open(file_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Metadata saved to {file_path}")
    
    def load_metadata(self, file_path: str) -> Dict[str, Any]:
        """Load embedding metadata"""
        with open(file_path, 'rb') as f:
            metadata = pickle.load(f)
        logger.info(f"Metadata loaded from {file_path}")
        return metadata
    
    def create_embedding_index(self, texts: List[str], 
                             embeddings: np.ndarray,
                             metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create an embedding index for efficient retrieval"""
        index = {
            'texts': texts,
            'embeddings': embeddings,
            'metadata': metadata or {},
            'model_name': self.model_name,
            'embedding_dim': embeddings.shape[1] if len(embeddings) > 0 else 0
        }
        
        logger.info(f"Created embedding index with {len(texts)} texts")
        return index
    
    def search_similar(self, query: str, 
                      embedding_index: Dict[str, Any], 
                      top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar texts in the embedding index"""
        # Generate query embedding
        query_embedding = self.generate_embeddings([query])[0]
        
        # Find most similar
        results = self.find_most_similar(
            query_embedding,
            embedding_index['embeddings'],
            top_k
        )
        
        # Add text information
        for result in results:
            idx = result['index']
            result['text'] = embedding_index['texts'][idx]
            if 'metadata' in embedding_index and idx < len(embedding_index['metadata']):
                result['metadata'] = embedding_index['metadata'][idx]
        
        return results