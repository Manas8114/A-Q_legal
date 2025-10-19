"""
Dense retrieval system using FAISS and sentence transformers
"""
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import torch
from loguru import logger
import pickle
from pathlib import Path


class DenseRetriever:
    """Dense retrieval using FAISS and sentence transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 index_type: str = "flat"):
        self.model_name = model_name
        self.index_type = index_type
        self.model = None
        self.index = None
        self.documents = []
        self.document_metadata = []
        self.embeddings = None
        self.is_fitted = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Load sentence transformer model"""
        try:
            logger.info(f"Loading dense retrieval model: {self.model_name}")
            # Force GPU usage if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(self.model_name, device=device)
            logger.info(f"Model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def _create_index(self, embedding_dim: int):
        """Create FAISS index"""
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, 100)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(embedding_dim, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        logger.info(f"Created FAISS index of type: {self.index_type}")
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms
    
    def fit(self, documents: List[str], metadata: List[Dict[str, Any]] = None,
            batch_size: int = 32):
        """Fit dense retriever on documents"""
        logger.info(f"Fitting dense retriever on {len(documents)} documents")
        
        self.documents = documents
        self.document_metadata = metadata or [{}] * len(documents)
        
        # Generate embeddings
        logger.info("Generating document embeddings...")
        self.embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Normalize embeddings for cosine similarity
        self.embeddings = self._normalize_embeddings(self.embeddings)
        
        # Create FAISS index
        embedding_dim = self.embeddings.shape[1]
        self._create_index(embedding_dim)
        
        # Add embeddings to index
        if self.index_type == "ivf":
            self.index.train(self.embeddings)
        
        self.index.add(self.embeddings.astype('float32'))
        self.is_fitted = True
        
        logger.info(f"Dense retriever fitted successfully with {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        if not self.is_fitted:
            raise ValueError("Dense retriever must be fitted before searching")
        
        if not query:
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = self._normalize_embeddings(query_embedding)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid index
                result = {
                    'index': int(idx),
                    'document': self.documents[idx],
                    'score': float(score),
                    'metadata': self.document_metadata[idx] if idx < len(self.document_metadata) else {}
                }
                results.append(result)
        
        logger.debug(f"Dense search returned {len(results)} results for query: {query[:50]}...")
        return results
    
    def add_documents(self, new_documents: List[str], 
                     new_metadata: List[Dict[str, Any]] = None,
                     batch_size: int = 32):
        """Add new documents to the index"""
        if not self.is_fitted:
            raise ValueError("Dense retriever must be fitted before adding documents")
        
        logger.info(f"Adding {len(new_documents)} new documents to dense index")
        
        # Generate embeddings for new documents
        new_embeddings = self.model.encode(
            new_documents,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        new_embeddings = self._normalize_embeddings(new_embeddings)
        
        # Add to existing data
        self.documents.extend(new_documents)
        if new_metadata:
            self.document_metadata.extend(new_metadata)
        else:
            self.document_metadata.extend([{}] * len(new_documents))
        
        # Update embeddings
        self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        # Add to FAISS index
        self.index.add(new_embeddings.astype('float32'))
        
        logger.info(f"Dense index now contains {len(self.documents)} documents")
    
    def remove_documents(self, indices: List[int]):
        """Remove documents from the index (requires rebuilding)"""
        if not self.is_fitted:
            raise ValueError("Dense retriever must be fitted before removing documents")
        
        logger.info(f"Removing {len(indices)} documents from dense index")
        
        # Sort indices in descending order to avoid index shifting
        indices = sorted(indices, reverse=True)
        
        # Remove from data structures
        for idx in indices:
            if 0 <= idx < len(self.documents):
                del self.documents[idx]
                if idx < len(self.document_metadata):
                    del self.document_metadata[idx]
        
        # Rebuild index
        if len(self.documents) > 0:
            # Regenerate embeddings
            self.embeddings = self.model.encode(
                self.documents,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            self.embeddings = self._normalize_embeddings(self.embeddings)
            
            # Recreate index
            embedding_dim = self.embeddings.shape[1]
            self._create_index(embedding_dim)
            
            if self.index_type == "ivf":
                self.index.train(self.embeddings)
            
            self.index.add(self.embeddings.astype('float32'))
        else:
            self.embeddings = None
            self.index = None
            self.is_fitted = False
        
        logger.info(f"Dense index now contains {len(self.documents)} documents")
    
    def save_model(self, filepath: str):
        """Save dense retriever model to disk"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'documents': self.documents,
            'document_metadata': self.document_metadata,
            'embeddings': self.embeddings,
            'model_name': self.model_name,
            'index_type': self.index_type
        }
        
        # Save FAISS index separately
        index_file = filepath.replace('.pkl', '_index.faiss')
        faiss.write_index(self.index, index_file)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Dense retriever model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load dense retriever model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.documents = model_data['documents']
        self.document_metadata = model_data['document_metadata']
        self.embeddings = model_data['embeddings']
        self.model_name = model_data['model_name']
        self.index_type = model_data['index_type']
        
        # Load FAISS index
        index_file = filepath.replace('.pkl', '_index.faiss')
        self.index = faiss.read_index(index_file)
        
        self.is_fitted = True
        
        logger.info(f"Dense retriever model loaded from {filepath}")
    
    def get_similarity_scores(self, query: str) -> np.ndarray:
        """Get similarity scores for all documents"""
        if not self.is_fitted:
            raise ValueError("Dense retriever must be fitted before getting scores")
        
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = self._normalize_embeddings(query_embedding)
        
        # Compute similarities with all documents
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        
        return similarities
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the dense retriever"""
        return {
            'is_fitted': self.is_fitted,
            'num_documents': len(self.documents),
            'model_name': self.model_name,
            'index_type': self.index_type,
            'embedding_dim': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'device': self.device
        }
    
    def explain_similarity(self, query: str, doc_index: int) -> Dict[str, Any]:
        """Explain similarity between query and document"""
        if not self.is_fitted:
            raise ValueError("Dense retriever must be fitted before explaining similarities")
        
        if doc_index >= len(self.documents):
            raise ValueError("Document index out of range")
        
        # Get embeddings
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = self._normalize_embeddings(query_embedding)
        
        doc_embedding = self.embeddings[doc_index:doc_index+1]
        
        # Compute similarity
        similarity = np.dot(query_embedding, doc_embedding.T)[0][0]
        
        return {
            'query': query,
            'document_index': doc_index,
            'document': self.documents[doc_index][:200] + "..." if len(self.documents[doc_index]) > 200 else self.documents[doc_index],
            'similarity': float(similarity),
            'query_embedding_norm': float(np.linalg.norm(query_embedding)),
            'doc_embedding_norm': float(np.linalg.norm(doc_embedding))
        }