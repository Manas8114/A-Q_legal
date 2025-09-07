"""
BM25-based lexical retrieval system
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi
from loguru import logger
import pickle
from pathlib import Path


class BM25Retriever:
    """BM25-based lexical retrieval for legal documents"""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.documents = []
        self.document_metadata = []
        self.is_fitted = False
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25 indexing"""
        if not text:
            return []
        
        # Simple tokenization (can be enhanced with legal-specific preprocessing)
        tokens = text.lower().split()
        
        # Remove very short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens
    
    def fit(self, documents: List[str], metadata: List[Dict[str, Any]] = None):
        """Fit BM25 model on documents"""
        logger.info(f"Fitting BM25 model on {len(documents)} documents")
        
        self.documents = documents
        self.document_metadata = metadata or [{}] * len(documents)
        
        # Preprocess documents
        tokenized_docs = [self.preprocess_text(doc) for doc in documents]
        
        # Create BM25 model
        self.bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
        self.is_fitted = True
        
        logger.info("BM25 model fitted successfully")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        if not self.is_fitted:
            raise ValueError("BM25 model must be fitted before searching")
        
        if not query:
            return []
        
        # Preprocess query
        query_tokens = self.preprocess_text(query)
        
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with positive scores
                result = {
                    'index': int(idx),
                    'document': self.documents[idx],
                    'score': float(scores[idx]),
                    'metadata': self.document_metadata[idx] if idx < len(self.document_metadata) else {}
                }
                results.append(result)
        
        logger.debug(f"BM25 search returned {len(results)} results for query: {query[:50]}...")
        return results
    
    def get_document_scores(self, query: str) -> np.ndarray:
        """Get BM25 scores for all documents"""
        if not self.is_fitted:
            raise ValueError("BM25 model must be fitted before getting scores")
        
        query_tokens = self.preprocess_text(query)
        if not query_tokens:
            return np.zeros(len(self.documents))
        
        return self.bm25.get_scores(query_tokens)
    
    def add_documents(self, new_documents: List[str], 
                     new_metadata: List[Dict[str, Any]] = None):
        """Add new documents to the index"""
        if not self.is_fitted:
            raise ValueError("BM25 model must be fitted before adding documents")
        
        logger.info(f"Adding {len(new_documents)} new documents to BM25 index")
        
        # Add to existing documents
        self.documents.extend(new_documents)
        if new_metadata:
            self.document_metadata.extend(new_metadata)
        else:
            self.document_metadata.extend([{}] * len(new_documents))
        
        # Refit model
        tokenized_docs = [self.preprocess_text(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
        
        logger.info(f"BM25 index now contains {len(self.documents)} documents")
    
    def remove_documents(self, indices: List[int]):
        """Remove documents from the index"""
        if not self.is_fitted:
            raise ValueError("BM25 model must be fitted before removing documents")
        
        logger.info(f"Removing {len(indices)} documents from BM25 index")
        
        # Sort indices in descending order to avoid index shifting
        indices = sorted(indices, reverse=True)
        
        for idx in indices:
            if 0 <= idx < len(self.documents):
                del self.documents[idx]
                if idx < len(self.document_metadata):
                    del self.document_metadata[idx]
        
        # Refit model
        tokenized_docs = [self.preprocess_text(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
        
        logger.info(f"BM25 index now contains {len(self.documents)} documents")
    
    def save_model(self, filepath: str):
        """Save BM25 model to disk"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'documents': self.documents,
            'document_metadata': self.document_metadata,
            'k1': self.k1,
            'b': self.b,
            'bm25': self.bm25
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"BM25 model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load BM25 model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.documents = model_data['documents']
        self.document_metadata = model_data['document_metadata']
        self.k1 = model_data['k1']
        self.b = model_data['b']
        self.bm25 = model_data['bm25']
        self.is_fitted = True
        
        logger.info(f"BM25 model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the BM25 model"""
        return {
            'is_fitted': self.is_fitted,
            'num_documents': len(self.documents),
            'k1': self.k1,
            'b': self.b,
            'avg_doc_length': np.mean([len(self.preprocess_text(doc)) for doc in self.documents]) if self.documents else 0
        }
    
    def explain_score(self, query: str, doc_index: int) -> Dict[str, Any]:
        """Explain BM25 score for a specific document"""
        if not self.is_fitted:
            raise ValueError("BM25 model must be fitted before explaining scores")
        
        if doc_index >= len(self.documents):
            raise ValueError("Document index out of range")
        
        query_tokens = self.preprocess_text(query)
        doc_tokens = self.preprocess_text(self.documents[doc_index])
        
        # Get term frequencies
        term_freqs = {}
        for token in doc_tokens:
            term_freqs[token] = term_freqs.get(token, 0) + 1
        
        # Calculate BM25 components
        doc_length = len(doc_tokens)
        avg_doc_length = np.mean([len(self.preprocess_text(doc)) for doc in self.documents])
        
        score = 0
        term_scores = {}
        
        for term in query_tokens:
            if term in term_freqs:
                tf = term_freqs[term]
                idf = self.bm25.idf[term] if term in self.bm25.idf else 0
                
                # BM25 formula
                term_score = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_length)))
                score += term_score
                term_scores[term] = {
                    'term_frequency': tf,
                    'idf': idf,
                    'score': term_score
                }
        
        return {
            'query': query,
            'document_index': doc_index,
            'document': self.documents[doc_index][:200] + "..." if len(self.documents[doc_index]) > 200 else self.documents[doc_index],
            'total_score': score,
            'term_scores': term_scores,
            'document_length': doc_length,
            'average_document_length': avg_doc_length
        }