"""
Hybrid retrieval system combining BM25 and dense retrieval
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever
from loguru import logger


class HybridRetriever:
    """Hybrid retrieval combining BM25 and dense retrieval"""
    
    def __init__(self, bm25_weight: float = 0.3, dense_weight: float = 0.7,
                 bm25_k1: float = 1.2, bm25_b: float = 0.75,
                 dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.bm25_retriever = BM25Retriever(k1=bm25_k1, b=bm25_b)
        self.dense_retriever = DenseRetriever(model_name=dense_model_name)
        self.is_fitted = False
    
    def fit(self, documents: List[str], metadata: List[Dict[str, Any]] = None,
            batch_size: int = 32):
        """Fit both BM25 and dense retrievers"""
        logger.info(f"Fitting hybrid retriever on {len(documents)} documents")
        
        # Fit BM25 retriever
        logger.info("Fitting BM25 retriever...")
        self.bm25_retriever.fit(documents, metadata)
        
        # Fit dense retriever
        logger.info("Fitting dense retriever...")
        self.dense_retriever.fit(documents, metadata, batch_size)
        
        self.is_fitted = True
        logger.info("Hybrid retriever fitted successfully")
    
    def search(self, query: str, top_k: int = 10, 
               normalize_scores: bool = True) -> List[Dict[str, Any]]:
        """Search using hybrid approach"""
        if not self.is_fitted:
            raise ValueError("Hybrid retriever must be fitted before searching")
        
        if not query:
            return []
        
        # Get BM25 results
        bm25_results = self.bm25_retriever.search(query, top_k * 2)  # Get more for better fusion
        bm25_scores = {r['index']: r['score'] for r in bm25_results}
        
        # Get dense results
        dense_results = self.dense_retriever.search(query, top_k * 2)  # Get more for better fusion
        dense_scores = {r['index']: r['score'] for r in dense_results}
        
        # Get all unique document indices
        all_indices = set(bm25_scores.keys()) | set(dense_scores.keys())
        
        # Normalize scores if requested
        if normalize_scores and all_indices:
            bm25_scores = self._normalize_scores(bm25_scores)
            dense_scores = self._normalize_scores(dense_scores)
        
        # Combine scores
        combined_scores = {}
        for idx in all_indices:
            bm25_score = bm25_scores.get(idx, 0.0)
            dense_score = dense_scores.get(idx, 0.0)
            combined_score = (self.bm25_weight * bm25_score + 
                            self.dense_weight * dense_score)
            combined_scores[idx] = combined_score
        
        # Sort by combined score
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create results
        results = []
        for idx, combined_score in sorted_indices[:top_k]:
            # Get document info from either retriever
            if idx in bm25_scores:
                bm25_result = next(r for r in bm25_results if r['index'] == idx)
                document = bm25_result['document']
                metadata = bm25_result['metadata']
            else:
                dense_result = next(r for r in dense_results if r['index'] == idx)
                document = dense_result['document']
                metadata = dense_result['metadata']
            
            result = {
                'index': idx,
                'document': document,
                'combined_score': combined_score,
                'bm25_score': bm25_scores.get(idx, 0.0),
                'dense_score': dense_scores.get(idx, 0.0),
                'metadata': metadata
            }
            results.append(result)
        
        logger.debug(f"Hybrid search returned {len(results)} results for query: {query[:50]}...")
        return results
    
    def _normalize_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
        """Normalize scores to [0, 1] range"""
        if not scores:
            return scores
        
        min_score = min(scores.values())
        max_score = max(scores.values())
        
        if max_score == min_score:
            return {idx: 1.0 for idx in scores.keys()}
        
        normalized = {}
        for idx, score in scores.items():
            normalized[idx] = (score - min_score) / (max_score - min_score)
        
        return normalized
    
    def get_retrieval_scores(self, query: str) -> Dict[str, Any]:
        """Get detailed retrieval scores for analysis"""
        if not self.is_fitted:
            raise ValueError("Hybrid retriever must be fitted before getting scores")
        
        # Get BM25 scores
        bm25_scores = self.bm25_retriever.get_document_scores(query)
        
        # Get dense scores
        dense_scores = self.dense_retriever.get_similarity_scores(query)
        
        # Combine scores
        combined_scores = (self.bm25_weight * bm25_scores + 
                          self.dense_weight * dense_scores)
        
        return {
            'bm25_scores': bm25_scores.tolist(),
            'dense_scores': dense_scores.tolist(),
            'combined_scores': combined_scores.tolist(),
            'bm25_weight': self.bm25_weight,
            'dense_weight': self.dense_weight
        }
    
    def add_documents(self, new_documents: List[str], 
                     new_metadata: List[Dict[str, Any]] = None,
                     batch_size: int = 32):
        """Add new documents to both retrievers"""
        if not self.is_fitted:
            raise ValueError("Hybrid retriever must be fitted before adding documents")
        
        logger.info(f"Adding {len(new_documents)} new documents to hybrid retriever")
        
        # Add to BM25 retriever
        self.bm25_retriever.add_documents(new_documents, new_metadata)
        
        # Add to dense retriever
        self.dense_retriever.add_documents(new_documents, new_metadata, batch_size)
        
        logger.info("Documents added to hybrid retriever successfully")
    
    def remove_documents(self, indices: List[int]):
        """Remove documents from both retrievers"""
        if not self.is_fitted:
            raise ValueError("Hybrid retriever must be fitted before removing documents")
        
        logger.info(f"Removing {len(indices)} documents from hybrid retriever")
        
        # Remove from BM25 retriever
        self.bm25_retriever.remove_documents(indices)
        
        # Remove from dense retriever
        self.dense_retriever.remove_documents(indices)
        
        logger.info("Documents removed from hybrid retriever successfully")
    
    def save_model(self, filepath: str):
        """Save hybrid retriever model"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        # Save BM25 model
        bm25_file = filepath.replace('.pkl', '_bm25.pkl')
        self.bm25_retriever.save_model(bm25_file)
        
        # Save dense model
        dense_file = filepath.replace('.pkl', '_dense.pkl')
        self.dense_retriever.save_model(dense_file)
        
        # Save hybrid configuration
        hybrid_config = {
            'bm25_weight': self.bm25_weight,
            'dense_weight': self.dense_weight
        }
        
        config_file = filepath.replace('.pkl', '_config.pkl')
        import pickle
        with open(config_file, 'wb') as f:
            pickle.dump(hybrid_config, f)
        
        logger.info(f"Hybrid retriever model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load hybrid retriever model"""
        # Load BM25 model
        bm25_file = filepath.replace('.pkl', '_bm25.pkl')
        self.bm25_retriever.load_model(bm25_file)
        
        # Load dense model
        dense_file = filepath.replace('.pkl', '_dense.pkl')
        self.dense_retriever.load_model(dense_file)
        
        # Load hybrid configuration
        config_file = filepath.replace('.pkl', '_config.pkl')
        import pickle
        with open(config_file, 'rb') as f:
            hybrid_config = pickle.load(f)
        
        self.bm25_weight = hybrid_config['bm25_weight']
        self.dense_weight = hybrid_config['dense_weight']
        self.is_fitted = True
        
        logger.info(f"Hybrid retriever model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the hybrid retriever"""
        return {
            'is_fitted': self.is_fitted,
            'bm25_weight': self.bm25_weight,
            'dense_weight': self.dense_weight,
            'bm25_info': self.bm25_retriever.get_model_info(),
            'dense_info': self.dense_retriever.get_model_info()
        }
    
    def explain_retrieval(self, query: str, doc_index: int) -> Dict[str, Any]:
        """Explain why a document was retrieved"""
        if not self.is_fitted:
            raise ValueError("Hybrid retriever must be fitted before explaining retrieval")
        
        # Get BM25 explanation
        bm25_explanation = self.bm25_retriever.explain_score(query, doc_index)
        
        # Get dense explanation
        dense_explanation = self.dense_retriever.explain_similarity(query, doc_index)
        
        # Calculate combined score
        bm25_score = bm25_explanation['total_score']
        dense_score = dense_explanation['similarity']
        combined_score = (self.bm25_weight * bm25_score + 
                         self.dense_weight * dense_score)
        
        return {
            'query': query,
            'document_index': doc_index,
            'combined_score': combined_score,
            'bm25_explanation': bm25_explanation,
            'dense_explanation': dense_explanation,
            'bm25_weight': self.bm25_weight,
            'dense_weight': self.dense_weight
        }