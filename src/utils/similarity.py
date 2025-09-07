"""
Similarity computation utilities including Tanimoto coefficient
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger


class TanimotoSimilarity:
    """Compute Tanimoto coefficient for similarity between vectors"""
    
    @staticmethod
    def compute_tanimoto(vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Compute Tanimoto coefficient between two vectors"""
        if len(vector1) != len(vector2):
            raise ValueError("Vectors must have the same length")
        
        # Convert to binary if not already
        v1_binary = (vector1 > 0).astype(int)
        v2_binary = (vector2 > 0).astype(int)
        
        # Compute intersection and union
        intersection = np.sum(v1_binary & v2_binary)
        union = np.sum(v1_binary | v2_binary)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    @staticmethod
    def compute_tanimoto_batch(query_vector: np.ndarray, 
                              candidate_vectors: np.ndarray) -> np.ndarray:
        """Compute Tanimoto coefficients for query against multiple candidates"""
        similarities = []
        for candidate in candidate_vectors:
            sim = TanimotoSimilarity.compute_tanimoto(query_vector, candidate)
            similarities.append(sim)
        return np.array(similarities)
    
    @staticmethod
    def find_most_similar_tanimoto(query_vector: np.ndarray,
                                 candidate_vectors: np.ndarray,
                                 top_k: int = 5) -> List[Dict[str, Any]]:
        """Find most similar vectors using Tanimoto coefficient"""
        similarities = TanimotoSimilarity.compute_tanimoto_batch(
            query_vector, candidate_vectors
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'index': int(idx),
                'similarity': float(similarities[idx])
            })
        
        return results


class SimilarityChecker:
    """Comprehensive similarity checking using multiple methods"""
    
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.similarity_cache = {}
    
    def check_embedding_similarity(self, query_embedding: np.ndarray,
                                 candidate_embeddings: np.ndarray,
                                 method: str = 'cosine') -> List[Dict[str, Any]]:
        """Check similarity using different methods"""
        if method == 'cosine':
            similarities = cosine_similarity([query_embedding], candidate_embeddings)[0]
        elif method == 'tanimoto':
            similarities = TanimotoSimilarity.compute_tanimoto_batch(
                query_embedding, candidate_embeddings
            )
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
        results = []
        for i, sim in enumerate(similarities):
            results.append({
                'index': i,
                'similarity': float(sim),
                'is_similar': sim >= self.threshold
            })
        
        return results
    
    def find_duplicates(self, embeddings: np.ndarray, 
                       method: str = 'cosine') -> List[Tuple[int, int, float]]:
        """Find duplicate pairs in a set of embeddings"""
        duplicates = []
        n = len(embeddings)
        
        for i in range(n):
            for j in range(i + 1, n):
                if method == 'cosine':
                    sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                elif method == 'tanimoto':
                    sim = TanimotoSimilarity.compute_tanimoto(embeddings[i], embeddings[j])
                else:
                    raise ValueError(f"Unknown similarity method: {method}")
                
                if sim >= self.threshold:
                    duplicates.append((i, j, sim))
        
        return duplicates
    
    def get_similarity_explanation(self, query_embedding: np.ndarray,
                                 candidate_embedding: np.ndarray) -> Dict[str, Any]:
        """Get detailed explanation of similarity between two embeddings"""
        cosine_sim = cosine_similarity([query_embedding], [candidate_embedding])[0][0]
        tanimoto_sim = TanimotoSimilarity.compute_tanimoto(query_embedding, candidate_embedding)
        
        # Compute additional metrics
        euclidean_dist = np.linalg.norm(query_embedding - candidate_embedding)
        manhattan_dist = np.sum(np.abs(query_embedding - candidate_embedding))
        
        return {
            'cosine_similarity': float(cosine_sim),
            'tanimoto_coefficient': float(tanimoto_sim),
            'euclidean_distance': float(euclidean_dist),
            'manhattan_distance': float(manhattan_dist),
            'is_similar_cosine': cosine_sim >= self.threshold,
            'is_similar_tanimoto': tanimoto_sim >= self.threshold
        }


class NearDuplicateDetector:
    """Detect near-duplicate questions using multiple similarity methods"""
    
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.similarity_checker = SimilarityChecker(threshold)
    
    def detect_duplicates(self, question_embeddings: np.ndarray,
                         question_texts: List[str]) -> List[Dict[str, Any]]:
        """Detect near-duplicate questions"""
        logger.info(f"Detecting duplicates among {len(question_embeddings)} questions")
        
        duplicates = []
        n = len(question_embeddings)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Check cosine similarity
                cosine_sim = cosine_similarity(
                    [question_embeddings[i]], 
                    [question_embeddings[j]]
                )[0][0]
                
                # Check Tanimoto coefficient
                tanimoto_sim = TanimotoSimilarity.compute_tanimoto(
                    question_embeddings[i], 
                    question_embeddings[j]
                )
                
                # Consider duplicate if either method exceeds threshold
                if cosine_sim >= self.threshold or tanimoto_sim >= self.threshold:
                    duplicates.append({
                        'index1': i,
                        'index2': j,
                        'question1': question_texts[i],
                        'question2': question_texts[j],
                        'cosine_similarity': float(cosine_sim),
                        'tanimoto_coefficient': float(tanimoto_sim),
                        'max_similarity': max(cosine_sim, tanimoto_sim)
                    })
        
        logger.info(f"Found {len(duplicates)} duplicate pairs")
        return duplicates
    
    def find_most_similar_question(self, query_embedding: np.ndarray,
                                 question_embeddings: np.ndarray,
                                 question_texts: List[str],
                                 top_k: int = 5) -> List[Dict[str, Any]]:
        """Find most similar questions to a query"""
        # Compute similarities
        cosine_sims = cosine_similarity([query_embedding], question_embeddings)[0]
        tanimoto_sims = TanimotoSimilarity.compute_tanimoto_batch(
            query_embedding, question_embeddings
        )
        
        # Combine similarities (weighted average)
        combined_sims = 0.7 * cosine_sims + 0.3 * tanimoto_sims
        
        # Get top-k
        top_indices = np.argsort(combined_sims)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'index': int(idx),
                'question': question_texts[idx],
                'cosine_similarity': float(cosine_sims[idx]),
                'tanimoto_coefficient': float(tanimoto_sims[idx]),
                'combined_similarity': float(combined_sims[idx]),
                'is_duplicate': combined_sims[idx] >= self.threshold
            })
        
        return results