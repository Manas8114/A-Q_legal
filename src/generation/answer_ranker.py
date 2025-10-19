"""
Answer ranking and scoring system
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger


class AnswerRanker:
    """Rank and score answers using multiple criteria"""
    
    def __init__(self, 
                 bayesian_weight: float = 0.3,
                 similarity_weight: float = 0.4,
                 confidence_weight: float = 0.3):
        self.bayesian_weight = bayesian_weight
        self.similarity_weight = similarity_weight
        self.confidence_weight = confidence_weight
    
    def rank_answers(self, 
                    answers: List[Dict[str, Any]],
                    question: str,
                    question_embedding: np.ndarray = None,
                    bayesian_probability: float = None) -> List[Dict[str, Any]]:
        """Rank answers based on multiple criteria"""
        if not answers:
            return []
        
        ranked_answers = []
        
        for answer in answers:
            # Calculate individual scores
            bayesian_score = self._calculate_bayesian_score(answer, bayesian_probability)
            similarity_score = self._calculate_similarity_score(answer, question, question_embedding)
            confidence_score = self._calculate_confidence_score(answer)
            
            # Calculate combined score
            combined_score = (
                self.bayesian_weight * bayesian_score +
                self.similarity_weight * similarity_score +
                self.confidence_weight * confidence_score
            )
            
            # Add scores to answer
            ranked_answer = answer.copy()
            ranked_answer.update({
                'bayesian_score': bayesian_score,
                'similarity_score': similarity_score,
                'confidence_score': confidence_score,
                'combined_score': combined_score,
                'ranking_criteria': {
                    'bayesian_weight': self.bayesian_weight,
                    'similarity_weight': self.similarity_weight,
                    'confidence_weight': self.confidence_weight
                }
            })
            
            ranked_answers.append(ranked_answer)
        
        # Sort by combined score
        ranked_answers.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Add rank
        for i, answer in enumerate(ranked_answers):
            answer['rank'] = i + 1
        
        return ranked_answers
    
    def _calculate_bayesian_score(self, answer: Dict[str, Any], 
                                 bayesian_probability: float = None) -> float:
        """Calculate Bayesian-based score"""
        if bayesian_probability is not None:
            return float(bayesian_probability)
        
        # Use answer confidence if available
        if 'confidence' in answer:
            return float(answer['confidence'])
        
        # Default score
        return 0.5
    
    def _calculate_similarity_score(self, answer: Dict[str, Any], 
                                   question: str,
                                   question_embedding: np.ndarray = None) -> float:
        """Calculate similarity-based score"""
        if 'similarity' in answer:
            return float(answer['similarity'])
        
        # If we have embeddings, calculate cosine similarity
        if question_embedding is not None and 'answer_embedding' in answer:
            answer_embedding = answer['answer_embedding']
            if answer_embedding is not None:
                similarity = cosine_similarity([question_embedding], [answer_embedding])[0][0]
                return float(similarity)
        
        # Simple text similarity (can be improved)
        if 'answer' in answer:
            return self._text_similarity(question, answer['answer'])
        
        return 0.5
    
    def _calculate_confidence_score(self, answer: Dict[str, Any]) -> float:
        """Calculate confidence-based score with improved quality assessment"""
        if 'confidence' in answer:
            return float(answer['confidence'])
        
        # Use extractive confidence if available
        if 'extractive_confidence' in answer:
            return float(answer['extractive_confidence'])
        
        # Use generative confidence if available
        if 'generative_confidence' in answer:
            return float(answer['generative_confidence'])
        
        # Calculate confidence based on answer quality indicators
        confidence = 0.4  # Lower base confidence for more realistic scoring
        
        if 'answer' in answer:
            answer_text = answer['answer']
            answer_length = len(answer_text)
            
            # Length-based confidence with better scaling
            if answer_length > 20:
                length_bonus = min(0.3, (answer_length - 20) / 300)  # Up to 0.3 bonus
                confidence += length_bonus
            
            # Quality indicators
            # Check for complete sentences
            sentence_count = answer_text.count('.') + answer_text.count('!') + answer_text.count('?')
            if sentence_count > 0:
                confidence += min(0.15, sentence_count * 0.03)  # Up to 0.15 bonus
            
            # Check for proper capitalization and structure
            if answer_text and answer_text[0].isupper() and answer_text.count(' ') > 2:
                confidence += 0.05
            
            # Penalty for very short or incomplete answers
            if answer_length < 10:
                confidence *= 0.7
            elif answer_length < 20:
                confidence *= 0.85
        
        # Boost confidence for answers with context
        if 'context' in answer and answer['context']:
            context_length = len(answer['context'])
            if context_length > 50:
                confidence += 0.1
            else:
                confidence += 0.05
        
        # Boost confidence for answers with metadata
        if 'metadata' in answer and answer['metadata']:
            confidence += 0.05
        
        # Source-based confidence adjustment
        if 'source' in answer:
            if answer['source'] == 'extractive':
                confidence *= 1.1  # Slight boost for extractive answers
            elif answer['source'] == 'generative':
                confidence *= 1.05  # Small boost for generative answers
        
        # Ensure confidence is within bounds
        return min(max(confidence, 0.1), 0.95)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        if not text1 or not text2:
            return 0.0
        
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def select_best_answer(self, ranked_answers: List[Dict[str, Any]], 
                          threshold: float = 0.5) -> Optional[Dict[str, Any]]:
        """Select the best answer based on threshold"""
        if not ranked_answers:
            return None
        
        best_answer = ranked_answers[0]
        
        # Check if best answer meets threshold
        if best_answer['combined_score'] >= threshold:
            return best_answer
        
        # If no answer meets threshold, return the best available if it's reasonable
        if best_answer['combined_score'] > 0.2:
            return best_answer
        
        # Return None only if all answers are very poor
        return None
    
    def get_answer_explanation(self, answer: Dict[str, Any]) -> Dict[str, Any]:
        """Get explanation for why an answer was ranked highly"""
        explanation = {
            'answer': answer.get('answer', ''),
            'combined_score': answer.get('combined_score', 0.0),
            'breakdown': {
                'bayesian_score': answer.get('bayesian_score', 0.0),
                'similarity_score': answer.get('similarity_score', 0.0),
                'confidence_score': answer.get('confidence_score', 0.0)
            },
            'weights': answer.get('ranking_criteria', {}),
            'rank': answer.get('rank', 0)
        }
        
        # Add reasoning
        if answer.get('bayesian_score', 0) > 0.7:
            explanation['reasoning'] = "High Bayesian probability indicates good category match"
        elif answer.get('similarity_score', 0) > 0.7:
            explanation['reasoning'] = "High similarity score indicates good semantic match"
        elif answer.get('confidence_score', 0) > 0.7:
            explanation['reasoning'] = "High confidence score indicates model certainty"
        else:
            explanation['reasoning'] = "Moderate scores across all criteria"
        
        return explanation
    
    def compare_answers(self, answers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple answers and provide analysis"""
        if len(answers) < 2:
            return {'message': 'Need at least 2 answers to compare'}
        
        # Calculate statistics
        scores = [answer.get('combined_score', 0) for answer in answers]
        
        comparison = {
            'num_answers': len(answers),
            'score_statistics': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores))
            },
            'score_distribution': {
                'high_confidence': len([s for s in scores if s >= 0.8]),
                'medium_confidence': len([s for s in scores if 0.5 <= s < 0.8]),
                'low_confidence': len([s for s in scores if s < 0.5])
            },
            'top_answer': answers[0] if answers else None,
            'score_gap': float(scores[0] - scores[1]) if len(scores) > 1 else 0.0
        }
        
        return comparison
    
    def filter_answers_by_confidence(self, answers: List[Dict[str, Any]], 
                                   min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """Filter answers by minimum confidence threshold"""
        filtered = []
        
        for answer in answers:
            if answer.get('combined_score', 0) >= min_confidence:
                filtered.append(answer)
        
        return filtered
    
    def get_ranking_summary(self, ranked_answers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary of ranking results"""
        if not ranked_answers:
            return {'message': 'No answers to rank'}
        
        summary = {
            'total_answers': len(ranked_answers),
            'top_score': ranked_answers[0].get('combined_score', 0),
            'score_range': {
                'min': min(answer.get('combined_score', 0) for answer in ranked_answers),
                'max': max(answer.get('combined_score', 0) for answer in ranked_answers)
            },
            'high_confidence_count': len([a for a in ranked_answers if a.get('combined_score', 0) >= 0.8]),
            'medium_confidence_count': len([a for a in ranked_answers if 0.5 <= a.get('combined_score', 0) < 0.8]),
            'low_confidence_count': len([a for a in ranked_answers if a.get('combined_score', 0) < 0.5])
        }
        
        return summary