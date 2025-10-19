"""
Answer caching system for similar questions
"""
import json
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path
import pickle
from loguru import logger
import time


class AnswerCache:
    """Cache system for storing and retrieving answers to similar questions"""
    
    def __init__(self, cache_file: str = "answer_cache.pkl", max_size: int = 10000):
        self.cache_file = Path(cache_file)
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        self.load_cache()
    
    def _generate_key(self, question: str, context: str = "") -> str:
        """Generate a unique key for a question"""
        # Normalize question
        normalized_question = question.lower().strip()
        normalized_context = context.lower().strip() if context else ""
        
        # Create hash
        content = f"{normalized_question}|{normalized_context}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _generate_similarity_key(self, question: str, similarity_threshold: float = 0.8) -> str:
        """Generate a key for similarity-based caching"""
        # Use a more lenient key for similarity matching
        normalized = question.lower().strip()
        # Remove common words that don't affect meaning
        common_words = {'what', 'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [w for w in normalized.split() if w not in common_words]
        content = " ".join(sorted(words))
        return hashlib.md5(content.encode()).hexdigest()
    
    def store_answer(self, question: str, answer: str, 
                    context: str = "", metadata: Dict[str, Any] = None) -> str:
        """Store an answer in the cache"""
        key = self._generate_key(question, context)
        
        cache_entry = {
            'question': question,
            'answer': answer,
            'context': context,
            'metadata': metadata or {},
            'timestamp': time.time(),
            'access_count': 0
        }
        
        self.cache[key] = cache_entry
        self.access_times[key] = time.time()
        
        # Check cache size
        if len(self.cache) > self.max_size:
            self._evict_oldest()
        
        logger.debug(f"Stored answer for question: {question[:50]}...")
        return key
    
    def get_answer(self, question: str, context: str = "") -> Optional[Dict[str, Any]]:
        """Get answer from cache"""
        key = self._generate_key(question, context)
        
        if key in self.cache:
            # Update access info
            self.cache[key]['access_count'] += 1
            self.access_times[key] = time.time()
            
            logger.debug(f"Cache hit for question: {question[:50]}...")
            return self.cache[key]
        
        logger.debug(f"Cache miss for question: {question[:50]}...")
        return None
    
    def find_similar_answer(self, question: str, 
                          similarity_threshold: float = 0.8) -> Optional[Dict[str, Any]]:
        """Find similar answer using similarity key"""
        sim_key = self._generate_similarity_key(question, similarity_threshold)
        
        # Look for entries with similar keys (simplified approach)
        for key, entry in self.cache.items():
            entry_sim_key = self._generate_similarity_key(entry['question'], similarity_threshold)
            if sim_key == entry_sim_key:
                # Update access info
                entry['access_count'] += 1
                self.access_times[key] = time.time()
                
                logger.debug(f"Found similar answer for question: {question[:50]}...")
                return entry
        
        # If no exact similarity match, try fuzzy matching
        question_words = set(question.lower().split())
        best_match = None
        best_score = 0.0
        
        for key, entry in self.cache.items():
            entry_words = set(entry['question'].lower().split())
            
            # Calculate word overlap
            intersection = len(question_words.intersection(entry_words))
            union = len(question_words.union(entry_words))
            
            if union > 0:
                similarity_score = intersection / union
                if similarity_score > best_score and similarity_score >= 0.6:  # Lower threshold
                    best_score = similarity_score
                    best_match = entry
        
        if best_match:
            # Update access info
            best_match['access_count'] += 1
            self.access_times[key] = time.time()
            
            logger.debug(f"Found fuzzy similar answer for question: {question[:50]}... (score: {best_score:.2f})")
            return best_match
        
        return None
    
    def get_cached_answers(self, question: str, 
                          similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Get all cached answers that might be relevant"""
        results = []
        
        # First try exact match
        exact_match = self.get_answer(question)
        if exact_match:
            results.append(exact_match)
        
        # Then try similarity match
        similar_match = self.find_similar_answer(question, similarity_threshold)
        if similar_match and similar_match not in results:
            results.append(similar_match)
        
        return results
    
    def _evict_oldest(self):
        """Evict oldest entries when cache is full"""
        if not self.access_times:
            return
        
        # Sort by access time (oldest first)
        sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
        
        # Remove oldest 10% of entries
        num_to_remove = max(1, len(sorted_items) // 10)
        
        for key, _ in sorted_items[:num_to_remove]:
            if key in self.cache:
                del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
        
        logger.info(f"Evicted {num_to_remove} entries from cache")
    
    def save_cache(self):
        """Save cache to disk"""
        try:
            # Ensure directory exists
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            cache_data = {
                'cache': self.cache,
                'access_times': self.access_times,
                'max_size': self.max_size
            }
            
            # Create temporary file first, then rename for atomic operation
            temp_file = self.cache_file.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # Atomic rename
            temp_file.replace(self.cache_file)
            
            logger.info(f"Cache saved to {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            # Try to clean up temp file if it exists
            temp_file = self.cache_file.with_suffix('.tmp')
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except:
                    pass
    
    def load_cache(self):
        """Load cache from disk"""
        if not self.cache_file.exists():
            logger.info("No existing cache file found")
            return
        
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.cache = cache_data.get('cache', {})
            self.access_times = cache_data.get('access_times', {})
            self.max_size = cache_data.get('max_size', self.max_size)
            
            logger.info(f"Loaded cache with {len(self.cache)} entries")
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            self.cache = {}
            self.access_times = {}
    
    def clear_cache(self):
        """Clear all cached entries"""
        self.cache.clear()
        self.access_times.clear()
        logger.info("Cache cleared")
    
    def force_save_cache(self):
        """Force save cache and ensure file is created"""
        try:
            self.save_cache()
            # Verify file was created
            if self.cache_file.exists():
                logger.info(f"Cache file verified: {self.cache_file}")
                return True
            else:
                logger.error("Cache file was not created after save")
                return False
        except Exception as e:
            logger.error(f"Failed to force save cache: {e}")
            return False
    
    def remove_answer(self, question: str, context: str = "") -> bool:
        """Remove a specific answer from cache"""
        key = self._generate_key(question, context)
        
        if key in self.cache:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
            logger.info(f"Removed answer for question: {question[:50]}...")
            return True
        
        logger.warning(f"No answer found to remove for question: {question[:50]}...")
        return False
    
    def remove_similar_answers(self, question: str, similarity_threshold: float = 0.8) -> int:
        """Remove all similar answers from cache"""
        sim_key = self._generate_similarity_key(question, similarity_threshold)
        removed_count = 0
        keys_to_remove = []
        
        # Find all similar entries
        for key, entry in self.cache.items():
            entry_sim_key = self._generate_similarity_key(entry['question'], similarity_threshold)
            if sim_key == entry_sim_key:
                keys_to_remove.append(key)
        
        # Remove found entries
        for key in keys_to_remove:
            if key in self.cache:
                del self.cache[key]
                removed_count += 1
            if key in self.access_times:
                del self.access_times[key]
        
        logger.info(f"Removed {removed_count} similar answers for question: {question[:50]}...")
        return removed_count
    
    def remove_old_entries(self, max_age_hours: int = 24) -> int:
        """Remove entries older than specified hours"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        removed_count = 0
        keys_to_remove = []
        
        for key, entry in self.cache.items():
            age = current_time - entry['timestamp']
            if age > max_age_seconds:
                keys_to_remove.append(key)
        
        # Remove old entries
        for key in keys_to_remove:
            if key in self.cache:
                del self.cache[key]
                removed_count += 1
            if key in self.access_times:
                del self.access_times[key]
        
        logger.info(f"Removed {removed_count} old entries (older than {max_age_hours} hours)")
        return removed_count
    
    def remove_low_access_entries(self, min_access_count: int = 1) -> int:
        """Remove entries with access count below threshold"""
        removed_count = 0
        keys_to_remove = []
        
        for key, entry in self.cache.items():
            if entry['access_count'] < min_access_count:
                keys_to_remove.append(key)
        
        # Remove low access entries
        for key in keys_to_remove:
            if key in self.cache:
                del self.cache[key]
                removed_count += 1
            if key in self.access_times:
                del self.access_times[key]
        
        logger.info(f"Removed {removed_count} low access entries (access count < {min_access_count})")
        return removed_count
    
    def remove_entries_by_pattern(self, pattern: str, search_in: str = "question") -> int:
        """Remove entries matching a pattern in specified field"""
        import re
        removed_count = 0
        keys_to_remove = []
        
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            logger.error(f"Invalid regex pattern: {pattern}")
            return 0
        
        for key, entry in self.cache.items():
            if search_in in entry and regex.search(str(entry[search_in])):
                keys_to_remove.append(key)
        
        # Remove matching entries
        for key in keys_to_remove:
            if key in self.cache:
                del self.cache[key]
                removed_count += 1
            if key in self.access_times:
                del self.access_times[key]
        
        logger.info(f"Removed {removed_count} entries matching pattern '{pattern}' in {search_in}")
        return removed_count
    
    def cleanup_cache(self, max_age_hours: int = 24, min_access_count: int = 1) -> Dict[str, int]:
        """Comprehensive cache cleanup"""
        logger.info("Starting comprehensive cache cleanup...")
        
        old_removed = self.remove_old_entries(max_age_hours)
        low_access_removed = self.remove_low_access_entries(min_access_count)
        
        # Save cleaned cache
        self.save_cache()
        
        cleanup_stats = {
            'old_entries_removed': old_removed,
            'low_access_entries_removed': low_access_removed,
            'total_removed': old_removed + low_access_removed,
            'remaining_entries': len(self.cache)
        }
        
        logger.info(f"Cache cleanup completed: {cleanup_stats}")
        return cleanup_stats
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.cache:
            return {
                'total_entries': 0,
                'cache_size_mb': 0,
                'avg_access_count': 0,
                'oldest_entry': None,
                'newest_entry': None
            }
        
        total_entries = len(self.cache)
        access_counts = [entry['access_count'] for entry in self.cache.values()]
        timestamps = [entry['timestamp'] for entry in self.cache.values()]
        
        # Estimate cache size
        cache_size_bytes = len(pickle.dumps(self.cache))
        cache_size_mb = cache_size_bytes / (1024 * 1024)
        
        return {
            'total_entries': total_entries,
            'cache_size_mb': round(cache_size_mb, 2),
            'avg_access_count': sum(access_counts) / len(access_counts) if access_counts else 0,
            'oldest_entry': min(timestamps) if timestamps else None,
            'newest_entry': max(timestamps) if timestamps else None
        }
    
    def search_cache(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search cache for entries containing query"""
        results = []
        query_lower = query.lower()
        
        for entry in self.cache.values():
            if (query_lower in entry['question'].lower() or 
                query_lower in entry['answer'].lower() or
                query_lower in entry['context'].lower()):
                results.append(entry)
        
        # Sort by access count (most accessed first)
        results.sort(key=lambda x: x['access_count'], reverse=True)
        
        return results[:limit]