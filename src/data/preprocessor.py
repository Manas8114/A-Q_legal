"""
Legal Text Preprocessor for normalization and tokenization
"""
import re
import string
from typing import List, Dict, Any, Optional
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from loguru import logger


class LegalTextPreprocessor:
    """Preprocesses legal text for better model performance"""
    
    def __init__(self, language: str = "en"):
        self.language = language
        self.nlp = None
        self.stop_words = set()
        self._setup_models()
    
    def _setup_models(self):
        """Setup spaCy and NLTK models"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.warning(f"Could not download NLTK data: {e}")
            self.stop_words = set()
    
    def normalize_text(self, text: str) -> str:
        """Normalize text by removing extra whitespace, special chars, etc."""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep legal references
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\(\)\[\]\{\}\-\'\"\/]', '', text)
        
        # Normalize legal references (e.g., "Article 14" -> "Article 14")
        text = re.sub(r'\b(article|section|clause|rule|regulation)\s+(\d+)', 
                     lambda m: f"{m.group(1).capitalize()} {m.group(2)}", 
                     text, flags=re.IGNORECASE)
        
        # Remove multiple periods
        text = re.sub(r'\.{2,}', '.', text)
        
        return text.strip()
    
    def tokenize_text(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """Tokenize text into words"""
        if not text:
            return []
        
        # Use spaCy if available, otherwise NLTK
        if self.nlp:
            doc = self.nlp(text)
            tokens = [token.text.lower() for token in doc if not token.is_space]
        else:
            tokens = word_tokenize(text.lower())
        
        # Remove stopwords if requested
        if remove_stopwords and self.stop_words:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Remove punctuation-only tokens
        tokens = [token for token in tokens if token not in string.punctuation]
        
        return tokens
    
    def extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract legal entities from text"""
        if not self.nlp or not text:
            return {'articles': [], 'sections': [], 'acts': [], 'cases': []}
        
        doc = self.nlp(text)
        entities = {
            'articles': [],
            'sections': [],
            'acts': [],
            'cases': []
        }
        
        # Extract articles
        article_pattern = r'\b(?:article|art\.?)\s+(\d+[a-z]?)'
        articles = re.findall(article_pattern, text, re.IGNORECASE)
        entities['articles'] = articles
        
        # Extract sections
        section_pattern = r'\b(?:section|sec\.?)\s+(\d+[a-z]?)'
        sections = re.findall(section_pattern, text, re.IGNORECASE)
        entities['sections'] = sections
        
        # Extract acts (simplified)
        act_pattern = r'\b(?:act|code|law)\s+of\s+(\d{4})'
        acts = re.findall(act_pattern, text, re.IGNORECASE)
        entities['acts'] = acts
        
        # Extract case names (simplified)
        case_pattern = r'\b[A-Z][a-z]+\s+v\.?\s+[A-Z][a-z]+'
        cases = re.findall(case_pattern, text)
        entities['cases'] = cases
        
        return entities
    
    def preprocess_question(self, question: str) -> Dict[str, Any]:
        """Preprocess a legal question"""
        normalized = self.normalize_text(question)
        tokens = self.tokenize_text(normalized)
        entities = self.extract_legal_entities(question)
        
        return {
            'original': question,
            'normalized': normalized,
            'tokens': tokens,
            'entities': entities,
            'length': len(tokens),
            'word_count': len(question.split())
        }
    
    def preprocess_answer(self, answer: str) -> Dict[str, Any]:
        """Preprocess a legal answer"""
        normalized = self.normalize_text(answer)
        tokens = self.tokenize_text(normalized, remove_stopwords=False)
        entities = self.extract_legal_entities(answer)
        
        return {
            'original': answer,
            'normalized': normalized,
            'tokens': tokens,
            'entities': entities,
            'length': len(tokens),
            'word_count': len(answer.split())
        }
    
    def preprocess_context(self, context: str) -> Dict[str, Any]:
        """Preprocess legal context/passage"""
        normalized = self.normalize_text(context)
        tokens = self.tokenize_text(normalized)
        entities = self.extract_legal_entities(context)
        
        return {
            'original': context,
            'normalized': normalized,
            'tokens': tokens,
            'entities': entities,
            'length': len(tokens),
            'word_count': len(context.split())
        }
    
    def batch_preprocess(self, texts: List[str], text_type: str = "question") -> List[Dict[str, Any]]:
        """Preprocess a batch of texts"""
        results = []
        
        for text in texts:
            if text_type == "question":
                result = self.preprocess_question(text)
            elif text_type == "answer":
                result = self.preprocess_answer(text)
            elif text_type == "context":
                result = self.preprocess_context(text)
            else:
                result = self.preprocess_question(text)  # Default to question
            
            results.append(result)
        
        return results
    
    def create_vocabulary(self, texts: List[str]) -> Dict[str, int]:
        """Create vocabulary from preprocessed texts"""
        vocab = {}
        
        for text in texts:
            preprocessed = self.preprocess_question(text)
            for token in preprocessed['tokens']:
                vocab[token] = vocab.get(token, 0) + 1
        
        # Sort by frequency
        vocab = dict(sorted(vocab.items(), key=lambda x: x[1], reverse=True))
        
        return vocab