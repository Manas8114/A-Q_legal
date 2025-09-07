"""
Syntactic feature extraction for legal text
"""
import spacy
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from loguru import logger


class SyntacticFeatureExtractor:
    """Extract syntactic features from legal text using spaCy"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self.nlp = None
        self._load_model()
    
    def _load_model(self):
        """Load spaCy model"""
        try:
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
        except OSError:
            logger.warning(f"spaCy model {self.model_name} not found. Install with: python -m spacy download {self.model_name}")
            self.nlp = None
    
    def extract_pos_features(self, text: str) -> Dict[str, int]:
        """Extract Part-of-Speech features"""
        if not self.nlp or not text:
            return {}
        
        doc = self.nlp(text)
        pos_counts = {}
        
        for token in doc:
            if not token.is_space and not token.is_punct:
                pos = token.pos_
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
        
        return pos_counts
    
    def extract_dependency_features(self, text: str) -> Dict[str, int]:
        """Extract dependency parsing features"""
        if not self.nlp or not text:
            return {}
        
        doc = self.nlp(text)
        dep_counts = {}
        
        for token in doc:
            if not token.is_space and not token.is_punct:
                dep = token.dep_
                dep_counts[dep] = dep_counts.get(dep, 0) + 1
        
        return dep_counts
    
    def extract_legal_syntactic_patterns(self, text: str) -> Dict[str, int]:
        """Extract legal-specific syntactic patterns"""
        if not self.nlp or not text:
            return {}
        
        doc = self.nlp(text)
        patterns = {
            'question_words': 0,
            'modal_verbs': 0,
            'passive_voice': 0,
            'conditional_clauses': 0,
            'legal_terms': 0,
            'citations': 0,
            'definitions': 0
        }
        
        # Question words
        question_words = {'what', 'when', 'where', 'who', 'why', 'how', 'which', 'whether'}
        for token in doc:
            if token.text.lower() in question_words:
                patterns['question_words'] += 1
        
        # Modal verbs
        modal_verbs = {'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would'}
        for token in doc:
            if token.text.lower() in modal_verbs:
                patterns['modal_verbs'] += 1
        
        # Passive voice (simplified detection)
        for token in doc:
            if token.dep_ == 'auxpass' or (token.tag_ == 'VBN' and token.dep_ == 'ROOT'):
                patterns['passive_voice'] += 1
        
        # Conditional clauses
        conditional_words = {'if', 'unless', 'provided', 'assuming', 'suppose'}
        for token in doc:
            if token.text.lower() in conditional_words:
                patterns['conditional_clauses'] += 1
        
        # Legal terms (simplified)
        legal_terms = {'act', 'article', 'section', 'clause', 'rule', 'regulation', 
                      'provision', 'statute', 'law', 'code', 'amendment'}
        for token in doc:
            if token.text.lower() in legal_terms:
                patterns['legal_terms'] += 1
        
        # Citations (simplified)
        import re
        citation_pattern = r'\b(?:article|section|clause|rule)\s+\d+'
        patterns['citations'] = len(re.findall(citation_pattern, text, re.IGNORECASE))
        
        # Definitions (simplified)
        definition_words = {'means', 'defined', 'definition', 'refers to', 'is defined as'}
        for token in doc:
            if token.text.lower() in definition_words:
                patterns['definitions'] += 1
        
        return patterns
    
    def extract_sentence_features(self, text: str) -> Dict[str, Any]:
        """Extract sentence-level features"""
        if not self.nlp or not text:
            return {}
        
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        features = {
            'num_sentences': len(sentences),
            'avg_sentence_length': 0,
            'max_sentence_length': 0,
            'min_sentence_length': 0,
            'sentence_lengths': []
        }
        
        if sentences:
            sentence_lengths = [len(sent) for sent in sentences]
            features['avg_sentence_length'] = np.mean(sentence_lengths)
            features['max_sentence_length'] = np.max(sentence_lengths)
            features['min_sentence_length'] = np.min(sentence_lengths)
            features['sentence_lengths'] = sentence_lengths
        
        return features
    
    def extract_entity_features(self, text: str) -> Dict[str, int]:
        """Extract named entity features"""
        if not self.nlp or not text:
            return {}
        
        doc = self.nlp(text)
        entity_counts = {}
        
        for ent in doc.ents:
            entity_type = ent.label_
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        return entity_counts
    
    def extract_all_features(self, text: str) -> Dict[str, Any]:
        """Extract all syntactic features"""
        if not text:
            return {}
        
        features = {}
        
        # POS features
        features['pos'] = self.extract_pos_features(text)
        
        # Dependency features
        features['dependencies'] = self.extract_dependency_features(text)
        
        # Legal patterns
        features['legal_patterns'] = self.extract_legal_syntactic_patterns(text)
        
        # Sentence features
        features['sentences'] = self.extract_sentence_features(text)
        
        # Entity features
        features['entities'] = self.extract_entity_features(text)
        
        return features
    
    def create_feature_vector(self, text: str, feature_names: List[str] = None) -> np.ndarray:
        """Create a feature vector from syntactic features"""
        features = self.extract_all_features(text)
        
        if feature_names is None:
            # Create default feature names
            feature_names = self._get_default_feature_names()
        
        vector = []
        for feature_name in feature_names:
            value = self._get_feature_value(features, feature_name)
            vector.append(value)
        
        return np.array(vector)
    
    def _get_default_feature_names(self) -> List[str]:
        """Get default feature names for vectorization"""
        # POS tags
        pos_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 'INTJ', 'NOUN', 
                   'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
        
        # Dependencies
        deps = ['ROOT', 'acl', 'acomp', 'advcl', 'advmod', 'agent', 'amod', 'appos',
               'attr', 'aux', 'auxpass', 'case', 'cc', 'ccomp', 'compound', 'conj',
               'cop', 'csubj', 'csubjpass', 'dative', 'dep', 'det', 'dobj', 'expl',
               'intj', 'mark', 'meta', 'neg', 'nmod', 'npadvmod', 'nsubj', 'nsubjpass',
               'nummod', 'oprd', 'parataxis', 'pcomp', 'pobj', 'poss', 'preconj',
               'predet', 'prep', 'prt', 'punct', 'quantmod', 'relcl', 'xcomp']
        
        # Legal patterns
        legal_patterns = ['question_words', 'modal_verbs', 'passive_voice', 
                         'conditional_clauses', 'legal_terms', 'citations', 'definitions']
        
        # Sentence features
        sentence_features = ['num_sentences', 'avg_sentence_length', 'max_sentence_length', 'min_sentence_length']
        
        # Entity types
        entity_types = ['PERSON', 'ORG', 'GPE', 'LAW', 'DATE', 'TIME', 'MONEY', 'PERCENT']
        
        feature_names = []
        feature_names.extend([f"pos_{tag}" for tag in pos_tags])
        feature_names.extend([f"dep_{dep}" for dep in deps])
        feature_names.extend([f"legal_{pattern}" for pattern in legal_patterns])
        feature_names.extend([f"sent_{feat}" for feat in sentence_features])
        feature_names.extend([f"ent_{etype}" for etype in entity_types])
        
        return feature_names
    
    def _get_feature_value(self, features: Dict[str, Any], feature_name: str) -> float:
        """Get value for a specific feature name"""
        if feature_name.startswith('pos_'):
            pos_tag = feature_name[4:]
            return features.get('pos', {}).get(pos_tag, 0)
        elif feature_name.startswith('dep_'):
            dep = feature_name[4:]
            return features.get('dependencies', {}).get(dep, 0)
        elif feature_name.startswith('legal_'):
            pattern = feature_name[6:]
            return features.get('legal_patterns', {}).get(pattern, 0)
        elif feature_name.startswith('sent_'):
            feat = feature_name[5:]
            return features.get('sentences', {}).get(feat, 0)
        elif feature_name.startswith('ent_'):
            etype = feature_name[4:]
            return features.get('entities', {}).get(etype, 0)
        else:
            return 0.0
    
    def batch_extract_features(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Extract features for a batch of texts"""
        results = []
        for text in texts:
            features = self.extract_all_features(text)
            results.append(features)
        return results