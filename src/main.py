"""
Main Legal QA System orchestrator
"""
import os
import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger

from .data import LegalDatasetLoader, LegalTextPreprocessor, EmbeddingGenerator
from .classification import BayesianLegalClassifier, SyntacticFeatureExtractor
from .retrieval import HybridRetriever
from .generation import ExtractiveAnswerModel, GenerativeAnswerModel, AnswerRanker
from .utils import TanimotoSimilarity, AnswerCache


class LegalQASystem:
    """Main Legal QA System that orchestrates all components"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.dataset_loader = LegalDatasetLoader(self.config['data_dir'])
        self.preprocessor = LegalTextPreprocessor()
        self.embedding_generator = EmbeddingGenerator(self.config['embedding_model'])
        
        # Classification
        self.classifier = BayesianLegalClassifier(self.config['question_categories'])
        self.syntactic_extractor = SyntacticFeatureExtractor()
        
        # Retrieval
        self.retriever = HybridRetriever(
            bm25_weight=self.config['bm25_weight'],
            dense_weight=self.config['dense_weight']
        )
        
        # Generation
        self.extractive_model = ExtractiveAnswerModel(self.config['extractive_model'])
        self.generative_model = GenerativeAnswerModel(
            self.config['generative_model'],
            api_key=self.config.get('gemini_api_key')
        )
        self.answer_ranker = AnswerRanker(
            bayesian_weight=self.config['bayesian_weight'],
            similarity_weight=self.config['similarity_weight'],
            confidence_weight=self.config['confidence_weight']
        )
        
        # Utilities
        self.answer_cache = AnswerCache(self.config['cache_file'])
        self.similarity_checker = TanimotoSimilarity()
        
        # System state
        self.is_initialized = False
        self.dataset_stats = None
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        try:
            # Try to import config from the main directory
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent))
            from config import get_config_for_mode
            return get_config_for_mode('demo')
        except ImportError:
            # Fallback to hardcoded config
            return {
                'data_dir': 'data/',
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'extractive_model': 'bert-base-uncased',
                'generative_model': 't5-small',
                'question_categories': ['fact', 'procedure', 'interpretive'],
                'bm25_weight': 0.3,
                'dense_weight': 0.7,
                'bayesian_weight': 0.3,
                'similarity_weight': 0.4,
                'confidence_weight': 0.3,
                'cache_file': 'answer_cache.pkl',
                'similarity_threshold': 0.8,
                'confidence_threshold': 0.7,
                'default_top_k': 15,
                'max_context_length': 2000,
                'batch_size': 32,
                'max_epochs': 3,
                'learning_rate': 2e-5
            }
    
    def initialize_system(self, dataset_paths: Dict[str, str] = None):
        """Initialize the system with datasets"""
        logger.info("Initializing Legal QA System...")
        
        # Load datasets
        if dataset_paths:
            self._load_datasets(dataset_paths)
        else:
            # Use default dataset paths
            default_paths = {
                'constitution': 'data/constitution_qa.json',
                'crpc': 'data/crpc_qa.json',
                'ipc': 'data/ipc_qa.json'
            }
            self._load_datasets(default_paths)
        
        # Preprocess data
        self._preprocess_data()
        
        # Generate embeddings
        self._generate_embeddings()
        
        # Fit retrieval system
        self._fit_retrieval_system()
        
        # Train models (if training data available)
        self._train_models()
        
        self.is_initialized = True
        logger.info("Legal QA System initialized successfully")
    
    def _load_datasets(self, dataset_paths: Dict[str, str]):
        """Load and merge datasets"""
        logger.info("Loading datasets...")
        
        datasets = []
        for dataset_name, path in dataset_paths.items():
            if dataset_name == 'constitution':
                dataset = self.dataset_loader.load_constitution_dataset(path)
            elif dataset_name == 'crpc':
                dataset = self.dataset_loader.load_crpc_dataset(path)
            elif dataset_name == 'ipc':
                dataset = self.dataset_loader.load_ipc_dataset(path)
            else:
                logger.warning(f"Unknown dataset: {dataset_name}")
                continue
            
            datasets.append(dataset)
        
        # Merge datasets
        self.dataset_df = self.dataset_loader.merge_datasets(datasets)
        self.dataset_stats = self.dataset_loader.get_dataset_stats(self.dataset_df)
        
        logger.info(f"Loaded {len(self.dataset_df)} Q&A pairs")
    
    def _preprocess_data(self):
        """Preprocess the loaded data"""
        logger.info("Preprocessing data...")
        
        # Preprocess questions
        questions = self.dataset_df['question'].tolist()
        self.preprocessed_questions = self.preprocessor.batch_preprocess(questions, "question")
        
        # Preprocess answers
        answers = self.dataset_df['answer'].tolist()
        self.preprocessed_answers = self.preprocessor.batch_preprocess(answers, "answer")
        
        # Preprocess contexts
        contexts = self.dataset_df['context'].fillna('').tolist()
        self.preprocessed_contexts = self.preprocessor.batch_preprocess(contexts, "context")
        
        logger.info("Data preprocessing completed")
    
    def _generate_embeddings(self):
        """Generate embeddings for questions and contexts"""
        logger.info("Generating embeddings...")
        
        # Generate question embeddings
        questions = [pq['normalized'] for pq in self.preprocessed_questions]
        self.question_embeddings = self.embedding_generator.generate_question_embeddings(questions)
        
        # Generate context embeddings
        contexts = [pc['normalized'] for pc in self.preprocessed_contexts]
        self.context_embeddings = self.embedding_generator.generate_context_embeddings(contexts)
        
        logger.info("Embeddings generated successfully")
    
    def _fit_retrieval_system(self):
        """Fit the hybrid retrieval system"""
        logger.info("Fitting retrieval system...")
        
        # Prepare documents for retrieval
        documents = [pc['normalized'] for pc in self.preprocessed_contexts]
        metadata = []
        
        for i, row in self.dataset_df.iterrows():
            meta = {
                'id': row['id'],
                'category': row['category'],
                'source': row['source'],
                'question': row['question'],
                'answer': row['answer']
            }
            metadata.append(meta)
        
        # Fit retriever
        self.retriever.fit(documents, metadata)
        
        logger.info("Retrieval system fitted successfully")
    
    def _train_models(self):
        """Train classification and generation models"""
        logger.info("Training models...")
        
        # Train question classifier
        questions = [pq['normalized'] for pq in self.preprocessed_questions]
        categories = self.dataset_df['category'].tolist()
        
        self.classifier.train(questions, categories)
        
        # Train extractive model (if enough data)
        if len(self.dataset_df) > 5:  # Lowered threshold for demo
            contexts = [pc['normalized'] for pc in self.preprocessed_contexts]
            answers = [pa['normalized'] for pa in self.preprocessed_answers]
            
            self.extractive_model.train(
                contexts, questions, answers,
                epochs=self.config.get('max_epochs', 3),
                batch_size=self.config.get('batch_size', 8),
                learning_rate=self.config.get('learning_rate', 2e-5)
            )
        
        # Fine-tune generative model (if enough data and not using Gemini)
        if len(self.dataset_df) > 5 and not self.generative_model.use_gemini:  # Skip fine-tuning for Gemini
            self.generative_model.fine_tune(
                contexts, questions, answers,
                epochs=self.config.get('max_epochs', 3),
                batch_size=self.config.get('batch_size', 4),
                learning_rate=self.config.get('learning_rate', 5e-5)
            )
        elif self.generative_model.use_gemini:
            logger.info("Using Gemini model - no fine-tuning needed")
            self.generative_model.is_trained = True
        
        logger.info("Model training completed")
    
    def ask_question(self, question: str, top_k: int = None, use_extractive_only: bool = False, use_generative_only: bool = False) -> Dict[str, Any]:
        """Main method to ask a question and get an answer"""
        if not self.is_initialized:
            raise ValueError("System must be initialized before asking questions")
        
        logger.info(f"Processing question: {question[:100]}...")
        
        # Use default top_k if not specified
        if top_k is None:
            top_k = self.config['default_top_k']
        
        # Step 1: Preprocess question
        preprocessed_q = self.preprocessor.preprocess_question(question)
        
        # Step 2: Check for similar questions in cache
        cached_answers = self.answer_cache.get_cached_answers(question)
        if cached_answers:
            logger.info("Found cached answer")
            return {
                'question': question,
                'answer': cached_answers[0]['answer'],
                'source': 'cache',
                'confidence': 0.9,
                'explanation': 'Answer retrieved from cache'
            }
        
        # Step 3: Classify question
        classification = self.classifier.predict_single(preprocessed_q['normalized'])
        
        # Step 4: Retrieve relevant contexts
        retrieved_docs = self.retriever.search(preprocessed_q['normalized'], top_k)
        
        if not retrieved_docs:
            return {
                'question': question,
                'answer': 'I could not find relevant information to answer your question.',
                'source': 'none',
                'confidence': 0.0,
                'explanation': 'No relevant documents found'
            }
        
        # Step 5: Generate answers
        contexts = [doc['document'] for doc in retrieved_docs]
        
        # Limit context length for generation to avoid memory issues
        max_context_length = self.config.get('max_context_length', 2000)
        truncated_contexts = []
        current_length = 0
        
        for context in contexts:
            if current_length + len(context) <= max_context_length:
                truncated_contexts.append(context)
                current_length += len(context)
            else:
                # Add partial context if there's still room
                remaining_length = max_context_length - current_length
                if remaining_length > 100:  # Only add if meaningful length remains
                    truncated_contexts.append(context[:remaining_length])
                break
        
        # Use truncated contexts for generation
        generation_contexts = truncated_contexts if truncated_contexts else contexts[:3]
        
        # Step 6: Generate answers based on model selection
        all_answers = []
        
        # Generate with extractive model (if requested and trained)
        if not use_generative_only and self.extractive_model.is_trained:
            extractive_results = self.extractive_model.predict(generation_contexts, [question] * len(generation_contexts))
            
            # Add extractive answers
            for i, result in enumerate(extractive_results):
                answer_data = {
                    'answer': result['answer'],
                    'confidence': result['confidence'],
                    'source': 'extractive',
                    'context': contexts[i],
                    'metadata': retrieved_docs[i]['metadata']
                }
                all_answers.append(answer_data)
        elif not use_generative_only:
            # Fallback: extract relevant sentences from contexts
            extractive_results = self._fallback_extractive_answer(generation_contexts, question)
            for i, result in enumerate(extractive_results):
                answer_data = {
                    'answer': result['answer'],
                    'confidence': result['confidence'],
                    'source': 'extractive',
                    'context': contexts[i],
                    'metadata': retrieved_docs[i]['metadata']
                }
                all_answers.append(answer_data)
        
        # Generate with generative model (if requested and trained)
        if not use_extractive_only and self.generative_model.is_trained:
            generative_result = self.generative_model.generate_with_retrieval(question, generation_contexts)
            
            # Add generative answer
            generative_answer = {
                'answer': generative_result['answer'],
                'confidence': generative_result['confidence'],
                'source': 'generative',
                'context': ' '.join(generation_contexts),
                'metadata': {
                    'used_contexts': len(generation_contexts),
                    'total_retrieved': len(contexts),
                    'context_truncated': len(generation_contexts) < len(contexts)
                }
            }
            all_answers.append(generative_answer)
        elif not use_extractive_only:
            # Fallback: simple answer from retrieved contexts
            generative_result = self._fallback_generative_answer(generation_contexts, question)
            generative_answer = {
                'answer': generative_result['answer'],
                'confidence': generative_result['confidence'],
                'source': 'generative',
                'context': ' '.join(generation_contexts),
                'metadata': {
                    'used_contexts': len(generation_contexts),
                    'total_retrieved': len(contexts),
                    'context_truncated': len(generation_contexts) < len(contexts)
                }
            }
            all_answers.append(generative_answer)
        
        # Rank answers
        question_embedding = self.embedding_generator.generate_embeddings([question])[0]
        ranked_answers = self.answer_ranker.rank_answers(
            all_answers, 
            question, 
            question_embedding,
            classification['confidence']
        )
        
        # Select best answer
        best_answer = self.answer_ranker.select_best_answer(
            ranked_answers, 
            self.config['confidence_threshold']
        )
        
        if not best_answer:
            best_answer = ranked_answers[0] if ranked_answers else {
                'answer': 'I could not generate a confident answer.',
                'confidence': 0.0,
                'source': 'none'
            }
        
        # Step 7: Cache the answer
        self.answer_cache.store_answer(
            question, 
            best_answer['answer'],
            best_answer.get('context', ''),
            {
                'classification': classification,
                'retrieved_docs': len(retrieved_docs),
                'confidence': best_answer['confidence']
            }
        )
        
        # Check if using fallback mode
        fallback_mode = not (self.extractive_model.is_trained and self.generative_model.is_trained)
        
        # Prepare response
        response = {
            'question': question,
            'answer': best_answer['answer'],
            'source': best_answer['source'],
            'confidence': best_answer['confidence'],
            'classification': classification,
            'retrieved_documents': len(retrieved_docs),
            'all_answers': ranked_answers,
            'explanation': self.answer_ranker.get_answer_explanation(best_answer),
            'fallback_mode': fallback_mode
        }
        
        logger.info(f"Question processed successfully. Confidence: {best_answer['confidence']:.3f}")
        return response
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics"""
        status = {
            'is_initialized': self.is_initialized,
            'dataset_stats': self.dataset_stats,
            'cache_stats': self.answer_cache.get_cache_stats(),
            'retriever_info': self.retriever.get_model_info(),
            'classifier_info': {
                'is_trained': self.classifier.is_trained,
                'categories': self.classifier.categories
            },
            'extractive_model_info': self.extractive_model.get_model_info(),
            'generative_model_info': self.generative_model.get_model_info()
        }
        return self._convert_numpy_types(status)
    
    def save_system(self, filepath: str):
        """Save the entire system state"""
        logger.info(f"Saving system to {filepath}")
        
        # Save individual components
        self.classifier.save_model(f"{filepath}_classifier.pkl")
        self.retriever.save_model(f"{filepath}_retriever.pkl")
        
        # Only save models if they are trained
        if self.extractive_model.is_trained:
            self.extractive_model.save_model(f"{filepath}_extractive.pkl")
        else:
            logger.info("Skipping extractive model save (not trained)")
            
        if self.generative_model.is_trained:
            self.generative_model.save_model(f"{filepath}_generative.pkl")
        else:
            logger.info("Skipping generative model save (not trained)")
            
        self.answer_cache.save_cache()
        
        # Save system metadata
        import pickle
        metadata = {
            'config': self.config,
            'dataset_stats': self.dataset_stats,
            'is_initialized': self.is_initialized
        }
        
        with open(f"{filepath}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info("System saved successfully")
    
    def load_system(self, filepath: str):
        """Load a saved system state"""
        logger.info(f"Loading system from {filepath}")
        
        # Load individual components
        self.classifier.load_model(f"{filepath}_classifier.pkl")
        self.retriever.load_model(f"{filepath}_retriever.pkl")
        
        # Load models only if they exist and are trained
        if os.path.exists(f"{filepath}_extractive.pkl"):
            try:
                self.extractive_model.load_model(f"{filepath}_extractive.pkl")
            except Exception as e:
                logger.warning(f"Failed to load extractive model: {e}")
        else:
            logger.info("Extractive model file not found, skipping")
            
        if os.path.exists(f"{filepath}_generative.pkl"):
            try:
                self.generative_model.load_model(f"{filepath}_generative.pkl")
            except Exception as e:
                logger.warning(f"Failed to load generative model: {e}")
        else:
            logger.info("Generative model file not found, skipping")
            
        self.answer_cache.load_cache()
        
        # Load system metadata
        import pickle
        with open(f"{filepath}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        self.config = metadata['config']
        self.dataset_stats = metadata['dataset_stats']
        self.is_initialized = metadata['is_initialized']
        
        logger.info("System loaded successfully")
    
    def _fallback_extractive_answer(self, contexts: List[str], question: str) -> List[Dict[str, Any]]:
        """Fallback extractive answer when model is not trained"""
        import re
        
        # Simple keyword-based extraction
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        results = []
        
        for context in contexts:
            # Find sentences that contain question keywords
            sentences = re.split(r'[.!?]+', context)
            best_sentence = ""
            max_score = 0
            
            for sentence in sentences:
                sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
                score = len(question_words.intersection(sentence_words))
                if score > max_score:
                    max_score = score
                    best_sentence = sentence.strip()
            
            if best_sentence:
                confidence = min(0.7, max_score / len(question_words)) if question_words else 0.3
                results.append({
                    'answer': best_sentence,
                    'confidence': confidence
                })
            else:
                # Fallback to first sentence
                first_sentence = sentences[0].strip() if sentences else context[:100]
                results.append({
                    'answer': first_sentence,
                    'confidence': 0.3
                })
        
        return results
    
    def _fallback_generative_answer(self, contexts: List[str], question: str) -> Dict[str, Any]:
        """Fallback generative answer when model is not trained"""
        # Combine contexts and create a simple answer
        combined_context = ' '.join(contexts[:3])  # Use top 3 contexts
        
        # Simple keyword matching to find relevant parts
        import re
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        
        # Find sentences with highest keyword overlap
        sentences = re.split(r'[.!?]+', combined_context)
        best_sentences = []
        
        for sentence in sentences:
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            score = len(question_words.intersection(sentence_words))
            if score > 0:
                best_sentences.append((score, sentence.strip()))
        
        # Sort by score and take top sentences
        best_sentences.sort(key=lambda x: x[0], reverse=True)
        
        if best_sentences:
            # Combine top 2 sentences
            answer_parts = [sent for _, sent in best_sentences[:2]]
            answer = '. '.join(answer_parts)
            confidence = min(0.6, best_sentences[0][0] / len(question_words)) if question_words else 0.4
        else:
            # Fallback to first part of context
            answer = combined_context[:200] + "..." if len(combined_context) > 200 else combined_context
            confidence = 0.3
        
        return {
            'answer': answer,
            'confidence': confidence
        }