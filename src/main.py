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
        # Use updated categories from config
        categories = self.config.get('question_categories', ['fact', 'procedure', 'interpretive', 'directive', 'duty'])
        self.classifier = BayesianLegalClassifier(categories)
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
            api_key=self.config.get('gemini_api_key')  # Optional, only used if explicitly requested
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
    
    def initialize_system(self, dataset_paths: Dict[str, str] = None, use_saved_models: bool = True):
        """Initialize the system with datasets"""
        logger.info("Initializing Legal QA System...")
        
        # Try to load saved models first if requested
        if use_saved_models:
            try:
                logger.info("Attempting to load saved models...")
                self.load_system("trained_legal_qa_system")
                logger.info("✅ Loaded saved models successfully!")
                self.is_initialized = True
                return
            except Exception as e:
                logger.warning(f"Failed to load saved models: {e}")
                logger.info("Proceeding with full initialization...")
        
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
        
        # Fine-tune generative model (if enough data and not using Gemini and not skipped)
        if (len(self.dataset_df) > 5 and 
            not self.generative_model.use_gemini and 
            not self.config.get('skip_generative_training', False)):  # Skip fine-tuning if configured
            try:
                self.generative_model.fine_tune(
                    contexts, questions, answers,
                    epochs=self.config.get('max_epochs', 3),
                    batch_size=self.config.get('batch_size', 4),
                    learning_rate=self.config.get('learning_rate', 5e-5),
                    gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 4),
                    max_grad_norm=self.config.get('max_grad_norm', 1.0)
                )
            except Exception as e:
                logger.warning(f"Generative model training failed: {e}")
                logger.info("Continuing without generative model training")
                self.generative_model.is_trained = False
        elif self.generative_model.use_gemini:
            logger.info("Using Gemini model - no fine-tuning needed")
            self.generative_model.is_trained = True
        elif self.config.get('skip_generative_training', False):
            logger.info("Skipping generative model training as configured")
            self.generative_model.is_trained = False
        
        logger.info("Model training completed")
    
    def ask_question(self, question: str, top_k: int = None, use_extractive_only: bool = False, use_generative_only: bool = False) -> Dict[str, Any]:
        """Main method to ask a question and get an answer"""
        if not self.is_initialized:
            raise ValueError("System must be initialized before asking questions")
        
        logger.info(f"Processing question: {question[:100]}...")
        
        # Use default top_k if not specified
        if top_k is None:
            top_k = self.config['default_top_k']
        
        # Initialize variables to avoid NameError
        retrieved_docs = []
        classification = {}
        best_answer = {}
        ranked_answers = []
        
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
        
        # Step 3: Classify question (skip if classifier not trained)
        try:
            classification = self.classifier.predict_single(preprocessed_q['normalized'])
        except Exception as e:
            logger.warning(f"Classification failed: {e}")
            # Use default classification
            classification = {
                'predicted_category': 'fact',
                'confidence': 0.5,
                'probabilities': {'fact': 0.5, 'procedure': 0.3, 'interpretive': 0.2}
            }
        
        # Step 4: Retrieve relevant contexts
        try:
            retrieved_docs = self.retriever.search(preprocessed_q['normalized'], top_k)
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return {
                'question': question,
                'answer': 'I encountered an error while searching for relevant information.',
                'source': 'error',
                'confidence': 0.0,
                'explanation': f'Retrieval error: {str(e)}'
            }
        
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
        try:
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
        except Exception as e:
            logger.warning(f"Failed to cache answer: {e}")
        
        # Check if using fallback mode
        fallback_mode = not (self.extractive_model.is_trained and self.generative_model.is_trained)
        
        # Prepare response
        try:
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
            
        except Exception as e:
            logger.error(f"Error preparing response: {e}")
            return {
                'question': question,
                'answer': 'I encountered an error while processing your question.',
                'source': 'error',
                'confidence': 0.0,
                'explanation': f'Processing error: {str(e)}'
            }
    
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
        """Load a saved system state with improved error handling"""
        logger.info(f"Loading system from {filepath}")
        
        # Check if metadata file exists
        metadata_file = f"{filepath}_metadata.pkl"
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        # Load system metadata first
        import pickle
        try:
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            self.config = metadata['config']
            self.dataset_stats = metadata['dataset_stats']
            self.is_initialized = metadata['is_initialized']
            logger.info("✅ System metadata loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            raise
        
        # Load individual components with error handling
        components_loaded = 0
        
        # Load classifier
        classifier_file = f"{filepath}_classifier.pkl"
        if os.path.exists(classifier_file):
            try:
                self.classifier.load_model(classifier_file)
                logger.info("✅ Classifier loaded successfully")
                components_loaded += 1
            except Exception as e:
                logger.warning(f"Failed to load classifier: {e}")
        else:
            logger.warning(f"Classifier file not found: {classifier_file}")
        
        # Load retriever (check for hybrid retriever files)
        retriever_base = f"{filepath}_retriever"
        bm25_file = f"{retriever_base}_bm25.pkl"
        dense_file = f"{retriever_base}_dense.pkl"
        config_file = f"{retriever_base}_config.pkl"
        
        if os.path.exists(bm25_file) and os.path.exists(dense_file):
            try:
                # Load hybrid retriever components (add .pkl extension)
                retriever_filepath = f"{retriever_base}.pkl"
                self.retriever.load_model(retriever_filepath)
                logger.info("✅ Hybrid retriever loaded successfully")
                components_loaded += 1
            except Exception as e:
                logger.warning(f"Failed to load hybrid retriever: {e}")
        else:
            # Try single retriever file
            retriever_file = f"{filepath}_retriever.pkl"
            if os.path.exists(retriever_file):
                try:
                    self.retriever.load_model(retriever_file)
                    logger.info("✅ Retriever loaded successfully")
                    components_loaded += 1
                except Exception as e:
                    logger.warning(f"Failed to load retriever: {e}")
            else:
                logger.warning(f"Retriever files not found: {bm25_file}, {dense_file}, {retriever_file}")
        
        # Load extractive model
        extractive_file = f"{filepath}_extractive.pkl"
        if os.path.exists(extractive_file):
            try:
                self.extractive_model.load_model(extractive_file)
                logger.info("✅ Extractive model loaded successfully")
                components_loaded += 1
            except Exception as e:
                logger.warning(f"Failed to load extractive model: {e}")
        else:
            logger.info("Extractive model file not found, skipping")
            
        # Load generative model
        generative_file = f"{filepath}_generative.pkl"
        if os.path.exists(generative_file):
            try:
                self.generative_model.load_model(generative_file)
                logger.info("✅ Generative model loaded successfully")
                components_loaded += 1
            except Exception as e:
                logger.warning(f"Failed to load generative model: {e}")
        else:
            logger.info("Generative model file not found, skipping")
            
        # Load cache
        try:
            self.answer_cache.load_cache()
            logger.info("✅ Answer cache loaded successfully")
            components_loaded += 1
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
        
        logger.info(f"✅ System loaded successfully! ({components_loaded} components loaded)")
        
        # Check if we have minimum required components
        if components_loaded < 2:
            logger.warning("⚠️  Warning: Very few components loaded. System may not function properly.")
        
        return components_loaded
    
    def _fallback_extractive_answer(self, contexts: List[str], question: str) -> List[Dict[str, Any]]:
        """Enhanced fallback extractive answer using training data patterns"""
        import re
        
        # Enhanced keyword-based extraction with legal terms
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        
        # Add legal synonyms and related terms
        legal_synonyms = {
            'kill': ['murder', 'homicide', 'death', 'killing'],
            'self-defense': ['self defence', 'private defense', 'self protection'],
            'harassment': ['harass', 'harassing', 'molestation', 'abuse'],
            'sexual': ['sex', 'gender', 'modesty', 'outrage'],
            'jail': ['prison', 'imprisonment', 'custody', 'detention'],
            'punishment': ['penalty', 'sentence', 'fine', 'consequence'],
            'theft': ['steal', 'stealing', 'robbery', 'larceny'],
            'robbery': ['rob', 'robbing', 'theft with force', 'armed robbery'],
            'lawyer': ['attorney', 'counsel', 'legal representative', 'advocate'],
            'court': ['tribunal', 'judiciary', 'legal proceeding', 'trial']
        }
        
        # Expand question words with synonyms
        expanded_question_words = question_words.copy()
        for word in question_words:
            if word in legal_synonyms:
                expanded_question_words.update(legal_synonyms[word])
        
        results = []
        
        for context in contexts:
            # Find sentences that contain question keywords or synonyms
            sentences = re.split(r'[.!?]+', context)
            best_sentences = []
            
            for sentence in sentences:
                sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
                score = len(expanded_question_words.intersection(sentence_words))
                if score > 0:
                    best_sentences.append((score, sentence.strip()))
            
            # Sort by relevance score
            best_sentences.sort(key=lambda x: x[0], reverse=True)
            
            if best_sentences:
                # Combine top 2-3 most relevant sentences
                top_sentences = [sent for _, sent in best_sentences[:3]]
                combined_answer = '. '.join(top_sentences)
                confidence = min(0.8, best_sentences[0][0] / len(expanded_question_words)) if expanded_question_words else 0.4
                
                results.append({
                    'answer': combined_answer,
                    'confidence': confidence
                })
            else:
                # Fallback to first meaningful sentence
                first_sentence = sentences[0].strip() if sentences else context[:150]
                results.append({
                    'answer': first_sentence,
                    'confidence': 0.3
                })
        
        return results
    
    def _fallback_generative_answer(self, contexts: List[str], question: str) -> Dict[str, Any]:
        """Enhanced fallback generative answer using training data patterns"""
        import re
        
        # Combine contexts and create a comprehensive answer
        combined_context = ' '.join(contexts[:5])  # Use top 5 contexts for better coverage
        
        # Enhanced keyword matching with legal synonyms
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        
        # Add legal synonyms and related terms
        legal_synonyms = {
            'kill': ['murder', 'homicide', 'death', 'killing'],
            'self-defense': ['self defence', 'private defense', 'self protection'],
            'harassment': ['harass', 'harassing', 'molestation', 'abuse'],
            'sexual': ['sex', 'gender', 'modesty', 'outrage'],
            'jail': ['prison', 'imprisonment', 'custody', 'detention'],
            'punishment': ['penalty', 'sentence', 'fine', 'consequence'],
            'theft': ['steal', 'stealing', 'robbery', 'larceny'],
            'robbery': ['rob', 'robbing', 'theft with force', 'armed robbery'],
            'lawyer': ['attorney', 'counsel', 'legal representative', 'advocate'],
            'court': ['tribunal', 'judiciary', 'legal proceeding', 'trial']
        }
        
        # Expand question words with synonyms
        expanded_question_words = question_words.copy()
        for word in question_words:
            if word in legal_synonyms:
                expanded_question_words.update(legal_synonyms[word])
        
        # Find sentences with highest keyword overlap
        sentences = re.split(r'[.!?]+', combined_context)
        best_sentences = []
        
        for sentence in sentences:
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            score = len(expanded_question_words.intersection(sentence_words))
            if score > 0:
                best_sentences.append((score, sentence.strip()))
        
        # Sort by score and take top sentences
        best_sentences.sort(key=lambda x: x[0], reverse=True)
        
        if best_sentences:
            # Combine top 3-4 most relevant sentences for comprehensive answer
            answer_parts = [sent for _, sent in best_sentences[:4]]
            answer = '. '.join(answer_parts)
            
            # Calculate confidence based on relevance and coverage
            relevance_score = best_sentences[0][0] / len(expanded_question_words) if expanded_question_words else 0.3
            coverage_score = min(1.0, len(best_sentences) / 3)  # Reward having multiple relevant sentences
            confidence = min(0.8, (relevance_score + coverage_score) / 2)
        else:
            # Fallback to first meaningful part of context
            answer = combined_context[:300] + "..." if len(combined_context) > 300 else combined_context
            confidence = 0.3
        
        return {
            'answer': answer,
            'confidence': confidence
        }
    
    # Cache Management Methods
    def clear_cache(self):
        """Clear all cached answers"""
        self.answer_cache.clear_cache()
        success = self.answer_cache.force_save_cache()
        if success:
            logger.info("Cache cleared successfully")
        else:
            logger.warning("Cache cleared but file save failed")
    
    def remove_cached_answer(self, question: str, context: str = "") -> bool:
        """Remove a specific cached answer"""
        success = self.answer_cache.remove_answer(question, context)
        if success:
            self.answer_cache.save_cache()
        return success
    
    def remove_similar_cached_answers(self, question: str, similarity_threshold: float = 0.8) -> int:
        """Remove all similar cached answers"""
        removed_count = self.answer_cache.remove_similar_answers(question, similarity_threshold)
        if removed_count > 0:
            self.answer_cache.save_cache()
        return removed_count
    
    def remove_old_cached_entries(self, max_age_hours: int = 24) -> int:
        """Remove cached entries older than specified hours"""
        removed_count = self.answer_cache.remove_old_entries(max_age_hours)
        if removed_count > 0:
            self.answer_cache.save_cache()
        return removed_count
    
    def remove_low_access_cached_entries(self, min_access_count: int = 1) -> int:
        """Remove cached entries with low access count"""
        removed_count = self.answer_cache.remove_low_access_entries(min_access_count)
        if removed_count > 0:
            self.answer_cache.save_cache()
        return removed_count
    
    def remove_cached_entries_by_pattern(self, pattern: str, search_in: str = "question") -> int:
        """Remove cached entries matching a pattern"""
        removed_count = self.answer_cache.remove_entries_by_pattern(pattern, search_in)
        if removed_count > 0:
            self.answer_cache.save_cache()
        return removed_count
    
    def cleanup_cache(self, max_age_hours: int = 24, min_access_count: int = 1) -> Dict[str, int]:
        """Comprehensive cache cleanup"""
        cleanup_stats = self.answer_cache.cleanup_cache(max_age_hours, min_access_count)
        logger.info(f"Cache cleanup completed: {cleanup_stats}")
        return cleanup_stats
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.answer_cache.get_cache_stats()
    
    def search_cache(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search cached entries"""
        return self.answer_cache.search_cache(query, limit)
    
    def list_cached_entries(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List cached entries"""
        entries = list(self.answer_cache.cache.values())
        # Sort by access count (most accessed first)
        entries.sort(key=lambda x: x['access_count'], reverse=True)
        return entries[:limit]