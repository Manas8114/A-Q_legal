"""
Configuration file for Legal QA System
"""
import os
from typing import Dict, Any

def get_config() -> Dict[str, Any]:
    """Get configuration with environment-specific overrides"""
    
    # Base configuration
    config = {
        'data_dir': 'data/',
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'extractive_model': 'bert-base-uncased',
        'generative_model': 'gemini-1.5-flash',  # Use Gemini by default
        'gemini_api_key': None,  # Will be set via environment variable
        'question_categories': ['fact', 'procedure', 'interpretive'],
        'bm25_weight': 0.3,
        'dense_weight': 0.7,
        'bayesian_weight': 0.3,
        'similarity_weight': 0.4,
        'confidence_weight': 0.3,
        'cache_file': 'answer_cache.pkl',
        'similarity_threshold': 0.8,
        'confidence_threshold': 0.7,
        'default_top_k': 15,  # Number of documents to retrieve
        'max_context_length': 2000,  # Maximum context length for generation
        'batch_size': 32,  # Batch size for processing
        'max_epochs': 3,  # Maximum training epochs
        'learning_rate': 2e-5,  # Learning rate for fine-tuning
    }
    
    # Environment-specific overrides
    if os.getenv('LOW_MEMORY_MODE', 'false').lower() == 'true':
        config.update({
            'generative_model': 'distilbert-base-uncased',  # Smaller model
            'default_top_k': 10,  # Fewer documents
            'max_context_length': 1000,  # Shorter context
            'batch_size': 16,  # Smaller batches
        })
    
    if os.getenv('HIGH_PERFORMANCE_MODE', 'false').lower() == 'true':
        config.update({
            'default_top_k': 25,  # More documents
            'max_context_length': 4000,  # Longer context
            'batch_size': 64,  # Larger batches
        })
    
    # Override with environment variables if set
    for key in config:
        env_key = f'LEGAL_QA_{key.upper()}'
        if env_key in os.environ:
            value = os.environ[env_key]
            # Try to convert to appropriate type
            if key in ['bm25_weight', 'dense_weight', 'bayesian_weight', 'similarity_weight', 
                      'confidence_weight', 'similarity_threshold', 'confidence_threshold', 'learning_rate']:
                config[key] = float(value)
            elif key in ['default_top_k', 'max_context_length', 'batch_size', 'max_epochs']:
                config[key] = int(value)
            else:
                config[key] = value
    
    # Set Gemini API key from environment
    if 'GEMINI_API_KEY' in os.environ:
        config['gemini_api_key'] = os.environ['GEMINI_API_KEY']
    
    return config

# Predefined configurations for different use cases
CONFIGS = {
    'demo': {
        'default_top_k': 10,
        'max_context_length': 1500,
        'batch_size': 16,
        'max_epochs': 2,
    },
    'production': {
        'default_top_k': 20,
        'max_context_length': 3000,
        'batch_size': 32,
        'max_epochs': 5,
    },
    'research': {
        'default_top_k': 30,
        'max_context_length': 5000,
        'batch_size': 64,
        'max_epochs': 10,
    }
}

def get_config_for_mode(mode: str = 'demo') -> Dict[str, Any]:
    """Get configuration for a specific mode"""
    base_config = get_config()
    if mode in CONFIGS:
        base_config.update(CONFIGS[mode])
    return base_config
