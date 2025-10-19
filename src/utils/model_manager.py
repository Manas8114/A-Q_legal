"""
Model Manager for A-Qlegal 2.0
Handles model caching, loading, and persistence for faster startup
"""

import os
import json
import pickle
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
from datetime import datetime
import hashlib


class ModelManager:
    """
    Centralized model management system
    - Caches models to disk for faster loading
    - Tracks model versions and metadata
    - Validates model integrity
    - Provides unified interface for all models
    """
    
    def __init__(self, cache_dir: str = "models/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.cache_dir / "model_metadata.json"
        self.metadata = self._load_metadata()
        
        # Model registry
        self.models = {}
        self.tokenizers = {}
        
        logger.info(f"📦 Model Manager initialized (cache: {self.cache_dir})")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load model metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save model metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _get_model_hash(self, model_name: str) -> str:
        """Generate hash for model identifier"""
        return hashlib.md5(model_name.encode()).hexdigest()[:16]
    
    def cache_model(
        self,
        model_name: str,
        model: Any,
        tokenizer: Any = None,
        metadata: Dict[str, Any] = None
    ) -> Path:
        """
        Cache model to disk for faster loading
        
        Args:
            model_name: Name/identifier of the model
            model: Model object
            tokenizer: Tokenizer object (optional)
            metadata: Additional metadata
        
        Returns:
            Path to cached model
        """
        logger.info(f"💾 Caching model: {model_name}")
        
        model_hash = self._get_model_hash(model_name)
        model_dir = self.cache_dir / model_hash
        model_dir.mkdir(exist_ok=True)
        
        try:
            # Save model
            if hasattr(model, 'save_pretrained'):
                # HuggingFace model
                model.save_pretrained(str(model_dir / "model"))
            else:
                # PyTorch model
                torch.save(model.state_dict(), str(model_dir / "model.pt"))
            
            # Save tokenizer
            if tokenizer is not None:
                if hasattr(tokenizer, 'save_pretrained'):
                    tokenizer.save_pretrained(str(model_dir / "tokenizer"))
                else:
                    with open(model_dir / "tokenizer.pkl", 'wb') as f:
                        pickle.dump(tokenizer, f)
            
            # Save metadata
            model_metadata = {
                'model_name': model_name,
                'cached_at': str(datetime.now()),
                'model_type': type(model).__name__,
                'has_tokenizer': tokenizer is not None,
                'custom_metadata': metadata or {}
            }
            
            with open(model_dir / "metadata.json", 'w') as f:
                json.dump(model_metadata, f, indent=2)
            
            # Update global metadata
            self.metadata[model_name] = {
                'hash': model_hash,
                'path': str(model_dir),
                'cached_at': model_metadata['cached_at']
            }
            self._save_metadata()
            
            logger.success(f"✅ Model cached: {model_dir}")
            return model_dir
            
        except Exception as e:
            logger.error(f"❌ Failed to cache model: {e}")
            raise
    
    def load_cached_model(
        self,
        model_name: str,
        model_class: Any = None,
        tokenizer_class: Any = None,
        device: str = "auto"
    ) -> tuple:
        """
        Load model from cache
        
        Args:
            model_name: Name/identifier of the model
            model_class: Class to instantiate model (if not HuggingFace)
            tokenizer_class: Class to instantiate tokenizer
            device: Device to load model on
        
        Returns:
            (model, tokenizer) tuple
        """
        if model_name not in self.metadata:
            raise ValueError(f"Model {model_name} not found in cache")
        
        logger.info(f"📂 Loading cached model: {model_name}")
        
        model_dir = Path(self.metadata[model_name]['path'])
        
        try:
            # Load metadata
            with open(model_dir / "metadata.json", 'r') as f:
                metadata = json.load(f)
            
            # Setup device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load model
            model = None
            if (model_dir / "model").exists():
                # HuggingFace model
                from transformers import AutoModel
                model = AutoModel.from_pretrained(str(model_dir / "model"))
                model.to(device)
            elif (model_dir / "model.pt").exists():
                # PyTorch model
                if model_class is None:
                    raise ValueError("model_class required for PyTorch models")
                model = model_class()
                model.load_state_dict(torch.load(str(model_dir / "model.pt"), map_location=device))
                model.to(device)
            
            # Load tokenizer
            tokenizer = None
            if metadata.get('has_tokenizer', False):
                if (model_dir / "tokenizer").exists():
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))
                elif (model_dir / "tokenizer.pkl").exists():
                    with open(model_dir / "tokenizer.pkl", 'rb') as f:
                        tokenizer = pickle.load(f)
            
            logger.success(f"✅ Loaded from cache: {model_name}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"❌ Failed to load cached model: {e}")
            raise
    
    def is_cached(self, model_name: str) -> bool:
        """Check if model is cached"""
        return model_name in self.metadata
    
    def clear_cache(self, model_name: str = None):
        """Clear model cache"""
        if model_name:
            if model_name in self.metadata:
                model_dir = Path(self.metadata[model_name]['path'])
                import shutil
                shutil.rmtree(model_dir, ignore_errors=True)
                del self.metadata[model_name]
                self._save_metadata()
                logger.info(f"🗑️  Cleared cache for: {model_name}")
        else:
            import shutil
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            self.cache_dir.mkdir(exist_ok=True)
            self.metadata = {}
            self._save_metadata()
            logger.info("🗑️  Cleared all model caches")
    
    def list_cached_models(self) -> Dict[str, Any]:
        """List all cached models"""
        return self.metadata.copy()
    
    def get_cache_size(self) -> int:
        """Get total cache size in bytes"""
        total_size = 0
        for path in self.cache_dir.rglob('*'):
            if path.is_file():
                total_size += path.stat().st_size
        return total_size
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        return {
            'num_models': len(self.metadata),
            'cache_dir': str(self.cache_dir),
            'total_size_mb': self.get_cache_size() / (1024 * 1024),
            'models': self.metadata
        }


def save_all_models():
    """Save all models to cache for faster loading"""
    logger.info("🔄 Saving all models to cache...")
    
    manager = ModelManager()
    
    # 1. Save RAG embeddings
    logger.info("\n1️⃣ Caching RAG system...")
    try:
        from src.retrieval.rag_system import AdvancedRAGSystem
        from sentence_transformers import SentenceTransformer
        
        # Cache embedding model
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        manager.cache_model(
            "sentence_transformer_all_minilm_l6_v2",
            embedding_model,
            metadata={'purpose': 'RAG embeddings'}
        )
        logger.success("✅ RAG embedding model cached")
    except Exception as e:
        logger.warning(f"⚠️  RAG embedding cache failed: {e}")
    
    # 2. Save Legal-BERT
    logger.info("\n2️⃣ Caching Legal-BERT...")
    try:
        from transformers import AutoTokenizer, AutoModel
        
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
        
        manager.cache_model(
            "legal_bert_base",
            model,
            tokenizer,
            metadata={'purpose': 'Legal text encoding'}
        )
        logger.success("✅ Legal-BERT cached")
    except Exception as e:
        logger.warning(f"⚠️  Legal-BERT cache failed: {e}")
    
    # 3. Save Flan-T5 Base
    logger.info("\n3️⃣ Caching Flan-T5 Base...")
    try:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
        
        manager.cache_model(
            "flan_t5_base",
            model,
            tokenizer,
            metadata={'purpose': 'Text generation and simplification'}
        )
        logger.success("✅ Flan-T5 Base cached")
    except Exception as e:
        logger.warning(f"⚠️  Flan-T5 cache failed: {e}")
    
    # Print summary
    logger.info("\n" + "="*60)
    cache_info = manager.get_cache_info()
    logger.info(f"📊 Cache Summary:")
    logger.info(f"  Models cached: {cache_info['num_models']}")
    logger.info(f"  Total size: {cache_info['total_size_mb']:.2f} MB")
    logger.info(f"  Cache directory: {cache_info['cache_dir']}")
    logger.info("="*60)
    
    return manager


def main():
    """Test model manager"""
    manager = save_all_models()
    
    # List cached models
    logger.info("\n📋 Cached Models:")
    for name, info in manager.list_cached_models().items():
        logger.info(f"  • {name}")
        logger.info(f"    Cached: {info['cached_at']}")
        logger.info(f"    Path: {info['path']}")


if __name__ == "__main__":
    main()

