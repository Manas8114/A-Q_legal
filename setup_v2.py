#!/usr/bin/env python3
"""
A-Qlegal 2.0 - Complete Setup Script
Automates the entire setup process for A-Qlegal 2.0
"""

import os
import sys
import subprocess
from pathlib import Path
from loguru import logger
import json


class AQlegalSetup:
    """Complete setup automation for A-Qlegal 2.0"""
    
    def __init__(self):
        self.base_dir = Path.cwd()
        self.steps_completed = []
        
        logger.info("🚀 A-Qlegal 2.0 Setup Wizard")
        logger.info("=" * 60)
    
    def check_python_version(self):
        """Check Python version"""
        logger.info("\n📋 Step 1: Checking Python version...")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 10):
            logger.error(f"❌ Python 3.10+ required, found {version.major}.{version.minor}")
            return False
        
        logger.success(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        self.steps_completed.append("python_version")
        return True
    
    def install_dependencies(self):
        """Install Python dependencies"""
        logger.info("\n📦 Step 2: Installing dependencies...")
        
        try:
            # Upgrade pip
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
            
            # Install requirements
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
            
            logger.success("✅ Dependencies installed")
            self.steps_completed.append("dependencies")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Dependency installation failed: {e}")
            return False
    
    def download_nlp_models(self):
        """Download required NLP models"""
        logger.info("\n🤖 Step 3: Downloading NLP models...")
        
        try:
            # spaCy model
            logger.info("Downloading spaCy model...")
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            
            # NLTK data
            logger.info("Downloading NLTK data...")
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            
            logger.success("✅ NLP models downloaded")
            self.steps_completed.append("nlp_models")
            return True
            
        except Exception as e:
            logger.error(f"❌ NLP model download failed: {e}")
            return False
    
    def create_directory_structure(self):
        """Create necessary directories"""
        logger.info("\n📁 Step 4: Creating directory structure...")
        
        directories = [
            "data/external_datasets",
            "models/pretrained",
            "models/lora_models",
            "models/rag_index",
            "models/faiss_index",
            "data/chroma_db",
            "logs",
            "outputs"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created: {dir_path}")
        
        logger.success(f"✅ Created {len(directories)} directories")
        self.steps_completed.append("directories")
        return True
    
    def download_datasets(self, download_external: bool = False):
        """Download and process datasets"""
        logger.info("\n📚 Step 5: Processing datasets...")
        
        if not download_external:
            logger.info("⏭️  Skipping external dataset download (can do this later)")
            logger.info("   Run: python src/data/dataset_downloader.py")
            self.steps_completed.append("datasets_skipped")
            return True
        
        try:
            logger.info("Downloading external datasets...")
            from src.data.dataset_downloader import LegalDatasetDownloader
            
            downloader = LegalDatasetDownloader()
            results = downloader.download_all()
            
            successful = sum(1 for v in results.values() if v is not None)
            logger.success(f"✅ Downloaded {successful}/{len(results)} datasets")
            
            self.steps_completed.append("datasets")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  Dataset download failed: {e}")
            logger.info("   You can download datasets later using dataset_downloader.py")
            self.steps_completed.append("datasets_partial")
            return True
    
    def process_existing_data(self):
        """Process existing legal dataset"""
        logger.info("\n🔄 Step 6: Processing existing data...")
        
        try:
            from src.data.enhanced_preprocessor import EnhancedLegalPreprocessor
            
            preprocessor = EnhancedLegalPreprocessor()
            
            # Check if dataset exists
            dataset_file = Path("data/expanded_legal_dataset.json")
            if not dataset_file.exists():
                logger.warning("⚠️  Original dataset not found, skipping preprocessing")
                self.steps_completed.append("processing_skipped")
                return True
            
            # Load and process
            with open(dataset_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            logger.info(f"Processing {len(dataset)} documents...")
            enhanced_dataset = preprocessor.process_dataset(
                dataset[:1000],  # Process first 1000 for quick setup
                output_file="data/enhanced_legal_dataset_v2.json"
            )
            
            # Get statistics
            stats = preprocessor.get_statistics(enhanced_dataset)
            
            # Save statistics
            with open("data/enhanced_dataset_statistics_v2.json", 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.success(f"✅ Processed {len(enhanced_dataset)} documents")
            self.steps_completed.append("processing")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data processing failed: {e}")
            return False
    
    def build_rag_index(self):
        """Build RAG index"""
        logger.info("\n🔍 Step 7: Building RAG index...")
        
        try:
            from src.retrieval.rag_system import AdvancedRAGSystem
            
            # Check if processed data exists
            enhanced_file = Path("data/enhanced_legal_dataset_v2.json")
            if not enhanced_file.exists():
                # Use original dataset
                enhanced_file = Path("data/expanded_legal_dataset.json")
            
            if not enhanced_file.exists():
                logger.warning("⚠️  No dataset found, skipping RAG index")
                self.steps_completed.append("rag_skipped")
                return True
            
            # Load documents
            with open(enhanced_file, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            # Initialize RAG
            logger.info("Initializing RAG system...")
            rag = AdvancedRAGSystem(use_gpu=False)  # Use CPU for setup
            
            # Add documents (limit for quick setup)
            logger.info(f"Indexing {min(len(documents), 1000)} documents...")
            rag.add_documents(documents[:1000], use_chromadb=True)
            
            # Save index
            rag.save_index("models/rag_index")
            
            logger.success("✅ RAG index built successfully")
            self.steps_completed.append("rag_index")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  RAG index build failed: {e}")
            logger.info("   You can build the index later")
            self.steps_completed.append("rag_failed")
            return True
    
    def create_config_file(self):
        """Create configuration file"""
        logger.info("\n⚙️  Step 8: Creating configuration...")
        
        config = {
            "version": "2.0.0",
            "models": {
                "legal_bert": "nlpaueb/legal-bert-base-uncased",
                "flan_t5": "google/flan-t5-base",
                "indic_bert": "ai4bharat/indic-bert",
                "use_8bit": True,
                "device": "auto"
            },
            "rag": {
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "faiss_weight": 0.4,
                "bm25_weight": 0.3,
                "chroma_weight": 0.3,
                "top_k": 10
            },
            "multilingual": {
                "supported_languages": ["en", "hi", "ta", "bn", "te", "mr", "gu", "kn", "ml", "pa"],
                "primary_translator": "deep"
            },
            "training": {
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "batch_size": 4,
                "learning_rate": 2e-4,
                "num_epochs": 3
            },
            "paths": {
                "data_dir": "data",
                "models_dir": "models",
                "cache_dir": "cache",
                "logs_dir": "logs"
            }
        }
        
        with open("config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.success("✅ Configuration file created")
        self.steps_completed.append("config")
        return True
    
    def run_tests(self):
        """Run basic tests"""
        logger.info("\n🧪 Step 9: Running tests...")
        
        try:
            # Test imports
            logger.info("Testing imports...")
            from src.generation.advanced_models import MultiModelLegalSystem
            from src.retrieval.rag_system import AdvancedRAGSystem
            from src.utils.multilingual import MultilingualLegalSystem
            from src.utils.evaluation_metrics import LegalEvaluationMetrics
            
            logger.success("✅ All imports successful")
            
            # Test basic functionality
            logger.info("Testing multilingual system...")
            multilingual = MultilingualLegalSystem()
            test_text = "This is a test"
            translated = multilingual.translate(test_text, target_lang='hi', source_lang='en')
            logger.debug(f"Translation test: {test_text} -> {translated}")
            
            logger.success("✅ Basic tests passed")
            self.steps_completed.append("tests")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  Some tests failed: {e}")
            logger.info("   The system may still work, but check the documentation")
            self.steps_completed.append("tests_partial")
            return True
    
    def generate_summary(self):
        """Generate setup summary"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 SETUP SUMMARY")
        logger.info("=" * 60)
        
        completed = len(self.steps_completed)
        logger.info(f"\n✅ Completed {completed} steps:\n")
        
        step_names = {
            "python_version": "Python version check",
            "dependencies": "Dependencies installation",
            "nlp_models": "NLP models download",
            "directories": "Directory structure",
            "datasets": "External datasets download",
            "datasets_skipped": "External datasets (skipped)",
            "datasets_partial": "External datasets (partial)",
            "processing": "Data processing",
            "processing_skipped": "Data processing (skipped)",
            "rag_index": "RAG index building",
            "rag_skipped": "RAG index (skipped)",
            "rag_failed": "RAG index (failed)",
            "config": "Configuration file",
            "tests": "System tests",
            "tests_partial": "System tests (partial)"
        }
        
        for step in self.steps_completed:
            logger.info(f"  ✓ {step_names.get(step, step)}")
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 SETUP COMPLETE!")
        logger.info("=" * 60)
        
        logger.info("\n📖 Next Steps:\n")
        logger.info("1. Review the configuration in config.json")
        logger.info("2. Read UPGRADE_TO_V2_GUIDE.md for features")
        logger.info("3. Run the application:")
        logger.info("   streamlit run legal_ai_app_enhanced.py --server.port 8504")
        logger.info("\n4. Or explore the new features:")
        logger.info("   python src/generation/advanced_models.py")
        logger.info("   python src/retrieval/rag_system.py")
        logger.info("   python src/utils/multilingual.py")
        
        logger.info("\n💡 Tips:")
        logger.info("  • Check README_V2.md for complete documentation")
        logger.info("  • Download more datasets: python src/data/dataset_downloader.py")
        logger.info("  • Train custom models: python src/training/lora_trainer.py")
        
        logger.info("\n📞 Support:")
        logger.info("  • Documentation: docs/")
        logger.info("  • Issues: GitHub Issues")
        logger.info("  • Community: Discord/Discussions")
        
        logger.info("\n" + "=" * 60)
    
    def run(self, download_datasets: bool = False):
        """Run complete setup"""
        steps = [
            (self.check_python_version, []),
            (self.install_dependencies, []),
            (self.download_nlp_models, []),
            (self.create_directory_structure, []),
            (self.download_datasets, [download_datasets]),
            (self.process_existing_data, []),
            (self.build_rag_index, []),
            (self.create_config_file, []),
            (self.run_tests, [])
        ]
        
        for step_func, args in steps:
            if not step_func(*args):
                logger.error(f"\n❌ Setup failed at: {step_func.__name__}")
                logger.info("Please fix the errors and run setup again")
                return False
        
        self.generate_summary()
        return True


def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="A-Qlegal 2.0 Setup Script")
    parser.add_argument(
        "--download-datasets",
        action="store_true",
        help="Download external datasets (requires time and bandwidth)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick setup (skip optional steps)"
    )
    
    args = parser.parse_args()
    
    setup = AQlegalSetup()
    
    if args.quick:
        logger.info("🚀 Running quick setup...")
        args.download_datasets = False
    
    success = setup.run(download_datasets=args.download_datasets)
    
    if success:
        logger.info("\n✅ Setup completed successfully!")
        sys.exit(0)
    else:
        logger.error("\n❌ Setup encountered errors")
        sys.exit(1)


if __name__ == "__main__":
    main()

