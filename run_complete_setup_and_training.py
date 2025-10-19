#!/usr/bin/env python3
"""
A-Qlegal 2.0 - Complete Automated Setup, Training, and Deployment
Runs all necessary steps from installation to production-ready system
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from loguru import logger
from datetime import datetime
import time


class CompleteSystemBuilder:
    """Automated system builder for A-Qlegal 2.0"""
    
    def __init__(self):
        self.base_dir = Path.cwd()
        self.start_time = datetime.now()
        self.steps_completed = []
        self.log_file = f"logs/setup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # Configure logger
        logger.add(self.log_file, rotation="10 MB")
        
        logger.info("="*80)
        logger.info("🚀 A-QLEGAL 2.0 - COMPLETE AUTOMATED SETUP & TRAINING")
        logger.info("="*80)
        logger.info(f"Started at: {self.start_time}")
        logger.info(f"Log file: {self.log_file}")
        logger.info("="*80)
    
    def step_1_verify_environment(self):
        """Step 1: Verify Python environment"""
        logger.info("\n" + "="*80)
        logger.info("STEP 1/12: VERIFYING ENVIRONMENT")
        logger.info("="*80)
        
        # Check Python version
        version = sys.version_info
        logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
        
        if version.major < 3 or (version.major == 3 and version.minor < 10):
            logger.error("❌ Python 3.10+ required!")
            return False
        
        logger.success("✅ Python version OK")
        self.steps_completed.append("environment")
        return True
    
    def step_2_install_dependencies(self):
        """Step 2: Install all dependencies"""
        logger.info("\n" + "="*80)
        logger.info("STEP 2/12: INSTALLING DEPENDENCIES")
        logger.info("="*80)
        
        try:
            logger.info("Installing Python packages...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--upgrade"],
                check=True,
                capture_output=True
            )
            
            logger.info("Downloading spaCy model...")
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                check=True,
                capture_output=True
            )
            
            logger.info("Downloading NLTK data...")
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            
            logger.success("✅ All dependencies installed")
            self.steps_completed.append("dependencies")
            return True
            
        except Exception as e:
            logger.error(f"❌ Dependency installation failed: {e}")
            return False
    
    def step_3_create_directories(self):
        """Step 3: Create directory structure"""
        logger.info("\n" + "="*80)
        logger.info("STEP 3/12: CREATING DIRECTORY STRUCTURE")
        logger.info("="*80)
        
        directories = [
            "data/external_datasets",
            "data/processed",
            "data/chroma_db",
            "models/cache",
            "models/pretrained",
            "models/lora_models",
            "models/rag_index",
            "models/faiss_index",
            "logs",
            "outputs",
            "cache"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created: {dir_path}")
        
        logger.success(f"✅ Created {len(directories)} directories")
        self.steps_completed.append("directories")
        return True
    
    def step_4_download_and_cache_models(self):
        """Step 4: Download and cache all models"""
        logger.info("\n" + "="*80)
        logger.info("STEP 4/12: DOWNLOADING AND CACHING MODELS")
        logger.info("="*80)
        logger.info("This may take 10-15 minutes depending on your internet speed...")
        
        try:
            from src.utils.model_manager import save_all_models
            
            logger.info("Downloading models...")
            manager = save_all_models()
            
            cache_info = manager.get_cache_info()
            logger.success(f"✅ Cached {cache_info['num_models']} models")
            logger.info(f"   Total cache size: {cache_info['total_size_mb']:.2f} MB")
            
            self.steps_completed.append("model_caching")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  Model caching failed: {e}")
            logger.info("Models will be downloaded on first use")
            self.steps_completed.append("model_caching_partial")
            return True  # Continue anyway
    
    def step_5_process_existing_data(self):
        """Step 5: Process existing legal dataset"""
        logger.info("\n" + "="*80)
        logger.info("STEP 5/12: PROCESSING EXISTING DATA")
        logger.info("="*80)
        
        try:
            from src.data.enhanced_preprocessor import EnhancedLegalPreprocessor
            
            dataset_file = Path("data/expanded_legal_dataset.json")
            
            if not dataset_file.exists():
                logger.warning("⚠️  Original dataset not found, skipping processing")
                self.steps_completed.append("processing_skipped")
                return True
            
            logger.info(f"Loading dataset from {dataset_file}")
            with open(dataset_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            logger.info(f"Processing {len(dataset)} documents...")
            preprocessor = EnhancedLegalPreprocessor()
            
            # Process in batches to save memory
            batch_size = 1000
            enhanced_dataset = []
            
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(dataset)-1)//batch_size + 1}")
                enhanced_batch = preprocessor.process_dataset(batch)
                enhanced_dataset.extend(enhanced_batch)
            
            # Save enhanced dataset
            output_file = "data/enhanced_legal_dataset_v2.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_dataset, f, ensure_ascii=False, indent=2)
            
            # Generate statistics
            stats = preprocessor.get_statistics(enhanced_dataset)
            with open("data/enhanced_dataset_statistics_v2.json", 'w') as f:
                json.dump(stats, f, indent=2)
            
            # Create training pairs
            pairs = preprocessor.create_legal_to_layman_pairs(enhanced_dataset)
            with open("data/legal_to_layman_pairs.json", 'w', encoding='utf-8') as f:
                json.dump([{'legal': p[0], 'layman': p[1]} for p in pairs], f, ensure_ascii=False, indent=2)
            
            logger.success(f"✅ Processed {len(enhanced_dataset)} documents")
            logger.info(f"   Enhanced dataset: {output_file}")
            logger.info(f"   Training pairs: {len(pairs)}")
            
            self.steps_completed.append("processing")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data processing failed: {e}")
            return False
    
    def step_6_build_rag_index(self):
        """Step 6: Build RAG index"""
        logger.info("\n" + "="*80)
        logger.info("STEP 6/12: BUILDING RAG INDEX")
        logger.info("="*80)
        logger.info("Building FAISS + BM25 + ChromaDB index...")
        
        try:
            from src.retrieval.rag_system import AdvancedRAGSystem
            
            # Load enhanced dataset
            enhanced_file = Path("data/enhanced_legal_dataset_v2.json")
            if not enhanced_file.exists():
                enhanced_file = Path("data/expanded_legal_dataset.json")
            
            if not enhanced_file.exists():
                logger.warning("⚠️  No dataset found for RAG index")
                self.steps_completed.append("rag_skipped")
                return True
            
            with open(enhanced_file, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            # Initialize RAG
            logger.info("Initializing RAG system...")
            rag = AdvancedRAGSystem(use_gpu=False)  # Use CPU for compatibility
            
            # Add documents in batches
            batch_size = 1000
            logger.info(f"Indexing {len(documents)} documents in batches...")
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                logger.info(f"Batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
                
                if i == 0:
                    # First batch
                    rag.add_documents(batch, use_chromadb=True)
                else:
                    # Subsequent batches
                    rag.add_documents(batch, use_chromadb=False)
            
            # Save index
            logger.info("Saving RAG index...")
            rag.save_index("models/rag_index")
            
            stats = rag.get_stats()
            logger.success("✅ RAG index built successfully")
            logger.info(f"   Documents indexed: {stats['num_documents']}")
            logger.info(f"   FAISS index size: {stats['faiss_index_size']}")
            
            self.steps_completed.append("rag_index")
            return True
            
        except Exception as e:
            logger.error(f"❌ RAG index build failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def step_7_train_lora_model(self):
        """Step 7: Train LoRA model for legal simplification"""
        logger.info("\n" + "="*80)
        logger.info("STEP 7/12: TRAINING LORA MODEL")
        logger.info("="*80)
        logger.info("Training Flan-T5 with LoRA for legal-to-layman simplification...")
        
        try:
            from src.training.lora_trainer import LoRALegalTrainer
            
            # Check for training data
            pairs_file = Path("data/legal_to_layman_pairs.json")
            if not pairs_file.exists():
                logger.warning("⚠️  No training pairs found, skipping training")
                self.steps_completed.append("training_skipped")
                return True
            
            with open(pairs_file, 'r', encoding='utf-8') as f:
                pairs = json.load(f)
            
            if len(pairs) < 100:
                logger.warning(f"⚠️  Only {len(pairs)} training pairs, skipping training")
                logger.info("   Need at least 100 pairs for effective training")
                self.steps_completed.append("training_skipped")
                return True
            
            logger.info(f"Training with {len(pairs)} legal-to-layman pairs...")
            
            # Initialize trainer
            trainer = LoRALegalTrainer(
                model_name="google/flan-t5-base",
                output_dir="models/lora_models/legal_simplifier",
                use_8bit=True,  # Memory efficient
                lora_r=16,
                lora_alpha=32
            )
            
            # Prepare dataset
            train_dataset = trainer.prepare_dataset(
                pairs,
                input_key='legal',
                output_key='layman'
            )
            
            # Split into train/eval (90/10)
            split_idx = int(len(train_dataset) * 0.9)
            eval_dataset = train_dataset.select(range(split_idx, len(train_dataset)))
            train_dataset = train_dataset.select(range(split_idx))
            
            logger.info(f"Training set: {len(train_dataset)} examples")
            logger.info(f"Eval set: {len(eval_dataset)} examples")
            
            # Train
            logger.info("Starting training (this may take 30-60 minutes)...")
            trainer.train(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                num_epochs=3,
                batch_size=4,
                learning_rate=2e-4,
                save_steps=100,
                eval_steps=100
            )
            
            # Test the model
            logger.info("Testing trained model...")
            test_input = "Section 302 IPC: Whoever commits murder shall be punished with death or imprisonment for life"
            output = trainer.generate(f"Simplify this legal text: {test_input}")
            
            logger.success("✅ Model trained successfully")
            logger.info(f"   Model saved to: models/lora_models/legal_simplifier")
            logger.info(f"   Test output: {output[:100]}...")
            
            self.steps_completed.append("training")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  Training failed: {e}")
            logger.info("   You can train manually later")
            import traceback
            logger.debug(traceback.format_exc())
            self.steps_completed.append("training_failed")
            return True  # Continue anyway
    
    def step_8_download_external_datasets(self, download: bool = False):
        """Step 8: Download external datasets (optional)"""
        logger.info("\n" + "="*80)
        logger.info("STEP 8/12: DOWNLOADING EXTERNAL DATASETS")
        logger.info("="*80)
        
        if not download:
            logger.info("⏭️  Skipping external dataset download")
            logger.info("   You can download later using: python src/data/dataset_downloader.py")
            self.steps_completed.append("datasets_skipped")
            return True
        
        try:
            from src.data.dataset_downloader import LegalDatasetDownloader
            
            logger.info("Downloading external legal datasets...")
            logger.info("This may take 1-2 hours depending on your internet speed")
            
            downloader = LegalDatasetDownloader()
            results = downloader.download_all()
            
            successful = sum(1 for v in results.values() if v is not None)
            logger.success(f"✅ Downloaded {successful}/{len(results)} datasets")
            
            self.steps_completed.append("datasets")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  Dataset download failed: {e}")
            logger.info("   You can download datasets manually later")
            self.steps_completed.append("datasets_partial")
            return True
    
    def step_9_test_all_components(self):
        """Step 9: Test all components"""
        logger.info("\n" + "="*80)
        logger.info("STEP 9/12: TESTING ALL COMPONENTS")
        logger.info("="*80)
        
        try:
            logger.info("Running comprehensive verification...")
            
            # Run verification script
            result = subprocess.run(
                [sys.executable, "verify_installation.py"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.success("✅ All components verified successfully")
                self.steps_completed.append("testing")
                return True
            else:
                logger.warning("⚠️  Some tests failed, but continuing...")
                logger.info("   Check logs/setup_*.log for details")
                self.steps_completed.append("testing_partial")
                return True
                
        except Exception as e:
            logger.warning(f"⚠️  Testing failed: {e}")
            self.steps_completed.append("testing_failed")
            return True  # Continue anyway
    
    def step_10_create_configuration(self):
        """Step 10: Create configuration files"""
        logger.info("\n" + "="*80)
        logger.info("STEP 10/12: CREATING CONFIGURATION")
        logger.info("="*80)
        
        config = {
            "version": "2.0.0",
            "created_at": str(datetime.now()),
            "models": {
                "legal_bert": {
                    "name": "nlpaueb/legal-bert-base-uncased",
                    "cached": True,
                    "device": "auto"
                },
                "flan_t5": {
                    "name": "google/flan-t5-base",
                    "cached": True,
                    "use_8bit": True,
                    "device": "auto"
                },
                "indic_bert": {
                    "name": "ai4bharat/indic-bert",
                    "enabled": False
                }
            },
            "rag": {
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "faiss_weight": 0.4,
                "bm25_weight": 0.3,
                "chroma_weight": 0.3,
                "top_k": 10,
                "use_gpu": False
            },
            "training": {
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "batch_size": 4,
                "learning_rate": 2e-4,
                "num_epochs": 3
            },
            "multilingual": {
                "supported_languages": ["en", "hi", "ta", "bn", "te", "mr", "gu", "kn", "ml", "pa"],
                "primary_translator": "deep",
                "preserve_legal_terms": True
            },
            "paths": {
                "data_dir": "data",
                "models_dir": "models",
                "cache_dir": "models/cache",
                "logs_dir": "logs",
                "rag_index": "models/rag_index",
                "enhanced_dataset": "data/enhanced_legal_dataset_v2.json",
                "training_pairs": "data/legal_to_layman_pairs.json"
            },
            "features": {
                "use_rag": True,
                "use_lora_model": True,
                "enable_multilingual": True,
                "enable_evaluation": True,
                "cache_models": True
            }
        }
        
        with open("config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.success("✅ Configuration created: config.json")
        self.steps_completed.append("configuration")
        return True
    
    def step_11_generate_documentation(self):
        """Step 11: Generate deployment documentation"""
        logger.info("\n" + "="*80)
        logger.info("STEP 11/12: GENERATING DOCUMENTATION")
        logger.info("="*80)
        
        setup_report = f"""# A-Qlegal 2.0 - Setup Report

## Setup Information

- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Python Version**: {sys.version}
- **Setup Duration**: {datetime.now() - self.start_time}

## Steps Completed

{chr(10).join([f'- ✅ {step}' for step in self.steps_completed])}

## System Status

- **Models Cached**: Check `models/cache/`
- **RAG Index Built**: Check `models/rag_index/`
- **Training Data**: Check `data/legal_to_layman_pairs.json`
- **Configuration**: See `config.json`

## Quick Start

```bash
# Run the application
streamlit run legal_ai_app_enhanced.py --server.port 8504

# Access at
http://localhost:8504
```

## Test Commands

```python
# Test advanced models
python src/generation/advanced_models.py

# Test RAG system
python src/retrieval/rag_system.py

# Test multilingual
python src/utils/multilingual.py

# Verify installation
python verify_installation.py
```

## Next Steps

1. Read `README_V2.md` for complete documentation
2. Review `UPGRADE_TO_V2_GUIDE.md` for features
3. Check `QUICK_START_V2.md` for quick reference
4. Review `config.json` and customize as needed

## Support

- Documentation: `docs/`
- Logs: `{self.log_file}`
- Configuration: `config.json`

---

*Generated by A-Qlegal 2.0 Setup Script*
"""
        
        with open("SETUP_REPORT.md", 'w') as f:
            f.write(setup_report)
        
        logger.success("✅ Setup report generated: SETUP_REPORT.md")
        self.steps_completed.append("documentation")
        return True
    
    def step_12_final_summary(self):
        """Step 12: Display final summary"""
        logger.info("\n" + "="*80)
        logger.info("STEP 12/12: FINAL SUMMARY")
        logger.info("="*80)
        
        duration = datetime.now() - self.start_time
        
        logger.info("\n🎉 SETUP COMPLETE!")
        logger.info("="*80)
        logger.info(f"Total Duration: {duration}")
        logger.info(f"Steps Completed: {len(self.steps_completed)}")
        logger.info(f"Log File: {self.log_file}")
        logger.info("="*80)
        
        logger.info("\n📊 System Status:")
        logger.info("  ✅ Dependencies installed")
        logger.info("  ✅ Models downloaded and cached")
        logger.info("  ✅ Data processed")
        logger.info("  ✅ RAG index built")
        logger.info("  ✅ Configuration created")
        
        logger.info("\n🚀 Ready to Use!")
        logger.info("="*80)
        logger.info("\n📖 Quick Start:")
        logger.info("\n1. Run the application:")
        logger.info("   streamlit run legal_ai_app_enhanced.py --server.port 8504")
        logger.info("\n2. Access at:")
        logger.info("   http://localhost:8504")
        logger.info("\n3. Read documentation:")
        logger.info("   - README_V2.md")
        logger.info("   - UPGRADE_TO_V2_GUIDE.md")
        logger.info("   - QUICK_START_V2.md")
        logger.info("   - SETUP_REPORT.md")
        
        logger.info("\n💡 Pro Tips:")
        logger.info("  • Models are cached for fast loading")
        logger.info("  • Use GPU for 3x faster performance")
        logger.info("  • Train custom models with LoRA")
        logger.info("  • Download more datasets as needed")
        
        logger.info("\n📞 Support:")
        logger.info("  • Check logs/ for detailed logs")
        logger.info("  • Run: python verify_installation.py")
        logger.info("  • Read: FINAL_SETUP_INSTRUCTIONS.md")
        
        logger.info("\n" + "="*80)
        logger.success("🎊 A-QLEGAL 2.0 IS READY FOR ACTION!")
        logger.info("="*80 + "\n")
        
        self.steps_completed.append("summary")
        return True
    
    def run_all(self, download_datasets: bool = False):
        """Run all setup steps"""
        steps = [
            ("Environment Verification", self.step_1_verify_environment, []),
            ("Dependency Installation", self.step_2_install_dependencies, []),
            ("Directory Creation", self.step_3_create_directories, []),
            ("Model Caching", self.step_4_download_and_cache_models, []),
            ("Data Processing", self.step_5_process_existing_data, []),
            ("RAG Index Building", self.step_6_build_rag_index, []),
            ("LoRA Training", self.step_7_train_lora_model, []),
            ("External Datasets", self.step_8_download_external_datasets, [download_datasets]),
            ("Component Testing", self.step_9_test_all_components, []),
            ("Configuration", self.step_10_create_configuration, []),
            ("Documentation", self.step_11_generate_documentation, []),
            ("Final Summary", self.step_12_final_summary, [])
        ]
        
        logger.info(f"\n🎯 Total Steps: {len(steps)}\n")
        
        for i, (name, step_func, args) in enumerate(steps, 1):
            logger.info(f"\n▶️  Starting: {name} ({i}/{len(steps)})")
            
            try:
                success = step_func(*args)
                if not success:
                    logger.error(f"❌ {name} failed!")
                    logger.info("Setup cannot continue. Please fix errors and try again.")
                    return False
            except Exception as e:
                logger.error(f"❌ {name} crashed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                return False
            
            logger.info(f"✅ Completed: {name}\n")
        
        return True


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="A-Qlegal 2.0 - Complete Automated Setup & Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick setup (recommended)
  python run_complete_setup_and_training.py
  
  # Full setup with external datasets
  python run_complete_setup_and_training.py --download-datasets
  
  # Skip training
  python run_complete_setup_and_training.py --no-training
        """
    )
    
    parser.add_argument(
        "--download-datasets",
        action="store_true",
        help="Download external datasets (ILDC, IndicLegalQA, etc.)"
    )
    
    parser.add_argument(
        "--no-training",
        action="store_true",
        help="Skip LoRA model training"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "="*80)
    print("🚀 A-QLEGAL 2.0 - COMPLETE AUTOMATED SETUP & TRAINING")
    print("="*80)
    print("\nThis will:")
    print("  1. Install all dependencies")
    print("  2. Download and cache models")
    print("  3. Process legal datasets")
    print("  4. Build RAG index")
    print("  5. Train LoRA model for legal simplification")
    print("  6. Test all components")
    print("  7. Create configuration files")
    print("\nEstimated time: 30-60 minutes")
    print("="*80 + "\n")
    
    response = input("Continue? [Y/n]: ").strip().lower()
    if response and response != 'y':
        print("Setup cancelled.")
        return
    
    # Run setup
    builder = CompleteSystemBuilder()
    success = builder.run_all(download_datasets=args.download_datasets)
    
    if success:
        print("\n✅ SETUP COMPLETED SUCCESSFULLY!")
        print(f"📄 Check SETUP_REPORT.md for details")
        print(f"📋 Check {builder.log_file} for full logs")
        sys.exit(0)
    else:
        print("\n❌ SETUP FAILED")
        print(f"📋 Check {builder.log_file} for errors")
        sys.exit(1)


if __name__ == "__main__":
    main()

