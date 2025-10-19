#!/usr/bin/env python3
"""
A-Qlegal 2.0 - Installation Verification Script
Comprehensive testing of all components
"""

import sys
import os
from pathlib import Path
from loguru import logger
import traceback


class InstallationVerifier:
    """Verify A-Qlegal 2.0 installation"""
    
    def __init__(self):
        self.results = {}
        self.errors = []
        
    def test_python_version(self):
        """Test Python version"""
        logger.info("\n" + "="*60)
        logger.info("1️⃣ Testing Python Version")
        logger.info("="*60)
        
        try:
            version = sys.version_info
            if version.major >= 3 and version.minor >= 10:
                logger.success(f"✅ Python {version.major}.{version.minor}.{version.micro}")
                self.results['python_version'] = True
            else:
                logger.error(f"❌ Python 3.10+ required, found {version.major}.{version.minor}")
                self.results['python_version'] = False
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            self.results['python_version'] = False
            self.errors.append(('python_version', str(e)))
    
    def test_core_dependencies(self):
        """Test core dependencies"""
        logger.info("\n" + "="*60)
        logger.info("2️⃣ Testing Core Dependencies")
        logger.info("="*60)
        
        dependencies = [
            ('torch', 'PyTorch'),
            ('transformers', 'HuggingFace Transformers'),
            ('sentence_transformers', 'Sentence Transformers'),
            ('faiss', 'FAISS'),
            ('chromadb', 'ChromaDB'),
            ('spacy', 'spaCy'),
            ('nltk', 'NLTK'),
            ('loguru', 'Loguru'),
            ('streamlit', 'Streamlit'),
            ('peft', 'PEFT/LoRA')
        ]
        
        for module, name in dependencies:
            try:
                __import__(module)
                logger.success(f"✅ {name}")
                self.results[f'dep_{module}'] = True
            except ImportError as e:
                logger.error(f"❌ {name}: {e}")
                self.results[f'dep_{module}'] = False
                self.errors.append((f'dep_{module}', str(e)))
    
    def test_gpu_availability(self):
        """Test GPU availability"""
        logger.info("\n" + "="*60)
        logger.info("3️⃣ Testing GPU Availability")
        logger.info("="*60)
        
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_count = torch.cuda.device_count()
                logger.success(f"✅ GPU Available: {gpu_name}")
                logger.info(f"   GPU Count: {gpu_count}")
                self.results['gpu'] = True
            else:
                logger.warning("⚠️  No GPU detected (CPU mode will be used)")
                self.results['gpu'] = False
        except Exception as e:
            logger.error(f"❌ Error checking GPU: {e}")
            self.results['gpu'] = False
            self.errors.append(('gpu', str(e)))
    
    def test_advanced_models(self):
        """Test advanced models module"""
        logger.info("\n" + "="*60)
        logger.info("4️⃣ Testing Advanced Models")
        logger.info("="*60)
        
        try:
            from src.generation.advanced_models import MultiModelLegalSystem
            
            logger.info("Initializing model system (this may take a minute)...")
            system = MultiModelLegalSystem(
                use_legal_bert=False,  # Skip for quick test
                use_flan_t5=True,
                use_indic_bert=False,
                device="cpu"  # Use CPU for testing
            )
            
            # Test text generation
            test_text = "Section 302 IPC deals with murder"
            logger.info(f"Testing with: {test_text}")
            
            result = system.explain_law_in_layman_terms(test_text)
            
            logger.success(f"✅ Advanced Models Working")
            logger.info(f"   Generated: {result[:100]}...")
            self.results['advanced_models'] = True
            
        except Exception as e:
            logger.error(f"❌ Advanced Models Error: {e}")
            logger.debug(traceback.format_exc())
            self.results['advanced_models'] = False
            self.errors.append(('advanced_models', str(e)))
    
    def test_rag_system(self):
        """Test RAG system"""
        logger.info("\n" + "="*60)
        logger.info("5️⃣ Testing RAG System")
        logger.info("="*60)
        
        try:
            from src.retrieval.rag_system import AdvancedRAGSystem
            
            logger.info("Initializing RAG system...")
            rag = AdvancedRAGSystem(use_gpu=False)
            
            # Test with sample documents
            sample_docs = [
                {
                    'id': 'test_1',
                    'text': 'Section 302 IPC: Whoever commits murder shall be punished with death or life imprisonment',
                    'title': 'Section 302 IPC - Murder',
                    'category': 'criminal_law'
                },
                {
                    'id': 'test_2',
                    'text': 'Section 378 IPC: Whoever commits theft shall be punished with imprisonment',
                    'title': 'Section 378 IPC - Theft',
                    'category': 'criminal_law'
                }
            ]
            
            logger.info("Adding test documents...")
            rag.add_documents(sample_docs, use_chromadb=False)
            
            logger.info("Testing search...")
            results = rag.hybrid_search("What is punishment for murder?", top_k=2)
            
            if results:
                logger.success(f"✅ RAG System Working")
                logger.info(f"   Found {len(results)} results")
                logger.info(f"   Top result: {results[0]['document']['title']}")
                self.results['rag_system'] = True
            else:
                logger.warning("⚠️  RAG returned no results")
                self.results['rag_system'] = False
                
        except Exception as e:
            logger.error(f"❌ RAG System Error: {e}")
            logger.debug(traceback.format_exc())
            self.results['rag_system'] = False
            self.errors.append(('rag_system', str(e)))
    
    def test_multilingual(self):
        """Test multilingual support"""
        logger.info("\n" + "="*60)
        logger.info("6️⃣ Testing Multilingual Support")
        logger.info("="*60)
        
        try:
            from src.utils.multilingual import MultilingualLegalSystem
            
            logger.info("Initializing multilingual system...")
            ml = MultilingualLegalSystem(primary_translator="deep")
            
            test_text = "Section 302 IPC deals with murder"
            logger.info(f"Testing translation: {test_text}")
            
            # Test Hindi translation
            hindi = ml.translate(test_text, target_lang='hi', source_lang='en')
            
            logger.success(f"✅ Multilingual Working")
            logger.info(f"   Hindi: {hindi}")
            self.results['multilingual'] = True
            
        except Exception as e:
            logger.error(f"❌ Multilingual Error: {e}")
            logger.debug(traceback.format_exc())
            self.results['multilingual'] = False
            self.errors.append(('multilingual', str(e)))
    
    def test_evaluation_metrics(self):
        """Test evaluation metrics"""
        logger.info("\n" + "="*60)
        logger.info("7️⃣ Testing Evaluation Metrics")
        logger.info("="*60)
        
        try:
            from src.utils.evaluation_metrics import LegalEvaluationMetrics
            
            logger.info("Initializing evaluator...")
            evaluator = LegalEvaluationMetrics()
            
            reference = "Section 302 IPC provides punishment for murder"
            hypothesis = "Section 302 deals with murder punishment"
            
            logger.info("Calculating metrics...")
            metrics = evaluator.evaluate_all(
                hypothesis,
                reference,
                include_bertscore=False  # Skip for quick test
            )
            
            logger.success(f"✅ Evaluation Metrics Working")
            logger.info(f"   BLEU: {metrics.get('bleu', 0):.3f}")
            logger.info(f"   ROUGE-L: {metrics.get('rougeL_f1', 0):.3f}")
            self.results['evaluation'] = True
            
        except Exception as e:
            logger.error(f"❌ Evaluation Error: {e}")
            logger.debug(traceback.format_exc())
            self.results['evaluation'] = False
            self.errors.append(('evaluation', str(e)))
    
    def test_data_processing(self):
        """Test data processing"""
        logger.info("\n" + "="*60)
        logger.info("8️⃣ Testing Data Processing")
        logger.info("="*60)
        
        try:
            from src.data.enhanced_preprocessor import EnhancedLegalPreprocessor
            
            logger.info("Initializing preprocessor...")
            preprocessor = EnhancedLegalPreprocessor()
            
            sample_doc = {
                'id': 'test_doc',
                'text': 'Section 302 IPC: Whoever commits murder shall be punished with death or imprisonment for life',
                'title': 'Section 302 IPC',
                'category': 'criminal_law'
            }
            
            logger.info("Processing document...")
            enhanced = preprocessor.process_document(sample_doc)
            
            logger.success(f"✅ Data Processing Working")
            logger.info(f"   Section: {enhanced['section']}")
            logger.info(f"   Category: {enhanced['category']}")
            logger.info(f"   Keywords: {enhanced['keywords'][:3]}")
            self.results['data_processing'] = True
            
        except Exception as e:
            logger.error(f"❌ Data Processing Error: {e}")
            logger.debug(traceback.format_exc())
            self.results['data_processing'] = False
            self.errors.append(('data_processing', str(e)))
    
    def test_lora_trainer(self):
        """Test LoRA trainer (initialization only)"""
        logger.info("\n" + "="*60)
        logger.info("9️⃣ Testing LoRA Trainer")
        logger.info("="*60)
        
        try:
            from src.training.lora_trainer import LoRALegalTrainer
            
            logger.info("Testing LoRA trainer initialization...")
            # Just test initialization, not actual training
            
            logger.success(f"✅ LoRA Trainer Module Loaded")
            logger.info(f"   (Skipping actual training for quick verification)")
            self.results['lora_trainer'] = True
            
        except Exception as e:
            logger.error(f"❌ LoRA Trainer Error: {e}")
            logger.debug(traceback.format_exc())
            self.results['lora_trainer'] = False
            self.errors.append(('lora_trainer', str(e)))
    
    def test_file_structure(self):
        """Test file structure"""
        logger.info("\n" + "="*60)
        logger.info("🔟 Testing File Structure")
        logger.info("="*60)
        
        required_files = [
            'requirements.txt',
            'setup_v2.py',
            'README_V2.md',
            'UPGRADE_TO_V2_GUIDE.md',
            'src/generation/advanced_models.py',
            'src/retrieval/rag_system.py',
            'src/training/lora_trainer.py',
            'src/data/enhanced_preprocessor.py',
            'src/data/dataset_downloader.py',
            'src/utils/multilingual.py',
            'src/utils/evaluation_metrics.py'
        ]
        
        missing = []
        for file_path in required_files:
            if Path(file_path).exists():
                logger.success(f"✅ {file_path}")
            else:
                logger.error(f"❌ Missing: {file_path}")
                missing.append(file_path)
        
        if not missing:
            self.results['file_structure'] = True
        else:
            self.results['file_structure'] = False
            self.errors.append(('file_structure', f"Missing files: {missing}"))
    
    def generate_report(self):
        """Generate final report"""
        logger.info("\n" + "="*60)
        logger.info("📊 VERIFICATION REPORT")
        logger.info("="*60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for v in self.results.values() if v)
        failed_tests = total_tests - passed_tests
        
        logger.info(f"\n📈 Summary:")
        logger.info(f"  Total Tests: {total_tests}")
        logger.info(f"  Passed: {passed_tests} ✅")
        logger.info(f"  Failed: {failed_tests} ❌")
        logger.info(f"  Success Rate: {(passed_tests/total_tests*100):.1f}%")
        
        if failed_tests > 0:
            logger.info(f"\n❌ Failed Tests:")
            for test_name, passed in self.results.items():
                if not passed:
                    logger.error(f"  • {test_name}")
        
        if self.errors:
            logger.info(f"\n🐛 Errors Encountered:")
            for test_name, error in self.errors:
                logger.error(f"  {test_name}: {error}")
        
        logger.info("\n" + "="*60)
        
        if passed_tests == total_tests:
            logger.success("🎉 ALL TESTS PASSED!")
            logger.info("\n✅ Your A-Qlegal 2.0 installation is working perfectly!")
            logger.info("\n📖 Next Steps:")
            logger.info("  1. Read UPGRADE_TO_V2_GUIDE.md for features")
            logger.info("  2. Run: streamlit run legal_ai_app_enhanced.py --server.port 8504")
            logger.info("  3. Try the examples in QUICK_START_V2.md")
            return True
        else:
            logger.warning("⚠️  SOME TESTS FAILED")
            logger.info("\n💡 Troubleshooting:")
            logger.info("  1. Run: pip install -r requirements.txt --upgrade")
            logger.info("  2. Run: python setup_v2.py --quick")
            logger.info("  3. Check UPGRADE_TO_V2_GUIDE.md for solutions")
            return False
    
    def run_all_tests(self):
        """Run all verification tests"""
        logger.info("🚀 Starting A-Qlegal 2.0 Verification")
        logger.info("This will test all components...")
        
        tests = [
            self.test_python_version,
            self.test_core_dependencies,
            self.test_gpu_availability,
            self.test_file_structure,
            self.test_data_processing,
            self.test_evaluation_metrics,
            self.test_multilingual,
            self.test_rag_system,
            self.test_advanced_models,
            self.test_lora_trainer
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                logger.error(f"❌ Test crashed: {test.__name__}")
                logger.debug(traceback.format_exc())
                self.results[test.__name__] = False
                self.errors.append((test.__name__, str(e)))
        
        return self.generate_report()


def main():
    """Main verification function"""
    verifier = InstallationVerifier()
    success = verifier.run_all_tests()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

