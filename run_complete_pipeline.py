#!/usr/bin/env python3
"""
🎯 Master Script - Complete Legal AI Pipeline
Runs the entire pipeline: merge → train → test
Author: A-Qlegal Team
Date: 2025
"""

import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner(text: str):
    """Print a styled banner"""
    banner = "=" * 80
    logger.info("\n" + banner)
    logger.info(text.center(80))
    logger.info(banner + "\n")


def run_script(script_name: str, description: str) -> bool:
    """Run a Python script and return success status"""
    print_banner(f"STEP: {description}")
    
    logger.info(f"📜 Running: {script_name}")
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"✅ {description} completed successfully!")
            logger.info(f"⏱️  Time taken: {elapsed_time:.2f} seconds\n")
            return True
        else:
            logger.error(f"❌ {description} failed with return code: {result.returncode}")
            logger.error(f"⏱️  Time taken: {elapsed_time:.2f} seconds\n")
            return False
            
    except FileNotFoundError:
        logger.error(f"❌ Script not found: {script_name}")
        return False
    except Exception as e:
        logger.error(f"❌ Error running {script_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_prerequisites() -> bool:
    """Check if all required files exist"""
    print_banner("CHECKING PREREQUISITES")
    
    BASE_DIR = Path(r"C:\Users\msgok\Desktop\A-Qlegal-main")
    
    required_files = [
        BASE_DIR / "data" / "enhanced_legal" / "enhanced_legal_documents.json",
        BASE_DIR / "data" / "indian_legal"
    ]
    
    all_exist = True
    
    for file_path in required_files:
        if file_path.exists():
            logger.info(f"✅ Found: {file_path}")
        else:
            logger.error(f"❌ Missing: {file_path}")
            all_exist = False
    
    if all_exist:
        logger.info("\n✅ All prerequisites met!\n")
    else:
        logger.error("\n❌ Some prerequisites are missing!\n")
    
    return all_exist


def check_trained_models() -> bool:
    """Check if trained models already exist"""
    BASE_DIR = Path(r"C:\Users\msgok\Desktop\A-Qlegal-main")
    
    classification_model = BASE_DIR / "models" / "legal_model" / "legal_classification_model"
    qa_model = BASE_DIR / "models" / "legal_model" / "legal_qa_model"
    
    return classification_model.exists() and qa_model.exists()


def main():
    """Main pipeline execution"""
    start_time = datetime.now()
    
    print_banner("🚀 A-QLEGAL COMPLETE PIPELINE")
    logger.info(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check if models already exist
    if check_trained_models():
        print_banner("✅ TRAINED MODELS FOUND!")
        logger.info("✅ Classification model exists")
        logger.info("✅ QA model exists")
        logger.info("\n💡 Skipping training and going straight to testing...\n")
        
        # Just test the existing models
        if not run_script("test_legal_model_enhanced.py", "Model Testing"):
            logger.error("❌ Model testing failed!")
            return False
    else:
        logger.info("⚠️  No trained models found. Running full pipeline...\n")
        
        # Check prerequisites
        if not check_prerequisites():
            logger.error("❌ Please ensure all required data files exist before running the pipeline.")
            return False
        
        # Step 1: Check if dataset exists
        BASE_DIR = Path(r"C:\Users\msgok\Desktop\A-Qlegal-main")
        dataset_path = BASE_DIR / "data" / "expanded_legal_dataset.json"
        
        if not dataset_path.exists():
            logger.info("📊 Dataset not found. Merging datasets...\n")
            if not run_script("merge_legal_datasets.py", "Dataset Merger"):
                logger.error("❌ Pipeline failed at dataset merging stage!")
                return False
        else:
            logger.info(f"✅ Dataset already exists: {dataset_path}\n")
        
        # Step 2: Train model
        if not run_script("train_legal_model.py", "Model Training"):
            logger.error("❌ Pipeline failed at model training stage!")
            logger.info("📝 Note: You can still use the merged dataset for other purposes.")
            return False
        
        # Step 3: Test model
        if not run_script("test_legal_model_enhanced.py", "Model Testing"):
            logger.error("❌ Pipeline failed at model testing stage!")
            logger.info("📝 Note: Your model may still be functional. Try testing manually.")
            return False
    
    # Calculate total time
    end_time = datetime.now()
    elapsed = end_time - start_time
    
    # Success message
    print_banner("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Ended: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total Time: {elapsed}")
    
    logger.info("\n" + "=" * 80)
    logger.info("📦 Generated Files:")
    logger.info("  1. data/merged_legal_dataset.json - Merged dataset")
    logger.info("  2. data/merge_statistics.json - Merger statistics")
    logger.info("  3. models/legal_model/legal_classification_model/ - Classification model")
    logger.info("  4. models/legal_model/legal_qa_model/ - QA model (if available)")
    logger.info("  5. models/legal_model/category_mapping.json - Category mapping")
    logger.info("=" * 80)
    
    logger.info("\n🎉 Your Legal AI models are ready to use!")
    logger.info("   Run test_legal_model.py again for interactive testing.\n")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Pipeline interrupted by user!")
        exit(1)
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

