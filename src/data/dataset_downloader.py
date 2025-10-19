"""
Dataset Downloader for A-Qlegal 2.0
Downloads and integrates external legal datasets
"""

import os
import json
import requests
from pathlib import Path
from typing import Dict, List, Any
from loguru import logger
from tqdm import tqdm
import kaggle
from huggingface_hub import hf_hub_download, list_repo_files
import gdown


class LegalDatasetDownloader:
    """Download and integrate external legal datasets"""
    
    def __init__(self, base_dir: str = "data/external_datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            'ildc': {
                'name': 'Indian Legal Documents Corpus (ILDC)',
                'source': 'huggingface',
                'repo': 'prakruthij/ILDC',
                'description': '35k Supreme Court judgments with outcomes',
                'size': '~2GB'
            },
            'indiclegal_qa': {
                'name': 'IndicLegalQA',
                'source': 'huggingface',
                'repo': 'ai4bharat/IndicLegalQA',
                'description': '10k QA pairs from Indian Supreme Court',
                'size': '~500MB'
            },
            'lawsum': {
                'name': 'LawSum',
                'source': 'custom',
                'url': 'https://arxiv.org/abs/2110.01638',
                'description': '10k judgments with summaries',
                'size': '~1GB'
            },
            'mildsum': {
                'name': 'MILDSum',
                'source': 'huggingface',
                'repo': 'csebuetnlp/mildsum',
                'description': 'Bilingual summaries (English + Hindi)',
                'size': '~300MB'
            },
            'tathyanyaya': {
                'name': 'TathyaNyaya',
                'source': 'custom',
                'url': 'https://github.com/Law-AI/TathyaNyaya',
                'description': 'Factual judgment prediction dataset',
                'size': '~200MB'
            },
            'justicehub': {
                'name': 'JusticeHub Datasets',
                'source': 'api',
                'base_url': 'https://justicehub.in/api',
                'description': 'Real court metadata and legal data',
                'size': 'Variable'
            },
            'opennyai': {
                'name': 'OpenNyAI Legal Corpus',
                'source': 'huggingface',
                'repo': 'opennyai/legal-bert-indian',
                'description': 'Legal texts for Indian law NLP',
                'size': '~500MB'
            },
            'indian_kanoon': {
                'name': 'Indian Kanoon Cases',
                'source': 'kaggle',
                'dataset': 'disisbig/indian-legal-dataset',
                'description': 'Scraped case data from Indian Kanoon',
                'size': '~1.5GB'
            }
        }
    
    def download_ildc(self) -> Path:
        """Download ILDC dataset from HuggingFace"""
        logger.info("📥 Downloading ILDC dataset...")
        
        try:
            output_dir = self.base_dir / "ildc"
            output_dir.mkdir(exist_ok=True)
            
            from datasets import load_dataset
            
            # Load ILDC dataset
            dataset = load_dataset("prakruthij/ILDC", split="train")
            
            # Save to JSON
            data_list = []
            for item in tqdm(dataset, desc="Processing ILDC"):
                data_list.append({
                    'id': f"ildc_{len(data_list)}",
                    'text': item.get('text', ''),
                    'summary': item.get('summary', ''),
                    'judgment': item.get('judgment', ''),
                    'category': 'supreme_court_judgment',
                    'source': 'ILDC',
                    'metadata': {
                        'court': item.get('court', 'Supreme Court'),
                        'date': item.get('date', ''),
                        'outcome': item.get('label', '')
                    }
                })
            
            output_file = output_dir / "ildc_processed.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data_list, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ ILDC downloaded: {len(data_list)} documents")
            return output_file
            
        except Exception as e:
            logger.error(f"❌ Error downloading ILDC: {e}")
            return None
    
    def download_indiclegal_qa(self) -> Path:
        """Download IndicLegalQA dataset"""
        logger.info("📥 Downloading IndicLegalQA...")
        
        try:
            output_dir = self.base_dir / "indiclegal_qa"
            output_dir.mkdir(exist_ok=True)
            
            from datasets import load_dataset
            
            dataset = load_dataset("ai4bharat/IndicLegalQA", split="train")
            
            data_list = []
            for item in tqdm(dataset, desc="Processing IndicLegalQA"):
                data_list.append({
                    'id': f"indiclegal_qa_{len(data_list)}",
                    'question': item.get('question', ''),
                    'answer': item.get('answer', ''),
                    'context': item.get('context', ''),
                    'category': 'legal_qa',
                    'source': 'IndicLegalQA',
                    'language': item.get('language', 'en'),
                    'metadata': {
                        'judgment_id': item.get('judgment_id', ''),
                        'court': item.get('court', '')
                    }
                })
            
            output_file = output_dir / "indiclegal_qa_processed.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data_list, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ IndicLegalQA downloaded: {len(data_list)} QA pairs")
            return output_file
            
        except Exception as e:
            logger.error(f"❌ Error downloading IndicLegalQA: {e}")
            return None
    
    def download_mildsum(self) -> Path:
        """Download MILDSum dataset (Bilingual summaries)"""
        logger.info("📥 Downloading MILDSum...")
        
        try:
            output_dir = self.base_dir / "mildsum"
            output_dir.mkdir(exist_ok=True)
            
            from datasets import load_dataset
            
            dataset = load_dataset("csebuetnlp/mildsum", split="train")
            
            data_list = []
            for item in tqdm(dataset, desc="Processing MILDSum"):
                data_list.append({
                    'id': f"mildsum_{len(data_list)}",
                    'text': item.get('document', ''),
                    'summary_en': item.get('summary_en', ''),
                    'summary_hi': item.get('summary_hi', ''),
                    'category': 'legal_summary',
                    'source': 'MILDSum',
                    'metadata': {
                        'language': 'bilingual',
                        'court': item.get('court', ''),
                        'date': item.get('date', '')
                    }
                })
            
            output_file = output_dir / "mildsum_processed.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data_list, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ MILDSum downloaded: {len(data_list)} documents")
            return output_file
            
        except Exception as e:
            logger.error(f"❌ Error downloading MILDSum: {e}")
            return None
    
    def download_opennyai(self) -> Path:
        """Download OpenNyAI legal corpus"""
        logger.info("📥 Downloading OpenNyAI corpus...")
        
        try:
            output_dir = self.base_dir / "opennyai"
            output_dir.mkdir(exist_ok=True)
            
            from datasets import load_dataset
            
            # OpenNyAI provides legal text corpus
            dataset = load_dataset("opennyai/legal-bert-indian", split="train")
            
            data_list = []
            for item in tqdm(dataset, desc="Processing OpenNyAI"):
                data_list.append({
                    'id': f"opennyai_{len(data_list)}",
                    'text': item.get('text', ''),
                    'category': item.get('category', 'legal_text'),
                    'source': 'OpenNyAI',
                    'metadata': {
                        'act': item.get('act', ''),
                        'section': item.get('section', '')
                    }
                })
            
            output_file = output_dir / "opennyai_processed.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data_list, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ OpenNyAI downloaded: {len(data_list)} documents")
            return output_file
            
        except Exception as e:
            logger.error(f"❌ Error downloading OpenNyAI: {e}")
            return None
    
    def download_indian_kanoon_kaggle(self) -> Path:
        """Download Indian Kanoon dataset from Kaggle"""
        logger.info("📥 Downloading Indian Kanoon from Kaggle...")
        
        try:
            output_dir = self.base_dir / "indian_kanoon"
            output_dir.mkdir(exist_ok=True)
            
            # Download using Kaggle API
            kaggle.api.dataset_download_files(
                'disisbig/indian-legal-dataset',
                path=str(output_dir),
                unzip=True
            )
            
            logger.info("✅ Indian Kanoon dataset downloaded")
            return output_dir
            
        except Exception as e:
            logger.error(f"❌ Error downloading Indian Kanoon: {e}")
            logger.info("💡 Make sure Kaggle API is configured: https://github.com/Kaggle/kaggle-api")
            return None
    
    def download_justicehub_data(self) -> Path:
        """Download data from JusticeHub API"""
        logger.info("📥 Fetching JusticeHub data...")
        
        try:
            output_dir = self.base_dir / "justicehub"
            output_dir.mkdir(exist_ok=True)
            
            # JusticeHub datasets (already in your data folder)
            # Just organize them
            justicehub_datasets = [
                'calpra_cases_2015_2023',
                'ccpd_cases',
                'central_acts_list',
                'child_protection_data_2023',
                'juvenile_justice_overview',
                'pcma_cases_2015_2023',
                'pocso_romantic_cases',
                'section_144_crpc'
            ]
            
            logger.info(f"✅ JusticeHub: {len(justicehub_datasets)} datasets available")
            return output_dir
            
        except Exception as e:
            logger.error(f"❌ Error accessing JusticeHub: {e}")
            return None
    
    def download_all(self) -> Dict[str, Path]:
        """Download all available datasets"""
        logger.info("🚀 Starting comprehensive dataset download...")
        
        results = {}
        
        # Download each dataset
        datasets_to_download = [
            ('ildc', self.download_ildc),
            ('indiclegal_qa', self.download_indiclegal_qa),
            ('mildsum', self.download_mildsum),
            ('opennyai', self.download_opennyai),
            ('indian_kanoon', self.download_indian_kanoon_kaggle),
            ('justicehub', self.download_justicehub_data),
        ]
        
        for name, download_func in datasets_to_download:
            logger.info(f"\n{'='*60}")
            logger.info(f"📦 Dataset: {self.datasets[name]['name']}")
            logger.info(f"📝 Description: {self.datasets[name]['description']}")
            logger.info(f"💾 Size: {self.datasets[name]['size']}")
            logger.info(f"{'='*60}\n")
            
            try:
                result = download_func()
                results[name] = result
                
                if result:
                    logger.success(f"✅ {name} downloaded successfully!")
                else:
                    logger.warning(f"⚠️  {name} download failed or skipped")
                    
            except Exception as e:
                logger.error(f"❌ Error downloading {name}: {e}")
                results[name] = None
        
        # Save download log
        log_file = self.base_dir / "download_log.json"
        with open(log_file, 'w') as f:
            json.dump({
                'datasets': {k: str(v) if v else None for k, v in results.items()},
                'timestamp': str(Path.cwd()),
                'total_downloaded': sum(1 for v in results.values() if v is not None)
            }, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🎉 Download Complete!")
        logger.info(f"✅ Successfully downloaded: {sum(1 for v in results.values() if v is not None)}/{len(results)}")
        logger.info(f"📁 Download log: {log_file}")
        logger.info(f"{'='*60}\n")
        
        return results
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about available datasets"""
        return self.datasets
    
    def check_downloaded(self) -> Dict[str, bool]:
        """Check which datasets have been downloaded"""
        status = {}
        
        for name in self.datasets.keys():
            dataset_dir = self.base_dir / name
            status[name] = dataset_dir.exists() and any(dataset_dir.iterdir())
        
        return status


def main():
    """Main function to download datasets"""
    downloader = LegalDatasetDownloader()
    
    # Show available datasets
    logger.info("📚 Available Datasets:")
    for name, info in downloader.get_dataset_info().items():
        logger.info(f"\n{name}:")
        logger.info(f"  Name: {info['name']}")
        logger.info(f"  Description: {info['description']}")
        logger.info(f"  Size: {info['size']}")
    
    # Check already downloaded
    logger.info("\n📋 Checking existing downloads...")
    status = downloader.check_downloaded()
    for name, downloaded in status.items():
        icon = "✅" if downloaded else "❌"
        logger.info(f"{icon} {name}: {'Downloaded' if downloaded else 'Not downloaded'}")
    
    # Download all
    logger.info("\n🚀 Starting download process...")
    results = downloader.download_all()
    
    logger.info("\n✅ All datasets processed!")
    return results


if __name__ == "__main__":
    main()

