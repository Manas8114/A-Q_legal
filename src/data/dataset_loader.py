"""
Legal Dataset Loader for Constitution, CrPC, and IPC datasets
"""
import json
import jsonlines
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from loguru import logger


class LegalDatasetLoader:
    """Loads and merges legal QA datasets"""
    
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = Path(data_dir)
        self.datasets = {}
        
    def load_constitution_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """Load Constitution QA dataset"""
        logger.info(f"Loading Constitution dataset from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Normalize structure
        normalized_data = []
        for item in data:
            normalized_item = {
                'question': item.get('question', ''),
                'answer': item.get('answer', ''),
                'context': item.get('context', ''),
                'article': item.get('article', ''),
                'category': item.get('category', 'fact'),  # Preserve original category
                'source': 'constitution'
            }
            normalized_data.append(normalized_item)
        
        logger.info(f"Loaded {len(normalized_data)} Constitution Q&A pairs")
        return normalized_data
    
    def load_crpc_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """Load CrPC QA dataset"""
        logger.info(f"Loading CrPC dataset from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Normalize structure
        normalized_data = []
        for item in data:
            normalized_item = {
                'question': item.get('question', ''),
                'answer': item.get('answer', ''),
                'context': item.get('context', ''),
                'section': item.get('section', ''),
                'category': item.get('category', 'fact'),  # Preserve original category
                'source': 'crpc'
            }
            normalized_data.append(normalized_item)
        
        logger.info(f"Loaded {len(normalized_data)} CrPC Q&A pairs")
        return normalized_data
    
    def load_ipc_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """Load IPC QA dataset"""
        logger.info(f"Loading IPC dataset from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Normalize structure
        normalized_data = []
        for item in data:
            normalized_item = {
                'question': item.get('question', ''),
                'answer': item.get('answer', ''),
                'context': item.get('context', ''),
                'section': item.get('section', ''),
                'category': item.get('category', 'fact'),  # Preserve original category
                'source': 'ipc'
            }
            normalized_data.append(normalized_item)
        
        logger.info(f"Loaded {len(normalized_data)} IPC Q&A pairs")
        return normalized_data
    
    def merge_datasets(self, datasets: List[List[Dict[str, Any]]]) -> pd.DataFrame:
        """Merge multiple datasets into a single DataFrame"""
        logger.info("Merging datasets...")
        
        all_data = []
        for dataset in datasets:
            all_data.extend(dataset)
        
        df = pd.DataFrame(all_data)
        
        # Add unique IDs
        df['id'] = range(len(df))
        
        # Add metadata
        df['dataset_size'] = len(df)
        df['created_at'] = pd.Timestamp.now()
        
        logger.info(f"Merged dataset contains {len(df)} total Q&A pairs")
        logger.info(f"Category distribution: {df['category'].value_counts().to_dict()}")
        
        return df
    
    def save_merged_dataset(self, df: pd.DataFrame, output_path: str):
        """Save merged dataset to file"""
        logger.info(f"Saving merged dataset to {output_path}")
        df.to_json(output_path, orient='records', indent=2)
        logger.info("Dataset saved successfully")
    
    def load_merged_dataset(self, file_path: str) -> pd.DataFrame:
        """Load previously merged dataset"""
        logger.info(f"Loading merged dataset from {file_path}")
        df = pd.read_json(file_path)
        logger.info(f"Loaded {len(df)} Q&A pairs from merged dataset")
        return df
    
    def get_dataset_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about the dataset"""
        stats = {
            'total_qa_pairs': len(df),
            'categories': df['category'].value_counts().to_dict(),
            'sources': df['source'].value_counts().to_dict(),
            'avg_question_length': df['question'].str.len().mean(),
            'avg_answer_length': df['answer'].str.len().mean(),
            'questions_with_context': df['context'].notna().sum(),
            'questions_with_articles': df['article'].notna().sum(),
            'questions_with_sections': df['section'].notna().sum()
        }
        return stats