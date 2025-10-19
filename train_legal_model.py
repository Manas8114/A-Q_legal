#!/usr/bin/env python3
"""
🎓 Legal Model Training Script
Trains a transformer-based model on the merged legal dataset.
Author: A-Qlegal Team
Date: 2025
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LegalDataset(Dataset):
    """Custom dataset for legal documents"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class LegalQADataset(Dataset):
    """Dataset for question answering tasks"""
    
    def __init__(self, questions: List[str], contexts: List[str], answers: List[str], 
                 tokenizer, max_length: int = 512):
        self.questions = questions
        self.contexts = contexts
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = str(self.questions[idx])
        context = str(self.contexts[idx])
        answer = str(self.answers[idx])
        
        # Tokenize question and context
        encoding = self.tokenizer(
            question,
            context,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation='only_second',  # Truncate context if too long
            return_tensors='pt'
        )
        
        # Find answer position in context (simplified)
        answer_start = context.find(answer)
        if answer_start == -1:
            answer_start = 0
            answer_end = 0
        else:
            answer_end = answer_start + len(answer)
        
        # Convert character positions to token positions (simplified)
        # In production, you'd use more sophisticated answer span detection
        answer_tokens = self.tokenizer(
            answer,
            add_special_tokens=False,
            return_tensors='pt'
        )
        
        start_positions = torch.tensor([answer_start], dtype=torch.long)
        end_positions = torch.tensor([answer_end], dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'start_positions': start_positions.flatten(),
            'end_positions': end_positions.flatten()
        }


class LegalModelTrainer:
    """Trainer for legal AI models"""
    
    def __init__(self, merged_dataset_path: str, output_dir: str):
        self.merged_dataset_path = Path(merged_dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️  Using device: {self.device}")
        
        if torch.cuda.is_available():
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Model configuration - IMPROVED with Legal-BERT
        self.model_name = "nlpaueb/legal-bert-base-uncased"  # Legal-specific BERT (12GB legal text)
        self.max_length = 256  # Reduced for 4GB GPU (from 512)
        self.batch_size = 16 if torch.cuda.is_available() else 4  # Increased for faster training
        self.num_epochs = 10  # Increased from 3 for better convergence on small dataset
        self.learning_rate = 1e-5  # Reduced for better fine-tuning
        self.warmup_ratio = 0.1  # Add warmup for stable training
        self.weight_decay = 0.01  # Add regularization
        
        # Data storage
        self.data = None
        self.categories = []
        self.category_to_id = {}
        self.id_to_category = {}
    
    def load_merged_dataset(self):
        """Load the merged dataset"""
        logger.info(f"📂 Loading merged dataset from: {self.merged_dataset_path}")
        
        try:
            with open(self.merged_dataset_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            
            logger.info(f"✅ Loaded {len(self.data)} entries")
            
            # Extract unique categories
            categories = set()
            for entry in self.data:
                category = entry.get('category', 'unknown')
                categories.add(category)
            
            self.categories = sorted(list(categories))
            self.category_to_id = {cat: idx for idx, cat in enumerate(self.categories)}
            self.id_to_category = {idx: cat for idx, cat in enumerate(self.categories)}
            
            logger.info(f"📊 Found {len(self.categories)} categories:")
            for cat in self.categories[:10]:
                count = sum(1 for entry in self.data if entry.get('category') == cat)
                logger.info(f"  - {cat}: {count} entries")
            
            if len(self.categories) > 10:
                logger.info(f"  ... and {len(self.categories) - 10} more categories")
            
            return True
            
        except FileNotFoundError:
            logger.error(f"❌ Merged dataset not found: {self.merged_dataset_path}")
            logger.error("   Please run merge_legal_datasets.py first!")
            return False
        except Exception as e:
            logger.error(f"❌ Error loading dataset: {e}")
            return False
    
    def prepare_classification_data(self) -> Tuple[List[str], List[int], List[str], List[int]]:
        """Prepare data for classification task with class balancing"""
        logger.info("🔧 Preparing classification data...")
        
        texts = []
        labels = []
        
        for entry in self.data:
            # Combine title and text for better context
            text = f"{entry.get('title', '')} {entry.get('text', '')}"
            texts.append(text)
            
            # Get category label
            category = entry.get('category', 'unknown')
            label = self.category_to_id.get(category, 0)
            labels.append(label)
        
        # Check class distribution
        from collections import Counter
        label_counts = Counter(labels)
        min_samples = min(label_counts.values())
        
        logger.info("📊 Class distribution:")
        for label_id, count in sorted(label_counts.items()):
            category = self.id_to_category[label_id]
            logger.info(f"   {category}: {count} samples")
        
        # Calculate class weights for imbalanced data (IMPROVEMENT)
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced',
            classes=np.array(sorted(label_counts.keys())),
            y=np.array(labels)
        )
        self.class_weights = torch.FloatTensor(class_weights)
        logger.info(f"✅ Computed class weights to handle imbalance")
        
        # Use stratify only if all classes have at least 2 samples
        stratify = labels if min_samples >= 2 else None
        
        if stratify is None:
            logger.warning(f"⚠️ Some categories have only 1 sample. Disabling stratification.")
        
        # Split into train/test (80-20)
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=stratify
        )
        
        logger.info(f"✅ Train size: {len(train_texts)}, Test size: {len(test_texts)}")
        
        return train_texts, train_labels, test_texts, test_labels
    
    def prepare_qa_data(self) -> Tuple[List, List, List, List, List, List]:
        """Prepare data for QA task"""
        logger.info("🔧 Preparing QA data...")
        
        questions = []
        contexts = []
        answers = []
        
        for entry in self.data:
            # Check if this entry has QA structure
            if 'metadata' in entry and 'question' in entry.get('metadata', {}):
                metadata = entry['metadata']
                questions.append(metadata.get('question', ''))
                contexts.append(metadata.get('context', entry.get('text', '')))
                answers.append(metadata.get('answer', ''))
            else:
                # Generate synthetic QA from title and text
                title = entry.get('title', '')
                text = entry.get('text', '')
                
                if title and text:
                    # Create a simple question from title
                    question = f"What is {title}?"
                    questions.append(question)
                    contexts.append(text)
                    # Use first sentence as answer
                    answer = text.split('.')[0] + '.' if '.' in text else text[:100]
                    answers.append(answer)
        
        if not questions:
            logger.warning("⚠️ No QA data found, skipping QA model training")
            return None, None, None, None, None, None
        
        # Split into train/test
        train_q, test_q, train_c, test_c, train_a, test_a = train_test_split(
            questions, contexts, answers, test_size=0.2, random_state=42
        )
        
        logger.info(f"✅ Train QA size: {len(train_q)}, Test QA size: {len(test_q)}")
        
        return train_q, train_c, train_a, test_q, test_c, test_a
    
    def train_classification_model(self, train_texts, train_labels, test_texts, test_labels):
        """Train classification model"""
        logger.info("=" * 80)
        logger.info("🎓 Training Classification Model")
        logger.info("=" * 80)
        
        try:
            # Load tokenizer and model
            logger.info(f"📥 Loading {self.model_name} model...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.categories),
                use_safetensors=True  # Use safetensors to bypass torch.load security restriction
            )
            model.to(self.device)
            
            # Create datasets
            logger.info("🔨 Creating datasets...")
            train_dataset = LegalDataset(train_texts, train_labels, tokenizer, self.max_length)
            test_dataset = LegalDataset(test_texts, test_labels, tokenizer, self.max_length)
            
            # Training arguments - IMPROVED
            training_args = TrainingArguments(
                output_dir=str(self.output_dir / "classification_checkpoints"),
                num_train_epochs=self.num_epochs,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                logging_dir=str(self.output_dir / "logs"),
                logging_steps=50,  # More frequent logging
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                warmup_ratio=self.warmup_ratio,  # Better warmup
                fp16=torch.cuda.is_available(),
                report_to="none",
                # Additional improvements
                gradient_accumulation_steps=2,  # Effective batch size x2
                max_grad_norm=1.0,  # Gradient clipping
                save_total_limit=2,  # Keep only best 2 checkpoints
            )
            
            # Custom trainer with class weights (IMPROVEMENT)
            from transformers import TrainerCallback
            
            class CustomTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                    labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    logits = outputs.logits
                    
                    # Use weighted loss for class imbalance
                    loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
                    loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
                    
                    return (loss, outputs) if return_outputs else loss
            
            # Create trainer with class weights
            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                tokenizer=tokenizer,
            )
            trainer.class_weights = self.class_weights
            
            logger.info("✅ Using weighted loss to handle class imbalance")
            
            # Train
            logger.info("🚀 Starting training...")
            train_result = trainer.train()
            
            logger.info("✅ Training completed!")
            logger.info(f"  Training Loss: {train_result.training_loss:.4f}")
            
            # Evaluate
            logger.info("📊 Evaluating model...")
            eval_result = trainer.evaluate()
            logger.info(f"  Evaluation Loss: {eval_result['eval_loss']:.4f}")
            
            # Save model
            model_path = self.output_dir / "legal_classification_model"
            logger.info(f"💾 Saving model to: {model_path}")
            trainer.save_model(str(model_path))
            tokenizer.save_pretrained(str(model_path))
            
            # Save category mapping
            mapping_path = self.output_dir / "category_mapping.json"
            with open(mapping_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'category_to_id': self.category_to_id,
                    'id_to_category': self.id_to_category,
                    'categories': self.categories
                }, f, indent=2)
            
            logger.info(f"💾 Saved category mapping to: {mapping_path}")
            logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training classification model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def train_qa_model(self, train_q, train_c, train_a, test_q, test_c, test_a):
        """Train QA model"""
        logger.info("=" * 80)
        logger.info("🎓 Training Question Answering Model")
        logger.info("=" * 80)
        
        if train_q is None:
            logger.warning("⚠️ No QA data available, skipping QA model training")
            return False
        
        try:
            # Load tokenizer and model
            logger.info(f"📥 Loading {self.model_name} for QA...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            model = AutoModelForQuestionAnswering.from_pretrained(
                self.model_name,
                use_safetensors=True  # Use safetensors to bypass torch.load security restriction
            )
            model.to(self.device)
            
            # Create datasets
            logger.info("🔨 Creating QA datasets...")
            train_dataset = LegalQADataset(train_q, train_c, train_a, tokenizer, self.max_length)
            test_dataset = LegalQADataset(test_q, test_c, test_a, tokenizer, self.max_length)
            
            # Training arguments - IMPROVED for QA
            training_args = TrainingArguments(
                output_dir=str(self.output_dir / "qa_checkpoints"),
                num_train_epochs=self.num_epochs,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                logging_dir=str(self.output_dir / "logs"),
                logging_steps=50,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                warmup_ratio=self.warmup_ratio,
                fp16=torch.cuda.is_available(),
                report_to="none",
                gradient_accumulation_steps=2,
                max_grad_norm=1.0,
                save_total_limit=2,
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                tokenizer=tokenizer,
            )
            
            # Train
            logger.info("🚀 Starting QA training...")
            train_result = trainer.train()
            
            logger.info("✅ QA Training completed!")
            logger.info(f"  Training Loss: {train_result.training_loss:.4f}")
            
            # Evaluate
            logger.info("📊 Evaluating QA model...")
            eval_result = trainer.evaluate()
            logger.info(f"  Evaluation Loss: {eval_result['eval_loss']:.4f}")
            
            # Save model
            model_path = self.output_dir / "legal_qa_model"
            logger.info(f"💾 Saving QA model to: {model_path}")
            trainer.save_model(str(model_path))
            tokenizer.save_pretrained(str(model_path))
            
            logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training QA model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def train_all(self):
        """Train all models"""
        logger.info("🚀 Starting Legal Model Training Pipeline")
        logger.info("=" * 80)
        
        # Load dataset
        if not self.load_merged_dataset():
            logger.error("❌ Failed to load dataset!")
            return False
        
        # Prepare classification data
        train_texts, train_labels, test_texts, test_labels = self.prepare_classification_data()
        
        # Train classification model
        if not self.train_classification_model(train_texts, train_labels, test_texts, test_labels):
            logger.error("❌ Classification model training failed!")
            return False
        
        # Prepare QA data
        train_q, train_c, train_a, test_q, test_c, test_a = self.prepare_qa_data()
        
        # Train QA model (optional, based on data availability)
        if train_q is not None:
            self.train_qa_model(train_q, train_c, train_a, test_q, test_c, test_a)
        
        logger.info("=" * 80)
        logger.info("✅ All training completed successfully!")
        logger.info("=" * 80)
        
        return True


def main():
    """Main execution function"""
    # Define paths
    BASE_DIR = Path(r"C:\Users\msgok\Desktop\A-Qlegal-main")
    MERGED_DATASET = BASE_DIR / "data" / "expanded_legal_dataset.json"
    OUTPUT_DIR = BASE_DIR / "models" / "legal_model"
    
    try:
        # Create trainer
        trainer = LegalModelTrainer(
            merged_dataset_path=str(MERGED_DATASET),
            output_dir=str(OUTPUT_DIR)
        )
        
        # Train all models
        success = trainer.train_all()
        
        if success:
            logger.info("✅ Legal model training completed successfully!")
            return True
        else:
            logger.error("❌ Training failed!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

