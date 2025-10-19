#!/usr/bin/env python3
"""
🚀 IMPROVED Legal Model Training Script v2.0
Enhanced with validation split, data augmentation, better scheduling, and ensemble support
Optimized for 4GB GPU with all accuracy improvements
Author: A-Qlegal Team
Date: 2025
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
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
    DataCollatorWithPadding,
    EarlyStoppingCallback
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
    """Custom dataset for legal documents with augmentation support"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 384):
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
                 tokenizer, max_length: int = 384):
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
            truncation='only_second',
            return_tensors='pt'
        )
        
        # Find answer position in context
        answer_start = context.find(answer)
        if answer_start == -1:
            answer_start = 0
            answer_end = 0
        else:
            answer_end = answer_start + len(answer)
        
        start_positions = torch.tensor([answer_start], dtype=torch.long)
        end_positions = torch.tensor([answer_end], dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'start_positions': start_positions.flatten(),
            'end_positions': end_positions.flatten()
        }


class DataAugmenter:
    """Data augmentation for text classification"""
    
    @staticmethod
    def simple_augment(text: str, n_aug: int = 1) -> List[str]:
        """Simple text augmentation using word replacement"""
        augmented = [text]
        
        # Synonym replacement (simple version without external libraries)
        words = text.split()
        
        for _ in range(n_aug):
            if len(words) > 3:
                # Random word shuffle (keep legal terms in place)
                import random
                legal_terms = ['section', 'article', 'ipc', 'crpc', 'constitution', 'act', 'law']
                
                new_words = words.copy()
                # Only shuffle non-legal terms
                indices = [i for i, w in enumerate(words) if w.lower() not in legal_terms]
                
                if len(indices) > 2:
                    random.shuffle(indices)
                    for i, idx in enumerate(indices[:2]):  # Shuffle only 2 words
                        if i + 1 < len(indices):
                            new_words[indices[i]], new_words[indices[i+1]] = new_words[indices[i+1]], new_words[indices[i]]
                    
                    augmented.append(' '.join(new_words))
        
        return augmented


class ImprovedLegalModelTrainer:
    """Enhanced trainer with all improvements"""
    
    def __init__(self, merged_dataset_path: str, output_dir: str, use_augmentation: bool = True):
        self.merged_dataset_path = Path(merged_dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️  Using device: {self.device}")
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"  Memory: {gpu_memory:.2f} GB")
            
            # Optimize for GPU memory
            if gpu_memory < 6:
                logger.info("  📊 Detected <6GB GPU - using memory-optimized settings")
        
        # IMPROVED Model configuration
        self.model_name = "nlpaueb/legal-bert-base-uncased"  # Legal-specific BERT
        
        # Adaptive max_length based on GPU memory
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 4
        self.max_length = 384 if gpu_mem >= 6 else 256  # 384 for 6GB+, 256 for 4GB
        
        self.batch_size = 8 if torch.cuda.is_available() else 4
        self.num_epochs = 20  # INCREASED from 10 for better convergence
        self.learning_rate = 2e-5  # OPTIMIZED learning rate
        self.warmup_ratio = 0.1
        self.weight_decay = 0.01
        self.use_augmentation = use_augmentation
        
        # Data storage
        self.data = None
        self.categories = []
        self.category_to_id = {}
        self.id_to_category = {}
        self.class_weights = None
        
        logger.info(f"✅ Configuration:")
        logger.info(f"   Max Length: {self.max_length} tokens")
        logger.info(f"   Batch Size: {self.batch_size}")
        logger.info(f"   Epochs: {self.num_epochs}")
        logger.info(f"   Learning Rate: {self.learning_rate}")
        logger.info(f"   Data Augmentation: {'Enabled' if use_augmentation else 'Disabled'}")
    
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
            
            logger.info(f"📊 Found {len(self.categories)} categories")
            
            return True
            
        except FileNotFoundError:
            logger.error(f"❌ Merged dataset not found: {self.merged_dataset_path}")
            return False
        except Exception as e:
            logger.error(f"❌ Error loading dataset: {e}")
            return False
    
    def augment_data(self, texts: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
        """Augment training data"""
        if not self.use_augmentation:
            return texts, labels
        
        logger.info("🔄 Applying data augmentation...")
        
        augmented_texts = []
        augmented_labels = []
        
        augmenter = DataAugmenter()
        
        for text, label in zip(texts, labels):
            # Add original
            augmented_texts.append(text)
            augmented_labels.append(label)
            
            # Add augmented versions (1 per sample)
            aug_texts = augmenter.simple_augment(text, n_aug=1)
            for aug_text in aug_texts[1:]:  # Skip first (original)
                augmented_texts.append(aug_text)
                augmented_labels.append(label)
        
        logger.info(f"✅ Augmented from {len(texts)} to {len(augmented_texts)} samples")
        
        return augmented_texts, augmented_labels
    
    def prepare_classification_data_with_validation(self) -> Dict[str, Any]:
        """Prepare data with VALIDATION SPLIT (70/15/15)"""
        logger.info("🔧 Preparing classification data with validation split...")
        
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
        for label_id, count in sorted(label_counts.items())[:10]:
            category = self.id_to_category[label_id]
            logger.info(f"   {category}: {count} samples")
        
        # Calculate class weights for imbalanced data
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced',
            classes=np.array(sorted(label_counts.keys())),
            y=np.array(labels)
        )
        self.class_weights = torch.FloatTensor(class_weights)
        logger.info(f"✅ Computed class weights to handle imbalance")
        
        # Use stratify only if all classes have at least 3 samples (for 3-way split)
        stratify = labels if min_samples >= 3 else None
        
        if stratify is None:
            logger.warning(f"⚠️ Some categories have <3 samples. Disabling stratification.")
        
        # IMPROVED: Three-way split (70% train, 15% val, 15% test)
        logger.info("✨ Using 70/15/15 train/val/test split...")
        
        # First split: 70% train, 30% temp
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=0.3, random_state=42, stratify=stratify
        )
        
        # Second split: 15% validation, 15% test (from 30% temp)
        temp_stratify = temp_labels if min_samples >= 3 else None
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_stratify
        )
        
        logger.info(f"✅ Data split:")
        logger.info(f"   Train: {len(train_texts)} samples (70%)")
        logger.info(f"   Validation: {len(val_texts)} samples (15%)")
        logger.info(f"   Test: {len(test_texts)} samples (15%)")
        
        # Apply data augmentation to training set only
        if self.use_augmentation:
            train_texts, train_labels = self.augment_data(train_texts, train_labels)
            logger.info(f"   Train (after augmentation): {len(train_texts)} samples")
        
        return {
            'train_texts': train_texts,
            'train_labels': train_labels,
            'val_texts': val_texts,
            'val_labels': val_labels,
            'test_texts': test_texts,
            'test_labels': test_labels
        }
    
    def train_classification_model(self, data_splits: Dict[str, Any]):
        """Train classification model with all improvements"""
        logger.info("=" * 80)
        logger.info("🎓 Training Classification Model (IMPROVED)")
        logger.info("=" * 80)
        
        try:
            # Load tokenizer and model
            logger.info(f"📥 Loading {self.model_name} model...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.categories),
                use_safetensors=True
            )
            model.to(self.device)
            
            # Create datasets
            logger.info("🔨 Creating datasets...")
            train_dataset = LegalDataset(
                data_splits['train_texts'], 
                data_splits['train_labels'], 
                tokenizer, 
                self.max_length
            )
            val_dataset = LegalDataset(
                data_splits['val_texts'], 
                data_splits['val_labels'], 
                tokenizer, 
                self.max_length
            )
            test_dataset = LegalDataset(
                data_splits['test_texts'], 
                data_splits['test_labels'], 
                tokenizer, 
                self.max_length
            )
            
            # IMPROVED Training arguments with cosine schedule and early stopping
            training_args = TrainingArguments(
                output_dir=str(self.output_dir / "classification_checkpoints"),
                num_train_epochs=self.num_epochs,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                logging_dir=str(self.output_dir / "logs"),
                logging_steps=50,
                
                # IMPROVED: Use validation set for evaluation
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                
                # IMPROVED: Cosine learning rate schedule
                warmup_ratio=self.warmup_ratio,
                lr_scheduler_type='cosine',  # Cosine decay
                
                # Optimization
                fp16=torch.cuda.is_available(),
                gradient_accumulation_steps=2,
                max_grad_norm=1.0,
                save_total_limit=3,  # Keep best 3 checkpoints
                
                report_to="none",
                
                # Additional improvements
                dataloader_num_workers=0,  # Avoid multiprocessing issues on Windows
                optim="adamw_torch",  # Use PyTorch AdamW
            )
            
            # Custom trainer with weighted loss for class imbalance
            class WeightedLossTrainer(Trainer):
                def __init__(self, *args, class_weights=None, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.class_weights = class_weights
                
                def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                    labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    logits = outputs.logits
                    
                    # Use weighted loss for class imbalance
                    loss_fct = torch.nn.CrossEntropyLoss(
                        weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
                    )
                    loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
                    
                    return (loss, outputs) if return_outputs else loss
            
            # Create trainer with early stopping
            trainer = WeightedLossTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,  # Use validation set
                tokenizer=tokenizer,
                class_weights=self.class_weights,
                callbacks=[
                    EarlyStoppingCallback(early_stopping_patience=3)  # Stop if no improvement for 3 epochs
                ]
            )
            
            logger.info("✅ Improvements applied:")
            logger.info("   - Validation set for monitoring")
            logger.info("   - Cosine learning rate schedule")
            logger.info("   - Weighted loss for class imbalance")
            logger.info("   - Early stopping (patience=3)")
            logger.info(f"   - Data augmentation: {self.use_augmentation}")
            
            # Train
            logger.info("🚀 Starting training...")
            train_result = trainer.train()
            
            logger.info("✅ Training completed!")
            logger.info(f"  Training Loss: {train_result.training_loss:.4f}")
            
            # Evaluate on validation set
            logger.info("📊 Evaluating on validation set...")
            val_result = trainer.evaluate(eval_dataset=val_dataset)
            logger.info(f"  Validation Loss: {val_result['eval_loss']:.4f}")
            
            # Final evaluation on test set
            logger.info("📊 Final evaluation on test set...")
            test_result = trainer.evaluate(eval_dataset=test_dataset)
            logger.info(f"  Test Loss: {test_result['eval_loss']:.4f}")
            
            # Save model
            model_path = self.output_dir / "legal_classification_model_v2"
            logger.info(f"💾 Saving improved model to: {model_path}")
            trainer.save_model(str(model_path))
            tokenizer.save_pretrained(str(model_path))
            
            # Save category mapping
            mapping_path = self.output_dir / "category_mapping_v2.json"
            with open(mapping_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'category_to_id': self.category_to_id,
                    'id_to_category': self.id_to_category,
                    'categories': self.categories,
                    'training_info': {
                        'epochs': self.num_epochs,
                        'learning_rate': self.learning_rate,
                        'max_length': self.max_length,
                        'batch_size': self.batch_size,
                        'augmentation': self.use_augmentation,
                        'train_loss': train_result.training_loss,
                        'val_loss': val_result['eval_loss'],
                        'test_loss': test_result['eval_loss']
                    }
                }, f, indent=2)
            
            logger.info(f"💾 Saved category mapping to: {mapping_path}")
            logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training classification model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def train_all(self):
        """Train all models with improvements"""
        logger.info("🚀 Starting IMPROVED Legal Model Training Pipeline")
        logger.info("=" * 80)
        
        # Load dataset
        if not self.load_merged_dataset():
            logger.error("❌ Failed to load dataset!")
            return False
        
        # Prepare classification data with validation split
        data_splits = self.prepare_classification_data_with_validation()
        
        # Train improved classification model
        if not self.train_classification_model(data_splits):
            logger.error("❌ Classification model training failed!")
            return False
        
        logger.info("=" * 80)
        logger.info("✅ IMPROVED training completed successfully!")
        logger.info("=" * 80)
        logger.info("📊 Summary of Improvements:")
        logger.info("   ✅ 70/15/15 train/val/test split (added validation set)")
        logger.info("   ✅ Increased epochs to 20 (from 10)")
        logger.info("   ✅ Cosine learning rate schedule")
        logger.info("   ✅ Weighted loss for class imbalance")
        logger.info("   ✅ Early stopping (patience=3)")
        logger.info(f"   ✅ Data augmentation: {self.use_augmentation}")
        logger.info(f"   ✅ Optimized max_length: {self.max_length} tokens")
        logger.info("=" * 80)
        
        return True


def main():
    """Main execution function"""
    # Define paths
    BASE_DIR = Path(r"C:\Users\msgok\Desktop\A-Qlegal-main")
    MERGED_DATASET = BASE_DIR / "data" / "expanded_legal_dataset.json"
    OUTPUT_DIR = BASE_DIR / "models" / "legal_model"
    
    try:
        # Create improved trainer
        trainer = ImprovedLegalModelTrainer(
            merged_dataset_path=str(MERGED_DATASET),
            output_dir=str(OUTPUT_DIR),
            use_augmentation=True  # Enable data augmentation
        )
        
        # Train all models
        success = trainer.train_all()
        
        if success:
            logger.info("✅ Improved legal model training completed successfully!")
            logger.info("📁 Models saved to: models/legal_model/legal_classification_model_v2/")
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


