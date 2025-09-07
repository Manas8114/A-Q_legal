"""
Extractive answer model using BiLSTM + Attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModel
from loguru import logger
import pickle
from pathlib import Path


class BiLSTMAttentionModel(nn.Module):
    """BiLSTM with attention mechanism for extractive QA"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, 
                 num_layers: int = 2, dropout: float = 0.3):
        super(BiLSTMAttentionModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # BiLSTM layers
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Answer span prediction
        self.start_classifier = nn.Linear(hidden_dim * 2, 1)
        self.end_classifier = nn.Linear(hidden_dim * 2, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        # Embedding
        embedded = self.embedding(input_ids)
        
        # BiLSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Self-attention
        # Convert attention mask to boolean (1 -> False, 0 -> True for padding)
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = None
            
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, 
                                   key_padding_mask=key_padding_mask)
        
        # Combine LSTM and attention outputs
        combined = lstm_out + attn_out
        combined = self.dropout(combined)
        
        # Predict start and end positions
        start_logits = self.start_classifier(combined).squeeze(-1)
        end_logits = self.end_classifier(combined).squeeze(-1)
        
        return start_logits, end_logits


class ExtractiveAnswerModel:
    """Extractive answer model for legal QA"""
    
    def __init__(self, model_name: str = "bert-base-uncased", 
                 max_length: int = 512, device: str = None):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.is_trained = False
        
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """Load tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info(f"Loaded tokenizer: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}")
            logger.info("Using fallback tokenizer for extractive model")
            # Use a simple tokenizer as fallback
            self.tokenizer = None
    
    def _prepare_data(self, contexts: List[str], questions: List[str], 
                     answers: List[str] = None) -> Dict[str, torch.Tensor]:
        """Prepare data for training/inference"""
        # Tokenize contexts and questions
        inputs = self.tokenizer(
            questions,
            contexts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Add answer spans if provided
        if answers is not None:
            start_positions, end_positions = self._find_answer_spans(
                contexts, answers, inputs['input_ids']
            )
            inputs['start_positions'] = torch.tensor(start_positions, device=self.device)
            inputs['end_positions'] = torch.tensor(end_positions, device=self.device)
        
        return inputs
    
    def _find_answer_spans(self, contexts: List[str], answers: List[str], 
                          input_ids: torch.Tensor) -> Tuple[List[int], List[int]]:
        """Find answer spans in tokenized contexts"""
        start_positions = []
        end_positions = []
        
        for i, (context, answer) in enumerate(zip(contexts, answers)):
            # Tokenize answer
            answer_tokens = self.tokenizer.tokenize(answer)
            
            # Find answer in context
            context_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
            
            start_pos = -1
            end_pos = -1
            
            # Simple substring matching (can be improved)
            for j in range(len(context_tokens) - len(answer_tokens) + 1):
                if context_tokens[j:j+len(answer_tokens)] == answer_tokens:
                    start_pos = j
                    end_pos = j + len(answer_tokens) - 1
                    break
            
            start_positions.append(start_pos)
            end_positions.append(end_pos)
        
        return start_positions, end_positions
    
    def train(self, contexts: List[str], questions: List[str], answers: List[str],
              epochs: int = 3, batch_size: int = 8, learning_rate: float = 2e-5):
        """Train the extractive model"""
        logger.info(f"Training extractive model on {len(contexts)} examples")
        
        # Check if tokenizer is available
        if self.tokenizer is None:
            logger.warning("Tokenizer not available, skipping extractive model training")
            self.is_trained = False
            return
        
        # Prepare data
        try:
            train_data = self._prepare_data(contexts, questions, answers)
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            self.is_trained = False
            return
        
        # Create model
        vocab_size = self.tokenizer.vocab_size
        self.model = BiLSTMAttentionModel(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2
        ).to(self.device)
        
        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            successful_batches = 0
            
            # Create batches
            num_batches = len(contexts) // batch_size
            for i in range(0, len(contexts), batch_size):
                try:
                    batch_contexts = contexts[i:i+batch_size]
                    batch_questions = questions[i:i+batch_size]
                    batch_answers = answers[i:i+batch_size]
                    
                    # Prepare batch
                    batch_data = self._prepare_data(batch_contexts, batch_questions, batch_answers)
                
                    # Forward pass
                    start_logits, end_logits = self.model(
                        batch_data['input_ids'],
                        batch_data['attention_mask']
                    )
                    
                    # Compute loss
                    start_loss = F.cross_entropy(
                        start_logits, 
                        batch_data['start_positions'],
                        ignore_index=-1
                    )
                    end_loss = F.cross_entropy(
                        end_logits, 
                        batch_data['end_positions'],
                        ignore_index=-1
                    )
                    
                    loss = start_loss + end_loss
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    successful_batches += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to process batch {i//batch_size + 1}: {e}")
                    continue
            
            if successful_batches > 0:
                avg_loss = total_loss / successful_batches
                logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            else:
                logger.warning(f"Epoch {epoch+1}/{epochs}, No successful batches")
        
        self.is_trained = True
        logger.info("Extractive model training completed")
    
    def predict(self, contexts: List[str], questions: List[str]) -> List[Dict[str, Any]]:
        """Predict answers for given contexts and questions"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        results = []
        
        with torch.no_grad():
            # Prepare data
            inputs = self._prepare_data(contexts, questions)
            
            # Get predictions
            start_logits, end_logits = self.model(
                inputs['input_ids'],
                inputs['attention_mask']
            )
            
            # Convert to probabilities
            start_probs = F.softmax(start_logits, dim=-1)
            end_probs = F.softmax(end_logits, dim=-1)
            
            # Find best spans
            for i in range(len(contexts)):
                # Get start and end positions
                start_pos = torch.argmax(start_probs[i]).item()
                end_pos = torch.argmax(end_probs[i]).item()
                
                # Ensure valid span
                if start_pos > end_pos:
                    start_pos, end_pos = end_pos, start_pos
                
                # Extract answer
                answer_tokens = self.tokenizer.convert_ids_to_tokens(
                    inputs['input_ids'][i][start_pos:end_pos+1]
                )
                answer = self.tokenizer.convert_tokens_to_string(answer_tokens)
                
                # Calculate confidence
                start_confidence = start_probs[i][start_pos].item()
                end_confidence = end_probs[i][end_pos].item()
                confidence = (start_confidence + end_confidence) / 2
                
                result = {
                    'answer': answer,
                    'start_position': start_pos,
                    'end_position': end_pos,
                    'confidence': confidence,
                    'start_confidence': start_confidence,
                    'end_confidence': end_confidence
                }
                results.append(result)
        
        return results
    
    def predict_single(self, context: str, question: str) -> Dict[str, Any]:
        """Predict answer for a single context-question pair"""
        return self.predict([context], [question])[0]
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'max_length': self.max_length,
            'vocab_size': self.tokenizer.vocab_size
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Extractive model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_name = model_data['model_name']
        self.max_length = model_data['max_length']
        
        # Create model
        self.model = BiLSTMAttentionModel(
            vocab_size=model_data['vocab_size'],
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2
        ).to(self.device)
        
        # Load state dict
        self.model.load_state_dict(model_data['model_state_dict'])
        self.is_trained = True
        
        logger.info(f"Extractive model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'is_trained': self.is_trained,
            'model_name': self.model_name,
            'max_length': self.max_length,
            'device': self.device,
            'vocab_size': self.tokenizer.vocab_size if self.tokenizer else 0
        }