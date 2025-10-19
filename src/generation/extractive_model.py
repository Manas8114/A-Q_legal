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
        """Find answer spans in tokenized contexts with improved matching"""
        start_positions = []
        end_positions = []
        
        for i, (context, answer) in enumerate(zip(contexts, answers)):
            # Tokenize answer
            answer_tokens = self.tokenizer.tokenize(answer)
            
            # Find answer in context
            context_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
            
            start_pos = -1
            end_pos = -1
            
            # Improved answer span detection
            if len(answer_tokens) > 0:
                # Method 1: Exact token matching
                for j in range(len(context_tokens) - len(answer_tokens) + 1):
                    if context_tokens[j:j+len(answer_tokens)] == answer_tokens:
                        start_pos = j
                        end_pos = j + len(answer_tokens) - 1
                        break
                
                # Method 2: Fuzzy matching if exact match fails
                if start_pos == -1:
                    best_match_score = 0
                    best_start = -1
                    best_end = -1
                    
                    for j in range(len(context_tokens) - len(answer_tokens) + 1):
                        context_chunk = context_tokens[j:j+len(answer_tokens)]
                        # Calculate similarity score
                        matches = sum(1 for a, c in zip(answer_tokens, context_chunk) if a.lower() == c.lower())
                        score = matches / len(answer_tokens)
                        
                        if score > best_match_score and score > 0.7:  # 70% similarity threshold
                            best_match_score = score
                            best_start = j
                            best_end = j + len(answer_tokens) - 1
                    
                    if best_start != -1:
                        start_pos = best_start
                        end_pos = best_end
                
                # Method 3: Keyword-based matching if still no match
                if start_pos == -1:
                    # Extract key terms from answer
                    answer_words = [token.lower().strip('.,!?;:') for token in answer_tokens if len(token.strip('.,!?;:')) > 2]
                    
                    if answer_words:
                        best_score = 0
                        best_start = -1
                        best_end = -1
                        
                        # Look for consecutive matches
                        for j in range(len(context_tokens) - len(answer_words) + 1):
                            context_chunk = [token.lower().strip('.,!?;:') for token in context_tokens[j:j+len(answer_words)]]
                            matches = sum(1 for a, c in zip(answer_words, context_chunk) if a == c)
                            score = matches / len(answer_words)
                            
                            if score > best_score and score > 0.5:  # 50% similarity threshold
                                best_score = score
                                best_start = j
                                best_end = j + len(answer_words) - 1
                        
                        if best_start != -1:
                            start_pos = best_start
                            end_pos = best_end
            
            # Fallback: Use first few tokens if no match found
            if start_pos == -1 and len(answer_tokens) > 0:
                # Find the first occurrence of any answer token
                for j, token in enumerate(context_tokens):
                    if token.lower() in [t.lower() for t in answer_tokens]:
                        start_pos = j
                        end_pos = min(j + len(answer_tokens) - 1, len(context_tokens) - 1)
                        break
            
            # Final fallback: Use middle of context
            if start_pos == -1:
                mid_point = len(context_tokens) // 2
                start_pos = max(0, mid_point - 2)
                end_pos = min(len(context_tokens) - 1, mid_point + 2)
            
            start_positions.append(start_pos)
            end_positions.append(end_pos)
        
        return start_positions, end_positions
    
    def train(self, contexts: List[str], questions: List[str], answers: List[str],
              epochs: int = 3, batch_size: int = 8, learning_rate: float = 2e-5,
              gradient_accumulation_steps: int = 2):
        """Train the extractive model with memory optimization"""
        logger.info(f"Training extractive model on {len(contexts)} examples")
        
        # Check if tokenizer is available
        if self.tokenizer is None:
            logger.warning("Tokenizer not available, skipping extractive model training")
            self.is_trained = False
            return
        
        # Create model with smaller dimensions for memory efficiency
        vocab_size = self.tokenizer.vocab_size
        self.model = BiLSTMAttentionModel(
            vocab_size=vocab_size,
            embedding_dim=64,  # Reduced from 128
            hidden_dim=128,    # Reduced from 256
            num_layers=1       # Reduced from 2
        ).to(self.device)
        
        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Enable mixed precision for memory efficiency
        scaler = torch.cuda.amp.GradScaler() if self.device == "cuda" else None
        
        # Training loop with memory optimization
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            successful_batches = 0
            optimizer.zero_grad()
            
            # Create batches with smaller batch size
            effective_batch_size = max(1, batch_size // 2)  # Reduce batch size
            num_batches = len(contexts) // effective_batch_size
            
            for i in range(0, len(contexts), effective_batch_size):
                try:
                    batch_contexts = contexts[i:i+effective_batch_size]
                    batch_questions = questions[i:i+effective_batch_size]
                    batch_answers = answers[i:i+effective_batch_size]
                    
                    # Prepare batch with shorter sequences
                    batch_data = self._prepare_data(batch_contexts, batch_questions, batch_answers)
                    
                    # Limit sequence length for memory efficiency
                    max_seq_len = 512
                    if batch_data['input_ids'].size(1) > max_seq_len:
                        batch_data['input_ids'] = batch_data['input_ids'][:, :max_seq_len]
                        batch_data['attention_mask'] = batch_data['attention_mask'][:, :max_seq_len]
                        # Adjust positions if they exceed the limit
                        batch_data['start_positions'] = torch.clamp(batch_data['start_positions'], 0, max_seq_len-1)
                        batch_data['end_positions'] = torch.clamp(batch_data['end_positions'], 0, max_seq_len-1)
                
                    # Forward pass with mixed precision
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
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
                            
                            loss = (start_loss + end_loss) / gradient_accumulation_steps
                        
                        scaler.scale(loss).backward()
                    else:
                        # CPU training
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
                        
                        loss = (start_loss + end_loss) / gradient_accumulation_steps
                        loss.backward()
                    
                    total_loss += loss.item() * gradient_accumulation_steps
                    successful_batches += 1
                    
                    # Gradient accumulation
                    if (i // effective_batch_size + 1) % gradient_accumulation_steps == 0:
                        if scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad()
                        
                        # Clear GPU cache periodically
                        if self.device == "cuda" and (i // effective_batch_size + 1) % 50 == 0:
                            torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.warning(f"Failed to process batch {i//effective_batch_size + 1}: {e}")
                    continue
            
            if successful_batches > 0:
                avg_loss = total_loss / successful_batches
                logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            else:
                logger.warning(f"Epoch {epoch+1}/{epochs}, No successful batches")
            
            # Clear GPU cache after each epoch
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        self.is_trained = True
        logger.info("Extractive model training completed")
    
    def predict(self, contexts: List[str], questions: List[str]) -> List[Dict[str, Any]]:
        """Predict answers for given contexts and questions with improved extraction"""
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
            
            # Find best spans with improved logic
            for i in range(len(contexts)):
                # Get top candidates for start and end positions
                top_k = min(5, len(start_probs[i]))
                start_candidates = torch.topk(start_probs[i], top_k)
                end_candidates = torch.topk(end_probs[i], top_k)
                
                best_answer = ""
                best_confidence = 0
                best_start = 0
                best_end = 0
                
                # Try different combinations of start and end positions
                for start_idx in start_candidates.indices:
                    for end_idx in end_candidates.indices:
                        start_pos = start_idx.item()
                        end_pos = end_idx.item()
                        
                        # Ensure valid span
                        if start_pos > end_pos:
                            start_pos, end_pos = end_pos, start_pos
                        
                        # Skip if span is too short (but allow longer spans for complete answers)
                        span_length = end_pos - start_pos + 1
                        if span_length < 2 or span_length > 100:  # Increased from 50 to 100
                            continue
                        
                        # Extract answer with better token handling
                        answer_tokens = self.tokenizer.convert_ids_to_tokens(
                            inputs['input_ids'][i][start_pos:end_pos+1]
                        )
                        
                        # Filter out special tokens before converting to string
                        filtered_tokens = []
                        for token in answer_tokens:
                            if token not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:
                                filtered_tokens.append(token)
                        
                        answer = self.tokenizer.convert_tokens_to_string(filtered_tokens)
                        
                        # Try to expand answer for completeness
                        expanded_answer = self._expand_answer_for_completeness(
                            answer, contexts[i], start_pos, end_pos, inputs['input_ids'][i]
                        )
                        if expanded_answer:
                            answer = expanded_answer
                        
                        # Clean up answer
                        answer = self._clean_extracted_answer(answer)
                        
                        # Skip if answer is too short (but be more lenient)
                        if len(answer.strip()) < 5:  # Reduced from 10 to 5
                            continue
                        
                        # Calculate confidence with improved scoring
                        start_confidence = start_probs[i][start_pos].item()
                        end_confidence = end_probs[i][end_pos].item()
                        
                        # Base confidence: weighted average with emphasis on start position
                        confidence = (start_confidence * 0.6 + end_confidence * 0.4)
                        
                        # Quality-based bonuses
                        # Length bonus: more substantial for longer answers
                        if len(answer) > 20:
                            length_bonus = min(0.2, (len(answer) - 20) / 200)  # Up to 0.2 bonus
                            confidence += length_bonus
                        
                        # Completeness bonus: check for complete sentences
                        sentence_count = answer.count('.') + answer.count('!') + answer.count('?')
                        if sentence_count > 0:
                            confidence += min(0.1, sentence_count * 0.02)  # Up to 0.1 bonus
                        
                        # Coherence bonus: check for proper sentence structure
                        if answer.count(' ') > 3 and not answer.startswith(('the', 'a', 'an', 'and', 'or', 'but')):
                            confidence += 0.05
                        
                        # Penalty for very short answers
                        if len(answer) < 10:
                            confidence *= 0.8
                        
                        # Check if this is the best answer so far
                        if confidence > best_confidence:
                            best_answer = answer
                            best_confidence = confidence
                            best_start = start_pos
                            best_end = end_pos
                
                # If no good answer found, use fallback
                if not best_answer or best_confidence < 0.3:
                    best_answer = self._fallback_answer_extraction(contexts[i], questions[i])
                    best_confidence = 0.3
                    best_start = 0
                    best_end = 0
                
                result = {
                    'answer': best_answer,
                    'start_position': best_start,
                    'end_position': best_end,
                    'confidence': min(0.95, best_confidence),  # Cap confidence at 95%
                    'start_confidence': start_probs[i][best_start].item() if best_start < len(start_probs[i]) else 0,
                    'end_confidence': end_probs[i][best_end].item() if best_end < len(end_probs[i]) else 0,
                    'extraction_method': 'model' if best_answer else 'fallback'
                }
                results.append(result)
        
        return results
    
    def _clean_extracted_answer(self, answer: str) -> str:
        """Clean and improve extracted answer"""
        # Remove leading/trailing whitespace
        answer = answer.strip()
        
        # Remove BERT special tokens and padding
        special_tokens = ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']
        for token in special_tokens:
            answer = answer.replace(token, '')
        
        # Remove multiple spaces and clean up
        answer = ' '.join(answer.split())
        
        # Remove incomplete sentences at the beginning
        sentences = answer.split('.')
        if len(sentences) > 1 and len(sentences[0].strip()) < 5:
            answer = '.'.join(sentences[1:]).strip()
        
        # Remove incomplete sentences at the end
        if answer.endswith('...') or answer.endswith('..'):
            answer = answer[:-3].strip()
        
        # Clean up any remaining special characters at the beginning/end
        while answer and answer[0] in ['[', ']', '#', '##']:
            answer = answer[1:].strip()
        while answer and answer[-1] in ['[', ']', '#', '##']:
            answer = answer[:-1].strip()
        
        # Try to complete incomplete sentences at the end
        if answer and not answer.endswith(('.', '!', '?')):
            last_sentence = answer.split('.')[-1].strip()
            if len(last_sentence) > 10:  # If last part is substantial, keep it
                answer += '.'
            else:
                # Remove the incomplete last sentence
                sentences = answer.split('.')
                if len(sentences) > 1:
                    answer = '.'.join(sentences[:-1]).strip()
                    if answer and not answer.endswith(('.', '!', '?')):
                        answer += '.'
        
        # Final cleanup: remove any remaining artifacts
        answer = ' '.join(answer.split())
        
        return answer
    
    def _expand_answer_for_completeness(self, answer: str, context: str, start_pos: int, 
                                      end_pos: int, input_ids: torch.Tensor) -> str:
        """Try to expand answer to make it more complete"""
        # If answer doesn't end with punctuation, try to extend it
        if not answer.endswith(('.', '!', '?')):
            # Try to extend to the next sentence boundary
            context_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            
            # Look for sentence boundaries after the current end position
            for i in range(end_pos + 1, min(end_pos + 20, len(context_tokens))):  # Look ahead up to 20 tokens
                token = context_tokens[i]
                if token in ['.', '!', '?']:
                    # Extend answer to include this sentence boundary
                    extended_tokens = self.tokenizer.convert_ids_to_tokens(
                        input_ids[start_pos:i+1]
                    )
                    extended_answer = self.tokenizer.convert_tokens_to_string(extended_tokens)
                    return extended_answer
            
            # If no sentence boundary found, try to extend by a few more words
            if end_pos + 5 < len(context_tokens):
                extended_tokens = self.tokenizer.convert_ids_to_tokens(
                    input_ids[start_pos:end_pos+6]  # Extend by 5 more tokens
                )
                extended_answer = self.tokenizer.convert_tokens_to_string(extended_tokens)
                return extended_answer
        
        return answer
    
    def _fallback_answer_extraction(self, context: str, question: str) -> str:
        """Fallback method to extract answer when model fails"""
        import re
        
        # Extract key terms from question
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        
        # Find sentences that contain question keywords
        sentences = re.split(r'[.!?]+', context)
        best_sentences = []
        
        for sentence in sentences:
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            overlap = len(question_words.intersection(sentence_words))
            if overlap > 0:
                best_sentences.append((overlap, sentence.strip()))
        
        # Sort by overlap and take best sentences
        best_sentences.sort(key=lambda x: x[0], reverse=True)
        
        if best_sentences:
            # Combine top 3 sentences for more complete answers
            answer_parts = [sent for _, sent in best_sentences[:3]]
            answer = '. '.join(answer_parts)
            if not answer.endswith('.'):
                answer += '.'
            return answer
        else:
            # Return first 2 sentences as fallback for more complete answer
            if len(sentences) >= 2:
                first_two = '. '.join([sentences[0].strip(), sentences[1].strip()])
                return first_two + '.' if not first_two.endswith('.') else first_two
            else:
                first_sentence = sentences[0].strip() if sentences else context[:150]  # Increased from 100
                return first_sentence + '.' if not first_sentence.endswith('.') else first_sentence
    
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