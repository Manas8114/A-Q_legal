"""
Generative answer model using T5/LLaMA/Gemini
"""
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any, Optional
from loguru import logger
import pickle
from pathlib import Path
from .gemini_model import GeminiGenerativeModel


class GenerativeAnswerModel:
    """Generative answer model for legal QA using T5, LLaMA, or Gemini"""
    
    def __init__(self, model_name: str = "t5-small", 
                 max_length: int = 512, device: str = None, api_key: str = None):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.api_key = api_key
        self.tokenizer = None
        self.model = None
        self.gemini_model = None
        self.is_trained = False
        self.use_gemini = False
        
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer"""
        # Prioritize local models first
        logger.info(f"Loading local generative model: {self.model_name}")
        
        # Load local model (GPT-2/T5/LLaMA) first
        try:
            if "t5" in self.model_name.lower():
                self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
                self.model = T5ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True
                )
            else:
                # For GPT-2, LLaMA or other causal models
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                
                # Add padding token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True
                )
            
            self.model.to(self.device)
            logger.info(f"Loaded local generative model: {self.model_name}")
            self.is_trained = True
        except Exception as e:
            logger.warning(f"Failed to load local model {self.model_name}: {e}")
            
            # Only try Gemini as fallback if explicitly requested and API key available
            if "gemini" in self.model_name.lower() and self.api_key:
                logger.info("Attempting to fallback to Gemini API...")
                try:
                    self.gemini_model = GeminiGenerativeModel(
                        api_key=self.api_key,
                        model_name=self.model_name
                    )
                    if self.gemini_model.is_available:
                        self.use_gemini = True
                        self.is_trained = True
                        logger.info(f"Using Gemini model as fallback: {self.gemini_model.model_name}")
                        return
                    else:
                        logger.warning("Gemini model not available")
                except Exception as gemini_error:
                    logger.warning(f"Failed to initialize Gemini model: {gemini_error}")
            
            logger.info("Using fallback generative model")
            # Set to None to indicate fallback mode
            self.tokenizer = None
            self.model = None
    
    def _prepare_input(self, context: str, question: str) -> str:
        """Prepare input for the model"""
        if "t5" in self.model_name.lower():
            # T5 format
            input_text = f"question: {question} context: {context}"
        else:
            # LLaMA format
            input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
        
        return input_text
    
    def generate_answer(self, context: str, question: str, 
                       max_new_tokens: int = 100,
                       temperature: float = 0.7,
                       do_sample: bool = True) -> Dict[str, Any]:
        """Generate answer for a single context-question pair"""
        # Use Gemini if available
        if self.use_gemini and self.gemini_model:
            return self.gemini_model.generate_answer(context, question, max_new_tokens, temperature)
        
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained or not loaded, using base model")
        
        # Check if model is available
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model must be loaded before making predictions")
        
        # Prepare input
        input_text = self._prepare_input(context, question)
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            if "t5" in self.model_name.lower():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            else:
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (remove input text)
        if "t5" in self.model_name.lower():
            answer = generated_text
        else:
            # For LLaMA, extract only the answer part
            answer_start = generated_text.find("Answer:") + len("Answer:")
            answer = generated_text[answer_start:].strip()
        
        return {
            'answer': answer,
            'generated_text': generated_text,
            'input_text': input_text,
            'confidence': 0.8  # Placeholder confidence
        }
    
    def generate_answers_batch(self, contexts: List[str], questions: List[str],
                              max_new_tokens: int = 100,
                              temperature: float = 0.7,
                              do_sample: bool = True) -> List[Dict[str, Any]]:
        """Generate answers for a batch of context-question pairs"""
        results = []
        
        for context, question in zip(contexts, questions):
            result = self.generate_answer(
                context, question, max_new_tokens, temperature, do_sample
            )
            results.append(result)
        
        return results
    
    def fine_tune(self, contexts: List[str], questions: List[str], answers: List[str],
                  epochs: int = 3, batch_size: int = 4, learning_rate: float = 5e-5,
                  gradient_accumulation_steps: int = 4, max_grad_norm: float = 1.0):
        """Fine-tune the generative model with memory optimization"""
        logger.info(f"Fine-tuning generative model on {len(contexts)} examples")
        
        # Check if model is available
        if self.model is None or self.tokenizer is None:
            logger.warning("Generative model not available, skipping fine-tuning")
            self.is_trained = False
            return
        
        # Prepare training data
        train_data = []
        for context, question, answer in zip(contexts, questions, answers):
            input_text = self._prepare_input(context, question)
            train_data.append({
                'input': input_text,
                'target': answer
            })
        
        # Set up training with memory optimization
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Enable mixed precision for memory efficiency
        scaler = torch.cuda.amp.GradScaler() if self.device == "cuda" else None
        
        # Training loop with memory optimization
        for epoch in range(epochs):
            total_loss = 0
            num_batches = len(train_data) // batch_size
            optimizer.zero_grad()
            
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                
                # Prepare batch with shorter sequences for memory efficiency
                inputs = self.tokenizer(
                    [item['input'] for item in batch],
                    max_length=min(self.max_length, 1024),  # Limit sequence length
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                targets = self.tokenizer(
                    [item['target'] for item in batch],
                    max_length=min(self.max_length, 512),  # Limit target length
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Forward pass with mixed precision
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        if "t5" in self.model_name.lower():
                            outputs = self.model(
                                input_ids=inputs.input_ids,
                                attention_mask=inputs.attention_mask,
                                labels=targets.input_ids
                            )
                            loss = outputs.loss
                        else:
                            # For GPT-2, use next token prediction
                            outputs = self.model(
                                input_ids=inputs.input_ids,
                                attention_mask=inputs.attention_mask,
                                labels=inputs.input_ids
                            )
                            loss = outputs.loss
                    
                    # Scale loss for mixed precision
                    loss = loss / gradient_accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    # CPU training
                    if "t5" in self.model_name.lower():
                        outputs = self.model(
                            input_ids=inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            labels=targets.input_ids
                        )
                        loss = outputs.loss
                    else:
                        outputs = self.model(
                            input_ids=inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            labels=inputs.input_ids
                        )
                        loss = outputs.loss
                    
                    loss = loss / gradient_accumulation_steps
                    loss.backward()
                
                total_loss += loss.item() * gradient_accumulation_steps
                
                # Gradient accumulation
                if (i // batch_size + 1) % gradient_accumulation_steps == 0:
                    if scaler is not None:
                        # Check for inf/nan before unscaling
                        if scaler._per_optimizer_states[optimizer]['stage'] == 'unscaled':
                            # Skip unscaling if already unscaled
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            # Safe unscaling with error handling
                            try:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                                scaler.step(optimizer)
                                scaler.update()
                            except ValueError as e:
                                if "FP16 gradients" in str(e):
                                    # Skip gradient clipping if FP16 issue
                                    scaler.step(optimizer)
                                    scaler.update()
                                else:
                                    raise e
                    else:
                        # Gradient clipping for CPU
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    
                    # Clear GPU cache periodically
                    if self.device == "cuda" and (i // batch_size + 1) % 100 == 0:
                        torch.cuda.empty_cache()
            
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
            # Clear GPU cache after each epoch
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        self.is_trained = True
        logger.info("Generative model fine-tuning completed")
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            logger.warning("Saving untrained model")
        
        # Handle Gemini model (no local model to save)
        if self.use_gemini:
            logger.info("Using Gemini model - no local model to save")
            metadata = {
                'model_name': self.model_name,
                'max_length': self.max_length,
                'is_trained': self.is_trained,
                'use_gemini': True,
                'gemini_model_name': self.model_name
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(metadata, f)
            return
        
        # Save local model and tokenizer (if available)
        if self.model is not None and self.tokenizer is not None:
            model_dir = Path(filepath).parent / "generative_model"
            model_dir.mkdir(exist_ok=True)
            
            self.model.save_pretrained(model_dir)
            self.tokenizer.save_pretrained(model_dir)
        else:
            logger.warning("No local model to save - model or tokenizer is None")
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'is_trained': self.is_trained,
            'use_gemini': False
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Generative model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        # Load metadata
        with open(filepath, 'rb') as f:
            metadata = pickle.load(f)
        
        self.model_name = metadata['model_name']
        self.max_length = metadata['max_length']
        self.is_trained = metadata['is_trained']
        self.use_gemini = metadata.get('use_gemini', False)
        
        # Handle Gemini model
        if self.use_gemini:
            gemini_model_name = metadata.get('gemini_model_name', 'gemini-1.5-flash')
            logger.info(f"Loaded Gemini model configuration: {gemini_model_name}")
            return
        
        # Load local model and tokenizer
        model_dir = Path(filepath).parent / "generative_model"
        
        if model_dir.exists():
            if "t5" in self.model_name.lower():
                self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
                self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
                self.model = AutoModelForCausalLM.from_pretrained(model_dir)
            
            self.model.to(self.device)
            logger.info(f"Generative model loaded from {filepath}")
        else:
            logger.warning(f"Model directory not found: {model_dir}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = {
            'is_trained': self.is_trained,
            'model_name': self.model_name,
            'max_length': self.max_length,
            'device': self.device,
            'vocab_size': self.tokenizer.vocab_size if self.tokenizer else 0,
            'use_gemini': self.use_gemini
        }
        
        if self.use_gemini and self.gemini_model:
            info.update(self.gemini_model.get_model_info())
        
        return info
    
    def generate_with_retrieval(self, question: str, retrieved_contexts: List[str],
                               max_new_tokens: int = 100) -> Dict[str, Any]:
        """Generate answer using retrieved contexts"""
        # Use Gemini if available
        if self.use_gemini and self.gemini_model:
            return self.gemini_model.generate_with_retrieval(question, retrieved_contexts, max_new_tokens)
        
        # Check if model is available
        if self.model is None or self.tokenizer is None:
            logger.warning("Generative model not available, using fallback")
            # Fallback: simple answer from retrieved contexts
            combined_context = " ".join(retrieved_contexts[:3])  # Use top 3 contexts
            return {
                'answer': combined_context[:200] + "..." if len(combined_context) > 200 else combined_context,
                'confidence': 0.3,
                'used_contexts': retrieved_contexts,
                'num_contexts': len(retrieved_contexts),
                'fallback': True
            }
        
        # Combine contexts
        combined_context = " ".join(retrieved_contexts)
        
        # Generate answer
        result = self.generate_answer(
            combined_context, 
            question, 
            max_new_tokens=max_new_tokens
        )
        
        # Add context information
        result['used_contexts'] = retrieved_contexts
        result['num_contexts'] = len(retrieved_contexts)
        
        return result