"""
LoRA/PEFT Training Pipeline for A-Qlegal 2.0
Efficient fine-tuning of large legal models using Parameter-Efficient Fine-Tuning
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset
from typing import List, Dict, Any, Optional
from loguru import logger
from pathlib import Path
import json


class LoRALegalTrainer:
    """
    Trainer for legal models using LoRA (Low-Rank Adaptation)
    
    Features:
    - Memory-efficient training with LoRA
    - 4-bit/8-bit quantization support
    - Legal-to-layman text simplification
    - Multi-task training
    """
    
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        output_dir: str = "models/lora_legal_model",
        use_4bit: bool = False,
        use_8bit: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        device: str = "auto"
    ):
        """
        Initialize LoRA trainer
        
        Args:
            model_name: Base model to fine-tune
            output_dir: Directory to save model
            use_4bit: Use 4-bit quantization (requires bitsandbytes)
            use_8bit: Use 8-bit quantization
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: Dropout for LoRA layers
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Setup device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"🔧 Initializing LoRA trainer for {model_name}")
        logger.info(f"💾 Device: {self.device}")
        logger.info(f"⚙️  Quantization: {'4-bit' if use_4bit else '8-bit' if use_8bit else 'None'}")
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
        self._load_model()
    
    def _load_model(self):
        """Load base model with optional quantization"""
        logger.info(f"🔄 Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Setup quantization config
        quantization_config = None
        if self.use_4bit or self.use_8bit:
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.use_4bit,
                load_in_8bit=self.use_8bit,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4" if self.use_4bit else None,
                bnb_4bit_use_double_quant=True if self.use_4bit else False
            )
        
        # Load model
        try:
            # Try as Seq2Seq model first (T5, BART, etc.)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.task_type = TaskType.SEQ_2_SEQ_LM
            logger.info("✅ Loaded as Seq2Seq model")
            
        except:
            # Try as Causal LM (GPT, LLaMA, etc.)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.task_type = TaskType.CAUSAL_LM
            logger.info("✅ Loaded as Causal LM")
        
        # Prepare model for training if using quantization
        if quantization_config is not None:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Setup LoRA
        self._setup_lora()
    
    def _setup_lora(self):
        """Setup LoRA configuration"""
        logger.info("🔧 Setting up LoRA...")
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self._get_target_modules(),
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=self.task_type
        )
        
        # Apply LoRA
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.peft_model.print_trainable_parameters()
        
        logger.info("✅ LoRA setup complete")
    
    def _get_target_modules(self) -> List[str]:
        """Get target modules for LoRA based on model type"""
        model_type = self.model.config.model_type.lower()
        
        # T5/Flan-T5
        if "t5" in model_type:
            return ["q", "v"]
        
        # BERT-based models
        elif "bert" in model_type:
            return ["query", "value"]
        
        # LLaMA/Mistral
        elif "llama" in model_type or "mistral" in model_type:
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        # Default
        else:
            return ["q", "v"]
    
    def prepare_dataset(
        self,
        data: List[Dict[str, str]],
        input_key: str = "legal_text",
        output_key: str = "simplified_summary",
        max_input_length: int = 512,
        max_output_length: int = 256
    ) -> Dataset:
        """
        Prepare dataset for training
        
        Args:
            data: List of dictionaries with input-output pairs
            input_key: Key for input text
            output_key: Key for output text
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
        """
        logger.info(f"📊 Preparing dataset with {len(data)} examples")
        
        def preprocess_function(examples):
            # Get inputs and outputs
            inputs = examples[input_key]
            targets = examples[output_key]
            
            # Tokenize inputs
            model_inputs = self.tokenizer(
                inputs,
                max_length=max_input_length,
                truncation=True,
                padding="max_length"
            )
            
            # Tokenize outputs
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    targets,
                    max_length=max_output_length,
                    truncation=True,
                    padding="max_length"
                )
            
            model_inputs["labels"] = labels["input_ids"]
            
            return model_inputs
        
        # Create HuggingFace dataset
        dataset = Dataset.from_list(data)
        
        # Preprocess
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        logger.info(f"✅ Dataset prepared: {len(processed_dataset)} examples")
        
        return processed_dataset
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        gradient_accumulation_steps: int = 4,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 50
    ):
        """Train the model with LoRA"""
        logger.info("🚀 Starting training...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            fp16=self.device == "cuda",
            optim="adamw_torch",
            report_to="none",  # Disable wandb
            remove_unused_columns=False
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.peft_model,
            padding=True
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )
        
        # Train
        logger.info("🔥 Training started...")
        trainer.train()
        
        # Save model
        logger.info("💾 Saving model...")
        self.save_model()
        
        logger.info("✅ Training complete!")
    
    def save_model(self, path: str = None):
        """Save the LoRA model"""
        if path is None:
            path = self.output_dir
        else:
            path = Path(path)
        
        path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA weights
        self.peft_model.save_pretrained(str(path))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(str(path))
        
        # Save config
        config = {
            'base_model': self.model_name,
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'task_type': str(self.task_type)
        }
        
        with open(path / "lora_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a saved LoRA model"""
        from peft import PeftModel
        
        path = Path(path)
        
        logger.info(f"📂 Loading LoRA model from {path}")
        
        # Load config
        with open(path / "lora_config.json", 'r') as f:
            config = json.load(f)
        
        # Load base model
        self.model_name = config['base_model']
        self._load_model()
        
        # Load LoRA weights
        self.peft_model = PeftModel.from_pretrained(self.model, str(path))
        
        logger.info("✅ Model loaded successfully")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_beams: int = 4
    ) -> str:
        """Generate text using the fine-tuned model"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                early_stopping=True
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text


def create_legal_to_layman_dataset(
    legal_texts: List[str],
    simplified_texts: List[str]
) -> List[Dict[str, str]]:
    """Create training dataset for legal-to-layman simplification"""
    dataset = []
    
    for legal, simple in zip(legal_texts, simplified_texts):
        dataset.append({
            'legal_text': legal,
            'simplified_summary': simple,
            'input': f"Simplify this legal text: {legal}",
            'output': simple
        })
    
    return dataset


def main():
    """Main training function"""
    # Initialize trainer
    trainer = LoRALegalTrainer(
        model_name="google/flan-t5-base",
        output_dir="models/lora_legal_simplifier",
        use_8bit=True,
        lora_r=16,
        lora_alpha=32
    )
    
    # Load or create training data
    # Example: Load from enhanced dataset
    dataset_file = Path("data/legal_to_layman_pairs.json")
    
    if dataset_file.exists():
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"📚 Loaded {len(data)} training pairs")
        
        # Prepare dataset
        train_dataset = trainer.prepare_dataset(
            data,
            input_key='legal',
            output_key='layman'
        )
        
        # Split into train/eval
        split_idx = int(len(train_dataset) * 0.9)
        eval_dataset = train_dataset.select(range(split_idx, len(train_dataset)))
        train_dataset = train_dataset.select(range(split_idx))
        
        # Train
        trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            num_epochs=3,
            batch_size=4,
            learning_rate=2e-4
        )
        
        # Test generation
        test_input = "Section 302 IPC: Whoever commits murder shall be punished with death or imprisonment for life."
        logger.info(f"\n🧪 Test Input: {test_input}")
        
        output = trainer.generate(f"Simplify this legal text: {test_input}")
        logger.info(f"📝 Generated Output: {output}")
    
    else:
        logger.error(f"❌ Training data not found: {dataset_file}")
        logger.info("💡 Please run enhanced_preprocessor.py first to create training pairs")


if __name__ == "__main__":
    main()

