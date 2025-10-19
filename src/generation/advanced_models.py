"""
Advanced Model Integration for A-Qlegal 2.0
Integrates Legal-BERT, Flan-T5, IndicBERT, and other SOTA models
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    T5Tokenizer,
    T5ForConditionalGeneration,
    pipeline
)
from typing import List, Dict, Any, Optional, Union
from loguru import logger
from pathlib import Path


class MultiModelLegalSystem:
    """
    Unified system for multiple legal models:
    - Legal-BERT (nlpaueb/legal-bert-base-uncased)
    - Flan-T5 (google/flan-t5-base, flan-t5-large)
    - IndicBERT (ai4bharat/indic-bert)
    - MuRIL (google/muril-base-cased)
    """
    
    def __init__(
        self,
        use_legal_bert: bool = True,
        use_flan_t5: bool = True,
        use_indic_bert: bool = True,
        device: str = "auto",
        model_cache_dir: str = "models/pretrained"
    ):
        self.device = self._setup_device(device)
        self.cache_dir = Path(model_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model storage
        self.models = {}
        self.tokenizers = {}
        
        # Load models based on configuration
        if use_legal_bert:
            self._load_legal_bert()
        
        if use_flan_t5:
            self._load_flan_t5()
        
        if use_indic_bert:
            self._load_indic_bert()
        
        logger.info(f"✅ Multi-Model System initialized with {len(self.models)} models")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                logger.info("💻 Using CPU")
        
        return torch.device(device)
    
    def _load_legal_bert(self):
        """Load Legal-BERT models"""
        logger.info("🔄 Loading Legal-BERT...")
        
        try:
            model_name = "nlpaueb/legal-bert-base-uncased"
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir)
            )
            
            # Load model
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir)
            ).to(self.device)
            
            model.eval()
            
            self.models['legal_bert'] = model
            self.tokenizers['legal_bert'] = tokenizer
            
            logger.info("✅ Legal-BERT loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading Legal-BERT: {e}")
    
    def _load_flan_t5(self, size: str = "base"):
        """
        Load Flan-T5 model
        
        Args:
            size: Model size - "small", "base", "large", "xl", "xxl"
        """
        logger.info(f"🔄 Loading Flan-T5 ({size})...")
        
        try:
            model_name = f"google/flan-t5-{size}"
            
            # Load tokenizer
            tokenizer = T5Tokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir)
            )
            
            # Load model
            model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir),
                device_map="auto" if self.device.type == "cuda" else None,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            )
            
            if self.device.type == "cpu":
                model = model.to(self.device)
            
            model.eval()
            
            self.models['flan_t5'] = model
            self.tokenizers['flan_t5'] = tokenizer
            
            logger.info(f"✅ Flan-T5 ({size}) loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading Flan-T5: {e}")
    
    def _load_indic_bert(self):
        """Load IndicBERT for multilingual support"""
        logger.info("🔄 Loading IndicBERT...")
        
        try:
            model_name = "ai4bharat/indic-bert"
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir)
            )
            
            # Load model
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir)
            ).to(self.device)
            
            model.eval()
            
            self.models['indic_bert'] = model
            self.tokenizers['indic_bert'] = tokenizer
            
            logger.info("✅ IndicBERT loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading IndicBERT: {e}")
    
    def encode_legal_text(
        self,
        texts: Union[str, List[str]],
        model_name: str = "legal_bert",
        max_length: int = 512,
        return_tensors: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Encode legal text using specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        # Tokenize
        inputs = tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Encode
        with torch.no_grad():
            outputs = model(**inputs)
            # Use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        if not return_tensors:
            embeddings = embeddings.cpu().numpy()
        
        if single_text:
            return embeddings[0]
        
        return embeddings
    
    def generate_with_flan_t5(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> Union[str, List[str]]:
        """Generate text using Flan-T5"""
        if 'flan_t5' not in self.models:
            raise ValueError("Flan-T5 model not loaded")
        
        model = self.models['flan_t5']
        tokenizer = self.tokenizers['flan_t5']
        
        # Tokenize input
        inputs = tokenizer(
            prompt,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True if temperature > 0 else False
            )
        
        # Decode
        generated_texts = [
            tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        if num_return_sequences == 1:
            return generated_texts[0]
        
        return generated_texts
    
    def explain_law_in_layman_terms(
        self,
        legal_text: str,
        context: str = "",
        language: str = "en"
    ) -> str:
        """
        Explain legal text in simple layman terms using Flan-T5
        
        Args:
            legal_text: Legal text to explain
            context: Additional context
            language: Target language (en, hi, ta, etc.)
        """
        if 'flan_t5' not in self.models:
            logger.warning("Flan-T5 not available, using fallback")
            return legal_text
        
        # Create prompt for Flan-T5
        prompt = f"""Explain the following legal text in simple, easy-to-understand language that a non-lawyer can understand:

Legal Text: {legal_text}

{f'Context: {context}' if context else ''}

Simplified Explanation:"""
        
        # Generate explanation
        explanation = self.generate_with_flan_t5(
            prompt,
            max_length=300,
            temperature=0.5
        )
        
        return explanation
    
    def categorize_legal_text(
        self,
        text: str,
        categories: List[str] = None
    ) -> Dict[str, float]:
        """Categorize legal text using Legal-BERT"""
        if categories is None:
            categories = ["Criminal", "Civil", "Constitutional", "Corporate", "Family"]
        
        if 'legal_bert' not in self.models:
            logger.warning("Legal-BERT not available")
            return {cat: 1.0/len(categories) for cat in categories}
        
        # Encode text
        text_embedding = self.encode_legal_text(text, model_name='legal_bert', return_tensors=True)
        
        # For now, use zero-shot classification approach
        # In production, this should be a fine-tuned classification head
        
        category_scores = {}
        for category in categories:
            category_text = f"This is a {category} law case."
            category_embedding = self.encode_legal_text(category_text, model_name='legal_bert', return_tensors=True)
            
            # Cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                text_embedding.unsqueeze(0),
                category_embedding.unsqueeze(0)
            ).item()
            
            category_scores[category] = similarity
        
        # Normalize scores
        total = sum(category_scores.values())
        if total > 0:
            category_scores = {k: v/total for k, v in category_scores.items()}
        
        return category_scores
    
    def answer_legal_question(
        self,
        question: str,
        context: str,
        max_answer_length: int = 200
    ) -> str:
        """Answer legal question using Flan-T5 with context"""
        if 'flan_t5' not in self.models:
            return "Model not available"
        
        # Create prompt
        prompt = f"""Answer the following legal question based on the provided context.

Context: {context}

Question: {question}

Answer:"""
        
        # Generate answer
        answer = self.generate_with_flan_t5(
            prompt,
            max_length=max_answer_length,
            temperature=0.3  # Lower temperature for more focused answers
        )
        
        return answer
    
    def translate_legal_text(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "hi"
    ) -> str:
        """
        Translate legal text to another language
        Uses IndicBERT or Flan-T5 for translation
        """
        if 'flan_t5' not in self.models:
            return text
        
        lang_names = {
            'en': 'English',
            'hi': 'Hindi',
            'ta': 'Tamil',
            'te': 'Telugu',
            'bn': 'Bengali',
            'mr': 'Marathi'
        }
        
        source_name = lang_names.get(source_lang, source_lang)
        target_name = lang_names.get(target_lang, target_lang)
        
        prompt = f"""Translate the following {source_name} legal text to {target_name}:

{source_name} text: {text}

{target_name} translation:"""
        
        translation = self.generate_with_flan_t5(
            prompt,
            max_length=len(text) * 2,
            temperature=0.3
        )
        
        return translation
    
    def summarize_judgment(
        self,
        judgment_text: str,
        max_summary_length: int = 200
    ) -> str:
        """Summarize a legal judgment"""
        if 'flan_t5' not in self.models:
            return judgment_text[:max_summary_length]
        
        prompt = f"""Summarize the following legal judgment in a concise manner:

Judgment: {judgment_text}

Summary:"""
        
        summary = self.generate_with_flan_t5(
            prompt,
            max_length=max_summary_length,
            temperature=0.4
        )
        
        return summary
    
    def extract_key_points(
        self,
        legal_text: str,
        num_points: int = 5
    ) -> List[str]:
        """Extract key points from legal text"""
        if 'flan_t5' not in self.models:
            return [legal_text[:100]]
        
        prompt = f"""Extract the {num_points} most important points from the following legal text:

Text: {legal_text}

Key Points (numbered list):"""
        
        response = self.generate_with_flan_t5(
            prompt,
            max_length=300,
            temperature=0.4
        )
        
        # Parse numbered list
        points = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Remove numbering/bullets
                point = re.sub(r'^[\d\.\-\•\*]\s*', '', line)
                points.append(point)
        
        return points[:num_points]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            'device': str(self.device),
            'loaded_models': list(self.models.keys()),
            'cache_dir': str(self.cache_dir)
        }
        
        for model_name, model in self.models.items():
            try:
                # Get model size
                param_count = sum(p.numel() for p in model.parameters())
                info[f'{model_name}_parameters'] = f"{param_count / 1e6:.1f}M"
                
                # Get model dtype
                info[f'{model_name}_dtype'] = str(next(model.parameters()).dtype)
            except:
                pass
        
        return info


# Import regex for extract_key_points
import re


def main():
    """Test advanced models"""
    # Initialize system
    system = MultiModelLegalSystem(
        use_legal_bert=True,
        use_flan_t5=True,
        use_indic_bert=False,  # Skip for quick test
        device="auto"
    )
    
    # Test legal text
    legal_text = """Section 302 IPC: Whoever commits murder shall be punished with death 
    or imprisonment for life, and shall also be liable to fine."""
    
    # Test 1: Encode text
    logger.info("\n📝 Test 1: Encoding legal text")
    embedding = system.encode_legal_text(legal_text, model_name='legal_bert')
    logger.info(f"Embedding shape: {embedding.shape}")
    
    # Test 2: Explain in layman terms
    logger.info("\n💬 Test 2: Explaining in layman terms")
    explanation = system.explain_law_in_layman_terms(legal_text)
    logger.info(f"Explanation: {explanation}")
    
    # Test 3: Answer question
    logger.info("\n❓ Test 3: Answering question")
    question = "What is the punishment for murder?"
    answer = system.answer_legal_question(question, legal_text)
    logger.info(f"Answer: {answer}")
    
    # Test 4: Categorize
    logger.info("\n🏷️  Test 4: Categorizing text")
    categories = system.categorize_legal_text(legal_text)
    for cat, score in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"{cat}: {score:.3f}")
    
    # Test 5: Extract key points
    logger.info("\n🔑 Test 5: Extracting key points")
    points = system.extract_key_points(legal_text)
    for i, point in enumerate(points, 1):
        logger.info(f"{i}. {point}")
    
    # Model info
    logger.info("\n📊 Model Information:")
    info = system.get_model_info()
    for key, value in info.items():
        logger.info(f"{key}: {value}")


if __name__ == "__main__":
    main()

