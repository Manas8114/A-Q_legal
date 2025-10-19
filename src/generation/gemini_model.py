"""
Gemini API-based generative answer model for legal QA
"""
import os
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from loguru import logger
import time
import requests
import socket


class GeminiGenerativeModel:
    """Gemini API-based generative model for legal QA"""
    
    def __init__(self, api_key: str = None, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model_name = model_name
        self.model = None
        self.is_trained = True  # Gemini models are pre-trained
        self.is_available = False
        
        if not self.api_key:
            logger.warning("No Gemini API key provided. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
            return
        
        # Check internet connection first
        if not self._check_internet_connection():
            raise ConnectionError("No internet connection available. Gemini API requires internet access.")
        
        self._initialize_model()
    
    def _check_internet_connection(self) -> bool:
        """Check if internet connection is available"""
        try:
            # Try to connect to Google's DNS
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            try:
                # Try to connect to Google's API endpoint
                response = requests.get("https://generativelanguage.googleapis.com", timeout=5)
                return response.status_code in [200, 404]  # 404 is fine, means endpoint exists
            except:
                return False
    
    def _initialize_model(self):
        """Initialize the Gemini model"""
        try:
            # Configure the API key
            genai.configure(api_key=self.api_key)
            
            # Initialize the model
            self.model = genai.GenerativeModel(self.model_name)
            self.is_available = True
            
            logger.info(f"Initialized Gemini model: {self.model_name}")
            
            # Test the connection with a simple prompt
            try:
                test_response = self.model.generate_content("Hello")
                logger.info("Gemini API connection test successful")
            except Exception as e:
                logger.warning(f"Gemini API connection test failed: {e}")
                self.is_available = False
                
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            self.is_available = False
    
    def _prepare_legal_prompt(self, context: str, question: str) -> str:
        """Prepare a legal-specific prompt for Gemini"""
        prompt = f"""You are a legal expert assistant specializing in Indian law. Based on the provided legal context, please answer the following question accurately and comprehensively.

Legal Context:
{context}

Question: {question}

Please provide a detailed and accurate answer based on the legal context provided. If the context doesn't contain sufficient information to answer the question completely, please indicate this in your response.

Answer:"""
        return prompt
    
    def generate_answer(self, context: str, question: str, 
                       max_tokens: int = 500, temperature: float = 0.3) -> Dict[str, Any]:
        """Generate an answer using Gemini API"""
        if not self.is_available:
            return {
                'answer': 'Gemini API is not available. Please check your API key and connection.',
                'confidence': 0.0,
                'error': 'API not available'
            }
        
        try:
            # Prepare the prompt
            prompt = self._prepare_legal_prompt(context, question)
            
            # Generate content
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=0.8,
                top_k=40
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if response.text:
                # Calculate confidence based on response length and completeness
                confidence = min(0.9, 0.5 + (len(response.text) / 1000) * 0.4)
                
                return {
                    'answer': response.text.strip(),
                    'confidence': confidence,
                    'model': self.model_name,
                    'tokens_used': len(response.text.split()),
                    'prompt_tokens': len(prompt.split())
                }
            else:
                logger.warning("Empty response from Gemini API")
                return {
                    'answer': 'Unable to generate a response. Please try again.',
                    'confidence': 0.0,
                    'error': 'Empty response'
                }
                
        except Exception as e:
            logger.error(f"Error generating answer with Gemini: {e}")
            return {
                'answer': f'Error generating answer: {str(e)}',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def generate_with_retrieval(self, question: str, retrieved_contexts: List[str],
                               max_tokens: int = 500) -> Dict[str, Any]:
        """Generate answer using retrieved contexts"""
        if not self.is_available:
            logger.warning("Gemini API not available, using fallback")
            # Fallback: simple answer from retrieved contexts
            combined_context = " ".join(retrieved_contexts[:3])  # Use top 3 contexts
            return {
                'answer': combined_context[:200] + "..." if len(combined_context) > 200 else combined_context,
                'confidence': 0.3,
                'used_contexts': retrieved_contexts,
                'num_contexts': len(retrieved_contexts),
                'fallback': True
            }
        
        # Combine contexts intelligently
        combined_context = self._combine_contexts(retrieved_contexts)
        
        # Generate answer
        result = self.generate_answer(combined_context, question, max_tokens)
        
        # Add context information
        result['used_contexts'] = retrieved_contexts
        result['num_contexts'] = len(retrieved_contexts)
        result['combined_context_length'] = len(combined_context)
        
        return result
    
    def _combine_contexts(self, contexts: List[str], max_length: int = 3000) -> str:
        """Intelligently combine contexts while staying within length limits"""
        if not contexts:
            return ""
        
        # Start with the first context
        combined = contexts[0]
        
        # Add additional contexts if there's room
        for context in contexts[1:]:
            if len(combined) + len(context) + 1 <= max_length:
                combined += "\n\n" + context
            else:
                # Add partial context if there's still room
                remaining_length = max_length - len(combined) - 2
                if remaining_length > 100:  # Only add if meaningful length remains
                    combined += "\n\n" + context[:remaining_length] + "..."
                break
        
        return combined
    
    def generate_answers_batch(self, contexts: List[str], questions: List[str],
                              max_tokens: int = 500, temperature: float = 0.3) -> List[Dict[str, Any]]:
        """Generate answers for a batch of context-question pairs"""
        results = []
        
        for context, question in zip(contexts, questions):
            result = self.generate_answer(context, question, max_tokens, temperature)
            results.append(result)
            
            # Add small delay to avoid rate limiting
            time.sleep(0.1)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'is_available': self.is_available,
            'is_trained': self.is_trained,
            'api_key_set': bool(self.api_key),
            'model_type': 'gemini_api'
        }
    
    def test_connection(self) -> bool:
        """Test the connection to Gemini API"""
        if not self.is_available:
            return False
        
        try:
            test_response = self.model.generate_content("Test connection")
            return bool(test_response.text)
        except Exception as e:
            logger.error(f"Gemini API connection test failed: {e}")
            return False
