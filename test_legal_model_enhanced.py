#!/usr/bin/env python3
"""
🧪 Enhanced Legal Model Testing Script with Beautiful Output
Tests trained models with improved visualization and reporting.
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class Colors:
    """ANSI color codes for beautiful console output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_box(text: str, color=Colors.BLUE):
    """Print text in a nice box"""
    width = len(text) + 4
    print(f"\n{color}{'=' * width}")
    print(f"  {text}  ")
    print(f"{'=' * width}{Colors.END}\n")


def print_result(label: str, value: str, success: bool = None):
    """Print a result line with color coding"""
    if success is None:
        color = Colors.CYAN
        icon = "ℹ️"
    elif success:
        color = Colors.GREEN
        icon = "✅"
    else:
        color = Colors.RED
        icon = "❌"
    
    print(f"{icon} {Colors.BOLD}{label}:{Colors.END} {color}{value}{Colors.END}")


def print_confidence_bar(confidence: float):
    """Print a visual confidence bar"""
    bar_length = 30
    filled = int(confidence * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    # Color based on confidence
    if confidence >= 0.7:
        color = Colors.GREEN
    elif confidence >= 0.4:
        color = Colors.YELLOW
    else:
        color = Colors.RED
    
    print(f"   {color}{bar}{Colors.END} {confidence*100:.1f}%")


class EnhancedLegalModelTester:
    """Enhanced tester with beautiful output"""
    
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.classification_model = None
        self.classification_tokenizer = None
        self.qa_model = None
        self.qa_tokenizer = None
        self.category_mapping = None
        
        self.test_results = {
            'classification': {'correct': 0, 'total': 0, 'details': []},
            'qa': {'answered': 0, 'total': 0, 'details': []}
        }
        
        self.load_models()
    
    def load_models(self):
        """Load trained models"""
        print_box("📥 LOADING TRAINED MODELS", Colors.CYAN)
        
        classification_path = self.model_dir / "legal_classification_model"
        if classification_path.exists():
            try:
                self.classification_tokenizer = AutoTokenizer.from_pretrained(str(classification_path))
                self.classification_model = AutoModelForSequenceClassification.from_pretrained(str(classification_path))
                self.classification_model.to(self.device)
                self.classification_model.eval()
                print_result("Classification Model", "Loaded successfully ✓", True)
                
                mapping_path = self.model_dir / "category_mapping.json"
                if mapping_path.exists():
                    with open(mapping_path, 'r', encoding='utf-8') as f:
                        self.category_mapping = json.load(f)
                    print_result("Categories", f"{len(self.category_mapping['categories'])} loaded", True)
            except Exception as e:
                print_result("Classification Model", f"Error: {e}", False)
        
        qa_path = self.model_dir / "legal_qa_model"
        if qa_path.exists():
            try:
                self.qa_tokenizer = AutoTokenizer.from_pretrained(str(qa_path))
                self.qa_model = AutoModelForQuestionAnswering.from_pretrained(str(qa_path))
                self.qa_model.to(self.device)
                self.qa_model.eval()
                print_result("QA Model", "Loaded successfully ✓", True)
            except Exception as e:
                print_result("QA Model", f"Error: {e}", False)
    
    def classify_text(self, text: str) -> Dict[str, Any]:
        """Classify legal text"""
        if self.classification_model is None:
            return {'error': 'Model not loaded'}
        
        try:
            inputs = self.classification_tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.classification_model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()
            
            predicted_category = self.category_mapping['id_to_category'].get(str(predicted_class), 'unknown')
            
            top_k = min(5, len(probabilities[0]))
            top_probs, top_indices = torch.topk(probabilities[0], top_k)
            
            top_predictions = []
            for prob, idx in zip(top_probs, top_indices):
                category = self.category_mapping['id_to_category'].get(str(idx.item()), 'unknown')
                top_predictions.append({'category': category, 'confidence': prob.item()})
            
            return {
                'predicted_category': predicted_category,
                'confidence': confidence,
                'top_predictions': top_predictions
            }
        except Exception as e:
            return {'error': str(e)}
    
    def answer_question(self, question: str, context: str) -> Dict[str, Any]:
        """Answer a question"""
        if self.qa_model is None:
            return {'error': 'Model not loaded'}
        
        try:
            inputs = self.qa_tokenizer(question, context, return_tensors='pt', truncation='only_second', max_length=512, padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.qa_model(**inputs)
                start_idx = torch.argmax(outputs.start_logits)
                end_idx = torch.argmax(outputs.end_logits)
                
                start_prob = torch.softmax(outputs.start_logits, dim=-1)[0][start_idx].item()
                end_prob = torch.softmax(outputs.end_logits, dim=-1)[0][end_idx].item()
                confidence = (start_prob + end_prob) / 2
                
                if end_idx >= start_idx and confidence > 0.1:
                    answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
                    answer = self.qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)
                else:
                    answer = "Unable to find answer in the context."
                    confidence = 0.0
            
            return {'answer': answer, 'confidence': confidence}
        except Exception as e:
            return {'error': str(e)}
    
    def test_classification(self):
        """Test classification with enhanced output"""
        print_box("🧪 CLASSIFICATION MODEL TESTING", Colors.BLUE)
        
        samples = [
            {'text': "Article 14 of the Indian Constitution states equality before law", 'expected': 'fundamental_rights'},
            {'text': "Section 302 IPC deals with murder and its punishment", 'expected': 'criminal_law'},
            {'text': "Code of Civil Procedure governs civil suits", 'expected': 'civil_procedure'},
            {'text': "Habeas Corpus is a writ for personal liberty", 'expected': 'legal_procedures'},
            {'text': "Supreme Court has judicial review powers", 'expected': 'constitutional_law'},
        ]
        
        for i, sample in enumerate(samples, 1):
            print(f"\n{Colors.BOLD}{'─' * 70}")
            print(f"Test Case #{i}{Colors.END}")
            print(f"{'─' * 70}")
            
            print(f"\n📄 {Colors.CYAN}Input Text:{Colors.END}")
            print(f"   {sample['text'][:100]}...")
            
            result = self.classify_text(sample['text'])
            
            print(f"\n🎯 {Colors.CYAN}Expected:{Colors.END} {Colors.BOLD}{sample['expected']}{Colors.END}")
            print(f"🔮 {Colors.CYAN}Predicted:{Colors.END} {Colors.BOLD}{result['predicted_category']}{Colors.END}")
            
            print(f"\n📊 {Colors.CYAN}Confidence:{Colors.END}")
            print_confidence_bar(result['confidence'])
            
            correct = result['predicted_category'] == sample['expected']
            self.test_results['classification']['total'] += 1
            if correct:
                self.test_results['classification']['correct'] += 1
            
            self.test_results['classification']['details'].append({
                'text': sample['text'][:50],
                'expected': sample['expected'],
                'predicted': result['predicted_category'],
                'confidence': result['confidence'],
                'correct': correct
            })
            
            if correct:
                print(f"\n{Colors.GREEN}✅ CORRECT!{Colors.END}")
            else:
                print(f"\n{Colors.RED}❌ INCORRECT{Colors.END}")
                print(f"\n💡 {Colors.YELLOW}Top 3 Predictions:{Colors.END}")
                for j, pred in enumerate(result['top_predictions'][:3], 1):
                    marker = "👉" if j == 1 else "  "
                    print(f"   {marker} {pred['category']}: {pred['confidence']*100:.1f}%")
    
    def test_qa(self):
        """Test QA with enhanced output"""
        print_box("🧪 QUESTION ANSWERING MODEL TESTING", Colors.BLUE)
        
        samples = [
            {
                'question': "What does Article 14 state?",
                'context': "Article 14 of the Indian Constitution states: The State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India.",
            },
            {
                'question': "What is the punishment for murder?",
                'context': "Section 302 of the Indian Penal Code deals with murder. Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine.",
            },
            {
                'question': "What is Article 21 about?",
                'context': "Article 21 states: No person shall be deprived of his life or personal liberty except according to procedure established by law.",
            },
        ]
        
        for i, sample in enumerate(samples, 1):
            print(f"\n{Colors.BOLD}{'─' * 70}")
            print(f"Test Case #{i}{Colors.END}")
            print(f"{'─' * 70}")
            
            print(f"\n❓ {Colors.CYAN}Question:{Colors.END}")
            print(f"   {sample['question']}")
            
            print(f"\n📖 {Colors.CYAN}Context:{Colors.END}")
            print(f"   {sample['context'][:100]}...")
            
            result = self.answer_question(sample['question'], sample['context'])
            
            print(f"\n💡 {Colors.CYAN}Answer:{Colors.END}")
            if result['confidence'] > 0:
                print(f"   {Colors.GREEN}{result['answer']}{Colors.END}")
            else:
                print(f"   {Colors.RED}{result['answer']}{Colors.END}")
            
            print(f"\n📊 {Colors.CYAN}Confidence:{Colors.END}")
            print_confidence_bar(result['confidence'])
            
            self.test_results['qa']['total'] += 1
            if result['confidence'] > 0.3:
                self.test_results['qa']['answered'] += 1
            
            self.test_results['qa']['details'].append({
                'question': sample['question'],
                'answer': result['answer'],
                'confidence': result['confidence']
            })
    
    def print_summary(self):
        """Print beautiful test summary"""
        print_box("📊 TEST RESULTS SUMMARY", Colors.CYAN)
        
        # Classification results
        cls_total = self.test_results['classification']['total']
        cls_correct = self.test_results['classification']['correct']
        cls_accuracy = (cls_correct / cls_total * 100) if cls_total > 0 else 0
        
        print(f"{Colors.BOLD}Classification Model:{Colors.END}")
        print(f"   Accuracy: {cls_correct}/{cls_total} tests")
        print(f"   Success Rate: ", end="")
        print_confidence_bar(cls_accuracy / 100)
        
        # QA results
        qa_total = self.test_results['qa']['total']
        qa_answered = self.test_results['qa']['answered']
        qa_rate = (qa_answered / qa_total * 100) if qa_total > 0 else 0
        
        print(f"\n{Colors.BOLD}QA Model:{Colors.END}")
        print(f"   Answered: {qa_answered}/{qa_total} questions")
        print(f"   Answer Rate: ", end="")
        print_confidence_bar(qa_rate / 100)
        
        # Overall assessment
        print(f"\n{Colors.BOLD}Overall Assessment:{Colors.END}")
        avg_performance = (cls_accuracy + qa_rate) / 2
        
        if avg_performance >= 70:
            status = f"{Colors.GREEN}Excellent ⭐⭐⭐{Colors.END}"
        elif avg_performance >= 50:
            status = f"{Colors.YELLOW}Good ⭐⭐{Colors.END}"
        elif avg_performance >= 30:
            status = f"{Colors.YELLOW}Fair ⭐{Colors.END}"
        else:
            status = f"{Colors.RED}Needs Improvement{Colors.END}"
        
        print(f"   Status: {status}")
        print(f"   Average: {avg_performance:.1f}%")
        
        # Save detailed report
        self.save_report()
    
    def save_report(self):
        """Save detailed test report"""
        report_path = self.model_dir.parent / "test_report.json"
        report = {
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'results': self.test_results,
            'summary': {
                'classification_accuracy': (self.test_results['classification']['correct'] / 
                                          self.test_results['classification']['total'] * 100) 
                                          if self.test_results['classification']['total'] > 0 else 0,
                'qa_answer_rate': (self.test_results['qa']['answered'] / 
                                  self.test_results['qa']['total'] * 100) 
                                  if self.test_results['qa']['total'] > 0 else 0
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved: {Colors.CYAN}{report_path}{Colors.END}")
    
    def run_all_tests(self):
        """Run all tests"""
        print_box("🚀 ENHANCED LEGAL MODEL TESTING", Colors.HEADER)
        
        if self.classification_model:
            self.test_classification()
        
        if self.qa_model:
            self.test_qa()
        
        self.print_summary()
        
        print_box("✅ TESTING COMPLETE!", Colors.GREEN)


def main():
    """Main execution"""
    BASE_DIR = Path(r"C:\Users\msgok\Desktop\A-Qlegal-main")
    MODEL_DIR = BASE_DIR / "models" / "legal_model"
    
    try:
        if not MODEL_DIR.exists():
            print_result("Error", f"Model directory not found: {MODEL_DIR}", False)
            return False
        
        tester = EnhancedLegalModelTester(str(MODEL_DIR))
        tester.run_all_tests()
        
        return True
    except Exception as e:
        print_result("Fatal Error", str(e), False)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

