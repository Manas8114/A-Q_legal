"""
Evaluation Metrics for A-Qlegal 2.0
Includes BLEU, ROUGE, BERTScore, and Flesch readability metrics
"""

from typing import List, Dict, Any, Union
from loguru import logger
import numpy as np


try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except:
    ROUGE_AVAILABLE = False
    logger.warning("rouge_score not available")

try:
    from sacrebleu import sentence_bleu, corpus_bleu, BLEU
    BLEU_AVAILABLE = True
except:
    BLEU_AVAILABLE = False
    logger.warning("sacrebleu not available")

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except:
    BERTSCORE_AVAILABLE = False
    logger.warning("bert_score not available")

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except:
    TEXTSTAT_AVAILABLE = False
    logger.warning("textstat not available")


class LegalEvaluationMetrics:
    """
    Comprehensive evaluation metrics for legal text generation
    
    Metrics:
    1. BLEU - Measures n-gram overlap
    2. ROUGE - Measures recall-oriented n-gram overlap
    3. BERTScore - Measures semantic similarity
    4. Flesch Reading Ease - Measures readability
    5. Legal Accuracy - Custom metric for legal correctness
    """
    
    def __init__(self):
        self.rouge_scorer = None
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
        
        logger.info("📊 Evaluation metrics initialized")
    
    def calculate_bleu(
        self,
        hypothesis: Union[str, List[str]],
        reference: Union[str, List[str]],
        use_corpus: bool = False
    ) -> Dict[str, float]:
        """
        Calculate BLEU score
        
        Args:
            hypothesis: Generated text(s)
            reference: Reference text(s)
            use_corpus: Whether to use corpus-level BLEU
        
        Returns:
            Dictionary with BLEU scores
        """
        if not BLEU_AVAILABLE:
            logger.warning("BLEU not available")
            return {'bleu': 0.0}
        
        try:
            if use_corpus:
                # Corpus-level BLEU
                if isinstance(hypothesis, str):
                    hypothesis = [hypothesis]
                if isinstance(reference, str):
                    reference = [reference]
                
                bleu = corpus_bleu(hypothesis, [[ref] for ref in reference])
                
                return {
                    'bleu': bleu.score / 100.0,
                    'bleu_1': bleu.precisions[0] / 100.0 if len(bleu.precisions) > 0 else 0.0,
                    'bleu_2': bleu.precisions[1] / 100.0 if len(bleu.precisions) > 1 else 0.0,
                    'bleu_3': bleu.precisions[2] / 100.0 if len(bleu.precisions) > 2 else 0.0,
                    'bleu_4': bleu.precisions[3] / 100.0 if len(bleu.precisions) > 3 else 0.0
                }
            else:
                # Sentence-level BLEU
                bleu = sentence_bleu(hypothesis, [reference])
                
                return {
                    'bleu': bleu.score / 100.0
                }
        
        except Exception as e:
            logger.error(f"BLEU calculation error: {e}")
            return {'bleu': 0.0}
    
    def calculate_rouge(
        self,
        hypothesis: str,
        reference: str
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores
        
        Args:
            hypothesis: Generated text
            reference: Reference text
        
        Returns:
            Dictionary with ROUGE scores
        """
        if not ROUGE_AVAILABLE or not self.rouge_scorer:
            logger.warning("ROUGE not available")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        try:
            scores = self.rouge_scorer.score(reference, hypothesis)
            
            return {
                'rouge1_precision': scores['rouge1'].precision,
                'rouge1_recall': scores['rouge1'].recall,
                'rouge1_f1': scores['rouge1'].fmeasure,
                'rouge2_precision': scores['rouge2'].precision,
                'rouge2_recall': scores['rouge2'].recall,
                'rouge2_f1': scores['rouge2'].fmeasure,
                'rougeL_precision': scores['rougeL'].precision,
                'rougeL_recall': scores['rougeL'].recall,
                'rougeL_f1': scores['rougeL'].fmeasure
            }
        
        except Exception as e:
            logger.error(f"ROUGE calculation error: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def calculate_bertscore(
        self,
        hypotheses: Union[str, List[str]],
        references: Union[str, List[str]],
        model_type: str = "bert-base-uncased",
        lang: str = "en"
    ) -> Dict[str, float]:
        """
        Calculate BERTScore (semantic similarity)
        
        Args:
            hypotheses: Generated text(s)
            references: Reference text(s)
            model_type: BERT model to use
            lang: Language code
        
        Returns:
            Dictionary with BERTScore metrics
        """
        if not BERTSCORE_AVAILABLE:
            logger.warning("BERTScore not available")
            return {'bert_precision': 0.0, 'bert_recall': 0.0, 'bert_f1': 0.0}
        
        try:
            # Ensure lists
            if isinstance(hypotheses, str):
                hypotheses = [hypotheses]
            if isinstance(references, str):
                references = [references]
            
            # Calculate BERTScore
            P, R, F1 = bert_score(
                hypotheses,
                references,
                model_type=model_type,
                lang=lang,
                verbose=False
            )
            
            return {
                'bert_precision': float(P.mean()),
                'bert_recall': float(R.mean()),
                'bert_f1': float(F1.mean())
            }
        
        except Exception as e:
            logger.error(f"BERTScore calculation error: {e}")
            return {'bert_precision': 0.0, 'bert_recall': 0.0, 'bert_f1': 0.0}
    
    def calculate_flesch_score(self, text: str) -> Dict[str, float]:
        """
        Calculate Flesch Reading Ease score
        
        Scale:
        - 90-100: Very easy (5th grade)
        - 80-90: Easy (6th grade)
        - 70-80: Fairly easy (7th grade)
        - 60-70: Standard (8th-9th grade)
        - 50-60: Fairly difficult (10th-12th grade)
        - 30-50: Difficult (College)
        - 0-30: Very difficult (College graduate)
        """
        if not TEXTSTAT_AVAILABLE:
            logger.warning("textstat not available")
            return {'flesch_reading_ease': 0.0, 'flesch_grade': 0.0}
        
        try:
            flesch_reading_ease = textstat.flesch_reading_ease(text)
            flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
            
            # Additional readability metrics
            gunning_fog = textstat.gunning_fog(text)
            automated_readability = textstat.automated_readability_index(text)
            
            return {
                'flesch_reading_ease': flesch_reading_ease,
                'flesch_kincaid_grade': flesch_kincaid_grade,
                'gunning_fog': gunning_fog,
                'automated_readability_index': automated_readability,
                'readability_level': self._interpret_flesch_score(flesch_reading_ease)
            }
        
        except Exception as e:
            logger.error(f"Flesch score calculation error: {e}")
            return {'flesch_reading_ease': 0.0, 'flesch_grade': 0.0}
    
    def _interpret_flesch_score(self, score: float) -> str:
        """Interpret Flesch Reading Ease score"""
        if score >= 90:
            return "Very Easy"
        elif score >= 80:
            return "Easy"
        elif score >= 70:
            return "Fairly Easy"
        elif score >= 60:
            return "Standard"
        elif score >= 50:
            return "Fairly Difficult"
        elif score >= 30:
            return "Difficult"
        else:
            return "Very Difficult"
    
    def calculate_legal_accuracy(
        self,
        hypothesis: str,
        reference: str,
        legal_terms: List[str] = None
    ) -> Dict[str, float]:
        """
        Custom metric for legal accuracy
        
        Checks:
        - Preservation of legal terms
        - Section/Article number accuracy
        - Punishment details accuracy
        """
        if legal_terms is None:
            legal_terms = [
                'Section', 'Article', 'IPC', 'CrPC', 'Constitution',
                'imprisonment', 'fine', 'punishment', 'death', 'life'
            ]
        
        # Convert to lowercase for comparison
        hyp_lower = hypothesis.lower()
        ref_lower = reference.lower()
        
        # Check legal term preservation
        legal_term_matches = 0
        legal_term_total = 0
        
        for term in legal_terms:
            term_lower = term.lower()
            if term_lower in ref_lower:
                legal_term_total += 1
                if term_lower in hyp_lower:
                    legal_term_matches += 1
        
        legal_term_accuracy = legal_term_matches / legal_term_total if legal_term_total > 0 else 1.0
        
        # Check section/article number accuracy
        import re
        ref_sections = set(re.findall(r'Section\s+(\d+[A-Z]?)|Article\s+(\d+[A-Z]?)', reference, re.IGNORECASE))
        hyp_sections = set(re.findall(r'Section\s+(\d+[A-Z]?)|Article\s+(\d+[A-Z]?)', hypothesis, re.IGNORECASE))
        
        section_accuracy = len(ref_sections & hyp_sections) / len(ref_sections) if ref_sections else 1.0
        
        # Overall accuracy
        overall_accuracy = (legal_term_accuracy + section_accuracy) / 2
        
        return {
            'legal_accuracy': overall_accuracy,
            'legal_term_accuracy': legal_term_accuracy,
            'section_accuracy': section_accuracy,
            'legal_terms_preserved': legal_term_matches,
            'legal_terms_total': legal_term_total
        }
    
    def evaluate_all(
        self,
        hypothesis: str,
        reference: str,
        include_bertscore: bool = False
    ) -> Dict[str, Any]:
        """
        Calculate all metrics at once
        
        Args:
            hypothesis: Generated text
            reference: Reference text
            include_bertscore: Whether to include BERTScore (slower)
        
        Returns:
            Dictionary with all metrics
        """
        logger.info("📊 Calculating all evaluation metrics...")
        
        metrics = {}
        
        # BLEU
        logger.debug("Calculating BLEU...")
        metrics.update(self.calculate_bleu(hypothesis, reference))
        
        # ROUGE
        logger.debug("Calculating ROUGE...")
        metrics.update(self.calculate_rouge(hypothesis, reference))
        
        # BERTScore (optional, slower)
        if include_bertscore:
            logger.debug("Calculating BERTScore...")
            metrics.update(self.calculate_bertscore(hypothesis, reference))
        
        # Flesch
        logger.debug("Calculating Flesch readability...")
        metrics.update(self.calculate_flesch_score(hypothesis))
        
        # Legal Accuracy
        logger.debug("Calculating legal accuracy...")
        metrics.update(self.calculate_legal_accuracy(hypothesis, reference))
        
        logger.info("✅ All metrics calculated")
        
        return metrics
    
    def format_metrics_report(self, metrics: Dict[str, Any]) -> str:
        """Format metrics as a readable report"""
        report = "📊 EVALUATION METRICS REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # BLEU
        if 'bleu' in metrics:
            report += "🔹 BLEU Scores:\n"
            report += f"  Overall BLEU: {metrics['bleu']:.4f}\n"
            if 'bleu_1' in metrics:
                report += f"  BLEU-1: {metrics['bleu_1']:.4f}\n"
                report += f"  BLEU-2: {metrics['bleu_2']:.4f}\n"
                report += f"  BLEU-3: {metrics['bleu_3']:.4f}\n"
                report += f"  BLEU-4: {metrics['bleu_4']:.4f}\n"
            report += "\n"
        
        # ROUGE
        if 'rouge1_f1' in metrics:
            report += "🔹 ROUGE Scores:\n"
            report += f"  ROUGE-1 F1: {metrics['rouge1_f1']:.4f}\n"
            report += f"  ROUGE-2 F1: {metrics['rouge2_f1']:.4f}\n"
            report += f"  ROUGE-L F1: {metrics['rougeL_f1']:.4f}\n"
            report += "\n"
        
        # BERTScore
        if 'bert_f1' in metrics:
            report += "🔹 BERTScore:\n"
            report += f"  F1: {metrics['bert_f1']:.4f}\n"
            report += f"  Precision: {metrics['bert_precision']:.4f}\n"
            report += f"  Recall: {metrics['bert_recall']:.4f}\n"
            report += "\n"
        
        # Readability
        if 'flesch_reading_ease' in metrics:
            report += "🔹 Readability:\n"
            report += f"  Flesch Reading Ease: {metrics['flesch_reading_ease']:.2f}\n"
            report += f"  Reading Level: {metrics.get('readability_level', 'N/A')}\n"
            report += f"  Grade Level: {metrics.get('flesch_kincaid_grade', 0):.2f}\n"
            report += "\n"
        
        # Legal Accuracy
        if 'legal_accuracy' in metrics:
            report += "🔹 Legal Accuracy:\n"
            report += f"  Overall: {metrics['legal_accuracy']:.4f}\n"
            report += f"  Legal Terms: {metrics['legal_term_accuracy']:.4f}\n"
            report += f"  Section Accuracy: {metrics['section_accuracy']:.4f}\n"
        
        report += "\n" + "=" * 60
        
        return report


def main():
    """Test evaluation metrics"""
    evaluator = LegalEvaluationMetrics()
    
    # Test texts
    reference = """Section 302 of the Indian Penal Code provides punishment for murder. 
    Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine."""
    
    hypothesis = """Section 302 IPC deals with the punishment for murder. 
    Anyone who commits murder can be punished with death penalty or life imprisonment, along with a fine."""
    
    logger.info("📝 Reference:")
    logger.info(reference)
    logger.info("\n📝 Hypothesis:")
    logger.info(hypothesis)
    
    # Calculate all metrics
    metrics = evaluator.evaluate_all(hypothesis, reference, include_bertscore=True)
    
    # Print report
    report = evaluator.format_metrics_report(metrics)
    logger.info(f"\n{report}")


if __name__ == "__main__":
    main()

