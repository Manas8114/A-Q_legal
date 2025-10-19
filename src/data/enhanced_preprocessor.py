"""
Enhanced Data Preprocessor for A-Qlegal 2.0
Processes legal texts with all required fields for training
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from loguru import logger
from tqdm import tqdm
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


class EnhancedLegalPreprocessor:
    """
    Enhanced preprocessor with all required fields:
    - Section
    - Legal Text
    - Simplified Summary
    - Real-Life Example
    - Punishment
    - Keywords
    - Category (Civil/Criminal)
    """
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        
        # Legal domain keywords
        self.criminal_keywords = {
            'murder', 'theft', 'robbery', 'assault', 'rape', 'kidnapping',
            'criminal', 'offense', 'punishment', 'imprisonment', 'fine',
            'accused', 'culpable', 'homicide', 'grievous', 'hurt', 'ipc'
        }
        
        self.civil_keywords = {
            'contract', 'agreement', 'property', 'damages', 'compensation',
            'suit', 'plaintiff', 'defendant', 'civil', 'breach', 'liability',
            'dispute', 'claim', 'injunction', 'decree'
        }
        
        self.constitutional_keywords = {
            'article', 'constitution', 'fundamental', 'rights', 'directive',
            'principles', 'freedoms', 'equality', 'liberty', 'justice'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean legal text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep legal formatting
        text = re.sub(r'[^\w\s\.,;:()\-\[\]\"\'\/]', '', text)
        
        # Preserve section numbers
        text = re.sub(r'Section\s+(\d+[A-Z]?)', r'Section \1', text, flags=re.IGNORECASE)
        text = re.sub(r'Article\s+(\d+[A-Z]?)', r'Article \1', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def extract_section_number(self, text: str) -> str:
        """Extract section or article number"""
        # Try to find IPC section
        ipc_match = re.search(r'Section\s+(\d+[A-Z]?)\s*(?:of\s*)?(?:the\s*)?IPC', text, re.IGNORECASE)
        if ipc_match:
            return f"Section {ipc_match.group(1)} IPC"
        
        # Try to find general section
        section_match = re.search(r'Section\s+(\d+[A-Z]?)', text, re.IGNORECASE)
        if section_match:
            return f"Section {section_match.group(1)}"
        
        # Try to find article
        article_match = re.search(r'Article\s+(\d+[A-Z]?)', text, re.IGNORECASE)
        if article_match:
            return f"Article {article_match.group(1)}"
        
        return "General"
    
    def generate_simplified_summary(self, legal_text: str, max_length: int = 200) -> str:
        """Generate simplified summary from legal text"""
        # Clean text
        text = self.clean_text(legal_text)
        
        # If text is short enough, return as is
        if len(text) <= max_length:
            return text
        
        # Extract first few sentences
        sentences = sent_tokenize(text)
        
        summary = ""
        for sent in sentences:
            if len(summary) + len(sent) <= max_length:
                summary += sent + " "
            else:
                break
        
        # Simplify legal jargon
        simplifications = {
            'aforementioned': 'mentioned above',
            'pursuant to': 'according to',
            'notwithstanding': 'despite',
            'hereinafter': 'later in this document',
            'wherein': 'in which',
            'whereby': 'by which',
            'thereof': 'of it',
            'therein': 'in it',
            'thereof': 'of that',
            'whosoever': 'whoever',
            'shall': 'will',
            'shall be': 'is'
        }
        
        for legal, simple in simplifications.items():
            summary = re.sub(r'\b' + legal + r'\b', simple, summary, flags=re.IGNORECASE)
        
        return summary.strip()
    
    def generate_real_life_example(self, section: str, legal_text: str, category: str) -> str:
        """Generate real-life example based on legal text"""
        # Extract key concepts
        text_lower = legal_text.lower()
        
        # Example templates based on category
        if 'murder' in text_lower or 'section 302' in section.lower():
            return "Example: If A intentionally shoots B causing B's death, A can be charged with murder under this section."
        
        elif 'theft' in text_lower or 'section 378' in section.lower():
            return "Example: If A takes B's mobile phone from B's pocket without permission, A commits theft."
        
        elif 'assault' in text_lower or 'hurt' in text_lower:
            return "Example: If A punches B causing injury, A can be charged with assault."
        
        elif 'contract' in text_lower:
            return "Example: If A agrees to sell goods to B for Rs. 10,000 but fails to deliver, it's a breach of contract."
        
        elif 'article 21' in text_lower or 'right to life' in text_lower:
            return "Example: The state cannot deprive anyone of life or liberty except by due process of law."
        
        elif 'property' in text_lower:
            return "Example: If A owns a house and B claims ownership without proof, A can file a suit."
        
        else:
            return f"Example: This provision applies in legal situations related to {category}."
    
    def extract_punishment(self, legal_text: str) -> str:
        """Extract punishment details from legal text"""
        text_lower = legal_text.lower()
        
        # Look for punishment patterns
        punishment_patterns = [
            r'punish(?:ed|able)\s+with\s+([^.]+)',
            r'imprisonment\s+(?:for\s+)?(?:a\s+term\s+)?(?:of\s+)?([^,\.]+)',
            r'fine\s+(?:of\s+)?(?:up\s+to\s+)?([^,\.]+)',
            r'death\s+penalty',
            r'life\s+imprisonment'
        ]
        
        punishments = []
        for pattern in punishment_patterns:
            matches = re.findall(pattern, text_lower)
            punishments.extend(matches)
        
        if punishments:
            return "Punishment: " + "; ".join(punishments[:2])
        
        # Category-based defaults
        if 'murder' in text_lower:
            return "Punishment: Death penalty or life imprisonment, and fine"
        elif 'theft' in text_lower:
            return "Punishment: Imprisonment up to 3 years, or fine, or both"
        elif 'assault' in text_lower:
            return "Punishment: Imprisonment up to 1 year, or fine up to Rs. 1000, or both"
        else:
            return "Punishment: As prescribed by law"
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract important keywords from legal text"""
        # Tokenize
        words = word_tokenize(text.lower())
        
        # Remove stopwords and short words
        keywords = [
            word for word in words 
            if word.isalnum() 
            and word not in self.stop_words 
            and len(word) > 3
        ]
        
        # Count frequency
        from collections import Counter
        word_freq = Counter(keywords)
        
        # Get top keywords
        top_keywords = [word for word, _ in word_freq.most_common(top_k)]
        
        # Add legal-specific keywords
        legal_keywords = []
        for word in ['section', 'article', 'act', 'law', 'court', 'judgment']:
            if word in text.lower():
                legal_keywords.append(word)
        
        # Combine and deduplicate
        all_keywords = list(set(legal_keywords + top_keywords))
        
        return all_keywords[:top_k]
    
    def determine_category(self, text: str, existing_category: str = None) -> str:
        """Determine if text is Civil, Criminal, or Constitutional"""
        if existing_category:
            # Map existing categories
            category_lower = existing_category.lower()
            if 'criminal' in category_lower or 'ipc' in category_lower:
                return "Criminal"
            elif 'civil' in category_lower or 'contract' in category_lower or 'property' in category_lower:
                return "Civil"
            elif 'constitutional' in category_lower or 'fundamental' in category_lower:
                return "Constitutional"
        
        # Analyze text
        text_lower = text.lower()
        text_words = set(word_tokenize(text_lower))
        
        # Count category keywords
        criminal_score = len(text_words & self.criminal_keywords)
        civil_score = len(text_words & self.civil_keywords)
        constitutional_score = len(text_words & self.constitutional_keywords)
        
        # Determine category
        scores = {
            'Criminal': criminal_score,
            'Civil': civil_score,
            'Constitutional': constitutional_score
        }
        
        max_category = max(scores, key=scores.get)
        
        # Default to Criminal if no clear winner
        if scores[max_category] == 0:
            return "Criminal"
        
        return max_category
    
    def process_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single document with all required fields"""
        # Get original text
        original_text = doc.get('text', '')
        title = doc.get('title', '')
        existing_category = doc.get('category', '')
        
        # Clean text
        legal_text = self.clean_text(original_text)
        
        # Extract/generate all required fields
        section = self.extract_section_number(title or legal_text)
        simplified_summary = self.generate_simplified_summary(legal_text)
        category = self.determine_category(legal_text, existing_category)
        real_life_example = self.generate_real_life_example(section, legal_text, category)
        punishment = self.extract_punishment(legal_text)
        keywords = self.extract_keywords(legal_text)
        
        # Create enhanced document
        enhanced_doc = {
            'id': doc.get('id', f"doc_{hash(legal_text)%100000}"),
            'section': section,
            'legal_text': legal_text,
            'simplified_summary': simplified_summary,
            'real_life_example': real_life_example,
            'punishment': punishment,
            'keywords': keywords,
            'category': category,
            'subcategory': existing_category,
            'title': title,
            'source': doc.get('source', 'Unknown'),
            'original_text': original_text,
            'metadata': doc.get('metadata', {})
        }
        
        return enhanced_doc
    
    def process_dataset(self, dataset: List[Dict[str, Any]], output_file: str = None) -> List[Dict[str, Any]]:
        """Process entire dataset"""
        logger.info(f"📊 Processing {len(dataset)} documents...")
        
        enhanced_dataset = []
        
        for doc in tqdm(dataset, desc="Processing documents"):
            try:
                enhanced_doc = self.process_document(doc)
                enhanced_dataset.append(enhanced_doc)
            except Exception as e:
                logger.warning(f"⚠️  Error processing document {doc.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"✅ Processed {len(enhanced_dataset)} documents")
        
        # Save if output file specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_dataset, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Saved to {output_file}")
        
        return enhanced_dataset
    
    def create_legal_to_layman_pairs(
        self, 
        dataset: List[Dict[str, Any]]
    ) -> List[Tuple[str, str]]:
        """Create Legal → Layman translation pairs for training"""
        pairs = []
        
        for doc in dataset:
            legal_text = doc.get('legal_text', '')
            simplified = doc.get('simplified_summary', '')
            
            if legal_text and simplified and legal_text != simplified:
                pairs.append((legal_text, simplified))
        
        logger.info(f"✅ Created {len(pairs)} legal-to-layman pairs")
        return pairs
    
    def get_statistics(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not dataset:
            return {}
        
        # Category distribution
        categories = [doc.get('category', 'Unknown') for doc in dataset]
        from collections import Counter
        category_counts = Counter(categories)
        
        # Average lengths
        avg_legal_length = sum(len(doc.get('legal_text', '')) for doc in dataset) / len(dataset)
        avg_summary_length = sum(len(doc.get('simplified_summary', '')) for doc in dataset) / len(dataset)
        
        # Keywords
        all_keywords = []
        for doc in dataset:
            all_keywords.extend(doc.get('keywords', []))
        keyword_counts = Counter(all_keywords)
        
        stats = {
            'total_documents': len(dataset),
            'category_distribution': dict(category_counts),
            'avg_legal_text_length': int(avg_legal_length),
            'avg_summary_length': int(avg_summary_length),
            'top_keywords': dict(keyword_counts.most_common(20)),
            'unique_sections': len(set(doc.get('section', '') for doc in dataset))
        }
        
        return stats


def main():
    """Main function for testing"""
    preprocessor = EnhancedLegalPreprocessor()
    
    # Load existing dataset
    dataset_file = Path("data/expanded_legal_dataset.json")
    
    if dataset_file.exists():
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        logger.info(f"📚 Loaded {len(dataset)} documents")
        
        # Process dataset
        enhanced_dataset = preprocessor.process_dataset(
            dataset,
            output_file="data/enhanced_legal_dataset_v2.json"
        )
        
        # Get statistics
        stats = preprocessor.get_statistics(enhanced_dataset)
        
        logger.info("\n📊 Dataset Statistics:")
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        
        # Save statistics
        with open("data/enhanced_dataset_statistics_v2.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Create training pairs
        pairs = preprocessor.create_legal_to_layman_pairs(enhanced_dataset)
        
        with open("data/legal_to_layman_pairs.json", 'w', encoding='utf-8') as f:
            json.dump([{'legal': p[0], 'layman': p[1]} for p in pairs], f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Created {len(pairs)} training pairs")
    
    else:
        logger.error(f"❌ Dataset not found: {dataset_file}")


if __name__ == "__main__":
    main()

