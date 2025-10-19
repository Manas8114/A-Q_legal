"""
Multilingual Support for A-Qlegal 2.0
Supports Hindi, Tamil, Bengali, Telugu, and other Indian languages
"""

from typing import Optional, List, Dict, Any
from loguru import logger
import re


try:
    from googletrans import Translator as GoogleTranslator
    GOOGLETRANS_AVAILABLE = True
except:
    GOOGLETRANS_AVAILABLE = False
    logger.warning("googletrans not available")

try:
    from deep_translator import GoogleTranslator as DeepGoogleTranslator
    DEEP_TRANSLATOR_AVAILABLE = True
except:
    DEEP_TRANSLATOR_AVAILABLE = False
    logger.warning("deep_translator not available")

try:
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
except:
    LANGDETECT_AVAILABLE = False
    logger.warning("langdetect not available")


class MultilingualLegalSystem:
    """
    Multilingual support for Indian legal system
    
    Supported languages:
    - English (en)
    - Hindi (hi) - हिंदी
    - Tamil (ta) - தமிழ்
    - Bengali (bn) - বাংলা
    - Telugu (te) - తెలుగు
    - Marathi (mr) - मराठी
    - Gujarati (gu) - ગુજરાતી
    - Kannada (kn) - ಕನ್ನಡ
    - Malayalam (ml) - മലയാളം
    - Punjabi (pa) - ਪੰਜਾਬੀ
    """
    
    def __init__(self, primary_translator: str = "deep"):
        """
        Initialize multilingual system
        
        Args:
            primary_translator: 'google' or 'deep' (deep_translator is more reliable)
        """
        self.primary_translator = primary_translator
        
        # Language codes and names
        self.languages = {
            'en': 'English',
            'hi': 'Hindi (हिंदी)',
            'ta': 'Tamil (தமிழ்)',
            'bn': 'Bengali (বাংলা)',
            'te': 'Telugu (తెలుగు)',
            'mr': 'Marathi (मराठी)',
            'gu': 'Gujarati (ગુજરાતી)',
            'kn': 'Kannada (ಕನ್ನಡ)',
            'ml': 'Malayalam (മലയാളം)',
            'pa': 'Punjabi (ਪੰਜਾਬੀ)'
        }
        
        # Initialize translators
        if primary_translator == "google" and GOOGLETRANS_AVAILABLE:
            self.translator = GoogleTranslator()
        elif DEEP_TRANSLATOR_AVAILABLE:
            self.translator = "deep"  # Create instances as needed
        else:
            self.translator = None
            logger.warning("No translation service available")
        
        # Legal terms dictionary (preserve these during translation)
        self.legal_terms = {
            'en': {
                'Section': 'Section',
                'Article': 'Article',
                'IPC': 'IPC',
                'CrPC': 'CrPC',
                'Constitution': 'Constitution',
                'Supreme Court': 'Supreme Court',
                'High Court': 'High Court',
                'Punishment': 'Punishment',
                'Imprisonment': 'Imprisonment',
                'Fine': 'Fine'
            },
            'hi': {
                'Section': 'धारा',
                'Article': 'अनुच्छेद',
                'IPC': 'आईपीसी',
                'CrPC': 'सीआरपीसी',
                'Constitution': 'संविधान',
                'Supreme Court': 'सर्वोच्च न्यायालय',
                'High Court': 'उच्च न्यायालय',
                'Punishment': 'सज़ा',
                'Imprisonment': 'कारावास',
                'Fine': 'जुर्माना'
            },
            'ta': {
                'Section': 'பிரிவு',
                'Article': 'கட்டுரை',
                'IPC': 'IPC',
                'CrPC': 'CrPC',
                'Constitution': 'அரசியலமைப்பு',
                'Supreme Court': 'உச்ச நீதிமன்றம்',
                'High Court': 'உயர் நீதிமன்றம்',
                'Punishment': 'தண்டனை',
                'Imprisonment': 'சிறைவாசம்',
                'Fine': 'அபராதம்'
            }
            # Add more languages as needed
        }
        
        logger.info(f"🌍 Multilingual system initialized with {len(self.languages)} languages")
    
    def detect_language(self, text: str) -> str:
        """Detect language of text"""
        if not LANGDETECT_AVAILABLE:
            return 'en'
        
        try:
            lang = detect(text)
            return lang if lang in self.languages else 'en'
        except:
            return 'en'
    
    def translate(
        self,
        text: str,
        target_lang: str = 'hi',
        source_lang: str = 'en',
        preserve_legal_terms: bool = True
    ) -> str:
        """
        Translate text to target language
        
        Args:
            text: Text to translate
            target_lang: Target language code
            source_lang: Source language code
            preserve_legal_terms: Keep legal terms untranslated
        """
        if not text or not text.strip():
            return text
        
        # If same language, return as is
        if source_lang == target_lang:
            return text
        
        # Check if target language is supported
        if target_lang not in self.languages:
            logger.warning(f"Language {target_lang} not supported")
            return text
        
        try:
            # Replace legal terms with placeholders if preserving
            placeholders = {}
            processed_text = text
            
            if preserve_legal_terms:
                processed_text, placeholders = self._replace_legal_terms(text, source_lang)
            
            # Translate
            if self.translator == "deep" and DEEP_TRANSLATOR_AVAILABLE:
                translator = DeepGoogleTranslator(source=source_lang, target=target_lang)
                
                # Translate in chunks if text is too long
                if len(processed_text) > 4000:
                    chunks = self._split_text(processed_text, 4000)
                    translated_chunks = [translator.translate(chunk) for chunk in chunks]
                    translated = ' '.join(translated_chunks)
                else:
                    translated = translator.translate(processed_text)
            
            elif GOOGLETRANS_AVAILABLE and hasattr(self.translator, 'translate'):
                result = self.translator.translate(processed_text, src=source_lang, dest=target_lang)
                translated = result.text
            
            else:
                logger.warning("No translator available, returning original text")
                return text
            
            # Restore legal terms
            if preserve_legal_terms:
                translated = self._restore_legal_terms(translated, placeholders, target_lang)
            
            return translated
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text
    
    def _replace_legal_terms(self, text: str, lang: str) -> tuple:
        """Replace legal terms with placeholders"""
        placeholders = {}
        processed_text = text
        
        if lang in self.legal_terms:
            for i, (term_key, term) in enumerate(self.legal_terms[lang].items()):
                # Case-insensitive replacement
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                matches = pattern.findall(processed_text)
                
                if matches:
                    placeholder = f"__LEGAL_TERM_{i}__"
                    placeholders[placeholder] = (term_key, matches[0])
                    processed_text = pattern.sub(placeholder, processed_text, count=1)
        
        return processed_text, placeholders
    
    def _restore_legal_terms(self, text: str, placeholders: Dict, target_lang: str) -> str:
        """Restore legal terms in target language"""
        restored_text = text
        
        for placeholder, (term_key, original) in placeholders.items():
            # Get target language term
            if target_lang in self.legal_terms and term_key in self.legal_terms[target_lang]:
                target_term = self.legal_terms[target_lang][term_key]
            else:
                target_term = original
            
            restored_text = restored_text.replace(placeholder, target_term)
        
        return restored_text
    
    def _split_text(self, text: str, max_length: int = 4000) -> List[str]:
        """Split text into chunks for translation"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 2 <= max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def translate_legal_document(
        self,
        document: Dict[str, Any],
        target_lang: str = 'hi'
    ) -> Dict[str, Any]:
        """Translate a legal document"""
        translated_doc = document.copy()
        
        # Translate text fields
        if 'text' in document:
            translated_doc['text'] = self.translate(document['text'], target_lang=target_lang)
        
        if 'title' in document:
            translated_doc['title'] = self.translate(document['title'], target_lang=target_lang)
        
        if 'simplified_summary' in document:
            translated_doc['simplified_summary'] = self.translate(
                document['simplified_summary'],
                target_lang=target_lang
            )
        
        if 'real_life_example' in document:
            translated_doc['real_life_example'] = self.translate(
                document['real_life_example'],
                target_lang=target_lang
            )
        
        # Add language metadata
        translated_doc['language'] = target_lang
        translated_doc['original_language'] = 'en'
        
        return translated_doc
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages"""
        return self.languages.copy()
    
    def format_multilingual_response(
        self,
        answer: str,
        languages: List[str] = ['en', 'hi']
    ) -> Dict[str, str]:
        """Format response in multiple languages"""
        responses = {}
        
        for lang in languages:
            if lang == 'en':
                responses[lang] = answer
            else:
                responses[lang] = self.translate(answer, target_lang=lang, source_lang='en')
        
        return responses


def main():
    """Test multilingual system"""
    multilingual = MultilingualLegalSystem(primary_translator="deep")
    
    # Test text
    legal_text = """Section 302 of the Indian Penal Code deals with punishment for murder. 
    Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine."""
    
    logger.info(f"📝 Original (English):\n{legal_text}\n")
    
    # Translate to Hindi
    hindi_text = multilingual.translate(legal_text, target_lang='hi', source_lang='en')
    logger.info(f"🇮🇳 Hindi:\n{hindi_text}\n")
    
    # Translate to Tamil
    tamil_text = multilingual.translate(legal_text, target_lang='ta', source_lang='en')
    logger.info(f"🇮🇳 Tamil:\n{tamil_text}\n")
    
    # Translate to Bengali
    bengali_text = multilingual.translate(legal_text, target_lang='bn', source_lang='en')
    logger.info(f"🇮🇳 Bengali:\n{bengali_text}\n")
    
    # Translate to Telugu
    telugu_text = multilingual.translate(legal_text, target_lang='te', source_lang='en')
    logger.info(f"🇮🇳 Telugu:\n{telugu_text}\n")
    
    # Detect language
    detected = multilingual.detect_language(hindi_text)
    logger.info(f"🔍 Detected language: {detected}")
    
    # Multilingual response
    response = multilingual.format_multilingual_response(
        "The punishment for murder is death penalty or life imprisonment.",
        languages=['en', 'hi', 'ta']
    )
    
    logger.info("\n📢 Multilingual Response:")
    for lang, text in response.items():
        logger.info(f"{lang}: {text}")


if __name__ == "__main__":
    main()

