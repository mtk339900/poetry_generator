"""
Text cleaning and normalization functionality.
"""

import re
import logging
from typing import List, Set, Optional


class TextCleaner:
    """Handles text cleaning, normalization, and preprocessing."""
    
    def __init__(self, language: str = 'en'):
        """
        Initialize text cleaner.
        
        Args:
            language: Language code for language-specific processing
        """
        self.language = language
        self.logger = logging.getLogger(__name__)
        
        # Common stopwords by language
        self._stopwords = {
            'en': {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that',
                'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
                'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her',
                'its', 'our', 'their'
            },
            'es': {
                'el', 'la', 'los', 'las', 'un', 'una', 'y', 'o', 'pero', 'en', 'de',
                'a', 'por', 'para', 'con', 'sin', 'es', 'son', 'era', 'fueron',
                'ser', 'estar', 'tiene', 'tienen', 'tuvo', 'haber', 'he', 'ha',
                'han', 'había', 'habían', 'que', 'qué', 'quien', 'quién', 'cual',
                'cuál', 'cuando', 'cuándo', 'donde', 'dónde', 'como', 'cómo',
                'yo', 'tú', 'él', 'ella', 'nosotros', 'vosotros', 'ellos', 'ellas',
                'me', 'te', 'le', 'nos', 'os', 'les', 'mi', 'tu', 'su', 'nuestro',
                'vuestro', 'este', 'esta', 'estos', 'estas', 'ese', 'esa', 'esos',
                'esas', 'aquel', 'aquella', 'aquellos', 'aquellas'
            }
        }
    
    def remove_punctuation(self, text: str, keep_sentence_endings: bool = True) -> str:
        """
        Remove punctuation from text.
        
        Args:
            text: Input text
            keep_sentence_endings: Whether to preserve sentence ending punctuation
            
        Returns:
            Text with punctuation removed
        """
        if keep_sentence_endings:
            # Replace sentence endings with placeholders
            text = re.sub(r'[.!?]+', ' SENTENCE_END ', text)
            # Remove other punctuation
            text = re.sub(r'[^\w\s]', ' ', text)
            # Restore sentence endings
            text = re.sub(r'\s*SENTENCE_END\s*', '. ', text)
        else:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        return text
    
    def normalize_case(self, text: str, method: str = 'lower') -> str:
        """
        Normalize text case.
        
        Args:
            text: Input text
            method: Normalization method ('lower', 'upper', 'title', 'sentence')
            
        Returns:
            Case-normalized text
        """
        if method == 'lower':
            return text.lower()
        elif method == 'upper':
            return text.upper()
        elif method == 'title':
            return text.title()
        elif method == 'sentence':
            # Capitalize first letter of each sentence
            sentences = re.split(r'([.!?]+)', text)
            normalized = []
            for i, sentence in enumerate(sentences):
                if i % 2 == 0 and sentence.strip():  # Text parts (not punctuation)
                    sentence = sentence.strip()
                    if sentence:
                        sentence = sentence[0].upper() + sentence[1:].lower()
                normalized.append(sentence)
            return ''.join(normalized)
        else:
            return text
    
    def remove_extra_whitespace(self, text: str) -> str:
        """
        Remove extra whitespace and normalize spacing.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        return text.strip()
    
    def filter_stopwords(self, words: List[str], custom_stopwords: Optional[Set[str]] = None) -> List[str]:
        """
        Remove stopwords from word list.
        
        Args:
            words: List of words to filter
            custom_stopwords: Additional custom stopwords to remove
            
        Returns:
            Filtered word list
        """
        stopwords = self._stopwords.get(self.language, set())
        if custom_stopwords:
            stopwords = stopwords.union(custom_stopwords)
        
        return [word for word in words if word.lower() not in stopwords]
    
    def remove_short_words(self, words: List[str], min_length: int = 2) -> List[str]:
        """
        Remove words shorter than specified length.
        
        Args:
            words: List of words to filter
            min_length: Minimum word length to keep
            
        Returns:
            Filtered word list
        """
        return [word for word in words if len(word) >= min_length]
    
    def remove_numbers(self, text: str) -> str:
        """
        Remove numeric content from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with numbers removed
        """
        return re.sub(r'\b\d+\b', '', text)
    
    def clean_text(self, text: str, 
                   remove_punct: bool = True,
                   normalize_case: bool = True,
                   remove_nums: bool = True,
                   remove_extra_space: bool = True,
                   case_method: str = 'lower') -> str:
        """
        Apply comprehensive text cleaning.
        
        Args:
            text: Input text
            remove_punct: Whether to remove punctuation
            normalize_case: Whether to normalize case
            remove_nums: Whether to remove numbers
            remove_extra_space: Whether to normalize whitespace
            case_method: Case normalization method
            
        Returns:
            Cleaned text
        """
        original_length = len(text)
        
        if remove_nums:
            text = self.remove_numbers(text)
        
        if remove_punct:
            text = self.remove_punctuation(text, keep_sentence_endings=False)
        
        if normalize_case:
            text = self.normalize_case(text, case_method)
        
        if remove_extra_space:
            text = self.remove_extra_whitespace(text)
        
        self.logger.debug(f"Cleaned text: {original_length} -> {len(text)} characters")
        return text
