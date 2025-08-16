"""
Text tokenization functionality for splitting text into words and sentences.
"""

import re
import logging
from typing import List, Tuple, Optional
from collections import Counter


class Tokenizer:
    """Handles text tokenization into words, sentences, and n-grams."""
    
    def __init__(self, language: str = 'en'):
        """
        Initialize tokenizer.
        
        Args:
            language: Language code for language-specific processing
        """
        self.language = language
        self.logger = logging.getLogger(__name__)
        
        # Sentence boundary patterns by language
        self._sentence_patterns = {
            'en': r'[.!?]+\s+',
            'es': r'[.!?¡¿]+\s+'
        }
    
    def tokenize_words(self, text: str, preserve_case: bool = False) -> List[str]:
        """
        Split text into individual words.
        
        Args:
            text: Input text
            preserve_case: Whether to preserve original case
            
        Returns:
            List of word tokens
        """
        # Split on whitespace and punctuation boundaries
        words = re.findall(r'\b\w+\b', text)
        
        if not preserve_case:
            words = [word.lower() for word in words]
        
        self.logger.debug(f"Tokenized {len(words)} words from text")
        return words
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentence strings
        """
        pattern = self._sentence_patterns.get(self.language, r'[.!?]+\s+')
        sentences = re.split(pattern, text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        self.logger.debug(f"Tokenized {len(sentences)} sentences from text")
        return sentences
    
    def tokenize_lines(self, text: str) -> List[str]:
        """
        Split text into lines (for poetry processing).
        
        Args:
            text: Input text
            
        Returns:
            List of line strings
        """
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        
        self.logger.debug(f"Tokenized {len(lines)} lines from text")
        return lines
    
    def create_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """
        Create n-grams from token list.
        
        Args:
            tokens: List of tokens
            n: Size of n-grams
            
        Returns:
            List of n-gram tuples
        """
        if n <= 0 or len(tokens) < n:
            return []
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)
        
        self.logger.debug(f"Created {len(ngrams)} {n}-grams")
        return ngrams
    
    def create_word_frequency(self, tokens: List[str], min_freq: int = 1) -> Counter:
        """
        Create word frequency distribution.
        
        Args:
            tokens: List of word tokens
            min_freq: Minimum frequency to include word
            
        Returns:
            Counter object with word frequencies
        """
        freq_dist = Counter(tokens)
        
        # Filter by minimum frequency
        if min_freq > 1:
            freq_dist = Counter({word: count for word, count in freq_dist.items() 
                               if count >= min_freq})
        
        self.logger.debug(f"Created frequency distribution with {len(freq_dist)} unique words")
        return freq_dist
    
    def extract_vocabulary(self, tokens: List[str], min_freq: int = 2, max_vocab: Optional[int] = None) -> List[str]:
        """
        Extract vocabulary from tokens with frequency filtering.
        
        Args:
            tokens: List of word tokens
            min_freq: Minimum frequency to include word
            max_vocab: Maximum vocabulary size (keeps most frequent)
            
        Returns:
            List of vocabulary words
        """
        freq_dist = self.create_word_frequency(tokens, min_freq)
        
        # Get words sorted by frequency (descending)
        vocab = [word for word, _ in freq_dist.most_common()]
        
        if max_vocab and len(vocab) > max_vocab:
            vocab = vocab[:max_vocab]
        
        self.logger.info(f"Extracted vocabulary of {len(vocab)} words")
        return vocab
    
    def tokenize_for_generation(self, text: str) -> dict:
        """
        Complete tokenization pipeline for text generation.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing various tokenized forms
        """
        result = {
            'words': self.tokenize_words(text),
            'sentences': self.tokenize_sentences(text),
            'lines': self.tokenize_lines(text),
            'bigrams': [],
            'trigrams': [],
            'vocabulary': [],
            'word_freq': Counter()
        }
        
        if result['words']:
            result['bigrams'] = self.create_ngrams(result['words'], 2)
            result['trigrams'] = self.create_ngrams(result['words'], 3)
            result['vocabulary'] = self.extract_vocabulary(result['words'])
            result['word_freq'] = self.create_word_frequency(result['words'])
        
        self.logger.info(f"Complete tokenization: {len(result['words'])} words, "
                        f"{len(result['sentences'])} sentences, "
                        f"{len(result['vocabulary'])} vocab")
        
        return result
