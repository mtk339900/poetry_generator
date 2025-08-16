"""
Corpus loading functionality for text data preparation.
"""

import os
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path


class CorpusLoader:
    """Handles loading and validation of text corpora from various sources."""
    
    def __init__(self, encoding: str = 'utf-8'):
        """
        Initialize corpus loader.
        
        Args:
            encoding: Text file encoding to use
        """
        self.encoding = encoding
        self.logger = logging.getLogger(__name__)
    
    def load_text_file(self, filepath: str) -> str:
        """
        Load text content from a single file.
        
        Args:
            filepath: Path to the text file
            
        Returns:
            Raw text content
            
        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file can't be read
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {filepath}")
        
        try:
            with open(path, 'r', encoding=self.encoding) as f:
                content = f.read()
            self.logger.info(f"Loaded corpus from {filepath}: {len(content)} characters")
            return content
        except IOError as e:
            self.logger.error(f"Failed to read corpus file {filepath}: {e}")
            raise
    
    def load_directory(self, directory: str, extensions: List[str] = None) -> Dict[str, str]:
        """
        Load all text files from a directory.
        
        Args:
            directory: Directory path containing corpus files
            extensions: List of file extensions to include (default: ['.txt'])
            
        Returns:
            Dictionary mapping filename to content
        """
        if extensions is None:
            extensions = ['.txt']
        
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Corpus directory not found: {directory}")
        
        corpus_data = {}
        for ext in extensions:
            for filepath in dir_path.glob(f"*{ext}"):
                try:
                    content = self.load_text_file(str(filepath))
                    corpus_data[filepath.name] = content
                except IOError:
                    self.logger.warning(f"Skipping unreadable file: {filepath}")
        
        self.logger.info(f"Loaded {len(corpus_data)} corpus files from {directory}")
        return corpus_data
    
    def load_multiple_files(self, filepaths: List[str]) -> str:
        """
        Load and concatenate multiple text files.
        
        Args:
            filepaths: List of file paths to load
            
        Returns:
            Concatenated text content
        """
        combined_content = []
        for filepath in filepaths:
            try:
                content = self.load_text_file(filepath)
                combined_content.append(content)
            except (FileNotFoundError, IOError):
                self.logger.warning(f"Skipping unavailable file: {filepath}")
        
        result = "\n\n".join(combined_content)
        self.logger.info(f"Combined {len(filepaths)} files into {len(result)} characters")
        return result
    
    def validate_corpus(self, content: str, min_words: int = 100) -> bool:
        """
        Validate that corpus content meets minimum requirements.
        
        Args:
            content: Text content to validate
            min_words: Minimum number of words required
            
        Returns:
            True if corpus is valid
        """
        if not content or not content.strip():
            self.logger.error("Corpus is empty")
            return False
        
        word_count = len(content.split())
        if word_count < min_words:
            self.logger.error(f"Corpus too small: {word_count} words (minimum {min_words})")
            return False
        
        self.logger.info(f"Corpus validation passed: {word_count} words")
        return True
