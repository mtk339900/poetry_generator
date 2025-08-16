# poetry_generator/__init__.py
"""Poetry Generator - A custom text generation system for poetry and prose."""

__version__ = "1.0.0"
__author__ = "Poetry Generator Team"

# poetry_generator/corpus/__init__.py
"""Corpus handling modules for text processing and preparation."""

from .loader import CorpusLoader
from .cleaner import TextCleaner
from .tokenizer import Tokenizer

__all__ = ['CorpusLoader', 'TextCleaner', 'Tokenizer']

# poetry_generator/generation/__init__.py
"""Text generation modules implementing various generation strategies."""

from .template import TemplateGenerator
from .markov import MarkovGenerator
from .hybrid import HybridGenerator

__all__ = ['TemplateGenerator', 'MarkovGenerator', 'HybridGenerator']

# poetry_generator/style/__init__.py
"""Style and poetic device modules."""

from .rhyme import RhymeEngine
from .figurative import FigurativeLanguage

__all__ = ['RhymeEngine', 'FigurativeLanguage']

# poetry_generator/cli/__init__.py
"""Command line interface modules."""

# poetry_generator/utils/__init__.py
"""Utility modules for configuration and helper functions."""

from .config import ConfigManager
from .random_utils import RandomUtils

__all__ = ['ConfigManager', 'RandomUtils']
