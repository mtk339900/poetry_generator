"""
Hybrid text generation combining template-based and Markov chain approaches.
"""

import logging
import random
from typing import List, Dict, Any, Optional, Tuple
from .template import TemplateGenerator
from .markov import MarkovGenerator
from ..utils.random_utils import RandomUtils


class HybridGenerator:
    """Combines template-based and Markov chain generation for varied output."""
    
    def __init__(self, template_generator: TemplateGenerator,
                 markov_generator: MarkovGenerator,
                 random_utils: Optional[RandomUtils] = None):
        """
        Initialize hybrid generator.
        
        Args:
            template_generator: Initialized template generator
            markov_generator: Trained Markov generator
            random_utils: Random utilities instance
        """
        self.template_gen = template_generator
        self.markov_gen = markov_generator
        self.random_utils = random_utils or RandomUtils()
        self.logger = logging.getLogger(__name__)
        
        # Hybrid generation parameters
        self.blend_ratio = 0.5  # 0.0 = all template, 1.0 = all Markov
        self.enhancement_probability = 0.3  # Probability of enhancing template with Markov
        self.variation_modes = ['alternate', 'blend', 'enhance', 'compete']
        self.current_mode = 'blend'
    
    def set_blend_ratio(self, ratio: float) -> None:
        """
        Set the blend ratio between template and Markov generation.
        
        Args:
            ratio: Blend ratio (0.0 = pure template, 1.0 = pure Markov)
        """
        self.blend_ratio = max(0.0, min(1.0, ratio))
        self.logger.debug(f"Blend ratio set to {self.blend_ratio}")
    
    def set_generation_mode(self, mode: str) -> None:
        """
        Set the hybrid generation mode.
        
        Args:
            mode: Generation mode ('alternate', 'blend', 'enhance', 'compete')
        """
        if mode in self.variation_modes:
            self.current_mode = mode
            self.logger.debug(f"Generation mode set to {mode}")
        else:
            self.logger.warning(f"Unknown mode {mode}, keeping {self.current_mode}")
    
    def generate_text(self, max_length: int = 50,
                     category: str = None,
                     seed_words: Optional[List[str]] = None,
                     custom_template: str = None) -> str:
        """
        Generate text using hybrid approach.
        
        Args:
            max_length: Maximum length for generated text
            category: Template category for template generation
            seed_words: Seed words for thematic guidance
            custom_template: Custom template to use
            
        Returns:
            Generated text string
        """
        if self.current_mode == 'alternate':
            return self._generate_alternating(max_length, category, seed_words)
        elif self.current_mode == 'blend':
            return self._generate_blended(max_length, category, seed_words, custom_template)
        elif self.current_mode == 'enhance':
            return self._generate_enhanced(category, seed_words, custom_template)
        elif self.current_mode == 'compete':
            return self._generate_competitive(max_length, category, seed_words, custom_template)
        else:
            return self._generate_blended(max_length, category, seed_words, custom_template)
    
    def _generate_alternating(self, max_length: int, category: str = None,
                            seed_words: Optional[List[str]] = None) -> str:
        """
        Generate by alternating between template and Markov methods.
        
        Args:
            max_length: Maximum text length
            category: Template category
            seed_words: Seed words
            
        Returns:
            Generated text
        """
        parts = []
        use_template = random.choice([True, False])
        
        remaining_length = max_length
        while remaining_length > 5:
            if use_template:
                part = self.template_gen.generate_from_template(category, seed_words)
                if part:
                    parts.append(part)
                    remaining_length -= len(part.split())
            else:
                # Generate short Markov sequence
                markov_length = min(remaining_length, random.randint(5, 15))
                markov_words = self.markov_gen.generate_text(
                    max_length=markov_length,
                    end_patterns=['.', '!', '?']
                )
                if markov_words:
                    part = ' '.join(markov_words)
                    parts.append(part)
                    remaining_length -= len(markov_words)
            
            use_template = not use_template
            
            if not parts or remaining_length <= 0:
                break
        
        result = ' '.join(parts)
        self.logger.debug(f"Generated alternating text: {len(result)} characters")
        return result
    
    def _generate_blended(self, max_length: int, category: str = None,
                         seed_words: Optional[List[str]] = None,
                         custom_template: str = None) -> str:
        """
        Generate by blending template and Markov based on blend ratio.
        
        Args:
            max_length: Maximum text length
            category: Template category
            seed_words: Seed words
            custom_template: Custom template
            
        Returns:
            Generated text
        """
        if random.random() < self.blend_ratio:
            # Use Markov generation
            words = self.markov_gen.generate_text(
                max_length=max_length,
                end_patterns=['.', '!', '?']
            )
            result = ' '.join(words) if words else ""
            
            # Optionally enhance with template elements
            if random.random() < 0.3 and result:
                template_part = self.template_gen.generate_from_template(category, seed_words)
                if template_part:
                    result = f"{template_part} {result}"
        else:
            # Use template generation
            result = self.template_gen.generate_from_template(
                category, seed_words, custom_template
            )
            
            # Optionally enhance with Markov elements
            if random.random() < 0.3 and result:
                markov_words = self.markov_gen.generate_text(
                    max_length=10,
                    end_patterns=['.', '!', '?']
                )
                if markov_words:
                    markov_part = ' '.join(markov_words)
                    result = f"{result} {markov_part}"
        
        self.logger.debug(f"Generated blended text: {len(result)} characters")
        return result
    
    def _generate_enhanced(self, category: str = None,
                          seed_words: Optional[List[str]] = None,
                          custom_template: str = None) -> str:
        """
        Generate template-based text enhanced with Markov elements.
        
        Args:
            category: Template category
            seed_words: Seed words
            custom_template: Custom template
            
        Returns:
            Generated text
        """
        # Start with template
        base_text = self.template_gen.generate_from_template(
            category, seed_words, custom_template
        )
        
        if not base_text:
            # Fallback to Markov
            words = self.markov_gen.generate_text(max_length=20)
            return ' '.join(words) if words else ""
        
        # Enhance with Markov-generated phrases
        enhanced_parts = []
        words = base_text.split()
        
        i = 0
        while i < len(words):
            enhanced_parts.append(words[i])
            
            # Randomly insert Markov-generated phrases
            if random.random() < self.enhancement_probability:
                # Use current word as seed for Markov generation
                seed_state = self._find_markov_state_with_word(words[i])
                if seed_state:
                    markov_words = self.markov_gen.generate_text(
                        max_length=random.randint(3, 8),
                        start_state=seed_state
                    )
                    if markov_words and len(markov_words) > len(seed_state):
                        # Add the generated continuation (skip the seed part)
                        continuation = markov_words[len(seed_state):]
                        enhanced_parts.extend(continuation)
            
            i += 1
        
        result = ' '.join(enhanced_parts)
        self.logger.debug(f"Generated enhanced text: {len(result)} characters")
        return result
    
    def _generate_competitive(self, max_length: int, category: str = None,
                            seed_words: Optional[List[str]] = None,
                            custom_template: str = None) -> str:
        """
        Generate multiple candidates and select the best one.
        
        Args:
            max_length: Maximum text length
            category: Template category
            seed_words: Seed words
            custom_template: Custom template
            
        Returns:
            Best generated text
        """
        candidates = []
        
        # Generate template candidate
        template_text = self.template_gen.generate_from_template(
            category, seed_words, custom_template
        )
        if template_text:
            candidates.append(('template', template_text))
        
        # Generate Markov candidate
        markov_words = self.markov_gen.generate_text(
            max_length=max_length,
            end_patterns=['.', '!', '?']
        )
        if markov_words:
            markov_text = ' '.join(markov_words)
            candidates.append(('markov', markov_text))
        
        # Generate hybrid candidate
        hybrid_text = self._generate_blended(max_length, category, seed_words, custom_template)
        if hybrid_text:
            candidates.append(('hybrid', hybrid_text))
        
        if not candidates:
            return ""
        
        # Select best candidate based on scoring
        best_candidate = max(candidates, key=lambda x: self._score_text(x[1]))
        result = best_candidate[1]
        
        self.logger.debug(f"Generated competitive text ({best_candidate[0]}): {len(result)} characters")
        return result
    
    def _find_markov_state_with_word(self, word: str) -> Optional[Tuple[str, ...]]:
        """
        Find a Markov state containing the given word.
        
        Args:
            word: Word to find in states
            
        Returns:
            Markov state tuple or None
        """
        word_lower = word.lower()
        matching_states = [
            state for state in self.markov_gen.chain.keys()
            if word_lower in [w.lower() for w in state]
        ]
        
        return random.choice(matching_states) if matching_states else None
    
    def _score_text(self, text: str) -> float:
        """
        Score generated text for quality selection.
        
        Args:
            text: Text to score
            
        Returns:
            Quality score (higher is better)
        """
        if not text:
            return 0.0
        
        score = 0.0
        words = text.split()
        
        # Length factor (prefer moderate length)
        length_score = min(1.0, len(words) / 20.0) * (1.0 - max(0.0, (len(words) - 50) / 50.0))
        score += length_score * 0.3
        
        # Vocabulary diversity
        unique_words = len(set(word.lower() for word in words))
        diversity_score = unique_words / len(words) if words else 0
        score += diversity_score * 0.4
        
        # Sentence completeness (ends with punctuation)
        if text.rstrip().endswith(('.', '!', '?')):
            score += 0.3
        
        return score
    
    def generate_multiple(self, count: int, **kwargs) -> List[str]:
        """
        Generate multiple hybrid texts.
        
        Args:
            count: Number of texts to generate
            **kwargs: Arguments passed to generate_text()
            
        Returns:
            List of generated texts
        """
        return [self.generate_text(**kwargs) for _ in range(count)]
