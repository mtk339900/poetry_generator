"""
Markov chain text generation using n-gram models.
"""

import logging
import random
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, Counter
from ..utils.random_utils import RandomUtils


class MarkovGenerator:
    """Generates text using Markov chain n-gram models."""
    
    def __init__(self, n: int = 2, temperature: float = 1.0, random_utils: Optional[RandomUtils] = None):
        """
        Initialize Markov generator.
        
        Args:
            n: Order of Markov chain (n-gram size)
            temperature: Temperature parameter for randomness control (0.1-2.0)
            random_utils: Random utilities instance
        """
        self.n = max(1, n)
        self.temperature = max(0.1, min(2.0, temperature))
        self.random_utils = random_utils or RandomUtils()
        self.logger = logging.getLogger(__name__)
        
        # Model data structures
        self.chain: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.sentence_starters: List[Tuple[str, ...]] = []
        self.vocabulary: set = set()
        self.is_trained = False
    
    def train(self, tokens: List[str], sentence_boundaries: Optional[List[int]] = None) -> None:
        """
        Train Markov model on token sequence.
        
        Args:
            tokens: List of word tokens
            sentence_boundaries: Optional list of sentence boundary indices
        """
        if len(tokens) < self.n:
            self.logger.warning(f"Token list too short for {self.n}-gram model")
            return
        
        self.vocabulary = set(tokens)
        self._build_chain(tokens, sentence_boundaries)
        self.is_trained = True
        
        chain_size = sum(len(counter) for counter in self.chain.values())
        self.logger.info(f"Trained {self.n}-gram model: {len(self.chain)} states, "
                        f"{chain_size} transitions, {len(self.vocabulary)} vocabulary")
    
    def _build_chain(self, tokens: List[str], sentence_boundaries: Optional[List[int]] = None) -> None:
        """
        Build the Markov chain from tokens.
        
        Args:
            tokens: List of word tokens
            sentence_boundaries: Optional sentence boundary indices
        """
        # Create n-grams and transitions
        for i in range(len(tokens) - self.n):
            state = tuple(tokens[i:i + self.n])
            next_word = tokens[i + self.n]
            
            self.chain[state][next_word] += 1
            
            # Track sentence starters
            if sentence_boundaries and i in sentence_boundaries:
                self.sentence_starters.append(state)
            elif i == 0:  # First n-gram is always a starter
                self.sentence_starters.append(state)
        
        # If no sentence boundaries provided, use heuristics
        if not sentence_boundaries:
            self._detect_sentence_starters(tokens)
    
    def _detect_sentence_starters(self, tokens: List[str]) -> None:
        """
        Detect likely sentence starters using capitalization heuristics.
        
        Args:
            tokens: List of word tokens
        """
        for i in range(len(tokens) - self.n):
            state = tuple(tokens[i:i + self.n])
            # Check if first word looks like sentence start
            if tokens[i][0].isupper() or i == 0:
                self.sentence_starters.append(state)
    
    def _select_next_word(self, state: Tuple[str, ...]) -> Optional[str]:
        """
        Select next word based on transition probabilities and temperature.
        
        Args:
            state: Current state (n-gram)
            
        Returns:
            Selected next word or None if no transitions available
        """
        if state not in self.chain:
            return None
        
        candidates = self.chain[state]
        if not candidates:
            return None
        
        # Apply temperature to probabilities
        words, counts = zip(*candidates.items())
        probabilities = self.random_utils.apply_temperature(list(counts), self.temperature)
        
        # Select word based on adjusted probabilities
        return self.random_utils.weighted_choice(list(words), probabilities)
    
    def generate_text(self, max_length: int = 50, 
                     start_state: Optional[Tuple[str, ...]] = None,
                     end_patterns: Optional[List[str]] = None) -> List[str]:
        """
        Generate text using the trained Markov model.
        
        Args:
            max_length: Maximum number of words to generate
            start_state: Optional starting state (uses random sentence starter if None)
            end_patterns: Optional list of patterns that trigger generation end
            
        Returns:
            List of generated words
        """
        if not self.is_trained:
            self.logger.error("Model must be trained before generating text")
            return []
        
        # Select starting state
        if start_state is None:
            if not self.sentence_starters:
                # Fallback to random state
                start_state = random.choice(list(self.chain.keys()))
            else:
                start_state = random.choice(self.sentence_starters)
        
        generated = list(start_state)
        current_state = start_state
        
        for _ in range(max_length - len(start_state)):
            next_word = self._select_next_word(current_state)
            
            if next_word is None:
                # No valid transitions, try to find continuation
                if not self._find_continuation(generated, current_state):
                    break
                continue
            
            generated.append(next_word)
            
            # Check for end patterns
            if end_patterns and any(pattern in next_word.lower() for pattern in end_patterns):
                break
            
            # Natural sentence ending
            if next_word.endswith(('.', '!', '?')):
                break
            
            # Update state for next iteration
            current_state = tuple(generated[-self.n:])
        
        self.logger.debug(f"Generated {len(generated)} words using Markov chain")
        return generated
    
    def _find_continuation(self, generated: List[str], current_state: Tuple[str, ...]) -> bool:
        """
        Attempt to find a continuation when current state has no transitions.
        
        Args:
            generated: Currently generated words
            current_state: Current state with no transitions
            
        Returns:
            True if continuation was found and added
        """
        # Try shorter n-grams as fallback
        for fallback_n in range(self.n - 1, 0, -1):
            if len(generated) < fallback_n:
                continue
            
            fallback_state = tuple(generated[-fallback_n:])
            next_word = self._select_next_word(fallback_state)
            
            if next_word:
                generated.append(next_word)
                return True
        
        return False
    
    def generate_multiple_texts(self, count: int, **kwargs) -> List[List[str]]:
        """
        Generate multiple text sequences.
        
        Args:
            count: Number of texts to generate
            **kwargs: Arguments passed to generate_text()
            
        Returns:
            List of generated word lists
        """
        return [self.generate_text(**kwargs) for _ in range(count)]
    
    def get_model_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the trained model.
        
        Returns:
            Dictionary with model statistics
        """
        if not self.is_trained:
            return {"trained": False}
        
        total_transitions = sum(sum(counter.values()) for counter in self.chain.values())
        avg_transitions = total_transitions / len(self.chain) if self.chain else 0
        
        return {
            "trained": True,
            "n_gram_order": self.n,
            "vocabulary_size": len(self.vocabulary),
            "chain_states": len(self.chain),
            "total_transitions": total_transitions,
            "average_transitions_per_state": avg_transitions,
            "sentence_starters": len(self.sentence_starters),
            "temperature": self.temperature
        }
