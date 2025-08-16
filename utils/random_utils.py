"""
Random utility functions for controlled randomness in text generation.
"""

import random
import logging
from typing import List, Optional, Any
import math


class RandomUtils:
    """Utilities for managing randomness with temperature control and seeding."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize random utilities.
        
        Args:
            seed: Random seed for reproducible results
        """
        self.logger = logging.getLogger(__name__)
        self._original_seed = seed
        
        if seed is not None:
            self.set_seed(seed)
        
    def set_seed(self, seed: int) -> None:
        """
        Set random seed for reproducible generation.
        
        Args:
            seed: Random seed value
        """
        random.seed(seed)
        self._original_seed = seed
        self.logger.debug(f"Random seed set to {seed}")
    
    def get_seed(self) -> Optional[int]:
        """Get the current random seed."""
        return self._original_seed
    
    def apply_temperature(self, probabilities: List[float], temperature: float) -> List[float]:
        """
        Apply temperature scaling to probability distribution.
        
        Args:
            probabilities: Original probability values
            temperature: Temperature parameter (0.1-2.0)
                        - Low temp (0.1-0.5): More deterministic
                        - High temp (1.5-2.0): More random
            
        Returns:
            Temperature-adjusted probabilities
        """
        if not probabilities:
            return []
        
        temperature = max(0.01, temperature)  # Prevent division by zero
        
        # Apply temperature scaling
        if temperature == 1.0:
            adjusted = probabilities
        else:
            # Use log-softmax for numerical stability
            max_prob = max(probabilities)
            log_probs = [math.log(p + 1e-10) - math.log(max_prob + 1e-10) for p in probabilities]
            scaled_log_probs = [lp / temperature for lp in log_probs]
            
            # Convert back to probabilities
            max_scaled = max(scaled_log_probs)
            exp_probs = [math.exp(lp - max_scaled) for lp in scaled_log_probs]
            
            # Normalize
            total = sum(exp_probs)
            adjusted = [p / total for p in exp_probs]
        
        self.logger.debug(f"Applied temperature {temperature} to {len(probabilities)} probabilities")
        return adjusted
    
    def weighted_choice(self, items: List[Any], weights: List[float]) -> Any:
        """
        Make weighted random choice from items.
        
        Args:
            items: List of items to choose from
            weights: Corresponding weights for each item
            
        Returns:
            Randomly selected item based on weights
        """
        if not items or not weights or len(items) != len(weights):
            return random.choice(items) if items else None
        
        total_weight = sum(weights)
        if total_weight <= 0:
            return random.choice(items)
        
        # Normalize weights
        normalized_weights = [w / total_weight for w in weights]
        
        # Use random.choices for weighted selection
        return random.choices(items, weights=normalized_weights, k=1)[0]
    
    def sample_with_replacement(self, items: List[Any], k: int, 
                              weights: Optional[List[float]] = None) -> List[Any]:
        """
        Sample k items with replacement.
        
        Args:
            items: Items to sample from
            k: Number of items to sample
            weights: Optional weights for sampling
            
        Returns:
            List of sampled items
        """
        if not items:
            return []
        
        if weights:
            return random.choices(items, weights=weights, k=k)
        else:
            return random.choices(items, k=k)
    
    def sample_without_replacement(self, items: List[Any], k: int) -> List[Any]:
        """
        Sample k items without replacement.
        
        Args:
            items: Items to sample from
            k: Number of items to sample
            
        Returns:
            List of sampled items (no duplicates)
        """
        if not items:
            return []
        
        k = min(k, len(items))  # Can't sample more than available
        return random.sample(items, k)
    
    def shuffle_preserving_structure(self, items: List[Any], 
                                   preserve_indices: List[int]) -> List[Any]:
        """
        Shuffle items while preserving elements at specific indices.
        
        Args:
            items: Items to shuffle
            preserve_indices: Indices to keep unchanged
            
        Returns:
            Shuffled list with preserved elements
        """
        if not items:
            return []
        
        result = items.copy()
        preserved_elements = {i: items[i] for i in preserve_indices 
                            if 0 <= i < len(items)}
        
        # Get indices that can be shuffled
        shuffleable_indices = [i for i in range(len(items)) 
                             if i not in preserved_elements]
        shuffleable_items = [items[i] for i in shuffleable_indices]
        
        # Shuffle the non-preserved items
        random.shuffle(shuffleable_items)
        
        # Reconstruct the list
        shuffled_iter = iter(shuffleable_items)
        for i in range(len(result)):
            if i not in preserved_elements:
                result[i] = next(shuffled_iter)
        
        return result
    
    def random_float_range(self, min_val: float, max_val: float) -> float:
        """
        Generate random float in range with proper bounds checking.
        
        Args:
            min_val: Minimum value (inclusive)
            max_val: Maximum value (inclusive)
            
        Returns:
            Random float in specified range
        """
        if min_val >= max_val:
            return min_val
        
        return random.uniform(min_val, max_val)
    
    def random_int_range(self, min_val: int, max_val: int) -> int:
        """
        Generate random integer in range.
        
        Args:
            min_val: Minimum value (inclusive)
            max_val: Maximum value (inclusive)
            
        Returns:
            Random integer in specified range
        """
        if min_val >= max_val:
            return min_val
        
        return random.randint(min_val, max_val)
    
    def probability_gate(self, probability: float) -> bool:
        """
        Return True with given probability.
        
        Args:
            probability: Probability of returning True (0.0-1.0)
            
        Returns:
            True with specified probability
        """
        return random.random() < max(0.0, min(1.0, probability))
    
    def exponential_decay_probability(self, iteration: int, 
                                    initial_prob: float = 0.9,
                                    decay_rate: float = 0.1) -> float:
        """
        Calculate probability that decreases exponentially with iteration.
        
        Args:
            iteration: Current iteration number
            initial_prob: Starting probability
            decay_rate: Rate of decay
            
        Returns:
            Decayed probability
        """
        return initial_prob * math.exp(-decay_rate * iteration)
    
    def generate_variation_seed(self, base_text: str) -> int:
        """
        Generate deterministic seed based on text content for variations.
        
        Args:
            base_text: Text to base seed on
            
        Returns:
            Deterministic seed value
        """
        # Simple hash-based seed generation
        seed = 0
        for char in base_text:
            seed = (seed * 31 + ord(char)) % (2**31 - 1)
        
        return seed
    
    def controlled_randomness(self, base_value: Any, variation_factor: float,
                            alternatives: List[Any]) -> Any:
        """
        Apply controlled randomness to choose between base value and alternatives.
        
        Args:
            base_value: Default/preferred value
            variation_factor: How much variation to allow (0.0-1.0)
            alternatives: Alternative values to choose from
            
        Returns:
            Selected value (base or alternative)
        """
        if not alternatives or variation_factor <= 0:
            return base_value
        
        if random.random() < variation_factor:
            return random.choice(alternatives)
        else:
            return base_value
    
    def smart_shuffle(self, items: List[Any], similarity_threshold: float = 0.3) -> List[Any]:
        """
        Shuffle items while avoiding placing similar items adjacent.
        
        Args:
            items: Items to shuffle
            similarity_threshold: Threshold for considering items similar
            
        Returns:
            Shuffled list with reduced adjacency of similar items
        """
        if len(items) <= 2:
            return items.copy()
        
        result = []
        remaining = items.copy()
        random.shuffle(remaining)
        
        result.append(remaining.pop(0))
        
        while remaining:
            # Try to find item that's not too similar to the last added
            best_choice = remaining[0]
            best_index = 0
            
            for i, item in enumerate(remaining):
                # Simple similarity check based on string representation
                if self._calculate_similarity(str(result[-1]), str(item)) < similarity_threshold:
                    best_choice = item
                    best_index = i
                    break
            
            result.append(remaining.pop(best_index))
        
        return result
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate simple similarity between two strings.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score (0.0-1.0)
        """
        if not str1 or not str2:
            return 0.0
        
        # Simple character overlap similarity
        set1 = set(str1.lower())
        set2 = set(str2.lower())
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
