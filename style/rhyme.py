"""
Rhyme detection and generation using phonetic approximation.
"""

import re
import logging
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict


class RhymeEngine:
    """Handles rhyme detection and generation using simple phonetic rules."""
    
    def __init__(self, language: str = 'en'):
        """
        Initialize rhyme engine.
        
        Args:
            language: Language code for language-specific rules
        """
        self.language = language
        self.logger = logging.getLogger(__name__)
        
        # Phonetic mapping rules by language
        self._vowel_sounds = {
            'en': {
                'a': ['a', 'ay', 'ai'],
                'e': ['e', 'ee', 'ea', 'ey'],
                'i': ['i', 'ie', 'y', 'igh'],
                'o': ['o', 'ow', 'oa', 'oe'],
                'u': ['u', 'oo', 'ou', 'ue']
            },
            'es': {
                'a': ['a'],
                'e': ['e'],
                'i': ['i'],
                'o': ['o'],
                'u': ['u']
            }
        }
        
        self._consonant_clusters = {
            'en': {
                'ch': 'ch', 'sh': 'sh', 'th': 'th', 'ph': 'f',
                'ck': 'k', 'ng': 'ng', 'qu': 'kw'
            },
            'es': {
                'ch': 'ch', 'rr': 'rr', 'll': 'll', 'ñ': 'ny'
            }
        }
        
        # Rhyme dictionaries
        self.rhyme_dict: Dict[str, Set[str]] = defaultdict(set)
        self.ending_sounds: Dict[str, List[str]] = defaultdict(list)
        
    def build_rhyme_dictionary(self, words: List[str]) -> None:
        """
        Build rhyme dictionary from word list.
        
        Args:
            words: List of words to analyze for rhymes
        """
        for word in words:
            if len(word) < 2:
                continue
                
            ending_sound = self._extract_ending_sound(word.lower())
            if ending_sound:
                self.rhyme_dict[ending_sound].add(word.lower())
                self.ending_sounds[word.lower()].append(ending_sound)
        
        self.logger.info(f"Built rhyme dictionary with {len(self.rhyme_dict)} sound groups")
    
    def _extract_ending_sound(self, word: str) -> str:
        """
        Extract phonetic ending sound from word.
        
        Args:
            word: Word to analyze
            
        Returns:
            Phonetic ending sound representation
        """
        if len(word) < 2:
            return word
        
        # Remove common suffixes that don't affect rhyme
        suffixes_to_strip = ['s', 'es', 'ed', 'ing', 'ly', 'er', 'est']
        original_word = word
        
        for suffix in sorted(suffixes_to_strip, key=len, reverse=True):
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                word = word[:-len(suffix)]
                break
        
        # Extract last 2-3 characters as ending sound approximation
        if len(word) >= 3:
            ending = word[-3:]
        elif len(word) >= 2:
            ending = word[-2:]
        else:
            ending = word
        
        # Apply phonetic transformations
        ending = self._apply_phonetic_rules(ending)
        
        return ending
    
    def _apply_phonetic_rules(self, ending: str) -> str:
        """
        Apply language-specific phonetic transformation rules.
        
        Args:
            ending: Raw ending string
            
        Returns:
            Phonetically transformed ending
        """
        # Handle consonant clusters
        clusters = self._consonant_clusters.get(self.language, {})
        for cluster, sound in clusters.items():
            ending = ending.replace(cluster, sound)
        
        # Normalize vowel sounds
        vowels = self._vowel_sounds.get(self.language, {})
        for vowel, variations in vowels.items():
            for variation in variations:
                if variation in ending and variation != vowel:
                    ending = ending.replace(variation, vowel)
        
        # Language-specific rules
        if self.language == 'en':
            # Silent 'e' rule
            if ending.endswith('e') and len(ending) > 1:
                ending = ending[:-1]
            
            # 'y' to 'i' sound
            ending = ending.replace('y', 'i')
            
        elif self.language == 'es':
            # Spanish phonetic consistency (less transformation needed)
            pass
        
        return ending
    
    def find_rhymes(self, word: str, max_rhymes: int = 10) -> List[str]:
        """
        Find words that rhyme with the given word.
        
        Args:
            word: Word to find rhymes for
            max_rhymes: Maximum number of rhymes to return
            
        Returns:
            List of rhyming words
        """
        word_lower = word.lower()
        ending_sound = self._extract_ending_sound(word_lower)
        
        if ending_sound not in self.rhyme_dict:
            return []
        
        rhymes = list(self.rhyme_dict[ending_sound])
        
        # Remove the original word
        rhymes = [w for w in rhymes if w != word_lower]
        
        # Sort by similarity to original word
        rhymes.sort(key=lambda w: self._rhyme_quality_score(word_lower, w), reverse=True)
        
        return rhymes[:max_rhymes]
    
    def _rhyme_quality_score(self, word1: str, word2: str) -> float:
        """
        Score the quality of rhyme between two words.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Rhyme quality score (higher is better)
        """
        if not word1 or not word2:
            return 0.0
        
        score = 0.0
        
        # Length similarity bonus
        len_diff = abs(len(word1) - len(word2))
        score += max(0, 1.0 - len_diff * 0.1)
        
        # Ending similarity (more characters matching = better)
        min_len = min(len(word1), len(word2))
        matching_chars = 0
        
        for i in range(1, min_len + 1):
            if word1[-i] == word2[-i]:
                matching_chars += 1
            else:
                break
        
        score += matching_chars / min_len if min_len > 0 else 0
        
        # Penalize very different word beginnings
        if word1[0] != word2[0]:
            score *= 0.9
        
        return score
    
    def check_rhyme(self, word1: str, word2: str) -> bool:
        """
        Check if two words rhyme.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            True if words rhyme
        """
        ending1 = self._extract_ending_sound(word1.lower())
        ending2 = self._extract_ending_sound(word2.lower())
        
        return ending1 == ending2 and ending1 != ""
    
    def get_rhyme_scheme_words(self, rhyme_scheme: str, vocabulary: List[str]) -> Dict[str, List[str]]:
        """
        Get words for a specific rhyme scheme pattern.
        
        Args:
            rhyme_scheme: Rhyme scheme pattern (e.g., 'ABAB', 'AABA')
            vocabulary: Available vocabulary
            
        Returns:
            Dictionary mapping rhyme letters to word lists
        """
        if not vocabulary:
            return {}
        
        # Build rhyme dictionary if not already done
        if not self.rhyme_dict:
            self.build_rhyme_dictionary(vocabulary)
        
        unique_rhymes = set(rhyme_scheme)
        scheme_words = {}
        
        used_endings = set()
        available_endings = list(self.rhyme_dict.keys())
        
        for rhyme_letter in unique_rhymes:
            # Find an unused ending sound with enough words
            suitable_ending = None
            for ending in available_endings:
                if ending not in used_endings and len(self.rhyme_dict[ending]) >= 2:
                    suitable_ending = ending
                    break
            
            if suitable_ending:
                scheme_words[rhyme_letter] = list(self.rhyme_dict[suitable_ending])
                used_endings.add(suitable_ending)
            else:
                # Fallback to any available words
                scheme_words[rhyme_letter] = vocabulary[:10]
        
        self.logger.debug(f"Generated rhyme scheme words for pattern '{rhyme_scheme}'")
        return scheme_words
    
    def create_rhyming_line(self, base_word: str, target_pattern: str = None) -> str:
        """
        Create a line ending with a word that rhymes with base_word.
        
        Args:
            base_word: Word to rhyme with
            target_pattern: Optional pattern for the line structure
            
        Returns:
            Generated rhyming line
        """
        rhymes = self.find_rhymes(base_word, max_rhymes=5)
        if not rhymes:
            return f"words that rhyme with {base_word}"
        
        chosen_rhyme = rhymes[0]
        
        if target_pattern:
            # Simple pattern filling (placeholder implementation)
            line = target_pattern.replace("{rhyme}", chosen_rhyme)
        else:
            # Simple line construction
            articles = ["the", "a", "an"]
            adjectives = ["bright", "dark", "gentle", "fierce", "quiet"]
            
            import random
            article = random.choice(articles)
            adjective = random.choice(adjectives)
            
            line = f"{article} {adjective} {chosen_rhyme}"
        
        return line
    
    def validate_rhyme_scheme(self, lines: List[str], expected_scheme: str) -> bool:
        """
        Validate that lines follow the expected rhyme scheme.
        
        Args:
            lines: List of text lines
            expected_scheme: Expected rhyme scheme pattern
            
        Returns:
            True if lines match the scheme
        """
        if len(lines) != len(expected_scheme):
            return False
        
        # Extract last words from each line
        last_words = []
        for line in lines:
            words = line.strip().split()
            if words:
                # Remove punctuation from last word
                last_word = re.sub(r'[^\w]', '', words[-1]).lower()
                last_words.append(last_word)
            else:
                return False
        
        # Group words by rhyme scheme letter
        rhyme_groups = {}
        for i, letter in enumerate(expected_scheme):
            if letter not in rhyme_groups:
                rhyme_groups[letter] = []
            rhyme_groups[letter].append(last_words[i])
        
        # Check that words in each group rhyme with each other
        for letter, words in rhyme_groups.items():
            if len(words) > 1:
                first_word = words[0]
                for word in words[1:]:
                    if not self.check_rhyme(first_word, word):
                        return False
        
        return True
    
    def suggest_rhyme_improvements(self, lines: List[str], target_scheme: str) -> List[Tuple[int, str]]:
        """
        Suggest improvements to make lines fit rhyme scheme better.
        
        Args:
            lines: Current lines
            target_scheme: Target rhyme scheme
            
        Returns:
            List of (line_index, suggested_replacement) tuples
        """
        suggestions = []
        
        if len(lines) != len(target_scheme):
            return suggestions
        
        # Analyze current rhyme patterns
        last_words = []
        for line in lines:
            words = line.strip().split()
            if words:
                last_word = re.sub(r'[^\w]', '', words[-1]).lower()
                last_words.append(last_word)
            else:
                last_words.append("")
        
        # Find lines that break the rhyme scheme
        rhyme_groups = defaultdict(list)
        for i, letter in enumerate(target_scheme):
            rhyme_groups[letter].append((i, last_words[i]))
        
        for letter, word_pairs in rhyme_groups.items():
            if len(word_pairs) > 1:
                # Check if all words in group rhyme
                base_word = word_pairs[0][1]
                for line_idx, word in word_pairs[1:]:
                    if not self.check_rhyme(base_word, word):
                        # Suggest rhyming replacement
                        rhymes = self.find_rhymes(base_word, max_rhymes=3)
                        if rhymes:
                            new_word = rhymes[0]
                            # Reconstruct line with new ending word
                            original_line = lines[line_idx]
                            line_words = original_line.strip().split()
                            if line_words:
                                line_words[-1] = new_word
                                suggested_line = ' '.join(line_words)
                                suggestions.append((line_idx, suggested_line))
        
        return suggestions
