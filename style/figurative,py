"""
Figurative language generation including metaphors, similes, personification, and alliteration.
"""

import re
import json
import logging
import random
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict


class FigurativeLanguage:
    """Handles creation and enhancement of figurative language devices."""
    
    def __init__(self, language: str = 'en'):
        """
        Initialize figurative language generator.
        
        Args:
            language: Language code for language-specific processing
        """
        self.language = language
        self.logger = logging.getLogger(__name__)
        
        # Figurative language databases
        self.metaphor_patterns: List[str] = []
        self.simile_patterns: List[str] = []
        self.personification_patterns: List[str] = []
        
        # Word categorization for figurative devices
        self.abstract_concepts: Set[str] = set()
        self.concrete_objects: Set[str] = set()
        self.human_actions: Set[str] = set()
        self.natural_elements: Set[str] = set()
        self.sensory_words: Dict[str, Set[str]] = defaultdict(set)
        
        # Alliteration support
        self.consonant_groups: Dict[str, List[str]] = defaultdict(list)
        
        self._initialize_default_patterns()
    
    def _initialize_default_patterns(self) -> None:
        """Initialize default figurative language patterns."""
        
        # Default metaphor patterns
        self.metaphor_patterns = [
            "{subject} is {metaphor_object}",
            "{subject} becomes {metaphor_object}",
            "The {metaphor_object} of {subject}",
            "{subject} transforms into {metaphor_object}",
            "{subject} - a {metaphor_object} in disguise"
        ]
        
        # Default simile patterns
        self.simile_patterns = [
            "{subject} like {comparison_object}",
            "as {adjective} as {comparison_object}",
            "{subject} moves like {comparison_object}",
            "{verb} like {comparison_object}",
            "resembling {comparison_object}"
        ]
        
        # Default personification patterns
        self.personification_patterns = [
            "{object} {human_action}",
            "{object} {human_emotion}",
            "The {object} whispers {message}",
            "{object} dances {manner}",
            "{object} watches {target}"
        ]
        
        # Default word sets
        self.abstract_concepts.update([
            'love', 'fear', 'hope', 'memory', 'time', 'freedom', 'peace', 'chaos',
            'beauty', 'truth', 'wisdom', 'courage', 'sorrow', 'joy', 'anger'
        ])
        
        self.concrete_objects.update([
            'ocean', 'mountain', 'tree', 'stone', 'fire', 'river', 'cloud', 'star',
            'flower', 'bird', 'lion', 'wolf', 'butterfly', 'storm', 'sunrise'
        ])
        
        self.human_actions.update([
            'whispers', 'dances', 'sleeps', 'weeps', 'laughs', 'sings', 'dreams',
            'watches', 'embraces', 'mourns', 'celebrates', 'remembers', 'forgets'
        ])
        
        self.sensory_words['visual'].update([
            'glowing', 'shimmering', 'sparkling', 'gleaming', 'radiant', 'dim',
            'bright', 'vivid', 'pale', 'colorful', 'transparent', 'opaque'
        ])
        
        self.sensory_words['auditory'].update([
            'echoing', 'whispering', 'thundering', 'melodic', 'harsh', 'soft',
            'rhythmic', 'silent', 'resonant', 'muted', 'sharp', 'gentle'
        ])
        
        self.sensory_words['tactile'].update([
            'smooth', 'rough', 'warm', 'cold', 'soft', 'hard', 'silky', 'coarse',
            'tender', 'sharp', 'gentle', 'firm', 'delicate', 'solid'
        ])
    
    def load_figurative_dictionary(self, dictionary_file: str) -> None:
        """
        Load figurative language patterns and words from JSON file.
        
        Args:
            dictionary_file: Path to JSON dictionary file
        """
        try:
            with open(dictionary_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load patterns
            self.metaphor_patterns = data.get('metaphor_patterns', self.metaphor_patterns)
            self.simile_patterns = data.get('simile_patterns', self.simile_patterns)
            self.personification_patterns = data.get('personification_patterns', self.personification_patterns)
            
            # Load word categories
            if 'abstract_concepts' in data:
                self.abstract_concepts.update(data['abstract_concepts'])
            if 'concrete_objects' in data:
                self.concrete_objects.update(data['concrete_objects'])
            if 'human_actions' in data:
                self.human_actions.update(data['human_actions'])
            if 'sensory_words' in data:
                for sense, words in data['sensory_words'].items():
                    self.sensory_words[sense].update(words)
            
            self.logger.info(f"Loaded figurative dictionary from {dictionary_file}")
            
        except FileNotFoundError:
            self.logger.warning(f"Figurative dictionary not found: {dictionary_file}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in figurative dictionary: {e}")
    
    def create_metaphor(self, subject: str = None, target_concept: str = None) -> str:
        """
        Create a metaphor connecting two concepts.
        
        Args:
            subject: Subject of the metaphor
            target_concept: Optional target concept to metaphorize
            
        Returns:
            Generated metaphor string
        """
        if not subject:
            subject = random.choice(list(self.abstract_concepts))
        
        if not target_concept:
            target_concept = random.choice(list(self.concrete_objects))
        
        pattern = random.choice(self.metaphor_patterns)
        
        # Fill in the pattern
        metaphor = pattern.format(
            subject=subject,
            metaphor_object=target_concept,
            adjective=self._get_descriptive_word(target_concept)
        )
        
        self.logger.debug(f"Created metaphor: {metaphor}")
        return metaphor
    
    def create_simile(self, subject: str = None, comparison_basis: str = None) -> str:
        """
        Create a simile comparing two things.
        
        Args:
            subject: Subject being compared
            comparison_basis: Basis for comparison (quality or action)
            
        Returns:
            Generated simile string
        """
        if not subject:
            subject = random.choice(list(self.concrete_objects | self.abstract_concepts))
        
        comparison_object = random.choice(list(self.concrete_objects))
        
        # Ensure different objects for comparison
        attempts = 0
        while comparison_object == subject and attempts < 5:
            comparison_object = random.choice(list(self.concrete_objects))
            attempts += 1
        
        pattern = random.choice(self.simile_patterns)
        
        # Fill in the pattern
        simile = pattern.format(
            subject=subject,
            comparison_object=comparison_object,
            adjective=self._get_descriptive_word(comparison_object),
            verb=random.choice(['moves', 'appears', 'sounds', 'feels'])
        )
        
        self.logger.debug(f"Created simile: {simile}")
        return simile
    
    def create_personification(self, object_word: str = None) -> str:
        """
        Create personification by giving human qualities to non-human things.
        
        Args:
            object_word: Object to personify
            
        Returns:
            Generated personification string
        """
        if not object_word:
            object_word = random.choice(list(self.concrete_objects | self.natural_elements))
        
        human_action = random.choice(list(self.human_actions))
        pattern = random.choice(self.personification_patterns)
        
        # Fill in the pattern
        personification = pattern.format(
            object=object_word,
            human_action=human_action,
            human_emotion=random.choice(['sighs', 'smiles', 'frowns', 'rejoices']),
            message=random.choice(['secrets', 'stories', 'songs', 'promises']),
            manner=random.choice(['gracefully', 'wildly', 'gently', 'proudly']),
            target=random.choice(['the world', 'the sky', 'passersby', 'the horizon'])
        )
        
        self.logger.debug(f"Created personification: {personification}")
        return personification
    
    def _get_descriptive_word(self, target: str) -> str:
        """
        Get appropriate descriptive word for target object.
        
        Args:
            target: Target word to describe
            
        Returns:
            Descriptive adjective
        """
        # Simple heuristic mapping
        if target in ['ocean', 'river', 'water']:
            return random.choice(['flowing', 'deep', 'vast', 'calm'])
        elif target in ['mountain', 'stone', 'rock']:
            return random.choice(['solid', 'towering', 'ancient', 'majestic'])
        elif target in ['fire', 'flame', 'sun']:
            return random.choice(['bright', 'fierce', 'warm', 'blazing'])
        elif target in ['cloud', 'mist', 'fog']:
            return random.choice(['soft', 'ethereal', 'drifting', 'gentle'])
        else:
            # Default to sensory words
            all_descriptors = []
            for sense_words in self.sensory_words.values():
                all_descriptors.extend(list(sense_words))
            return random.choice(all_descriptors) if all_descriptors else 'mysterious'
    
    def enhance_with_alliteration(self, text: str, target_sound: str = None) -> str:
        """
        Enhance text with alliterative elements.
        
        Args:
            text: Original text
            target_sound: Target consonant sound for alliteration
            
        Returns:
            Enhanced text with alliteration
        """
        words = text.split()
        if len(words) < 2:
            return text
        
        if not target_sound:
            # Choose consonant from first word
            first_word = words[0].lower()
            target_sound = first_word[0] if first_word and first_word[0].isalpha() else 'w'
        
        # Find replacement words that start with target sound
        enhanced_words = []
        for i, word in enumerate(words):
            if i == 0 or word.lower()[0] == target_sound:
                enhanced_words.append(word)
            else:
                # Try to find alliterative replacement
                replacement = self._find_alliterative_replacement(word, target_sound)
                enhanced_words.append(replacement or word)
        
        result = ' '.join(enhanced_words)
        self.logger.debug(f"Enhanced with alliteration ({target_sound}): {result}")
        return result
    
    def _find_alliterative_replacement(self, word: str, target_sound: str) -> Optional[str]:
        """
        Find word starting with target sound to replace given word.
        
        Args:
            word: Original word
            target_sound: Target starting sound
            
        Returns:
            Replacement word or None
        """
        # Simple replacement dictionary for common words
        replacements = {
            'the': {'w': 'wild', 's': 'strong', 'b': 'bright', 'g': 'great'},
            'and': {'w': 'with', 's': 'so', 'b': 'but', 'f': 'for'},
            'is': {'w': 'was', 's': 'seems', 'b': 'becomes', 'f': 'feels'},
            'in': {'w': 'within', 's': 'softly', 'b': 'beneath', 'f': 'from'},
            'of': {'w': 'with', 's': 'so', 'b': 'by', 'f': 'for'},
            'on': {'w': 'with', 's': 'softly', 'b': 'by', 'f': 'from'}
        }
        
        word_lower = word.lower()
        if word_lower in replacements and target_sound in replacements[word_lower]:
            return replacements[word_lower][target_sound]
        
        return None
    
    def add_sensory_details(self, text: str, sense_type: str = None) -> str:
        """
        Add sensory details to enhance imagery.
        
        Args:
            text: Original text
            sense_type: Type of sensory detail ('visual', 'auditory', 'tactile', etc.)
            
        Returns:
            Enhanced text with sensory details
        """
        if not sense_type:
            sense_type = random.choice(list(self.sensory_words.keys()))
        
        if sense_type not in self.sensory_words:
            return text
        
        sensory_word = random.choice(list(self.sensory_words[sense_type]))
        
        # Simple insertion strategy
        words = text.split()
        if len(words) >= 2:
            # Insert before a noun (heuristic: longer words are often nouns)
            noun_index = 0
            for i, word in enumerate(words):
                if len(word) > 4 and word.lower() not in {'that', 'with', 'from', 'they', 'were'}:
                    noun_index = i
                    break
            
            words.insert(noun_index, sensory_word)
            result = ' '.join(words)
        else:
            result = f"{sensory_word} {text}"
        
        self.logger.debug(f"Added {sense_type} detail: {result}")
        return result
    
    def create_extended_metaphor(self, base_subject: str, base_metaphor: str, 
                               extension_length: int = 3) -> List[str]:
        """
        Create an extended metaphor across multiple lines.
        
        Args:
            base_subject: Primary subject of the metaphor
            base_metaphor: Primary metaphor object
            extension_length: Number of lines to extend the metaphor
            
        Returns:
            List of metaphor lines
        """
        lines = []
        
        # Start with basic metaphor
        lines.append(self.create_metaphor(base_subject, base_metaphor))
        
        # Extend with related concepts
        related_actions = self._get_related_actions(base_metaphor)
        related_qualities = self._get_related_qualities(base_metaphor)
        
        for i in range(extension_length - 1):
            if i % 2 == 0 and related_actions:
                # Action-based extension
                action = random.choice(related_actions)
                line = f"{base_subject} {action}"
            elif related_qualities:
                # Quality-based extension
                quality = random.choice(related_qualities)
                line = f"With {quality}, {base_subject} continues"
            else:
                # Fallback extension
                line = f"The essence of {base_metaphor} within {base_subject}"
            
            lines.append(line)
        
        self.logger.debug(f"Created extended metaphor with {len(lines)} lines")
        return lines
    
    def _get_related_actions(self, metaphor_object: str) -> List[str]:
        """Get actions related to metaphor object."""
        action_mappings = {
            'ocean': ['flows', 'crashes', 'swells', 'retreats'],
            'mountain': ['stands', 'towers', 'endures', 'watches'],
            'fire': ['burns', 'flickers', 'consumes', 'illuminates'],
            'bird': ['soars', 'sings', 'glides', 'nests'],
            'tree': ['grows', 'sways', 'shelters', 'reaches'],
            'river': ['meanders', 'rushes', 'carves', 'nourishes'],
            'storm': ['rages', 'passes', 'thunders', 'cleanses']
        }
        
        return action_mappings.get(metaphor_object, ['moves', 'exists', 'transforms'])
    
    def _get_related_qualities(self, metaphor_object: str) -> List[str]:
        """Get qualities related to metaphor object."""
        quality_mappings = {
            'ocean': ['depth', 'vastness', 'power', 'mystery'],
            'mountain': ['strength', 'permanence', 'majesty', 'solitude'],
            'fire': ['passion', 'energy', 'destruction', 'renewal'],
            'bird': ['freedom', 'grace', 'lightness', 'song'],
            'tree': ['growth', 'stability', 'life', 'wisdom'],
            'river': ['persistence', 'change', 'life', 'journey'],
            'storm': ['power', 'chaos', 'cleansing', 'intensity']
        }
        
        return quality_mappings.get(metaphor_object, ['essence', 'nature', 'spirit'])
