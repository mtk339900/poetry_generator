"""
Template-based text generation with configurable sentence structures.
"""

import re
import json
import logging
import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from ..utils.random_utils import RandomUtils


class TemplateGenerator:
    """Generates text using predefined templates with variable placeholders."""
    
    def __init__(self, language: str = 'en', random_utils: Optional[RandomUtils] = None):
        """
        Initialize template generator.
        
        Args:
            language: Language code for template selection
            random_utils: Random utilities instance
        """
        self.language = language
        self.random_utils = random_utils or RandomUtils()
        self.logger = logging.getLogger(__name__)
        
        # Template and word data
        self.templates: Dict[str, List[str]] = {}
        self.word_sets: Dict[str, List[str]] = {}
        self.metaphor_patterns: List[str] = []
        self.simile_patterns: List[str] = []
        
        # Generation parameters
        self.style_params = {
            'formality': 0.5,      # 0.0 = casual, 1.0 = formal
            'emotion_intensity': 0.5,  # 0.0 = neutral, 1.0 = intense
            'imagery_density': 0.5     # 0.0 = sparse, 1.0 = dense
        }
    
    def load_templates(self, template_file: str) -> None:
        """
        Load templates from JSON configuration file.
        
        Args:
            template_file: Path to template JSON file
        """
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.templates = data.get('templates', {})
            self.word_sets = data.get('word_sets', {})
            self.metaphor_patterns = data.get('metaphor_patterns', [])
            self.simile_patterns = data.get('simile_patterns', [])
            
            self.logger.info(f"Loaded templates from {template_file}: "
                           f"{len(self.templates)} categories, "
                           f"{len(self.word_sets)} word sets")
        
        except FileNotFoundError:
            self.logger.error(f"Template file not found: {template_file}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in template file: {e}")
            raise
    
    def set_style_parameters(self, formality: float = None, 
                           emotion_intensity: float = None,
                           imagery_density: float = None) -> None:
        """
        Set style parameters for generation.
        
        Args:
            formality: Formality level (0.0-1.0)
            emotion_intensity: Emotional intensity (0.0-1.0)
            imagery_density: Imagery density (0.0-1.0)
        """
        if formality is not None:
            self.style_params['formality'] = max(0.0, min(1.0, formality))
        if emotion_intensity is not None:
            self.style_params['emotion_intensity'] = max(0.0, min(1.0, emotion_intensity))
        if imagery_density is not None:
            self.style_params['imagery_density'] = max(0.0, min(1.0, imagery_density))
        
        self.logger.debug(f"Style parameters updated: {self.style_params}")
    
    def _select_template(self, category: str = None) -> str:
        """
        Select a template from available templates.
        
        Args:
            category: Optional category to select from
            
        Returns:
            Selected template string
        """
        if category and category in self.templates:
            return random.choice(self.templates[category])
        
        # Select from all templates based on style parameters
        suitable_templates = []
        
        for cat, template_list in self.templates.items():
            # Filter templates based on style
            if self._template_matches_style(cat):
                suitable_templates.extend(template_list)
        
        if not suitable_templates:
            # Fallback to any available template
            all_templates = []
            for template_list in self.templates.values():
                all_templates.extend(template_list)
            suitable_templates = all_templates
        
        return random.choice(suitable_templates) if suitable_templates else ""
    
    def _template_matches_style(self, category: str) -> bool:
        """
        Check if template category matches current style parameters.
        
        Args:
            category: Template category name
            
        Returns:
            True if category matches style
        """
        formality = self.style_params['formality']
        emotion = self.style_params['emotion_intensity']
        
        # Simple style matching heuristics
        formal_categories = {'academic', 'classical', 'philosophical', 'formal'}
        casual_categories = {'conversational', 'informal', 'modern', 'simple'}
        emotional_categories = {'passionate', 'melancholic', 'joyful', 'dramatic'}
        neutral_categories = {'descriptive', 'narrative', 'observational'}
        
        category_lower = category.lower()
        
        # Formality check
        if formality > 0.7 and any(cat in category_lower for cat in formal_categories):
            return True
        if formality < 0.3 and any(cat in category_lower for cat in casual_categories):
            return True
        
        # Emotion check
        if emotion > 0.6 and any(cat in category_lower for cat in emotional_categories):
            return True
        if emotion < 0.4 and any(cat in category_lower for cat in neutral_categories):
            return True
        
        # Default to accepting template
        return True
    
    def _fill_template(self, template: str, seed_words: Optional[List[str]] = None) -> str:
        """
        Fill template placeholders with appropriate words.
        
        Args:
            template: Template string with placeholders
            seed_words: Optional seed words to influence word choice
            
        Returns:
            Template with placeholders filled
        """
        # Extract all placeholders
        placeholders = re.findall(r'\{(\w+)\}', template)
        filled_template = template
        
        for placeholder in placeholders:
            word = self._get_word_for_placeholder(placeholder, seed_words)
            filled_template = filled_template.replace(f'{{{placeholder}}}', word, 1)
        
        return filled_template
    
    def _get_word_for_placeholder(self, placeholder: str, seed_words: Optional[List[str]] = None) -> str:
        """
        Get appropriate word for template placeholder.
        
        Args:
            placeholder: Placeholder name
            seed_words: Optional seed words to influence selection
            
        Returns:
            Selected word for placeholder
        """
        # Check if placeholder matches a word set
        if placeholder in self.word_sets:
            candidates = self.word_sets[placeholder]
        else:
            # Try to find similar word set
            candidates = self._find_similar_word_set(placeholder)
        
        if not candidates:
            return f"[{placeholder}]"  # Fallback placeholder
        
        # Apply seed word influence if available
        if seed_words:
            candidates = self._apply_seed_influence(candidates, seed_words)
        
        # Select word based on style parameters
        return self._select_stylistically_appropriate_word(candidates, placeholder)
    
    def _find_similar_word_set(self, placeholder: str) -> List[str]:
        """
        Find word set similar to placeholder name.
        
        Args:
            placeholder: Placeholder name
            
        Returns:
            List of candidate words
        """
        placeholder_lower = placeholder.lower()
        
        # Map common placeholder patterns to word sets
        placeholder_mappings = {
            'noun': ['nouns', 'objects', 'things'],
            'verb': ['verbs', 'actions'],
            'adj': ['adjectives', 'descriptors'],
            'adv': ['adverbs'],
            'emotion': ['emotions', 'feelings'],
            'color': ['colors', 'hues'],
            'nature': ['natural_objects', 'landscape'],
            'person': ['characters', 'people'],
            'place': ['locations', 'places']
        }
        
        for pattern, word_sets in placeholder_mappings.items():
            if pattern in placeholder_lower:
                for word_set in word_sets:
                    if word_set in self.word_sets:
                        return self.word_sets[word_set]
        
        return []
    
    def _apply_seed_influence(self, candidates: List[str], seed_words: List[str]) -> List[str]:
        """
        Modify candidate list based on seed words.
        
        Args:
            candidates: Original candidate words
            seed_words: Seed words for thematic influence
            
        Returns:
            Modified candidate list
        """
        # Simple semantic similarity based on shared letters/sounds
        influenced_candidates = []
        
        for candidate in candidates:
            similarity_score = 0
            for seed in seed_words:
                # Basic similarity metrics
                shared_letters = set(candidate.lower()) & set(seed.lower())
                similarity_score += len(shared_letters) / max(len(candidate), len(seed))
            
            # Boost candidates with higher similarity
            if similarity_score > 0.3:
                influenced_candidates.extend([candidate] * 3)  # Triple weight
            elif similarity_score > 0.1:
                influenced_candidates.extend([candidate] * 2)  # Double weight
            else:
                influenced_candidates.append(candidate)
        
        return influenced_candidates
    
    def _select_stylistically_appropriate_word(self, candidates: List[str], placeholder_type: str) -> str:
        """
        Select word that matches current style parameters.
        
        Args:
            candidates: List of candidate words
            placeholder_type: Type of placeholder being filled
            
        Returns:
            Selected word
        """
        # Basic style-based filtering
        formality = self.style_params['formality']
        emotion = self.style_params['emotion_intensity']
        
        # Length-based formality heuristic
        if formality > 0.6:
            # Prefer longer, more formal words
            candidates = sorted(candidates, key=len, reverse=True)
            candidates = candidates[:len(candidates)//2 + 1]
        elif formality < 0.4:
            # Prefer shorter, simpler words
            candidates = sorted(candidates, key=len)
            candidates = candidates[:len(candidates)//2 + 1]
        
        # Emotional intensity could influence word choice
        # For now, use random selection from filtered candidates
        return random.choice(candidates)
    
    def generate_from_template(self, category: str = None, 
                             seed_words: Optional[List[str]] = None,
                             custom_template: str = None) -> str:
        """
        Generate text from a template.
        
        Args:
            category: Template category to use
            seed_words: Seed words for thematic guidance
            custom_template: Custom template string to use
            
        Returns:
            Generated text
        """
        if custom_template:
            template = custom_template
        else:
            template = self._select_template(category)
        
        if not template:
            self.logger.warning("No suitable template found")
            return ""
        
        generated_text = self._fill_template(template, seed_words)
        
        # Apply figurative language enhancement
        if random.random() < self.style_params['imagery_density']:
            generated_text = self._enhance_with_figurative_language(generated_text)
        
        self.logger.debug(f"Generated from template: {generated_text}")
        return generated_text
    
    def _enhance_with_figurative_language(self, text: str) -> str:
        """
        Add figurative language elements to generated text.
        
        Args:
            text: Original generated text
            
        Returns:
            Enhanced text with figurative elements
        """
        # Simple probability-based enhancement
        if random.random() < 0.3 and self.simile_patterns:
            # Add simile
            simile = random.choice(self.simile_patterns)
            if '{object}' in simile and 'objects' in self.word_sets:
                obj = random.choice(self.word_sets['objects'])
                simile = simile.replace('{object}', obj)
            text += f" {simile}"
        
        if random.random() < 0.2 and self.metaphor_patterns:
            # Add metaphor
            metaphor = random.choice(self.metaphor_patterns)
            if '{concept}' in metaphor and 'concepts' in self.word_sets:
                concept = random.choice(self.word_sets['concepts'])
                metaphor = metaphor.replace('{concept}', concept)
            text = f"{metaphor} {text}"
        
        return text
    
    def generate_multiple(self, count: int, **kwargs) -> List[str]:
        """
        Generate multiple texts from templates.
        
        Args:
            count: Number of texts to generate
            **kwargs: Arguments passed to generate_from_template()
            
        Returns:
            List of generated texts
        """
        return [self.generate_from_template(**kwargs) for _ in range(count)]
