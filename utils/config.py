"""
Configuration management for the poetry generator.
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path


class ConfigManager:
    """Manages configuration loading, validation, and access."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file (JSON or YAML)
        """
        self.logger = logging.getLogger(__name__)
        self.config: Dict[str, Any] = {}
        self.config_file = config_file
        
        # Default configuration
        self._load_defaults()
        
        if config_file:
            self.load_config(config_file)
    
    def _load_defaults(self) -> None:
        """Load default configuration values."""
        self.config = {
            "generation": {
                "mode": "hybrid",
                "max_length": 50,
                "temperature": 1.0,
                "blend_ratio": 0.5,
                "language": "en"
            },
            "corpus": {
                "min_words": 100,
                "remove_stopwords": True,
                "min_word_freq": 1,
                "max_vocabulary": None,
                "encoding": "utf-8"
            },
            "markov": {
                "n_gram_order": 2,
                "temperature": 1.0,
                "sentence_boundaries": True
            },
            "template": {
                "formality": 0.5,
                "emotion_intensity": 0.5,
                "imagery_density": 0.5
            },
            "style": {
                "poem_form": "free_verse",
                "rhyme_scheme": None,
                "syllable_pattern": None,
                "stanza_count": 1,
                "lines_per_stanza": 4
            },
            "figurative": {
                "metaphor_probability": 0.3,
                "simile_probability": 0.3,
                "personification_probability": 0.2,
                "alliteration_probability": 0.1,
                "sensory_enhancement": True
            },
            "output": {
                "format": "text",
                "include_title": True,
                "title_generation": "keyword_based",
                "line_endings": "auto",
                "export_path": None
            },
            "logging": {
                "level": "INFO",
                "file": None,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "random": {
                "seed": None,
                "reproducible": False
            },
            "files": {
                "corpus_dir": "corpora",
                "template_dir": "config/templates",
                "dictionary_dir": "config/dictionaries",
                "output_dir": "output"
            }
        }
    
    def load_config(self, config_file: str) -> None:
        """
        Load configuration from file.
        
        Args:
            config_file: Path to configuration file
        """
        config_path = Path(config_file)
        
        if not config_path.exists():
            self.logger.warning(f"Configuration file not found: {config_file}")
            return
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    file_config = yaml.safe_load(f)
                else:
                    file_config = json.load(f)
            
            # Merge with defaults
            self._merge_config(self.config, file_config)
            self.config_file = config_file
            
            self.logger.info(f"Loaded configuration from {config_file}")
            
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            self.logger.error(f"Invalid configuration file format: {e}")
            raise
        except IOError as e:
            self.logger.error(f"Error reading configuration file: {e}")
            raise
    
    def _merge_config(self, base_config: Dict[str, Any], new_config: Dict[str, Any]) -> None:
        """
        Recursively merge new configuration into base configuration.
        
        Args:
            base_config: Base configuration dictionary
            new_config: New configuration to merge
        """
        for key, value in new_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (supports dot notation like 'generation.mode')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the final key
        config[keys[-1]] = value
        
        self.logger.debug(f"Set configuration: {key} = {value}")
    
    def validate_config(self) -> List[str]:
        """
        Validate configuration values and return list of issues.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate generation settings
        mode = self.get('generation.mode')
        if mode not in ['template', 'markov', 'hybrid']:
            errors.append(f"Invalid generation mode: {mode}")
        
        temperature = self.get('generation.temperature')
        if not isinstance(temperature, (int, float)) or temperature <= 0:
            errors.append(f"Invalid temperature: {temperature}")
        
        max_length = self.get('generation.max_length')
        if not isinstance(max_length, int) or max_length <= 0:
            errors.append(f"Invalid max_length: {max_length}")
        
        # Validate Markov settings
        n_gram = self.get('markov.n_gram_order')
        if not isinstance(n_gram, int) or n_gram < 1 or n_gram > 5:
            errors.append(f"Invalid n_gram_order: {n_gram}")
        
        # Validate style settings
        formality = self.get('template.formality')
        if not isinstance(formality, (int, float)) or not 0 <= formality <= 1:
            errors.append(f"Invalid formality: {formality}")
        
        emotion = self.get('template.emotion_intensity')
        if not isinstance(emotion, (int, float)) or not 0 <= emotion <= 1:
            errors.append(f"Invalid emotion_intensity: {emotion}")
        
        imagery = self.get('template.imagery_density')
        if not isinstance(imagery, (int, float)) or not 0 <= imagery <= 1:
            errors.append(f"Invalid imagery_density: {imagery}")
        
        # Validate file paths
        corpus_dir = self.get('files.corpus_dir')
        if corpus_dir and not Path(corpus_dir).exists():
            errors.append(f"Corpus directory does not exist: {corpus_dir}")
        
        template_dir = self.get('files.template_dir')
        if template_dir and not Path(template_dir).exists():
            errors.append(f"Template directory does not exist: {template_dir}")
        
        return errors
    
    def save_config(self, output_file: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_file: Output file path (uses original file if None)
        """
        target_file = output_file or self.config_file
        
        if not target_file:
            raise ValueError("No output file specified and no original config file")
        
        config_path = Path(target_file)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                else:
                    json.dump(self.config, f, indent=2)
            
            self.logger.info(f"Saved configuration to {target_file}")
            
        except IOError as e:
            self.logger.error(f"Error saving configuration file: {e}")
            raise
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Configuration section dictionary
        """
        return self.config.get(section, {})
    
    def update_section(self, section: str, updates: Dict[str, Any]) -> None:
        """
        Update entire configuration section.
        
        Args:
            section: Section name
            updates: Dictionary of updates
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section].update(updates)
        self.logger.debug(f"Updated configuration section: {section}")
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self._load_defaults()
        self.logger.info("Reset configuration to defaults")
    
    def get_file_paths(self) -> Dict[str, str]:
        """
        Get resolved file paths based on configuration.
        
        Returns:
            Dictionary of resolved file paths
        """
        base_paths = self.get_section('files')
        resolved_paths = {}
        
        for key, path in base_paths.items():
            if path:
                resolved_path = Path(path).resolve()
                resolved_paths[key] = str(resolved_path)
            else:
                resolved_paths[key] = None
        
        return resolved_paths
    
    def setup_logging(self) -> None:
        """Setup logging based on configuration."""
        log_config = self.get_section('logging')
        
        level = getattr(logging, log_config.get('level', 'INFO').upper())
        format_str = log_config.get('format', 
                                  '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(
            level=level,
            format=format_str,
            filename=log_config.get('file'),
            filemode='a' if log_config.get('file') else None
        )
        
        self.logger.info("Logging configured from settings")
    
    def create_default_config_file(self, output_path: str) -> None:
        """
        Create a default configuration file.
        
        Args:
            output_path: Path where to create the default config
        """
        self.reset_to_defaults()
        self.save_config(output_path)
        self.logger.info(f"Created default configuration file: {output_path}")
    
    def __str__(self) -> str:
        """String representation of current configuration."""
        return json.dumps(self.config, indent=2)
