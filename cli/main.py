"""
Command line interface for the poetry generator.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Import the generator modules
from ..corpus.loader import CorpusLoader
from ..corpus.cleaner import TextCleaner
from ..corpus.tokenizer import Tokenizer
from ..generation.template import TemplateGenerator
from ..generation.markov import MarkovGenerator
from ..generation.hybrid import HybridGenerator
from ..style.rhyme import RhymeEngine
from ..style.figurative import FigurativeLanguage
from ..utils.config import ConfigManager
from ..utils.random_utils import RandomUtils


class PoetryGeneratorCLI:
    """Command line interface for the poetry generator."""
    
    def __init__(self):
        """Initialize the CLI application."""
        self.config_manager = ConfigManager()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.corpus_loader = None
        self.text_cleaner = None
        self.tokenizer = None
        self.template_gen = None
        self.markov_gen = None
        self.hybrid_gen = None
        self.rhyme_engine = None
        self.figurative = None
        self.random_utils = None
        
    def create_argument_parser(self) -> argparse.ArgumentParser:
        """
        Create command line argument parser.
        
        Returns:
            Configured argument parser
        """
        parser = argparse.ArgumentParser(
            description="Poetry Generator - Create original poems using custom algorithms",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s --mode template --category lyrical --seed-words love,nature
  %(prog)s --mode markov --corpus corpus1.txt,corpus2.txt --max-length 30
  %(prog)s --mode hybrid --config my_config.yaml --count 5
  %(prog)s --mode template --rhyme-scheme ABAB --stanzas 2
            """
        )
        
        # Generation mode and basic parameters
        parser.add_argument(
            '--mode', choices=['template', 'markov', 'hybrid'],
            default='hybrid', help='Text generation mode (default: hybrid)'
        )
        
        parser.add_argument(
            '--max-length', type=int, default=50,
            help='Maximum number of words to generate (default: 50)'
        )
        
        parser.add_argument(
            '--temperature', type=float, default=1.0,
            help='Temperature for randomness control (0.1-2.0, default: 1.0)'
        )
        
        parser.add_argument(
            '--count', type=int, default=1,
            help='Number of poems to generate (default: 1)'
        )
        
        # Input sources
        parser.add_argument(
            '--corpus', type=str, nargs='*',
            help='Corpus files for training (comma-separated or multiple args)'
        )
        
        parser.add_argument(
            '--corpus-dir', type=str,
            help='Directory containing corpus files'
        )
        
        parser.add_argument(
            '--template-file', type=str,
            help='Template configuration file'
        )
        
        parser.add_argument(
            '--dictionary-file', type=str,
            help='Figurative language dictionary file'
        )
        
        # Style and content parameters
        parser.add_argument(
            '--language', choices=['en', 'es'], default='en',
            help='Language for generation (default: en)'
        )
        
        parser.add_argument(
            '--seed-words', type=str,
            help='Comma-separated seed words for thematic guidance'
        )
        
        parser.add_argument(
            '--category', type=str,
            help='Template category (e.g., lyrical, narrative, descriptive)'
        )
        
        parser.add_argument(
            '--formality', type=float, default=0.5,
            help='Formality level (0.0-1.0, default: 0.5)'
        )
        
        parser.add_argument(
            '--emotion-intensity', type=float, default=0.5,
            help='Emotional intensity (0.0-1.0, default: 0.5)'
        )
        
        parser.add_argument(
            '--imagery-density', type=float, default=0.5,
            help='Imagery density (0.0-1.0, default: 0.5)'
        )
        
        # Poetic structure
        parser.add_argument(
            '--poem-form', choices=['free_verse', 'rhymed', 'fixed_pattern'],
            default='free_verse', help='Poem form (default: free_verse)'
        )
        
        parser.add_argument(
            '--rhyme-scheme', type=str,
            help='Rhyme scheme pattern (e.g., ABAB, AABA)'
        )
        
        parser.add_argument(
            '--syllable-pattern', type=str,
            help='Syllable pattern (e.g., 5,7,5 for haiku-like)'
        )
        
        parser.add_argument(
            '--stanzas', type=int, default=1,
            help='Number of stanzas (default: 1)'
        )
        
        parser.add_argument(
            '--lines-per-stanza', type=int, default=4,
            help='Lines per stanza (default: 4)'
        )
        
        # Markov-specific parameters
        parser.add_argument(
            '--n-gram-order', type=int, default=2,
            help='N-gram order for Markov chains (default: 2)'
        )
        
        # Hybrid mode parameters
        parser.add_argument(
            '--blend-ratio', type=float, default=0.5,
            help='Template/Markov blend ratio for hybrid mode (0.0-1.0, default: 0.5)'
        )
        
        parser.add_argument(
            '--hybrid-mode', choices=['alternate', 'blend', 'enhance', 'compete'],
            default='blend', help='Hybrid generation mode (default: blend)'
        )
        
        # Output options
        parser.add_argument(
            '--output-format', choices=['text', 'markdown'], default='text',
            help='Output format (default: text)'
        )
        
        parser.add_argument(
            '--include-title', action='store_true',
            help='Generate and include poem title'
        )
        
        parser.add_argument(
            '--export-file', type=str,
            help='Export generated poems to file'
        )
        
        # Configuration and debugging
        parser.add_argument(
            '--config', type=str,
            help='Configuration file (YAML or JSON)'
        )
        
        parser.add_argument(
            '--seed', type=int,
            help='Random seed for reproducible generation'
        )
        
        parser.add_argument(
            '--verbose', '-v', action='store_true',
            help='Enable verbose logging'
        )
        
        parser.add_argument(
            '--debug', action='store_true',
            help='Enable debug logging'
        )
        
        return parser
    
    def setup_logging(self, verbose: bool = False, debug: bool = False) -> None:
        """
        Setup logging configuration.
        
        Args:
            verbose: Enable verbose logging
            debug: Enable debug logging
        """
        if debug:
            level = logging.DEBUG
        elif verbose:
            level = logging.INFO
        else:
            level = logging.WARNING
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def load_configuration(self, args: argparse.Namespace) -> None:
        """
        Load and apply configuration from file and command line arguments.
        
        Args:
            args: Parsed command line arguments
        """
        # Load config file if specified
        if args.config:
            self.config_manager.load_config(args.config)
        
        # Override with command line arguments
        cli_overrides = {
            'generation.mode': args.mode,
            'generation.max_length': args.max_length,
            'generation.temperature': args.temperature,
            'generation.language': args.language,
            'markov.n_gram_order': args.n_gram_order,
            'template.formality': args.formality,
            'template.emotion_intensity': args.emotion_intensity,
            'template.imagery_density': args.imagery_density,
            'style.poem_form': args.poem_form,
            'style.rhyme_scheme': args.rhyme_scheme,
            'style.stanza_count': args.stanzas,
            'style.lines_per_stanza': args.lines_per_stanza,
            'output.format': args.output_format,
            'output.include_title': args.include_title,
            'random.seed': args.seed
        }
        
        # Apply non-None values
        for key, value in cli_overrides.items():
            if value is not None:
                self.config_manager.set(key, value)
        
        # Handle hybrid-specific settings
        if args.mode == 'hybrid':
            self.config_manager.set('generation.blend_ratio', args.blend_ratio)
            if hasattr(args, 'hybrid_mode'):
                self.config_manager.set('generation.hybrid_mode', args.hybrid_mode)
        
        # Validate configuration
        errors = self.config_manager.validate_config()
        if errors:
            self.logger.error("Configuration validation errors:")
            for error in errors:
                self.logger.error(f"  - {error}")
            sys.exit(1)
    
    def initialize_components(self, args: argparse.Namespace) -> None:
        """
        Initialize generator components based on configuration.
        
        Args:
            args: Parsed command line arguments
        """
        language = self.config_manager.get('generation.language')
        seed = self.config_manager.get('random.seed')
        
        # Initialize utilities
        self.random_utils = RandomUtils(seed)
        
        # Initialize corpus processing components
        self.corpus_loader = CorpusLoader()
        self.text_cleaner = TextCleaner(language)
        self.tokenizer = Tokenizer(language)
        
        # Initialize style components
        self.rhyme_engine = RhymeEngine(language)
        self.figurative = FigurativeLanguage(language)
        
        # Load dictionaries if specified
        if args.dictionary_file:
            self.figurative.load_figurative_dictionary(args.dictionary_file)
        
        # Initialize generators based on mode
        mode = self.config_manager.get('generation.mode')
        
        if mode in ['template', 'hybrid']:
            self.template_gen = TemplateGenerator(language, self.random_utils)
            
            # Load templates
            template_file = args.template_file
            if not template_file:
                # Use default template file
                template_dir = self.config_manager.get('files.template_dir', 'config/templates')
                template_file = f"{template_dir}/{language}.json"
            
            if Path(template_file).exists():
                self.template_gen.load_templates(template_file)
            else:
                self.logger.warning(f"Template file not found: {template_file}")
            
            # Set style parameters
            self.template_gen.set_style_parameters(
                formality=self.config_manager.get('template.formality'),
                emotion_intensity=self.config_manager.get('template.emotion_intensity'),
                imagery_density=self.config_manager.get('template.imagery_density')
            )
        
        if mode in ['markov', 'hybrid']:
            temperature = self.config_manager.get('generation.temperature')
            n_gram = self.config_manager.get('markov.n_gram_order')
            
            self.markov_gen = MarkovGenerator(n_gram, temperature, self.random_utils)
            
            # Train Markov model if corpus provided
            self._train_markov_model(args)
        
        if mode == 'hybrid' and self.template_gen and self.markov_gen:
            self.hybrid_gen = HybridGenerator(
                self.template_gen, self.markov_gen, self.random_utils
            )
            
            # Configure hybrid mode
            blend_ratio = self.config_manager.get('generation.blend_ratio', 0.5)
            hybrid_mode = getattr(args, 'hybrid_mode', 'blend')
            
            self.hybrid_gen.set_blend_ratio(blend_ratio)
            self.hybrid_gen.set_generation_mode(hybrid_mode)
    
    def _train_markov_model(self, args: argparse.Namespace) -> None:
        """
        Train the Markov model on provided corpus data.
        
        Args:
            args: Command line arguments containing corpus information
        """
        corpus_text = ""
        
        # Load corpus from files
        if args.corpus:
            corpus_files = []
            for corpus_arg in args.corpus:
                # Handle comma-separated files
                corpus_files.extend(corpus_arg.split(','))
            
            try:
                corpus_text = self.corpus_loader.load_multiple_files(corpus_files)
            except Exception as e:
                self.logger.error(f"Error loading corpus files: {e}")
                sys.exit(1)
        
        elif args.corpus_dir:
            try:
                corpus_data = self.corpus_loader.load_directory(args.corpus_dir)
                corpus_text = "\n\n".join(corpus_data.values())
            except Exception as e:
                self.logger.error(f"Error loading corpus directory: {e}")
                sys.exit(1)
        
        else:
            # Try default corpus directory
            default_corpus_dir = self.config_manager.get('files.corpus_dir', 'corpora')
            if Path(default_corpus_dir).exists():
                try:
                    corpus_data = self.corpus_loader.load_directory(default_corpus_dir)
                    corpus_text = "\n\n".join(corpus_data.values())
                except Exception:
                    pass
        
        if not corpus_text:
            self.logger.warning("No corpus data available for Markov training")
            return
        
        # Validate corpus
        if not self.corpus_loader.validate_corpus(corpus_text):
            self.logger.error("Corpus validation failed")
            sys.exit(1)
        
        # Clean and tokenize
        cleaned_text = self.text_cleaner.clean_text(corpus_text)
        tokenized = self.tokenizer.tokenize_for_generation(cleaned_text)
        
        # Train Markov model
        self.markov_gen.train(tokenized['words'])
        
        # Build rhyme dictionary from corpus vocabulary
        self.rhyme_engine.build_rhyme_dictionary(tokenized['vocabulary'])
        
        self.logger.info(f"Trained Markov model on {len(tokenized['words'])} tokens")
    
    def parse_seed_words(self, seed_words_str: str) -> List[str]:
        """
        Parse comma-separated seed words string.
        
        Args:
            seed_words_str: Comma-separated seed words
            
        Returns:
            List of seed words
        """
        if not seed_words_str:
            return []
        
        return [word.strip() for word in seed_words_str.split(',') if word.strip()]
    
    def generate_single_poem(self, args: argparse.Namespace) -> str:
        """
        Generate a single poem based on configuration.
        
        Args:
            args: Command line arguments
            
        Returns:
            Generated poem text
        """
        mode = self.config_manager.get('generation.mode')
        max_length = self.config_manager.get('generation.max_length')
        seed_words = self.parse_seed_words(args.seed_words) if args.seed_words else None
        
        if mode == 'template':
            return self.generate_template_poem(args, seed_words)
        elif mode == 'markov':
            return self.generate_markov_poem(max_length, seed_words)
        elif mode == 'hybrid':
            return self.generate_hybrid_poem(args, max_length, seed_words)
        else:
            raise ValueError(f"Unknown generation mode: {mode}")
    
    def generate_template_poem(self, args: argparse.Namespace, seed_words: Optional[List[str]]) -> str:
        """Generate poem using template method."""
        if not self.template_gen:
            raise RuntimeError("Template generator not initialized")
        
        # Handle structured poems
        poem_form = self.config_manager.get('style.poem_form')
        rhyme_scheme = self.config_manager.get('style.rhyme_scheme')
        stanza_count = self.config_manager.get('style.stanza_count')
        lines_per_stanza = self.config_manager.get('style.lines_per_stanza')
        
        if poem_form == 'rhymed' and rhyme_scheme:
            return self.generate_rhyming_poem(rhyme_scheme, stanza_count, seed_words, args.category)
        else:
            # Generate free verse or simple structured poem
            lines = []
            total_lines = stanza_count * lines_per_stanza
            
            for _ in range(total_lines):
                line = self.template_gen.generate_from_template(
                    category=args.category,
                    seed_words=seed_words
                )
                if line:
                    lines.append(line)
            
            # Format into stanzas
            return self.format_poem_stanzas(lines, lines_per_stanza)
    
    def generate_markov_poem(self, max_length: int, seed_words: Optional[List[str]]) -> str:
        """Generate poem using Markov chain method."""
        if not self.markov_gen:
            raise RuntimeError("Markov generator not initialized")
        
        words = self.markov_gen.generate_text(
            max_length=max_length,
            end_patterns=['.', '!', '?']
        )
        
        poem_text = ' '.join(words) if words else "Unable to generate poem"
        
        # Format as lines (simple line breaking)
        return self.format_as_poem_lines(poem_text)
    
    def generate_hybrid_poem(self, args: argparse.Namespace, max_length: int, 
                           seed_words: Optional[List[str]]) -> str:
        """Generate poem using hybrid method."""
        if not self.hybrid_gen:
            raise RuntimeError("Hybrid generator not initialized")
        
        poem_text = self.hybrid_gen.generate_text(
            max_length=max_length,
            category=args.category,
            seed_words=seed_words
        )
        
        return self.format_as_poem_lines(poem_text)
    
    def generate_rhyming_poem(self, rhyme_scheme: str, stanza_count: int,
                            seed_words: Optional[List[str]], category: str = None) -> str:
        """
        Generate a poem following a specific rhyme scheme.
        
        Args:
            rhyme_scheme: Rhyme scheme pattern
            stanza_count: Number of stanzas
            seed_words: Optional seed words
            category: Template category
            
        Returns:
            Generated rhyming poem
        """
        if not self.template_gen or not self.rhyme_engine:
            return "Rhyming generation requires template generator and rhyme engine"
        
        # Get vocabulary for rhyme generation
        vocabulary = list(self.rhyme_engine.rhyme_dict.keys()) if self.rhyme_engine.rhyme_dict else []
        if not vocabulary:
            # Fallback vocabulary
            vocabulary = ['love', 'heart', 'light', 'night', 'day', 'way', 'time', 'rhyme', 
                         'dream', 'stream', 'fire', 'desire', 'hope', 'scope']
        
        # Get rhyme scheme words
        scheme_words = self.rhyme_engine.get_rhyme_scheme_words(rhyme_scheme, vocabulary)
        
        poem_lines = []
        
        for stanza in range(stanza_count):
            stanza_lines = []
            
            for i, rhyme_letter in enumerate(rhyme_scheme):
                # Generate line ending with appropriate rhyme
                if rhyme_letter in scheme_words:
                    rhyme_words = scheme_words[rhyme_letter]
                    target_word = rhyme_words[i % len(rhyme_words)] if rhyme_words else None
                else:
                    target_word = None
                
                # Generate template line
                line = self.template_gen.generate_from_template(
                    category=category,
                    seed_words=seed_words
                )
                
                # Modify line to end with rhyming word
                if target_word and line:
                    words = line.split()
                    if words:
                        words[-1] = target_word
                        line = ' '.join(words)
                
                stanza_lines.append(line if line else f"Line ending with {target_word or 'word'}")
            
            poem_lines.extend(stanza_lines)
            poem_lines.append("")  # Empty line between stanzas
        
        return '\n'.join(poem_lines).strip()
    
    def format_poem_stanzas(self, lines: List[str], lines_per_stanza: int) -> str:
        """
        Format lines into stanzas.
        
        Args:
            lines: List of poem lines
            lines_per_stanza: Number of lines per stanza
            
        Returns:
            Formatted poem with stanzas
        """
        formatted_lines = []
        
        for i, line in enumerate(lines):
            formatted_lines.append(line)
            
            # Add empty line after each stanza (except the last)
            if (i + 1) % lines_per_stanza == 0 and i < len(lines) - 1:
                formatted_lines.append("")
        
        return '\n'.join(formatted_lines)
    
    def format_as_poem_lines(self, text: str, max_words_per_line: int = 8) -> str:
        """
        Format text as poem lines with reasonable line breaks.
        
        Args:
            text: Input text
            max_words_per_line: Maximum words per line
            
        Returns:
            Formatted poem
        """
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            
            # Break line on natural boundaries or max length
            if (len(current_line) >= max_words_per_line or 
                word.endswith(('.', '!', '?', ',')) or
                len(' '.join(current_line)) > 60):
                
                lines.append(' '.join(current_line))
                current_line = []
        
        # Add remaining words
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    def generate_title(self, poem_text: str, seed_words: Optional[List[str]] = None) -> str:
        """
        Generate title for the poem.
        
        Args:
            poem_text: Generated poem text
            seed_words: Optional seed words
            
        Returns:
            Generated title
        """
        title_generation = self.config_manager.get('output.title_generation', 'keyword_based')
        
        if title_generation == 'keyword_based':
            # Extract interesting words from poem
            words = poem_text.split()
            interesting_words = [w.strip('.,!?') for w in words 
                               if len(w) > 4 and w.lower() not in {'the', 'and', 'that', 'with'}]
            
            if interesting_words:
                title_word = interesting_words[0].title()
                return f"On {title_word}" if len(title_word) > 6 else f"{title_word} Dreams"
            
        elif title_generation == 'first_line':
            lines = poem_text.split('\n')
            if lines and lines[0].strip():
                first_line = lines[0].strip()
                # Use first few words as title
                title_words = first_line.split()[:4]
                return ' '.join(title_words).title()
        
        # Fallback titles
        if seed_words:
            return f"Reflections on {seed_words[0].title()}"
        
        return "Untitled Poem"
    
    def format_output(self, poem_text: str, title: str = None) -> str:
        """
        Format poem for output based on configuration.
        
        Args:
            poem_text: Generated poem text
            title: Optional poem title
            
        Returns:
            Formatted output
        """
        output_format = self.config_manager.get('output.format')
        include_title = self.config_manager.get('output.include_title')
        
        output_lines = []
        
        if include_title and title:
            if output_format == 'markdown':
                output_lines.append(f"# {title}")
                output_lines.append("")
            else:
                output_lines.append(title)
                output_lines.append("=" * len(title))
                output_lines.append("")
        
        output_lines.append(poem_text)
        
        return '\n'.join(output_lines)
    
    def export_to_file(self, content: str, filename: str) -> None:
        """
        Export generated content to file.
        
        Args:
            content: Content to export
            filename: Output filename
        """
        try:
            output_path = Path(filename)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"Exported to {filename}")
            
        except IOError as e:
            self.logger.error(f"Error exporting to file: {e}")
    
    def run(self, args: List[str] = None) -> int:
        """
        Run the CLI application.
        
        Args:
            args: Command line arguments (uses sys.argv if None)
            
        Returns:
            Exit code
        """
        parser = self.create_argument_parser()
        parsed_args = parser.parse_args(args)
        
        # Setup logging
        self.setup_logging(parsed_args.verbose, parsed_args.debug)
        
        try:
            # Load configuration
            self.load_configuration(parsed_args)
            
            # Initialize components
            self.initialize_components(parsed_args)
            
            # Generate poems
            all_output = []
            
            for i in range(parsed_args.count):
                self.logger.info(f"Generating poem {i + 1}/{parsed_args.count}")
                
                poem_text = self.generate_single_poem(parsed_args)
                
                # Generate title if requested
                title = None
                if self.config_manager.get('output.include_title'):
                    seed_words = self.parse_seed_words(parsed_args.seed_words) if parsed_args.seed_words else None
                    title = self.generate_title(poem_text, seed_words)
                
                # Format output
                formatted_output = self.format_output(poem_text, title)
                
                if parsed_args.count > 1:
                    all_output.append(f"--- Poem {i + 1} ---")
                    all_output.append("")
                
                all_output.append(formatted_output)
                all_output.append("")
            
            final_output = '\n'.join(all_output).strip()
            
            # Print to console
            print(final_output)
            
            # Export if requested
            if parsed_args.export_file:
                self.export_to_file(final_output, parsed_args.export_file)
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            if parsed_args.debug:
                import traceback
                traceback.print_exc()
            return 1


def main():
    """Main entry point for the CLI application."""
    cli = PoetryGeneratorCLI()
    sys.exit(cli.run())
