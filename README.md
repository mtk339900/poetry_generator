# Poetry Generator

A production-ready Python system for generating original short poems and prose using custom-built algorithms. This implementation uses rule-based templates, Markov chain probabilistic models, and hybrid approaches without relying on external AI/ML/NLP libraries.

## Features

### Core Generation Methods
- **Template-based generation**: Predefined sentence structures with variable placeholders
- **Markov chain generation**: N-gram models trained on custom corpora
- **Hybrid generation**: Combines template and Markov approaches with multiple modes

### Poetic Structure Support
- **Free verse**: No rhyme or strict syllable requirements
- **Rhymed verse**: End rhymes using custom phonetic approximation
- **Fixed patterns**: Configurable syllable counts (e.g., haiku-like structures)
- **Multi-stanza poems**: Configurable stanza and line counts

### Stylistic Features
- **Temperature-controlled randomness**: Adjustable determinism vs. creativity
- **Figurative language**: Automatic insertion of metaphors, similes, personification, and alliteration
- **Thematic guidance**: Seed words to influence tone and imagery
- **Style parameters**: Formality, emotional intensity, and imagery density controls
- **Multi-language support**: English and Spanish with extensible framework

### Advanced Capabilities
- **Rhyme engine**: Custom phonetic matching without external libraries
- **Corpus preparation**: Built-from-scratch tokenization, cleaning, and frequency analysis
- **Memory-efficient n-grams**: Optimized data structures for large corpora
- **Reproducible generation**: Random seed support for consistent outputs
- **Comprehensive logging**: Debug-level traceability of generation decisions

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd poetry_generator

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### Basic Usage

```bash
# Generate a simple poem using hybrid mode
python main.py --mode hybrid --count 1

# Generate with seed words for thematic guidance
python main.py --seed-words "love,nature,time" --include-title

# Generate rhyming poem
python main.py --poem-form rhymed --rhyme-scheme ABAB --stanzas 2
```

### Using Different Generation Modes

```bash
# Template-based generation
python main.py --mode template --category lyrical --formality 0.7

# Markov chain generation
python main.py --mode markov --corpus corpora/sample_en.txt --max-length 40

# Hybrid with custom blend ratio
python main.py --mode hybrid --blend-ratio 0.3 --hybrid-mode enhance
```

## Configuration

### Configuration File Format

The system supports YAML and JSON configuration files. Use `config/default_config.yaml` as a starting point:

```yaml
generation:
  mode: hybrid                    # template, markov, hybrid
  max_length: 50                  # maximum words to generate
  temperature: 1.0                # randomness control (0.1-2.0)
  language: en                    # en, es

template:
  formality: 0.5                  # formality level (0.0-1.0)
  emotion_intensity: 0.5          # emotional intensity (0.0-1.0)
  imagery_density: 0.5            # imagery density (0.0-1.0)

style:
  poem_form: free_verse           # free_verse, rhymed, fixed_pattern
  rhyme_scheme: null              # ABAB, AABA, etc.
  stanza_count: 1                 # number of stanzas
  lines_per_stanza: 4             # lines per stanza
```

### Command Line Options

#### Generation Parameters
- `--mode {template,markov,hybrid}`: Generation method
- `--max-length INT`: Maximum words to generate (default: 50)
- `--temperature FLOAT`: Randomness control 0.1-2.0 (default: 1.0)
- `--count INT`: Number of poems to generate (default: 1)

#### Input Sources
- `--corpus FILE [FILE ...]`: Corpus files for Markov training
- `--corpus-dir DIR`: Directory containing corpus files
- `--template-file FILE`: Custom template configuration
- `--dictionary-file FILE`: Figurative language dictionary

#### Style Control
- `--language {en,es}`: Generation language (default: en)
- `--seed-words WORDS`: Comma-separated thematic seed words
- `--formality FLOAT`: Formality level 0.0-1.0 (default: 0.5)
- `--emotion-intensity FLOAT`: Emotional intensity 0.0-1.0 (default: 0.5)
- `--imagery-density FLOAT`: Imagery density 0.0-1.0 (default: 0.5)

#### Poetic Structure
- `--poem-form {free_verse,rhymed,fixed_pattern}`: Poem structure
- `--rhyme-scheme PATTERN`: Rhyme pattern (e.g., ABAB, AABA)
- `--stanzas INT`: Number of stanzas (default: 1)
- `--lines-per-stanza INT`: Lines per stanza (default: 4)

#### Output Options
- `--output-format {text,markdown}`: Output format
- `--include-title`: Generate poem titles
- `--export-file FILE`: Export to file
- `--seed INT`: Random seed for reproducibility

## Corpus Preparation

### Input Format
- **Text files**: UTF-8 encoded plain text
- **Directory processing**: Automatic processing of .txt files
- **Multiple files**: Comma-separated file lists or multiple --corpus arguments

### Preprocessing Pipeline
1. **Loading**: Flexible file and directory input
2. **Cleaning**: Punctuation handling, case normalization, number removal
3. **Tokenization**: Word and sentence boundary detection
4. **Filtering**: Stopword removal, frequency thresholding
5. **N-gram generation**: Configurable order (1-5)

### Corpus Requirements
- **Minimum size**: 100 words (configurable)
- **Language consistency**: Match generation language setting
- **Quality**: Clean, well-formed text produces better results

### Example Corpus Structure
```
corpora/
├── english_poetry.txt          # Primary English corpus
├── nature_themes.txt           # Thematic corpus
├── spanish/
│   ├── poesia_clasica.txt     # Spanish poetry
│   └── literatura_moderna.txt # Modern literature
└── specialized/
    ├── love_poems.txt         # Genre-specific corpus
    └── philosophical.txt      # Thematic corpus
```

## Template System

### Template Syntax
Templates use `{placeholder}` syntax for variable substitution:

```json
{
  "templates": {
    "lyrical": [
      "In the {time_of_day}, {emotion} fills the {location}",
      "The {natural_element} whispers {message} to the {listener}",
      "Between {place1} and {place2}, {abstract_concept} dwells"
    ]
  },
  "word_sets": {
    "time_of_day": ["dawn", "morning", "twilight", "dusk"],
    "emotions": ["love", "longing", "hope", "joy"],
    "locations": ["garden", "forest", "meadow", "shore"]
  }
}
```

### Template Categories
- **lyrical**: Emotional and artistic expressions
- **narrative**: Story-telling structures
- **descriptive**: Observational and sensory descriptions
- **philosophical**: Abstract and contemplative themes
- **melancholic**: Sad and reflective moods
- **joyful**: Happy and celebratory expressions

### Custom Templates
Create custom template files following the JSON structure:
1. Define template patterns with placeholders
2. Create word_sets for each placeholder type
3. Add metaphor_patterns and simile_patterns for figurative language
4. Place in `config/templates/` directory

## Advanced Features

### Hybrid Generation Modes
- **alternate**: Alternates between template and Markov generation
- **blend**: Uses blend_ratio to choose method probabilistically
- **enhance**: Template-based with Markov enhancements
- **compete**: Generates multiple candidates and selects best

### Rhyme Engine
Custom phonetic approximation without external libraries:
- **Ending sound extraction**: Analyzes word endings phonetically
- **Rhyme matching**: Groups words by similar ending sounds
- **Quality scoring**: Ranks rhyme quality by multiple factors
- **Scheme validation**: Checks adherence to rhyme patterns

### Figurative Language Generation
- **Metaphor creation**: Subject-object relationship patterns
- **Simile generation**: Comparison-based figurative expressions
- **Personification**: Human qualities to non-human objects
- **Alliteration**: Consonant sound repetition
- **Sensory enhancement**: Visual, auditory, and tactile details

### Performance Optimization
- **Efficient n-grams**: Dictionary-of-dictionaries structure
- **Memory management**: Configurable vocabulary limits
- **Lazy loading**: On-demand resource initialization
- **Batch processing**: Multiple poem generation with shared models

## Usage Examples

### Example 1: Simple Lyrical Poem
```bash
python main.py --mode template --category lyrical --seed-words "sunset,peace" --include-title
```

Output:
```
Peaceful Evening
===============

In the twilight, peace fills the meadow
The sunlight whispers promises to the heart
Between earth and sky, serenity dwells
With golden hues, beauty embraces all
```

### Example 2: Markov Chain Generation
```bash
python main.py --mode markov --corpus corpora/sample_en.txt --max-length 30 --temperature 1.2
```

Output:
```
The wind whispers through ancient valleys where
rivers sing their eternal dance with grace
carrying stories of forgotten dreams that
bloom in unexpected places among the
rustling leaves and gentle morning light
```

### Example 3: Rhyming Poem
```bash
python main.py --poem-form rhymed --rhyme-scheme ABAB --stanzas 2 --lines-per-stanza 4
```

Output:
```
In gardens where the roses grow (A)
The morning light begins to dance (B)
Through petals soft that gently glow (A)
In nature's sweet and pure romance (B)

The whispered songs that breezes bring (A)
Across the meadow's verdant face (B)
Inspire the heart to rise and sing (A)
Of beauty found in this quiet place (B)
```

### Example 4: Multi-language Generation
```bash
python main.py --language es --mode template --category lírico --seed-words "amor,tiempo" --include-title
```

Output:
```
Sobre el Tiempo
==============

En el amanecer, amor llena el jardín
El viento susurra secretos al alma
Entre corazón y mente, eternidad habita
Con ternura, esperanza abraza toda existencia
```

### Example 5: Hybrid with Custom Parameters
```bash
python main.py --mode hybrid --blend-ratio 0.7 --hybrid-mode enhance --formality 0.8 --emotion-intensity 0.9 --imagery-density 0.6 --count 3
```

### Example 6: Reproducible Generation
```bash
python main.py --seed 42 --mode hybrid --count 5 --export-file "generated_poems.txt"
```

### Example 7: High Imagery Density
```bash
python main.py --imagery-density 0.9 --category descriptive --seed-words "ocean,storm,light" --lines-per-stanza 6
```

## Configuration Examples

### Custom Style Configuration
```yaml
template:
  formality: 0.2              # Casual, conversational
  emotion_intensity: 0.8      # Highly emotional
  imagery_density: 0.9        # Rich imagery

figurative:
  metaphor_probability: 0.5   # Frequent metaphors
  simile_probability: 0.4     # Common similes
  alliteration_probability: 0.2  # Occasional alliteration
```

### Markov Chain Tuning
```yaml
markov:
  n_gram_order: 3             # Longer context
  temperature: 0.8            # Less random
  sentence_boundaries: true   # Respect sentence structure

corpus:
  min_word_freq: 2           # Filter rare words
  max_vocabulary: 5000       # Limit vocabulary size
  remove_stopwords: false    # Keep all words
```

### Output Customization
```yaml
output:
  format: markdown           # Markdown formatting
  include_title: true        # Generate titles
  title_generation: first_line  # Use first line as title

style:
  poem_form: fixed_pattern   # Structured poems
  syllable_pattern: "5,7,5"  # Haiku-like structure
```

## Troubleshooting

### Common Issues

#### "No corpus data available for Markov training"
- Ensure corpus files exist and are readable
- Check file encoding (should be UTF-8)
- Verify minimum word count requirements
- Use `--corpus-dir` or `--corpus` flags explicitly

#### "Template file not found"
- Check `config/templates/` directory exists
- Verify language-specific template file (e.g., `english.json`, `spanish.json`)
- Use `--template-file` to specify custom template location
- Ensure JSON syntax is valid

#### "Rhyme generation produces poor results"
- Build larger rhyme dictionary with more diverse vocabulary
- Increase corpus size for better word coverage
- Adjust `--imagery-density` and `--emotion-intensity` for better word selection
- Use `--seed-words` to guide rhyme theme

#### "Generated text seems repetitive"
- Increase `--temperature` for more randomness
- Use larger, more diverse corpus
- Try different `--hybrid-mode` settings
- Increase `max_vocabulary` in configuration

#### "Memory usage too high"
- Reduce `max_vocabulary` in corpus configuration
- Lower `n_gram_order` for Markov chains
- Process corpus in smaller chunks
- Enable `remove_stopwords` to reduce vocabulary size

### Performance Tips

#### For Large Corpora
```yaml
corpus:
  max_vocabulary: 10000       # Limit vocabulary size
  min_word_freq: 3           # Filter uncommon words
  
markov:
  n_gram_order: 2            # Use smaller n-grams
```

#### For Better Quality
```yaml
generation:
  temperature: 0.8           # Less randomness
  
template:
  formality: 0.6            # More structured output
  
figurative:
  metaphor_probability: 0.4  # Moderate figurative language
```

#### For Faster Generation
```yaml
generation:
  max_length: 30            # Shorter poems
  
style:
  stanza_count: 1           # Single stanza
  lines_per_stanza: 4       # Fewer lines
```

## API Reference

### Core Classes

#### `CorpusLoader`
```python
from poetry_generator.corpus.loader import CorpusLoader

loader = CorpusLoader(encoding='utf-8')
text = loader.load_text_file('corpus.txt')
corpus_dict = loader.load_directory('corpora/')
```

#### `MarkovGenerator`
```python
from poetry_generator.generation.markov import MarkovGenerator

markov = MarkovGenerator(n=2, temperature=1.0)
markov.train(tokens)
words = markov.generate_text(max_length=50)
```

#### `TemplateGenerator`
```python
from poetry_generator.generation.template import TemplateGenerator

template_gen = TemplateGenerator(language='en')
template_gen.load_templates('config/templates/english.json')
template_gen.set_style_parameters(formality=0.7)
text = template_gen.generate_from_template(category='lyrical')
```

#### `HybridGenerator`
```python
from poetry_generator.generation.hybrid import HybridGenerator

hybrid = HybridGenerator(template_gen, markov_gen)
hybrid.set_blend_ratio(0.6)
hybrid.set_generation_mode('enhance')
text = hybrid.generate_text(max_length=40)
```

#### `RhymeEngine`
```python
from poetry_generator.style.rhyme import RhymeEngine

rhyme_engine = RhymeEngine(language='en')
rhyme_engine.build_rhyme_dictionary(vocabulary)
rhymes = rhyme_engine.find_rhymes('love', max_rhymes=5)
```

### Programmatic Usage

```python
from poetry_generator.utils.config import ConfigManager
from poetry_generator.utils.random_utils import RandomUtils
from poetry_generator.corpus.loader import CorpusLoader
from poetry_generator.generation.hybrid import HybridGenerator

# Setup
config = ConfigManager('my_config.yaml')
random_utils = RandomUtils(seed=42)

# Load and prepare corpus
loader = CorpusLoader()
corpus_text = loader.load_text_file('my_corpus.txt')

# Initialize generators
# ... (initialization code)

# Generate poem
poem = hybrid_gen.generate_text(
    max_length=50,
    seed_words=['nature', 'peace']
)
print(poem)
```

## Development

### Project Structure
```
poetry_generator/
├── poetry_generator/          # Main package
│   ├── corpus/               # Corpus processing modules
│   ├── generation/           # Text generation algorithms  
│   ├── style/               # Poetic style and structure
│   ├── cli/                 # Command line interface
│   └── utils/               # Utilities and configuration
├── config/                   # Configuration files
│   ├── templates/           # Template definitions
│   └── dictionaries/        # Word lists and mappings
├── corpora/                 # Sample training corpora
└── tests/                   # Test suite (not included)
```

### Extension Points

#### Adding New Languages
1. Create template file in `config/templates/{language}.json`
2. Add language-specific cleaning rules in `TextCleaner`
3. Update phonetic rules in `RhymeEngine`
4. Create figurative language dictionary

#### Custom Generation Modes
1. Inherit from base generator classes
2. Implement required methods (`generate_text`, etc.)
3. Register in CLI argument parser
4. Add configuration options

#### New Poetic Forms
1. Define structure in configuration schema
2. Implement structure validation in `RhymeEngine`
3. Add formatting logic in CLI module
4. Create templates for the new form

### Code Quality
- **Type hints**: All public functions use type annotations
- **Docstrings**: Comprehensive documentation for all modules
- **Logging**: Debug-level traceability throughout
- **Error handling**: Graceful failure with informative messages
- **Configuration validation**: Comprehensive parameter checking

## License

This project is released under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Code Style**: Follow PEP 8 conventions
2. **Documentation**: Update docstrings and README for new features  
3. **Testing**: Add tests for new functionality
4. **Compatibility**: Ensure Python 3.8+ compatibility
5. **Dependencies**: Avoid adding external AI/ML/NLP libraries

### Areas for Contribution
- Additional language support
- New poetic forms and structures
- Enhanced figurative language patterns
- Performance optimizations
- Better phonetic approximation algorithms
- Advanced corpus preprocessing techniques

## Acknowledgments

This project implements custom algorithms for text generation without relying on external AI/ML libraries, making it suitable for educational purposes and environments where dependency minimization is important.

The system draws inspiration from classical computational linguistics techniques while providing a modern, configurable interface for creative text generation.
