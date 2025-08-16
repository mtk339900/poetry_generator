from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="poetry-generator",
    version="1.0.0",
    author="mohammed mostafa (mtk339900)",
    description="A custom text generation system for poetry and prose using rule-based and probabilistic algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Artistic Software",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=0.991",
        ],
    },
    entry_points={
        "console_scripts": [
            "poetry-generator=poetry_generator.cli.main:main",
        ],
    },
    package_data={
        "poetry_generator": [
            "config/templates/*.json",
            "config/dictionaries/*.json",
            "config/*.yaml",
            "corpora/*.txt",
        ],
    },
    include_package_data=True,
)
