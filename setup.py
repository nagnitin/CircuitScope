"""
CircuitScope: Mechanistic Interpretability of GPT-2 Small
==========================================================
Package setup file. Install in development mode with:

    pip install -e .

This allows importing `src` modules as `from src.model import loader`
from anywhere in the project without modifying PYTHONPATH manually.
"""

from setuptools import setup, find_packages

setup(
    name="circuitscope",
    version="0.1.0",
    description=(
        "Research-quality mechanistic interpretability toolkit for "
        "reverse-engineering the IOI circuit in GPT-2 Small using TransformerLens."
    ),
    author="CircuitScope Research",
    python_requires=">=3.11",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        "torch>=2.2.0",
        "transformer_lens>=1.19.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "plotly>=5.20.0",
        "matplotlib>=3.8.0",
        "circuitsvis>=1.4.0",
        "tqdm>=4.66.0",
        "einops>=0.7.0",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.1.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
