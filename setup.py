"""
Setup script for fuzzy-soft-circuit package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fuzzy-soft-circuit",
    version="0.1.0",
    author="Alexander Towell",
    description="Automatic fuzzy rule discovery through differentiable soft circuits",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/queelius/fuzzy-soft-circuit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "autograd>=1.3",
        "scipy>=1.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "isort",
            "mypy",
        ],
        "viz": [
            "matplotlib>=3.0",
            "seaborn>=0.11",
        ],
        "benchmarks": [
            "scikit-learn>=0.24",
            "pandas>=1.2",
        ],
    },
)
