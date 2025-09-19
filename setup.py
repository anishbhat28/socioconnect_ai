"""
Setup script for SocioConnect AI - Immigrant Community Growth Prediction
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text(encoding="utf-8").strip().split("\n")
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]

setup(
    name="socioconnect-ai",
    version="1.0.0",
    description="Predict immigrant community growth and recommend culturally relevant businesses",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SocioConnect AI Team",
    author_email="contact@socioconnect.ai",
    url="https://github.com/socioconnect/immigrant-growth-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "socioconnect-etl=etl.main:main",
            "socioconnect-train=modeling.main:main",
            "socioconnect-api=api.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "socioconnect_ai": [
            "data/*.parquet",
            "models/*.pkl",
            "docs/*.md",
        ],
    },
    keywords=[
        "immigration",
        "community",
        "prediction",
        "machine-learning",
        "urban-planning",
        "business-recommendations",
        "census-data",
        "spatial-analysis",
    ],
    project_urls={
        "Bug Reports": "https://github.com/socioconnect/immigrant-growth-prediction/issues",
        "Source": "https://github.com/socioconnect/immigrant-growth-prediction",
        "Documentation": "https://docs.socioconnect.ai",
        "Homepage": "https://socioconnect.ai",
    },
)
