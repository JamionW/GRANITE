#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for GRANITE: Graph-Refined Accessibility Network for Integrated Transit Equity

GRANITE: A framework for Social Vulnerability Index disaggregation using Graph Neural 
Networks integrated with MetricGraph's Whittle-Matérn spatial models.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_file(filename):
    """Read a file and return its contents"""
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, filename), encoding='utf-8') as f:
        return f.read()

# Read version from __init__.py
def get_version():
    """Extract version from __init__.py"""
    here = os.path.abspath(os.path.dirname(__file__))
    init_file = os.path.join(here, 'granite', '__init__.py')
    try:
        with open(init_file, encoding='utf-8') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return "0.1.0"

# Core dependencies (required)
INSTALL_REQUIRES = [
    # Core scientific computing
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "scipy>=1.9.0",
    "scikit-learn>=1.0.0",
    
    # Geospatial libraries
    "geopandas>=0.10.0",
    "shapely>=1.8.0",
    "fiona>=1.8.0",
    "pyproj>=3.0.0",
    "rasterio>=1.2.0",  # For NLCD data processing
    
    # Deep learning
    "torch>=1.10.0",
    "torch-geometric>=2.0.0",
    
    # Network analysis
    "networkx>=2.6.0",
    
    # Visualization
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    
    # Configuration and utilities
    "PyYAML>=5.4.0",
    "tqdm>=4.62.0",
    "requests>=2.25.0",
    
    # R interface for MetricGraph
    "rpy2>=3.4.0",
]

# Optional dependencies for enhanced functionality
EXTRAS_REQUIRE = {
    'dev': [
        "pytest>=6.0",
        "pytest-cov>=2.0",
        "black>=21.0",
        "flake8>=3.9",
        "mypy>=0.910",
        "pre-commit>=2.15",
    ],
    'docs': [
        "sphinx>=4.0",
        "sphinx-rtd-theme>=1.0",
        "nbsphinx>=0.8",
        "pandoc>=1.0",
    ],
    'jupyter': [
        "jupyter>=1.0.0",
        "ipykernel>=6.0.0",
        "ipywidgets>=7.6.0",
        "jupyterlab>=3.0.0",
    ],
    'transit': [
        "gtfs-kit>=4.0.0",  # For GTFS data processing
        "osmium>=3.0.0",    # For OpenStreetMap data
    ],
    'performance': [
        "numba>=0.55.0",    # For JIT compilation
        "dask[complete]>=2021.0.0",  # For parallel processing
    ],
    'all': [
        # Include all optional dependencies
        "pytest>=6.0", "pytest-cov>=2.0", "black>=21.0", "flake8>=3.9", "mypy>=0.910", "pre-commit>=2.15",
        "sphinx>=4.0", "sphinx-rtd-theme>=1.0", "nbsphinx>=0.8", "pandoc>=1.0",
        "jupyter>=1.0.0", "ipykernel>=6.0.0", "ipywidgets>=7.6.0", "jupyterlab>=3.0.0",
        "gtfs-kit>=4.0.0", "osmium>=3.0.0",
        "numba>=0.55.0", "dask[complete]>=2021.0.0",
    ]
}

# Package data to include
PACKAGE_DATA = {
    'granite': [
        'config/*.yaml',
        'config/*.yml', 
        'data/templates/*.json',
        'data/templates/*.geojson',
    ]
}

# Entry points for command-line tools
ENTRY_POINTS = {
    'console_scripts': [
        'granite=granite.scripts.run_granite:main',
        'granite-pipeline=granite.scripts.run_granite:main',  # Alternative name
    ],
}

# Classifiers for PyPI
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "Topic :: Scientific/Engineering :: GIS",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Operating System :: OS Independent",
]

# Keywords for discoverability
KEYWORDS = [
    "spatial-analysis", "gis", "graph-neural-networks", "social-vulnerability",
    "spatial-disaggregation", "accessibility", "transit-equity", "metricgraph",
    "whittle-matern", "spde", "r-interface", "urban-planning"
]

setup(
    # Basic package information
    name="granite-svi",  # Note: 'granite' might be taken on PyPI
    version=get_version(),
    author="Jamion Williams",
    author_email="your.email@example.com",  # Update with your email
    description="Graph-Refined Accessibility Network for Integrated Transit Equity",
    long_description=read_file("README.md") if os.path.exists("README.md") else __doc__,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/granite",  # Update with your repo
    
    # Package discovery and structure
    packages=find_packages(),
    package_data=PACKAGE_DATA,
    include_package_data=True,
    
    # Dependencies
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires=">=3.8",
    
    # Entry points
    entry_points=ENTRY_POINTS,
    
    # Metadata
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    license="MIT",
    
    # Zip safety
    zip_safe=False,
)