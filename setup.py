#!/usr/bin/env python
"""Setup script for GRANITE."""

from setuptools import setup, find_packages
import os


def get_version():
    """Read version from __init__.py."""
    here = os.path.abspath(os.path.dirname(__file__))
    init_file = os.path.join(here, 'granite', '__init__.py')
    try:
        with open(init_file, encoding='utf-8') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return "0.2.0"


INSTALL_REQUIRES = [
    # scientific computing
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "scipy>=1.9.0",
    "scikit-learn>=1.0.0",

    # geospatial
    "geopandas>=0.10.0",
    "shapely>=1.8.0",
    "fiona>=1.8.0",
    "pyproj>=3.0.0",
    "geopy>=2.3.0",

    # deep learning
    "torch>=1.10.0",
    "torch-geometric>=2.0.0",

    # network analysis
    "networkx>=2.6.0",

    # visualization
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",

    # utilities
    "PyYAML>=5.4.0",
    "tqdm>=4.62.0",
    "requests>=2.25.0",
]

EXTRAS_REQUIRE = {
    'spatial': [
        "libpysal>=4.6.0",
        "esda>=2.4.0",
    ],
    'dev': [
        "pytest>=6.0",
        "pytest-cov>=2.0",
        "black>=21.0",
        "flake8>=3.9",
    ],
    'jupyter': [
        "jupyter>=1.0.0",
        "ipykernel>=6.0.0",
    ],
}

setup(
    name="granite-svi",
    version=get_version(),
    author="Jamion Williams",
    description="Graph-Refined Accessibility Network for Transportation Equity",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",

    packages=find_packages(),
    include_package_data=True,

    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires=">=3.8",

    entry_points={
        'console_scripts': [
            'granite=granite.scripts.run_granite:main',
        ],
    },

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],

    zip_safe=False,
)