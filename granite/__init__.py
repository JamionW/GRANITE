"""
GRANITE: Graph-Refined Accessibility Network for Integrated Transit Equity
==========================================================================

A framework for accessibility pattern learning using Graph Neural Networks
for transportation equity and social vulnerability analysis.

Author: Jamion Williams
Date: September 2025
"""

__version__ = "0.1.0"
__author__ = "Jamion Williams"

# Import main components
from .data import loaders
from .models import gnn
from .disaggregation import pipeline
from .visualization import plots

# New disaggregation modules
from .evaluation import baselines
from .visualization import plots

# Package metadata
__all__ = [
    "loaders",
    "gnn", 
    "interface",
    "pipeline",
    "plots",
    "disaggregation_baselines",
    "disaggregation_plots"
]