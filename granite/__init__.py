"""
GRANITE: Graph-Refined Accessibility Network for Integrated Transit Equity
==========================================================================

A framework for Social Vulnerability Index disaggregation using Graph Neural 
Networks integrated with MetricGraph's Whittle-Matérn spatial models.

Author: [Your Name]
Date: July 2025
"""

__version__ = "0.1.0"
__author__ = "[Your Name]"

# Import main components
from .data import loaders
from .models import gnn
from .metricgraph import interface
from .disaggregation import pipeline
from .visualization import plots

# Package metadata
__all__ = [
    "loaders",
    "gnn", 
    "interface",
    "pipeline",
    "plots"
]