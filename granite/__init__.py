"""
GRANITE
=======

Constraint-preserving graph neural network for spatial disaggregation of
the CDC Social Vulnerability Index from census tract resolution to individual
addresses. The acronym is retained as a project codename.

Study area: Hamilton County, Tennessee (FIPS 47065).

Author: Jamion Williams
"""

__version__ = "0.2.0"
__author__ = "Jamion Williams"

from .data import loaders
from .models import gnn
from .disaggregation import pipeline
from .evaluation import baselines
from .visualization import plots

__all__ = [
    "loaders",
    "gnn",
    "pipeline",
    "baselines",
    "plots",
]