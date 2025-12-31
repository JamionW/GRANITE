"""
GRANITE: Graph-Refined Accessibility Network for Transportation Equity

Spatial Version: Uses coordinate-based features and graph topology
for disaggregation of tract-level SVI to address resolution.
"""

__version__ = '0.2.0'

from .models.gnn import (
    SpatialDisaggregationGNN,
    SpatialGNNTrainer,
    MultiTractGNNTrainer,
    set_random_seed
)

from .features.spatial_features import (
    SpatialFeatureComputer,
    normalize_spatial_features
)

from .data.loaders import DataLoader
from .disaggregation.pipeline import GRANITEPipeline

__all__ = [
    'SpatialDisaggregationGNN',
    'SpatialGNNTrainer', 
    'MultiTractGNNTrainer',
    'SpatialFeatureComputer',
    'DataLoader',
    'GRANITEPipeline',
    'set_random_seed'
]