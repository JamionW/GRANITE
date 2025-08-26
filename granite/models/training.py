"""
Training utilities for GRANITE GNN models

This module provides training functionality for Graph Neural Networks
that learn accessibility features for MetricGraph integration.
"""
# Standard library imports
import time
from datetime import datetime
from typing import Dict, Tuple, Optional

# Third-party imports
import numpy as np

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def train_accessibility_corrections(graph_data, idm_baseline: np.ndarray,
                                    tract_svi: float, **kwargs):
        """
        NEW FUNCTION: Train GNN for accessibility corrections (hybrid approach).
        """
        from .gnn import AccessibilityGNNCorrector, HybridCorrectionTrainer
        
        # Create correction model
        input_dim = graph_data.x.shape[1] + 1  # +1 for IDM baseline
        model = AccessibilityGNNCorrector(
            input_dim=input_dim,
            hidden_dim=kwargs.get('hidden_dim', 64)
        )
        
        # Add IDM baseline as feature
        idm_tensor = torch.tensor(idm_baseline, dtype=torch.float32).unsqueeze(1)
        graph_data.x = torch.cat([graph_data.x, idm_tensor], dim=1)
        
        # Train corrections
        trainer = HybridCorrectionTrainer(model, config=kwargs)
        result = trainer.train_corrections(
            graph_data, idm_baseline, tract_svi, 
            epochs=kwargs.get('epochs', 100)
        )
        
        return model, result