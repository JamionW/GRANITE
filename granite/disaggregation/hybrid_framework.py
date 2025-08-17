"""
granite/disaggregation/hybrid_framework.py

Hybrid IDM+GNN disaggregation framework that combines empirically-validated 
land cover coefficients with learned network accessibility patterns.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DisaggregationConfig:
    """Configuration for hybrid disaggregation"""
    idm_weight: float = 0.7  # Weight for IDM baseline (0.7 = 70% IDM, 30% GNN)
    gnn_correction_scale: float = 0.3  # Maximum correction magnitude from GNN
    constraint_tolerance: float = 0.01  # Tolerance for tract-level constraint
    enable_uncertainty: bool = True
    min_variation_cv: float = 0.15  # Minimum coefficient of variation to prevent collapse


class HybridDisaggregator:
    """
    Main class for hybrid IDM+GNN disaggregation.
    
    This combines:
    1. IDM baseline using empirically-validated NLCD coefficients
    2. GNN accessibility corrections based on network topology
    3. Whittle-Matérn spatial integration for uncertainty
    """
    
    def __init__(self, config: DisaggregationConfig = None):
        self.config = config or DisaggregationConfig()
        
        # Component models (will be initialized separately)
        self.idm_baseline = None
        self.gnn_corrector = None
        self.spatial_integrator = None
    
    def set_components(self, idm_baseline, gnn_corrector, spatial_integrator):
        """Set the three component models"""
        self.idm_baseline = idm_baseline
        self.gnn_corrector = gnn_corrector
        self.spatial_integrator = spatial_integrator
    
    def disaggregate(self, 
                     tract_svi: float,
                     addresses: pd.DataFrame,
                     nlcd_features: pd.DataFrame,
                     road_network: object,
                     tract_geometry: object) -> Dict:
        """
        Perform hybrid disaggregation combining IDM and GNN.
        
        Parameters:
        -----------
        tract_svi : float
            Tract-level SVI value to disaggregate
        addresses : pd.DataFrame
            Address locations for prediction
        nlcd_features : pd.DataFrame
            NLCD land cover features per address
        road_network : object
            Road network graph structure
        tract_geometry : object
            Tract boundary geometry
            
        Returns:
        --------
        Dict with predictions, uncertainty, and diagnostics
        """
        
        # Step 1: Generate IDM baseline
        print("Step 1: Computing IDM baseline...")
        idm_result = self.idm_baseline.disaggregate_svi(
            tract_svi=tract_svi,
            prediction_locations=addresses,
            nlcd_features=nlcd_features,
            tract_geometry=tract_geometry
        )
        
        if not idm_result['success']:
            return idm_result  # Fallback to pure IDM if it fails
        
        idm_predictions = idm_result['predictions']['svi_prediction'].values
        
        # Step 2: Compute GNN corrections based on network accessibility
        print("Step 2: Computing GNN accessibility corrections...")
        gnn_corrections = self.gnn_corrector.compute_corrections(
            road_network=road_network,
            addresses=addresses,
            idm_baseline=idm_predictions,
            nlcd_features=nlcd_features
        )
        
        # Step 3: Combine IDM baseline with GNN corrections
        print("Step 3: Combining IDM + GNN predictions...")
        combined_predictions = self._combine_predictions(
            idm_baseline=idm_predictions,
            gnn_corrections=gnn_corrections,
            tract_svi=tract_svi
        )
        
        # Step 4: Apply spatial integration if available
        if self.spatial_integrator is not None:
            print("Step 4: Applying Whittle-Matérn spatial integration...")
            final_predictions, uncertainty = self.spatial_integrator.integrate(
                initial_predictions=combined_predictions,
                addresses=addresses,
                tract_constraint=tract_svi
            )
        else:
            final_predictions = combined_predictions
            uncertainty = np.ones_like(combined_predictions) * 0.05
        
        # Step 5: Ensure constraint satisfaction
        final_predictions = self._enforce_tract_constraint(
            predictions=final_predictions,
            target_mean=tract_svi
        )
        
        # Step 6: Validate spatial variation
        self._validate_spatial_variation(final_predictions)
        
        # Prepare results
        results_df = addresses.copy()
        results_df['svi_prediction'] = final_predictions
        results_df['uncertainty'] = uncertainty
        results_df['idm_baseline'] = idm_predictions
        results_df['gnn_correction'] = gnn_corrections
        results_df['tract_svi'] = tract_svi
        
        diagnostics = {
            'method': 'hybrid_idm_gnn',
            'idm_weight': self.config.idm_weight,
            'mean_prediction': float(np.mean(final_predictions)),
            'std_prediction': float(np.std(final_predictions)),
            'cv_prediction': float(np.std(final_predictions) / np.mean(final_predictions)),
            'constraint_error': float(abs(np.mean(final_predictions) - tract_svi)),
            'mean_correction': float(np.mean(np.abs(gnn_corrections))),
            'max_correction': float(np.max(np.abs(gnn_corrections)))
        }
        
        return {
            'success': True,
            'predictions': results_df,
            'diagnostics': diagnostics
        }
    
    def _combine_predictions(self, idm_baseline: np.ndarray, 
                           gnn_corrections: np.ndarray,
                           tract_svi: float) -> np.ndarray:
        """
        Combine IDM baseline with GNN corrections using weighted approach.
        
        Mathematical formulation:
        Final_SVI = w * IDM_baseline + (1-w) * (IDM_baseline + α * GNN_correction)
        
        Where w is idm_weight and α scales corrections to maintain constraint.
        """
        
        # Scale corrections to prevent extreme values
        scaled_corrections = gnn_corrections * self.config.gnn_correction_scale
        
        # Apply corrections with weighting
        combined = (self.config.idm_weight * idm_baseline + 
                   (1 - self.config.idm_weight) * (idm_baseline + scaled_corrections))
        
        # Clip to valid SVI range [0, 1]
        combined = np.clip(combined, 0, 1)
        
        return combined
    
    def _enforce_tract_constraint(self, predictions: np.ndarray, 
                                 target_mean: float) -> np.ndarray:
        """
        Adjust predictions to exactly satisfy tract-level constraint.
        Uses multiplicative scaling to preserve relative patterns.
        """
        current_mean = np.mean(predictions)
        
        if abs(current_mean - target_mean) < self.config.constraint_tolerance:
            return predictions
        
        # Use multiplicative adjustment to preserve relative patterns
        if current_mean > 0:
            scale_factor = target_mean / current_mean
            adjusted = predictions * scale_factor
        else:
            # Fallback to additive adjustment if mean is zero
            adjusted = predictions + (target_mean - current_mean)
        
        # Ensure still in valid range
        adjusted = np.clip(adjusted, 0, 1)
        
        # Final check and minor adjustment if needed
        final_mean = np.mean(adjusted)
        if abs(final_mean - target_mean) > self.config.constraint_tolerance:
            # Small additive correction
            adjusted = adjusted + (target_mean - final_mean)
            adjusted = np.clip(adjusted, 0, 1)
        
        return adjusted
    
    def _validate_spatial_variation(self, predictions: np.ndarray):
        """
        Check that predictions have sufficient spatial variation.
        Warns if coefficient of variation is too low (indicating collapse).
        """
        cv = np.std(predictions) / (np.mean(predictions) + 1e-8)
        
        if cv < self.config.min_variation_cv:
            print(f"⚠️ Warning: Low spatial variation detected (CV={cv:.3f})")
            print(f"   Target minimum CV: {self.config.min_variation_cv}")
            print("   Consider adjusting GNN training or correction scaling")


class AccessibilityGNNCorrector(nn.Module):
    """
    GNN model that learns accessibility-based corrections to IDM baseline.
    
    Key difference from original: This outputs CORRECTIONS, not full SPDE parameters.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        # Feature processing layers
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Graph convolution layers (using PyTorch Geometric)
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
        
        # Output single correction value per node
        self.correction_head = nn.Linear(hidden_dim // 2, 1)
        
        # Activation and normalization
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Learned scaling parameter for corrections
        self.correction_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                idm_baseline: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass computing accessibility corrections.
        
        Parameters:
        -----------
        x : torch.Tensor
            Node features [num_nodes, input_dim]
        edge_index : torch.Tensor
            Graph connectivity [2, num_edges]
        idm_baseline : torch.Tensor, optional
            IDM baseline predictions to inform corrections
            
        Returns:
        --------
        torch.Tensor
            Accessibility corrections [num_nodes, 1]
        """
        
        # Initial projection
        h = self.input_projection(x)
        h = self.relu(h)
        
        # Graph convolutions with residual connections
        h1 = self.conv1(h, edge_index)
        h1 = self.batch_norm1(h1)
        h1 = self.relu(h1)
        h1 = self.dropout(h1)
        
        h2 = self.conv2(h1, edge_index)
        h2 = self.batch_norm2(h2)
        h2 = self.relu(h2)
        h2 = h2 + h1  # Residual connection
        
        h3 = self.conv3(h2, edge_index)
        h3 = self.relu(h3)
        
        # Generate corrections (centered around zero)
        corrections = self.correction_head(h3)
        corrections = torch.tanh(corrections) * self.correction_scale
        
        return corrections
    
    def compute_corrections(self, road_network, addresses: pd.DataFrame,
                          idm_baseline: np.ndarray, 
                          nlcd_features: pd.DataFrame) -> np.ndarray:
        """
        Compute accessibility corrections for given addresses.
        
        This is the main interface method called by HybridDisaggregator.
        """
        # Convert inputs to graph format
        from ..models.gnn import prepare_graph_data_with_nlcd
        graph_data, _ = prepare_graph_data_with_nlcd(
            road_network, nlcd_features, addresses
        )
        
        # Add IDM baseline as additional feature if available
        if idm_baseline is not None:
            baseline_tensor = torch.tensor(idm_baseline, dtype=torch.float32).unsqueeze(1)
            graph_data.x = torch.cat([graph_data.x, baseline_tensor], dim=1)
        
        # Forward pass
        self.eval()
        with torch.no_grad():
            corrections = self.forward(graph_data.x, graph_data.edge_index)
            corrections = corrections.squeeze().numpy()
        
        # Ensure corrections have zero mean (mass-preserving)
        corrections = corrections - np.mean(corrections)
        
        return corrections


class SpatialIntegrator:
    """
    Whittle-Matérn spatial integration for uncertainty quantification.
    
    This class interfaces with MetricGraph to provide spatial smoothing
    and uncertainty estimates.
    """
    
    def __init__(self, metricgraph_interface):
        self.mg_interface = metricgraph_interface
    
    def integrate(self, initial_predictions: np.ndarray,
                 addresses: pd.DataFrame,
                 tract_constraint: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply spatial integration with Whittle-Matérn fields.
        
        Returns:
        --------
        Tuple of (smoothed_predictions, uncertainty_estimates)
        """
        
        # This would interface with the MetricGraph R package
        # For now, return a simple spatial smoothing
        
        # Apply local smoothing (simplified version)
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(initial_predictions, sigma=1.0)
        
        # Ensure constraint is maintained
        smoothed = smoothed * (tract_constraint / np.mean(smoothed))
        
        # Simple uncertainty based on local variation
        uncertainty = np.abs(initial_predictions - smoothed) * 0.5 + 0.02
        
        return smoothed, uncertainty