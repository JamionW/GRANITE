"""
Graph Neural Network models for GRANITE framework

This module implements GNN architectures for learning SPDE parameters
from road network structure.
"""
# Standard library imports
from typing import Tuple, Dict, Optional

# Third-party imports
import numpy as np
import pandas as pd
import networkx as nx

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch Geometric imports
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class AccessibilityGNNCorrector(nn.Module):
    """
    GNN model that learns accessibility-based corrections to IDM baseline.
    
    Key difference from traditional GNN: This outputs CORRECTIONS, not full SPDE parameters.
    Corrections are bounded to prevent unrealistic adjustments.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, max_correction: float = 0.15):
        super().__init__()
        
        self.max_correction = max_correction
        
        # Feature processing layers
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Graph convolution layers
        try:
            from torch_geometric.nn import GCNConv
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
        except ImportError:
            # Fallback to simple linear layers if PyTorch Geometric not available
            self.conv1 = nn.Linear(hidden_dim, hidden_dim)
            self.conv2 = nn.Linear(hidden_dim, hidden_dim)
            self.conv3 = nn.Linear(hidden_dim, hidden_dim // 2)
            self._use_pyg = False
        else:
            self._use_pyg = True
            
        # Accessibility-specific layers
        self.accessibility_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2, num_heads=4, batch_first=True
        )
        
        # Output single correction value per node
        self.correction_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Normalization and activation
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                idm_baseline: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass computing accessibility corrections.
        
        Parameters:
        -----------
        x : torch.Tensor [num_nodes, input_dim]
            Node features (network topology + NLCD + optionally IDM baseline)
        edge_index : torch.Tensor [2, num_edges] 
            Graph connectivity
        idm_baseline : torch.Tensor [num_nodes, 1], optional
            IDM baseline predictions to inform corrections
            
        Returns:
        --------
        torch.Tensor [num_nodes, 1]
            Accessibility corrections bounded to [-max_correction, +max_correction]
        """

        if idm_baseline is not None:
            num_addresses = idm_baseline.shape[0]
        else:
            num_addresses = 2394  # Debug hardcode for now 
        
        num_addresses = idm_baseline.shape[0] if idm_baseline is not None else x.shape[0]

        # Initial feature projection
        h = self.input_projection(x)
        h = self.relu(h)
        
        # Graph convolutions with residual connections
        if self._use_pyg:
            # PyTorch Geometric path
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
        else:
            # Fallback for simpler graphs (no edge_index needed)
            h1 = self.conv1(h)
            h1 = self.batch_norm1(h1)
            h1 = self.relu(h1)
            
            h2 = self.conv2(h1)
            h2 = self.batch_norm2(h2)
            h2 = self.relu(h2) + h1
            
            h3 = self.conv3(h2)
            h3 = self.relu(h3)
        
        # Accessibility attention mechanism
        h3_expanded = h3.unsqueeze(0)  # Add batch dimension
        attended_features, _ = self.accessibility_attention(
            h3_expanded, h3_expanded, h3_expanded
        )
        h3 = attended_features.squeeze(0)  # Remove batch dimension
        
        # Generate corrections for ALL graph nodes
        corrections = self.correction_head(h3)  # [3098, 1]

        # FIXED: Return only corrections for address locations
        address_corrections = corrections[:num_addresses]  # [2394, 1] 

        # Bound corrections to reasonable range using tanh
        address_corrections = torch.tanh(address_corrections) * self.max_correction

        # Ensure corrections are mean-centered (mass preserving)
        address_corrections = address_corrections - torch.mean(address_corrections)

        # Generate corrections for ALL graph nodes
        all_corrections = self.correction_head(h3)  # [3098, 1]
        
        # SAFE BOUNDS CHECKING:
        if num_addresses > all_corrections.shape[0]:
            num_addresses = all_corrections.shape[0]
        
        address_corrections = all_corrections[:num_addresses]  # SAFE
        
        # Bound corrections to reasonable range using tanh
        address_corrections = torch.tanh(address_corrections) * self.max_correction
        
        # Ensure corrections are mean-centered (mass preserving)
        address_corrections = address_corrections - torch.mean(address_corrections)

        return address_corrections

class HybridCorrectionTrainer:
    """
    IMPROVED trainer for the accessibility GNN corrector.
    
    Key improvements:
    1. Restored spatial_loss weight (1.0)
    2. Reduced feature_utilization weight (0.1)  
    3. Added variance preservation term to prevent collapse
    4. Soft tract constraints instead of hard constraints
    5. Explicit coefficient of variation target (~20% of mean)
    """
    
    def __init__(self, model: AccessibilityGNNCorrector, config: dict = None):
        self.model = model
        self.config = config or {}
        
        # REBALANCED loss function weights (from your research analysis)
        self.loss_weights = {
            'spatial_weight': self.config.get('spatial_weight', 0.2),            # LOWER
            'smoothness_weight': self.config.get('smoothness_weight', 0.2),      # LOWER
            'diversity_weight': self.config.get('diversity_weight', 30.0),       # MUCH HIGHER
            'variation_weight': self.config.get('variation_weight', 20.0),       # MUCH HIGHER
            'constraint_weight': self.config.get('constraint_weight', 0.002),    # MUCH LOWER
            'feature_weight': self.config.get('feature_weight', 0.0001),         # MUCH LOWER
            'variance_preservation_weight': self.config.get('variance_preservation_weight', 75.0)  # MUCH HIGHER
        }
        
        # Target coefficient of variation (~40% of mean)
        self.target_cv = self.config.get('target_cv', 0.40)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
    def train_corrections(self, graph_data: object, idm_baseline: np.ndarray,
                         tract_svi: float, epochs: int = 100, 
                         verbose: bool = False) -> dict:
        """
        Train the GNN with improved loss function to prevent parameter collapse.
        """
        
        self.model.train()
        
        # Convert inputs to tensors and handle size mismatch
        idm_tensor = torch.tensor(idm_baseline, dtype=torch.float32)
        target_mean = torch.tensor(tract_svi, dtype=torch.float32)
        
        loss_history = []
        correction_history = []
        variance_history = []  # Track parameter variance over training

        last_training_corrections = None
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            corrections = self.model(
                graph_data.x, graph_data.edge_index, idm_tensor
            ).squeeze()
            
            # SAVE the corrections from the LAST epoch (bypass eval mode issue)
            if epoch == epochs - 1:  # Last epoch
                last_training_corrections = corrections.detach().clone()

            # Ensure corrections match IDM size (model should handle this internally now)
            if corrections.shape[0] != idm_tensor.shape[0]:
                corrections = corrections[:idm_tensor.shape[0]]
            
            # Compute enhanced predictions
            enhanced_predictions = idm_tensor + corrections
            
            # IMPROVED multi-component loss function
            total_loss, loss_components = self._compute_improved_loss(
                corrections, enhanced_predictions, target_mean, graph_data, idm_tensor
            )

            if verbose and epoch % 5 == 0:  # More frequent logging
                pred_std = torch.std(enhanced_predictions).item()
                pred_mean = torch.mean(enhanced_predictions).item()
                corr_std = torch.std(corrections).item()
                corr_var = torch.var(corrections).item()
                constraint_error = abs(pred_mean - target_mean.item())
                current_cv = pred_std / (pred_mean + 1e-8)
      
                # Emergency early stopping
                if corr_var < 1e-8 and epoch > 10:
                    print(f"🛑 EMERGENCY STOP: Complete parameter collapse at epoch {epoch}!")
                    break
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping to prevent instability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Enhanced monitoring
            loss_history.append(total_loss.item())
            correction_history.append(corrections.detach().numpy().copy())
            
            # Track parameter variance to detect collapse
            param_variance = torch.var(corrections).item()
            variance_history.append(param_variance)
            
            if verbose and epoch % 10 == 0:  # More frequent logging
                pred_std = torch.std(enhanced_predictions).item()
                pred_mean = torch.mean(enhanced_predictions).item()
                corr_std = torch.std(corrections).item()
                constraint_error = abs(pred_mean - target_mean.item())
                current_cv = pred_std / (pred_mean + 1e-8)
      
                # Warn about parameter collapse
                if param_variance < 1e-6:
                    print("  ⚠️  WARNING: Parameter variance very low - possible collapse!")
        
        self.model.eval()

        with torch.no_grad():
            # Get final corrections
            final_corrections = self.model(
                graph_data.x, graph_data.edge_index, idm_tensor
            ).squeeze()
            
            if last_training_corrections is not None:
                final_corrections = last_training_corrections
            else:
                self.model.eval()
                with torch.no_grad():
                    final_corrections = self.model(graph_data.x, graph_data.edge_index, idm_tensor).squeeze()

            
            # Check if size mismatch
            if final_corrections.shape[0] != idm_tensor.shape[0]:
                print(f"   ⚠️  Size mismatch: corrections {final_corrections.shape[0]} vs idm {idm_tensor.shape[0]}")
                final_corrections = final_corrections[:idm_tensor.shape[0]]
                print(f"   Truncated to: {final_corrections.shape}")
                print(f"   After truncation: std={final_corrections.std():.6f}")
                
            final_predictions = idm_tensor + final_corrections

            if epoch % 3 == 0:  # Every 3 epochs
                corr_var = torch.var(corrections).item()
                corr_range = corrections.max().item() - corrections.min().item()
                pred_cv = torch.std(enhanced_predictions).item() / torch.mean(enhanced_predictions).item()

                if corr_var < 1e-6:
                    print(f"   🚨 SEVERE COLLAPSE! Variance = {corr_var:.2e}")
            
            # Truncate back to address count for output
            original_address_count = len(idm_baseline)
            final_corrections = final_corrections[:original_address_count]
            final_predictions = final_predictions[:original_address_count]
            
        return {
            'success': True,
            'final_corrections': final_corrections.numpy(),
            'final_predictions': final_predictions.numpy(),
            'loss_history': loss_history,
            'correction_history': correction_history,
            'variance_history': variance_history,
            'final_metrics': {
                'prediction_std': torch.std(final_predictions).item(),
                'prediction_cv': torch.std(final_predictions).item() / (torch.mean(final_predictions).item() + 1e-8),
                'correction_std': torch.std(final_corrections).item(),
                'correction_variance': torch.var(final_corrections).item(),
                'constraint_error': abs(torch.mean(final_predictions) - target_mean).item(),
                'mean_correction_magnitude': torch.mean(torch.abs(final_corrections)).item(),
                'parameter_collapse_detected': torch.var(final_corrections).item() < 1e-6
            }
        }
    
    def _compute_improved_loss(self, corrections: torch.Tensor, 
                              enhanced_predictions: torch.Tensor,
                              target_mean: torch.Tensor,
                              graph_data: object,
                              idm_baseline: torch.Tensor) -> tuple:
        """
        IMPROVED loss function that prevents trivial solutions.
        
        Key changes from original:
        1. Added spatial_loss with weight 1.0
        2. Added variance preservation term (weight 3.0)
        3. Reduced feature_weight to 0.1 
        4. Increased diversity_weight to 5.0
        5. Soft constraint instead of hard constraint
        """
        
        loss_components = {}
        
        # 1. SOFT tract constraint (reduced weight, allows learning)
        constraint_loss = F.mse_loss(
            torch.mean(enhanced_predictions), target_mean
        )
        loss_components['constraint'] = constraint_loss.item()
        
        # 2. SPATIAL SMOOTHNESS LOSS (restored with weight 1.0)
        spatial_loss = self._compute_spatial_loss(corrections, graph_data.edge_index)
        loss_components['spatial'] = spatial_loss.item()
        
        # 3. VARIANCE PRESERVATION (NEW - prevents parameter collapse)
        variance_loss = self._compute_variance_preservation_loss(
            corrections, enhanced_predictions, target_mean
        )
        loss_components['variance_preservation'] = variance_loss.item()
        
        # 4. COEFFICIENT OF VARIATION TARGET (improved from variation_loss)
        cv_loss = self._compute_cv_target_loss(enhanced_predictions)
        loss_components['cv_target'] = cv_loss.item()
        
        # 5. DIVERSITY (increased weight - prevents uniform predictions)
        diversity_loss = self._compute_diversity_loss(enhanced_predictions)
        loss_components['diversity'] = diversity_loss.item()
        
        # 6. GENTLE feature utilization (reduced weight from 10.0 to 0.1)
        feature_loss = self._compute_feature_utilization_loss(corrections, graph_data.x)
        loss_components['feature'] = feature_loss.item()
        
        # REBALANCED total loss
        total_loss = (
            self.loss_weights['constraint_weight'] * constraint_loss +
            self.loss_weights['spatial_weight'] * spatial_loss +           # RESTORED
            self.loss_weights['variance_preservation_weight'] * variance_loss +  # NEW
            self.loss_weights['variation_weight'] * cv_loss +
            self.loss_weights['diversity_weight'] * diversity_loss +        # INCREASED
            self.loss_weights['feature_weight'] * feature_loss              # REDUCED
        )
        
        return total_loss, loss_components
    
    def _compute_spatial_loss(self, corrections: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        FIXED: Safe spatial loss that only uses address-to-address edges.
        
        The issue: corrections[2394], but edge_index references nodes up to 3098.
        The fix: Only use edges where both endpoints are < len(corrections).
        """
        if edge_index.shape[1] == 0:
            return torch.tensor(0.0)
        
        # CRITICAL FIX: Filter to address-only edges
        num_addresses = corrections.shape[0]  # 2394
        valid_edge_mask = (edge_index[0] < num_addresses) & (edge_index[1] < num_addresses)
        
        if not valid_edge_mask.any():
            return torch.tensor(0.0)
        
        # Use only valid edges
        valid_edge_index = edge_index[:, valid_edge_mask]
        
        # Safe indexing - all indices guaranteed < num_addresses
        source_corrections = corrections[valid_edge_index[0]]
        target_corrections = corrections[valid_edge_index[1]]
        
        # Standard spatial smoothness
        smoothness = F.mse_loss(source_corrections, target_corrections)
        
        # Add uniformity penalty
        uniformity_penalty = torch.exp(-torch.var(corrections) * 100) * 0.1
        
        return smoothness + uniformity_penalty
    
    def _compute_variance_preservation_loss(self, corrections: torch.Tensor,
                                        predictions: torch.Tensor,
                                        target_mean: torch.Tensor) -> torch.Tensor:
        """
        MUCH MORE AGGRESSIVE variance preservation to prevent parameter collapse.
        """
        # Target: corrections should have meaningful variance
        correction_variance = torch.var(corrections)
        
        # SUPER STRONG penalty for low variance (exponential punishment)
        collapse_penalty = torch.exp(-correction_variance * 50000)  # MUCH stronger
        
        # Target: predictions should have realistic coefficient of variation
        pred_mean = torch.mean(predictions)
        pred_std = torch.std(predictions)
        
        if pred_mean > 0:
            current_cv = pred_std / pred_mean
            cv_penalty = (current_cv - self.target_cv) ** 2
        else:
            cv_penalty = torch.tensor(0.0)
        
        # STRONG penalty for corrections that are too small in magnitude
        magnitude_penalty = torch.exp(-torch.mean(torch.abs(corrections)) * 1000)
        
        # Combine penalties with aggressive weighting
        return 20.0 * collapse_penalty + 5.0 * cv_penalty + 15.0 * magnitude_penalty

    
    def _compute_cv_target_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Target coefficient of variation around 20% of mean.
        """
        pred_mean = torch.mean(predictions)
        pred_std = torch.std(predictions)
        
        if pred_mean > 0:
            current_cv = pred_std / pred_mean
            return F.mse_loss(current_cv, torch.tensor(self.target_cv))
        else:
            return torch.tensor(0.0)
    
    def _compute_diversity_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        MUCH STRONGER diversity loss to prevent uniform predictions
        """
        # Much stronger penalties for low diversity
        std_loss = torch.exp(-torch.std(predictions) * 100)  # Increased from 10
        var_loss = torch.exp(-torch.var(predictions) * 1000)  # Increased from 100
        
        return 10.0 * (std_loss + var_loss)  # Much higher multiplier

    
    def _compute_feature_utilization_loss(self, corrections: torch.Tensor,
                                        features: torch.Tensor) -> torch.Tensor:
        """
        REDUCED weight - gentle encouragement to use available features
        """
        if features.shape[1] == 0:
            return torch.tensor(0.0)
            
        # Much gentler feature utilization than before
        feature_correlations = []
        for i in range(min(features.shape[1], 5)):  # Limit to avoid dominance
            try:
                corr = torch.corrcoef(torch.stack([
                    corrections.flatten(), 
                    features[:, i].flatten()
                ]))[0, 1]
                if not torch.isnan(corr):
                    feature_correlations.append(torch.abs(corr))
            except:
                continue
        
        if feature_correlations:
            mean_correlation = torch.mean(torch.stack(feature_correlations))
            return -mean_correlation * 0.1  # Much reduced impact
        else:
            return torch.tensor(0.0)

# Convenience function for easy use in pipeline
def create_accessibility_corrector(input_dim: int, hidden_dim: int = 64, 
                                 max_correction: float = 0.15) -> AccessibilityGNNCorrector:
    """
    Convenience function to create AccessibilityGNNCorrector.
    Can be used in pipeline.py for easy model creation.
    """
    return AccessibilityGNNCorrector(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        max_correction=max_correction
    )

class SPDEParameterGNN(nn.Module):
    """GNN with explicit variance preservation"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 3):
        super(SPDEParameterGNN, self).__init__()
        
        # Residual connections to preserve input variance
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Use BatchNorm to maintain variance
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Direct feature bypass to preserve input signal
        self.feature_bypass = nn.Linear(input_dim, output_dim)
        
        # Output heads
        self.param_head = nn.Linear(hidden_dim + output_dim, output_dim)
        
        # Minimal dropout
        self.dropout = nn.Dropout(0.05)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Save original features for bypass
        original_features = x
        
        # Project input
        x = self.input_projection(x)
        identity = x  # For residual
        
        # Conv block 1 with residual
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x + identity  # Residual connection
        
        # Conv block 2 with residual
        identity = x
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x + identity
        
        # Conv block 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Combine with bypassed features
        bypass_params = self.feature_bypass(original_features)
        combined = torch.cat([x, bypass_params], dim=1)
        
        # Final parameters with no compression
        params = self.param_head(combined)

        kappa = params[:, 0] * 5.0 + 2.5    # Kappa in [2.5-5, 2.5+5] = [-2.5, 7.5]
        alpha = params[:, 1] * 1.0           
        tau = params[:, 2] * 1.0 + 0.5       # Tau in [0.5-1, 0.5+1] = [-0.5, 1.5]

        # But clamp to valid ranges
        kappa = torch.clamp(kappa, min=0.2, max=10.0)  
        tau = torch.clamp(tau, min=0.05, max=2.0) 
        
        params = torch.stack([kappa, alpha, tau], dim=1)
        
        return params

def safe_feature_normalization_vectorized(node_features):
    """
    Fully vectorized normalization
    """
    # Compute min/max across all features at once
    col_mins = torch.min(node_features, dim=0, keepdim=True)[0]
    col_maxs = torch.max(node_features, dim=0, keepdim=True)[0]
    
    # Avoid division by zero
    col_ranges = col_maxs - col_mins
    col_ranges = torch.where(col_ranges > 0, col_ranges, torch.ones_like(col_ranges))
    
    # Vectorized normalization (creates new tensor)
    normalized_features = (node_features - col_mins) / col_ranges
    
    return normalized_features

def prepare_graph_data_with_nlcd(road_network: nx.Graph, 
                                nlcd_features: pd.DataFrame,
                                addresses: pd.DataFrame = None) -> Tuple[Data, Dict]:
    """
    Prepare graph data for GNN with enhanced NLCD-based features
    """
    import numpy as np
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import cdist
    
    nodes = list(road_network.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Build spatial index for NLCD feature lookup
    feature_tree = None
    feature_coords = None
    
    if len(nlcd_features) > 0 and addresses is not None and len(addresses) > 0:
        # Match nlcd_features to addresses by address_id
        if 'address_id' in nlcd_features.columns and 'address_id' in addresses.columns:
            features_with_coords = nlcd_features.merge(
                addresses[['address_id', 'geometry']], 
                on='address_id', 
                how='left'
            )
            
            valid_features = features_with_coords.dropna(subset=['geometry'])
            if len(valid_features) > 0:
                feature_coords = np.array([
                    [geom.x, geom.y] for geom in valid_features.geometry
                ])
                feature_tree = cKDTree(feature_coords)
    
    # Extract enhanced node features
    node_features = []
    successful_lookups = 0
    
    # Pre-compute spatial features for efficiency
    water_classes = [11, 12, 90, 95]
    forest_classes = [41, 42, 43]
    developed_classes = [21, 22, 23, 24]
    
    for node in nodes:
        node_x, node_y = node[0], node[1]
        
        # Default feature values
        development_intensity = 0.5
        svi_coefficient = 0.3
        land_cover_diversity = 0.0
        development_gradient = 0.0  
        distance_to_water = 1.0
        distance_to_forest = 1.0
        normalized_nlcd_class = 0.23
        
        # Try spatial lookup if available
        if feature_tree is not None and len(nlcd_features) > 0:
            try:
                # Find nearest address
                distance, nearest_idx = feature_tree.query([node_x, node_y])
                
                if distance < 1000:  # Within 1km
                    # Get corresponding feature row
                    feature_row = nlcd_features.iloc[nearest_idx]
                    
                    # Extract basic features
                    development_intensity = feature_row.get('development_intensity', 0.5)
                    svi_coefficient = feature_row.get('svi_coefficient', 
                                                   feature_row.get('svi_vulnerability_coeff', 0.3))
                    
                    # 1. Land Cover Diversity (Shannon entropy in neighborhood)
                    land_cover_diversity = calculate_land_cover_diversity(
                        node_x, node_y, feature_tree, feature_coords, 
                        nlcd_features, radius=500
                    )
                    
                    # 2. Development Gradient (rate of development change)
                    development_gradient = calculate_development_gradient(
                        node_x, node_y, feature_tree, feature_coords,
                        nlcd_features, radius=300
                    )
                    
                    # 3. Distance to Water Features  
                    distance_to_water = calculate_distance_to_feature_class(
                        node_x, node_y, feature_tree, feature_coords,
                        nlcd_features, target_classes=water_classes
                    )
                    
                    # 4. Distance to Forest Features
                    distance_to_forest = calculate_distance_to_feature_class(
                        node_x, node_y, feature_tree, feature_coords,
                        nlcd_features, target_classes=forest_classes
                    )
                    
                    # Normalize NLCD class
                    nlcd_class = feature_row.get('nlcd_class', 22)
                    normalized_nlcd_class = nlcd_class / 95.0
                    
                    successful_lookups += 1
                    
            except Exception as e:
                print(f"Error in spatial lookup for node {node}: {e}")
        
        # Construct enhanced feature vector
        features = [
            # Core NLCD-derived features (both GNN and IDM can use)
            development_intensity,     # 0.0-1.0 based on NLCD class
            svi_coefficient,          # 0.0-1.5 based on land cover vulnerability
            development_gradient,     # 0.0-1.0 rate of development change  
            
            # Network topology (GNN-specific)
            min(road_network.degree(node), 10) / 10.0,  # Normalized degree
        ]
        
        node_features.append(features)
    
    # Convert to tensor and normalize
    node_features = torch.FloatTensor(node_features)
    node_features = safe_feature_normalization_vectorized(node_features)
    
    # Build edges (unchanged)
    edge_list = []
    edge_attrs = []
    
    for u, v, data in road_network.edges(data=True):
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        
        edge_list.append([u_idx, v_idx])
        edge_list.append([v_idx, u_idx])
        
        length = data.get('length', 1.0)
        edge_attrs.append([length])
        edge_attrs.append([length])
    
    # Create PyTorch Geometric data object
    x = node_features
    edge_index = torch.LongTensor(edge_list).t().contiguous()
    edge_attr = torch.FloatTensor(edge_attrs) if edge_attrs else None
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data, node_to_idx

def prepare_graph_data_topological(road_network: nx.Graph) -> Tuple[Data, Dict]:
    """
    Prepare graph data for GNN from road network
    
    Parameters:
    -----------
    road_network : nx.Graph
        Road network graph
        
    Returns:
    --------
    Tuple[Data, Dict]
        PyTorch Geometric Data object and node ID mapping
    """
    # Create node mapping
    nodes = list(road_network.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Extract node features
    node_features = []
    
    for node in nodes:
        # Compute node features
        features = [
            road_network.degree(node),  # Degree
            nx.closeness_centrality(road_network, node) if len(nodes) > 1 else 0.5,  # Closeness
            node[0],  # X coordinate (normalized later)
            node[1],  # Y coordinate (normalized later)
            len(road_network[node]),  # Local connectivity
            np.mean([1.0 for neighbor in road_network[node]]) if road_network[node] else 0.0,  # Simple neighbor count
            min(road_network.degree(node), 10) / 10.0,  # Normalized degree (0-1)
        ]

        node_features.append(features)
    
    # Convert to PyTorch tensor, then normalize
    node_features = torch.FloatTensor(node_features)
    node_features = safe_feature_normalization_vectorized(node_features)
    
    # Extract edges
    edge_list = []
    edge_attrs = []
    
    for u, v, data in road_network.edges(data=True):
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        
        # Add both directions for undirected graph
        edge_list.append([u_idx, v_idx])
        edge_list.append([v_idx, u_idx])
        
        # Edge attributes (e.g., length)
        length = data.get('length', 1.0)
        edge_attrs.append([length])
        edge_attrs.append([length])
    
    # Convert to tensors
    x = node_features
    edge_index = torch.LongTensor(edge_list).t().contiguous()
    edge_attr = torch.FloatTensor(edge_attrs) if edge_attrs else None
    
    # Create PyTorch Geometric data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data, node_to_idx

def create_gnn_model(input_dim: int = 5, hidden_dim: int = 128, 
                    output_dim: int = 3) -> nn.Module:
    """
    Create GNN model for SPDE parameter learning
    
    Parameters:
    -----------
    input_dim : int
        Number of input features
    hidden_dim : int
        Hidden layer dimension
    output_dim : int
        Output dimension (3 for SPDE parameters)
        
    Returns:
    --------
    nn.Module
        GNN model
    """
    return SPDEParameterGNN(input_dim, hidden_dim, output_dim)

def calculate_land_cover_diversity(node_x, node_y, feature_tree, feature_coords, 
                                  nlcd_features, radius=500):
    """Simplified land cover diversity calculation"""
    try:
        # Check if we have the required data
        if feature_tree is None or 'nlcd_class' not in nlcd_features.columns:
            return 0.0
            
        # Find neighbors within radius  
        neighbor_indices = feature_tree.query_ball_point([node_x, node_y], radius)
        
        if len(neighbor_indices) < 2:
            return 0.0
        
        # Get NLCD classes for neighbors
        neighbor_classes = nlcd_features.iloc[neighbor_indices]['nlcd_class'].values
        unique_classes = len(set(neighbor_classes))
        
        # Simple diversity measure: number of unique classes / max possible
        return min(unique_classes / 4.0, 1.0)  # Normalize by 4 expected classes
        
    except Exception as e:
        print(f"Land cover diversity error: {e}")
        return 0.0


def calculate_development_gradient(node_x, node_y, feature_tree, feature_coords,
                                 nlcd_features, radius=300):
    """Calculate rate of development intensity change in neighborhood"""
    try:
        # Find neighbors within radius
        neighbor_indices = feature_tree.query_ball_point([node_x, node_y], radius)
        
        if len(neighbor_indices) < 2:
            return 0.0
        
        # Get development intensities
        neighbor_features = nlcd_features.iloc[neighbor_indices]
        dev_intensities = neighbor_features['development_intensity'].values
        
        # Calculate distances from node
        neighbor_coords = feature_coords[neighbor_indices]
        distances = np.sqrt(np.sum((neighbor_coords - [node_x, node_y])**2, axis=1))
        
        # Calculate gradient (correlation between distance and development)
        if len(set(distances)) > 1 and len(set(dev_intensities)) > 1:
            correlation = np.corrcoef(distances, dev_intensities)[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.0
        
        return 0.0
        
    except:
        return 0.0


def calculate_distance_to_feature_class(node_x, node_y, feature_tree, feature_coords,
                                       nlcd_features, target_classes):
    """Simplified distance calculation"""
    try:
        if feature_tree is None or 'nlcd_class' not in nlcd_features.columns:
            return 1.0
            
        # Check if any target classes exist in the data
        has_targets = nlcd_features['nlcd_class'].isin(target_classes).any()
        if not has_targets:
            return 1.0  # No target classes found
            
        # Find all addresses within reasonable distance
        all_indices = feature_tree.query_ball_point([node_x, node_y], 2000)  # 2km radius
        
        if len(all_indices) == 0:
            return 1.0
            
        # Check which ones have target classes
        for idx in all_indices:
            if idx < len(nlcd_features):
                nlcd_class = nlcd_features.iloc[idx]['nlcd_class']
                if nlcd_class in target_classes:
                    # Calculate distance to this target
                    target_coord = feature_coords[idx]
                    distance = np.sqrt((target_coord[0] - node_x)**2 + (target_coord[1] - node_y)**2)
                    return min(distance / 2000.0, 1.0)  # Normalize by 2km
                    
        return 1.0  # No targets found
        
    except Exception as e:
        print(f"Distance calculation error: {e}")
        return 1.0