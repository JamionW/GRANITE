"""
granite/models/correction_training.py

Training module for GNN that learns accessibility corrections to IDM baseline.
Key innovation: Loss functions that encourage meaningful corrections without
collapsing to trivial solutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime


class CorrectionLoss(nn.Module):
    """
    Custom loss function for learning accessibility corrections.
    
    Balances multiple objectives:
    1. Corrections should be meaningful (not zero everywhere)
    2. Corrections should be spatially smooth
    3. Combined predictions should maintain realistic variation
    4. Network features should influence corrections
    """
    
    def __init__(self, config: Dict = None):
        super().__init__()
        
        # Default weights for loss components
        default_config = {
            'smoothness_weight': 1.0,      # Spatial smoothness of corrections
            'diversity_weight': 0.5,        # Encourage non-zero corrections
            'variation_weight': 0.3,        # Maintain spatial variation
            'feature_weight': 0.2,          # Feature utilization
            'constraint_weight': 0.1        # Soft constraint on mean
        }
        
        self.config = config if config else default_config
    
    def forward(self, corrections: torch.Tensor,
               edge_index: torch.Tensor,
               node_features: torch.Tensor,
               idm_baseline: torch.Tensor,
               tract_svi: float) -> Tuple[torch.Tensor, Dict]:
        """
        Compute multi-component loss for accessibility corrections.
        
        Parameters:
        -----------
        corrections : torch.Tensor [N, 1]
            GNN-predicted corrections
        edge_index : torch.Tensor [2, E]
            Graph edges
        node_features : torch.Tensor [N, F]
            Input features to GNN
        idm_baseline : torch.Tensor [N, 1]
            IDM baseline predictions
        tract_svi : float
            Target tract-level mean
            
        Returns:
        --------
        Tuple of (total_loss, loss_components_dict)
        """
        
        # Ensure corrections have variation (prevent collapse to zero)
        diversity_loss = self._diversity_loss(corrections)
        
        # Spatial smoothness (connected nodes should have similar corrections)
        smoothness_loss = self._spatial_smoothness_loss(corrections, edge_index)
        
        # Combined predictions should maintain variation
        combined = idm_baseline + corrections
        variation_loss = self._variation_preservation_loss(combined, idm_baseline)
        
        # Feature utilization (corrections should depend on input features)
        feature_loss = self._feature_utilization_loss(corrections, node_features)
        
        # Soft constraint on tract mean (not too strict to allow learning)
        constraint_loss = self._soft_constraint_loss(combined, tract_svi)
        
        # Combine losses
        total_loss = (
            self.config['smoothness_weight'] * smoothness_loss +
            self.config['diversity_weight'] * diversity_loss +
            self.config['variation_weight'] * variation_loss +
            self.config['feature_weight'] * feature_loss +
            self.config['constraint_weight'] * constraint_loss
        )
        
        # Return components for monitoring
        components = {
            'total': total_loss.item(),
            'smoothness': smoothness_loss.item(),
            'diversity': diversity_loss.item(),
            'variation': variation_loss.item(),
            'feature': feature_loss.item(),
            'constraint': constraint_loss.item()
        }
        
        return total_loss, components
    
    def _diversity_loss(self, corrections: torch.Tensor) -> torch.Tensor:
        """
        Encourage non-zero corrections with meaningful variation.
        Penalizes both all-zero and constant corrections.
        """
        # Penalize low standard deviation
        std = torch.std(corrections)
        std_loss = torch.exp(-std * 10)  # Exponential penalty for low std
        
        # Penalize corrections being too small
        magnitude = torch.mean(torch.abs(corrections))
        magnitude_loss = torch.exp(-magnitude * 100)
        
        return std_loss + magnitude_loss
    
    def _spatial_smoothness_loss(self, corrections: torch.Tensor, 
                                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Encourage spatial smoothness in corrections.
        Connected nodes should have similar (but not identical) corrections.
        """
        row, col = edge_index
        
        # Difference between connected nodes
        edge_diff = corrections[row] - corrections[col]
        
        # L2 penalty on differences
        smoothness = torch.mean(edge_diff ** 2)
        
        # But don't make it too smooth (prevent constant)
        # Add small penalty if variance is too low
        var_penalty = torch.exp(-torch.var(corrections) * 50) * 0.1
        
        return smoothness + var_penalty
    
    def _variation_preservation_loss(self, combined: torch.Tensor,
                                    baseline: torch.Tensor) -> torch.Tensor:
        """
        Ensure combined predictions maintain realistic spatial variation.
        Should not collapse to constant or have extreme variation.
        """
        # Target coefficient of variation (std/mean)
        target_cv = 0.2  # 20% variation is realistic for SVI
        
        combined_cv = torch.std(combined) / (torch.mean(combined) + 1e-8)
        baseline_cv = torch.std(baseline) / (torch.mean(baseline) + 1e-8)
        
        # Penalize deviation from target CV
        cv_loss = (combined_cv - target_cv) ** 2
        
        # Also penalize if combined has much less variation than baseline
        relative_loss = torch.relu(baseline_cv - combined_cv) * 2
        
        return cv_loss + relative_loss
    
    def _feature_utilization_loss(self, corrections: torch.Tensor,
                                 features: torch.Tensor) -> torch.Tensor:
        """
        Encourage corrections to utilize input features.
        Prevents learning trivial solutions that ignore the graph structure.
        """
        # Compute correlation between features and corrections
        # Higher correlation means features are being used
        
        # Standardize for correlation computation
        corr_std = torch.std(corrections) + 1e-8
        feat_std = torch.std(features, dim=0) + 1e-8
        
        # Compute correlations with each feature
        correlations = []
        for i in range(features.shape[1]):
            corr = torch.abs(torch.mean(
                (corrections - torch.mean(corrections)) * 
                (features[:, i:i+1] - torch.mean(features[:, i]))
            ) / (corr_std * feat_std[i]))
            correlations.append(corr)
        
        # Want at least some features to be correlated
        max_correlation = torch.max(torch.stack(correlations))
        
        # Penalize low correlation
        return torch.exp(-max_correlation * 5)
    
    def _soft_constraint_loss(self, combined: torch.Tensor,
                             target_mean: float) -> torch.Tensor:
        """
        Soft constraint on tract-level mean.
        Not too strict to allow the model to learn patterns.
        """
        predicted_mean = torch.mean(combined)
        
        # Use Huber loss for robustness
        constraint_error = F.smooth_l1_loss(
            predicted_mean, 
            torch.tensor(target_mean, device=combined.device)
        )
        
        return constraint_error


class AccessibilityCorrectionTrainer:
    """
    Trainer for GNN learning accessibility corrections.
    """
    
    def __init__(self, model: nn.Module, config: Dict = None):
        self.model = model
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            patience=10,
            factor=0.5
        )
        
        # Initialize loss function
        self.loss_fn = CorrectionLoss(config=self.config.get('loss_config'))
        
        # Training history
        self.history = {
            'epoch': [],
            'total_loss': [],
            'loss_components': []
        }
    
    def train(self, graph_data, idm_baseline: np.ndarray,
             tract_svi: float, epochs: int = 100,
             verbose: bool = True) -> Dict:
        """
        Train the GNN to learn accessibility corrections.
        
        Parameters:
        -----------
        graph_data : torch_geometric.data.Data
            Graph structure with features
        idm_baseline : np.ndarray
            IDM baseline predictions
        tract_svi : float
            Target tract-level SVI
        epochs : int
            Number of training epochs
        verbose : bool
            Print training progress
            
        Returns:
        --------
        Dict with training history and final model state
        """
        
        # Move data to device
        graph_data = graph_data.to(self.device)
        idm_tensor = torch.tensor(idm_baseline, dtype=torch.float32).to(self.device)
        if idm_tensor.dim() == 1:
            idm_tensor = idm_tensor.unsqueeze(1)
        
        # Training loop
        self.model.train()
        
        for epoch in range(epochs):
            # Forward pass
            corrections = self.model(
                graph_data.x, 
                graph_data.edge_index,
                idm_baseline=idm_tensor
            )
            
            # Compute loss
            loss, components = self.loss_fn(
                corrections=corrections,
                edge_index=graph_data.edge_index,
                node_features=graph_data.x,
                idm_baseline=idm_tensor,
                tract_svi=tract_svi
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update learning rate
            self.scheduler.step(loss)
            
            # Store history
            self.history['epoch'].append(epoch)
            self.history['total_loss'].append(loss.item())
            self.history['loss_components'].append(components)
            
            # Print progress
            if verbose and epoch % 10 == 0:
                self._print_progress(epoch, loss.item(), components)
            
            # Early stopping check
            if self._check_convergence():
                if verbose:
                    print(f"\nConverged at epoch {epoch}")
                break
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            final_corrections = self.model(
                graph_data.x,
                graph_data.edge_index,
                idm_baseline=idm_tensor
            )
            final_corrections = final_corrections.cpu().numpy()
        
        # Compute final statistics
        final_stats = self._compute_correction_statistics(
            final_corrections, idm_baseline, tract_svi
        )
        
        return {
            'history': self.history,
            'final_corrections': final_corrections,
            'statistics': final_stats,
            'model_state': self.model.state_dict()
        }
    
    def _print_progress(self, epoch: int, loss: float, components: Dict):
        """Print training progress."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] Epoch {epoch:3d} | Loss: {loss:.4f} | "
              f"Smooth: {components['smoothness']:.3f} | "
              f"Divers: {components['diversity']:.3f} | "
              f"Var: {components['variation']:.3f}")
    
    def _check_convergence(self, window: int = 20, threshold: float = 1e-4):
        """Check if training has converged."""
        if len(self.history['total_loss']) < window:
            return False
        
        recent_losses = self.history['total_loss'][-window:]
        loss_std = np.std(recent_losses)
        
        return loss_std < threshold
    
    def _compute_correction_statistics(self, corrections: np.ndarray,
                                      baseline: np.ndarray,
                                      tract_svi: float) -> Dict:
        """Compute statistics about learned corrections."""
        combined = baseline + corrections
        
        return {
            'correction_mean': float(np.mean(corrections)),
            'correction_std': float(np.std(corrections)),
            'correction_min': float(np.min(corrections)),
            'correction_max': float(np.max(corrections)),
            'combined_mean': float(np.mean(combined)),
            'combined_std': float(np.std(combined)),
            'combined_cv': float(np.std(combined) / np.mean(combined)),
            'constraint_error': float(abs(np.mean(combined) - tract_svi)),
            'improvement_over_baseline': float(
                np.std(combined) / np.std(baseline)
            )
        }