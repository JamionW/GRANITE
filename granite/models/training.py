"""
Training utilities for GRANITE GNN models

This module provides training functionality for Graph Neural Networks
that learn accessibility features for MetricGraph integration.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from datetime import datetime
from typing import Dict, Tuple, Optional
import time

class AccessibilityTrainer:
    """Trainer class for GNN models learning SPDE parameters"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu', verbose=None, config=None):
        """
        Initialize trainer
        
        Parameters:
        -----------
        model : nn.Module
            GNN model to train
        device : str
            Device to use for training
        verbose : bool
            Enable verbose logging
        """
        self.model = model.to(device)
        self.device = device
        if config and 'processing' in config:
            self.verbose = config['processing'].get('verbose', False)
        else:
            self.verbose = verbose if verbose is not None else False
        self.training_history = {
            'loss': [],
            'spatial_loss': [],
            'reg_loss': []
        }
    
    def _log(self, message: str):
        """Log message with timestamp"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] Trainer: {message}")
    
    def spatial_smoothness_loss(self, params: torch.Tensor, edge_index: torch.Tensor, 
                               edge_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute spatial smoothness loss for SPDE parameters
        
        This encourages similar parameters for connected nodes, which is
        important for the Whittle-Matérn model's spatial coherence.
        
        Parameters:
        -----------
        params : torch.Tensor
            SPDE parameters [num_nodes, 3] (kappa, alpha, tau)
        edge_index : torch.Tensor
            Edge connectivity [2, num_edges]
        edge_weights : torch.Tensor, optional
            Edge weights (e.g., distances)
            
        Returns:
        --------
        torch.Tensor
            Spatial smoothness loss
        """
        # Get parameters at edge endpoints
        params_u = params[edge_index[0]]
        params_v = params[edge_index[1]]
        
        # Compute squared differences
        diff = params_u - params_v
        squared_diff = torch.sum(diff ** 2, dim=1)
        
        # Weight by edge lengths if provided
        if edge_weights is not None:
            # Shorter edges should have more similar parameters
            weights = 1.0 / (edge_weights + 0.1)
            loss = torch.mean(weights * squared_diff)
        else:
            loss = torch.mean(squared_diff)
        
        return loss
    
    def parameter_regularization(self, params: torch.Tensor) -> torch.Tensor:
        """
        Regularization to keep SPDE parameters in reasonable ranges
        
        Parameters:
        -----------
        params : torch.Tensor
            SPDE parameters [num_nodes, 3]
            
        Returns:
        --------
        torch.Tensor
            Regularization loss
        """
        # Extract individual parameters
        kappa = params[:, 0]  # Precision parameter
        alpha = params[:, 1]  # Smoothness parameter
        tau = params[:, 2]    # Nugget effect
        
        # L2 regularization
        reg_loss = torch.mean(params ** 2) * 0.01
        
        # Add penalties for extreme values
        # Kappa should be positive and not too large
        kappa_penalty = torch.mean(F.relu(kappa - 5.0) ** 2)
        
        # Alpha should be between 0 and 2 (typical range)
        alpha_penalty = torch.mean(F.relu(alpha - 2.0) ** 2) + torch.mean(F.relu(-alpha) ** 2)
        
        # Tau should be positive and small
        tau_penalty = torch.mean(F.relu(tau - 1.0) ** 2) + torch.mean(F.relu(-tau) ** 2)
        
        return reg_loss + kappa_penalty + alpha_penalty + tau_penalty
    
    def diversity_penalty(self, params: torch.Tensor) -> torch.Tensor:
        """
        Penalty to prevent all parameters from converging to same values
        """
        kappa = params[:, 0]  # κ variance
        alpha = params[:, 1]  # α variance  
        tau = params[:, 2]    # τ variance
        
        # Current variances
        kappa_var = torch.var(kappa)
        alpha_var = torch.var(alpha)
        tau_var = torch.var(tau)
        
        # Target variances for good spatial variation
        target_kappa_var = 0.05  # κ should vary significantly
        target_alpha_var = 0.02   # α should vary moderately  
        target_tau_var = 0.01    # τ should vary less
        
        # Strong penalty if variance is too low
        variance_penalty = (
            F.relu(target_kappa_var - kappa_var) * 2.0 +  # Very strong penalty
            F.relu(target_alpha_var - alpha_var) * 1.0 +   # Strong penalty
            F.relu(target_tau_var - tau_var) * 0.5         # Moderate penalty
        )
        
        # Additional: encourage parameter ranges to be meaningful
        kappa_range = kappa.max() - kappa.min()
        alpha_range = alpha.max() - alpha.min()
        tau_range = tau.max() - tau.min()
        
        range_penalty = (
            F.relu(0.5 - kappa_range) * 1.0 +  # κ range should be at least 0.5
            F.relu(0.3 - alpha_range) * 0.5 +  # α range should be at least 0.3
            F.relu(0.2 - tau_range) * 0.2      # τ range should be at least 0.2
        )
        
        return variance_penalty + range_penalty
    
    def physical_realism_penalty(self, params: torch.Tensor) -> torch.Tensor:
        """
        Keep parameters in physically reasonable ranges
        
        Parameters:
        -----------
        params : torch.Tensor
            SPDE parameters [num_nodes, 3]
            
        Returns:
        --------
        torch.Tensor
            Physical realism penalty
        """
        kappa = params[:, 0]
        alpha = params[:, 1] 
        tau = params[:, 2]
        
        # Kappa should be in [0.1, 5.0] for numerical stability
        kappa_penalty = (torch.mean(F.relu(0.1 - kappa).pow(2)) +  # Lower bound
                        torch.mean(F.relu(kappa - 5.0).pow(2)))   # Upper bound
        
        # Alpha should be in [0.5, 2.5] for reasonable smoothness
        alpha_penalty = (torch.mean(F.relu(0.5 - alpha).pow(2)) +  # Lower bound
                        torch.mean(F.relu(alpha - 2.5).pow(2)))   # Upper bound
        
        # Tau should be in [0.01, 1.0] for proper nugget effect  
        tau_penalty = (torch.mean(F.relu(0.01 - tau).pow(2)) +     # Lower bound
                    torch.mean(F.relu(tau - 1.0).pow(2)))       # Upper bound
        
        return kappa_penalty + alpha_penalty + tau_penalty
    
    def train(self, graph_data, epochs: int = 100, lr: float = 0.0001, 
             weight_decay: float = 1e-4, spatial_weight: float = 0.1, 
             reg_weight: float = 0.01, collect_history: bool = False, 
             feature_history: Optional[list] = None) -> Dict:
        """
        Train the GNN model with optional feature history collection
        
        Parameters:
        -----------
        graph_data : torch_geometric.data.Data
            Graph data
        epochs : int
            Number of training epochs
        lr : float
            Learning rate
        weight_decay : float
            Weight decay for optimizer
        spatial_weight : float
            Weight for spatial smoothness loss
        reg_weight : float
            Weight for parameter regularization
        collect_history : bool
            Whether to collect feature evolution history
        feature_history : list, optional
            List to store feature snapshots during training
            
        Returns:
        --------
        Dict
            Training metrics
        """
        print(f"TRAINING DEBUG: spatial_weight received = {spatial_weight}")
    
        start_time = time.time()
        self._log(f"Starting GNN training for {epochs} epochs...")

        print(f"TRAINING DEBUG: Training started with spatial_weight = {spatial_weight}")
        
        # Move data to device
        graph_data = graph_data.to(self.device)
        
        # Setup optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Setup learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        
        # Training loop
        self.model.train()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass - learn SPDE parameters
            params = self.model(graph_data.x, graph_data.edge_index)

            # Feature diversity analysis every 10 epochs
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    # Analyze input feature diversity
                    feature_std = torch.std(graph_data.x, dim=0)
                    feature_mean_std = torch.mean(feature_std)
                    
                    # Analyze output parameter diversity 
                    param_std = torch.std(params, dim=0)
                    param_mean_std = torch.mean(param_std)
            
            # Collect feature snapshots for visualization
            if collect_history and feature_history is not None:
                if epoch % max(1, epochs // 10) == 0:  # Collect every 10% of training
                    with torch.no_grad():
                        feature_snapshot = params.detach().cpu().numpy().copy()
                        feature_history.append(feature_snapshot)
            
            # Compute losses
            spatial_loss = self.spatial_smoothness_loss(
                params, 
                graph_data.edge_index,
                graph_data.edge_attr[:, 0] if graph_data.edge_attr is not None else None
            )
            
            reg_loss = self.parameter_regularization(params)
            
            # Additional loss components
            diversity_loss = self.diversity_penalty(params)
            realism_loss = self.physical_realism_penalty(params)

            # Total loss 
            loss = (10.0 * self.feature_utilization_loss(params, graph_data.x) +
                5.0 * self.diversity_penalty(params))
                        
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update scheduler
            scheduler.step(loss)
            
            # Record history
            self.training_history['loss'].append(loss.item())
            self.training_history['spatial_loss'].append(spatial_loss.item())
            self.training_history['reg_loss'].append(reg_loss.item())
            
            # Track additional losses
            if 'diversity_loss' not in self.training_history:
                self.training_history['diversity_loss'] = []
                self.training_history['realism_loss'] = []
            self.training_history['diversity_loss'].append(diversity_loss.item())
            self.training_history['realism_loss'].append(realism_loss.item())
            
            # Log progress
            if (epoch + 1) % 20 == 0:
                self._log(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} "
                    f"(Spatial: {spatial_loss.item():.4f}, Reg: {reg_loss.item():.4f}, "
                    f"Div: {diversity_loss.item():.4f}, Real: {realism_loss.item():.4f})")

        # Final feature snapshot for history
        if collect_history and feature_history is not None:
            with torch.no_grad():
                final_params = self.model(graph_data.x, graph_data.edge_index)
                final_snapshot = final_params.detach().cpu().numpy().copy()
                feature_history.append(final_snapshot)
        
        # Training complete
        training_time = time.time() - start_time
        self._log(f"Training complete in {training_time:.2f}s")
        self._log(f"Final loss: {loss.item():.4f}")

        feature_data, param_data = self.complete_feature_diagnostic(
            graph_data=graph_data,
            feature_names=[
                'development_intensity',
                'svi_coefficient', 
                'is_developed',
                'is_uninhabited',
                'normalized_nlcd_class',
                'normalized_degree'
            ]
        )
        
        # Return training metrics
        return {
            'final_loss': loss.item(),
            'spatial_loss': spatial_loss.item(),
            'regularization_loss': reg_loss.item(),
            'training_time': training_time,
            'history': self.training_history
        }
    
    def minimal_validity_constraint(self, params: torch.Tensor) -> torch.Tensor:
        """
        Ensure only mathematical validity (Tau > 0)
        """
        tau = params[:, 2]
        # Strong penalty for negative Tau (breaks SPDE math)
        tau_positivity = torch.mean(F.relu(-tau + 0.01).pow(2)) * 10.0
        return tau_positivity

    def complete_feature_diagnostic(self, graph_data, feature_names):
        """
        Comprehensive diagnostic of all features and their correlations
        """
        import numpy as np
        
        self.model.eval()
        with torch.no_grad():
            params = self.model(graph_data.x, graph_data.edge_index)
        
        features_np = graph_data.x.cpu().numpy()
        params_np = params.cpu().numpy()
        
        print("\n🔍 COMPLETE FEATURE DIAGNOSTIC")
        print("=" * 60)
        
        # Check feature array dimensions
        print(f"Feature array shape: {features_np.shape}")
        print(f"Feature names provided: {len(feature_names)}")
        print(f"Expected features: {feature_names}")
        
        # Analyze each feature individually
        print(f"\n📊 INDIVIDUAL FEATURE ANALYSIS:")
        for f_idx, feature_name in enumerate(feature_names):
            if f_idx < features_np.shape[1]:
                feature_vals = features_np[:, f_idx]
                f_mean = np.mean(feature_vals)
                f_std = np.std(feature_vals)
                f_min = np.min(feature_vals)
                f_max = np.max(feature_vals)
                f_unique = len(np.unique(feature_vals))
                
                status = "✅ GOOD" if f_std > 0.01 else "⚠️ LOW_VAR" if f_std > 0.001 else "❌ NO_VAR"
                
                print(f"  {f_idx}. {feature_name}:")
                print(f"     Range: [{f_min:.6f}, {f_max:.6f}]")
                print(f"     Mean: {f_mean:.6f}, Std: {f_std:.6f}")
                print(f"     Unique values: {f_unique}, Status: {status}")
            else:
                print(f"  {f_idx}. {feature_name}: ❌ INDEX OUT OF BOUNDS")
        
        # Test correlations with ALL features
        print(f"\n🔗 CORRELATION MATRIX (All Features vs All Parameters):")
        param_names = ['Kappa', 'Alpha', 'Tau']
        
        print(f"{'Feature':<25} {'Kappa':<10} {'Alpha':<10} {'Tau':<10}")
        print("-" * 55)
        
        for f_idx, feature_name in enumerate(feature_names):
            if f_idx < features_np.shape[1]:
                feature_vals = features_np[:, f_idx]
                row = f"{feature_name:<25}"
                
                for p_idx in range(3):
                    param_vals = params_np[:, p_idx]
                    try:
                        if np.std(feature_vals) > 0 and np.std(param_vals) > 0:
                            corr = np.corrcoef(feature_vals, param_vals)[0, 1]
                            row += f"{corr:>9.3f} "
                        else:
                            row += f"{'NO_VAR':>9} "
                    except:
                        row += f"{'ERROR':>9} "
                print(row)
            else:
                print(f"{feature_name:<25} OUT_OF_BOUNDS")
        
        # Check if feature array matches expectations
        print(f"\n🚨 POTENTIAL ISSUES:")
        if features_np.shape[1] != len(feature_names):
            print(f"❌ MISMATCH: Feature array has {features_np.shape[1]} columns but {len(feature_names)} names provided")
        
        zero_var_features = []
        for f_idx in range(min(features_np.shape[1], len(feature_names))):
            if np.std(features_np[:, f_idx]) < 0.001:
                zero_var_features.append(feature_names[f_idx] if f_idx < len(feature_names) else f"Feature_{f_idx}")
        
        if zero_var_features:
            print(f"❌ ZERO VARIATION: {zero_var_features}")
        
        return features_np, params_np
    
    def feature_utilization_loss(self, params: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Encourage GNN to use input features meaningfully
        """
        if not features.requires_grad:
            features = features.clone().detach().requires_grad_(True)
        
        # Calculate how much each feature influences the parameters
        # Use sum to create a scalar for gradient computation
        param_sum = params.sum()
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=param_sum,
            inputs=features,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]
        
        if gradients is None:
            return torch.tensor(0.0, device=self.device)
        
        # Encourage high feature importance (negative because we want to maximize)
        feature_importance = torch.mean(torch.abs(gradients))
        return -feature_importance

    def extract_features(self, graph_data) -> np.ndarray:
        """
        Extract learned SPDE parameters from trained model
        
        Parameters:
        -----------
        graph_data : torch_geometric.data.Data
            Graph data
            
        Returns:
        --------
        np.ndarray
            Learned SPDE parameters [num_nodes, 3]
        """
        self.model.eval()
        graph_data = graph_data.to(self.device)
        
        with torch.no_grad():
            features = self.model(graph_data.x, graph_data.edge_index)
            features = features.cpu().numpy()
        
        # Log parameter statistics
        self._log("Extracted SPDE parameters:")
        self._log(f"  Kappa (precision): [{features[:, 0].min():.3f}, {features[:, 0].max():.3f}] "
                f"(var: {np.var(features[:, 0]):.4f})")
        self._log(f"  Alpha (smoothness): [{features[:, 1].min():.3f}, {features[:, 1].max():.3f}] "
                f"(var: {np.var(features[:, 1]):.4f})")
        self._log(f"  Tau (nugget): [{features[:, 2].min():.3f}, {features[:, 2].max():.3f}] "
                f"(var: {np.var(features[:, 2]):.4f})")
        
        return features


def train_accessibility_gnn(graph_data, model: Optional[nn.Module] = None, 
                          epochs: int = 100, collect_history: bool = False,
                          feature_history: Optional[list] = None, **kwargs) -> Tuple[nn.Module, np.ndarray, Dict]:
    """
    Train GNN model to learn SPDE parameters for accessibility
    
    This is the main training function used by the pipeline. It now supports
    feature history collection for visualization purposes.
    
    Parameters:
    -----------
    graph_data : torch_geometric.data.Data
        Graph data
    model : nn.Module, optional
        GNN model (creates default if None)
    epochs : int
        Number of training epochs
    collect_history : bool
        Whether to collect feature evolution during training
    feature_history : list, optional
        List to store feature snapshots (will be modified in-place)
    **kwargs : dict
        Additional training parameters (lr, spatial_weight, reg_weight, etc.)
        
    Returns:
    --------
    Tuple[nn.Module, np.ndarray, Dict]
        (trained_model, spde_parameters, training_metrics)
    """
    # Create model if not provided
    if model is None:
        from ..models.gnn import create_gnn_model
        input_dim = graph_data.x.shape[1]
        model = create_gnn_model(
            input_dim=input_dim,
            hidden_dim=kwargs.get('hidden_dim', 64),
            output_dim=3  # Three SPDE parameters
        )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create trainer
    trainer = AccessibilityTrainer(model, device=device, verbose=kwargs.get('verbose', True))
    
    # Train model with feature history collection
    metrics = trainer.train(
        graph_data, 
        epochs=epochs, 
        collect_history=collect_history,
        feature_history=feature_history,
        **kwargs
    )
    
    # Extract learned SPDE parameters
    features = trainer.extract_features(graph_data)
    
    return model, features, metrics