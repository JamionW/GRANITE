"""
GRANITE GNN Architecture and Training
Unified file with both single-tract and multi-tract training capabilities
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, BatchNorm
from torch_geometric.data import Data
import numpy as np
from typing import Dict

class AccessibilitySVIGNN(nn.Module):
    """
    Shared GNN model architecture for both single-tract and multi-tract training.
    Learns accessibility-vulnerability relationships from graph-structured data.
    """
    def __init__(self, accessibility_features_dim, hidden_dim=64, dropout=0.3):
        super(AccessibilitySVIGNN, self).__init__()
        
        self.accessibility_features_dim = accessibility_features_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        
        # Input normalization
        self.input_norm = nn.LayerNorm(accessibility_features_dim)
        
        # Feature encoding
        self.feature_encoder = nn.Sequential(
            nn.Linear(accessibility_features_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Graph convolution layers
        self.spatial_conv1 = GCNConv(hidden_dim, hidden_dim)
        self.spatial_norm1 = BatchNorm(hidden_dim)
        
        self.attention_conv = GATConv(hidden_dim, hidden_dim//2, heads=2, concat=True, dropout=dropout*0.5)
        self.attention_norm = BatchNorm(hidden_dim)
        
        self.spatial_conv2 = GCNConv(hidden_dim, hidden_dim//2)
        self.spatial_norm2 = BatchNorm(hidden_dim//2)
        
        # Accessibility learning layer
        self.accessibility_learner = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim//4, accessibility_features_dim),
            nn.ReLU()
        )
        
        # SVI prediction head
        self.prediction_layers = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//4, 1)
        )
        
        self._initialize_weights()
        self.dropout = nn.Dropout(dropout)

    def _initialize_weights(self):
        """Proper weight initialization for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
    def forward(self, accessibility_features, edge_index, return_accessibility=False):
        """
        Forward pass through the GNN.
        
        Args:
            accessibility_features: Input node features
            edge_index: Graph edge connectivity
            return_accessibility: If True, return both predictions and learned accessibility
        
        Returns:
            svi_predictions: Predicted SVI values [0,1]
            learned_accessibility: (optional) Learned accessibility representations
        """
        # Input processing
        x = self.input_norm(accessibility_features)
        x = self.feature_encoder(x)
        
        # Graph convolution
        x = self.spatial_conv1(x, edge_index)
        x = self.spatial_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x_att = self.attention_conv(x, edge_index)
        x_att = self.attention_norm(x_att)
        x_att = F.relu(x_att)
        
        x = self.spatial_conv2(x_att, edge_index)
        x = self.spatial_norm2(x)
        x = F.relu(x)
        
        # Learn accessibility representations
        learned_accessibility = self.accessibility_learner(x)
        
        # SVI prediction
        svi_predictions = self.prediction_layers(x)
        svi_predictions = torch.sigmoid(svi_predictions.squeeze())
        
        if return_accessibility:
            return svi_predictions, learned_accessibility
        else:
            return svi_predictions


class AccessibilityGNNTrainer:
    """
    Single-tract trainer for standard GRANITE training.
    Enforces tract-level mean constraint while learning spatial patterns.
    """
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or {}
        
        learning_rate = float(self.config.get('learning_rate', 0.001))
        weight_decay = float(self.config.get('weight_decay', 1e-4))
        
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.8, min_lr=1e-5
        )
        
        # Training state
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'losses': [], 
            'constraint_errors': [], 
            'spatial_stds': []
        }
        
    def train(self, graph_data, tract_svi, epochs=100, verbose=True):
        """
        Train GNN on single tract.
        
        Args:
            graph_data: PyTorch Geometric Data object
            tract_svi: Target tract SVI value
            epochs: Number of training epochs
            verbose: Print progress
        
        Returns:
            Dict with predictions and training metrics
        """
        self.model.train()
        target_svi = torch.FloatTensor([tract_svi])
        n_addresses = graph_data.x.shape[0]
        
        learned_accessibility_history = []
        
        if verbose:
            print(f"Training GNN: {n_addresses} addresses, target SVI: {tract_svi:.4f}")
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            svi_predictions, learned_accessibility = self.model(
                graph_data.x, graph_data.edge_index, return_accessibility=True
            )
            
            # Store learned accessibility periodically
            if epoch % 10 == 0:
                learned_accessibility_history.append(learned_accessibility.detach().numpy())
            
            # Compute losses
            losses = self._compute_losses(svi_predictions, target_svi, n_addresses)
            total_loss = losses['total']
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step(total_loss)
            
            # Track metrics
            constraint_error = float(abs(svi_predictions.mean() - tract_svi) / tract_svi * 100)
            spatial_std = float(svi_predictions.std())
            
            self.training_history['losses'].append(total_loss.item())
            self.training_history['constraint_errors'].append(constraint_error)
            self.training_history['spatial_stds'].append(spatial_std)
            
            # Early stopping
            if total_loss.item() < self.best_loss:
                self.best_loss = total_loss.item()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                    
            if self.patience_counter >= 15:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
            
            # Progress reporting
            if verbose and epoch % 20 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:3d}: Loss={total_loss.item():.6f}, "
                    f"Constraint={constraint_error:.2f}%, "
                    f"Std={spatial_std:.4f}, "
                    f"LR={current_lr:.6f}")
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            final_predictions, final_learned_accessibility = self.model(
                graph_data.x, graph_data.edge_index, return_accessibility=True
            )
        
        results = {
            'final_predictions': final_predictions.detach().numpy(),
            'learned_accessibility': final_learned_accessibility.detach().numpy(),
            'learned_accessibility_history': learned_accessibility_history,
            'training_history': self.training_history,
            'final_spatial_std': float(final_predictions.std()),
            'constraint_error': float(abs(final_predictions.mean() - tract_svi)),
            'epochs_trained': epoch + 1,
            'final_loss': total_loss.item(),
            'learning_converged': self.patience_counter < 15,
            'success': True
        }
        
        return results
    
    def _compute_losses(self, predictions, target_svi, n_addresses):
        """Single-tract loss computation"""
        
        # 1. Constraint preservation loss
        predicted_mean = predictions.mean()
        constraint_loss = F.mse_loss(predicted_mean.unsqueeze(0), target_svi)
        
        # 2. Spatial variation encouragement
        spatial_std = predictions.std()
        min_variation = 0.02
        variation_loss = F.relu(min_variation - spatial_std)
        
        # 3. Bounds enforcement
        bounds_loss = torch.mean(F.relu(predictions - 1.0)) + torch.mean(F.relu(-predictions))
        
        # 4. Distribution regularization
        if n_addresses > 10:
            prediction_range = predictions.max() - predictions.min()
            min_range = 0.05
            range_loss = F.relu(min_range - prediction_range)
        else:
            range_loss = torch.tensor(0.0)
        
        # 5. Accessibility consistency
        accessibility_consistency_loss = self._compute_accessibility_consistency_loss(predictions)
        
        # Weighted combination
        total_loss = (
            3.0 * constraint_loss +
            0.5 * variation_loss +
            1.0 * bounds_loss +
            0.3 * range_loss +
            0.2 * accessibility_consistency_loss
        )
        
        return {
            'total': total_loss,
            'constraint': constraint_loss,
            'variation': variation_loss,
            'bounds': bounds_loss,
            'range': range_loss,
            'accessibility': accessibility_consistency_loss
        }
    
    def _compute_accessibility_consistency_loss(self, predictions):
        """Encourage structured predictions"""
        
        if len(predictions) < 4:
            return torch.tensor(0.0)
        
        sorted_preds = torch.sort(predictions)[0]
        
        if len(sorted_preds) > 1:
            pred_gradient = sorted_preds[1:] - sorted_preds[:-1]
            gradient_loss = F.relu(0.001 - pred_gradient.mean())
        else:
            gradient_loss = torch.tensor(0.0)
        
        return gradient_loss


class MultiTractGNNTrainer:
    """
    Multi-tract trainer for GRANITE with per-tract constraint enforcement.
    
    Addresses single-tract confounding by learning from cross-tract
    accessibility-vulnerability gradients while maintaining per-tract means.
    """
    
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or {}
        
        learning_rate = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.8, min_lr=1e-5
        )
        
        # Training state
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'losses': [], 
            'constraint_errors': [], 
            'spatial_stds': [],
            'per_tract_errors': {}
        }
    
    def train(self, graph_data, tract_svis: Dict[str, float], 
              tract_masks: Dict[str, np.ndarray], epochs=100, verbose=True):
        """
        Train GNN across multiple tracts with per-tract constraints.
        
        Args:
            graph_data: PyTorch Geometric Data object with all addresses
            tract_svis: Dict mapping tract FIPS to target SVI values
            tract_masks: Dict mapping tract FIPS to boolean masks
            epochs: Number of training epochs
            verbose: Print training progress
        
        Returns:
            Dict with final predictions, learned accessibility, and training metrics
        """
        self.model.train()
        n_addresses = graph_data.x.shape[0]
        
        # Convert tract SVIs to tensors
        tract_targets = {
            fips: torch.FloatTensor([svi]) 
            for fips, svi in tract_svis.items()
        }
        
        # Convert masks to tensors
        tract_masks_tensor = {
            fips: torch.BoolTensor(mask) 
            for fips, mask in tract_masks.items()
        }
        
        if verbose:
            print(f"\nTraining Multi-Tract GNN: {n_addresses} addresses across {len(tract_svis)} tracts")
            for fips, svi in tract_svis.items():
                n_addrs = tract_masks[fips].sum()
                print(f"  Tract {fips}: {n_addrs} addresses, SVI={svi:.4f}")
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            svi_predictions, learned_accessibility = self.model(
                graph_data.x, graph_data.edge_index, return_accessibility=True
            )
            
            # Compute multi-tract losses
            losses = self._compute_multi_tract_losses(
                svi_predictions, 
                tract_targets, 
                tract_masks_tensor,
                n_addresses
            )
            
            total_loss = losses['total']
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step(total_loss)
            
            # Track metrics
            overall_constraint_error = self._compute_overall_constraint_error(
                svi_predictions, tract_targets, tract_masks_tensor
            )
            
            per_tract_errors = self._compute_per_tract_errors(
                svi_predictions, tract_targets, tract_masks_tensor
            )
            
            spatial_std = float(svi_predictions.std())
            
            self.training_history['losses'].append(total_loss.item())
            self.training_history['constraint_errors'].append(overall_constraint_error)
            self.training_history['spatial_stds'].append(spatial_std)
            
            # Store per-tract errors
            for fips, error in per_tract_errors.items():
                if fips not in self.training_history['per_tract_errors']:
                    self.training_history['per_tract_errors'][fips] = []
                self.training_history['per_tract_errors'][fips].append(error)
            
            # Early stopping
            if total_loss.item() < self.best_loss:
                self.best_loss = total_loss.item()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= 15:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
            
            # Progress reporting
            if verbose and epoch % 20 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:3d}: Loss={total_loss.item():.6f}, "
                      f"Overall Constraint={overall_constraint_error:.2f}%, "
                      f"Std={spatial_std:.4f}, LR={current_lr:.6f}")
                
                # Show per-tract errors (abbreviated)
                for fips, error in list(per_tract_errors.items())[:3]:
                    print(f"  Tract {fips[-6:]}: {error:.2f}% error")
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            final_predictions, final_learned_accessibility = self.model(
                graph_data.x, graph_data.edge_index, return_accessibility=True
            )
        
        # Compute final metrics
        final_per_tract_errors = self._compute_per_tract_errors(
            final_predictions, tract_targets, tract_masks_tensor
        )
        
        results = {
            'final_predictions': final_predictions.detach().numpy(),
            'learned_accessibility': final_learned_accessibility.detach().numpy(),
            'training_history': self.training_history,
            'final_spatial_std': float(final_predictions.std()),
            'overall_constraint_error': self._compute_overall_constraint_error(
                final_predictions, tract_targets, tract_masks_tensor
            ),
            'per_tract_errors': final_per_tract_errors,
            'epochs_trained': epoch + 1,
            'final_loss': total_loss.item(),
            'learning_converged': self.patience_counter < 15,
            'success': True
        }
        
        return results
    
    def _compute_multi_tract_losses(self, predictions, tract_targets, 
                                     tract_masks, n_addresses):
        """
        Multi-tract loss computation with per-tract constraints.
        
        Key innovation: Enforce mean preservation for EACH tract independently.
        """
        
        # 1. Per-tract constraint losses (HIGHEST PRIORITY)
        constraint_losses = []
        
        for fips, target_svi in tract_targets.items():
            mask = tract_masks[fips]
            tract_predictions = predictions[mask]
            
            if len(tract_predictions) > 0:
                tract_mean = tract_predictions.mean()
                tract_loss = F.mse_loss(tract_mean.unsqueeze(0), target_svi)
                constraint_losses.append(tract_loss)
        
        if len(constraint_losses) > 0:
            constraint_loss = torch.mean(torch.stack(constraint_losses))
        else:
            constraint_loss = torch.tensor(0.0)
        
        # 2. Within-tract variation encouragement
        variation_losses = []
        for fips, mask in tract_masks.items():
            tract_predictions = predictions[mask]
            
            if len(tract_predictions) > 10:
                tract_std = tract_predictions.std()
                min_variation = 0.02
                variation_loss = F.relu(min_variation - tract_std)
                variation_losses.append(variation_loss)
        
        if len(variation_losses) > 0:
            variation_loss = torch.mean(torch.stack(variation_losses))
        else:
            variation_loss = torch.tensor(0.0)
        
        # 3. Bounds enforcement
        bounds_loss = torch.mean(F.relu(predictions - 1.0)) + torch.mean(F.relu(-predictions))
        
        # 4. Cross-tract smoothness
        smoothness_loss = self._compute_cross_tract_smoothness(predictions, tract_masks)
        
        # Weighted combination
        total_loss = (
            5.0 * constraint_loss +      # Per-tract constraints (highest weight)
            0.3 * variation_loss +        # Within-tract variation
            1.0 * bounds_loss +           # Bounds enforcement
            0.1 * smoothness_loss         # Cross-tract smoothness
        )
        
        return {
            'total': total_loss,
            'constraint': constraint_loss,
            'variation': variation_loss,
            'bounds': bounds_loss,
            'smoothness': smoothness_loss
        }
    
    def _compute_cross_tract_smoothness(self, predictions, tract_masks):
        """Gentle smoothness penalty for extreme tract differences"""
        
        tract_means = []
        for mask in tract_masks.values():
            if mask.sum() > 0:
                tract_means.append(predictions[mask].mean())
        
        if len(tract_means) > 1:
            tract_means_tensor = torch.stack(tract_means)
            range_penalty = (tract_means_tensor.max() - tract_means_tensor.min()) * 0.05
            return range_penalty
        else:
            return torch.tensor(0.0)
    
    def _compute_overall_constraint_error(self, predictions, tract_targets, tract_masks):
        """Compute weighted average constraint error across all tracts"""
        
        errors = []
        weights = []
        
        for fips, target_svi in tract_targets.items():
            mask = tract_masks[fips]
            tract_predictions = predictions[mask]
            
            if len(tract_predictions) > 0:
                tract_mean = float(tract_predictions.mean())
                target_val = target_svi.item()
                if target_val > 0:
                    error = abs(tract_mean - target_val) / target_val * 100
                else:
                    error = abs(tract_mean - target_val) * 100
                errors.append(error)
                weights.append(len(tract_predictions))
        
        if len(errors) > 0:
            total_addresses = sum(weights)
            weighted_error = sum(e * w for e, w in zip(errors, weights)) / total_addresses
            return weighted_error
        else:
            return 0.0
    
    def _compute_per_tract_errors(self, predictions, tract_targets, tract_masks):
        """Compute constraint error for each tract individually"""
        
        per_tract_errors = {}
        
        for fips, target_svi in tract_targets.items():
            mask = tract_masks[fips]
            tract_predictions = predictions[mask]
            
            if len(tract_predictions) > 0:
                tract_mean = float(tract_predictions.mean())
                target_val = target_svi.item()
                if target_val > 0:
                    error = abs(tract_mean - target_val) / target_val * 100
                else:
                    error = abs(tract_mean - target_val) * 100
                per_tract_errors[fips] = error
            else:
                per_tract_errors[fips] = 0.0
        
        return per_tract_errors


def normalize_accessibility_features(features, method='robust'):
    """
    Robust feature normalization to prevent training instability.
    
    Args:
        features: Numpy array of accessibility features
        method: 'robust' (default) or 'standard'
    
    Returns:
        Tuple of (normalized_features, scaler)
    """
    if method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    
    # Handle edge cases
    if features.shape[1] == 0:
        return features, scaler
    
    # Check for zero variance features
    feature_stds = np.std(features, axis=0)
    zero_var_mask = feature_stds < 1e-8
    
    if np.any(zero_var_mask):
        print(f"Warning: {np.sum(zero_var_mask)} features have zero variance and will be removed")
        features = features[:, ~zero_var_mask]
        if features.shape[1] == 0:
            raise ValueError("All features have zero variance")
    
    # Apply normalization
    normalized_features = scaler.fit_transform(features)
    
    # Validation
    if np.any(np.isnan(normalized_features)):
        print("Error: NaN values after normalization")
        normalized_features = np.nan_to_num(normalized_features)
    
    if np.any(np.isinf(normalized_features)):
        print("Error: Infinite values after normalization")
        normalized_features = np.nan_to_num(normalized_features)
    
    return normalized_features, scaler