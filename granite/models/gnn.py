"""
GRANITE GNN Architecture (Spatial Version)

Simplified Graph Neural Network for spatial disaggregation.
Uses coordinate-based features and graph topology to learn
within-tract vulnerability patterns.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, BatchNorm
from torch_geometric.data import Data
import numpy as np
import random
from typing import Dict, Optional


def set_random_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class SpatialDisaggregationGNN(nn.Module):
    """
    GNN for spatial disaggregation of tract-level SVI to addresses.
    
    Architecture:
        Input (spatial features) -> MLP encoder -> GCN -> GAT -> GCN -> Output (SVI)
    
    The model learns to allocate known tract-level SVI across addresses
    based on spatial position and graph structure.
    """
    
    def __init__(self, 
                 input_dim: int = 6,
                 hidden_dim: int = 32,
                 dropout: float = 0.2,
                 seed: int = 42):
        """
        Args:
            input_dim: Number of spatial features (default: 6)
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            seed: Random seed for reproducibility
        """
        super(SpatialDisaggregationGNN, self).__init__()
        
        set_random_seed(seed)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Feature encoder (MLP)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Graph convolution layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.norm1 = BatchNorm(hidden_dim)
        
        self.attention = GATConv(
            hidden_dim, 
            hidden_dim // 2, 
            heads=2, 
            concat=True,
            dropout=dropout * 0.5
        )
        self.norm2 = BatchNorm(hidden_dim)
        
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.norm3 = BatchNorm(hidden_dim // 2)
        
        # Output head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self._initialize_weights(seed)
    
    def _initialize_weights(self, seed: int):
        """Initialize weights deterministically."""
        torch.manual_seed(seed)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GNN.
        
        Args:
            x: Node features [n_nodes, input_dim]
            edge_index: Graph connectivity [2, n_edges]
        
        Returns:
            predictions: SVI predictions [n_nodes] in range [0, 1]
        """
        # Input processing
        x = self.input_norm(x)
        x = self.encoder(x)
        
        # Graph convolutions
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.attention(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index)
        x = self.norm3(x)
        x = F.relu(x)
        
        # Prediction
        out = self.predictor(x)
        predictions = torch.sigmoid(out.squeeze())
        
        return predictions


class SpatialGNNTrainer:
    """
    Trainer for spatial disaggregation GNN.
    
    Enforces tract-level mean constraint while encouraging
    meaningful spatial variation.
    """
    
    def __init__(self, 
                 model: SpatialDisaggregationGNN,
                 learning_rate: float = 0.001,
                 constraint_weight: float = 2.0,
                 seed: int = 42):
        """
        Args:
            model: SpatialDisaggregationGNN instance
            learning_rate: Optimizer learning rate
            constraint_weight: Weight for tract mean constraint loss
            seed: Random seed
        """
        self.model = model
        self.seed = seed
        self.constraint_weight = constraint_weight
        
        set_random_seed(seed)
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.8, min_lr=1e-5
        )
        
        self.training_history = {
            'losses': [],
            'constraint_errors': [],
            'spatial_stds': []
        }
    
    def train(self, 
              graph_data: Data,
              tract_svi: float,
              epochs: int = 100,
              verbose: bool = True) -> Dict:
        """
        Train the GNN with tract mean constraint.
        
        Args:
            graph_data: PyG Data object with x (features) and edge_index
            tract_svi: Known tract-level SVI value (constraint target)
            epochs: Number of training epochs
            verbose: Print progress
        
        Returns:
            Dict with training results
        """
        set_random_seed(self.seed)
        
        self.model.train()
        target_svi = torch.FloatTensor([tract_svi])
        n_addresses = graph_data.x.shape[0]
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(graph_data.x, graph_data.edge_index)
            
            # Compute losses
            losses = self._compute_losses(predictions, target_svi)
            total_loss = losses['total']
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step(total_loss)
            
            # Track metrics
            constraint_error = float(abs(predictions.mean() - tract_svi) / tract_svi * 100)
            spatial_std = float(predictions.std())
            
            self.training_history['losses'].append(total_loss.item())
            self.training_history['constraint_errors'].append(constraint_error)
            self.training_history['spatial_stds'].append(spatial_std)
            
            # Early stopping
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 15:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
            
            # Progress
            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch:3d}: Loss={total_loss.item():.6f}, "
                      f"Constraint={constraint_error:.2f}%, Std={spatial_std:.4f}")
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            final_predictions = self.model(graph_data.x, graph_data.edge_index)
        
        return {
            'success': True,
            'raw_predictions': final_predictions.numpy(),
            'final_spatial_std': float(final_predictions.std()),
            'constraint_error': float(abs(final_predictions.mean() - tract_svi)),
            'epochs_trained': epoch + 1,
            'training_history': self.training_history
        }
    
    def _compute_losses(self, predictions: torch.Tensor, target_svi: torch.Tensor) -> Dict:
        """Compute training losses."""
        
        # 1. Constraint loss (tract mean must match)
        predicted_mean = predictions.mean()
        constraint_loss = F.mse_loss(predicted_mean.unsqueeze(0), target_svi)
        
        # 2. Variation loss (encourage spatial heterogeneity)
        spatial_std = predictions.std()
        min_variation = 0.02
        variation_loss = F.relu(min_variation - spatial_std)
        
        # 3. Bounds loss (keep predictions in [0, 1])
        bounds_loss = (
            torch.mean(F.relu(predictions - 1.0)) + 
            torch.mean(F.relu(-predictions))
        )
        
        # 4. Range loss (encourage spread)
        pred_range = predictions.max() - predictions.min()
        range_loss = F.relu(0.05 - pred_range)
        
        # Combine
        total_loss = (
            self.constraint_weight * constraint_loss +
            1.5 * variation_loss +
            1.0 * bounds_loss +
            0.3 * range_loss
        )
        
        return {
            'total': total_loss,
            'constraint': constraint_loss,
            'variation': variation_loss,
            'bounds': bounds_loss,
            'range': range_loss
        }
    
    def predict(self, graph_data: Data) -> np.ndarray:
        """Generate predictions without training."""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(graph_data.x, graph_data.edge_index)
        return predictions.numpy()


class MultiTractGNNTrainer(SpatialGNNTrainer):
    """
    Extended trainer for multi-tract training with per-tract constraints.
    """
    
    def train_multi_tract(self,
                          graph_data: Data,
                          tract_svis: Dict[str, float],
                          tract_masks: Dict[str, np.ndarray],
                          epochs: int = 100,
                          verbose: bool = True) -> Dict:
        """
        Train on multiple tracts simultaneously.
        
        Args:
            graph_data: Combined graph for all tracts
            tract_svis: Dict mapping FIPS -> target SVI
            tract_masks: Dict mapping FIPS -> boolean mask for addresses
            epochs: Training epochs
            verbose: Print progress
        
        Returns:
            Training results dict
        """
        set_random_seed(self.seed)
        self.model.train()
        
        # Convert masks to tensors
        tract_masks_tensor = {
            fips: torch.BoolTensor(mask) 
            for fips, mask in tract_masks.items()
        }
        tract_targets = {
            fips: torch.FloatTensor([svi])
            for fips, svi in tract_svis.items()
        }
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            predictions = self.model(graph_data.x, graph_data.edge_index)
            
            # Per-tract constraint losses
            constraint_losses = []
            for fips, target in tract_targets.items():
                mask = tract_masks_tensor[fips]
                tract_preds = predictions[mask]
                if len(tract_preds) > 0:
                    tract_mean = tract_preds.mean()
                    loss = F.mse_loss(tract_mean.unsqueeze(0), target)
                    constraint_losses.append(loss)
            
            constraint_loss = torch.stack(constraint_losses).mean() if constraint_losses else torch.tensor(0.0)
            
            # Other losses
            spatial_std = predictions.std()
            variation_loss = F.relu(0.02 - spatial_std)
            bounds_loss = torch.mean(F.relu(predictions - 1.0)) + torch.mean(F.relu(-predictions))
            
            total_loss = (
                self.constraint_weight * constraint_loss +
                1.5 * variation_loss +
                1.0 * bounds_loss
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step(total_loss)
            
            # Early stopping
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 15:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
            
            if verbose and epoch % 20 == 0:
                avg_error = self._compute_avg_constraint_error(predictions, tract_targets, tract_masks_tensor)
                print(f"Epoch {epoch:3d}: Loss={total_loss.item():.6f}, "
                      f"AvgConstraint={avg_error:.2f}%, Std={float(spatial_std):.4f}")
        
        self.model.eval()
        with torch.no_grad():
            final_predictions = self.model(graph_data.x, graph_data.edge_index)
        
        return {
            'success': True,
            'raw_predictions': final_predictions.numpy(),
            'epochs_trained': epoch + 1
        }
    
    def _compute_avg_constraint_error(self, predictions, tract_targets, tract_masks):
        """Compute average constraint error across tracts."""
        errors = []
        for fips, target in tract_targets.items():
            mask = tract_masks[fips]
            tract_preds = predictions[mask]
            if len(tract_preds) > 0:
                tract_mean = float(tract_preds.mean())
                target_val = float(target)
                if target_val > 0:
                    error = abs(tract_mean - target_val) / target_val * 100
                    errors.append(error)
        return np.mean(errors) if errors else 0.0