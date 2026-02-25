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

def compute_block_group_loss(predictions: torch.Tensor,
                              bg_masks: Dict[str, torch.Tensor],
                              bg_targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Supervised loss: predicted block group means must match known BG SVI.
    
    Args:
        predictions: [n_addresses] tensor of predicted SVI
        bg_masks: {bg_id: boolean tensor} indicating addresses in each BG
        bg_targets: {bg_id: scalar tensor} of known BG SVI values
    
    Returns:
        Mean squared error across all block groups
    """
    losses = []
    for bg_id, mask in bg_masks.items():
        if bg_id not in bg_targets:
            continue
        n_addresses = mask.sum()
        if n_addresses < 3:
            continue
        bg_pred_mean = predictions[mask].mean()
        bg_target = bg_targets[bg_id]
        losses.append(F.mse_loss(bg_pred_mean, bg_target))
    
    if len(losses) == 0:
        return torch.tensor(0.0, requires_grad=True)
    
    return torch.stack(losses).mean()

def compute_within_bg_variance_loss(predictions: torch.Tensor,
                                     bg_masks: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Penalize high variance within each block group.
    Addresses in the same BG should have similar predictions.
    """
    variances = []
    for bg_id, mask in bg_masks.items():
        n_addr = mask.sum()
        if n_addr < 3:
            continue
        bg_preds = predictions[mask]
        bg_var = bg_preds.var()
        variances.append(bg_var)
    
    if len(variances) == 0:
        return torch.tensor(0.0, requires_grad=True)
    
    return torch.stack(variances).mean()

def compute_cross_bg_discrimination_loss(predictions: torch.Tensor,
                                          bg_masks: Dict[str, torch.Tensor],
                                          bg_targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Encourage predicted BG means to be separated proportional to their target separation.
    If two BGs have very different target SVIs, their predicted means should also differ.
    """
    bg_ids = list(bg_masks.keys())
    if len(bg_ids) < 2:
        return torch.tensor(0.0, requires_grad=True)
    
    # Compute predicted means for each BG
    pred_means = {}
    for bg_id in bg_ids:
        mask = bg_masks[bg_id]
        if mask.sum() >= 3 and bg_id in bg_targets:
            pred_means[bg_id] = predictions[mask].mean()
    
    if len(pred_means) < 2:
        return torch.tensor(0.0, requires_grad=True)
    
    # For each pair, penalize if predicted separation < target separation
    loss_terms = []
    bg_list = list(pred_means.keys())
    
    for i in range(len(bg_list)):
        for j in range(i + 1, len(bg_list)):
            bg_i, bg_j = bg_list[i], bg_list[j]
            
            target_diff = abs(float(bg_targets[bg_i]) - float(bg_targets[bg_j]))
            pred_diff = torch.abs(pred_means[bg_i] - pred_means[bg_j])
            
            # Penalize if predicted difference is smaller than target difference
            margin_loss = F.relu(target_diff - pred_diff)
            loss_terms.append(margin_loss)
    
    if len(loss_terms) == 0:
        return torch.tensor(0.0, requires_grad=True)
    
    return torch.stack(loss_terms).mean()

def compute_smoothness_loss(predictions: torch.Tensor,
                            edge_index: torch.Tensor) -> torch.Tensor:
    """
    Spatial smoothness prior: neighboring addresses should have similar SVI.
    Implements Tobler's first law as a soft constraint.
    
    Args:
        predictions: [n_addresses] tensor
        edge_index: [2, n_edges] graph connectivity
    
    Returns:
        Mean squared difference between connected nodes
    """
    src, dst = edge_index[0], edge_index[1]
    neighbor_diff = predictions[src] - predictions[dst]
    return torch.mean(neighbor_diff ** 2)


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
                constraint_weight: float = 0.0,
                bg_weight: float = 2.0,
                coherence_weight: float = 1.0,      # NEW
                discrimination_weight: float = 0.5,  # NEW
                smoothness_weight: float = 0.3,
                variation_weight: float = 1.0,
                seed: int = 42):
        """
        Args:
            model: SpatialDisaggregationGNN instance
            learning_rate: Optimizer learning rate
            constraint_weight: Deprecated tract-level constraint (default 0)
            bg_weight: Weight for block group supervision loss
            smoothness_weight: Weight for spatial smoothness prior
            variation_weight: Weight for variation regularization
            seed: Random seed
        """
        self.model = model
        self.seed = seed
        self.constraint_weight = constraint_weight
        self.bg_weight = bg_weight
        self.smoothness_weight = smoothness_weight
        self.variation_weight = variation_weight
        self.coherence_weight = coherence_weight
        self.discrimination_weight = discrimination_weight
        
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
            'bg_losses': [],
            'smoothness_losses': [],
            'constraint_errors': [],
            'spatial_stds': []
        }
    
    def train(self, 
            graph_data: Data,
            tract_svi: float,
            epochs: int = 100,
            verbose: bool = True,
            bg_masks: Dict[str, np.ndarray] = None,
            bg_svis: Dict[str, float] = None) -> Dict:
        """
        Train the GNN with block group supervision.
        
        Args:
            graph_data: PyG Data object with x (features) and edge_index
            tract_svi: Known tract-level SVI value (for post-hoc correction)
            epochs: Number of training epochs
            verbose: Print progress
            bg_masks: {bg_id: boolean array} for training block groups
            bg_svis: {bg_id: SVI value} for training block groups
        
        Returns:
            Dict with training results
        """
        set_random_seed(self.seed)
        self.model.train()
        
        # Convert block group data to tensors
        bg_masks_tensor = {}
        bg_targets_tensor = {}
        if bg_masks is not None and bg_svis is not None:
            for bg_id, mask in bg_masks.items():
                if bg_id in bg_svis and not np.isnan(bg_svis[bg_id]):
                    bg_masks_tensor[bg_id] = torch.BoolTensor(mask)
                    bg_targets_tensor[bg_id] = torch.FloatTensor([bg_svis[bg_id]])
        
        has_bg_supervision = len(bg_masks_tensor) > 0
        if verbose:
            if has_bg_supervision:
                print(f"Training with {len(bg_masks_tensor)} block group constraints")
            else:
                print("Warning: No block group supervision available")
            
            print(f"Loss weights:")
            print(f"  bg_weight:            {self.bg_weight}")
            print(f"  coherence_weight:     {self.coherence_weight}")
            print(f"  discrimination_weight:{self.discrimination_weight}")
            print(f"  smoothness_weight:    {self.smoothness_weight}")
            print(f"  variation_weight:     {self.variation_weight}")
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            predictions = self.model(graph_data.x, graph_data.edge_index)
            
            # Compute losses
            losses = self._compute_losses_v2(
                predictions, 
                graph_data.edge_index,
                bg_masks_tensor,
                bg_targets_tensor,
                tract_svi
            )
            total_loss = losses['total']
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step(total_loss)
            
            # Track metrics
            spatial_std = float(predictions.std())
            self.training_history['losses'].append(total_loss.item())
            self.training_history['bg_losses'].append(losses['bg'].item())
            self.training_history['smoothness_losses'].append(losses['smoothness'].item())
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
            
            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch:3d}: Loss={total_loss.item():.6f}, "
                    f"BG={losses['bg'].item():.4f}, Smooth={losses['smoothness'].item():.4f}, "
                    f"Std={spatial_std:.4f}")
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            final_predictions = self.model(graph_data.x, graph_data.edge_index)
        
        return {
            'success': True,
            'raw_predictions': final_predictions.numpy(),
            'final_spatial_std': float(final_predictions.std()),
            'epochs_trained': epoch + 1,
            'training_history': self.training_history,
            'n_bg_constraints': len(bg_masks_tensor)
        }

    def _compute_losses_v2(self, 
                        predictions: torch.Tensor,
                        edge_index: torch.Tensor,
                        bg_masks: Dict[str, torch.Tensor],
                        bg_targets: Dict[str, torch.Tensor],
                        tract_svi: float) -> Dict:
        """Compute training losses with block group supervision."""
        
        # 1. Block group mean constraint (PRIMARY LEARNING SIGNAL)
        bg_loss = compute_block_group_loss(predictions, bg_masks, bg_targets)
        
        # 2. Within-BG coherence (NEW - reduce within-BG variance)
        coherence_loss = compute_within_bg_variance_loss(predictions, bg_masks)
        
        # 3. Cross-BG discrimination (NEW - separate BG predictions)
        discrimination_loss = compute_cross_bg_discrimination_loss(
            predictions, bg_masks, bg_targets
        )
        
        # 4. Spatial smoothness (GEOGRAPHIC PRIOR)
        smoothness_loss = compute_smoothness_loss(predictions, edge_index)
        
        # 5. Variation regularization (PREVENT COLLAPSE)
        spatial_std = predictions.std()
        variation_loss = F.relu(torch.tensor(0.02) - spatial_std)
        
        # Combine
        total_loss = (
            self.bg_weight * bg_loss +
            self.coherence_weight * coherence_loss +
            self.discrimination_weight * discrimination_loss +
            self.smoothness_weight * smoothness_loss +
            self.variation_weight * variation_loss
        )
        
        return {
            'total': total_loss,
            'bg': bg_loss,
            'coherence': coherence_loss,
            'discrimination': discrimination_loss,
            'smoothness': smoothness_loss,
            'variation': variation_loss
        }
    
    def _compute_losses_legacy(self, predictions: torch.Tensor, target_svi: torch.Tensor) -> Dict:
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