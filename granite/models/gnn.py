"""
GRANITE GNN Architecture and Training

Provides graph neural network models and trainers for accessibility-based
social vulnerability prediction with constraint enforcement.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, BatchNorm
from torch_geometric.data import Data
import numpy as np
import random
from typing import Dict, Tuple, Optional

def set_random_seed(seed=42):
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Force deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set PyTorch Geometric reproducibility
    torch.use_deterministic_algorithms(True, warn_only=True)


class AccessibilitySVIGNN(nn.Module):
    """
    Context-aware GNN for accessibility-vulnerability prediction.
    
    Supports context-gated feature modulation and multi-task learning.
    """
    def __init__(self, accessibility_features_dim, context_features_dim=5, 
             hidden_dim=64, dropout=0.3, seed=42, use_context_gating=True,
             use_multitask=True):

        super(AccessibilitySVIGNN, self).__init__()
        
        # Set seed for reproducible initialization
        set_random_seed(seed)
        
        self.accessibility_features_dim = accessibility_features_dim
        self.context_features_dim = context_features_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.use_context_gating = use_context_gating  
        
        if use_context_gating:
            self.context_gate = ContextGatedFeatureModulator(
                accessibility_dim=accessibility_features_dim,
                context_dim=context_features_dim,
                hidden_dim=32
            )

        self.use_multitask = use_multitask
        
        # Input normalization (applied to potentially modulated features)
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
        
        # PRIMARY TASK: SVI prediction head
        self.svi_predictor = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//4, 1)
        )
        
        # AUXILIARY TASK HEADS (only if using multi-task)
        if use_multitask:
            self.accessibility_classifier = nn.Sequential(
                nn.Linear(hidden_dim//2, hidden_dim//4),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim//4, 5)
            )
            
            self.vehicle_predictor = nn.Sequential(
                nn.Linear(hidden_dim//2, hidden_dim//4),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim//4, 1)
            )
            
            self.employment_classifier = nn.Sequential(
                nn.Linear(hidden_dim//2, hidden_dim//4),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim//4, 3)
            )
        
        self._initialize_weights(seed)
        self.dropout = nn.Dropout(dropout)

    def _initialize_weights(self, seed):
        """Initialize weights deterministically using the provided seed."""
        # Set seed again to ensure consistent initialization
        torch.manual_seed(seed)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
    def forward(self, accessibility_features, edge_index, context_features=None, 
                return_accessibility=False, return_all_tasks=False):
        """
        Context-aware forward pass through the GNN.
        
        Args:
            accessibility_features: Input node features [n_nodes, accessibility_dim]
            edge_index: Graph edge connectivity
            context_features: Context features for gating [n_nodes, context_dim]
            return_accessibility: If True, return predictions, accessibility, and attention
        
        Returns:
            svi_predictions: Predicted SVI values [0,1]
            learned_accessibility: (optional) Learned accessibility representations
            attention_weights: (optional) Context-gating attention weights
        """
        # NEW: Apply context-gating if available
        attention_weights = None
        if self.use_context_gating and context_features is not None:
            # Modulate accessibility features based on context
            modulated_features, attention_weights = self.context_gate(
                accessibility_features,
                context_features
            )
            # Use modulated features for rest of pipeline
            x = self.input_norm(modulated_features)
        else:
            # Standard path (no context-gating)
            x = self.input_norm(accessibility_features)
        
        x = self.feature_encoder(x)
        
        # Graph convolution (unchanged)
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
        svi_predictions = self.svi_predictor(x)
        svi_predictions = torch.sigmoid(svi_predictions.squeeze())
        
        # Return based on what's requested
        if not self.use_multitask or not return_all_tasks:
            if return_accessibility:
                return svi_predictions, learned_accessibility, attention_weights
            else:
                return svi_predictions
        
        # MULTI-TASK OUTPUT
        return {
            'svi': svi_predictions,
            'accessibility_quintile_logits': self.accessibility_classifier(x),
            'vehicle_ownership': torch.sigmoid(self.vehicle_predictor(x).squeeze()),
            'employment_category_logits': self.employment_classifier(x),
            'learned_accessibility': learned_accessibility,
            'attention_weights': attention_weights,
            'embeddings': x
        }


class GraphSAGEAccessibilitySVIGNN(nn.Module):
    """
    GraphSAGE variant of AccessibilitySVIGNN.

    Identical to AccessibilitySVIGNN except the GCN+GAT+GCN convolution stack
    is replaced with three SAGEConv layers.  All other components (context
    gating, feature encoder, auxiliary heads, constraint logic) are unchanged.
    """
    def __init__(self, accessibility_features_dim, context_features_dim=5,
                 hidden_dim=64, dropout=0.3, seed=42, use_context_gating=True,
                 use_multitask=True):

        super(GraphSAGEAccessibilitySVIGNN, self).__init__()

        set_random_seed(seed)

        self.accessibility_features_dim = accessibility_features_dim
        self.context_features_dim = context_features_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.use_context_gating = use_context_gating

        if use_context_gating:
            self.context_gate = ContextGatedFeatureModulator(
                accessibility_dim=accessibility_features_dim,
                context_dim=context_features_dim,
                hidden_dim=32
            )

        self.use_multitask = use_multitask

        self.input_norm = nn.LayerNorm(accessibility_features_dim)

        self.feature_encoder = nn.Sequential(
            nn.Linear(accessibility_features_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # three SAGEConv layers replacing GCN+GAT+GCN
        self.spatial_conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.spatial_norm1 = BatchNorm(hidden_dim)

        self.spatial_conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.spatial_norm2 = BatchNorm(hidden_dim)

        self.spatial_conv3 = SAGEConv(hidden_dim, hidden_dim // 2)
        self.spatial_norm3 = BatchNorm(hidden_dim // 2)

        self.accessibility_learner = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 4, accessibility_features_dim),
            nn.ReLU()
        )

        self.svi_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )

        if use_multitask:
            self.accessibility_classifier = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim // 4, 5)
            )

            self.vehicle_predictor = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim // 4, 1)
            )

            self.employment_classifier = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim // 4, 3)
            )

        self._initialize_weights(seed)
        self.dropout = nn.Dropout(dropout)

    def _initialize_weights(self, seed):
        torch.manual_seed(seed)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, accessibility_features, edge_index, context_features=None,
                return_accessibility=False, return_all_tasks=False):
        attention_weights = None
        if self.use_context_gating and context_features is not None:
            modulated_features, attention_weights = self.context_gate(
                accessibility_features, context_features
            )
            x = self.input_norm(modulated_features)
        else:
            x = self.input_norm(accessibility_features)

        x = self.feature_encoder(x)

        x = self.spatial_conv1(x, edge_index)
        x = self.spatial_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.spatial_conv2(x, edge_index)
        x = self.spatial_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.spatial_conv3(x, edge_index)
        x = self.spatial_norm3(x)
        x = F.relu(x)

        learned_accessibility = self.accessibility_learner(x)

        svi_predictions = self.svi_predictor(x)
        svi_predictions = torch.sigmoid(svi_predictions.squeeze())

        if not self.use_multitask or not return_all_tasks:
            if return_accessibility:
                return svi_predictions, learned_accessibility, attention_weights
            else:
                return svi_predictions

        return {
            'svi': svi_predictions,
            'accessibility_quintile_logits': self.accessibility_classifier(x),
            'vehicle_ownership': torch.sigmoid(self.vehicle_predictor(x).squeeze()),
            'employment_category_logits': self.employment_classifier(x),
            'learned_accessibility': learned_accessibility,
            'attention_weights': attention_weights,
            'embeddings': x
        }


class ContextGatedFeatureModulator(nn.Module):
    """
    Dynamically weight accessibility features based on socioeconomic context.
    """
    def __init__(self, accessibility_dim, context_dim, hidden_dim=32):
        super(ContextGatedFeatureModulator, self).__init__()
        
        # Context encoder: Demographics -> embedding
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention mechanism: Context -> feature importance weights
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, accessibility_dim),
            nn.Softmax(dim=-1)
        )
        
        # Base feature importance (learned universal patterns)
        self.base_importance = nn.Parameter(torch.ones(accessibility_dim))
    
    def forward(self, accessibility_features, context_features):
        """Apply context-dependent feature weighting."""
        # Encode context
        context_embedding = self.context_encoder(context_features)
        
        # Compute feature importance weights from context
        attention_weights = self.attention(context_embedding)
        
        # Combine with base importance
        combined_weights = attention_weights * self.base_importance
        
        # Apply weights to features
        modulated_features = accessibility_features * combined_weights
        
        return modulated_features, attention_weights

class AccessibilityGNNTrainer:
    """
    Single-tract trainer for GRANITE training.
    Enforces tract-level mean constraint while learning spatial patterns.
    """
    def __init__(self, model, config=None, seed=42):
        self.model = model
        self.config = config or {}
        self.seed = seed
        
        self.use_multitask = config.get('use_multitask', True)
        self.multitask_weights = {
            'accessibility': 0.3,
            'vehicle': 0.3,
            'employment': 0.2
        }

        self.enforce_constraints = config.get('enforce_constraints', True)
        self.constraint_weight = config.get('constraint_weight', 
                                        2.0 if self.enforce_constraints else 0.0)
        
        # Set seed for optimizer initialization
        set_random_seed(seed)
        
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
            'spatial_stds': [],
            'raw_predictions_history': [],
            'attention_weights': [] 
        }
        
    def train(self, graph_data, tract_svi, epochs=100, verbose=True):
        """
        Train GNN on single tract with deterministic behavior.
        
        Args:
            graph_data: PyTorch Geometric Data object
            tract_svi: Target SVI value for the tract
            epochs: Number of training epochs
            verbose: Print training progress
        
        Returns:
            Dict with training results and diagnostics
        """
        # Ensure reproducibility
        set_random_seed(self.seed)
        
        self.model.train()
        device = graph_data.x.device
        target_svi = torch.FloatTensor([tract_svi]).to(device)
        n_addresses = graph_data.x.shape[0]
        
        learned_accessibility_history = []
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # Check if context features available
            context = getattr(graph_data, 'context', None)
            
            # Forward pass with context-gating
            predictions, learned_accessibility, attention_weights = self.model(
                graph_data.x, graph_data.edge_index, return_accessibility=True, context_features=context
            )

            # Optional: Track attention weights for analysis
            if attention_weights is not None:
                if epoch == 0:
                    self.training_history['attention_weights'] = []
                self.training_history['attention_weights'].append(
                    attention_weights.detach().cpu().numpy()
                )
            
            # Compute losses with REBALANCED weights
            losses = self._compute_losses(predictions, target_svi, n_addresses)
            total_loss = losses['total']
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step(total_loss)
            
            # Track metrics
            constraint_error = float(abs(predictions.mean() - tract_svi) / tract_svi * 100)
            spatial_std = float(predictions.std())
            
            self.training_history['losses'].append(total_loss.item())
            self.training_history['constraint_errors'].append(constraint_error)
            self.training_history['spatial_stds'].append(spatial_std)
            self.training_history['raw_predictions_history'].append(predictions.detach().cpu().numpy())
            
            # Store accessibility learning evolution
            if epoch % 10 == 0 or epoch == epochs - 1:
                learned_accessibility_history.append({
                    'epoch': epoch,
                    'learned_features': learned_accessibility.detach().cpu().numpy()
                })
            
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
        context = getattr(graph_data, 'context', None)
        final_predictions, final_learned_accessibility, attention_weights = self.model(
            graph_data.x, graph_data.edge_index, return_accessibility=True, context_features=context
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
        """Compute training losses with optional constraint enforcement."""
        
        # 1. Constraint preservation loss
        predicted_mean = predictions.mean()
        constraint_loss = F.mse_loss(predicted_mean.unsqueeze(0), target_svi)
        
        # 2. Spatial variation encouragement
        spatial_std = predictions.std()
        min_variation = 0.02
        variation_loss = F.relu(min_variation - spatial_std)
        
        # 3. Bounds enforcement (always active)
        bounds_loss = torch.mean(F.relu(predictions - 1.0)) + torch.mean(F.relu(-predictions))
        
        # 4. Distribution regularization
        if n_addresses > 10:
            prediction_range = predictions.max() - predictions.min()
            min_range = 0.05
            range_loss = F.relu(min_range - prediction_range)
        else:
            range_loss = torch.tensor(0.0, device=predictions.device)
        
        # 5. Accessibility consistency
        accessibility_consistency_loss = self._compute_accessibility_consistency_loss(predictions)
        
        if self.enforce_constraints:
            # Standard constrained training
            total_loss = (
                self.constraint_weight * constraint_loss +     # Use configured weight
                1.5 * variation_loss +
                1.0 * bounds_loss +
                0.3 * range_loss +
                0.5 * accessibility_consistency_loss
            )
        else:
            # Unconstrained: learn from structure only
            total_loss = (
                0.0 * constraint_loss +           # No constraint pressure
                2.0 * variation_loss +            # Strong variation encouragement
                1.0 * bounds_loss +               # Keep valid range
                0.5 * range_loss +                # Distribution shape
                1.0 * accessibility_consistency_loss  # Structure learning
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
            return torch.tensor(0.0, device=predictions.device)
        
        sorted_preds = torch.sort(predictions)[0]
        
        if len(sorted_preds) > 1:
            pred_gradient = sorted_preds[1:] - sorted_preds[:-1]
            gradient_loss = F.relu(0.001 - pred_gradient.mean())
        else:
            gradient_loss = torch.tensor(0.0, device=predictions.device)
        
        return gradient_loss

    def predict_unconstrained(self, graph_data):
        """
        Generate predictions without any correction.
        Returns raw model outputs for validation.
        
        Returns:
            dict with 'predictions' and 'learned_accessibility'
        """
        self.model.eval()
        # Check if context features available
        context = getattr(graph_data, 'context', None)

        with torch.no_grad():
            predictions, learned_accessibility, attention_weights = self.model(  # <- RIGHT: unpacking 3
                graph_data.x, 
                graph_data.edge_index, 
                return_accessibility=True,
                context_features=context
            )
        
        return {
            'predictions': predictions.detach().numpy(),
            'learned_accessibility': learned_accessibility.detach().numpy()
        }


class MultiTractGNNTrainer:
    """
    Multi-tract trainer for GRANITE with per-tract constraint enforcement.
    
    CRITICAL STABILITY IMPROVEMENTS:
    - Deterministic training with seed control
    - REBALANCED loss weights (constraint 5.0 -> 2.0)
    - Enhanced diagnostic tracking
    - Raw prediction preservation for analysis
    """
    
    def __init__(self, model, config=None, seed=42):
        self.model = model
        self.config = config or {}
        self.seed = seed

        # Multi-task learning configuration
        self.use_multitask = config.get('use_multitask', True)
        self.multitask_weights = {
            'accessibility': 0.3,
            'vehicle': 0.3,
            'employment': 0.2
        }

        # NEW: Training mode controls
        self.enforce_constraints = config.get('enforce_constraints', True)
        self.constraint_weight = config.get('constraint_weight',
                                        2.0 if self.enforce_constraints else 0.0)
        self.bg_constraint_weight = config.get('bg_constraint_weight', 1.0)

        # Pairwise ordering loss configuration
        self.ordering_weight = config.get('ordering_weight', 0.5)
        self.ordering_min_gap = config.get('ordering_min_gap', 0.5)
        self.ordering_margin = config.get('ordering_margin', 0.02)

        # Set seed for optimizer initialization
        set_random_seed(seed)

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
            'per_tract_errors': {},
            'raw_predictions_history': [],
            'attention_weights': [] 
        }
    
    def train(self, graph_data, tract_svis: Dict[str, float],
          tract_masks: Dict[str, np.ndarray], epochs=100, verbose=True,
          feature_names=None, block_group_targets=None,
          block_group_masks=None, ordering_values=None):
        """
        Train GNN across multiple tracts with per-tract constraints.

        STABILITY GUARANTEE: Identical results across runs with same seed.

        Args:
            graph_data: PyTorch Geometric Data object with all addresses
            tract_svis: Dict mapping tract FIPS to target SVI values
            tract_masks: Dict mapping tract FIPS to boolean masks
            epochs: Number of training epochs
            verbose: Print training progress
            block_group_targets: Optional dict mapping BG GEOID to target SVI
            block_group_masks: Optional dict mapping BG GEOID to boolean masks
            ordering_values: Optional 1D numpy array of raw log_appvalue per address

        Returns:
            Dict with training results including raw predictions
        """
        # Ensure complete reproducibility
        set_random_seed(self.seed)
        
        self.model.train()
        
        # Convert tract SVIs to tensors (match graph device)
        device = graph_data.x.device
        tract_targets = {
            fips: torch.FloatTensor([svi]).to(device)
            for fips, svi in tract_svis.items()
        }

        # Convert masks to tensors
        tract_masks_tensor = {
            fips: torch.BoolTensor(mask).to(device)
            for fips, mask in tract_masks.items()
        }

        # Convert block group data to tensors if provided
        bg_targets_tensor = None
        bg_masks_tensor = None
        if block_group_targets is not None and block_group_masks is not None:
            bg_targets_tensor = {
                bg_id: torch.FloatTensor([svi]).to(device)
                for bg_id, svi in block_group_targets.items()
            }
            bg_masks_tensor = {
                bg_id: torch.BoolTensor(mask).to(device)
                for bg_id, mask in block_group_masks.items()
            }

        n_addresses = graph_data.x.shape[0]

        # Convert ordering values to tensor if provided
        ordering_tensor = None
        ordering_group_masks = None
        if ordering_values is not None:
            ordering_tensor = torch.FloatTensor(ordering_values).to(device)
            n_valid = int((~torch.isnan(ordering_tensor)).sum().item())
            # prefer block group masks for pair sampling, fall back to tract masks
            if bg_masks_tensor is not None:
                ordering_group_masks = bg_masks_tensor
            else:
                ordering_group_masks = tract_masks_tensor
            n_groups = len(ordering_group_masks)
            if verbose:
                print(f"Pairwise ordering: {n_valid} of {n_addresses} addresses have valid log_appvalue")
                print(f"Ordering pairs per epoch: ~{min(100, n_valid) * n_groups} across "
                      f"{n_groups} groups (min_gap={self.ordering_min_gap}, margin={self.ordering_margin})")

        # Generate auxiliary labels for multi-task learning
        auxiliary_labels = None
        if self.use_multitask:
            if feature_names is None:
                if verbose:
                    print("WARNING: feature_names not provided, multi-task disabled")
                self.use_multitask = False
            else:
                # Extract features as numpy
                accessibility_np = graph_data.x.cpu().numpy()
                context_np = graph_data.context.cpu().numpy()
                
                # Generate labels
                labels_dict = generate_auxiliary_labels(
                    accessibility_np, context_np, feature_names
                )
                
                # Convert to tensors
                device = graph_data.x.device
                auxiliary_labels = {
                    'accessibility_quintile': torch.tensor(
                        labels_dict['accessibility_quintile'],
                        dtype=torch.long, device=device
                    ),
                    'vehicle_ownership': torch.tensor(
                        labels_dict['vehicle_ownership'],
                        dtype=torch.float32, device=device
                    ),
                    'employment_category': torch.tensor(
                        labels_dict['employment_category'],
                        dtype=torch.long, device=device
                    )
                }
                
                if verbose:
                    print("\n=== Multi-Task Learning Enabled ===")
                    print(f"Generated auxiliary labels:")
                    print(f"  Accessibility quintiles: {np.bincount(labels_dict['accessibility_quintile'])}")
                    print(f"  Vehicle ownership range: [{labels_dict['vehicle_ownership'].min():.3f}, {labels_dict['vehicle_ownership'].max():.3f}]")
                    print(f"  Employment categories: {np.bincount(labels_dict['employment_category'])}")
                    print("=" * 40 + "\n")

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # sample ordering pairs for this epoch
            ordering_pairs = None
            if ordering_tensor is not None and ordering_group_masks is not None:
                epoch_seed = self.seed + epoch
                low_idx, high_idx, _pairs_per_group = self._sample_ordering_pairs(
                    ordering_tensor, ordering_group_masks,
                    n_pairs_per_group=100, min_gap=self.ordering_min_gap,
                    seed=epoch_seed
                )
                if len(low_idx) > 0:
                    ordering_pairs = (low_idx, high_idx)

            # Check if context features available
            context = getattr(graph_data, 'context', None)

            if self.use_multitask and auxiliary_labels is not None:
                # === MULTI-TASK PATH ===
                outputs = self.model(
                    graph_data.x, 
                    graph_data.edge_index,
                    context_features=context,
                    return_all_tasks=True
                )
                
                predictions = outputs['svi']
                learned_accessibility = outputs['learned_accessibility']
                attention_weights = outputs['attention_weights']
                
                # Track attention weights
                if attention_weights is not None:
                    if epoch == 0:
                        self.training_history['attention_weights'] = []
                    self.training_history['attention_weights'].append(
                        attention_weights.detach().cpu().numpy()
                    )
                
                # Compute constraint losses
                losses = self._compute_multi_tract_losses(
                    predictions, tract_targets, tract_masks_tensor, n_addresses,
                    block_group_targets=bg_targets_tensor,
                    block_group_masks=bg_masks_tensor,
                    ordering_pairs=ordering_pairs
                )

                # Compute auxiliary losses
                aux_loss_dict = compute_multitask_loss(
                    outputs, auxiliary_labels, weights=self.multitask_weights
                )
                
                # Combined loss
                total_loss = losses['total'] + 1.0 * aux_loss_dict['total']
                
                # Enhanced logging every 10 epochs
                if verbose and epoch % 10 == 0:
                    print(f"\nEpoch {epoch}:")
                    print(f"  Total Loss: {total_loss:.4f}")
                    print(f"  Constraint: {losses['constraint']:.4f}")
                    print(f"  Auxiliary Total: {aux_loss_dict['total']:.4f}")
                    print(f"    - Accessibility cls: {aux_loss_dict['accessibility']:.4f}")
                    print(f"    - Vehicle reg: {aux_loss_dict['vehicle']:.4f}")
                    print(f"    - Employment cls: {aux_loss_dict['employment']:.4f}")
                    
                    # Evaluate auxiliary task accuracy
                    with torch.no_grad():
                        acc_pred = outputs['accessibility_quintile_logits'].argmax(dim=1)
                        acc_acc = (acc_pred == auxiliary_labels['accessibility_quintile']).float().mean().item()
                        
                        emp_pred = outputs['employment_category_logits'].argmax(dim=1)
                        emp_acc = (emp_pred == auxiliary_labels['employment_category']).float().mean().item()
                        
                        veh_corr = np.corrcoef(
                            outputs['vehicle_ownership'].cpu().numpy(),
                            auxiliary_labels['vehicle_ownership'].cpu().numpy()
                        )[0, 1]
                        
                        print(f"  Auxiliary Performance:")
                        print(f"    - Accessibility accuracy: {acc_acc:.3f} (target: >0.70)")
                        print(f"    - Employment accuracy: {emp_acc:.3f} (target: >0.70)")
                        print(f"    - Vehicle correlation: {veh_corr:.3f} (target: >0.80)")
            
            else:
                # === SINGLE-TASK PATH (original) ===
                predictions, learned_accessibility, attention_weights = self.model(
                    graph_data.x, graph_data.edge_index, 
                    return_accessibility=True, context_features=context
                )

                # Track attention weights
                if attention_weights is not None:
                    if epoch == 0:
                        self.training_history['attention_weights'] = []
                    self.training_history['attention_weights'].append(
                        attention_weights.detach().cpu().numpy()
                    )
                
                # Compute losses
                losses = self._compute_multi_tract_losses(
                    predictions, tract_targets, tract_masks_tensor, n_addresses,
                    block_group_targets=bg_targets_tensor,
                    block_group_masks=bg_masks_tensor,
                    ordering_pairs=ordering_pairs
                )
                total_loss = losses['total']

                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}: Loss={total_loss:.4f}")
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step(total_loss)
            
            # Track metrics
            overall_constraint_error = self._compute_overall_constraint_error(
                predictions, tract_targets, tract_masks_tensor
            )
            spatial_std = float(predictions.std())

            self.training_history['losses'].append(total_loss.item())
            self.training_history['constraint_errors'].append(overall_constraint_error)
            self.training_history['spatial_stds'].append(spatial_std)
            self.training_history['raw_predictions_history'].append(predictions.detach().cpu().numpy())

            # Track block group constraint error
            if bg_targets_tensor is not None:
                bg_error = self._compute_overall_constraint_error(
                    predictions, bg_targets_tensor, bg_masks_tensor
                )
                self.training_history.setdefault('bg_constraint_errors', []).append(bg_error)

            # Track ordering loss
            if ordering_pairs is not None:
                ord_loss_val = losses['ordering'].item()
                self.training_history.setdefault('ordering_losses', []).append(ord_loss_val)

            # Track per-tract errors
            per_tract_errors = self._compute_per_tract_errors(
                predictions, tract_targets, tract_masks_tensor
            )
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
                bg_msg = ""
                if bg_targets_tensor is not None:
                    bg_err = self.training_history['bg_constraint_errors'][-1]
                    bg_msg = f", BG Constraint={bg_err:.2f}%"
                print(f"Epoch {epoch:3d}: Loss={total_loss.item():.6f}, "
                    f"Overall Constraint={overall_constraint_error:.2f}%{bg_msg}, "
                    f"Std={spatial_std:.4f}, "
                    f"LR={current_lr:.6f}")

                # Show per-tract errors
                for fips, error in list(per_tract_errors.items())[:3]:
                    tract_id = fips[-6:]  # Last 6 digits
                    print(f"  Tract {tract_id}: {error:.2f}% error")

                # Show ordering loss
                if ordering_pairs is not None:
                    ord_val = losses['ordering'].item()
                    n_active = int((torch.relu(
                        predictions[ordering_pairs[1]] - predictions[ordering_pairs[0]] + self.ordering_margin
                    ) > 0).sum().item())
                    n_total = len(ordering_pairs[0])
                    print(f"  Ordering: loss={ord_val:.4f}, violated={n_active}/{n_total} pairs")

        # Final evaluation
        self.model.eval()
        context = getattr(graph_data, 'context', None)
        final_predictions, final_learned_accessibility, attention_weights = self.model(
            graph_data.x, graph_data.edge_index, return_accessibility=True, context_features=context
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

        # block group constraint error (final)
        if bg_targets_tensor is not None:
            results['bg_constraint_error'] = self._compute_overall_constraint_error(
                final_predictions, bg_targets_tensor, bg_masks_tensor
            )
            results['per_bg_errors'] = self._compute_per_tract_errors(
                final_predictions, bg_targets_tensor, bg_masks_tensor
            )

        return results
    
    def _sample_ordering_pairs(self, ordering_values, group_masks, n_pairs_per_group=100, min_gap=0.5, seed=None):
        """sample pairwise ordering pairs within groups based on property value gaps."""
        low_indices = []
        high_indices = []
        pairs_per_group = {}

        for group_id, mask in group_masks.items():
            # get indices with valid (non-nan) ordering values within this group
            group_indices = torch.where(mask)[0]
            group_values = ordering_values[group_indices]
            valid = ~torch.isnan(group_values)
            valid_indices = group_indices[valid]
            valid_values = group_values[valid]

            if len(valid_indices) < 2:
                pairs_per_group[group_id] = 0
                continue

            # find all pairs where value[i] - value[j] >= min_gap
            # i = low value (higher vulnerability), j = high value (lower vulnerability)
            n_valid = len(valid_indices)

            # sort by value to efficiently find pairs with sufficient gap
            sorted_order = torch.argsort(valid_values)
            sorted_values = valid_values[sorted_order]
            sorted_indices = valid_indices[sorted_order]

            # build candidate pairs: for each low-value address, find high-value addresses
            # with gap >= min_gap using binary search on sorted values
            candidate_low = []
            candidate_high = []
            for i in range(n_valid):
                # find first j where sorted_values[j] - sorted_values[i] >= min_gap
                threshold = sorted_values[i] + min_gap
                j_start = torch.searchsorted(sorted_values, threshold).item()
                if j_start < n_valid:
                    for j in range(j_start, n_valid):
                        candidate_low.append(sorted_indices[i].item())
                        candidate_high.append(sorted_indices[j].item())
                        # cap candidates to avoid O(N^2) explosion
                        if len(candidate_low) > n_pairs_per_group * 10:
                            break
                if len(candidate_low) > n_pairs_per_group * 10:
                    break

            n_candidates = len(candidate_low)
            if n_candidates == 0:
                pairs_per_group[group_id] = 0
                continue

            # sample up to n_pairs_per_group
            rng_seed = seed if seed is not None else 0
            rng = torch.Generator()
            rng.manual_seed(rng_seed)

            if n_candidates <= n_pairs_per_group:
                sampled = list(range(n_candidates))
            else:
                sampled = torch.randperm(n_candidates, generator=rng)[:n_pairs_per_group].tolist()

            for idx in sampled:
                low_indices.append(candidate_low[idx])
                high_indices.append(candidate_high[idx])

            pairs_per_group[group_id] = len(sampled)

        device = ordering_values.device
        if len(low_indices) == 0:
            return torch.tensor([], dtype=torch.long, device=device), torch.tensor([], dtype=torch.long, device=device), pairs_per_group

        return (
            torch.tensor(low_indices, dtype=torch.long, device=device),
            torch.tensor(high_indices, dtype=torch.long, device=device),
            pairs_per_group
        )

    def _compute_ordering_loss(self, predictions, low_value_indices, high_value_indices, margin=0.02):
        """margin-based ranking loss: low-value properties should predict higher vulnerability."""
        if len(low_value_indices) == 0:
            return torch.tensor(0.0, device=predictions.device)

        pred_low = predictions[low_value_indices]
        pred_high = predictions[high_value_indices]
        # penalize when high-value property predicts higher vulnerability than low-value
        pair_loss = torch.relu(pred_high - pred_low + margin)
        return pair_loss.mean()

    def _compute_multi_tract_losses(self, predictions, tract_targets,
                                    tract_masks, n_addresses,
                                    block_group_targets=None,
                                    block_group_masks=None,
                                    ordering_pairs=None):
        """
        Multi-tract loss computation with configurable constraint enforcement.

        When enforce_constraints=False, the model learns purely from
        accessibility patterns without mean-matching pressure.
        """

        # 1. Per-tract constraint losses (weight controlled by config)
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
            constraint_loss = torch.tensor(0.0, device=predictions.device)

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
            variation_loss = torch.tensor(0.0, device=predictions.device)

        # 3. Bounds enforcement (always active)
        bounds_loss = torch.mean(F.relu(predictions - 1.0)) + torch.mean(F.relu(-predictions))

        # 4. Cross-tract smoothness
        smoothness_loss = self._compute_cross_tract_smoothness(predictions, tract_masks)

        # 5. Block group constraint losses
        bg_constraint_loss = torch.tensor(0.0, device=predictions.device)
        if block_group_targets is not None and block_group_masks is not None:
            bg_losses = []
            for bg_id, bg_target in block_group_targets.items():
                bg_mask = block_group_masks[bg_id]
                bg_predictions = predictions[bg_mask]
                if len(bg_predictions) > 0:
                    bg_mean = bg_predictions.mean()
                    bg_loss = F.mse_loss(bg_mean.unsqueeze(0), bg_target)
                    bg_losses.append(bg_loss)
            if len(bg_losses) > 0:
                bg_constraint_loss = torch.mean(torch.stack(bg_losses))

        # 6. Pairwise ordering loss
        ordering_loss = torch.tensor(0.0, device=predictions.device)
        if ordering_pairs is not None:
            low_idx, high_idx = ordering_pairs
            ordering_loss = self._compute_ordering_loss(
                predictions, low_idx, high_idx, margin=self.ordering_margin
            )

        # Conditional weighting based on mode
        if self.enforce_constraints:
            # Standard constrained training
            total_loss = (
                self.constraint_weight * constraint_loss +
                0.8 * variation_loss +
                1.0 * bounds_loss +
                0.1 * smoothness_loss
            )
            # add block group constraint if provided
            if block_group_targets is not None:
                total_loss = total_loss + self.bg_constraint_weight * bg_constraint_loss
            # add ordering loss if provided
            if ordering_pairs is not None:
                total_loss = total_loss + self.ordering_weight * ordering_loss
        else:
            # Unconstrained: learn from structure only
            total_loss = (
                0.0 * constraint_loss +     # No constraint pressure
                2.0 * variation_loss +      # Strong variation encouragement
                1.0 * bounds_loss +         # Keep valid range
                0.5 * smoothness_loss       # Spatial structure
            )

        return {
            'total': total_loss,
            'constraint': constraint_loss,
            'variation': variation_loss,
            'bounds': bounds_loss,
            'smoothness': smoothness_loss,
            'bg_constraint': bg_constraint_loss,
            'ordering': ordering_loss
        }

    def predict_unconstrained(self, graph_data):
        """
        Generate predictions without any correction.
        Returns raw model outputs for validation.
        
        Returns:
            dict with 'predictions' and 'learned_accessibility'
        """
        self.model.eval()

        # Check if context features available
        context = getattr(graph_data, 'context', None)

        with torch.no_grad():
            predictions, learned_accessibility, attention_weights = self.model(
                graph_data.x, 
                graph_data.edge_index, 
                return_accessibility=True,
                context_features=context
            )
        
        return {
            'predictions': predictions.detach().numpy(),
            'learned_accessibility': learned_accessibility.detach().numpy()
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
            return torch.tensor(0.0, device=predictions.device)
    
    def _compute_overall_constraint_error(self, predictions, tract_targets, tract_masks):
        """Compute weighted average constraint error across tracts"""
        
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
        print(f"{np.sum(zero_var_mask)} features have zero variance; proceeding")
    
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

def generate_auxiliary_labels(accessibility_features, context_features, feature_names):
    """
    Generate ground truth labels for auxiliary tasks.
    
    Args:
        accessibility_features: [n_addresses, n_features] numpy array
        context_features: [n_addresses, 5] numpy array
        feature_names: List of feature names
    
    Returns:
        dict with 'accessibility_quintile', 'vehicle_ownership', 'employment_category'
    """
    n_addresses = len(accessibility_features)
    
    # Task 1: Accessibility quintile from travel times
    time_indices = [i for i, name in enumerate(feature_names) if 'min_time' in name]
    if len(time_indices) > 0:
        avg_travel_time = accessibility_features[:, time_indices].mean(axis=1)
        quintiles = np.argsort(np.argsort(avg_travel_time)) // (n_addresses // 5)
        quintiles = np.clip(quintiles, 0, 4).astype(np.int64)
    else:
        quintiles = np.random.randint(0, 5, size=n_addresses)
    
    # Task 2: Vehicle ownership from context features
    vehicle_ownership = context_features[:, 0]  # First context feature
    
    # Task 3: Employment category from employment counts
    emp_indices = [i for i, name in enumerate(feature_names) if 'employment_count' in name]
    if len(emp_indices) > 0:
        emp_access = accessibility_features[:, emp_indices].sum(axis=1)
        low_thresh = np.percentile(emp_access, 33)
        high_thresh = np.percentile(emp_access, 67)
        
        employment_category = np.zeros(n_addresses, dtype=np.int64)
        employment_category[emp_access >= low_thresh] = 1
        employment_category[emp_access >= high_thresh] = 2
    else:
        employment_category = np.random.randint(0, 3, size=n_addresses)
    
    return {
        'accessibility_quintile': quintiles,
        'vehicle_ownership': vehicle_ownership,
        'employment_category': employment_category
    }


def compute_multitask_loss(outputs, auxiliary_targets, weights=None):
    """
    Compute auxiliary task losses.
    
    Args:
        outputs: Dict from model forward pass with all task predictions
        auxiliary_targets: Dict with ground truth tensors
        weights: Optional dict of loss weights
    
    Returns:
        dict with 'total' and individual task losses
    """
    if weights is None:
        weights = {'accessibility': 0.3, 'vehicle': 0.3, 'employment': 0.2}
    
    # Classification losses
    acc_loss = F.cross_entropy(
        outputs['accessibility_quintile_logits'],
        auxiliary_targets['accessibility_quintile']
    )
    
    emp_loss = F.cross_entropy(
        outputs['employment_category_logits'],
        auxiliary_targets['employment_category']
    )
    
    # Regression loss
    vehicle_loss = F.mse_loss(
        outputs['vehicle_ownership'],
        auxiliary_targets['vehicle_ownership']
    )
    
    # Weighted combination
    total = (
        weights['accessibility'] * acc_loss +
        weights['vehicle'] * vehicle_loss +
        weights['employment'] * emp_loss
    )
    
    return {
        'total': total,
        'accessibility': acc_loss,
        'vehicle': vehicle_loss,
        'employment': emp_loss
    }

# ============================================================================
# MIXTURE OF EXPERTS SUPPORT
# ============================================================================
# These functions enable MoE usage while keeping single-model code unchanged

def create_standard_model(accessibility_features_dim, context_features_dim=5,
                         hidden_dim=64, dropout=0.3, seed=42):
    """Factory for standard single-expert GNN."""
    return AccessibilitySVIGNN(
        accessibility_features_dim=accessibility_features_dim,
        context_features_dim=context_features_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        seed=seed,
        use_context_gating=True,
        use_multitask=True
    )


def get_model_type(config):
    """Determine if training single model or mixture."""
    use_mixture = config.get('training', {}).get('use_mixture', False)
    return 'mixture' if use_mixture else 'standard'