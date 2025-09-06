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

# ADD THESE CLASSES TO YOUR models/gnn.py FILE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class AccessibilityLearningGNN(nn.Module):
    """
    Stage 1: Learn accessibility patterns from network structure and features
    FIXED: Robust tensor handling and type safety
    """
    def __init__(self, input_dim, hidden_dim=64, accessibility_output_dim=9):
        super(AccessibilityLearningGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.accessibility_output_dim = accessibility_output_dim
        
        # Multi-layer GCN for accessibility learning
        try:
            from torch_geometric.nn import GCNConv
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
            self._use_pyg = True
        except ImportError:
            # Fallback to linear layers
            self.conv1 = nn.Linear(input_dim, hidden_dim)
            self.conv2 = nn.Linear(hidden_dim, hidden_dim)
            self.conv3 = nn.Linear(hidden_dim, hidden_dim // 2)
            self._use_pyg = False
        
        # Accessibility-specific heads
        self.employment_head = nn.Linear(hidden_dim // 2, 3)  # min_time, accessible_count, transit_share
        self.healthcare_head = nn.Linear(hidden_dim // 2, 3)
        self.grocery_head = nn.Linear(hidden_dim // 2, 3)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, batch=None):
        # Ensure input is float tensor
        if x.dtype != torch.float32:
            x = x.float()
        
        # Graph convolutions for accessibility pattern learning
        if self._use_pyg:
            h1 = F.relu(self.conv1(x, edge_index))
            h1 = self.dropout(h1)
            
            h2 = F.relu(self.conv2(h1, edge_index))
            h2 = self.dropout(h2)
            
            h3 = F.relu(self.conv3(h2, edge_index))
        else:
            # Fallback for no PyG
            h1 = F.relu(self.conv1(x))
            h1 = self.dropout(h1)
            
            h2 = F.relu(self.conv2(h1))
            h2 = self.dropout(h2)
            
            h3 = F.relu(self.conv3(h2))
        
        # Multi-head accessibility prediction
        employment_accessibility = self.employment_head(h3)
        healthcare_accessibility = self.healthcare_head(h3)
        grocery_accessibility = self.grocery_head(h3)
        
        # Concatenate all accessibility features
        accessibility_features = torch.cat([
            employment_accessibility, 
            healthcare_accessibility, 
            grocery_accessibility
        ], dim=1)
        
        return accessibility_features

class AccessibilitySVIGNN(nn.Module):
    """
    Stage 2: Predict SVI using learned accessibility features
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(AccessibilitySVIGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Attention-based GNN for SVI prediction
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
        
        # SVI prediction head
        self.svi_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, output_dim),
            nn.Sigmoid()  # Ensure SVI in [0,1] range
        )
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, batch=None):
        # Attention-based feature aggregation
        h1 = F.relu(self.conv1(x, edge_index))
        h1 = self.dropout(h1)
        
        h2 = F.relu(self.conv2(h1, edge_index))
        h2 = self.dropout(h2)
        
        h3 = F.relu(self.conv3(h2, edge_index))
        
        # SVI prediction
        svi_predictions = self.svi_head(h3)
        
        return svi_predictions.squeeze()

class AccessibilityGNNTrainer:
    """
    Trainer for Stage 1: Accessibility Learning with comprehensive type safety
    """
    def __init__(self, model, config=None):
        """
        FIXED: Trainer for Stage 1 with proper config type conversion
        """
        self.model = model
        self.config = config or {}
        
        # CRITICAL FIX: Convert string config values to proper numeric types
        try:
            learning_rate = self.config.get('learning_rate', 0.001)
            if isinstance(learning_rate, str):
                learning_rate = float(learning_rate)
            
            weight_decay = self.config.get('weight_decay', 1e-5)
            if isinstance(weight_decay, str):
                weight_decay = float(weight_decay)
            
            print(f"🔧 Config conversion: lr={learning_rate} (type: {type(learning_rate)}), wd={weight_decay} (type: {type(weight_decay)})")
            
        except (ValueError, TypeError) as e:
            print(f"❌ Config conversion error: {e}")
            learning_rate = 0.001
            weight_decay = 1e-5
        
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    def train_accessibility_prediction(self, graph_data, accessibility_targets, epochs=50, verbose=False):
        """
        FIXED: Train GNN to predict accessibility features with robust error handling
        """
        print(f"🔍 Starting accessibility training with {epochs} epochs...")
        
        self.model.train()
        
        # COMPREHENSIVE INPUT VALIDATION AND TYPE CONVERSION
        try:
            features_array = accessibility_targets['features_per_address']
            
            # Debug information
            print(f"🔍 Input type: {type(features_array)}")
            print(f"🔍 Input shape: {getattr(features_array, 'shape', 'No shape attr')}")
            print(f"🔍 Input dtype: {getattr(features_array, 'dtype', 'No dtype attr')}")
            
            # STEP 1: Convert to numpy array if needed
            if not isinstance(features_array, np.ndarray):
                if hasattr(features_array, 'values'):  # DataFrame
                    features_array = features_array.values
                else:
                    features_array = np.array(features_array)
                print(f"🔍 Converted to numpy array: {features_array.shape}")
            
            # STEP 2: Handle mixed data types
            if features_array.dtype == 'object':
                print(f"🔍 Found object dtype, converting to numeric...")
                # Handle mixed types by converting each element
                numeric_array = np.zeros(features_array.shape, dtype=np.float64)
                for i in range(features_array.shape[0]):
                    for j in range(features_array.shape[1]):
                        try:
                            val = features_array[i, j]
                            if isinstance(val, str):
                                # Try to convert string to float
                                numeric_array[i, j] = float(val) if val.replace('.', '').replace('-', '').isdigit() else 0.0
                            else:
                                numeric_array[i, j] = float(val)
                        except (ValueError, TypeError):
                            numeric_array[i, j] = 0.0
                features_array = numeric_array
                print(f"🔍 Converted object array to numeric: {features_array.dtype}")
            
            # STEP 3: Ensure numeric dtype
            if not np.issubdtype(features_array.dtype, np.number):
                print(f"🔍 Converting non-numeric dtype {features_array.dtype} to float64")
                features_array = features_array.astype(np.float64)
            
            # STEP 4: Handle NaN and infinite values
            if np.any(np.isnan(features_array)):
                print(f"🔍 Found {np.sum(np.isnan(features_array))} NaN values, replacing with 0")
                features_array = np.nan_to_num(features_array, nan=0.0)
            
            if np.any(np.isinf(features_array)):
                print(f"🔍 Found infinite values, replacing with finite values")
                features_array = np.nan_to_num(features_array, posinf=999.0, neginf=0.0)
            
            # STEP 5: Create tensor with explicit dtype
            target_tensor = torch.tensor(features_array, dtype=torch.float32)
            print(f"🔍 Final tensor: shape={target_tensor.shape}, dtype={target_tensor.dtype}")
            
            # STEP 6: Validate tensor contents
            if torch.any(torch.isnan(target_tensor)):
                print(f"❌ WARNING: Tensor contains NaN values!")
                target_tensor = torch.nan_to_num(target_tensor, nan=0.0)
            
            if torch.any(torch.isinf(target_tensor)):
                print(f"❌ WARNING: Tensor contains infinite values!")
                target_tensor = torch.nan_to_num(target_tensor, posinf=999.0, neginf=0.0)
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR in tensor conversion: {str(e)}")
            print(f"❌ Error type: {type(e).__name__}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            raise
        
        training_losses = []
        
        for epoch in range(epochs):
            try:
                self.optimizer.zero_grad()
                
                # SAFE Forward pass with type checking
                predicted_accessibility = self.model(graph_data.x, graph_data.edge_index)
                
                # Ensure prediction tensor is the right type
                if predicted_accessibility.dtype != torch.float32:
                    predicted_accessibility = predicted_accessibility.float()
                
                if epoch == 0:
                    print(f"🔍 Prediction tensor: shape={predicted_accessibility.shape}, dtype={predicted_accessibility.dtype}")
                    print(f"🔍 Target tensor: shape={target_tensor.shape}, dtype={target_tensor.dtype}")
                
                # SAFE size matching
                num_addresses = target_tensor.shape[0]  # 2394
                if predicted_accessibility.shape[0] > num_addresses:
                    predicted_accessibility = predicted_accessibility[:num_addresses]
                    if epoch == 0:
                        print(f"🔧 Fixed: Sliced predictions to {num_addresses} address nodes")

                target_tensor_epoch = target_tensor
                
                # SAFE loss calculation
                accessibility_loss = F.mse_loss(predicted_accessibility, target_tensor_epoch)
                
                # SAFE regularization with explicit indexing
                try:
                    # Time penalties: indices 0, 3, 6 (employment, healthcare, grocery min_time)
                    if predicted_accessibility.shape[1] >= 7:
                        time_columns = predicted_accessibility[:, [0, 3, 6]]
                        time_penalty = F.relu(time_columns - 120.0).mean()
                    else:
                        time_penalty = torch.tensor(0.0)
                    
                    # Count penalties: indices 1, 4, 7 (accessible_count)
                    if predicted_accessibility.shape[1] >= 8:
                        count_columns = predicted_accessibility[:, [1, 4, 7]]
                        count_penalty = F.relu(count_columns - 10.0).mean()
                    else:
                        count_penalty = torch.tensor(0.0)
                    
                except IndexError as ie:
                    print(f"⚠️ Index error in regularization: {ie}")
                    time_penalty = torch.tensor(0.0)
                    count_penalty = torch.tensor(0.0)
                
                total_loss = accessibility_loss + 0.1 * (time_penalty + count_penalty)
                
                # Check loss is valid
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"❌ Invalid loss at epoch {epoch}: {total_loss}")
                    break
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                training_losses.append(total_loss.item())
                
                if verbose and epoch % 10 == 0:
                    print(f"Accessibility Training Epoch {epoch}: Loss = {total_loss.item():.6f}")
            
            except Exception as e:
                print(f"❌ ERROR in training epoch {epoch}: {str(e)}")
                print(f"❌ Error type: {type(e).__name__}")
                import traceback
                print(f"❌ Traceback: {traceback.format_exc()}")
                # Continue with next epoch instead of crashing
                training_losses.append(float('inf'))
                continue
        
        # Final prediction
        try:
            self.model.eval()
            with torch.no_grad():
                final_accessibility = self.model(graph_data.x, graph_data.edge_index)
            
            print(f"✅ Training completed successfully with {len([l for l in training_losses if l != float('inf')])} valid epochs")
            
            return {
                'predicted_accessibility': final_accessibility.detach().numpy(),
                'training_losses': training_losses,
                'final_loss': training_losses[-1] if training_losses else float('inf'),
                'target_features': accessibility_targets['feature_names']
            }
        
        except Exception as e:
            print(f"❌ ERROR in final prediction: {str(e)}")
            # Return fallback result
            return {
                'predicted_accessibility': np.zeros((target_tensor.shape[0], target_tensor.shape[1])),
                'training_losses': training_losses,
                'final_loss': float('inf'),
                'target_features': accessibility_targets['feature_names']
            }

class AccessibilitySVITrainer:
    """
    Trainer for Stage 2: SVI Prediction from Accessibility
    """
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or {}
        
        # SAME FIX: Convert string config values to proper numeric types
        try:
            learning_rate = self.config.get('learning_rate', 0.001)
            if isinstance(learning_rate, str):
                learning_rate = float(learning_rate)
            
            weight_decay = self.config.get('weight_decay', 1e-5)
            if isinstance(weight_decay, str):
                weight_decay = float(weight_decay)
                
        except (ValueError, TypeError):
            learning_rate = 0.001
            weight_decay = 1e-5
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    def train_svi_from_accessibility(self, graph_data, tract_svi, epochs=100, verbose=False):
        """
        Train GNN to predict SVI using accessibility features
        """
        self.model.train()
        target_svi = torch.FloatTensor([tract_svi])
        
        training_losses = []
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            predicted_svi = self.model(graph_data.x, graph_data.edge_index)
            
            # SVI prediction loss
            svi_loss = F.mse_loss(predicted_svi.mean().unsqueeze(0), target_svi)
            
            # Spatial variation encouragement
            spatial_variation = predicted_svi.std()
            variation_loss = F.relu(0.02 - spatial_variation)  # Encourage min 0.02 std
            
            # Constraint preservation (mean should equal tract SVI)
            constraint_loss = F.mse_loss(predicted_svi.mean().unsqueeze(0), target_svi)
            
            total_loss = svi_loss + 0.3 * variation_loss + 0.5 * constraint_loss
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            training_losses.append(total_loss.item())
            
            if verbose and epoch % 20 == 0:
                print(f"SVI Training Epoch {epoch}: Loss = {total_loss.item():.6f}, "
                      f"Std = {spatial_variation.item():.6f}")
        
        # Final prediction
        self.model.eval()
        with torch.no_grad():
            final_svi = self.model(graph_data.x, graph_data.edge_index)
        
        return {
            'svi_predictions': final_svi.detach().numpy(),
            'training_losses': training_losses,
            'final_loss': training_losses[-1],
            'spatial_variation': float(final_svi.std())
        }

class AccessibilityGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, embedding_dim=32):
        super().__init__()
        self.gnn_layers = torch.nn.ModuleList([
            GCNConv(input_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, embedding_dim)
        ])
        
        # Multi-task heads for different destination types
        self.employment_head = torch.nn.Linear(embedding_dim, 7)   # 7 employers
        self.healthcare_head = torch.nn.Linear(embedding_dim, 8)   # 8 hospitals  
        self.grocery_head = torch.nn.Linear(embedding_dim, 57)     # 57 stores
        
    def forward(self, x, edge_index):
        # Standard GNN forward pass
        for layer in self.gnn_layers[:-1]:
            x = F.relu(layer(x, edge_index))
        embeddings = self.gnn_layers[-1](x, edge_index)  # Final embeddings
        
        # Multi-task predictions
        employment_pred = self.employment_head(embeddings)
        healthcare_pred = self.healthcare_head(embeddings)
        grocery_pred = self.grocery_head(embeddings)
        
        return {
            'embeddings': embeddings,
            'employment': employment_pred,
            'healthcare': healthcare_pred, 
            'grocery': grocery_pred
        }

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
        Forward pass computing accessibility corrections with SVI range enforcement.
        
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
            and ensuring final predictions stay within [0, 1]
        """
        if idm_baseline is not None:
            num_addresses = idm_baseline.shape[0]
        else:
            num_addresses = x.shape[0]

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
        all_corrections = self.correction_head(h3)  # [total_nodes, 1]
        
        # SAFE BOUNDS CHECKING - ensure we don't exceed available corrections
        if num_addresses > all_corrections.shape[0]:
            num_addresses = all_corrections.shape[0]
        
        address_corrections = all_corrections[:num_addresses]  # Extract address corrections
        
        # Bound corrections to reasonable range using tanh
        address_corrections = torch.tanh(address_corrections) * self.max_correction
        
        # Ensure corrections are mean-centered (mass preserving)
        address_corrections = address_corrections - torch.mean(address_corrections)
        
        # NEW: Ensure final predictions stay in valid SVI range [0,1]
        if idm_baseline is not None:
            final_predictions = idm_baseline.squeeze() + address_corrections.squeeze()
            final_predictions = torch.clamp(final_predictions, min=0.0, max=1.0)
            # Recompute corrections to maintain range constraint
            address_corrections = (final_predictions - idm_baseline.squeeze()).unsqueeze(-1)
            # Re-center to preserve mass
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
            'constraint_weight': 5.0,         # MUCH HIGHER - enforce tract constraint
            'spatial_weight': 1.0,            # Moderate spatial smoothness  
            'diversity_weight': 0.5,          # MUCH LOWER - gentle diversity encouragement
            'variation_weight': 0.3,          # MUCH LOWER - gentle variation encouragement
            'variance_preservation_weight': 0.1,  # MUCH LOWER - gentle collapse prevention
            'feature_weight': 0.05,           # MUCH LOWER - minimal feature utilization
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
        Train the GNN with FIXED loss function to prevent parameter collapse.
        
        Key fixes:
        1. Rebalanced loss weights (constraint_weight=5.0, diversity_weight=0.5)
        2. SVI range enforcement [0, 1]
        3. Simplified training loop without numerical instabilities
        4. Proper gradient clipping
        """
        
        self.model.train()
        
        # Convert inputs to tensors
        idm_tensor = torch.tensor(idm_baseline, dtype=torch.float32)
        target_mean = torch.tensor(tract_svi, dtype=torch.float32)
        
        # FIXED loss weights - constraint is now dominant
        loss_weights = {
            'constraint_weight': 5.0,         # HIGH - enforce tract constraint
            'spatial_weight': 1.0,            # Moderate spatial smoothness  
            'diversity_weight': 0.5,          # LOW - gentle diversity encouragement
            'variation_weight': 0.3,          # LOW - gentle variation encouragement
            'variance_preservation_weight': 0.1,  # LOW - gentle collapse prevention
            'feature_weight': 0.05,           # MINIMAL - feature utilization
        }
        
        loss_history = []
        correction_history = []
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            corrections = self.model(
                graph_data.x, graph_data.edge_index, idm_tensor
            ).squeeze()
            
            # Ensure corrections match IDM size
            if corrections.shape[0] != idm_tensor.shape[0]:
                corrections = corrections[:idm_tensor.shape[0]]
            
            # Compute enhanced predictions with RANGE ENFORCEMENT
            enhanced_predictions = idm_tensor + corrections
            enhanced_predictions = torch.clamp(enhanced_predictions, min=0.0, max=1.0)  # CRITICAL FIX
            
            # SIMPLIFIED loss computation
            # 1. Constraint loss (HIGH weight)
            constraint_loss = F.mse_loss(torch.mean(enhanced_predictions), target_mean)
            
            # 2. Spatial smoothness (moderate weight)
            spatial_loss = self._compute_simple_spatial_loss(corrections, graph_data.edge_index)
            
            # 3. Diversity encouragement (low weight)
            diversity_loss = -torch.std(enhanced_predictions)  # Negative to encourage diversity
            
            # 4. Simple variance preservation (low weight)
            variance_loss = 1.0 / (1.0 + torch.var(corrections))  # Bounded, no exponentials
            
            # 5. Feature utilization (minimal weight)
            feature_loss = -torch.mean(torch.abs(corrections))  # Encourage meaningful corrections
            
            # REBALANCED total loss
            total_loss = (
                loss_weights['constraint_weight'] * constraint_loss +
                loss_weights['spatial_weight'] * spatial_loss +
                loss_weights['diversity_weight'] * diversity_loss +
                loss_weights['variance_preservation_weight'] * variance_loss +
                loss_weights['feature_weight'] * feature_loss
            )
            
            # Backward pass with gradient clipping
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track history
            loss_history.append(total_loss.item())
            correction_history.append(corrections.detach().numpy().copy())
            
            # Monitor progress
            if verbose and epoch % 10 == 0:
                pred_mean = torch.mean(enhanced_predictions).item()
                pred_std = torch.std(enhanced_predictions).item()
                corr_std = torch.std(corrections).item()
                constraint_error = abs(pred_mean - tract_svi)
                
                print(f"Epoch {epoch:3d}: Loss={total_loss.item():.4f} "
                    f"Constraint={constraint_loss.item():.4f} "
                    f"Mean={pred_mean:.4f} (target={tract_svi:.4f}) "
                    f"Std={pred_std:.4f} CorrStd={corr_std:.4f}")
                
                # Check for improvement
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                
                # Early warning for issues
                if constraint_error > 0.05:  # 5% error threshold
                    print(f"    ⚠️  High constraint error: {constraint_error:.4f}")
                if corr_std < 0.001:
                    print(f"    ⚠️  Very low correction variance: {corr_std:.6f}")
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            final_corrections = self.model(
                graph_data.x, graph_data.edge_index, idm_tensor
            ).squeeze()
            
            if final_corrections.shape[0] != idm_tensor.shape[0]:
                final_corrections = final_corrections[:idm_tensor.shape[0]]
                
            final_predictions = idm_tensor + final_corrections
            final_predictions = torch.clamp(final_predictions, min=0.0, max=1.0)
            
            # Final metrics
            final_constraint_error = abs(torch.mean(final_predictions) - target_mean).item()
            final_pred_std = torch.std(final_predictions).item()
            final_corr_std = torch.std(final_corrections).item()
            
            # Success criteria
            success_criteria = {
                'constraint_satisfied': final_constraint_error < 0.01,  # 1% tolerance
                'valid_range': torch.all(final_predictions >= 0.0) and torch.all(final_predictions <= 1.0),
                'meaningful_variation': final_pred_std > 0.005,  # At least some variation
                'stable_training': loss_history[-1] < loss_history[0] if len(loss_history) > 1 else True
            }
            
            training_success = all(success_criteria.values())
            
            if verbose:
                print(f"\n🏁 TRAINING COMPLETE:")
                print(f"   Final constraint error: {final_constraint_error:.4f}")
                print(f"   Final prediction std: {final_pred_std:.4f}")
                print(f"   Final correction std: {final_corr_std:.4f}")
                print(f"   Valid SVI range: {success_criteria['valid_range']}")
                print(f"   Training success: {'✅' if training_success else '❌'}")
        
        return {
            'success': training_success,
            'final_corrections': final_corrections.numpy(),
            'final_predictions': final_predictions.numpy(),
            'loss_history': loss_history,
            'correction_history': correction_history,
            'final_metrics': {
                'prediction_std': final_pred_std,
                'correction_std': final_corr_std,
                'constraint_error': final_constraint_error,
                'mean_prediction': torch.mean(final_predictions).item(),
                'valid_range': success_criteria['valid_range'],
                'parameter_collapse_detected': final_corr_std < 1e-4,
                'training_successful': training_success
            },
            'success_criteria': success_criteria
        }

    def _compute_simple_spatial_loss(self, corrections: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        SIMPLIFIED spatial loss without numerical instabilities.
        """
        if edge_index.shape[1] == 0:
            return torch.tensor(0.0)
        
        # Filter to valid address edges
        num_addresses = corrections.shape[0]
        valid_edge_mask = (edge_index[0] < num_addresses) & (edge_index[1] < num_addresses)
        
        if not valid_edge_mask.any():
            return torch.tensor(0.0)
        
        valid_edge_index = edge_index[:, valid_edge_mask]
        
        # Simple MSE between connected nodes
        source_corrections = corrections[valid_edge_index[0]]
        target_corrections = corrections[valid_edge_index[1]]
        
        return F.mse_loss(source_corrections, target_corrections)
    
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
    
    def _compute_variance_preservation_loss(self, corrections, predictions, target_mean):
        # SAFE variance preservation without explosions
        correction_variance = torch.var(corrections)
        
        # Use sigmoid instead of exponential to bound values
        collapse_penalty = 1.0 / (1.0 + correction_variance * 10.0)  # Bounded [0,1]
        
        # Simple L2 penalty instead of exponential
        magnitude_penalty = 1.0 / (1.0 + torch.mean(torch.abs(corrections)) * 5.0)
        
        return collapse_penalty + magnitude_penalty  # No huge multipliers

    
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