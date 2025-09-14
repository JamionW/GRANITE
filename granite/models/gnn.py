"""
Simplified GNN for direct accessibility → SVI prediction
Eliminates two-stage complexity and systematic bias issues
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, BatchNorm
from torch_geometric.data import Data
import numpy as np

class AccessibilitySVIGNN(nn.Module):
    def __init__(self, accessibility_features_dim, hidden_dim=64, dropout=0.5):  # Increased dropout
        super(AccessibilitySVIGNN, self).__init__()
        
        self.accessibility_features_dim = accessibility_features_dim
        self.hidden_dim = hidden_dim
        
        # More regularized architecture
        self.input_projection = nn.Linear(accessibility_features_dim, hidden_dim)
        self.input_norm = nn.BatchNorm1d(hidden_dim)
        
        # Reduced complexity to prevent overfitting
        self.attention_conv = GATConv(hidden_dim, hidden_dim//2, heads=2, concat=False, dropout=dropout)
        self.spatial_conv = GCNConv(hidden_dim, hidden_dim//2)
        self.spatial_norm = nn.BatchNorm1d(hidden_dim//2)
        
        # Simpler fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),  # attention + spatial = hidden_dim total
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # SVI prediction head with more regularization
        self.svi_head = nn.Sequential(
            nn.Linear(hidden_dim//4, 16),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(dropout/3),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, accessibility_features, edge_index):
        """Regularized forward pass"""
        # Input processing
        x = F.relu(self.input_norm(self.input_projection(accessibility_features)))
        
        # Attention processing
        x_att = F.elu(self.attention_conv(x, edge_index))
        x_att = self.dropout(x_att)
        
        # Spatial processing  
        x_spatial = F.relu(self.spatial_norm(self.spatial_conv(x, edge_index)))
        
        # Fusion
        x_fused = torch.cat([x_att, x_spatial], dim=1)  # Should be hidden_dim total
        x_final = self.fusion(x_fused)
        
        # SVI prediction
        svi_predictions = self.svi_head(x_final).squeeze()
        
        return svi_predictions

class AccessibilityGNNTrainer:
    """
    Trainer for accessibility → SVI prediction
    
    Includes anti-bias mechanisms:
    1. Balanced loss function
    2. Spatial constraint preservation
    3. Variation encouragement
    """
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or {}
        
        # Training parameters
        learning_rate = float(self.config.get('learning_rate', 0.001))
        weight_decay = float(self.config.get('weight_decay', 1e-4))
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.7
        )
        
    def train(self, graph_data, tract_svi, epochs=100, verbose=True):
        """
        Train GNN for accessibility → SVI prediction
        
        Loss components:
        1. Prediction accuracy (MSE)
        2. Spatial constraint (tract mean preservation)
        3. Variation encouragement (prevent over-smoothing)
        4. Regularization (prevent systematic bias)
        """
        self.model.train()
        target_svi = torch.FloatTensor([tract_svi])
        
        training_history = {
            'losses': [], 'prediction_errors': [], 'spatial_variations': []
        }
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            svi_predictions = self.model(graph_data.x, graph_data.edge_index)

            prediction_loss = F.mse_loss(svi_predictions.mean().unsqueeze(0), target_svi)

            bias_detected = self._detect_systematic_bias(svi_predictions, graph_data.x, epoch)
            
            # Constraint enforcement
            predicted_mean = svi_predictions.mean()
            constraint_loss = F.mse_loss(predicted_mean.unsqueeze(0), target_svi)
            
            # Spatial variation encouragement
            spatial_std = svi_predictions.std()
            variation_loss = F.relu(0.02 - spatial_std)
            
            # Prediction bounds
            extreme_penalty = torch.mean(F.relu(svi_predictions - 1.0)) + torch.mean(F.relu(-svi_predictions))
            
            if len(svi_predictions) > 10:
                total_loss = 2.0 * constraint_loss + 0.3 * variation_loss + 0.1 * extreme_penalty
            else:
                total_loss = constraint_loss + variation_loss + extreme_penalty
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping to prevent instability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step(total_loss)
            
            # Track metrics
            training_history['losses'].append(total_loss.item())
            training_history['prediction_errors'].append(prediction_loss.item())
            training_history['spatial_variations'].append(spatial_std.item())
            
            # Early stopping
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= 20:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
            
            # Progress reporting
            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch:3d}: Loss={total_loss.item():.6f}, "
                      f"Pred={prediction_loss.item():.6f}, "
                      f"Std={spatial_std.item():.4f}, "
                      f"LR={self.optimizer.param_groups[0]['lr']:.6f}")
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            final_predictions = self.model(graph_data.x, graph_data.edge_index)
            
        return {
            'final_predictions': final_predictions.detach().numpy(),
            'training_history': training_history,
            'final_spatial_std': float(final_predictions.std()),
            'constraint_error': float(abs(final_predictions.mean() - target_svi)),
            'epochs_trained': epoch + 1
        }
    
    def _detect_systematic_bias(self, predictions, accessibility_features, epoch):
        """Detect and warn about systematic bias during training"""
        if accessibility_features.shape[0] != len(predictions):
            return False
        
        # Calculate overall accessibility per address
        overall_accessibility = torch.mean(accessibility_features, dim=1)
        
        # Calculate correlation
        if len(predictions) > 10:  # Need sufficient data points
            correlation = torch.corrcoef(torch.stack([overall_accessibility, predictions]))[0,1]
            
            if torch.abs(correlation) > 0.6:  # Threshold for concern
                if epoch % 10 == 0:  # Don't spam warnings
                    print(f"    WARNING Epoch {epoch}: Systematic bias detected (r={correlation:.3f})")
                    print(f"    Consider reducing learning rate or increasing regularization")
                return True
        
        return False

def normalize_accessibility_features(features):
    """
    Robust feature normalization to prevent systematic bias
    
    Uses robust scaling (median/MAD) instead of mean/std
    to handle outliers better
    """
    from sklearn.preprocessing import RobustScaler
    
    scaler = RobustScaler()
    normalized_features = scaler.fit_transform(features)
    
    return normalized_features, scaler