"""
Mixture of Experts for GRANITE

Implements context-dependent accessibility-vulnerability prediction using
multiple specialized experts for different SVI contexts.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings

from granite.models.gnn import AccessibilitySVIGNN, set_random_seed, AccessibilityGNNTrainer

class ContextGatedGateNetwork(nn.Module):
    """
    Learns soft routing of addresses to appropriate experts based on accessibility and context.
    
    Input: 54 accessibility features + 5 context features (59-dim vector)
    Output: 3-way softmax distribution [p_low, p_medium, p_high]
    """
    def __init__(self, accessibility_dim=54, context_dim=5, hidden_dim=32):
        super(ContextGatedGateNetwork, self).__init__()
        
        input_dim = accessibility_dim + context_dim
        
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 3 experts
        )
        
        self.accessibility_dim = accessibility_dim
        self.context_dim = context_dim
    
    def forward(self, accessibility_features, context_features=None):
        """
        Route addresses to experts.
        
        Args:
            accessibility_features: (n_addresses, 54)
            context_features: (n_addresses, 5) or None
            
        Returns:
            gate_probs: (n_addresses, 3) - softmax probabilities for [low, medium, high]
        """
        if context_features is not None:
            x = torch.cat([accessibility_features, context_features], dim=1)
        else:
            # Default context if not provided
            batch_size = accessibility_features.shape[0]
            context_features = torch.zeros(batch_size, self.context_dim, 
                                          device=accessibility_features.device)
            x = torch.cat([accessibility_features, context_features], dim=1)
        
        logits = self.gate(x)
        gate_probs = F.softmax(logits, dim=1)  # (n_addresses, 3)
        
        return gate_probs


class MixtureOfExpertsGNN(nn.Module):
    """
    Mixture of Experts combining three specialist GNN experts trained on different SVI contexts.
    
    Experts:
    - expert_low: SVI 0.01-0.40 (suburban, car-dependent)
    - expert_medium: SVI 0.30-0.70 (transition, mixed)
    - expert_high: SVI 0.55-1.00 (urban, vulnerable)
    
    Routing:
    - Gate network learns to route addresses to appropriate experts
    - Soft mixture combines all three expert predictions
    """
    def __init__(self, accessibility_features_dim, context_features_dim=5,
                 hidden_dim=64, dropout=0.3, seed=42,
                 use_context_gating=True, use_multitask=True):
        super(MixtureOfExpertsGNN, self).__init__()
        
        set_random_seed(seed)
        
        # Three specialist experts
        self.expert_low = AccessibilitySVIGNN(
            accessibility_features_dim=accessibility_features_dim,
            context_features_dim=context_features_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            seed=seed,
            use_context_gating=use_context_gating,
            use_multitask=use_multitask
        )
        
        self.expert_medium = AccessibilitySVIGNN(
            accessibility_features_dim=accessibility_features_dim,
            context_features_dim=context_features_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            seed=seed + 1,
            use_context_gating=use_context_gating,
            use_multitask=use_multitask
        )
        
        self.expert_high = AccessibilitySVIGNN(
            accessibility_features_dim=accessibility_features_dim,
            context_features_dim=context_features_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            seed=seed + 2,
            use_context_gating=use_context_gating,
            use_multitask=use_multitask
        )
        
        # Gate network for soft routing
        self.gate_network = ContextGatedGateNetwork(
            accessibility_dim=accessibility_features_dim,
            context_dim=context_features_dim,
            hidden_dim=32
        )
        
        self.experts = [self.expert_low, self.expert_medium, self.expert_high]
        self.accessibility_features_dim = accessibility_features_dim
        self.context_features_dim = context_features_dim
        
    def forward(self, x, edge_index, context_features=None, 
                return_gate_weights=False, return_expert_predictions=False):
        """
        Mixture inference: route through experts, combine via gating.
        
        Args:
            x: accessibility features (n_addresses, 54)
            edge_index: graph edges
            context_features: (n_addresses, 5) socioeconomic context
            return_gate_weights: if True, return gate probabilities
            return_expert_predictions: if True, return individual expert outputs
            
        Returns:
            svi_predictions: (n_addresses,) - mixture predictions
            gate_weights: (n_addresses, 3) - if return_gate_weights=True
            expert_preds: list of 3 predictions - if return_expert_predictions=True
        """
        # Gate routing
        gate_weights = self.gate_network(x, context_features)  # (n_addresses, 3)
        
        # Expert predictions
        pred_low = self.expert_low(x, edge_index, return_accessibility=False, 
                                  context_features=context_features)
        pred_medium = self.expert_medium(x, edge_index, return_accessibility=False,
                                        context_features=context_features)
        pred_high = self.expert_high(x, edge_index, return_accessibility=False,
                                    context_features=context_features)
        
        # Soft mixture: weighted sum of expert predictions
        mixture = (
            gate_weights[:, 0].unsqueeze(1) * pred_low.unsqueeze(1) +
            gate_weights[:, 1].unsqueeze(1) * pred_medium.unsqueeze(1) +
            gate_weights[:, 2].unsqueeze(1) * pred_high.unsqueeze(1)
        ).squeeze(1)
        
        if return_gate_weights and return_expert_predictions:
            return mixture, gate_weights, [pred_low, pred_medium, pred_high]
        elif return_gate_weights:
            return mixture, gate_weights
        elif return_expert_predictions:
            return mixture, [pred_low, pred_medium, pred_high]
        else:
            return mixture
    
    def freeze_experts(self):
        """Freeze expert parameters for gate-only training."""
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False
    
    def unfreeze_experts(self):
        """Unfreeze expert parameters for fine-tuning."""
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = True
    
    def freeze_gate(self):
        """Freeze gate parameters."""
        for param in self.gate_network.parameters():
            param.requires_grad = False
    
    def unfreeze_gate(self):
        """Unfreeze gate parameters."""
        for param in self.gate_network.parameters():
            param.requires_grad = True


class MixtureOfExpertsTrainer:
    """
    Training orchestrator for Mixture of Experts.
    
    Workflow:
    1. Stratify training data by SVI range
    2. Train three experts independently on homogeneous data
    3. Train gate network with frozen experts
    4. Optional: fine-tune all parameters jointly
    """
    
    def __init__(self, model, config=None, seed=42):
        self.model = model
        self.config = config or {}
        self.seed = seed
        
        set_random_seed(seed)
        
        self.learning_rate = float(self.config.get('learning_rate', 0.001))
        self.weight_decay = float(self.config.get('weight_decay', 1e-4))
        
        # Expert training configuration
        self.expert_epochs = self.config.get('expert_epochs', 150)
        self.gate_epochs = self.config.get('gate_epochs', 100)
        self.finetune_epochs = self.config.get('finetune_epochs', 50)
        
        # SVI stratification boundaries
        self.svi_boundaries = {
            'low': (0.01, 0.40),
            'medium': (0.30, 0.70),
            'high': (0.55, 1.00)
        }
        
        self.training_history = {
            'expert_low': [],
            'expert_medium': [],
            'expert_high': [],
            'gate': [],
            'finetune': []
        }
    
    def stratify_training_data(self, graph_data_list, tract_svi_list):
        """
        Partition training data into expert-specific subsets.
        
        Args:
            graph_data_list: List of PyTorch Geometric Data objects
            tract_svi_list: List of tract SVI values
            
        Returns:
            Dict with stratified data:
            {
                'low': {'data': [...], 'svi': [...]},
                'medium': {'data': [...], 'svi': [...]},
                'high': {'data': [...], 'svi': [...]}
            }
        """
        stratified = {
            'low': {'data': [], 'svi': []},
            'medium': {'data': [], 'svi': []},
            'high': {'data': [], 'svi': []}
        }
        
        for graph_data, svi in zip(graph_data_list, tract_svi_list):
            # Low SVI expert (suburbs, car-dependent)
            if svi < self.svi_boundaries['low'][1]:
                stratified['low']['data'].append(graph_data)
                stratified['low']['svi'].append(svi)
            
            # Medium SVI expert (transition zones)
            if self.svi_boundaries['medium'][0] <= svi <= self.svi_boundaries['medium'][1]:
                stratified['medium']['data'].append(graph_data)
                stratified['medium']['svi'].append(svi)
            
            # High SVI expert (urban cores)
            if svi > self.svi_boundaries['high'][0]:
                stratified['high']['data'].append(graph_data)
                stratified['high']['svi'].append(svi)
        
        return stratified
    
    def train_single_expert(self, expert, graph_data_list, tract_svi_list,
                          expert_name='expert', verbose=True):
        """
        Train a single expert on stratified data.
        
        Args:
            expert: AccessibilitySVIGNN model
            graph_data_list: List of training graphs
            tract_svi_list: List of training SVI values
            expert_name: Name for logging
            verbose: Print progress
            
        Returns:
            Dict with training history
        """
        if len(graph_data_list) == 0:
            warnings.warn(f"No training data for {expert_name}, skipping")
            return {'status': 'skipped'}
        
        if verbose:
            print(f"\nTraining {expert_name} on {len(graph_data_list)} tracts...")
        
        expert.train()
        optimizer = torch.optim.Adam(expert.parameters(), 
                                    lr=self.learning_rate,
                                    weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.8, min_lr=1e-5
        )
        
        history = {'losses': [], 'constraint_errors': []}
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.expert_epochs):
            epoch_loss = 0.0
            
            for graph_data, tract_svi in zip(graph_data_list, tract_svi_list):
                optimizer.zero_grad()
                
                context = getattr(graph_data, 'context', None)
                predictions = expert(graph_data.x, graph_data.edge_index,
                                   context_features=context)
                
                # Constraint loss: match tract SVI mean
                constraint_loss = torch.abs(predictions.mean() - tract_svi)

                # Variation loss: encourage spread, penalize constant output
                pred_std = predictions.std()
                variation_loss = torch.exp(-pred_std * 10)  # Penalize low variance

                # Bounds loss: keep predictions in valid SVI range [0, 1]
                bounds_loss = (F.relu(-predictions) + F.relu(predictions - 1)).mean()

                loss = 2.0 * constraint_loss + 0.5 * variation_loss + bounds_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(expert.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= len(graph_data_list)
            scheduler.step(epoch_loss)
            
            history['losses'].append(epoch_loss)
            
            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch}")
                    break
            
            if verbose and (epoch % 20 == 0 or epoch == self.expert_epochs - 1):
                print(f"  Epoch {epoch:3d}: loss={epoch_loss:.6f}")
        
        return {
            'status': 'completed',
            'epochs': epoch,
            'final_loss': epoch_loss,
            'history': history
        }
    
    def train_gate_network(self, graph_data_list, tract_svi_list, verbose=True):
        """
        Train gate network with frozen experts.
        
        Args:
            graph_data_list: List of training graphs
            tract_svi_list: List of training SVI values
            verbose: Print progress
            
        Returns:
            Dict with training history
        """
        if verbose:
            print(f"\nTraining gate network with frozen experts...")
        
        # Freeze experts, unfreeze gate
        self.model.freeze_experts()
        self.model.unfreeze_gate()
        
        gate_optimizer = torch.optim.Adam(
            self.model.gate_network.parameters(),
            lr=self.learning_rate * 2,  # Gate learns faster
            weight_decay=self.weight_decay
        )
        
        history = {'losses': [], 'gate_entropies': []}
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.gate_epochs):
            epoch_loss = 0.0
            epoch_entropy = 0.0
            
            for graph_data, tract_svi in zip(graph_data_list, tract_svi_list):
                gate_optimizer.zero_grad()
                
                context = getattr(graph_data, 'context', None)
                
                # Mixture forward pass
                mixture, gate_weights = self.model(
                    graph_data.x, graph_data.edge_index,
                    context_features=context,
                    return_gate_weights=True
                )
                
                # Gate loss: mixture should match tract SVI
                gate_loss = torch.abs(mixture.mean() - tract_svi)
                
                # Regularize to prevent gate collapsing to single expert
                # (maximize entropy of gate weights)
                gate_entropy = -torch.mean(
                    torch.sum(gate_weights * torch.log(gate_weights + 1e-8), dim=1)
                )
                
                total_loss = gate_loss - 2.00 * gate_entropy
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.gate_network.parameters(),
                                              max_norm=1.0)
                gate_optimizer.step()
                
                epoch_loss += gate_loss.item()
                epoch_entropy += gate_entropy.item()
            
            epoch_loss /= len(graph_data_list)
            epoch_entropy /= len(graph_data_list)
            
            history['losses'].append(epoch_loss)
            history['gate_entropies'].append(epoch_entropy)
            
            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch}")
                    break
            
            if verbose and (epoch % 20 == 0 or epoch == self.gate_epochs - 1):
                print(f"  Epoch {epoch:3d}: gate_loss={epoch_loss:.6f}, entropy={epoch_entropy:.4f}")
        
        return {
            'status': 'completed',
            'epochs': epoch,
            'final_loss': epoch_loss,
            'history': history
        }
    
    def train(self, graph_data_list, tract_svi_list, finetune=False, verbose=True):
        """
        Complete MoE training: experts -> gate -> optional fine-tuning.
        
        Args:
            graph_data_list: List of training graphs
            tract_svi_list: List of training SVI values
            finetune: Whether to do joint fine-tuning after gate training
            verbose: Print progress
            
        Returns:
            Dict with complete training results
        """
        if verbose:
            print(f"\n{'='*80}")
            print("MIXTURE OF EXPERTS TRAINING")
            print(f"{'='*80}")
            print(f"Total training tracts: {len(graph_data_list)}")
            print(f"SVI range: {min(tract_svi_list):.3f} - {max(tract_svi_list):.3f}")
        
        set_random_seed(self.seed)
        
        # Step 1: Stratify data
        if verbose:
            print(f"\nStep 1: Stratifying training data by SVI...")
        
        stratified = self.stratify_training_data(graph_data_list, tract_svi_list)
        
        for expert_name, data in stratified.items():
            if verbose:
                print(f"  {expert_name}: {len(data['data'])} tracts, "
                      f"SVI range {min(data['svi']):.3f}-{max(data['svi']):.3f}")
        
        # Step 2: Train experts independently
        if verbose:
            print(f"\nStep 2: Training specialist experts on homogeneous data...")
        
        expert_results = {
            'low': self.train_single_expert(
                self.model.expert_low,
                stratified['low']['data'],
                stratified['low']['svi'],
                expert_name='Expert_Low (suburbs)',
                verbose=verbose
            ),
            'medium': self.train_single_expert(
                self.model.expert_medium,
                stratified['medium']['data'],
                stratified['medium']['svi'],
                expert_name='Expert_Medium (transition)',
                verbose=verbose
            ),
            'high': self.train_single_expert(
                self.model.expert_high,
                stratified['high']['data'],
                stratified['high']['svi'],
                expert_name='Expert_High (urban)',
                verbose=verbose
            )
        }
        
        # Step 3: Train gate network
        if verbose:
            print(f"\nStep 3: Training gate network with frozen experts...")
        
        gate_result = self.train_gate_network(
            graph_data_list, tract_svi_list, verbose=verbose
        )
        
        # Step 4: Optional fine-tuning
        if finetune and verbose:
            print(f"\nStep 4: Joint fine-tuning of all parameters...")
            self.model.unfreeze_experts()
            self.finetune(graph_data_list, tract_svi_list, verbose=verbose)
        
        return {
            'expert_results': expert_results,
            'gate_result': gate_result,
            'stratification': {k: len(v['data']) for k, v in stratified.items()}
        }
    
    def finetune(self, graph_data_list, tract_svi_list, verbose=True):
        """
        Joint fine-tuning of experts and gate (optional, after initial training).
        """
        self.model.train()
        
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate * 0.1,  # Smaller learning rate for fine-tuning
            weight_decay=self.weight_decay
        )
        
        history = {'losses': []}
        
        for epoch in range(self.finetune_epochs):
            epoch_loss = 0.0
            
            for graph_data, tract_svi in zip(graph_data_list, tract_svi_list):
                optimizer.zero_grad()
                
                context = getattr(graph_data, 'context', None)
                mixture = self.model(graph_data.x, graph_data.edge_index,
                                    context_features=context)
                
                loss = torch.abs(mixture.mean() - tract_svi)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                              max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= len(graph_data_list)
            history['losses'].append(epoch_loss)
            
            if verbose and (epoch % 10 == 0):
                print(f"  Fine-tune epoch {epoch:3d}: loss={epoch_loss:.6f}")
        
        self.training_history['finetune'] = history


class MoEInferenceAnalyzer:
    """Analyze MoE inference patterns and expert utilization."""
    
    def __init__(self, model, verbose=True):
        self.model = model
        self.verbose = verbose
    
    def analyze_expert_usage(self, graph_data_list, tract_svi_list):
        """
        Analyze which experts are selected for which tracts.
        
        Returns:
            Dict with expert usage patterns
        """
        self.model.eval()
        
        usage = {
            'low': [],
            'medium': [],
            'high': [],
            'gate_weights': []
        }
        
        with torch.no_grad():
            for graph_data, tract_svi in zip(graph_data_list, tract_svi_list):
                context = getattr(graph_data, 'context', None)
                
                _, gate_weights = self.model(
                    graph_data.x, graph_data.edge_index,
                    context_features=context,
                    return_gate_weights=True
                )
                
                # Average gate weights for this tract
                mean_weights = gate_weights.mean(dim=0).cpu().numpy()
                usage['gate_weights'].append(mean_weights)
                usage['low'].append(mean_weights[0])
                usage['medium'].append(mean_weights[1])
                usage['high'].append(mean_weights[2])
        
        return {
            'expert_low_usage': np.array(usage['low']),
            'expert_medium_usage': np.array(usage['medium']),
            'expert_high_usage': np.array(usage['high']),
            'tract_svi': np.array(tract_svi_list),
            'all_gate_weights': np.array(usage['gate_weights'])
        }
    
    def get_dominant_expert(self, gate_weights):
        """For each address, return which expert was selected."""
        return np.argmax(gate_weights, axis=1)
    
    def get_expert_confidence(self, gate_weights):
        """For each address, return the max gate weight (confidence in expert selection)."""
        return np.max(gate_weights, axis=1)


def create_moe_model(accessibility_features_dim, context_features_dim=5,
                     hidden_dim=64, dropout=0.3, seed=42):
    """Factory function to create initialized MoE model."""
    model = MixtureOfExpertsGNN(
        accessibility_features_dim=accessibility_features_dim,
        context_features_dim=context_features_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        seed=seed,
        use_context_gating=True,
        use_multitask=True
    )
    return model