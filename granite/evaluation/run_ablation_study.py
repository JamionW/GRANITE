"""
GRANITE Robust Ablation Study (GNN Version)

Answers: Do accessibility features carry signal for SVI prediction,
or does constraint enforcement do all the work?

Uses the ACTUAL GNN architecture (AccessibilitySVIGNN) - not a simplified MLP.
This properly tests whether GRANITE learns from accessibility patterns.

Features:
- Checkpointing: saves progress every N tracts, resumes on restart
- Heartbeat: periodic status updates for long runs
- Graceful shutdown: Ctrl+C saves state before exit
- Dual mode: fast Euclidean or cached OSRM features
- Curated tracts: uses validated train/test split
- ACTUAL GNN: Uses AccessibilitySVIGNN with graph convolution
"""
import os
import sys
import time
import json
import signal
import pickle
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/workspaces/GRANITE')


# =============================================================================
# CONFIGURATION
# =============================================================================

class AblationConfig:
    """Central configuration for ablation study."""
    
    def __init__(
        self,
        output_dir: str = './output/ablation_study',
        checkpoint_every: int = 5,        # save every N tracts
        heartbeat_every: int = 60,        # log status every N seconds
        epochs: int = 100,
        batch_size: int = 256,
        hidden_dim: int = 64,
        learning_rate: float = 0.001,
        seed: int = 42,
        use_cached_osrm: bool = True,     # try OSRM cache first
        n_train_tracts: int = 12,
        n_test_tracts: int = 10,
    ):
        self.output_dir = Path(output_dir)
        self.checkpoint_every = checkpoint_every
        self.heartbeat_every = heartbeat_every
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.seed = seed
        self.use_cached_osrm = use_cached_osrm
        self.n_train_tracts = n_train_tracts
        self.n_test_tracts = n_test_tracts
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / 'checkpoint.pkl'
        self.results_path = self.output_dir / 'ablation_results.json'
        self.log_path = self.output_dir / 'ablation_log.txt'


# =============================================================================
# CURATED TRACT LISTS (from holdout validation)
# =============================================================================

def get_curated_training_tracts() -> List[str]:
    """12 non-overlapping training tracts with balanced SVI coverage."""
    return [
        '47065012000', '47065011205', '47065011100', # Very Low SVI
        '47065000600', '47065010413', '47065010501', # Low SVI
        '47065012400', '47065002800',                  # Medium SVI
        '47065010902', '47065011442',                  # High SVI
        '47065003000', '47065001300',                  # Very High SVI
    ]


def get_curated_test_tracts() -> List[str]:
    """10 non-overlapping test tracts with balanced SVI coverage."""
    return [
        '47065000700', '47065010411',                  # Very Low SVI
        '47065011900', '47065010502',                  # Low SVI
        '47065010433', '47065000800',                  # Medium SVI
        '47065011206', '47065011444',                  # High SVI
        '47065000400', '47065012300',                  # Very High SVI
    ]


# =============================================================================
# LOGGING & HEARTBEAT
# =============================================================================

class AblationLogger:
    """Logger with heartbeat and file output."""
    
    def __init__(self, config: AblationConfig):
        self.config = config
        self.start_time = time.time()
        self.last_heartbeat = time.time()
        self.log_file = open(config.log_path, 'a')
        
    def log(self, message: str, level: str = 'INFO'):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {level}: {message}"
        print(line)
        self.log_file.write(line + '\n')
        self.log_file.flush()
        
    def heartbeat(self, current_step: int, total_steps: int, extra: str = ''):
        """Periodic status update."""
        now = time.time()
        if now - self.last_heartbeat < self.config.heartbeat_every:
            return
            
        self.last_heartbeat = now
        elapsed = now - self.start_time
        
        if current_step > 0:
            rate = current_step / elapsed
            remaining = (total_steps - current_step) / rate if rate > 0 else 0
            eta = datetime.now() + timedelta(seconds=remaining)
            
            self.log(
                f"HEARTBEAT: {current_step}/{total_steps} "
                f"({100*current_step/total_steps:.1f}%) | "
                f"Elapsed: {elapsed/60:.1f}m | "
                f"ETA: {eta.strftime('%H:%M:%S')} | {extra}"
            )
        else:
            self.log(f"HEARTBEAT: Starting... | {extra}")
            
    def close(self):
        elapsed = time.time() - self.start_time
        self.log(f"Total runtime: {elapsed/60:.1f} minutes")
        self.log_file.close()


# =============================================================================
# CHECKPOINTING
# =============================================================================

class CheckpointManager:
    """Save and restore ablation progress."""
    
    def __init__(self, config: AblationConfig):
        self.config = config
        
    def save(self, state: dict):
        """Save checkpoint to disk."""
        state['timestamp'] = datetime.now().isoformat()
        with open(self.config.checkpoint_path, 'wb') as f:
            pickle.dump(state, f)
            
    def load(self) -> Optional[dict]:
        """Load checkpoint if exists."""
        if self.config.checkpoint_path.exists():
            try:
                with open(self.config.checkpoint_path, 'rb') as f:
                    state = pickle.load(f)
                return state
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
        return None
        
    def clear(self):
        """Remove checkpoint after successful completion."""
        if self.config.checkpoint_path.exists():
            self.config.checkpoint_path.unlink()


# =============================================================================
# FEATURE COMPUTATION
# =============================================================================

def compute_euclidean_accessibility(
    addresses_gdf, 
    destinations_gdf, 
    dest_type: str
) -> np.ndarray:
    """
    Compute 10 accessibility features using Euclidean distances.
    Fast proxy for network distances (~1000x faster than OSRM).
    """
    addr_coords = np.column_stack([
        addresses_gdf.geometry.x.values,
        addresses_gdf.geometry.y.values
    ])
    
    dest_coords = np.column_stack([
        destinations_gdf.geometry.x.values,
        destinations_gdf.geometry.y.values
    ])
    
    tree = cKDTree(dest_coords)
    n_addresses = len(addr_coords)
    
    k = min(20, len(dest_coords))
    distances, indices = tree.query(addr_coords, k=k)
    
    # Convert degrees to km at ~35°N latitude
    km_per_degree = 111 * np.cos(np.radians(35))
    distances_km = distances * km_per_degree
    
    features = np.zeros((n_addresses, 10))
    
    # Distance features
    features[:, 0] = distances_km[:, 0]                    # min_distance
    features[:, 1] = np.mean(distances_km, axis=1)         # mean_distance
    features[:, 2] = np.median(distances_km, axis=1)       # median_distance
    features[:, 3] = np.std(distances_km, axis=1)          # distance_std
    
    # Count features (destinations within threshold)
    features[:, 4] = np.sum(distances_km < 1.0, axis=1)    # count_1km
    features[:, 5] = np.sum(distances_km < 3.0, axis=1)    # count_3km
    features[:, 6] = np.sum(distances_km < 5.0, axis=1)    # count_5km
    features[:, 7] = np.sum(distances_km < 10.0, axis=1)   # count_10km
    
    # Concentration features
    features[:, 8] = distances_km[:, 0] / (distances_km[:, -1] + 0.01) # concentration
    features[:, 9] = np.max(distances_km, axis=1) - np.min(distances_km, axis=1) # range
    
    return features


def try_load_cached_osrm_features(
    tract_fips: str, 
    cache_dir: str = './granite_cache/absolute'
) -> Optional[np.ndarray]:
    """
    Try to load pre-computed OSRM features from cache.
    
    NOTE: Disabled for ablation study - cache lookup was returning wrong data.
    The cache uses MD5 hashes that we can't reliably reconstruct here.
    """
    # Disabled: cache lookup was broken and returned mismatched features
    return None


def compute_tract_features(
    addresses, 
    data: dict, 
    config: AblationConfig
) -> np.ndarray:
    """Compute accessibility features for a tract."""
    
    # Try cached OSRM first if enabled
    if config.use_cached_osrm:
        cached = try_load_cached_osrm_features(
            addresses.iloc[0].get('tract_fips', 'unknown')
        )
        if cached is not None and len(cached) == len(addresses):
            return cached[:, :30] # First 30 features (base accessibility)
    
    # Fall back to Euclidean
    emp_features = compute_euclidean_accessibility(
        addresses, data['employment'], 'employment'
    )
    health_features = compute_euclidean_accessibility(
        addresses, data['healthcare'], 'healthcare'
    )
    grocery_features = compute_euclidean_accessibility(
        addresses, data['grocery'], 'grocery'
    )
    
    return np.hstack([emp_features, health_features, grocery_features])


# =============================================================================
# GRAPH CONSTRUCTION (Inductive Learning Approach)
# =============================================================================

def create_unified_graph(
    tract_data: Dict[str, dict],
    train_fips: List[str],
    test_fips: List[str],
    use_random: bool = False,
    use_constant: bool = False,
    use_coordinates: bool = False,
    use_laplacian: bool = False,
    feature_indices: Optional[List[int]] = None,
    k: int = 8
) -> Tuple[Data, Dict[str, torch.BoolTensor], Dict[str, float], torch.BoolTensor, torch.BoolTensor, RobustScaler]:
    """
    Create ONE unified graph containing all tracts for inductive learning.
    
    Key insight: GNNs need consistent graph topology between train and test.
    We build one graph, then mask which nodes to train on vs evaluate.
    
    Returns:
        graph: Combined PyG Data object with ALL tracts
        tract_masks: Dict mapping FIPS -> boolean mask for each tract's nodes
        tract_svis: Dict mapping FIPS -> SVI value
        train_mask: Boolean mask for all training nodes
        test_mask: Boolean mask for all test nodes
        scaler: Feature scaler (fit on training data only)
    """
    all_fips = train_fips + test_fips
    
    # First pass: collect all data and compute total size
    all_features = []
    all_coords = []
    tract_node_ranges = {} # {fips: (start_idx, end_idx)}
    tract_svis = {}
    expected_n_features = None
    
    current_idx = 0
    
    for fips in all_fips:
        if fips not in tract_data:
            continue
            
        td = tract_data[fips]
        n_addr = td['n_addresses']
        
        features = td['features'].copy()
        features = np.nan_to_num(features, nan=0.0)
        
        # Validate feature dimensions
        if expected_n_features is None:
            expected_n_features = features.shape[1]
        elif features.shape[1] != expected_n_features:
            raise ValueError(
                f"Feature dimension mismatch for tract {fips}: "
                f"expected {expected_n_features}, got {features.shape[1]}"
            )
        
        all_features.append(features)
        all_coords.append(td['coords'])
        
        tract_node_ranges[fips] = (current_idx, current_idx + n_addr)
        tract_svis[fips] = td['svi']
        
        current_idx += n_addr
    
    if len(all_features) == 0:
        raise ValueError("No valid tracts to create graph")
    
    # Combine all data
    combined_features = np.vstack(all_features)
    combined_coords = np.vstack(all_coords)
    n_total = len(combined_features)
    
    # Normalize using ONLY training data (proper ML practice)
    train_indices = []
    for fips in train_fips:
        if fips in tract_node_ranges:
            start, end = tract_node_ranges[fips]
            train_indices.extend(range(start, end))
    
    scaler = RobustScaler()
    scaler.fit(combined_features[train_indices]) # Fit only on training
    features_normalized = scaler.transform(combined_features) # Transform all
    features_normalized = np.nan_to_num(features_normalized, nan=0.0)
    
    # Feature replacement options
    if use_constant:
        # Pure topology test: all nodes get identical constant features
        features_normalized = np.ones_like(features_normalized)
    elif use_random:
        # Random baseline: replace with random noise
        features_normalized = np.random.randn(*features_normalized.shape)
    elif use_coordinates:
        # Pure position: just x,y coordinates (normalized)
        coord_scaler = RobustScaler()
        features_normalized = coord_scaler.fit_transform(combined_coords)
    elif use_laplacian:
        # Laplacian positional encoding: eigenvectors of graph Laplacian
        # First build the graph to get edge_index
        k_graph = min(k, n_total - 1)
        if k_graph < 1:
            k_graph = 1
        nbrs = NearestNeighbors(n_neighbors=k_graph + 1, algorithm='ball_tree')
        nbrs.fit(combined_coords)
        _, indices = nbrs.kneighbors(combined_coords)
        
        # Build adjacency matrix
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        
        rows, cols = [], []
        for i in range(n_total):
            for j in indices[i][1:]:
                rows.extend([i, j])
                cols.extend([j, i])
        
        adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_total, n_total))
        
        # Compute normalized Laplacian
        degree = np.array(adj.sum(axis=1)).flatten()
        degree_inv_sqrt = np.power(degree, -0.5, where=degree > 0)
        degree_inv_sqrt[degree == 0] = 0
        D_inv_sqrt = csr_matrix((degree_inv_sqrt, (range(n_total), range(n_total))))
        L_norm = csr_matrix(np.eye(n_total)) - D_inv_sqrt @ adj @ D_inv_sqrt
        
        # Get smallest eigenvectors (excluding trivial one)
        n_pe = min(8, n_total - 2) # 8 positional encoding dimensions
        try:
            eigenvalues, eigenvectors = eigsh(L_norm, k=n_pe + 1, which='SM')
            features_normalized = eigenvectors[:, 1:n_pe + 1] # Skip first (constant)
        except:
            # Fallback to coordinates if eigsh fails
            coord_scaler = RobustScaler()
            features_normalized = coord_scaler.fit_transform(combined_coords)
    
    # Feature subsetting (for pruned experiments)
    if feature_indices is not None and not use_constant and not use_random and not use_coordinates and not use_laplacian:
        features_normalized = features_normalized[:, feature_indices]
    
    # Build KNN graph on combined coordinates
    k = min(k, n_total - 1)
    if k < 1:
        k = 1
        
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree')
    nbrs.fit(combined_coords)
    _, indices = nbrs.kneighbors(combined_coords)
    
    edge_list = []
    for i in range(n_total):
        for j in indices[i][1:]: # Skip self
            edge_list.extend([[i, j], [j, i]])
    
    edge_index = torch.LongTensor(edge_list).t().contiguous()
    edge_weight = torch.ones(edge_index.shape[1])
    
    # Create graph
    graph = Data(
        x=torch.FloatTensor(features_normalized),
        edge_index=edge_index,
        edge_attr=edge_weight
    )
    
    # Create masks
    tract_masks = {}
    train_mask = torch.zeros(n_total, dtype=torch.bool)
    test_mask = torch.zeros(n_total, dtype=torch.bool)
    
    for fips, (start, end) in tract_node_ranges.items():
        # Per-tract mask
        mask = torch.zeros(n_total, dtype=torch.bool)
        mask[start:end] = True
        tract_masks[fips] = mask
        
        # Train/test split masks
        if fips in train_fips:
            train_mask[start:end] = True
        elif fips in test_fips:
            test_mask[start:end] = True
    
    return graph, tract_masks, tract_svis, train_mask, test_mask, scaler


# =============================================================================
# MODEL (Using actual GRANITE GNN)
# =============================================================================

def create_gnn_model(n_features: int, hidden_dim: int = 64, seed: int = 42):
    """Create the actual GRANITE GNN model."""
    from granite.models.gnn import AccessibilitySVIGNN, set_random_seed
    
    set_random_seed(seed)
    
    model = AccessibilitySVIGNN(
        accessibility_features_dim=n_features,
        context_features_dim=5, # Standard context dim
        hidden_dim=hidden_dim,
        dropout=0.3,
        seed=seed,
        use_context_gating=False, # Simpler for ablation
        use_multitask=False
    )
    
    return model


def train_gnn_inductive(
    graph: Data,
    tract_masks: Dict[str, torch.BoolTensor],
    tract_svis: Dict[str, float],
    train_mask: torch.BoolTensor,
    train_fips: List[str],
    config: AblationConfig,
    logger: AblationLogger
) -> nn.Module:
    """
    Train GNN using inductive approach: only compute loss on training nodes,
    but message passing occurs over the ENTIRE graph (including test nodes).
    
    This is the correct GNN evaluation protocol.
    """
    from granite.models.gnn import set_random_seed
    import torch.nn.functional as F
    
    set_random_seed(config.seed)
    
    n_features = graph.x.shape[1]
    model = create_gnn_model(n_features, config.hidden_dim, config.seed)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.8, min_lr=1e-5
    )
    
    # Only use training tract SVIs for loss
    train_tract_targets = {fips: torch.FloatTensor([tract_svis[fips]]) 
                          for fips in train_fips if fips in tract_svis}
    
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.epochs):
        optimizer.zero_grad()
        
        # Forward pass on ENTIRE graph (message passing includes test nodes)
        predictions = model(graph.x, graph.edge_index)
        
        # Compute loss ONLY on training tracts
        constraint_losses = []
        for fips, target_svi in train_tract_targets.items():
            mask = tract_masks[fips]
            tract_preds = predictions[mask]
            if len(tract_preds) > 0:
                tract_mean = tract_preds.mean()
                loss = F.mse_loss(tract_mean.unsqueeze(0), target_svi)
                constraint_losses.append(loss)
        
        if constraint_losses:
            constraint_loss = torch.mean(torch.stack(constraint_losses))
        else:
            constraint_loss = torch.tensor(0.0)
        
        # Variation on training nodes only
        train_preds = predictions[train_mask]
        spatial_std = train_preds.std() if len(train_preds) > 0 else torch.tensor(0.0)
        variation_loss = F.relu(0.02 - spatial_std)
        
        # Bounds on all predictions
        bounds_loss = torch.mean(F.relu(predictions - 1.0)) + torch.mean(F.relu(-predictions))
        
        # Combined loss
        total_loss = (
            2.0 * constraint_loss +
            0.8 * variation_loss +
            1.0 * bounds_loss
        )
        
        if torch.isnan(total_loss):
            logger.log(f" Epoch {epoch}: NaN loss, skipping", level='WARN')
            continue
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(total_loss)
        
        # Early stopping
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= 15:
            logger.log(f" Early stopping at epoch {epoch}")
            break
        
        if (epoch + 1) % 25 == 0:
            errors = []
            for fips, target_svi in train_tract_targets.items():
                mask = tract_masks[fips]
                tract_preds = predictions[mask]
                if len(tract_preds) > 0:
                    pred_mean = tract_preds.mean().item()
                    target = target_svi.item()
                    if target > 0:
                        err = abs(pred_mean - target) / target * 100
                        errors.append(err)
            
            mean_err = np.mean(errors) if errors else 0
            logger.log(f" Epoch {epoch+1}: loss={total_loss.item():.4f}, "
                      f"constraint_err={mean_err:.1f}%, std={spatial_std.item():.4f}")
    
    return model


def evaluate_gnn_inductive(
    model: nn.Module,
    graph: Data,
    tract_masks: Dict[str, torch.BoolTensor],
    tract_svis: Dict[str, float],
    test_fips: List[str],
    logger: AblationLogger
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate on test tracts using inductive approach.
    
    The model has already seen test node FEATURES during message passing,
    but never saw test node LABELS during training.
    """
    model.eval()
    
    test_actual = []
    test_predicted = []
    
    with torch.no_grad():
        # One forward pass on entire graph
        predictions = model(graph.x, graph.edge_index)
        
        for fips in test_fips:
            if fips not in tract_masks or fips not in tract_svis:
                continue
            
            mask = tract_masks[fips]
            tract_preds = predictions[mask].numpy().flatten()
            
            if len(tract_preds) == 0:
                continue
            
            pred_mean = np.mean(tract_preds)
            actual_svi = tract_svis[fips]
            
            test_actual.append(actual_svi)
            test_predicted.append(pred_mean)
            
            logger.log(f" {fips}: actual={actual_svi:.3f}, pred={pred_mean:.3f}, "
                      f"err={abs(pred_mean-actual_svi)/actual_svi*100:.1f}%")
    
    return np.array(test_actual), np.array(test_predicted)


# =============================================================================
# MAIN ABLATION LOGIC
# =============================================================================

class RobustAblationStudy:
    """Main ablation study runner with robustness features."""
    
    def __init__(self, config: AblationConfig):
        self.config = config
        self.logger = AblationLogger(config)
        self.checkpoint = CheckpointManager(config)
        self.shutdown_requested = False
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        self.logger.log("Shutdown requested, saving checkpoint...")
        self.shutdown_requested = True
        
    def run(self) -> dict:
        """Run the full ablation study."""
        
        self.logger.log("="*70)
        self.logger.log("GRANITE ROBUST ABLATION STUDY")
        self.logger.log("="*70)
        
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        
        # Check for existing checkpoint
        state = self.checkpoint.load()
        if state:
            self.logger.log(f"Resuming from checkpoint (saved {state['timestamp']})")
            tract_data = state.get('tract_data', {})
            processed_tracts = state.get('processed_tracts', [])
        else:
            tract_data = {}
            processed_tracts = []
        
        # Load data
        self.logger.log("Loading spatial data...")
        data = self._load_data()
        
        if data is None:
            self.logger.log("Failed to load data", level='ERROR')
            return {'error': 'data_load_failed'}
        
        tracts = data['tracts']
        
        # Get tract lists
        train_fips = get_curated_training_tracts()[:self.config.n_train_tracts]
        test_fips = get_curated_test_tracts()[:self.config.n_test_tracts]
        all_fips = train_fips + test_fips
        
        self.logger.log(f"Training tracts: {len(train_fips)}")
        self.logger.log(f"Test tracts: {len(test_fips)}")
        
        # Compute features for each tract
        self.logger.log("\nComputing accessibility features...")
        total_tracts = len(all_fips)
        
        for i, fips in enumerate(all_fips):
            if self.shutdown_requested:
                self._save_and_exit(tract_data, processed_tracts)
                return {'interrupted': True}
            
            if fips in processed_tracts:
                self.logger.heartbeat(i + 1, total_tracts, f"Skipping {fips} (cached)")
                continue
            
            self.logger.heartbeat(i + 1, total_tracts, f"Processing {fips}")
            
            try:
                addresses = data['loader'].get_addresses_for_tract(fips)
                if len(addresses) < 10:
                    self.logger.log(f" Skipping {fips}: only {len(addresses)} addresses")
                    continue
                
                tract_svi = tracts[tracts['FIPS'] == fips]['RPL_THEMES'].values
                if len(tract_svi) == 0:
                    self.logger.log(f" Skipping {fips}: no SVI data")
                    continue
                    
                tract_svi = tract_svi[0]
                features = compute_tract_features(addresses, data, self.config)
                
                # Store coordinates for graph construction
                coords = np.column_stack([
                    addresses.geometry.x.values,
                    addresses.geometry.y.values
                ])
                
                tract_data[fips] = {
                    'features': features,
                    'coords': coords,
                    'svi': tract_svi,
                    'n_addresses': len(addresses)
                }
                processed_tracts.append(fips)
                
                self.logger.log(f" {fips}: {len(addresses)} addresses, SVI={tract_svi:.3f}")
                
            except Exception as e:
                self.logger.log(f" Error processing {fips}: {e}", level='ERROR')
                continue
            
            # Checkpoint periodically
            if (i + 1) % self.config.checkpoint_every == 0:
                self.checkpoint.save({
                    'tract_data': tract_data,
                    'processed_tracts': processed_tracts,
                    'stage': 'feature_computation'
                })
                self.logger.log(f" Checkpoint saved ({len(processed_tracts)} tracts)")
        
        # Filter to valid tracts
        valid_train = [f for f in train_fips if f in tract_data]
        valid_test = [f for f in test_fips if f in tract_data]
        
        self.logger.log(f"\nValid training tracts: {len(valid_train)}")
        self.logger.log(f"Valid test tracts: {len(valid_test)}")
        
        if len(valid_train) < 3 or len(valid_test) < 2:
            self.logger.log("Insufficient valid tracts", level='ERROR')
            return {'error': 'insufficient_tracts'}
        
        # Run configurations
        results = self._run_configurations(tract_data, valid_train, valid_test)
        
        # Generate report
        self._generate_report(results)
        
        # Cleanup
        self.checkpoint.clear()
        self.logger.close()
        
        return results
    
    def _load_data(self) -> Optional[dict]:
        """Load spatial data."""
        try:
            from granite.data.loaders import DataLoader
            
            loader = DataLoader()
            census_tracts = loader.load_census_tracts('47', '065')
            svi = loader.load_svi_data('47', 'Hamilton')
            tracts = census_tracts.merge(svi, on='FIPS', how='inner')
            
            employment = loader.create_employment_destinations(use_real_data=True)
            healthcare = loader.create_healthcare_destinations(use_real_data=True)
            grocery = loader.create_grocery_destinations(use_real_data=True)
            
            self.logger.log(f" Tracts: {len(tracts)}")
            self.logger.log(f" Employment: {len(employment)}")
            self.logger.log(f" Healthcare: {len(healthcare)}")
            self.logger.log(f" Grocery: {len(grocery)}")
            
            return {
                'loader': loader,
                'tracts': tracts,
                'employment': employment,
                'healthcare': healthcare,
                'grocery': grocery
            }
        except Exception as e:
            self.logger.log(f"Data load error: {e}", level='ERROR')
            return None
    
    def _analyze_feature_importance(
        self, 
        tract_data: dict, 
        train_fips: List[str]
    ) -> Tuple[List[int], List[dict]]:
        """
        Analyze which features correlate most strongly with tract-level SVI.
        Returns indices of top features sorted by absolute correlation.
        """
        self.logger.log("\n--- FEATURE IMPORTANCE ANALYSIS ---")
        
        # Feature names for the 30 Euclidean features
        dest_types = ['employment', 'healthcare', 'grocery']
        feature_names_per_type = [
            'min_distance', 'mean_distance', 'median_distance', 'distance_std',
            'count_1km', 'count_3km', 'count_5km', 'count_10km',
            'concentration', 'range'
        ]
        
        all_feature_names = []
        for dest in dest_types:
            for feat in feature_names_per_type:
                all_feature_names.append(f"{dest}_{feat}")
        
        # Compute tract-level feature means and correlate with SVI
        tract_features = []
        tract_svis = []
        
        for fips in train_fips:
            if fips not in tract_data:
                continue
            td = tract_data[fips]
            # Mean of each feature across all addresses in tract
            tract_mean = np.mean(td['features'], axis=0)
            tract_features.append(tract_mean)
            tract_svis.append(td['svi'])
        
        tract_features = np.array(tract_features)
        tract_svis = np.array(tract_svis)
        
        # Compute correlation of each feature with SVI
        correlations = []
        for i in range(tract_features.shape[1]):
            feat_values = tract_features[:, i]
            if np.std(feat_values) > 0:
                r = np.corrcoef(feat_values, tract_svis)[0, 1]
            else:
                r = 0.0
            correlations.append({
                'index': i,
                'name': all_feature_names[i] if i < len(all_feature_names) else f'feature_{i}',
                'correlation': r,
                'abs_correlation': abs(r)
            })
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        # Log top 10
        self.logger.log("\nTop 10 features by |correlation| with tract SVI:")
        for i, feat in enumerate(correlations[:10]):
            self.logger.log(f" {i+1}. {feat['name']}: r = {feat['correlation']:+.3f}")
        
        # Log bottom 5
        self.logger.log("\nBottom 5 features (weakest signal):")
        for feat in correlations[-5:]:
            self.logger.log(f" {feat['name']}: r = {feat['correlation']:+.3f}")
        
        # Return indices of top features
        top_indices = [f['index'] for f in correlations]
        
        return top_indices, correlations
    
    def _run_configurations(
        self, 
        tract_data: dict, 
        train_fips: List[str], 
        test_fips: List[str]
    ) -> dict:
        """Run accessibility-only and random baseline using inductive GNN evaluation."""
        
        results = {}
        
        # First, analyze feature importance to select top features
        top_features, feature_correlations = self._analyze_feature_importance(
            tract_data, train_fips
        )
        
        configs = {
            'accessibility_only': {
                'use_random': False,
                'use_constant': False,
                'use_coordinates': False,
                'use_laplacian': False,
                'feature_indices': None,
                'description': 'All 30 Euclidean accessibility features'
            },
            'coordinates_only': {
                'use_random': False,
                'use_constant': False,
                'use_coordinates': True,
                'use_laplacian': False,
                'feature_indices': None,
                'description': 'Raw x,y coordinates as features (pure position)'
            },
            'laplacian_pe': {
                'use_random': False,
                'use_constant': False,
                'use_coordinates': False,
                'use_laplacian': True,
                'feature_indices': None,
                'description': 'Laplacian positional encoding (graph structure)'
            },
            'random_baseline': {
                'use_random': True,
                'use_constant': False,
                'use_coordinates': False,
                'use_laplacian': False,
                'feature_indices': None,
                'description': 'Random features (null hypothesis)'
            },
        }
        
        # Store feature analysis in results
        results['feature_analysis'] = feature_correlations
        
        for config_name, cfg in configs.items():
            self.logger.log(f"\n{'='*60}")
            self.logger.log(f"CONFIGURATION: {config_name}")
            if cfg.get('use_constant'):
                self.logger.log(f"Pure graph topology test: all nodes get constant features")
            elif cfg.get('use_coordinates'):
                self.logger.log(f"Position encoding: raw x,y coordinates")
            elif cfg.get('use_laplacian'):
                self.logger.log(f"Position encoding: Laplacian eigenvectors")
            elif cfg.get('feature_indices'):
                self.logger.log(f"Pruned features: using {len(cfg['feature_indices'])} top features")
            else:
                self.logger.log(f"Inductive GNN: train on 12 tracts, test on 10 (same graph)")
            self.logger.log(f"{'='*60}")
            
            if self.shutdown_requested:
                break
            
            start = time.time()
            
            try:
                # Build ONE unified graph with all tracts
                self.logger.log("Building unified graph (train + test tracts)...")
                
                graph, tract_masks, tract_svis, train_mask, test_mask, scaler = create_unified_graph(
                    tract_data, train_fips, test_fips, 
                    use_random=cfg.get('use_random', False),
                    use_constant=cfg.get('use_constant', False),
                    use_coordinates=cfg.get('use_coordinates', False),
                    use_laplacian=cfg.get('use_laplacian', False),
                    feature_indices=cfg.get('feature_indices', None)
                )
                
                n_total = graph.x.shape[0]
                n_train = train_mask.sum().item()
                n_test = test_mask.sum().item()
                n_features = graph.x.shape[1]
                n_edges = graph.edge_index.shape[1] // 2
                
                self.logger.log(f" Graph: {n_total} nodes ({n_train} train, {n_test} test)")
                self.logger.log(f" Edges: {n_edges}, Features: {n_features}")
                self.logger.log(f" Train tracts: {len(train_fips)}, Test tracts: {len(test_fips)}")
                
                # Train GNN (loss only on training nodes)
                self.logger.log("\nTraining GNN (inductive)...")
                model = train_gnn_inductive(
                    graph, tract_masks, tract_svis, train_mask, train_fips,
                    self.config, self.logger
                )
                
                # Evaluate on test tracts (same graph, different nodes)
                self.logger.log("\nEvaluating on test tracts...")
                actual, predicted = evaluate_gnn_inductive(
                    model, graph, tract_masks, tract_svis, test_fips, self.logger
                )
                
                # Compute metrics
                if len(actual) >= 3:
                    correlation = np.corrcoef(actual, predicted)[0, 1]
                    mae = np.mean(np.abs(actual - predicted))
                    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
                else:
                    correlation, mae, rmse = np.nan, np.nan, np.nan
                
                elapsed = time.time() - start
                
                results[config_name] = {
                    'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
                    'mae': float(mae) if not np.isnan(mae) else 1.0,
                    'rmse': float(rmse) if not np.isnan(rmse) else 1.0,
                    'n_total_nodes': n_total,
                    'n_train_nodes': n_train,
                    'n_test_nodes': n_test,
                    'n_train_tracts': len([f for f in train_fips if f in tract_masks]),
                    'n_test_tracts': len(actual),
                    'elapsed_seconds': elapsed,
                    'description': cfg['description'],
                    'model_type': 'AccessibilitySVIGNN',
                    'evaluation': 'inductive'
                }
                
                self.logger.log(f"\nResults for {config_name}:")
                self.logger.log(f" Correlation: r = {correlation:.3f}")
                self.logger.log(f" MAE: {mae:.3f}")
                self.logger.log(f" RMSE: {rmse:.3f}")
                self.logger.log(f" Time: {elapsed:.1f}s")
                
            except Exception as e:
                self.logger.log(f"Configuration {config_name} failed: {e}", level='ERROR')
                import traceback
                self.logger.log(traceback.format_exc(), level='ERROR')
                results[config_name] = {
                    'error': str(e),
                    'description': cfg['description']
                }
        
        return results
    
    def _generate_report(self, results: dict):
        """Generate final report."""
        
        self.logger.log("\n" + "="*70)
        self.logger.log("ABLATION STUDY RESULTS")
        self.logger.log("="*70)
        
        acc_r = results.get('accessibility_only', {}).get('correlation', np.nan)
        rand_r = results.get('random_baseline', {}).get('correlation', np.nan)
        
        self.logger.log(f"\n{'Configuration':<25} {'Correlation':>12} {'MAE':>10} {'RMSE':>10}")
        self.logger.log("-" * 60)
        
        for name, res in results.items():
            # Skip non-configuration entries
            if name in ['feature_analysis', 'interpretation', 'timestamp']:
                continue
            if not isinstance(res, dict) or 'correlation' not in res:
                continue
            r = res.get('correlation', np.nan)
            mae = res.get('mae', np.nan)
            rmse = res.get('rmse', np.nan)
            self.logger.log(f"{name:<25} {r:>12.3f} {mae:>10.3f} {rmse:>10.3f}")
        
        # Interpretation
        self.logger.log("\n" + "-"*70)
        self.logger.log("INTERPRETATION")
        self.logger.log("-"*70)
        
        topo_r = results.get('topology_only', {}).get('correlation', np.nan)
        
        if not np.isnan(acc_r) and not np.isnan(rand_r):
            diff = acc_r - rand_r
            
            if acc_r > 0.5:
                interpretation = "STRONG_SIGNAL"
                self.logger.log(f"\nAccessibility features carry STRONG signal (r={acc_r:.3f})")
                self.logger.log("=> Methodological Contribution narrative")
                
            elif acc_r > 0.3:
                interpretation = "MODERATE_SIGNAL"
                self.logger.log(f"\nAccessibility features carry MODERATE signal (r={acc_r:.3f})")
                self.logger.log("=> Methodological Contribution with caveats")
                
            else:
                interpretation = "WEAK_SIGNAL"
                self.logger.log(f"\nAccessibility features carry WEAK signal (r={acc_r:.3f})")
                self.logger.log("=> Valuable Negative Result narrative")
            
            self.logger.log(f"\nAccessibility r={acc_r:.3f} vs Random r={rand_r:.3f} (diff={diff:+.3f})")
            
            if diff > 0.15:
                self.logger.log("=> Accessibility significantly outperforms random")
            elif diff > 0.05:
                self.logger.log("=> Accessibility slightly better than random")
            else:
                self.logger.log("=> No significant difference from random")
        else:
            interpretation = "UNKNOWN"
        
        # Topology analysis
        topo_r = results.get('topology_only', {}).get('correlation', np.nan)
        if not np.isnan(topo_r):
            self.logger.log(f"\n--- GRAPH TOPOLOGY ANALYSIS ---")
            self.logger.log(f"Topology-only r={topo_r:.3f}")
            
            if topo_r > 0.5:
                self.logger.log("=> Graph structure alone provides strong predictive signal")
            elif topo_r > 0.3:
                self.logger.log("=> Graph structure provides moderate signal")
            else:
                self.logger.log("=> Graph structure alone is insufficient")
        
        # Position encoding analysis
        coord_r = results.get('coordinates_only', {}).get('correlation', np.nan)
        lap_r = results.get('laplacian_pe', {}).get('correlation', np.nan)
        
        if not np.isnan(coord_r) or not np.isnan(lap_r):
            self.logger.log(f"\n--- POSITION ENCODING ANALYSIS ---")
            
            if not np.isnan(coord_r):
                self.logger.log(f"Coordinates (x,y): r={coord_r:.3f}")
            if not np.isnan(lap_r):
                self.logger.log(f"Laplacian PE: r={lap_r:.3f}")
            
            best_pe = max(
                coord_r if not np.isnan(coord_r) else -999,
                lap_r if not np.isnan(lap_r) else -999
            )
            
            if best_pe > rand_r - 0.1:
                self.logger.log("=> Position encodings match or exceed random baseline")
                self.logger.log("=> RECOMMENDATION: Use position encoding instead of accessibility")
            
            if not np.isnan(acc_r) and best_pe > acc_r + 0.2:
                self.logger.log("=> Position encoding significantly outperforms accessibility!")
                self.logger.log("=> CONCLUSION: Spatial position, not accessibility, is the signal")
        
        # Pruned features analysis
        pruned_r = results.get('pruned_top5', {}).get('correlation', np.nan)
        if not np.isnan(pruned_r):
            self.logger.log(f"\n--- PRUNED FEATURES ANALYSIS ---")
            self.logger.log(f"Top 5 features r={pruned_r:.3f}")
            
            if not np.isnan(acc_r):
                improvement = pruned_r - acc_r
                self.logger.log(f"Improvement over all 30: {improvement:+.3f}")
                
                if pruned_r > acc_r + 0.1:
                    self.logger.log("=> Feature pruning HELPS: noise removal improves signal")
                    self.logger.log("=> RECOMMENDATION: Focus on high-correlation features")
                elif pruned_r < acc_r - 0.1:
                    self.logger.log("=> Feature pruning HURTS: additional features provide value")
                else:
                    self.logger.log("=> Feature pruning has minimal effect")
                    
            if not np.isnan(rand_r):
                if pruned_r > rand_r:
                    self.logger.log("=> Pruned features OUTPERFORM random baseline!")
                    self.logger.log("=> CONCLUSION: Selected accessibility features carry real signal")
        
        # Save JSON results
        results['interpretation'] = interpretation
        results['timestamp'] = datetime.now().isoformat()
        
        # Convert numpy types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            return obj
        
        results_clean = convert_for_json(results)
        
        with open(self.config.results_path, 'w') as f:
            json.dump(results_clean, f, indent=2)
        
        self.logger.log(f"\nResults saved: {self.config.results_path}")
        self.logger.log("="*70)
    
    def _save_and_exit(self, tract_data: dict, processed_tracts: list):
        """Save state and exit gracefully."""
        self.checkpoint.save({
            'tract_data': tract_data,
            'processed_tracts': processed_tracts,
            'stage': 'interrupted'
        })
        self.logger.log("Checkpoint saved. Run again to resume.")
        self.logger.close()


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='GRANITE Robust Ablation Study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Features:
  - Checkpoints progress every N tracts (resumes on restart)
  - Heartbeat logging for long runs
  - Graceful Ctrl+C handling
  - Uses curated train/test split from holdout validation

Examples:
  python run_ablation_study_robust.py
  python run_ablation_study_robust.py --epochs 150 --checkpoint-every 3
  python run_ablation_study_robust.py --no-osrm-cache
        """
    )
    
    parser.add_argument('--output-dir', type=str, default='./output/ablation_study')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--checkpoint-every', type=int, default=5)
    parser.add_argument('--heartbeat-every', type=int, default=60)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-osrm-cache', action='store_true',
                       help='Skip OSRM cache lookup, use Euclidean only')
    parser.add_argument('--n-train', type=int, default=12)
    parser.add_argument('--n-test', type=int, default=10)
    
    args = parser.parse_args()
    
    config = AblationConfig(
        output_dir=args.output_dir,
        epochs=args.epochs,
        checkpoint_every=args.checkpoint_every,
        heartbeat_every=args.heartbeat_every,
        seed=args.seed,
        use_cached_osrm=not args.no_osrm_cache,
        n_train_tracts=args.n_train,
        n_test_tracts=args.n_test,
    )
    
    study = RobustAblationStudy(config)
    results = study.run()
    
    if results.get('interrupted'):
        print("\nStudy interrupted. Run again to resume from checkpoint.")
        sys.exit(1)
    elif 'error' in results:
        print(f"\nStudy failed: {results['error']}")
        sys.exit(1)
    else:
        print("\nAblation study completed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()