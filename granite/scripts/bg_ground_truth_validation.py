"""
Block Group Ground Truth Validation for GRANITE

Tests whether GRANITE can learn accessibility-vulnerability relationships
at a scale where actual ground truth exists (block groups have ACS-derived SVI).

Key insight: Instead of validating by circular aggregation, we train and test
at the block group level where we can directly compare predictions to reality.

Usage:
    python bg_ground_truth_validation.py
"""
import sys
import os
sys.path.insert(0, '/workspaces/GRANITE')

import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(verbose: bool = True):
    """Load all required data: addresses, features, block groups with SVI."""
    from granite.data.loaders import DataLoader
    
    loader = DataLoader()
    
    if verbose:
        print("Loading spatial data...")
    
    # Load tract-level data (for context)
    tracts = loader.load_census_tracts('47', '065')
    svi = loader.load_svi_data('47', 'Hamilton')
    tract_data = tracts.merge(svi, on='FIPS', how='inner')
    
    # Load block groups with computed SVI
    if verbose:
        print("Loading block group data with ACS-derived SVI...")
    
    bg_gdf = loader.block_group_loader.get_block_groups_with_demographics('47', '065')
    
    # Filter to BGs with valid SVI
    valid_bg = bg_gdf[bg_gdf['SVI'].notna()].copy()
    
    if verbose:
        print(f"  Total block groups: {len(bg_gdf)}")
        print(f"  With valid SVI: {len(valid_bg)}")
        print(f"  SVI range: {valid_bg['SVI'].min():.3f} - {valid_bg['SVI'].max():.3f}")
    
    return {
        'loader': loader,
        'tract_data': tract_data,
        'block_groups': valid_bg
    }


def load_addresses_for_tracts(loader, tract_fips_list: List[str], 
                               verbose: bool = True) -> gpd.GeoDataFrame:
    """Load addresses for given tracts."""
    all_addresses = []
    
    if verbose:
        print(f"\nLoading addresses for {len(tract_fips_list)} tracts...")
    
    for fips in tract_fips_list:
        try:
            addresses = loader.get_addresses_for_tract(fips)
            if len(addresses) > 0:
                addresses['tract_fips'] = fips
                all_addresses.append(addresses)
        except Exception as e:
            if verbose:
                print(f"  {fips}: Error - {e}")
    
    if not all_addresses:
        return None
    
    combined = gpd.GeoDataFrame(
        pd.concat(all_addresses, ignore_index=True),
        geometry='geometry',
        crs='EPSG:4326'
    )
    
    if verbose:
        print(f"  Total: {len(combined)} addresses")
    
    return combined


def assign_addresses_to_block_groups(addresses: gpd.GeoDataFrame,
                                      block_groups: gpd.GeoDataFrame,
                                      verbose: bool = True) -> gpd.GeoDataFrame:
    """Spatial join addresses to their containing block groups."""
    if addresses.crs != block_groups.crs:
        addresses = addresses.to_crs(block_groups.crs)
    
    joined = gpd.sjoin(
        addresses,
        block_groups[['GEOID', 'SVI', 'geometry']],
        how='left',
        predicate='within'
    )
    
    joined = joined.rename(columns={'GEOID': 'bg_id', 'SVI': 'bg_svi'})
    
    if 'index_right' in joined.columns:
        joined = joined.drop(columns=['index_right'])
    
    n_unmatched = joined['bg_id'].isna().sum()
    if verbose and n_unmatched > 0:
        print(f"  Warning: {n_unmatched} addresses not in any block group")
    
    # Drop unmatched
    joined = joined[joined['bg_id'].notna()].copy()
    
    return joined


# =============================================================================
# FEATURE COMPUTATION
# =============================================================================

def compute_spatial_features(addresses: gpd.GeoDataFrame, 
                              block_groups: gpd.GeoDataFrame,
                              data_loader=None,
                              verbose: bool = True) -> Tuple[np.ndarray, List[str]]:
    """
    Compute spatial features for addresses using SpatialFeatureComputer.
    """
    from granite.features.spatial_features import SpatialFeatureComputer
    
    if verbose:
        print("Computing spatial features...")
    
    computer = SpatialFeatureComputer(verbose=verbose)
    
    # Get combined geometry of all block groups as the "tract" boundary
    combined_geometry = block_groups.unary_union
    
    # Compute features
    features, feature_names = computer.compute_features(
        addresses=addresses,
        tract_geometry=combined_geometry,
        include_density=True,
        include_boundary=True,
        data_loader=data_loader
    )
    
    if verbose:
        print(f"  Feature matrix: {features.shape}")
    
    return features, feature_names


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_spatial_graph(addresses: gpd.GeoDataFrame, 
                        features: np.ndarray,
                        k_neighbors: int = 8) -> Data:
    """Build PyTorch Geometric graph from addresses."""
    coords = np.array([[g.x, g.y] for g in addresses.geometry])
    
    # K-nearest neighbors for edges
    n = len(coords)
    k = min(k_neighbors, n - 1)
    
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree')
    nbrs.fit(coords)
    _, indices = nbrs.kneighbors(coords)
    
    edge_list = []
    for i in range(n):
        for j in indices[i][1:]:
            edge_list.extend([[i, j], [j, i]])
    
    edge_index = torch.LongTensor(edge_list).t().contiguous()
    
    # Normalize features
    scaler = RobustScaler()
    features_norm = scaler.fit_transform(features)
    
    graph = Data(
        x=torch.FloatTensor(features_norm),
        edge_index=edge_index
    )
    
    return graph, scaler


# =============================================================================
# GRANITE GNN
# =============================================================================

def train_granite_bg_constraints(graph: Data,
                                  bg_masks: Dict[str, np.ndarray],
                                  bg_svis: Dict[str, float],
                                  train_bg_ids: List[str],
                                  epochs: int = 150,
                                  hidden_dim: int = 32,
                                  lr: float = 0.01,
                                  verbose: bool = False) -> torch.nn.Module:
    """
    Train GRANITE GNN with block-group-level constraints.
    
    This is the key methodological change: instead of tract constraints,
    we use BG-level constraints where we have ground truth.
    """
    from granite.models.gnn import SpatialDisaggregationGNN, set_random_seed
    
    set_random_seed(42)
    
    n_features = graph.x.shape[1]
    
    model = SpatialDisaggregationGNN(
        input_dim=n_features,
        hidden_dim=hidden_dim,
        dropout=0.2
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.8, min_lr=1e-5
    )
    
    # Convert masks to tensors
    train_bg_targets = {bg_id: torch.FloatTensor([bg_svis[bg_id]]) 
                        for bg_id in train_bg_ids if bg_id in bg_svis}
    
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(graph.x, graph.edge_index)
        
        # Constraint loss: each training BG's mean should match its SVI
        constraint_losses = []
        for bg_id, target_svi in train_bg_targets.items():
            if bg_id not in bg_masks:
                continue
            mask = bg_masks[bg_id]
            bg_preds = predictions[mask]
            if len(bg_preds) > 0:
                bg_mean = bg_preds.mean()
                loss = F.mse_loss(bg_mean.unsqueeze(0), target_svi)
                constraint_losses.append(loss)
        
        if constraint_losses:
            constraint_loss = torch.mean(torch.stack(constraint_losses))
        else:
            constraint_loss = torch.tensor(0.0)
        
        # Variation loss: encourage spatial variation
        spatial_std = predictions.std()
        variation_loss = F.relu(0.02 - spatial_std)
        
        # Bounds loss
        bounds_loss = (torch.mean(F.relu(predictions - 1.0)) + 
                       torch.mean(F.relu(-predictions)))
        
        # Combined loss
        total_loss = (
            2.0 * constraint_loss +
            1.0 * variation_loss +
            1.0 * bounds_loss
        )
        
        if torch.isnan(total_loss):
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
        
        if patience_counter >= 20:
            if verbose:
                print(f"    Early stopping at epoch {epoch}")
            break
        
        if verbose and epoch % 50 == 0:
            print(f"    Epoch {epoch}: loss={total_loss.item():.4f}, "
                  f"std={spatial_std.item():.4f}")
    
    return model


def predict_bg_means(model: torch.nn.Module,
                     graph: Data,
                     bg_masks: Dict[str, np.ndarray],
                     test_bg_ids: List[str]) -> Dict[str, float]:
    """Get BG-level predictions from trained model."""
    model.eval()
    
    with torch.no_grad():
        predictions = model(graph.x, graph.edge_index).numpy().flatten()
    
    bg_predictions = {}
    for bg_id in test_bg_ids:
        if bg_id not in bg_masks:
            continue
        mask = bg_masks[bg_id]
        bg_preds = predictions[mask]
        if len(bg_preds) > 0:
            bg_predictions[bg_id] = float(np.mean(bg_preds))
    
    return bg_predictions


# =============================================================================
# IDW BASELINE
# =============================================================================

def compute_idw_predictions(block_groups: gpd.GeoDataFrame,
                            train_bg_ids: List[str],
                            test_bg_ids: List[str],
                            power: float = 2.0) -> Dict[str, float]:
    """
    Compute IDW predictions for test block groups.
    Uses training BG centroids as known points.
    """
    train_bg = block_groups[block_groups['GEOID'].isin(train_bg_ids)].copy()
    test_bg = block_groups[block_groups['GEOID'].isin(test_bg_ids)].copy()
    
    # Extract centroid coordinates
    train_coords = np.array([[g.centroid.x, g.centroid.y] for g in train_bg.geometry])
    test_coords = np.array([[g.centroid.x, g.centroid.y] for g in test_bg.geometry])
    train_svi = train_bg['SVI'].values
    
    # Compute distances and IDW weights
    distances = cdist(test_coords, train_coords)
    weights = 1.0 / (distances ** power + 1e-10)
    weights = weights / weights.sum(axis=1, keepdims=True)
    idw_predictions = weights @ train_svi
    
    return dict(zip(test_bg['GEOID'].values, idw_predictions))


# =============================================================================
# MAIN VALIDATION
# =============================================================================

def run_bg_validation(data: Dict, 
                      n_folds: int = 5,
                      max_addresses: int = 10000,
                      verbose: bool = True) -> pd.DataFrame:
    """
    Run k-fold cross-validation at the block group level.
    
    GRANITE vs IDW with actual ground truth.
    """
    block_groups = data['block_groups']
    loader = data['loader']
    
    # Get tracts containing our block groups
    tract_fips_list = block_groups['tract_fips'].unique().tolist()
    
    if verbose:
        print(f"\n{'='*70}")
        print("BLOCK GROUP GROUND TRUTH VALIDATION")
        print(f"{'='*70}")
        print(f"Block groups with valid SVI: {len(block_groups)}")
        print(f"Tracts containing these BGs: {len(tract_fips_list)}")
    
    # Load addresses
    addresses = load_addresses_for_tracts(loader, tract_fips_list, verbose)
    if addresses is None:
        print("ERROR: No addresses loaded")
        return None
    
    # Subsample if needed (for speed during testing)
    if len(addresses) > max_addresses:
        if verbose:
            print(f"  Subsampling to {max_addresses} addresses for speed...")
        addresses = addresses.sample(n=max_addresses, random_state=42)
    
    # Assign addresses to block groups
    if verbose:
        print("\nAssigning addresses to block groups...")
    addresses = assign_addresses_to_block_groups(addresses, block_groups, verbose)
    
    # Compute spatial features
    features, feature_cols = compute_spatial_features(
        addresses, block_groups, data_loader=data['loader'], verbose=verbose
    )
    
    # Build graph
    if verbose:
        print("\nBuilding spatial graph...")
    graph, scaler = build_spatial_graph(addresses, features)
    if verbose:
        print(f"  Nodes: {graph.x.shape[0]}, Edges: {graph.edge_index.shape[1]}")
    
    # Create BG masks (which addresses belong to which BG)
    bg_masks = {}
    bg_svis = {}
    for bg_id in addresses['bg_id'].unique():
        mask = (addresses['bg_id'] == bg_id).values
        bg_masks[bg_id] = mask
        bg_svi = addresses[addresses['bg_id'] == bg_id]['bg_svi'].iloc[0]
        bg_svis[bg_id] = bg_svi
    
    if verbose:
        print(f"  Block groups with addresses: {len(bg_masks)}")
    
    # K-fold cross-validation on block groups
    bg_ids = np.array(list(bg_masks.keys()))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_results = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(bg_ids)):
        train_bg_ids = bg_ids[train_idx].tolist()
        test_bg_ids = bg_ids[test_idx].tolist()
        
        if verbose:
            print(f"\n--- Fold {fold+1}/{n_folds} ---")
            print(f"  Train BGs: {len(train_bg_ids)}, Test BGs: {len(test_bg_ids)}")
        
        # Train GRANITE with BG-level constraints
        if verbose:
            print("  Training GRANITE...")
        model = train_granite_bg_constraints(
            graph=graph,
            bg_masks=bg_masks,
            bg_svis=bg_svis,
            train_bg_ids=train_bg_ids,
            epochs=150,
            verbose=verbose
        )
        
        # Get GRANITE predictions for test BGs
        granite_preds = predict_bg_means(model, graph, bg_masks, test_bg_ids)
        
        # Get IDW predictions
        idw_preds = compute_idw_predictions(
            block_groups, train_bg_ids, test_bg_ids
        )
        
        # Record results
        for bg_id in test_bg_ids:
            actual_svi = bg_svis.get(bg_id, np.nan)
            
            for method, pred_dict in [('GRANITE', granite_preds), ('IDW', idw_preds)]:
                predicted_svi = pred_dict.get(bg_id, np.nan)
                
                all_results.append({
                    'fold': fold,
                    'method': method,
                    'bg_id': bg_id,
                    'actual_svi': actual_svi,
                    'predicted_svi': predicted_svi,
                    'error': predicted_svi - actual_svi if not np.isnan(predicted_svi) else np.nan,
                    'abs_error': abs(predicted_svi - actual_svi) if not np.isnan(predicted_svi) else np.nan
                })
    
    return pd.DataFrame(all_results)


def summarize_results(results_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Compute summary statistics for each method."""
    summary = []
    
    for method in results_df['method'].unique():
        method_data = results_df[results_df['method'] == method]
        
        # Filter valid predictions
        valid = method_data['predicted_svi'].notna() & method_data['actual_svi'].notna()
        valid_data = method_data[valid]
        
        if len(valid_data) < 3:
            continue
        
        r, p = stats.pearsonr(valid_data['actual_svi'], valid_data['predicted_svi'])
        mae = valid_data['abs_error'].mean()
        rmse = np.sqrt((valid_data['error'] ** 2).mean())
        
        summary.append({
            'method': method,
            'n_predictions': len(valid_data),
            'pearson_r': r,
            'p_value': p,
            'mae': mae,
            'rmse': rmse
        })
    
    summary_df = pd.DataFrame(summary)
    
    if verbose:
        print(f"\n{'='*70}")
        print("RESULTS: Block Group Ground Truth Validation")
        print(f"{'='*70}")
        print(f"\n{'Method':<12} {'N':>6} {'r':>8} {'p-value':>10} {'MAE':>8} {'RMSE':>8}")
        print("-" * 56)
        for _, row in summary_df.iterrows():
            print(f"{row['method']:<12} {row['n_predictions']:>6} {row['pearson_r']:>8.3f} "
                  f"{row['p_value']:>10.4f} {row['mae']:>8.3f} {row['rmse']:>8.3f}")
        
        # Statistical test: is GRANITE significantly different from IDW?
        if len(summary_df) >= 2:
            granite_r = summary_df[summary_df['method'] == 'GRANITE']['pearson_r'].values[0]
            idw_r = summary_df[summary_df['method'] == 'IDW']['pearson_r'].values[0]
            
            print(f"\n{'='*70}")
            print("INTERPRETATION")
            print(f"{'='*70}")
            print(f"GRANITE r = {granite_r:.3f}, IDW r = {idw_r:.3f}")
            print(f"Difference: {granite_r - idw_r:+.3f}")
            
            if granite_r > idw_r + 0.05:
                print("\n→ GRANITE shows meaningful improvement over IDW.")
                print("  This suggests accessibility features add predictive value.")
            elif abs(granite_r - idw_r) < 0.05:
                print("\n→ GRANITE performs similarly to IDW.")
                print("  Accessibility features may not add value at BG scale.")
            else:
                print("\n→ IDW outperforms GRANITE.")
                print("  Simple spatial smoothing beats learned features.")
    
    return summary_df


def main():
    """Run the block group ground truth validation."""
    print("="*70)
    print("GRANITE: Block Group Ground Truth Validation")
    print("="*70)
    print("\nThis validates disaggregation where ground truth actually exists.")
    print("Block groups have ACS-derived SVI we can compare against directly.")
    print("\nGRANITE vs IDW - fair comparison with real ground truth.\n")
    
    # Load data
    data = load_data(verbose=True)
    
    # Run validation (limit addresses for reasonable runtime)
    results = run_bg_validation(
        data, 
        n_folds=5, 
        max_addresses=15000,  # Increase for more thorough test
        verbose=True
    )
    
    if results is not None:
        summary = summarize_results(results, verbose=True)
        
        # Save
        results.to_csv('bg_validation_results.csv', index=False)
        summary.to_csv('bg_validation_summary.csv', index=False)
        print("\nResults saved to bg_validation_*.csv")
        
        return results, summary
    
    return None, None


if __name__ == '__main__':
    results, summary = main()