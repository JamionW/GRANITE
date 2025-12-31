"""
Quick comparison: Coordinate-based GNN vs IDW for within-tract disaggregation.

Question: Does the GNN produce meaningfully different spatial patterns than IDW
when both are constrained to the same tract-level mean?
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/workspaces/GRANITE')


def compute_idw_predictions(addresses_coords, tract_svi, neighbor_centroids, neighbor_svis, power=2):
    """
    Compute IDW predictions for addresses within a tract.
    Uses neighboring tract centroids as anchor points.
    """
    n_addresses = len(addresses_coords)
    predictions = np.zeros(n_addresses)
    
    # Filter out neighbors with NaN SVI
    valid_mask = ~np.isnan(neighbor_svis)
    neighbor_centroids = neighbor_centroids[valid_mask]
    neighbor_svis = neighbor_svis[valid_mask]
    
    if len(neighbor_svis) == 0:
        print("    WARNING: No valid neighbor SVIs for IDW")
        return np.full(n_addresses, np.nan)
    
    for i, coord in enumerate(addresses_coords):
        # Distances to all neighbor centroids
        distances = cdist([coord], neighbor_centroids, metric='euclidean')[0]
        
        # Avoid division by zero
        distances = np.maximum(distances, 1e-10)
        
        # IDW weights
        weights = 1.0 / (distances ** power)
        
        if np.sum(weights) == 0 or np.isinf(np.sum(weights)):
            predictions[i] = np.nan
            continue
            
        weights = weights / weights.sum()
        
        # Weighted prediction
        predictions[i] = np.sum(weights * neighbor_svis)
    
    # Check for NaN
    if np.all(np.isnan(predictions)):
        print("    WARNING: All IDW predictions are NaN")
        return predictions
    
    # Rescale to match tract mean (constraint enforcement)
    current_mean = np.nanmean(predictions)
    if current_mean > 0:
        predictions = predictions * (tract_svi / current_mean)
    
    # Clip to valid range
    predictions = np.clip(predictions, 0, 1)
    
    return predictions


def create_tract_graph(coords, features, k=8):
    """Create a KNN graph for a single tract."""
    n = len(coords)
    k = min(k, n - 1)
    
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
    
    return graph


def train_constrained_gnn(graph, tract_svi, epochs=150, lr=0.01):
    """
    Train GNN with hard constraint enforcement.
    The model must produce predictions that average to tract_svi.
    """
    from granite.models.gnn import AccessibilitySVIGNN, set_random_seed
    
    set_random_seed(42)
    
    n_features = graph.x.shape[1]
    model = AccessibilitySVIGNN(
        accessibility_features_dim=n_features,
        context_features_dim=5,
        hidden_dim=32,
        dropout=0.2,
        seed=42,
        use_context_gating=False,
        use_multitask=False
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    target = torch.FloatTensor([tract_svi])
    
    model.train()
    best_loss = float('inf')
    patience = 0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        predictions = model(graph.x, graph.edge_index)
        pred_mean = predictions.mean()
        
        # Constraint loss: mean must match tract SVI
        constraint_loss = F.mse_loss(pred_mean.unsqueeze(0), target)
        
        # Variation loss: encourage spatial variation
        spatial_std = predictions.std()
        variation_loss = F.relu(0.05 - spatial_std)
        
        # Smoothness loss: nearby nodes should have similar values
        src, dst = graph.edge_index
        smoothness_loss = F.mse_loss(predictions[src], predictions[dst])
        
        # Bounds
        bounds_loss = torch.mean(F.relu(predictions - 1.0)) + torch.mean(F.relu(-predictions))
        
        total_loss = (
            3.0 * constraint_loss +
            1.0 * variation_loss +
            0.5 * smoothness_loss +
            1.0 * bounds_loss
        )
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            patience = 0
        else:
            patience += 1
        
        if patience > 20:
            break
    
    # Get final predictions
    model.eval()
    with torch.no_grad():
        predictions = model(graph.x, graph.edge_index).numpy().flatten()
    
    # Hard constraint: rescale to exact tract mean
    current_mean = predictions.mean()
    if current_mean > 0:
        predictions = predictions * (tract_svi / current_mean)
    
    predictions = np.clip(predictions, 0, 1)
    
    return predictions


def main():
    print("=" * 70)
    print("GNN vs IDW: Within-Tract Disaggregation Comparison")
    print("=" * 70)
    
    # Load data
    from granite.data.loaders import DataLoader
    
    loader = DataLoader()
    census_tracts = loader.load_census_tracts('47', '065')
    svi = loader.load_svi_data('47', 'Hamilton')
    tracts = census_tracts.merge(svi, on='FIPS', how='inner')
    
    # Pick a few test tracts with different SVI levels
    test_tracts = [
        ('47065000600', 'Low SVI (0.224)'),      # Low vulnerability
        ('47065010433', 'Medium SVI (0.454)'),   # Medium
        ('47065001300', 'High SVI (0.873)'),     # High vulnerability
    ]
    
    # Get neighbor tracts for IDW
    all_centroids = []
    all_svis = []
    tract_lookup = {}
    
    for idx, row in tracts.iterrows():
        fips = str(row['FIPS'])
        centroid = row.geometry.centroid
        all_centroids.append([centroid.x, centroid.y])
        all_svis.append(row['RPL_THEMES'])
        tract_lookup[fips] = len(all_centroids) - 1
    
    all_centroids = np.array(all_centroids)
    all_svis = np.array(all_svis)
    
    # Results storage
    results = []
    
    fig, axes = plt.subplots(len(test_tracts), 3, figsize=(15, 5 * len(test_tracts)))
    
    for t_idx, (fips, label) in enumerate(test_tracts):
        print(f"\n--- Tract {fips} ({label}) ---")
        
        # Load addresses for this tract
        addresses = loader.get_addresses_for_tract(fips)
        if len(addresses) == 0:
            print(f"  No addresses found, skipping")
            continue
        
        # Get tract SVI
        tract_idx = tract_lookup[fips]
        tract_svi = all_svis[tract_idx]
        tract_centroid = all_centroids[tract_idx]
        
        # Get coordinates
        coords = np.column_stack([
            addresses.geometry.x.values,
            addresses.geometry.y.values
        ])
        
        print(f"  Addresses: {len(coords)}")
        print(f"  Tract SVI: {tract_svi:.3f}")
        
        # Get neighbor centroids (exclude self)
        neighbor_mask = np.arange(len(all_centroids)) != tract_idx
        neighbor_centroids = all_centroids[neighbor_mask]
        neighbor_svis = all_svis[neighbor_mask]
        
        # Debug: check for coordinate/SVI issues
        print(f"  Debug: {len(neighbor_centroids)} neighbors, {np.sum(~np.isnan(neighbor_svis))} with valid SVI")
        print(f"  Debug: Address coord range: x=[{coords[:, 0].min():.1f}, {coords[:, 0].max():.1f}], y=[{coords[:, 1].min():.1f}, {coords[:, 1].max():.1f}]")
        print(f"  Debug: Centroid coord range: x=[{neighbor_centroids[:, 0].min():.1f}, {neighbor_centroids[:, 0].max():.1f}], y=[{neighbor_centroids[:, 1].min():.1f}, {neighbor_centroids[:, 1].max():.1f}]")
        
        # --- IDW Predictions ---
        print("  Running IDW...")
        idw_preds = compute_idw_predictions(
            coords, tract_svi, neighbor_centroids, neighbor_svis
        )
        
        # --- GNN Predictions (coordinate features) ---
        print("  Running GNN (coordinates only)...")
        graph = create_tract_graph(coords, coords)  # Use coords as features
        gnn_preds = train_constrained_gnn(graph, tract_svi)
        
        # --- Statistics ---
        print(f"\n  Results:")
        print(f"  {'Method':<15} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Range':>8}")
        print(f"  {'-'*55}")
        print(f"  {'IDW':<15} {idw_preds.mean():>8.3f} {idw_preds.std():>8.3f} {idw_preds.min():>8.3f} {idw_preds.max():>8.3f} {idw_preds.max()-idw_preds.min():>8.3f}")
        print(f"  {'GNN (coords)':<15} {gnn_preds.mean():>8.3f} {gnn_preds.std():>8.3f} {gnn_preds.min():>8.3f} {gnn_preds.max():>8.3f} {gnn_preds.max()-gnn_preds.min():>8.3f}")
        
        # Correlation between methods
        corr = np.corrcoef(idw_preds, gnn_preds)[0, 1]
        print(f"\n  IDW-GNN correlation: r = {corr:.3f}")
        
        results.append({
            'fips': fips,
            'label': label,
            'tract_svi': tract_svi,
            'n_addresses': len(coords),
            'idw_std': idw_preds.std(),
            'gnn_std': gnn_preds.std(),
            'idw_range': idw_preds.max() - idw_preds.min(),
            'gnn_range': gnn_preds.max() - gnn_preds.min(),
            'correlation': corr
        })
        
        # --- Visualization ---
        # IDW spatial pattern
        ax1 = axes[t_idx, 0]
        scatter1 = ax1.scatter(coords[:, 0], coords[:, 1], c=idw_preds, 
                               cmap='RdYlGn_r', s=3, vmin=0, vmax=1)
        ax1.set_title(f'IDW - {label}')
        ax1.set_aspect('equal')
        plt.colorbar(scatter1, ax=ax1, label='SVI')
        
        # GNN spatial pattern
        ax2 = axes[t_idx, 1]
        scatter2 = ax2.scatter(coords[:, 0], coords[:, 1], c=gnn_preds,
                               cmap='RdYlGn_r', s=3, vmin=0, vmax=1)
        ax2.set_title(f'GNN (coords) - {label}')
        ax2.set_aspect('equal')
        plt.colorbar(scatter2, ax=ax2, label='SVI')
        
        # Difference map
        ax3 = axes[t_idx, 2]
        diff = gnn_preds - idw_preds
        scatter3 = ax3.scatter(coords[:, 0], coords[:, 1], c=diff,
                               cmap='coolwarm', s=3, vmin=-0.2, vmax=0.2)
        ax3.set_title(f'Difference (GNN - IDW)')
        ax3.set_aspect('equal')
        plt.colorbar(scatter3, ax=ax3, label='Δ SVI')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('./output/gnn_vs_idw')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n\nVisualization saved: {output_dir / 'comparison.png'}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Tract':<20} {'IDW Std':>10} {'GNN Std':>10} {'Correlation':>12}")
    print("-" * 55)
    for r in results:
        print(f"{r['label']:<20} {r['idw_std']:>10.3f} {r['gnn_std']:>10.3f} {r['correlation']:>12.3f}")
    
    avg_corr = np.mean([r['correlation'] for r in results])
    avg_idw_std = np.mean([r['idw_std'] for r in results])
    avg_gnn_std = np.mean([r['gnn_std'] for r in results])
    
    print("-" * 55)
    print(f"{'Average':<20} {avg_idw_std:>10.3f} {avg_gnn_std:>10.3f} {avg_corr:>12.3f}")
    
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)
    
    if avg_corr > 0.9:
        print("\nGNN and IDW produce nearly IDENTICAL patterns.")
        print("=> GNN is learning the same spatial interpolation as IDW.")
        print("=> No methodological advantage; prefer IDW for simplicity.")
        print("=> RECOMMENDATION: Option 1 (Understanding Contribution)")
    elif avg_corr > 0.7:
        print("\nGNN and IDW produce SIMILAR patterns with minor differences.")
        print("=> GNN captures most of IDW's structure but adds some variation.")
        print("=> Marginal improvement may not justify added complexity.")
        print("=> RECOMMENDATION: Likely Option 1, but Option 2 possible with tuning")
    elif avg_corr > 0.4:
        print("\nGNN and IDW produce MODERATELY DIFFERENT patterns.")
        print("=> GNN is learning different spatial structure than IDW.")
        print("=> Worth investigating whether GNN patterns are more realistic.")
        print("=> RECOMMENDATION: Consider Option 2 with validation")
    else:
        print("\nGNN and IDW produce SUBSTANTIALLY DIFFERENT patterns.")
        print("=> GNN is capturing fundamentally different spatial relationships.")
        print("=> Could be better OR worse than IDW - needs ground truth validation.")
        print("=> RECOMMENDATION: Option 2 with careful validation")
    
    if avg_gnn_std > avg_idw_std * 1.5:
        print(f"\nGNN produces {avg_gnn_std/avg_idw_std:.1f}x MORE within-tract variation than IDW.")
        print("=> GNN may capture sharper spatial gradients.")
    elif avg_gnn_std < avg_idw_std * 0.67:
        print(f"\nGNN produces {avg_idw_std/avg_gnn_std:.1f}x LESS within-tract variation than IDW.")
        print("=> GNN may be over-smoothing.")


if __name__ == "__main__":
    main()