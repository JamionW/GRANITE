"""
GRANITE Spatial Autocorrelation Analysis (Moran's I)

Tests whether prediction residuals exhibit spatial clustering.
Validates that spatial patterns are real geographic structure, not random noise.

Key question: Do areas with similar residuals cluster together?
- Positive Moran's I: Nearby areas have similar residuals (spatial structure)
- Near-zero Moran's I: Random spatial distribution (no structure)
- Negative Moran's I: Nearby areas have opposite residuals (dispersion)

For GRANITE validation:
- We want LOW spatial autocorrelation in residuals (model captures spatial pattern)
- HIGH autocorrelation in predictions (learned spatial structure)
"""
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import sparse, stats
from scipy.spatial import distance
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/workspaces/GRANITE')


def compute_spatial_weights(coordinates, method='knn', k=8, threshold=None):
    """
    Compute spatial weights matrix for Moran's I.
    
    Args:
        coordinates: Array of (x, y) coordinates
        method: 'knn' (k-nearest neighbors) or 'threshold' (distance threshold)
        k: Number of neighbors for KNN
        threshold: Distance threshold (only for method='threshold')
        
    Returns:
        Sparse row-standardized weights matrix
    """
    n = len(coordinates)
    
    # Compute pairwise distances
    dist_matrix = distance.cdist(coordinates, coordinates)
    
    if method == 'knn':
        # K-nearest neighbors
        W = np.zeros((n, n))
        for i in range(n):
            # Get k nearest neighbors (excluding self)
            distances = dist_matrix[i]
            neighbors = np.argsort(distances)[1:k+1]  # Exclude self (index 0)
            W[i, neighbors] = 1
    
    elif method == 'threshold':
        # Distance threshold
        if threshold is None:
            # Use median distance to k-th nearest as default
            kth_distances = np.sort(dist_matrix, axis=1)[:, k]
            threshold = np.median(kth_distances)
        
        W = (dist_matrix <= threshold) & (dist_matrix > 0)
        W = W.astype(float)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # symmetrize before row-normalizing (k-NN graphs are asymmetric)
    W = (W + W.T) / 2

    # Row-standardize
    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    W = W / row_sums[:, np.newaxis]
    
    return sparse.csr_matrix(W)


def morans_i(values, W, permutations=999, seed=42):
    """
    Compute Moran's I statistic with permutation-based inference.
    
    Args:
        values: Array of values to test for spatial autocorrelation
        W: Spatial weights matrix (row-standardized)
        permutations: Number of permutations for p-value
        seed: Random seed
        
    Returns:
        dict with I statistic, expected I, z-score, and p-value
    """
    np.random.seed(seed)
    
    n = len(values)
    values = np.asarray(values).flatten()
    
    # Center the values
    z = values - values.mean()
    
    # Ensure W is dense for computation
    if sparse.issparse(W):
        W = W.toarray()
    
    # Moran's I numerator: sum of (z_i * sum_j(w_ij * z_j))
    # Denominator: sum of z^2
    numerator = np.sum(z * (W @ z))
    denominator = np.sum(z ** 2)
    
    # Total weight
    S0 = W.sum()
    
    # Moran's I
    I = (n / S0) * (numerator / denominator)
    
    # Expected I under null hypothesis
    E_I = -1 / (n - 1)
    
    # Permutation inference
    perm_Is = []
    for _ in range(permutations):
        z_perm = np.random.permutation(z)
        num_perm = np.sum(z_perm * (W @ z_perm))
        I_perm = (n / S0) * (num_perm / denominator)
        perm_Is.append(I_perm)
    
    perm_Is = np.array(perm_Is)
    
    # Pseudo p-value (two-tailed)
    p_value = (np.sum(np.abs(perm_Is) >= np.abs(I)) + 1) / (permutations + 1)
    
    # Z-score based on permutation distribution
    z_score = (I - np.mean(perm_Is)) / np.std(perm_Is)
    
    return {
        'I': float(I),
        'E_I': float(E_I),
        'z_score': float(z_score),
        'p_value': float(p_value),
        'permutation_mean': float(np.mean(perm_Is)),
        'permutation_std': float(np.std(perm_Is)),
        'permutation_dist': perm_Is
    }


def local_morans_i(values, W, permutations=999, seed=42):
    """
    Compute Local Moran's I (LISA) to identify spatial clusters.
    
    Returns local I for each observation, identifying:
    - High-High clusters: High values surrounded by high values
    - Low-Low clusters: Low values surrounded by low values
    - High-Low outliers: High values surrounded by low values
    - Low-High outliers: Low values surrounded by high values
    """
    np.random.seed(seed)
    
    n = len(values)
    values = np.asarray(values).flatten()
    
    # Standardize
    z = (values - values.mean()) / values.std()
    
    if sparse.issparse(W):
        W = W.toarray()
    
    # Local Moran's I for each observation
    lag_z = W @ z  # Spatial lag of z
    I_local = z * lag_z
    
    # Permutation inference for each location
    p_values = np.zeros(n)
    for i in range(n):
        perm_Is = []
        for _ in range(permutations):
            z_perm = np.random.permutation(z)
            lag_perm = W[i] @ z_perm
            I_perm = z[i] * lag_perm
            perm_Is.append(I_perm)
        
        perm_Is = np.array(perm_Is)
        p_values[i] = (np.sum(np.abs(perm_Is) >= np.abs(I_local[i])) + 1) / (permutations + 1)
    
    # Classify clusters (significant only)
    alpha = 0.05
    clusters = np.array(['Not Sig'] * n, dtype=object)
    
    significant = p_values < alpha
    high_value = z > 0
    high_lag = lag_z > 0
    
    clusters[significant & high_value & high_lag] = 'High-High'
    clusters[significant & ~high_value & ~high_lag] = 'Low-Low'
    clusters[significant & high_value & ~high_lag] = 'High-Low'
    clusters[significant & ~high_value & high_lag] = 'Low-High'
    
    return {
        'I_local': I_local,
        'p_values': p_values,
        'clusters': clusters,
        'lag_z': lag_z,
        'z': z
    }


def analyze_spatial_autocorrelation(addresses, predictions, ground_truth=None,
                                    method='knn', k=8, permutations=999, seed=42):
    """
    spatial autocorrelation analysis for GRANITE validation.
    
    Args:
        addresses: GeoDataFrame with address points
        predictions: Array of predicted values
        ground_truth: Optional array of actual values (for residual analysis)
        
    Returns:
        dict with Moran's I results for predictions and residuals
    """
    
    # Get coordinates
    if hasattr(addresses, 'geometry'):
        coords = np.column_stack([addresses.geometry.x, addresses.geometry.y])
    else:
        coords = np.column_stack([addresses['x'], addresses['y']])
    
    # Compute weights
    W = compute_spatial_weights(coords, method=method, k=k)
    
    results = {}
    
    # Moran's I for predictions
    print("\nAnalyzing spatial autocorrelation in PREDICTIONS...")
    moran_pred = morans_i(predictions, W, permutations=permutations, seed=seed)
    results['predictions'] = moran_pred
    
    print(f"  Moran's I = {moran_pred['I']:.4f}")
    print(f"  Expected I = {moran_pred['E_I']:.4f}")
    print(f"  Z-score = {moran_pred['z_score']:.2f}")
    print(f"  p-value = {moran_pred['p_value']:.4f}")
    
    if moran_pred['p_value'] < 0.05:
        if moran_pred['I'] > 0:
            print("  -> SIGNIFICANT positive spatial autocorrelation")
            print("     Nearby addresses have similar predicted values (good!)")
        else:
            print("  -> SIGNIFICANT negative spatial autocorrelation (dispersed)")
    else:
        print("  -> No significant spatial autocorrelation")
    
    # Moran's I for residuals (if ground truth provided)
    if ground_truth is not None:
        print("\nAnalyzing spatial autocorrelation in RESIDUALS...")
        residuals = predictions - ground_truth
        moran_resid = morans_i(residuals, W, permutations=permutations, seed=seed)
        results['residuals'] = moran_resid
        
        print(f"  Moran's I = {moran_resid['I']:.4f}")
        print(f"  Expected I = {moran_resid['E_I']:.4f}")
        print(f"  Z-score = {moran_resid['z_score']:.2f}")
        print(f"  p-value = {moran_resid['p_value']:.4f}")
        
        if moran_resid['p_value'] < 0.05:
            if moran_resid['I'] > 0:
                print("  -> SIGNIFICANT positive autocorrelation in residuals")
                print("     Model is missing some spatial structure")
            else:
                print("  -> SIGNIFICANT negative autocorrelation in residuals")
        else:
            print("  -> No significant spatial autocorrelation in residuals")
            print("     Model captures spatial structure well!")
    
    return results


def create_morans_i_plot(results, output_path='morans_i_analysis.png'):
    """Create visualization of Moran's I results."""
    import matplotlib.pyplot as plt
    
    n_panels = 2 if 'residuals' in results else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6*n_panels, 5))
    
    if n_panels == 1:
        axes = [axes]
    
    # Panel 1: Predictions
    ax = axes[0]
    perm_dist = results['predictions']['permutation_dist']
    I_obs = results['predictions']['I']
    
    ax.hist(perm_dist, bins=50, color='steelblue', alpha=0.7, edgecolor='white',
            label='Permutation Distribution')
    ax.axvline(I_obs, color='red', linewidth=2, label=f"Observed I = {I_obs:.4f}")
    ax.axvline(results['predictions']['E_I'], color='black', linestyle='--',
               label=f"Expected I = {results['predictions']['E_I']:.4f}")
    ax.set_xlabel("Moran's I")
    ax.set_ylabel('Frequency')
    ax.set_title(f"Predictions\np = {results['predictions']['p_value']:.4f}")
    ax.legend(fontsize=9)
    
    # Panel 2: Residuals (if available)
    if 'residuals' in results:
        ax = axes[1]
        perm_dist = results['residuals']['permutation_dist']
        I_obs = results['residuals']['I']
        
        ax.hist(perm_dist, bins=50, color='coral', alpha=0.7, edgecolor='white',
                label='Permutation Distribution')
        ax.axvline(I_obs, color='red', linewidth=2, label=f"Observed I = {I_obs:.4f}")
        ax.axvline(results['residuals']['E_I'], color='black', linestyle='--',
                   label=f"Expected I = {results['residuals']['E_I']:.4f}")
        ax.set_xlabel("Moran's I")
        ax.set_ylabel('Frequency')
        ax.set_title(f"Residuals\np = {results['residuals']['p_value']:.4f}")
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to: {output_path}")


def demo_with_synthetic_data():
    """Demonstrate Moran's I analysis with synthetic spatially-autocorrelated data."""
    
    print("\n" + "="*70)
    print("DEMO: Moran's I Analysis with Synthetic Data")
    print("="*70)
    
    np.random.seed(42)
    n = 500
    
    # Create spatially clustered data
    # Generate coordinates on a grid
    side = int(np.sqrt(n))
    x = np.tile(np.arange(side), side)
    y = np.repeat(np.arange(side), side)
    coords = np.column_stack([x, y])[:n]
    
    # Create spatially autocorrelated values
    # Start with random values
    base = np.random.randn(n)
    
    # Apply spatial smoothing to create autocorrelation
    W = compute_spatial_weights(coords, method='knn', k=8)
    W_dense = W.toarray()
    
    # Smooth: take average of neighbors
    smoothed = 0.3 * base + 0.7 * (W_dense @ base)
    
    # Create "ground truth" and predictions
    ground_truth = smoothed
    predictions = ground_truth + np.random.randn(n) * 0.2  # Add noise
    
    # Create mock GeoDataFrame
    import geopandas as gpd
    from shapely.geometry import Point
    
    addresses = gpd.GeoDataFrame({
        'geometry': [Point(x[i], y[i]) for i in range(n)]
    })
    
    # Run analysis
    results = analyze_spatial_autocorrelation(
        addresses, predictions, ground_truth,
        method='knn', k=8, permutations=999
    )
    
    # Interpretation
    print("\n" + "-"*70)
    print("INTERPRETATION GUIDE")
    print("-"*70)
    print("""
For GRANITE validation:

PREDICTIONS should have POSITIVE spatial autocorrelation:
  - Nearby addresses should have similar vulnerability predictions
  - This indicates the GNN learned spatial structure
  - I > 0 with p < 0.05 is expected and good

RESIDUALS should have LOW or NO spatial autocorrelation:
  - If residuals cluster spatially, the model misses some pattern
  - I near 0 with p > 0.05 means model captures spatial structure well
  - Significant positive I in residuals = room for improvement
""")
    
    # Create plot
    create_morans_i_plot(results, '/home/claude/morans_i_demo.png')
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Moran's I Spatial Autocorrelation Analysis")
    parser.add_argument('--demo', action='store_true', help='Run demo with synthetic data')
    parser.add_argument('--permutations', type=int, default=999, help='Permutations for inference')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_with_synthetic_data()
    else:
        print("Use --demo to run demonstration with synthetic data")
        print("For real data, import and call analyze_spatial_autocorrelation() directly")
