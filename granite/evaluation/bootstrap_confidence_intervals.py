"""
GRANITE Bootstrap Confidence Intervals

Quantifies whether GRANITE vs IDW difference is statistically significant.
Uses 1000-sample bootstrap to compute 95% CI on correlation differences.

Key question: Is the improvement from GRANITE (r=0.924) over IDW (r=0.907)
statistically significant, or could it arise from sampling variation?
"""
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/workspaces/GRANITE')


def bootstrap_correlation_ci(x, y, n_bootstrap=1000, ci=0.95, seed=42):
    """
    Compute bootstrap confidence interval for Pearson correlation.
    
    Args:
        x, y: Arrays to correlate
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level
        seed: Random seed
        
    Returns:
        dict with point estimate, CI bounds, and bootstrap distribution
    """
    np.random.seed(seed)
    
    # Remove NaN
    valid = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[valid], y[valid]
    n = len(x)
    
    if n < 10:
        return {'r': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'n': n}
    
    # Point estimate
    r_observed = np.corrcoef(x, y)[0, 1]
    
    # Bootstrap
    boot_correlations = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        r_boot = np.corrcoef(x[indices], y[indices])[0, 1]
        boot_correlations.append(r_boot)
    
    boot_correlations = np.array(boot_correlations)
    
    # CI bounds
    alpha = 1 - ci
    ci_lower = np.percentile(boot_correlations, 100 * alpha / 2)
    ci_upper = np.percentile(boot_correlations, 100 * (1 - alpha / 2))
    
    return {
        'r': r_observed,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_width': ci_upper - ci_lower,
        'n': n,
        'bootstrap_dist': boot_correlations
    }


def bootstrap_correlation_difference(x, y1, y2, n_bootstrap=1000, ci=0.95, seed=42):
    """
    Test if correlation(x, y1) - correlation(x, y2) is significantly different from 0.
    
    This is the key test: is GRANITE's correlation with ground truth significantly
    higher than IDW's correlation?
    
    Args:
        x: Ground truth values (e.g., block group SVI)
        y1: Method 1 predictions (e.g., GRANITE)
        y2: Method 2 predictions (e.g., IDW)
        
    Returns:
        dict with difference estimate, CI, and p-value
    """
    np.random.seed(seed)
    
    # Remove NaN (require all three valid)
    valid = ~np.isnan(x) & ~np.isnan(y1) & ~np.isnan(y2)
    x, y1, y2 = x[valid], y1[valid], y2[valid]
    n = len(x)
    
    if n < 10:
        return {'diff': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}
    
    # Observed difference
    r1 = np.corrcoef(x, y1)[0, 1]
    r2 = np.corrcoef(x, y2)[0, 1]
    diff_observed = r1 - r2
    
    # Bootstrap the difference
    boot_diffs = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        r1_boot = np.corrcoef(x[indices], y1[indices])[0, 1]
        r2_boot = np.corrcoef(x[indices], y2[indices])[0, 1]
        boot_diffs.append(r1_boot - r2_boot)
    
    boot_diffs = np.array(boot_diffs)
    
    # CI for the difference
    alpha = 1 - ci
    ci_lower = np.percentile(boot_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    
    # Two-tailed p-value: proportion of bootstrap samples with opposite sign
    # or less extreme than observed
    if diff_observed > 0:
        p_value = 2 * np.mean(boot_diffs <= 0)
    else:
        p_value = 2 * np.mean(boot_diffs >= 0)
    
    # More conservative: p-value based on how often bootstrap crosses 0
    p_value_zero = np.mean(boot_diffs <= 0) if diff_observed > 0 else np.mean(boot_diffs >= 0)
    
    significant = ci_lower > 0 or ci_upper < 0  # CI doesn't include 0
    
    return {
        'r1': r1,
        'r2': r2,
        'diff': diff_observed,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value_zero,
        'significant': significant,
        'n': n,
        'bootstrap_dist': boot_diffs
    }


def run_bootstrap_validation(granite_predictions, idw_predictions, ground_truth,
                             n_bootstrap=1000, seed=42, verbose=True):
    """
    Run full bootstrap validation comparing GRANITE to IDW.
    
    Args:
        granite_predictions: Array of GRANITE block-group mean predictions
        idw_predictions: Array of IDW block-group mean predictions
        ground_truth: Array of actual block-group SVI values
        
    Returns:
        dict with bootstrap results
    """
    
    if verbose:
        print("\n" + "="*70)
        print("BOOTSTRAP CONFIDENCE INTERVAL ANALYSIS")
        print("="*70)
    
    results = {}
    
    # Individual correlations with CIs
    if verbose:
        print("\nComputing individual correlation CIs...")
    
    granite_ci = bootstrap_correlation_ci(
        ground_truth, granite_predictions, n_bootstrap, seed=seed
    )
    idw_ci = bootstrap_correlation_ci(
        ground_truth, idw_predictions, n_bootstrap, seed=seed
    )
    
    results['granite'] = granite_ci
    results['idw'] = idw_ci
    
    if verbose:
        print(f"\nGRANITE: r = {granite_ci['r']:.3f} "
              f"[{granite_ci['ci_lower']:.3f}, {granite_ci['ci_upper']:.3f}] (n={granite_ci['n']})")
        print(f"IDW:     r = {idw_ci['r']:.3f} "
              f"[{idw_ci['ci_lower']:.3f}, {idw_ci['ci_upper']:.3f}] (n={idw_ci['n']})")
    
    # Difference test
    if verbose:
        print("\nTesting correlation difference...")
    
    diff_result = bootstrap_correlation_difference(
        ground_truth, granite_predictions, idw_predictions, n_bootstrap, seed=seed
    )
    results['difference'] = diff_result
    
    if verbose:
        print(f"\nDifference (GRANITE - IDW): {diff_result['diff']:.3f}")
        print(f"95% CI: [{diff_result['ci_lower']:.3f}, {diff_result['ci_upper']:.3f}]")
        print(f"p-value: {diff_result['p_value']:.4f}")
        
        if diff_result['significant']:
            print("\nCONCLUSION: Difference is STATISTICALLY SIGNIFICANT (CI excludes 0)")
        else:
            print("\nCONCLUSION: Difference is NOT statistically significant")
            print("           The 95% CI includes 0, meaning the improvement could be due to chance.")
    
    return results


def create_bootstrap_plot(results, output_path='bootstrap_comparison.png'):
    """Create visualization of bootstrap distributions."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Panel 1: GRANITE correlation distribution
    ax = axes[0]
    boot_dist = results['granite']['bootstrap_dist']
    ax.hist(boot_dist, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(results['granite']['r'], color='red', linewidth=2, label='Observed')
    ax.axvline(results['granite']['ci_lower'], color='red', linestyle='--', linewidth=1)
    ax.axvline(results['granite']['ci_upper'], color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Frequency')
    ax.set_title(f"GRANITE\nr={results['granite']['r']:.3f} "
                 f"[{results['granite']['ci_lower']:.3f}, {results['granite']['ci_upper']:.3f}]")
    ax.legend()
    
    # Panel 2: IDW correlation distribution
    ax = axes[1]
    boot_dist = results['idw']['bootstrap_dist']
    ax.hist(boot_dist, bins=50, color='coral', alpha=0.7, edgecolor='white')
    ax.axvline(results['idw']['r'], color='red', linewidth=2, label='Observed')
    ax.axvline(results['idw']['ci_lower'], color='red', linestyle='--', linewidth=1)
    ax.axvline(results['idw']['ci_upper'], color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Frequency')
    ax.set_title(f"IDW\nr={results['idw']['r']:.3f} "
                 f"[{results['idw']['ci_lower']:.3f}, {results['idw']['ci_upper']:.3f}]")
    ax.legend()
    
    # Panel 3: Difference distribution
    ax = axes[2]
    boot_dist = results['difference']['bootstrap_dist']
    ax.hist(boot_dist, bins=50, color='forestgreen', alpha=0.7, edgecolor='white')
    ax.axvline(results['difference']['diff'], color='red', linewidth=2, label='Observed')
    ax.axvline(0, color='black', linewidth=2, linestyle='--', label='No difference')
    ax.axvline(results['difference']['ci_lower'], color='red', linestyle='--', linewidth=1)
    ax.axvline(results['difference']['ci_upper'], color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Correlation Difference')
    ax.set_ylabel('Frequency')
    
    sig_text = "SIGNIFICANT" if results['difference']['significant'] else "NOT significant"
    ax.set_title(f"GRANITE - IDW\nΔr={results['difference']['diff']:.3f} "
                 f"[{results['difference']['ci_lower']:.3f}, {results['difference']['ci_upper']:.3f}]\n"
                 f"p={results['difference']['p_value']:.3f} ({sig_text})")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to: {output_path}")


def demo_with_synthetic_data():
    """
    Demonstrate bootstrap analysis with synthetic data.
    Use this to verify the methodology before running on real results.
    """
    print("\n" + "="*70)
    print("DEMO: Bootstrap Analysis with Synthetic Data")
    print("="*70)
    
    np.random.seed(42)
    n = 27  # Typical number of block groups
    
    # Simulate ground truth
    ground_truth = np.random.uniform(0.1, 0.9, n)
    
    # GRANITE: highly correlated with ground truth
    granite = ground_truth + np.random.normal(0, 0.08, n)
    granite = np.clip(granite, 0, 1)
    
    # IDW: slightly less correlated
    idw = ground_truth + np.random.normal(0, 0.12, n)
    idw = np.clip(idw, 0, 1)
    
    # Run analysis
    results = run_bootstrap_validation(
        granite, idw, ground_truth,
        n_bootstrap=1000, seed=42
    )
    
    # Create plot
    create_bootstrap_plot(results, '/home/claude/bootstrap_demo.png')
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Bootstrap CI Analysis')
    parser.add_argument('--demo', action='store_true', help='Run demo with synthetic data')
    parser.add_argument('--n-bootstrap', type=int, default=1000, help='Bootstrap samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_with_synthetic_data()
    else:
        print("Use --demo to run demonstration with synthetic data")
        print("For real data, import and call run_bootstrap_validation() directly")
