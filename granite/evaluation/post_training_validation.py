"""
GRANITE Post-Training Validation Suite

Single integrated script to run all validation analyses after global training.

Usage:
    python run_granite.py --global-training
    python post_training_validation.py --results-dir ./output/global_validation

Runs:
    1. Block group validation (GRANITE vs Dasymetric vs Pycnophylactic)
    2. Bootstrap confidence intervals (statistical significance)
    3. Moran's I spatial autocorrelation
    4. Expert routing feature analysis
    5. (Optional) Ablation study - accessibility-only model

Outputs:
    - validation_report.txt (comprehensive text report)
    - validation_summary.csv (key metrics)
    - bootstrap_comparison.png
    - morans_i_analysis.png
    - expert_routing_summary.csv
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/workspaces/GRANITE')


# =============================================================================
# DATA LOADING
# =============================================================================

def load_global_training_results(results_dir):
    """Load all outputs from global training run."""
    
    print("\n" + "="*70)
    print("LOADING GLOBAL TRAINING RESULTS")
    print("="*70)
    
    results = {}
    
    # Load validation results CSV
    csv_path = os.path.join(results_dir, 'validation_results.csv')
    if os.path.exists(csv_path):
        results['validation_df'] = pd.read_csv(csv_path)
        print(f"Loaded {len(results['validation_df'])} tract results")
    else:
        raise FileNotFoundError(f"No validation_results.csv found in {results_dir}")
    
    # Convert to dict format for compatibility
    results['tract_results'] = {}
    for _, row in results['validation_df'].iterrows():
        fips = str(row['fips'])
        results['tract_results'][fips] = {
            'actual_svi': row['actual_svi'],
            'predicted_mean': row['predicted_svi'],
            'error_pct': row['error_pct'],
            'dominant_expert': row.get('dominant_expert', 'Unknown')
        }
    
    return results


def load_spatial_data():
    """Load census tracts, addresses, and destinations."""
    
    print("\nLoading spatial data...")
    
    from granite.data.loaders import DataLoader
    
    loader = DataLoader()
    
    data = {}
    data['loader'] = loader
    data['census_tracts'] = loader.load_census_tracts('47', '065')
    data['svi'] = loader.load_svi_data('47', 'Hamilton')
    data['tracts'] = data['census_tracts'].merge(data['svi'], on='FIPS', how='inner')
    data['employment_destinations'] = loader.create_employment_destinations(use_real_data=True)
    data['healthcare_destinations'] = loader.create_healthcare_destinations(use_real_data=True)
    data['grocery_destinations'] = loader.create_grocery_destinations(use_real_data=True)
    
    # Load block groups for validation
    try:
        bg_geometries = loader.block_group_loader.load_block_group_geometries('47', '065')
        bg_demographics = loader.block_group_loader.fetch_acs_demographics('47', '065')
        data['block_groups'] = (bg_geometries, bg_demographics)
        print(f"Block group data loaded: {len(bg_geometries)} geometries")
    except Exception as e:
        print(f"WARNING: Block group data not available: {e}")
        data['block_groups'] = None
    
    print(f"Loaded {len(data['tracts'])} tracts")
    
    return data


def load_test_tract_data(tract_fips_list, data, verbose=True):
    """Load addresses for test tracts (fast - no feature computation)."""
    
    if verbose:
        print(f"\nLoading addresses for {len(tract_fips_list)} test tracts...")
    
    loader = data['loader']
    
    all_addresses = []
    
    for fips in tract_fips_list:
        try:
            addresses = loader.get_addresses_for_tract(fips)
            if len(addresses) == 0:
                if verbose:
                    print(f"  {fips}: No addresses, skipping")
                continue
            
            addresses['tract_fips'] = fips
            all_addresses.append(addresses)
            
            if verbose:
                print(f"  {fips}: {len(addresses)} addresses")
                
        except Exception as e:
            if verbose:
                print(f"  {fips}: Error - {e}")
    
    if all_addresses:
        combined_addresses = pd.concat(all_addresses, ignore_index=True)
        combined_addresses = gpd.GeoDataFrame(
            combined_addresses, geometry='geometry', crs='EPSG:4326'
        )
    else:
        combined_addresses = None
    
    print(f"  Total: {len(combined_addresses) if combined_addresses is not None else 0} addresses")
    
    return combined_addresses


# =============================================================================
# VALIDATION ANALYSES
# =============================================================================

def run_block_group_validation(addresses, predictions_dict, data, output_dir):
    """
    Run block group validation comparing GRANITE, Dasymetric, and Pycnophylactic.
    
    Args:
        addresses: GeoDataFrame with address points
        predictions_dict: Dict of {method_name: predictions_array}
        data: Spatial data dict with block_groups
        output_dir: Output directory
    """
    
    print("\n" + "="*70)
    print("BLOCK GROUP VALIDATION")
    print("="*70)
    
    if data['block_groups'] is None:
        print("Block group data not available, skipping")
        return None
    
    bg_geometries, bg_demographics = data['block_groups']
    
    # Merge geometries with demographics
    block_groups = bg_geometries.merge(bg_demographics, on='GEOID', how='left')
    
    # Ensure CRS match
    if addresses.crs != block_groups.crs:
        addresses = addresses.to_crs(block_groups.crs)
    
    # Assign addresses to block groups
    print("\nAssigning addresses to block groups...")
    joined = gpd.sjoin(
        addresses,
        block_groups[['GEOID', 'geometry']],
        how='left',
        predicate='within'
    )
    joined = joined.rename(columns={'GEOID': 'block_group_id'})
    
    if 'index_right' in joined.columns:
        joined = joined.drop(columns=['index_right'])
    
    n_assigned = joined['block_group_id'].notna().sum()
    print(f"  Assigned {n_assigned}/{len(addresses)} addresses to block groups")
    
    results = {}
    
    for method_name, predictions in predictions_dict.items():
        print(f"\nValidating {method_name}...")
        
        # Aggregate predictions to block group level
        df = pd.DataFrame({
            'block_group_id': joined['block_group_id'].values,
            'prediction': predictions
        })
        df = df.dropna(subset=['block_group_id'])
        
        agg = df.groupby('block_group_id').agg(
            predicted_svi=('prediction', 'mean'),
            prediction_std=('prediction', 'std'),
            n_addresses=('prediction', 'count')
        ).reset_index()
        agg = agg.rename(columns={'block_group_id': 'GEOID'})
        
        # Merge with block group SVI
        svi_cols = ['GEOID', 'SVI', 'svi_complete']
        available_cols = [c for c in svi_cols if c in block_groups.columns]
        
        if 'SVI' not in block_groups.columns:
            print(f"  No SVI column in block groups, skipping correlation")
            results[method_name] = {
                'correlations': {},
                'n_block_groups': len(agg),
                'validation_data': agg
            }
            continue
        
        merged = agg.merge(block_groups[available_cols], on='GEOID', how='inner')
        
        # Require minimum addresses
        merged = merged[merged['n_addresses'] >= 5]
        
        # Compute correlation
        valid = merged['SVI'].notna() & merged['predicted_svi'].notna()
        
        if valid.sum() >= 10:
            r, p = stats.pearsonr(
                merged.loc[valid, 'predicted_svi'],
                merged.loc[valid, 'SVI']
            )
            rho, _ = stats.spearmanr(
                merged.loc[valid, 'predicted_svi'],
                merged.loc[valid, 'SVI']
            )
            
            print(f"  Pearson r = {r:.3f} (p = {p:.4f})")
            print(f"  Spearman rho = {rho:.3f}")
            print(f"  N block groups = {valid.sum()}")
            
            correlations = {
                'svi_correlation': {
                    'pearson_r': float(r),
                    'spearman_rho': float(rho),
                    'p_value': float(p),
                    'n': int(valid.sum())
                }
            }
        else:
            print(f"  Insufficient valid block groups ({valid.sum()})")
            correlations = {}
        
        results[method_name] = {
            'correlations': correlations,
            'n_block_groups': len(merged),
            'validation_data': merged
        }
    
    # Create comparison plot
    _create_block_group_comparison_plot(results, output_dir)
    
    return results


def _create_block_group_comparison_plot(results, output_dir):
    """Create block group validation comparison plot."""
    import matplotlib.pyplot as plt
    
    methods = list(results.keys())
    n_methods = len(methods)
    
    if n_methods == 0:
        return
    
    fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 4))
    if n_methods == 1:
        axes = [axes]
    
    colors = {'GRANITE': 'steelblue', 'Dasymetric': '#E65100', 'Pycnophylactic': '#1565C0'}

    for i, method in enumerate(methods):
        ax = axes[i]
        data = results[method]
        df = data['validation_data']

        if 'SVI' not in df.columns:
            ax.text(0.5, 0.5, 'No SVI data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(method)
            continue

        valid = df['SVI'].notna() & df['predicted_svi'].notna()

        if valid.sum() > 0:
            ax.scatter(df.loc[valid, 'SVI'], df.loc[valid, 'predicted_svi'],
                      alpha=0.6, color=colors.get(method, 'gray'), s=30)
            
            # Add diagonal line
            lims = [0, 1]
            ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect')
            
            # Add correlation
            corr = data['correlations'].get('svi_correlation', {})
            r = corr.get('pearson_r', np.nan)
            
            ax.set_xlabel('Block Group SVI (Ground Truth)')
            ax.set_ylabel('Predicted SVI (Aggregated)')
            ax.set_title(f'{method}\nr = {r:.3f}')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'block_group_validation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved: {os.path.join(output_dir, 'block_group_validation.png')}")


def run_bootstrap_analysis(bg_results, output_dir, n_bootstrap=1000, seed=42):
    """
    Run bootstrap confidence intervals on block group correlations.
    """
    
    print("\n" + "="*70)
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print("="*70)
    
    if bg_results is None or len(bg_results) < 2:
        print("Insufficient block group results for bootstrap analysis")
        return None
    
    # Extract validation data
    methods = list(bg_results.keys())
    
    # Get ground truth and predictions from first method's validation data
    base_df = bg_results[methods[0]]['validation_data']
    
    if 'SVI' not in base_df.columns:
        print("No ground truth SVI in block group data")
        return None
    
    ground_truth = base_df['SVI'].values
    
    results = {}
    
    # Bootstrap each method's correlation
    np.random.seed(seed)
    
    for method in methods:
        df = bg_results[method]['validation_data']
        predictions = df['predicted_svi'].values
        
        # Align with ground truth
        valid = ~np.isnan(ground_truth) & ~np.isnan(predictions)
        gt_valid = ground_truth[valid]
        pred_valid = predictions[valid]
        
        if len(gt_valid) < 10:
            print(f"  {method}: Insufficient valid samples")
            continue
        
        # Point estimate
        r_obs = np.corrcoef(gt_valid, pred_valid)[0, 1]
        
        # Bootstrap
        boot_rs = []
        n = len(gt_valid)
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            r_boot = np.corrcoef(gt_valid[idx], pred_valid[idx])[0, 1]
            boot_rs.append(r_boot)
        
        boot_rs = np.array(boot_rs)
        ci_lower = np.percentile(boot_rs, 2.5)
        ci_upper = np.percentile(boot_rs, 97.5)
        
        results[method] = {
            'r': r_obs,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n': len(gt_valid),
            'bootstrap_dist': boot_rs
        }
        
        print(f"  {method}: r = {r_obs:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    # Test difference between methods (if GRANITE and Dasymetric present)
    if 'GRANITE' in results and 'Dasymetric' in results:
        print("\nTesting GRANITE vs Dasymetric difference...")

        granite_dist = results['GRANITE']['bootstrap_dist']
        dasy_dist = results['Dasymetric']['bootstrap_dist']
        diff_dist = granite_dist - dasy_dist

        diff_obs = results['GRANITE']['r'] - results['Dasymetric']['r']
        diff_ci_lower = np.percentile(diff_dist, 2.5)
        diff_ci_upper = np.percentile(diff_dist, 97.5)
        
        # two-tailed p-value: shift to null (diff=0), measure extremity
        diff_dist_null = diff_dist - diff_obs
        p_value = np.mean(np.abs(diff_dist_null) >= np.abs(diff_obs))
        p_value = max(p_value, 1.0 / len(diff_dist))
        significant = diff_ci_lower > 0 or diff_ci_upper < 0
        
        results['difference'] = {
            'diff': diff_obs,
            'ci_lower': diff_ci_lower,
            'ci_upper': diff_ci_upper,
            'p_value': p_value,
            'significant': significant
        }
        
        print(f"  Difference: {diff_obs:.3f} [{diff_ci_lower:.3f}, {diff_ci_upper:.3f}]")
        print(f"  p-value: {p_value:.4f}")
        
        if significant:
            print("  SIGNIFICANT: CI excludes 0")
        else:
            print("  NOT significant: CI includes 0")
    
    # Create plot
    _create_bootstrap_plot(results, output_dir)
    
    return results


def _create_bootstrap_plot(results, output_dir):
    """Create bootstrap distribution visualization."""
    import matplotlib.pyplot as plt
    
    methods = [m for m in results.keys() if m != 'difference']
    n_panels = len(methods) + (1 if 'difference' in results else 0)
    
    fig, axes = plt.subplots(1, n_panels, figsize=(5*n_panels, 4))
    if n_panels == 1:
        axes = [axes]
    
    colors = {'GRANITE': 'steelblue', 'Dasymetric': '#E65100', 'Pycnophylactic': '#1565C0'}

    for i, method in enumerate(methods):
        ax = axes[i]
        data = results[method]

        ax.hist(data['bootstrap_dist'], bins=50, color=colors.get(method, 'gray'),
                alpha=0.7, edgecolor='white')
        ax.axvline(data['r'], color='red', linewidth=2, label=f"r = {data['r']:.3f}")
        ax.axvline(data['ci_lower'], color='red', linestyle='--', linewidth=1)
        ax.axvline(data['ci_upper'], color='red', linestyle='--', linewidth=1)
        ax.set_xlabel('Correlation')
        ax.set_ylabel('Frequency')
        ax.set_title(f"{method}\nr = {data['r']:.3f} [{data['ci_lower']:.3f}, {data['ci_upper']:.3f}]")
        ax.legend()
    
    if 'difference' in results:
        ax = axes[-1]
        diff = results['difference']
        
        # Need to compute diff distribution
        if 'GRANITE' in results and 'Dasymetric' in results:
            diff_dist = results['GRANITE']['bootstrap_dist'] - results['Dasymetric']['bootstrap_dist']

            ax.hist(diff_dist, bins=50, color='purple', alpha=0.7, edgecolor='white')
            ax.axvline(diff['diff'], color='red', linewidth=2)
            ax.axvline(0, color='black', linewidth=2, linestyle='--', label='No difference')
            ax.axvline(diff['ci_lower'], color='red', linestyle='--', linewidth=1)
            ax.axvline(diff['ci_upper'], color='red', linestyle='--', linewidth=1)
            ax.set_xlabel('Correlation Difference')
            ax.set_ylabel('Frequency')

            sig_text = "SIGNIFICANT" if diff['significant'] else "NOT significant"
            ax.set_title(f"GRANITE - Dasymetric\n\u0394 = {diff['diff']:.3f} [{diff['ci_lower']:.3f}, {diff['ci_upper']:.3f}]\n({sig_text})")
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bootstrap_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved: {os.path.join(output_dir, 'bootstrap_comparison.png')}")


def run_morans_i_analysis(addresses, predictions, ground_truth=None, output_dir=None,
                          k=8, permutations=999, seed=42):
    """
    Run Moran's I spatial autocorrelation analysis.
    """
    
    print("\n" + "="*70)
    print("MORAN'S I SPATIAL AUTOCORRELATION")
    print("="*70)
    
    from scipy.spatial import distance
    from scipy import sparse
    
    # Get coordinates
    coords = np.column_stack([addresses.geometry.x, addresses.geometry.y])
    n = len(coords)
    
    print(f"Analyzing {n} addresses...")
    
    # Compute spatial weights (KNN)
    dist_matrix = distance.cdist(coords, coords)
    W = np.zeros((n, n))
    for i in range(n):
        neighbors = np.argsort(dist_matrix[i])[1:k+1]
        W[i, neighbors] = 1
    
    # symmetrize before row-normalizing (k-NN graphs are asymmetric)
    W = (W + W.T) / 2

    # Row-standardize
    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1
    W = W / row_sums[:, np.newaxis]

    def compute_morans_i(values, W, permutations, seed):
        np.random.seed(seed)
        n = len(values)
        z = values - values.mean()
        
        numerator = np.sum(z * (W @ z))
        denominator = np.sum(z ** 2)
        S0 = W.sum()
        
        I = (n / S0) * (numerator / denominator)
        E_I = -1 / (n - 1)
        
        # Permutation test
        perm_Is = []
        for _ in range(permutations):
            z_perm = np.random.permutation(z)
            num_perm = np.sum(z_perm * (W @ z_perm))
            I_perm = (n / S0) * (num_perm / denominator)
            perm_Is.append(I_perm)
        
        perm_Is = np.array(perm_Is)
        p_value = (np.sum(np.abs(perm_Is) >= np.abs(I)) + 1) / (permutations + 1)
        z_score = (I - np.mean(perm_Is)) / np.std(perm_Is)
        
        return {
            'I': float(I),
            'E_I': float(E_I),
            'z_score': float(z_score),
            'p_value': float(p_value),
            'perm_dist': perm_Is
        }
    
    results = {}
    
    # Predictions
    print("\nPredictions:")
    moran_pred = compute_morans_i(predictions, W, permutations, seed)
    results['predictions'] = moran_pred
    print(f"  Moran's I = {moran_pred['I']:.4f}")
    print(f"  Z-score = {moran_pred['z_score']:.2f}")
    print(f"  p-value = {moran_pred['p_value']:.4f}")
    
    if moran_pred['p_value'] < 0.05 and moran_pred['I'] > 0:
        print("  -> SIGNIFICANT positive spatial autocorrelation (good!)")
    elif moran_pred['p_value'] < 0.05:
        print("  -> SIGNIFICANT but negative (unexpected)")
    else:
        print("  -> Not significant")
    
    # Residuals (if ground truth provided)
    if ground_truth is not None:
        print("\nResiduals:")
        residuals = predictions - ground_truth
        moran_resid = compute_morans_i(residuals, W, permutations, seed)
        results['residuals'] = moran_resid
        print(f"  Moran's I = {moran_resid['I']:.4f}")
        print(f"  Z-score = {moran_resid['z_score']:.2f}")
        print(f"  p-value = {moran_resid['p_value']:.4f}")
        
        if moran_resid['p_value'] >= 0.05:
            print("  -> Not significant (model captures spatial structure well)")
        elif moran_resid['I'] > 0:
            print("  -> Significant positive autocorrelation in residuals")
            print("     Model may be missing some spatial pattern")
    
    # Create plot
    if output_dir:
        _create_morans_plot(results, output_dir)
    
    return results


def _create_morans_plot(results, output_dir):
    """Create Moran's I visualization."""
    import matplotlib.pyplot as plt
    
    n_panels = len(results)
    fig, axes = plt.subplots(1, n_panels, figsize=(6*n_panels, 4))
    if n_panels == 1:
        axes = [axes]
    
    for i, (key, data) in enumerate(results.items()):
        ax = axes[i]
        
        ax.hist(data['perm_dist'], bins=50, color='steelblue' if key == 'predictions' else 'coral',
                alpha=0.7, edgecolor='white')
        ax.axvline(data['I'], color='red', linewidth=2, label=f"I = {data['I']:.4f}")
        ax.axvline(data['E_I'], color='black', linestyle='--', label=f"E[I] = {data['E_I']:.4f}")
        ax.set_xlabel("Moran's I")
        ax.set_ylabel('Frequency')
        ax.set_title(f"{key.title()}\nI = {data['I']:.4f}, p = {data['p_value']:.4f}")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'morans_i_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved: {os.path.join(output_dir, 'morans_i_analysis.png')}")


def run_expert_routing_simplified(tract_results, output_dir):
    """
    Simplified expert routing analysis using only validation results.
    Does not require accessibility feature computation.
    """
    
    print("\n" + "="*70)
    print("EXPERT ROUTING ANALYSIS (Simplified)")
    print("="*70)
    
    # Group by expert
    expert_groups = defaultdict(list)
    for fips, result in tract_results.items():
        expert = result.get('dominant_expert', 'Unknown')
        expert_groups[expert].append(fips)
    
    print("\nTracts per expert:")
    for expert, tracts in sorted(expert_groups.items()):
        print(f"  {expert}: {len(tracts)} tracts")
    
    results = {
        'expert_groups': dict(expert_groups),
        'expert_profiles': {},
        'counterintuitive_cases': []
    }
    
    # Compute SVI profiles per expert
    for expert, tracts in expert_groups.items():
        svis = [tract_results[fips]['actual_svi'] for fips in tracts]
        errors = [tract_results[fips]['error_pct'] for fips in tracts]
        
        results['expert_profiles'][expert] = {
            'n_tracts': len(tracts),
            'svi_mean': np.mean(svis),
            'svi_std': np.std(svis),
            'svi_range': (min(svis), max(svis)),
            'mean_error': np.mean(errors)
        }
        
        print(f"\n{expert} Expert:")
        print(f"  Tracts: {len(tracts)}")
        print(f"  SVI range: {min(svis):.3f} - {max(svis):.3f}")
        print(f"  SVI mean: {np.mean(svis):.3f}")
        print(f"  Mean error: {np.mean(errors):.1f}%")
    
    # Identify counterintuitive cases
    print("\n" + "-"*70)
    print("COUNTERINTUITIVE ROUTING")
    print("-"*70)
    
    for fips, result in tract_results.items():
        svi = result['actual_svi']
        expert = result.get('dominant_expert', 'Unknown')
        
        is_counter = False
        reason = ""
        
        if svi < 0.3 and 'High' in expert:
            is_counter = True
            reason = f"Low SVI ({svi:.3f}) -> {expert} expert"
        elif svi > 0.7 and 'Low' in expert:
            is_counter = True
            reason = f"High SVI ({svi:.3f}) -> {expert} expert"
        
        if is_counter:
            results['counterintuitive_cases'].append({
                'fips': fips,
                'svi': svi,
                'expert': expert,
                'reason': reason
            })
            print(f"  Tract {fips}: {reason}")
    
    if len(results['counterintuitive_cases']) == 0:
        print("  No counterintuitive cases found.")
    
    # Save summary
    rows = []
    for fips, result in tract_results.items():
        rows.append({
            'FIPS': fips,
            'Actual_SVI': result['actual_svi'],
            'Predicted_SVI': result['predicted_mean'],
            'Expert': result.get('dominant_expert', 'Unknown'),
            'Error_Pct': result['error_pct']
        })
    
    summary_df = pd.DataFrame(rows).sort_values('Actual_SVI')
    summary_df.to_csv(os.path.join(output_dir, 'expert_routing_summary.csv'), index=False)
    print(f"\nSummary saved: {os.path.join(output_dir, 'expert_routing_summary.csv')}")
    
    # Print summary table
    print("\n" + "-"*70)
    print("ROUTING SUMMARY")
    print("-"*70)
    print(summary_df.to_string(index=False))
    
    return results


def run_expert_routing_analysis(tract_results, tract_features, feature_names, output_dir):
    """
    Run expert routing feature analysis.
    """
    
    print("\n" + "="*70)
    print("EXPERT ROUTING FEATURE ANALYSIS")
    print("="*70)
    
    # Group by expert
    expert_groups = defaultdict(list)
    for fips, result in tract_results.items():
        expert = result.get('dominant_expert', 'Unknown')
        expert_groups[expert].append(fips)
    
    print("\nTracts per expert:")
    for expert, tracts in sorted(expert_groups.items()):
        print(f"  {expert}: {len(tracts)} tracts")
    
    results = {
        'expert_groups': dict(expert_groups),
        'feature_profiles': {},
        'counterintuitive_cases': []
    }
    
    # Compute feature profiles
    for expert, tracts in expert_groups.items():
        expert_feats = []
        expert_svis = []
        
        for fips in tracts:
            if fips in tract_features:
                expert_feats.append(tract_features[fips])
            expert_svis.append(tract_results[fips]['actual_svi'])
        
        if len(expert_feats) == 0:
            continue
        
        expert_feats = np.vstack(expert_feats)
        
        results['feature_profiles'][expert] = {
            'n_tracts': len(tracts),
            'svi_mean': np.mean(expert_svis),
            'svi_range': (min(expert_svis), max(expert_svis)),
            'feature_means': np.mean(expert_feats, axis=0),
            'feature_stds': np.std(expert_feats, axis=0)
        }
        
        print(f"\n{expert} Expert:")
        print(f"  Tracts: {len(tracts)}")
        print(f"  SVI range: {min(expert_svis):.3f} - {max(expert_svis):.3f}")
    
    # Find distinguishing features
    print("\n" + "-"*70)
    print("DISTINGUISHING FEATURES")
    print("-"*70)
    
    experts = list(results['feature_profiles'].keys())
    
    for i, exp1 in enumerate(experts):
        for exp2 in experts[i+1:]:
            prof1 = results['feature_profiles'][exp1]
            prof2 = results['feature_profiles'][exp2]
            
            if prof1['n_tracts'] < 2 or prof2['n_tracts'] < 2:
                continue
            
            diffs = []
            for j, fname in enumerate(feature_names):
                if j >= len(prof1['feature_means']):
                    continue
                m1, m2 = prof1['feature_means'][j], prof2['feature_means'][j]
                std_pool = np.sqrt((prof1['feature_stds'][j]**2 + prof2['feature_stds'][j]**2) / 2)
                effect = abs(m1 - m2) / std_pool if std_pool > 0 else 0
                diffs.append((fname, effect, m1, m2))
            
            diffs.sort(key=lambda x: -x[1])
            
            print(f"\n{exp1} vs {exp2}:")
            print(f"  {'Feature':<40} {'Effect':>8} {exp1:>10} {exp2:>10}")
            print(f"  {'-'*70}")
            for fname, eff, m1, m2 in diffs[:5]:
                print(f"  {fname:<40} {eff:>8.2f} {m1:>10.3f} {m2:>10.3f}")
    
    # Find counterintuitive cases
    print("\n" + "-"*70)
    print("COUNTERINTUITIVE ROUTING")
    print("-"*70)
    
    for fips, result in tract_results.items():
        svi = result['actual_svi']
        expert = result.get('dominant_expert', 'Unknown')
        
        is_counter = False
        reason = ""
        
        if svi < 0.3 and 'High' in expert:
            is_counter = True
            reason = f"Low SVI ({svi:.3f}) -> {expert} expert"
        elif svi > 0.7 and 'Low' in expert:
            is_counter = True
            reason = f"High SVI ({svi:.3f}) -> {expert} expert"
        
        if is_counter:
            results['counterintuitive_cases'].append({
                'fips': fips,
                'svi': svi,
                'expert': expert,
                'reason': reason
            })
            
            print(f"\n  Tract {fips}: {reason}")
            
            # Explain via features
            if fips in tract_features and expert in results['feature_profiles']:
                feats = tract_features[fips]
                prof = results['feature_profiles'][expert]
                
                # Find features closest to expert profile
                matches = []
                for j, fname in enumerate(feature_names):
                    if j >= len(feats) or j >= len(prof['feature_means']):
                        continue
                    if 'min_time' in fname or 'transit' in fname or 'dependence' in fname:
                        val = feats[j]
                        exp_mean = prof['feature_means'][j]
                        exp_std = prof['feature_stds'][j]
                        if exp_std > 0:
                            z = abs(val - exp_mean) / exp_std
                            matches.append((fname, val, exp_mean, z))
                
                matches.sort(key=lambda x: x[3])
                
                print(f"    Accessibility features matching {expert} profile:")
                for fname, val, exp_mean, z in matches[:3]:
                    print(f"      {fname}: {val:.3f} (expert mean: {exp_mean:.3f})")
    
    if len(results['counterintuitive_cases']) == 0:
        print("  No counterintuitive cases found.")
    
    # Save summary
    rows = []
    for fips, result in tract_results.items():
        row = {
            'FIPS': fips,
            'Actual_SVI': result['actual_svi'],
            'Predicted_SVI': result['predicted_mean'],
            'Expert': result.get('dominant_expert', 'Unknown'),
            'Error_Pct': result['error_pct']
        }
        
        if fips in tract_features:
            feats = tract_features[fips]
            for j, fname in enumerate(feature_names):
                if j < len(feats) and 'min_time' in fname:
                    row[fname.replace('_min_time', '_MinTime')] = feats[j]
        
        rows.append(row)
    
    summary_df = pd.DataFrame(rows).sort_values('Actual_SVI')
    summary_df.to_csv(os.path.join(output_dir, 'expert_routing_summary.csv'), index=False)
    print(f"\nSummary saved: {os.path.join(output_dir, 'expert_routing_summary.csv')}")
    
    return results


# =============================================================================
# BASELINE COMPUTATION
# =============================================================================

def compute_baseline_predictions(addresses, tract_gdf, tract_results):
    """
    Compute Dasymetric and Pycnophylactic baseline predictions for comparison.
    """
    
    print("\n" + "="*70)
    print("COMPUTING BASELINE PREDICTIONS")
    print("="*70)
    
    from granite.evaluation.baselines import DasymetricDisaggregation, PycnophylacticDisaggregation

    n_addresses = len(addresses)
    predictions = {}

    # Dasymetric
    print("\nComputing Dasymetric predictions...")
    try:
        dasy = DasymetricDisaggregation(ancillary_column='nlcd_impervious_pct')
        dasy.fit(tract_gdf, svi_column='RPL_THEMES')

        dasy_preds = np.zeros(n_addresses)

        for fips, result in tract_results.items():
            mask = addresses['tract_fips'] == fips
            if mask.sum() == 0:
                continue

            addr_coords = np.column_stack([
                addresses.loc[mask, 'geometry'].apply(lambda g: g.x),
                addresses.loc[mask, 'geometry'].apply(lambda g: g.y)
            ])

            tract_preds = dasy.disaggregate(
                addr_coords, fips, result['actual_svi'],
                address_gdf=addresses.loc[mask]
            )
            dasy_preds[mask] = tract_preds

        predictions['Dasymetric'] = dasy_preds
        print(f"  Dasymetric: mean={np.mean(dasy_preds):.3f}, std={np.std(dasy_preds):.3f}")

    except Exception as e:
        print(f"  Dasymetric failed: {e}")
        import traceback
        traceback.print_exc()

    # Pycnophylactic
    print("Computing Pycnophylactic predictions...")
    try:
        pycno = PycnophylacticDisaggregation(n_iterations=50, k_neighbors=8)
        pycno.fit(tract_gdf, svi_column='RPL_THEMES')

        pycno_preds = np.zeros(n_addresses)

        for fips, result in tract_results.items():
            mask = addresses['tract_fips'] == fips
            if mask.sum() == 0:
                continue

            addr_coords = np.column_stack([
                addresses.loc[mask, 'geometry'].apply(lambda g: g.x),
                addresses.loc[mask, 'geometry'].apply(lambda g: g.y)
            ])

            tract_preds = pycno.disaggregate(addr_coords, fips, result['actual_svi'])
            pycno_preds[mask] = tract_preds

        predictions['Pycnophylactic'] = pycno_preds
        print(f"  Pycnophylactic: mean={np.mean(pycno_preds):.3f}, std={np.std(pycno_preds):.3f}")

    except Exception as e:
        print(f"  Pycnophylactic failed: {e}")
        import traceback
        traceback.print_exc()

    return predictions


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_validation_report(all_results, output_dir):
    """Generate text report."""
    
    lines = [
        "="*75,
        "GRANITE VALIDATION REPORT",
        "="*75,
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    
    # Block group validation
    if 'block_group' in all_results and all_results['block_group']:
        lines.extend([
            "-"*75,
            "1. BLOCK GROUP VALIDATION",
            "-"*75,
            "",
            f"{'Method':<15} {'r':>10} {'95% CI':>20} {'N':>8}",
            "-"*55
        ])
        
        bg = all_results['block_group']
        for method, data in bg.items():
            corr = data['correlations'].get('svi_correlation', {})
            r = corr.get('pearson_r', np.nan)
            n = data.get('n_block_groups', 0)
            lines.append(f"{method:<15} {r:>10.3f} {'N/A':>20} {n:>8}")
        
        lines.append("")
    
    # Bootstrap
    if 'bootstrap' in all_results and all_results['bootstrap']:
        lines.extend([
            "-"*75,
            "2. BOOTSTRAP CONFIDENCE INTERVALS",
            "-"*75,
            ""
        ])
        
        boot = all_results['bootstrap']
        for method, data in boot.items():
            if method == 'difference':
                continue
            lines.append(f"{method}: r = {data['r']:.3f} [{data['ci_lower']:.3f}, {data['ci_upper']:.3f}]")
        
        if 'difference' in boot:
            diff = boot['difference']
            sig = "SIGNIFICANT" if diff['significant'] else "NOT significant"
            lines.extend([
                "",
                f"GRANITE - Dasymetric: {diff['diff']:.3f} [{diff['ci_lower']:.3f}, {diff['ci_upper']:.3f}]",
                f"p-value: {diff['p_value']:.4f} ({sig})"
            ])
        
        lines.append("")
    
    # Moran's I
    if 'morans_i' in all_results and all_results['morans_i']:
        lines.extend([
            "-"*75,
            "3. MORAN'S I SPATIAL AUTOCORRELATION",
            "-"*75,
            ""
        ])
        
        moran = all_results['morans_i']
        for key, data in moran.items():
            sig = "significant" if data['p_value'] < 0.05 else "not significant"
            lines.append(f"{key.title()}: I = {data['I']:.4f}, p = {data['p_value']:.4f} ({sig})")
        
        lines.append("")
    
    # Expert routing
    if 'expert_routing' in all_results and all_results['expert_routing']:
        lines.extend([
            "-"*75,
            "4. EXPERT ROUTING ANALYSIS",
            "-"*75,
            ""
        ])
        
        routing = all_results['expert_routing']
        profiles_key = 'expert_profiles' if 'expert_profiles' in routing else 'feature_profiles'
        
        for expert, tracts in routing['expert_groups'].items():
            if profiles_key in routing and expert in routing[profiles_key]:
                prof = routing[profiles_key][expert]
                lines.append(f"{expert}: {len(tracts)} tracts, SVI range {prof['svi_range'][0]:.3f}-{prof['svi_range'][1]:.3f}")
        
        lines.append("")
        lines.append(f"Counterintuitive cases: {len(routing['counterintuitive_cases'])}")
        
        for case in routing['counterintuitive_cases']:
            lines.append(f"  {case['fips']}: {case['reason']}")
        
        lines.append("")
    
    # Summary
    lines.extend([
        "="*75,
        "SUMMARY",
        "="*75,
        ""
    ])
    
    # Checklist
    checks = []
    
    if 'bootstrap' in all_results and all_results['bootstrap']:
        boot = all_results['bootstrap']
        if 'difference' in boot:
            checks.append(("GRANITE > Dasymetric significant", boot['difference']['significant']))
    
    if 'morans_i' in all_results and all_results['morans_i']:
        moran = all_results['morans_i']
        pred_sig = moran['predictions']['p_value'] < 0.05 and moran['predictions']['I'] > 0
        checks.append(("Predictions spatially autocorrelated", pred_sig))
    
    for check, passed in checks:
        status = "PASS" if passed else "FAIL"
        lines.append(f"[{status}] {check}")
    
    lines.extend(["", "="*75])
    
    # Write report
    report_path = os.path.join(output_dir, 'validation_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\nReport saved: {report_path}")
    
    # Also print to console
    print('\n'.join(lines))


# =============================================================================
# MAIN
# =============================================================================

def run_post_training_validation(results_dir='./output/global_validation',
                                  output_dir=None,
                                  skip_baselines=False,
                                  seed=42):
    """
    Run complete post-training validation suite.
    """
    
    print("\n" + "="*70)
    print("GRANITE POST-TRAINING VALIDATION SUITE")
    print("="*70)
    
    if output_dir is None:
        output_dir = os.path.join(results_dir, 'validation')
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    all_results = {}
    
    # Load results
    training_results = load_global_training_results(results_dir)
    tract_results = training_results['tract_results']
    
    # Load spatial data
    data = load_spatial_data()
    
    # Load test tract addresses (fast - no feature computation)
    test_fips = list(tract_results.keys())
    addresses = load_test_tract_data(test_fips, data)
    
    if addresses is None or len(addresses) == 0:
        print("ERROR: No address data loaded")
        return None
    
    print(f"\nLoaded {len(addresses)} addresses across {len(test_fips)} tracts")
    
    # Create GRANITE predictions array (using tract means)
    granite_preds = np.zeros(len(addresses))
    for fips, result in tract_results.items():
        mask = addresses['tract_fips'] == fips
        granite_preds[mask] = result['predicted_mean']
    
    # Compute baselines
    if not skip_baselines:
        baseline_preds = compute_baseline_predictions(addresses, data['tracts'], tract_results)
    else:
        baseline_preds = {}
    
    # Combine all predictions
    all_predictions = {'GRANITE': granite_preds}
    all_predictions.update(baseline_preds)
    
    # 1. Block group validation
    if data['block_groups'] is not None:
        all_results['block_group'] = run_block_group_validation(
            addresses, all_predictions, data, output_dir
        )
    
    # 2. Bootstrap CIs
    if all_results.get('block_group'):
        all_results['bootstrap'] = run_bootstrap_analysis(
            all_results['block_group'], output_dir, seed=seed
        )
    
    # 3. Moran's I
    all_results['morans_i'] = run_morans_i_analysis(
        addresses, granite_preds, output_dir=output_dir, seed=seed
    )
    
    # 4. Expert routing (simplified - no feature analysis without cached features)
    all_results['expert_routing'] = run_expert_routing_simplified(
        tract_results, output_dir
    )
    
    # Generate report
    generate_validation_report(all_results, output_dir)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"VALIDATION COMPLETE - {elapsed/60:.1f} minutes")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='GRANITE Post-Training Validation Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python run_granite.py --global-training
    python post_training_validation.py --results-dir ./output/global_validation
        """
    )
    
    parser.add_argument('--results-dir', type=str, default='./output/global_validation',
                        help='Directory containing global training results')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: results-dir/validation)')
    parser.add_argument('--skip-baselines', action='store_true',
                        help='Skip Dasymetric/Pycnophylactic baseline computation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    run_post_training_validation(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        skip_baselines=args.skip_baselines,
        seed=args.seed
    )