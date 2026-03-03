"""
Global training validation with manually curated diverse tracts.
Uses confirmed tracts with sufficient addresses spanning full SVI spectrum.

UPDATED: Now includes baseline comparison aggregation for dissertation metrics.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from granite.disaggregation.pipeline import GRANITEPipeline
from granite.models.gnn import set_random_seed
from granite.data.loaders import DataLoader


def get_curated_training_tracts():
    """12 non-overlapping training tracts with balanced SVI coverage"""
    return [
        '47065012000', '47065011205', '47065011100', # Very Low
        '47065000600', '47065010413', '47065010501', # Low
        '47065012400', '47065002800',                  # Medium
        '47065010902', '47065011442',                  # High
        '47065003000', '47065001300',                  # Very High
    ]


def get_curated_test_tracts():
    """10 non-overlapping test tracts with balanced SVI coverage"""
    return [
        '47065000700', '47065010411',                  # Very Low
        '47065011900', '47065010502',                  # Low
        '47065010433', '47065000800',                  # Medium
        '47065011206', '47065011444',                  # High
        '47065000400', '47065012300',                  # Very High
    ]


def print_tract_summary(tract_list, label, loader):
    """Print summary statistics for a tract list"""
    
    tracts = loader.load_census_tracts('47', '065')
    svi = loader.load_svi_data('47', 'Hamilton')
    tract_data = tracts.merge(svi, on='FIPS', how='inner')
    
    selected = tract_data[tract_data['FIPS'].isin(tract_list)].copy()
    selected = selected.sort_values('RPL_THEMES')
    
    print(f"\n{'='*80}")
    print(f"{label} ({len(selected)} tracts)")
    print(f"{'='*80}")
    print(f"SVI Range: {selected['RPL_THEMES'].min():.3f} - {selected['RPL_THEMES'].max():.3f}")
    print(f"Mean SVI: {selected['RPL_THEMES'].mean():.3f}")
    print(f"Median SVI: {selected['RPL_THEMES'].median():.3f}")
    
    selected['quintile'] = pd.cut(
        selected['RPL_THEMES'],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    print(f"\nDistribution by SVI Quintile:")
    quintile_counts = selected['quintile'].value_counts().sort_index()
    for quintile, count in quintile_counts.items():
        print(f" {quintile}: {count} tracts")
    
    print(f"\nDetailed Tract List:")
    for _, row in selected.iterrows():
        try:
            addresses = loader.get_addresses_for_tract(row['FIPS'])
            n_addr = len(addresses)
        except:
            n_addr = 0
        
        quintile = row['quintile']
        print(f" {row['FIPS']}: SVI={row['RPL_THEMES']:.3f} ({quintile}), {n_addr:,} addresses")


def aggregate_baseline_results(test_results: dict) -> pd.DataFrame:
    """
    Aggregate baseline comparison results across all holdout tracts.
    
    Args:
        test_results: Dict of {fips: result_dict} from run_mixture_training
        
    Returns:
        DataFrame with per-tract baseline metrics
    """
    rows = []
    
    for fips, result in test_results.items():
        if result.get('mean_error_pct') is None:
            continue
            
        baseline = result.get('baseline_comparison', {})
        methods = baseline.get('methods', {})
        
        row = {
            'fips': fips,
            'actual_svi': result['actual_svi'],
            'gnn_mean': result['predicted_mean'],
            'gnn_error_pct': result['mean_error_pct'],
            'gnn_correlation': result['correlation'],
            'dominant_expert': result.get('dominant_expert', 'N/A'),
        }
        
        # Extract baseline metrics
        for method_name in ['GNN', 'Naive_Uniform', 'IDW_p2.0', 'IDW_p3.0', 'Kriging']:
            method_data = methods.get(method_name, {})
            prefix = method_name.lower().replace('_', '').replace('.', '')
            
            row[f'{prefix}_std'] = method_data.get('std', np.nan)
            row[f'{prefix}_constraint_err'] = method_data.get('constraint_error_pct', np.nan)
            row[f'{prefix}_access_corr'] = method_data.get('accessibility_correlation', np.nan)
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def print_baseline_summary(baseline_df: pd.DataFrame):
    """Print baseline comparison summary."""
    
    print(f"\n{'='*80}")
    print("BASELINE COMPARISON SUMMARY (GNN vs Traditional Methods)")
    print(f"{'='*80}")
    
    # Spatial variation comparison
    print(f"\n--- Spatial Variation (std) ---")
    print(f"{'Method':<20} {'Mean Std':<12} {'Median Std':<12}")
    print("-" * 44)
    
    for method, col in [('GNN', 'gnn_std'), ('Naive Uniform', 'naiveuniform_std'), 
                        ('IDW (p=2)', 'idwp20_std'), ('Kriging', 'kriging_std')]:
        if col in baseline_df.columns:
            mean_val = baseline_df[col].mean()
            median_val = baseline_df[col].median()
            print(f"{method:<20} {mean_val:<12.4f} {median_val:<12.4f}")
    
    # GNN advantage
    if 'gnn_std' in baseline_df.columns and 'idwp20_std' in baseline_df.columns:
        gnn_advantage = baseline_df['gnn_std'].mean() - baseline_df['idwp20_std'].mean()
        print(f"\nGNN variation advantage over IDW: {gnn_advantage:+.4f}")
        
        if gnn_advantage > 0:
            print(" -> GNN produces MORE spatial variation (better disaggregation)")
        else:
            print(" -> IDW produces more variation (investigate)")
    
    # Accessibility correlation
    print(f"\n--- Accessibility-SVI Correlation ---")
    if 'gnn_access_corr' in baseline_df.columns:
        gnn_corr = baseline_df['gnn_access_corr'].dropna()
        if len(gnn_corr) > 0:
            print(f"GNN mean correlation: {gnn_corr.mean():.3f}")
            print(f"Negative correlations: {(gnn_corr < 0).sum()}/{len(gnn_corr)}")
            print(f"Strong negative (r<-0.3): {(gnn_corr < -0.3).sum()}/{len(gnn_corr)}")
    
    # Per-tract comparison table
    print(f"\n--- Per-Tract Results ---")
    cols_to_show = ['fips', 'actual_svi', 'gnn_error_pct', 'gnn_std', 'idwp20_std', 'gnn_access_corr']
    available_cols = [c for c in cols_to_show if c in baseline_df.columns]
    
    if available_cols:
        display_df = baseline_df[available_cols].copy()
        display_df = display_df.sort_values('actual_svi')
        
        print(f"\n{'FIPS':<12} {'SVI':<8} {'Err%':<8} {'GNN_std':<10} {'IDW_std':<10} {'Access_r':<10}")
        print("-" * 68)
        
        for _, row in display_df.iterrows():
            fips = str(row['fips'])[-5:]
            svi = row['actual_svi']
            err = row.get('gnn_error_pct', np.nan)
            gnn_std = row.get('gnn_std', np.nan)
            idw_std = row.get('idwp20_std', np.nan)
            acc_corr = row.get('gnn_access_corr', np.nan)
            
            print(f"{fips:<12} {svi:<8.3f} {err:<8.1f} {gnn_std:<10.4f} {idw_std:<10.4f} {acc_corr:<10.3f}")


def run_global_training_validation(seed=42):
    """
    Global training validation with manually curated tracts.
    Now includes baseline comparison analysis.
    """
    set_random_seed(seed)
    
    loader = DataLoader()
    
    training_tracts = get_curated_training_tracts()
    test_tracts = get_curated_test_tracts()
    
    print(f"\n{'='*80}")
    print("GRANITE GLOBAL TRAINING VALIDATION")
    print("Using Manually Curated Diverse Tracts")
    print("With Baseline Comparisons (IDW, Kriging, Naive)")
    print(f"{'='*80}")
    
    print_tract_summary(training_tracts, "TRAINING SET", loader)
    print_tract_summary(test_tracts, "TEST SET", loader)
    
    # Verify no overlap
    overlap = set(training_tracts) & set(test_tracts)
    if overlap:
        print(f"\nWARNING: Overlap detected: {overlap}")
        return None
    else:
        print(f"\nNo overlap between training and test sets")
    
    # Configure pipeline
    config = {
        'data': {
            'state_fips': '47',
            'county_fips': '065'
        },
        'model': {
            'epochs': 150,
            'hidden_dim': 64,
            'dropout': 0.3
        },
        'training': {
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'enforce_constraints': True,
            'constraint_weight': 1.0,
            'use_multitask': True,
            'use_mixture': True,
            'epochs': 150,
            'gate_epochs': 100,
            'finetune_epochs': 50
        },
        'processing': {
            'verbose': True,
            'enable_caching': True,
            'random_seed': seed
        }
    }
    
    pipeline = GRANITEPipeline(config, output_dir='./output/global_validation')
    
    print(f"\n{'='*80}")
    print("TRAINING GLOBAL MODEL")
    print(f"{'='*80}\n")
    
    results = pipeline.run_mixture_training(
        training_fips_list=training_tracts,
        test_fips_list=test_tracts
    )
    
    if results['success']:
        print(f"\n{'='*80}")
        print("GLOBAL TRAINING VALIDATION RESULTS")
        print(f"{'='*80}")
        
        test_results = results['test_results']
        
        valid_results = {
            fips: r for fips, r in test_results.items() 
            if r['mean_error_pct'] is not None
        }
        
        if len(valid_results) == 0:
            print("\nERROR: No valid test results")
            return results
        
        # Standard metrics
        errors = [r['mean_error_pct'] for r in valid_results.values()]
        correlations = [r['correlation'] for r in valid_results.values()]
        actual_svis = [r['actual_svi'] for r in valid_results.values()]
        predicted_svis = [r['predicted_mean'] for r in valid_results.values()]
        
        print(f"\nTest Set Performance ({len(valid_results)}/{len(test_tracts)} tracts):")
        print(f"\n--- Constraint Satisfaction ---")
        print(f" Mean Error: {np.mean(errors):.2f}% +/- {np.std(errors):.2f}%")
        print(f" Median Error: {np.median(errors):.2f}%")
        print(f" Min Error: {np.min(errors):.2f}%")
        print(f" Max Error: {np.max(errors):.2f}%")
        print(f" Tracts < 10% error: {sum(1 for e in errors if e < 10)}/{len(errors)}")
        print(f" Tracts < 20% error: {sum(1 for e in errors if e < 20)}/{len(errors)}")
        
        print(f"\n--- Cross-Tract Generalization ---")
        print(f" R^2 (actual vs predicted): {np.corrcoef(actual_svis, predicted_svis)[0,1]**2:.3f}")
        
        # NEW: Aggregate and print baseline comparisons
        baseline_df = aggregate_baseline_results(valid_results)
        
        if len(baseline_df) > 0:
            print_baseline_summary(baseline_df)
            
            # Save detailed results
            baseline_path = './output/global_validation/baseline_comparison.csv'
            baseline_df.to_csv(baseline_path, index=False)
            print(f"\nBaseline comparison saved to: {baseline_path}")
        
        # Save standard results
        results_df = pd.DataFrame([
            {
                'fips': fips,
                'actual_svi': r['actual_svi'],
                'predicted_svi': r['predicted_mean'],
                'error_pct': r['mean_error_pct'],
                'correlation': r['correlation'],
                'dominant_expert': r.get('dominant_expert', 'N/A')
            }
            for fips, r in valid_results.items()
        ])
        
        output_file = './output/global_validation/validation_results.csv'
        results_df.to_csv(output_file, index=False)
        print(f"Validation results saved to: {output_file}")
        
        # Generate visualizations
        try:
            from granite.visualization.disaggregation_plots import DisaggregationVisualizer
            viz = DisaggregationVisualizer()
            
            # Create aggregate comparison plot
            _create_aggregate_baseline_plot(baseline_df, 
                './output/global_validation/aggregate_baseline_comparison.png')
            print(f"Aggregate visualization saved")
            
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
        
        # Final assessment
        print(f"\n{'='*80}")
        print("DISSERTATION METRICS SUMMARY")
        print(f"{'='*80}")
        
        mean_error = np.mean(errors)
        gnn_std_mean = baseline_df['gnn_std'].mean() if 'gnn_std' in baseline_df.columns else 0
        idw_std_mean = baseline_df['idwp20_std'].mean() if 'idwp20_std' in baseline_df.columns else 0
        
        print(f"\n1. Constraint Satisfaction: {mean_error:.1f}% mean error")
        if mean_error < 10:
            print("   -> EXCELLENT (<10%)")
        elif mean_error < 20:
            print("   -> GOOD (<20%)")
        else:
            print("   -> NEEDS IMPROVEMENT")
        
        print(f"\n2. Disaggregation Quality:")
        print(f"   GNN spatial variation: {gnn_std_mean:.4f}")
        print(f"   IDW spatial variation: {idw_std_mean:.4f}")
        print(f"   GNN advantage: {gnn_std_mean - idw_std_mean:+.4f}")
        
        if gnn_std_mean > idw_std_mean:
            print("   -> GNN produces finer-grained disaggregation than IDW")
        
        if 'gnn_access_corr' in baseline_df.columns:
            mean_corr = baseline_df['gnn_access_corr'].mean()
            print(f"\n3. Accessibility-Vulnerability Link:")
            print(f"   Mean correlation: {mean_corr:.3f}")
            if mean_corr < -0.2:
                print("   -> Strong equity pattern (high access = low vulnerability)")
    
    return results


def _create_aggregate_baseline_plot(baseline_df: pd.DataFrame, output_path: str):
    """Create aggregate baseline comparison visualization."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Spatial variation by method
    ax1 = axes[0, 0]
    methods = ['gnn_std', 'naiveuniform_std', 'idwp20_std', 'kriging_std']
    labels = ['GNN', 'Naive', 'IDW', 'Kriging']
    colors = ['green', 'gray', 'blue', 'purple']
    
    means = [baseline_df[m].mean() if m in baseline_df.columns else 0 for m in methods]
    bars = ax1.bar(labels, means, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Mean Spatial Variation (std)')
    ax1.set_title('Disaggregation Quality by Method')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. GNN vs IDW scatter
    ax2 = axes[0, 1]
    if 'gnn_std' in baseline_df.columns and 'idwp20_std' in baseline_df.columns:
        ax2.scatter(baseline_df['idwp20_std'], baseline_df['gnn_std'], 
                   s=80, alpha=0.7, c='green', edgecolors='black')
        
        # Diagonal line
        max_val = max(baseline_df['gnn_std'].max(), baseline_df['idwp20_std'].max())
        ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='1:1 line')
        
        ax2.set_xlabel('IDW Variation (std)')
        ax2.set_ylabel('GNN Variation (std)')
        ax2.set_title('GNN vs IDW Variation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Variation vs SVI
    ax3 = axes[1, 0]
    if 'actual_svi' in baseline_df.columns and 'gnn_std' in baseline_df.columns:
        ax3.scatter(baseline_df['actual_svi'], baseline_df['gnn_std'],
                   s=80, alpha=0.7, c='green', label='GNN', edgecolors='black')
        if 'idwp20_std' in baseline_df.columns:
            ax3.scatter(baseline_df['actual_svi'], baseline_df['idwp20_std'],
                       s=80, alpha=0.7, c='blue', label='IDW', edgecolors='black')
        
        ax3.set_xlabel('Tract SVI')
        ax3.set_ylabel('Spatial Variation (std)')
        ax3.set_title('Disaggregation vs Tract Vulnerability')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    n_tracts = len(baseline_df)
    gnn_mean = baseline_df['gnn_std'].mean() if 'gnn_std' in baseline_df.columns else 0
    idw_mean = baseline_df['idwp20_std'].mean() if 'idwp20_std' in baseline_df.columns else 0
    
    summary = f"""Baseline Comparison Summary
{'='*35}

Holdout Tracts: {n_tracts}

Mean Spatial Variation:
  GNN:     {gnn_mean:.4f}
  IDW:     {idw_mean:.4f}
  Diff:    {gnn_mean - idw_mean:+.4f}

Interpretation:
  {'GNN produces more variation' if gnn_mean > idw_mean else 'IDW produces more variation'}
  {'(Better disaggregation)' if gnn_mean > idw_mean else '(Investigate)'}
"""
    
    ax4.text(0.1, 0.9, summary, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    print("="*80)
    print("GRANITE GLOBAL TRAINING VALIDATION")
    print("With Baseline Comparisons (IDW, Kriging, Naive)")
    print("="*80)
    print("\nThis script will:")
    print("1. Train ONE global MoE model on 12 diverse tracts")
    print("2. Test on 10 separate holdout tracts")
    print("3. Compare GNN disaggregation vs IDW/Kriging/Naive baselines")
    print("4. Generate performance analysis")
    print("\nEstimated time:")
    print(" - First run: ~60-90 minutes (computing accessibility features)")
    print(" - Cached runs: ~10-15 minutes (training + validation only)")
    print("\nPress Ctrl+C to cancel, or wait 5 seconds to continue...")
    
    import time
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(0)
    
    results = run_global_training_validation(seed=42)
    
    if results is not None and results['success']:
        print("\n" + "="*80)
        print("VALIDATION COMPLETE")
        print("="*80)
        print(f"\nSuccessfully validated on {len(results['test_results'])} tracts")
        print("Review the output files in ./output/global_validation/ for detailed results.")