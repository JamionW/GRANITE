"""
Global training validation with manually curated diverse tracts.
Uses confirmed tracts with sufficient addresses spanning full SVI spectrum.
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
        '47065012000', '47065011205', '47065011100',  # Very Low
        '47065000600', '47065010413', '47065010501',  # Low
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
    
    # Create quintile labels
    selected['quintile'] = pd.cut(
        selected['RPL_THEMES'],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    print(f"\nDistribution by SVI Quintile:")
    quintile_counts = selected['quintile'].value_counts().sort_index()
    for quintile, count in quintile_counts.items():
        print(f"  {quintile}: {count} tracts")
    
    print(f"\nDetailed Tract List:")
    for _, row in selected.iterrows():
        # Get address count
        try:
            addresses = loader.get_addresses_for_tract(row['FIPS'])
            n_addr = len(addresses)
        except:
            n_addr = 0
        
        quintile = row['quintile']
        print(f"  {row['FIPS']}: SVI={row['RPL_THEMES']:.3f} ({quintile}), {n_addr:,} addresses")

def run_global_training_validation(seed=42):
    """
    Global training validation with manually curated tracts.
    """
    set_random_seed(seed)
    
    loader = DataLoader()
    
    # Get curated tract lists
    training_tracts = get_curated_training_tracts()
    test_tracts = get_curated_test_tracts()
    
    print(f"\n{'='*80}")
    print("GRANITE GLOBAL TRAINING VALIDATION")
    print("Using Manually Curated Diverse Tracts")
    print(f"{'='*80}")
    
    # Print summaries
    print_tract_summary(training_tracts, "TRAINING SET", loader)
    print_tract_summary(test_tracts, "TEST SET", loader)
    
    # Verify no overlap
    overlap = set(training_tracts) & set(test_tracts)
    if overlap:
        print(f"\n⚠️  WARNING: Overlap detected: {overlap}")
        return None
    else:
        print(f"\n✓ No overlap between training and test sets")
    
    # Configure pipeline for global training
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
            'use_multitask': True
        },
        'processing': {
            'verbose': True,
            'enable_caching': True,
            'random_seed': seed
        }
    }
    
    pipeline = GRANITEPipeline(config, output_dir='./output/global_validation')
    
    # Run global training
    print(f"\n{'='*80}")
    print("TRAINING GLOBAL MODEL")
    print(f"{'='*80}\n")
    
    results = pipeline.run_global_training(
        training_fips_list=training_tracts,
        test_fips_list=test_tracts
    )
    
    # Analyze and summarize results
    if results['success']:
        print(f"\n{'='*80}")
        print("GLOBAL TRAINING VALIDATION RESULTS")
        print(f"{'='*80}")
        
        test_results = results['test_results']
        
        # Filter out any failed tracts
        valid_results = {
            fips: r for fips, r in test_results.items() 
            if r['mean_error_pct'] is not None
        }
        
        if len(valid_results) == 0:
            print("\n✗ ERROR: No valid test results")
            return results
        
        # Calculate statistics
        errors = [r['mean_error_pct'] for r in valid_results.values()]
        correlations = [r['correlation'] for r in valid_results.values()]
        actual_svis = [r['actual_svi'] for r in valid_results.values()]
        predicted_svis = [r['predicted_mean'] for r in valid_results.values()]
        
        print(f"\nTest Set Performance ({len(valid_results)}/{len(test_tracts)} tracts):")
        print(f"\n--- Error Statistics ---")
        print(f"  Mean Error: {np.mean(errors):.2f}% ± {np.std(errors):.2f}%")
        print(f"  Median Error: {np.median(errors):.2f}%")
        print(f"  Min Error: {np.min(errors):.2f}%")
        print(f"  Max Error: {np.max(errors):.2f}%")
        print(f"  Tracts < 20% error: {sum(1 for e in errors if e < 20)}/{len(errors)}")
        print(f"  Tracts < 30% error: {sum(1 for e in errors if e < 30)}/{len(errors)}")
        
        print(f"\n--- Correlation Statistics ---")
        print(f"  Mean Correlation: {np.mean(correlations):.3f} ± {np.std(correlations):.3f}")
        print(f"  Median Correlation: {np.median(correlations):.3f}")
        print(f"  Positive Correlations: {sum(1 for c in correlations if c > 0)}/{len(correlations)}")
        print(f"  Strong Correlations (|r|>0.3): {sum(1 for c in correlations if abs(c) > 0.3)}/{len(correlations)}")
        
        print(f"\n--- SVI Prediction Analysis ---")
        print(f"  Actual SVI Range: {min(actual_svis):.3f} - {max(actual_svis):.3f}")
        print(f"  Predicted SVI Range: {min(predicted_svis):.3f} - {max(predicted_svis):.3f}")
        print(f"  R² (overall): {np.corrcoef(actual_svis, predicted_svis)[0,1]**2:.3f}")
        
        # Error by SVI category
        results_df = pd.DataFrame([
            {
                'fips': fips,
                'actual_svi': r['actual_svi'],
                'predicted_svi': r['predicted_mean'],
                'error_pct': r['mean_error_pct'],
                'correlation': r['correlation']
            }
            for fips, r in valid_results.items()
        ])
        
        results_df['svi_category'] = pd.cut(
            results_df['actual_svi'],
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['Low (<0.3)', 'Medium (0.3-0.5)', 'High (0.5-0.7)', 'Very High (>0.7)']
        )
        
        print(f"\n--- Error by SVI Category ---")
        error_by_svi = results_df.groupby('svi_category')['error_pct'].agg(['count', 'mean', 'std', 'min', 'max'])
        print(error_by_svi.to_string())
        
        # Detailed per-tract results
        print(f"\n{'='*80}")
        print("DETAILED PER-TRACT RESULTS")
        print(f"{'='*80}")
        
        results_df_sorted = results_df.sort_values('actual_svi')
        print(f"\n{'FIPS':<15} {'Actual SVI':<12} {'Pred SVI':<12} {'Error %':<10} {'Corr':<8} {'Category'}")
        print("-" * 80)
        
        for _, row in results_df_sorted.iterrows():
            print(f"{row['fips']:<15} {row['actual_svi']:<12.4f} {row['predicted_svi']:<12.4f} "
                  f"{row['error_pct']:<10.1f} {row['correlation']:<8.3f} {row['svi_category']}")
        
        # Save results
        output_file = './output/global_validation_results.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n{'='*80}")
        print(f"Results saved to: {output_file}")
        print(f"{'='*80}")
        
        # Compare to baseline (your previous per-tract results)
        print(f"\n{'='*80}")
        print("COMPARISON TO PER-TRACT BASELINE")
        print(f"{'='*80}")
        print(f"Previous per-tract training (from your validation):")
        print(f"  Mean Error: 41.7%")
        print(f"  Median Error: 55.4%")
        print(f"  Range: 13.8% - 62.1%")
        print(f"\nGlobal training (current results):")
        print(f"  Mean Error: {np.mean(errors):.1f}%")
        print(f"  Median Error: {np.median(errors):.1f}%")
        print(f"  Range: {np.min(errors):.1f}% - {np.max(errors):.1f}%")
        
        improvement = ((41.7 - np.mean(errors)) / 41.7) * 100
        print(f"\n{'🎉' if improvement > 0 else '⚠️'} Mean Error Improvement: {improvement:+.1f}%")
        
        if np.mean(errors) < 30:
            print("\n✓✓✓ PUBLICATION READY: Mean error < 30%")
        elif np.mean(errors) < 35:
            print("\n✓✓ VERY GOOD: Mean error < 35%")
        elif np.mean(errors) < 40:
            print("\n✓ GOOD: Mean error < 40%")
        else:
            print("\n⚠️ NEEDS IMPROVEMENT: Consider adding more training tracts")
    
    return results

if __name__ == '__main__':
    print("="*80)
    print("GRANITE GLOBAL TRAINING VALIDATION")
    print("="*80)
    print("\nThis script will:")
    print("1. Train ONE global model on 15 diverse tracts (SVI 0.014-0.867)")
    print("2. Test on 10 separate holdout tracts (SVI 0.019-0.873)")
    print("3. Compare results to per-tract baseline")
    print("4. Generate comprehensive performance analysis")
    print("\nEstimated time:")
    print("  - First run: ~60-90 minutes (computing accessibility features)")
    print("  - Cached runs: ~10-15 minutes (training + validation only)")
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