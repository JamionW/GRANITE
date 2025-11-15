"""
Comprehensive holdout validation across multiple tracts.
Tests GNN learning quality without constraint enforcement.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from granite.disaggregation.pipeline import GRANITEPipeline
from granite.models.gnn import set_random_seed

def run_comprehensive_validation(test_tracts, neighbor_count=3, seed=42):
    """
    Run holdout validation across multiple tracts.
    
    Args:
        test_tracts: List of FIPS codes to test
        neighbor_count: Number of neighboring tracts for training
        seed: Random seed for reproducibility
    """
    set_random_seed(seed)
    
    results = []
    
    for target_fips in test_tracts:
        print(f"\n{'='*80}")
        print(f"Holdout Validation: {target_fips}")
        print(f"{'='*80}")
        
        config = {
            'data': {
                'target_fips': target_fips,
                'state_fips': target_fips[:2],
                'county_fips': target_fips[2:5],
                'neighbor_tracts': neighbor_count
            },
            'model': {
                'epochs': 150,
                'hidden_dim': 64,
                'dropout': 0.3
            },
            'training': {
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'enforce_constraints': False  # CRITICAL
            },
            'validation': {
                'holdout_mode': True
            },
            'processing': {
                'verbose': True,
                'enable_caching': True,
                'random_seed': seed
            }
        }
        
        pipeline = GRANITEPipeline(config, output_dir=f'./output/holdout_{target_fips}')
        
        try:
            result = pipeline.run()
            
            if result['success']:
                results.append({
                    'fips': target_fips,
                    'actual_svi': result['actual_svi'],
                    'predicted_mean': result['predicted_mean'],
                    'mean_error_pct': result['mean_error_pct'],
                    'natural_convergence': result['natural_convergence'],
                    'correlation': result['accessibility_svi_correlation'],
                    'std': result['predicted_std'],
                    'n_training_tracts': len(result['training_fips']),
                    'n_holdout_addresses': result['n_holdout_addresses']
                })
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            continue
    
    # Summary
    if not results:
        print("\nNo successful validations!")
        return None
    
    results_df = pd.DataFrame(results)
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"\nTested {len(results)} tracts")
    print(f"\nMean Error: {results_df['mean_error_pct'].mean():.2f}% ± {results_df['mean_error_pct'].std():.2f}%")
    print(f"Median Error: {results_df['mean_error_pct'].median():.2f}%")
    print(f"Mean Correlation: {results_df['correlation'].mean():.4f} ± {results_df['correlation'].std():.4f}")
    print(f"\nNatural Convergence: {results_df['natural_convergence'].sum()} / {len(results)} tracts")
    print(f"Negative Correlation: {(results_df['correlation'] < 0).sum()} / {len(results)} tracts")
    
    results_df.to_csv('./output/holdout_validation_summary.csv', index=False)
    print(f"\nResults saved: ./output/holdout_validation_summary.csv")
    
    return results_df


if __name__ == '__main__':
    # Test diverse tracts
    test_tracts = [
        '47065000600',  # Current problematic tract
        '47065012300',  # High SVI tract
        '47065010100',
        '47065010200',
    ]
    
    results = run_comprehensive_validation(test_tracts, neighbor_count=3, seed=42)