#!/usr/bin/env python3
"""
GRANITE Disaggregation Validation Script

Runs GNN-based spatial disaggregation with baseline comparisons (IDW, Kriging).

Usage:
    python run_disaggregation_validation.py
    
Options:
    --tracts: Comma-separated list of tract FIPS codes (default: curated set)
    --output: Output directory (default: ./output/disaggregation_validation)
    --skip-plots: Skip visualization generation
    --verbose: Verbose logging
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import numpy as np
import pandas as pd
from datetime import datetime

# GRANITE imports
from granite.disaggregation.pipeline import GRANITEPipeline
from granite.models.gnn import set_random_seed
from granite.data.loaders import DataLoader


def get_validation_tracts():
    """
    Return balanced set of validation tracts spanning full SVI spectrum.
    Each tract should have sufficient addresses and valid SVI data.
    """
    return [
        # Very Low SVI (0.0-0.2)
        '47065012000',  # Signal Mountain area
        '47065011205',  # North Hamilton
        
        # Low SVI (0.2-0.4)
        '47065000600',  # Downtown fringe
        '47065010501',  # East Ridge area
        
        # Medium SVI (0.4-0.6)
        '47065012400',  # Mixed area
        '47065002800',  # Central corridor
        
        # High SVI (0.6-0.8)
        '47065010902',  # Southeast
        '47065011442',  # Older suburbs
        
        # Very High SVI (0.8-1.0)
        '47065003000',  # High vulnerability
        '47065001300',  # Urban core
    ]


def run_single_tract_disaggregation(tract_fips: str, config: dict, 
                                     loader: DataLoader, verbose: bool = True):
    """
    Run disaggregation for a single tract with baseline comparisons.
    
    Returns:
        Dict with disaggregation results and baseline comparisons
    """
    from granite.evaluation.disaggregation_baselines import (
        DisaggregationComparison, NaiveUniformBaseline, 
        IDWDisaggregation, OrdinaryKrigingDisaggregation
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"DISAGGREGATING TRACT: {tract_fips}")
        print(f"{'='*60}")
    
    # Create pipeline for this tract
    tract_config = config.copy()
    tract_config['data']['target_fips'] = tract_fips
    
    output_dir = os.path.join(
        config.get('output_dir', './output'),
        f'tract_{tract_fips}'
    )
    
    pipeline = GRANITEPipeline(tract_config, output_dir=output_dir)
    
    # Run GNN disaggregation
    try:
        result = pipeline.run()
        
        if not result.get('success', False):
            return {
                'tract_fips': tract_fips,
                'success': False,
                'error': result.get('error', 'Unknown error')
            }
        
        # Extract key results
        predictions = result['predictions']['mean'].values
        tract_svi = result['tract_info']['RPL_THEMES']
        accessibility_features = result.get('accessibility_features')
        
        # Get address GeoDataFrame
        addresses = loader.get_addresses_for_tract(tract_fips)
        
        # Get tract data for baselines
        state_fips = tract_fips[:2]
        county_fips = tract_fips[2:5]
        tracts = loader.load_census_tracts(state_fips, county_fips)
        county_name = loader._get_county_name(state_fips, county_fips)
        svi_data = loader.load_svi_data(state_fips, county_name)
        tract_gdf = tracts.merge(svi_data, on='FIPS', how='inner')
        
        # Run baseline comparison
        comparison = DisaggregationComparison(verbose=verbose)
        comparison.add_baseline(NaiveUniformBaseline())
        comparison.add_baseline(IDWDisaggregation(power=2.0))
        comparison.add_baseline(IDWDisaggregation(power=3.0))
        comparison.add_baseline(OrdinaryKrigingDisaggregation())
        
        baseline_results = comparison.run_comparison(
            tract_gdf=tract_gdf,
            address_gdf=addresses,
            gnn_predictions=predictions,
            tract_fips=tract_fips,
            tract_svi=tract_svi,
            accessibility_features=accessibility_features
        )
        
        return {
            'tract_fips': tract_fips,
            'success': True,
            'tract_svi': tract_svi,
            'gnn_results': {
                'mean': np.mean(predictions),
                'std': np.std(predictions),
                'range': np.ptp(predictions),
                'constraint_error': abs(np.mean(predictions) - tract_svi) / tract_svi * 100
            },
            'baseline_results': baseline_results,
            'n_addresses': len(addresses),
            'pipeline_result': result
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'tract_fips': tract_fips,
            'success': False,
            'error': str(e)
        }


def run_disaggregation_validation(tracts: list = None, output_dir: str = None,
                                   skip_plots: bool = False, verbose: bool = True,
                                   seed: int = 42):
    """
    Run full disaggregation validation across multiple tracts.
    
    Args:
        tracts: List of tract FIPS codes (default: curated validation set)
        output_dir: Output directory
        skip_plots: Skip visualization generation
        verbose: Verbose logging
        seed: Random seed
        
    Returns:
        Dict with aggregate results and per-tract details
    """
    set_random_seed(seed)
    
    if tracts is None:
        tracts = get_validation_tracts()
    
    if output_dir is None:
        output_dir = './output/disaggregation_validation'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("GRANITE DISAGGREGATION VALIDATION")
    print(f"{'='*70}")
    print(f"Tracts: {len(tracts)}")
    print(f"Output: {output_dir}")
    print(f"Seed: {seed}")
    print(f"{'='*70}\n")
    
    # Configuration
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
            'verbose': verbose,
            'enable_caching': True,
            'random_seed': seed
        },
        'output_dir': output_dir
    }
    
    # Initialize loader
    loader = DataLoader(config=config)
    
    # Run disaggregation for each tract
    all_results = []
    successful = 0
    failed = 0
    
    start_time = time.time()
    
    for i, tract_fips in enumerate(tracts, 1):
        print(f"\n[{i}/{len(tracts)}] Processing tract {tract_fips}...")
        
        result = run_single_tract_disaggregation(
            tract_fips=tract_fips,
            config=config,
            loader=loader,
            verbose=verbose
        )
        
        all_results.append(result)
        
        if result['success']:
            successful += 1
            gnn = result['gnn_results']
            print(f"  -> SVI: {result['tract_svi']:.4f}, "
                  f"GNN std: {gnn['std']:.4f}, "
                  f"Constraint err: {gnn['constraint_error']:.2f}%")
        else:
            failed += 1
            print(f"  -> FAILED: {result.get('error', 'Unknown')}")
    
    elapsed = time.time() - start_time
    
    # Aggregate results
    successful_results = [r for r in all_results if r['success']]
    
    if not successful_results:
        print("\nERROR: All tracts failed")
        return {'success': False, 'error': 'All tracts failed'}
    
    # Build summary DataFrame
    summary_rows = []
    for r in successful_results:
        gnn = r['gnn_results']
        baseline = r['baseline_results']['methods']
        
        row = {
            'tract_fips': r['tract_fips'],
            'tract_svi': r['tract_svi'],
            'n_addresses': r['n_addresses'],
            'gnn_mean': gnn['mean'],
            'gnn_std': gnn['std'],
            'gnn_constraint_error': gnn['constraint_error'],
            'naive_std': baseline.get('Naive_Uniform', {}).get('std', 0),
            'idw_std': baseline.get('IDW_p2.0', {}).get('std', 0),
            'kriging_std': baseline.get('Kriging', {}).get('std', 0),
            'gnn_access_corr': baseline.get('GNN', {}).get('accessibility_correlation'),
        }
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save summary
    summary_path = os.path.join(output_dir, 'disaggregation_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    # Print aggregate results
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")
    print(f"Successful tracts: {successful}/{len(tracts)}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/len(tracts):.1f}s per tract)")
    
    print(f"\nConstraint Satisfaction:")
    print(f"  Mean error: {summary_df['gnn_constraint_error'].mean():.2f}%")
    print(f"  Median error: {summary_df['gnn_constraint_error'].median():.2f}%")
    print(f"  Best: {summary_df['gnn_constraint_error'].min():.2f}%")
    print(f"  Worst: {summary_df['gnn_constraint_error'].max():.2f}%")
    
    print(f"\nSpatial Variation (std):")
    print(f"  GNN mean: {summary_df['gnn_std'].mean():.4f}")
    print(f"  IDW mean: {summary_df['idw_std'].mean():.4f}")
    print(f"  Kriging mean: {summary_df['kriging_std'].mean():.4f}")
    print(f"  Naive (uniform): {summary_df['naive_std'].mean():.4f}")
    
    gnn_vs_idw = summary_df['gnn_std'].mean() - summary_df['idw_std'].mean()
    print(f"\n  GNN variation advantage vs IDW: {gnn_vs_idw:+.4f}")
    
    # Accessibility correlation summary
    valid_corr = summary_df['gnn_access_corr'].dropna()
    if len(valid_corr) > 0:
        print(f"\nAccessibility-SVI Correlation:")
        print(f"  Mean: {valid_corr.mean():.3f}")
        print(f"  Negative correlations: {(valid_corr < 0).sum()}/{len(valid_corr)}")
    
    # Generate visualizations
    if not skip_plots:
        print(f"\nGenerating visualizations...")
        try:
            from granite.visualization.disaggregation_plots import (
                DisaggregationVisualizer, create_disaggregation_visualizations
            )
            
            viz = DisaggregationVisualizer()
            
            # Aggregate comparison plot
            agg_plot_path = os.path.join(output_dir, 'aggregate_comparison.png')
            _create_aggregate_comparison_plot(summary_df, agg_plot_path)
            
            print(f"  Saved: {agg_plot_path}")
            
        except Exception as e:
            print(f"  Warning: Visualization failed: {e}")
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}")
    
    return {
        'success': True,
        'summary_df': summary_df,
        'all_results': all_results,
        'elapsed_time': elapsed,
        'successful_count': successful,
        'failed_count': failed
    }


def _create_aggregate_comparison_plot(summary_df: pd.DataFrame, output_path: str):
    """Create aggregate comparison visualization."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Constraint error distribution
    ax1 = axes[0, 0]
    ax1.hist(summary_df['gnn_constraint_error'], bins=10, color='steelblue', 
             alpha=0.7, edgecolor='black')
    ax1.axvline(5, color='green', linestyle='--', label='5% threshold')
    ax1.set_xlabel('Constraint Error (%)')
    ax1.set_ylabel('Count')
    ax1.set_title('GNN Constraint Satisfaction Distribution')
    ax1.legend()
    
    # 2. Variation comparison by tract
    ax2 = axes[0, 1]
    x = range(len(summary_df))
    width = 0.25
    ax2.bar([i - width for i in x], summary_df['gnn_std'], width, 
            label='GNN', color='green', alpha=0.7)
    ax2.bar(x, summary_df['idw_std'], width, 
            label='IDW', color='blue', alpha=0.7)
    ax2.bar([i + width for i in x], summary_df['kriging_std'], width,
            label='Kriging', color='purple', alpha=0.7)
    ax2.set_xlabel('Tract')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_title('Spatial Variation by Method')
    ax2.legend()
    ax2.set_xticks(x)
    ax2.set_xticklabels([f[-4:] for f in summary_df['tract_fips']], rotation=45)
    
    # 3. Variation vs SVI
    ax3 = axes[1, 0]
    ax3.scatter(summary_df['tract_svi'], summary_df['gnn_std'], 
               s=80, color='green', alpha=0.7, label='GNN')
    ax3.scatter(summary_df['tract_svi'], summary_df['idw_std'],
               s=80, color='blue', alpha=0.7, label='IDW')
    ax3.set_xlabel('Tract SVI')
    ax3.set_ylabel('Prediction Std')
    ax3.set_title('Spatial Variation vs Tract SVI')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Accessibility correlation
    ax4 = axes[1, 1]
    valid_corr = summary_df.dropna(subset=['gnn_access_corr'])
    if len(valid_corr) > 0:
        colors = ['green' if c < 0 else 'red' for c in valid_corr['gnn_access_corr']]
        ax4.bar(range(len(valid_corr)), valid_corr['gnn_access_corr'], 
               color=colors, alpha=0.7, edgecolor='black')
        ax4.axhline(0, color='black', linewidth=0.5)
        ax4.axhline(-0.3, color='green', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Tract')
        ax4.set_ylabel('Correlation')
        ax4.set_title('GNN Accessibility-SVI Correlation\n(Green = Expected Direction)')
        ax4.set_xticks(range(len(valid_corr)))
        ax4.set_xticklabels([f[-4:] for f in valid_corr['tract_fips']], rotation=45)
    else:
        ax4.text(0.5, 0.5, 'No accessibility\ndata available',
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='GRANITE Disaggregation Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_disaggregation_validation.py
    python run_disaggregation_validation.py --tracts 47065012000,47065000600
    python run_disaggregation_validation.py --output ./my_results --verbose
        """
    )
    
    parser.add_argument('--tracts', type=str, default=None,
                       help='Comma-separated tract FIPS codes')
    parser.add_argument('--output', type=str, default='./output/disaggregation_validation',
                       help='Output directory')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Parse tract list
    tracts = None
    if args.tracts:
        tracts = [t.strip() for t in args.tracts.split(',')]
    
    # Run validation
    results = run_disaggregation_validation(
        tracts=tracts,
        output_dir=args.output,
        skip_plots=args.skip_plots,
        verbose=args.verbose,
        seed=args.seed
    )
    
    if results['success']:
        print("\nValidation completed successfully.")
        return 0
    else:
        print(f"\nValidation failed: {results.get('error', 'Unknown error')}")
        return 1


if __name__ == '__main__':
    sys.exit(main())