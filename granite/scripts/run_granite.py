"""
GRANITE Command-Line Interface

Unified entry point for all GRANITE workflows:
- Single-tract analysis
- Multi-tract analysis  
- Global MoE training with holdout validation
"""
import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='GRANITE: constraint-preserving GNN for spatial disaggregation of CDC SVI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflows:
  Single-tract:    granite --fips 47065010100
  Multi-tract:     granite --fips 47065010100 --neighbor-tracts 3
  Global training: granite --global-training

Examples:
  granite --fips 47065010100 --verbose
  granite --fips 47065010100 --epochs 200 --neighbor-tracts 5
  granite --global-training --verbose
  granite --global-training --epochs 150 --output ./results
        """
    )
    
    # Workflow selection (mutually exclusive)
    workflow = parser.add_mutually_exclusive_group(required=True)
    workflow.add_argument(
        '--fips', type=str,
        help='Target census tract FIPS code for single/multi-tract analysis'
    )
    workflow.add_argument(
        '--global-training', action='store_true',
        help='Run global MoE training with curated train/test split'
    )

    # Feature recovery mode (requires --fips; incompatible with --global-training)
    parser.add_argument(
        '--recover-feature', type=str, default=None, metavar='FEATURE_NAME',
        help=(
            'Hold out FEATURE_NAME from the input matrix and train with its '
            'per-tract mean as the constraint. Requires --fips. '
            'Incompatible with --global-training.'
        )
    )
    
    # Configuration
    parser.add_argument(
        '--config', type=str, default='./config.yaml',
        help='Path to configuration file (default: ./config.yaml)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output directory (default: ./output)'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Training epochs (overrides config)'
    )
    parser.add_argument(
        '--neighbor-tracts', type=int, default=None,
        help='Number of neighboring tracts for multi-tract mode'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    
    # Flags
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--no-cache', action='store_true',
        help='Disable accessibility feature caching'
    )
    parser.add_argument(
        '--cache-dir', type=str, default=None,
        help='Override cache directory'
    )
    parser.add_argument(
        '--no-constraints', action='store_true',
        help='Train without tract mean constraint enforcement'
    )
    parser.add_argument(
        '--skip-baselines', action='store_true',
        help='Skip baseline comparisons (Dasymetric, Pycnophylactic)'
    )
    parser.add_argument(
        '--skip-importance', action='store_true',
        help='Skip feature importance analysis'
    )
    parser.add_argument(
        '--architecture', choices=['gcn_gat', 'sage'], default='gcn_gat',
        help='GNN architecture: gcn_gat (default) or sage'
    )
    parser.add_argument(
        '--target', choices=['svi', 'property_value'], default=None,
        help='Disaggregation target: svi (default) or property_value'
    )
    parser.add_argument(
        '--prune-features', type=str, default=None, metavar='REPORT_PATH',
        help='Path to importance_report.txt; drops features with negative importance before training'
    )
    parser.add_argument(
        '--diagnostics', action='store_true',
        help='Generate diagnostic plots (stage validation, accessibility validation)'
    )

    return parser.parse_args()


def load_config(args):
    """Load configuration from YAML with CLI overrides."""
    import yaml
    
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f) or {}
    
    # Ensure nested dicts exist
    config.setdefault('data', {})
    config.setdefault('model', {})
    config.setdefault('training', {})
    config.setdefault('processing', {})
    config.setdefault('validation', {})
    config.setdefault('mixture', {})
    
    # Apply CLI overrides for single-tract mode
    if args.fips:
        config['data']['target_fips'] = args.fips
        config['data']['state_fips'] = args.fips[:2]
        config['data']['county_fips'] = args.fips[2:5]
    
    if args.neighbor_tracts is not None:
        config['data']['neighbor_tracts'] = args.neighbor_tracts
    
    if args.epochs is not None:
        config['model']['epochs'] = args.epochs
        config['training']['epochs'] = args.epochs
    
    config['processing']['verbose'] = args.verbose
    config['processing']['enable_caching'] = not args.no_cache
    config['processing']['random_seed'] = args.seed
    config['processing']['skip_importance'] = args.skip_importance
    
    if args.cache_dir:
        config['processing']['cache_dir'] = args.cache_dir
    
    config['training']['enforce_constraints'] = not args.no_constraints
    config['validation']['compare_baselines'] = not args.skip_baselines
    config['model']['architecture'] = args.architecture
    if args.target:
        config['data']['target'] = args.target
    if args.prune_features:
        config['processing']['prune_features_path'] = args.prune_features
    config['diagnostics'] = args.diagnostics

    return config


def get_curated_training_tracts():
    """13 non-overlapping training tracts with balanced SVI coverage."""
    return [
        '47065012000', '47065011205', '47065011100',  # Very Low SVI
        '47065000600', '47065010413', '47065010501',  # Low SVI
        '47065012400', '47065002800',                  # Medium SVI
        '47065010902', '47065011442',                  # High SVI
        '47065003000', '47065001300', '47065002300'    # Very High SVI
    ]


def get_curated_test_tracts():
    """10 non-overlapping test tracts with balanced SVI coverage."""
    return [
        '47065000700', '47065010411',                  # Very Low SVI
        '47065011900', '47065010502',                  # Low SVI
        '47065010433', '47065000800',                  # Medium SVI
        '47065011206', '47065011444',                  # High SVI
        '47065000400', '47065012300',                  # Very High SVI
    ]


def run_single_tract(args, config):
    """Run single or multi-tract analysis."""
    from granite.models.gnn import set_random_seed
    from granite.disaggregation.pipeline import GRANITEPipeline
    
    set_random_seed(args.seed)
    
    output_dir = args.output or config.get('output', {}).get('directory', './output')
    os.makedirs(output_dir, exist_ok=True)
    
    target = config['data'].get('target', 'svi')
    print(f"\n{'='*60}")
    print("GRANITE: Single/Multi-Tract Analysis")
    print(f"{'='*60}")
    print(f"Target FIPS: {args.fips}")
    print(f"Disaggregation target: {target}")
    if config['data'].get('neighbor_tracts', 0) > 0:
        print(f"Mode: Multi-tract ({config['data']['neighbor_tracts']} neighbors)")
    else:
        print(f"Mode: Single-tract")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    pipeline = GRANITEPipeline(config, output_dir=output_dir)
    results = pipeline.run()
    
    if results.get('success', False):
        print(f"\n{'='*60}")
        print("Analysis completed successfully")
        print(f"{'='*60}")
        
        summary = results.get('summary', {})
        print(f"Addresses processed: {summary.get('addresses_processed', 'N/A')}")
        print(f"Accessibility features: {summary.get('accessibility_features', 'N/A')}")
        print(f"Spatial variation: {summary.get('spatial_variation', 0):.4f}")
        print(f"Constraint error: {summary.get('constraint_error', 0):.2f}%")
        
        pipeline.save_results(results)
        print(f"\nResults saved to: {output_dir}")
        return 0
    else:
        print(f"\nAnalysis failed: {results.get('error', 'Unknown error')}")
        return 1


def run_global_training(args, config):
    """Run global MoE training with holdout validation."""
    import numpy as np
    import pandas as pd
    from granite.models.gnn import set_random_seed
    from granite.disaggregation.pipeline import GRANITEPipeline
    from granite.data.loaders import DataLoader
    
    set_random_seed(args.seed)
    
    output_dir = args.output or './output/global_validation'
    os.makedirs(output_dir, exist_ok=True)
    
    training_tracts = get_curated_training_tracts()
    test_tracts = get_curated_test_tracts()
    
    print(f"\n{'='*70}")
    print("GRANITE: Global MoE Training with Holdout Validation")
    print(f"{'='*70}")
    print(f"Training tracts: {len(training_tracts)}")
    print(f"Test tracts: {len(test_tracts)}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")
    
    # Load data for summaries
    loader = DataLoader()
    
    try:
        tracts = loader.load_census_tracts('47', '065')
        svi = loader.load_svi_data('47', 'Hamilton')
        tract_data = tracts.merge(svi, on='FIPS', how='inner')
    except Exception as e:
        print(f"Failed to load tract data: {e}")
        return 1
    
    # Training set summary
    print(f"\nTraining Set:")
    train_data = tract_data[tract_data['FIPS'].isin(training_tracts)]
    print(f"  SVI range: {train_data['RPL_THEMES'].min():.3f} - {train_data['RPL_THEMES'].max():.3f}")
    print(f"  Mean SVI: {train_data['RPL_THEMES'].mean():.3f}")
    
    # Test set summary
    print(f"\nTest Set:")
    test_data = tract_data[tract_data['FIPS'].isin(test_tracts)]
    print(f"  SVI range: {test_data['RPL_THEMES'].min():.3f} - {test_data['RPL_THEMES'].max():.3f}")
    print(f"  Mean SVI: {test_data['RPL_THEMES'].mean():.3f}")
    
    # Check for overlap
    overlap = set(training_tracts) & set(test_tracts)
    if overlap:
        print(f"\nError: Overlap detected between train/test: {overlap}")
        return 1
    print(f"\nNo overlap between training and test sets")
    
    # Configure for global training
    config['validation']['mode'] = 'global'
    config['validation']['training_fips'] = training_tracts
    config['validation']['test_fips'] = test_tracts
    config['training']['use_mixture'] = True
    config['mixture']['enabled'] = True
    
    epochs = args.epochs or config.get('training', {}).get('epochs', 150)
    config['training']['epochs'] = epochs
    config['model']['epochs'] = epochs
    
    print(f"\nStarting training ({epochs} epochs)...")
    print(f"This may take 10-90 minutes depending on caching.\n")
    
    start_time = time.time()
    
    pipeline = GRANITEPipeline(config, output_dir=output_dir)
    results = pipeline.run()
    
    elapsed = time.time() - start_time
    
    if not results.get('success', False):
        print(f"\nTraining failed: {results.get('error', 'Unknown error')}")
        return 1
    
    # Process results
    test_results = results.get('test_results', {})
    
    if not test_results:
        print("\nNo test results available")
        return 1
    
    valid_results = {k: v for k, v in test_results.items() if v.get('mean_error_pct') is not None}
    
    if not valid_results:
        print("\nNo valid test results")
        return 1
    
    errors = [r['mean_error_pct'] for r in valid_results.values()]
    actual_svis = [r['actual_svi'] for r in valid_results.values()]
    predicted_svis = [r['predicted_mean'] for r in valid_results.values()]
    
    print(f"\n{'='*70}")
    print("GLOBAL TRAINING RESULTS")
    print(f"{'='*70}")
    print(f"Elapsed time: {elapsed/60:.1f} minutes")
    print(f"Test tracts evaluated: {len(valid_results)}/{len(test_tracts)}")
    
    print(f"\nConstraint Satisfaction:")
    print(f"  Mean error: {np.mean(errors):.2f}% +/- {np.std(errors):.2f}%")
    print(f"  Median error: {np.median(errors):.2f}%")
    print(f"  Min/Max error: {np.min(errors):.2f}% / {np.max(errors):.2f}%")
    print(f"  Tracts < 10% error: {sum(1 for e in errors if e < 10)}/{len(errors)}")
    print(f"  Tracts < 20% error: {sum(1 for e in errors if e < 20)}/{len(errors)}")
    
    r_squared = np.corrcoef(actual_svis, predicted_svis)[0, 1] ** 2
    print(f"\nCross-Tract Generalization:")
    print(f"  R-squared: {r_squared:.3f}")
    
    print(f"\nPer-Tract Results:")
    print(f"{'FIPS':<14} {'Actual':<8} {'Predicted':<10} {'Error%':<8} {'Expert':<10}")
    print("-" * 55)
    
    for fips in sorted(valid_results.keys(), key=lambda x: valid_results[x]['actual_svi']):
        r = valid_results[fips]
        print(f"{fips:<14} {r['actual_svi']:<8.3f} {r['predicted_mean']:<10.3f} "
              f"{r['mean_error_pct']:<8.1f} {r.get('dominant_expert', 'N/A'):<10}")
    
    # Save results
    results_df = pd.DataFrame([
        {
            'fips': fips,
            'actual_svi': r['actual_svi'],
            'predicted_svi': r['predicted_mean'],
            'error_pct': r['mean_error_pct'],
            'dominant_expert': r.get('dominant_expert', 'N/A')
        }
        for fips, r in valid_results.items()
    ])
    
    results_path = os.path.join(output_dir, 'validation_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Quality assessment
    print(f"\n{'='*70}")
    print("QUALITY ASSESSMENT")
    print(f"{'='*70}")
    
    mean_error = np.mean(errors)
    if mean_error < 10:
        print(f"Constraint satisfaction: EXCELLENT ({mean_error:.1f}% < 10%)")
    elif mean_error < 20:
        print(f"Constraint satisfaction: GOOD ({mean_error:.1f}% < 20%)")
    else:
        print(f"Constraint satisfaction: NEEDS IMPROVEMENT ({mean_error:.1f}%)")
    
    if r_squared > 0.5:
        print(f"Cross-tract generalization: STRONG (R2={r_squared:.3f})")
    elif r_squared > 0.25:
        print(f"Cross-tract generalization: MODERATE (R2={r_squared:.3f})")
    else:
        print(f"Cross-tract generalization: WEAK (R2={r_squared:.3f})")
    
    return 0


def run_multi_fips_experiment(args):
    """Run pipeline sequentially over a comma-separated list of FIPS codes.

    Saves per-tract metrics to output/tract_results_<arch>.csv and prints
    a summary table.  Returns 0 on success (partial failures are logged but
    do not affect the exit code).
    """
    import csv
    import copy
    import yaml
    from granite.models.gnn import set_random_seed
    from granite.disaggregation.pipeline import GRANITEPipeline

    fips_list = [f.strip() for f in args.fips.split(',') if f.strip()]
    arch = args.architecture
    output_dir = args.output or './output'
    os.makedirs(output_dir, exist_ok=True)

    # load base config once
    base_config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            base_config = yaml.safe_load(f) or {}
    base_config.setdefault('data', {})
    base_config.setdefault('model', {})
    base_config.setdefault('training', {})
    base_config.setdefault('processing', {})
    base_config.setdefault('validation', {})
    base_config.setdefault('mixture', {})

    # apply shared CLI overrides
    if args.epochs is not None:
        base_config['model']['epochs'] = args.epochs
        base_config['training']['epochs'] = args.epochs
    base_config['processing']['verbose'] = args.verbose
    base_config['processing']['enable_caching'] = not args.no_cache
    base_config['processing']['random_seed'] = args.seed
    base_config['processing']['skip_importance'] = True   # skip per-tract importance for speed
    base_config['validation']['compare_baselines'] = not args.skip_baselines
    base_config['training']['enforce_constraints'] = not args.no_constraints
    base_config['model']['architecture'] = arch
    if args.cache_dir:
        base_config['processing']['cache_dir'] = args.cache_dir
    if args.prune_features:
        base_config['processing']['prune_features_path'] = args.prune_features

    print(f"\n{'='*60}")
    print(f"GRANITE: Multi-FIPS Experiment ({arch.upper()})")
    print(f"{'='*60}")
    print(f"Tracts : {len(fips_list)}")
    print(f"Epochs : {args.epochs or base_config['model'].get('epochs', 100)}")
    print(f"Output : {output_dir}")
    print(f"{'='*60}\n")

    rows = []
    succeeded = []
    failed = []
    multi_tract_viz_data = {}  # collect for multi-tract heatmap

    for fips in fips_list:
        print(f"\n--- Processing {fips} ---")
        tract_config = copy.deepcopy(base_config)
        tract_config['data']['target_fips'] = fips
        tract_config['data']['state_fips'] = fips[:2]
        tract_config['data']['county_fips'] = fips[2:5]
        tract_config['data']['neighbor_tracts'] = 0  # single-tract mode

        set_random_seed(args.seed)
        try:
            pipeline = GRANITEPipeline(tract_config, output_dir=output_dir)
            results = pipeline.run()
        except Exception as e:
            print(f"  ERROR: {fips} raised exception: {e}")
            failed.append((fips, str(e)))
            rows.append({
                'fips': fips, 'svi': None, 'addresses': None,
                'architecture': arch,
                'constraint_error_pct': None, 'spatial_std': None,
                'morans_i': None, 'pred_min': None, 'pred_max': None,
                'status': 'FAILED'
            })
            continue

        if not results.get('success', False):
            err = results.get('error', 'unknown')
            print(f"  FAILED: {fips}: {err}")
            failed.append((fips, err))
            rows.append({
                'fips': fips, 'svi': None, 'addresses': None,
                'architecture': arch,
                'constraint_error_pct': None, 'spatial_std': None,
                'morans_i': None, 'pred_min': None, 'pred_max': None,
                'status': 'FAILED'
            })
            continue

        summary = results.get('summary', {})
        tract_info = results.get('tract_info', {})
        val = results.get('validation_results', {})
        preds = results.get('predictions')

        svi = tract_info.get('RPL_THEMES')
        addrs = summary.get('addresses_processed')
        constraint_err = summary.get('constraint_error')
        spatial_std = summary.get('spatial_variation')
        morans_i = None
        try:
            morans_i = val['spatial_diagnostics']['spatial_autocorrelation']
        except (KeyError, TypeError):
            pass
        pred_min = pred_max = None
        try:
            pred_vals = preds['mean'].values
            pred_min = float(pred_vals.min())
            pred_max = float(pred_vals.max())
        except Exception:
            pass

        rows.append({
            'fips': fips,
            'svi': svi,
            'addresses': addrs,
            'architecture': arch,
            'constraint_error_pct': constraint_err,
            'spatial_std': spatial_std,
            'morans_i': morans_i,
            'pred_min': pred_min,
            'pred_max': pred_max,
            'status': 'OK'
        })
        succeeded.append(fips)
        morans_str = f"{morans_i:.4f}" if morans_i is not None else 'N/A'
        print(f"  OK  svi={svi:.4f}  addrs={addrs}  err={constraint_err:.2f}%  std={spatial_std:.4f}  morans={morans_str}")

        # collect data for multi-tract heatmap
        address_gdf = results.get('address_gdf')
        if address_gdf is not None and preds is not None and svi is not None:
            multi_tract_viz_data[fips] = {
                'address_gdf': address_gdf,
                'predictions': preds['mean'].values,
                'tract_svi': svi
            }

    # generate multi-tract heatmap if we have results from 2+ tracts
    if len(multi_tract_viz_data) >= 2:
        try:
            from granite.visualization.plots import DisaggregationVisualizer
            viz = DisaggregationVisualizer()
            heatmap_path = os.path.join(output_dir, f'multi_tract_heatmap_{arch}.png')
            viz.plot_spatial_disaggregation(
                tract_results=multi_tract_viz_data,
                title=f'GRANITE Disaggregation ({len(multi_tract_viz_data)} tracts, {arch.upper()})',
                output_path=heatmap_path
            )
            print(f"Multi-tract heatmap saved to: {heatmap_path}")
        except Exception as e:
            print(f"Warning: multi-tract heatmap failed: {e}")

    # save per-arch CSV
    csv_path = os.path.join(output_dir, f'tract_results_{arch}.csv')
    fieldnames = ['fips', 'svi', 'addresses', 'architecture',
                  'constraint_error_pct', 'spatial_std', 'morans_i',
                  'pred_min', 'pred_max', 'status']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-tract results saved to: {csv_path}")

    print(f"\n{'='*60}")
    print(f"SUMMARY ({arch.upper()}): {len(succeeded)}/{len(fips_list)} tracts succeeded")
    if failed:
        print("Failed tracts:")
        for fips, err in failed:
            print(f"  {fips}: {err}")
    print(f"{'='*60}")
    return 0


def run_recovery_workflow(args, config):
    """Run held-out feature recovery experiment."""
    from granite.disaggregation.recovery_harness import run_recovery

    target_feature = args.recover_feature
    arch = args.architecture
    fips = args.fips
    seed = args.seed

    default_output = os.path.join(
        './output/recovery',
        f'{target_feature}_{arch}_{fips}_{seed}'
    )
    output_dir = args.output or default_output
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("GRANITE: Feature Recovery Experiment")
    print(f"{'='*60}")
    print(f"Target FIPS   : {fips}")
    print(f"Hold-out feat : {target_feature}")
    print(f"Architecture  : {arch}")
    print(f"Seed          : {seed}")
    print(f"Output        : {output_dir}")
    print(f"{'='*60}\n")

    result = run_recovery(
        config=config,
        target_feature=target_feature,
        output_dir=output_dir,
        verbose=args.verbose,
    )

    if result.get('success', False):
        import pandas as pd
        print(f"\n{'='*60}")
        print("Recovery experiment completed")
        print(f"{'='*60}")
        print(f"Constraint error: {result['overall_constraint_error']:.2f}%")
        print(f"Epochs trained  : {result['epochs_trained']}")
        print(f"Outputs written : {output_dir}")

        metrics_path = os.path.join(output_dir, 'per_tract_metrics.csv')
        if os.path.exists(metrics_path):
            metrics = pd.read_csv(metrics_path)
            print("\nPer-tract metrics:")
            print(metrics.to_string(index=False))
        return 0
    else:
        print(f"\nRecovery experiment failed: {result.get('error', 'unknown error')}")
        return 1


def main():
    """Main entry point."""
    args = parse_arguments()

    # validate --recover-feature constraints
    if args.recover_feature:
        if args.global_training:
            print("error: --recover-feature is incompatible with --global-training")
            return 1
        if not args.fips:
            print("error: --recover-feature requires --fips")
            return 1

    # comma-separated FIPS triggers the multi-FIPS experiment path
    if args.fips and ',' in args.fips:
        return run_multi_fips_experiment(args)

    config = load_config(args)

    if args.recover_feature:
        return run_recovery_workflow(args, config)
    elif args.global_training:
        return run_global_training(args, config)
    else:
        return run_single_tract(args, config)


if __name__ == "__main__":
    sys.exit(main())