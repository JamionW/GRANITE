"""
Updated command-line interface for simplified GRANITE
"""
import os
import sys
import argparse
import yaml
from datetime import datetime

# Add your project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description='GRANITE Simplified: Accessibility → SVI')
    parser.add_argument('--fips', type=str, required=True, help='Target FIPS code (e.g., 47065010100)')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--neighbor-tracts', type=int, default=0, help='Number of neighboring tracts (0=single, 3-5=multi-tract)')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--cache-dir', type=str, default='./granite_cache', help='Cache directory')
    parser.add_argument('--skip-importance', action='store_true', help='Skip feature importance analysis') 
    parser.add_argument('--holdout-validation', action='store_true',
                    help='Run holdout validation (train on neighbors, predict on target)')
    parser.add_argument('--evaluate-unconstrained', action='store_true',
                    help='Evaluate model without constraint enforcement')
    parser.add_argument('--no-constraints', action='store_true',
                    help='Train without constraint enforcement')

    args = parser.parse_args()
    
    # Load configuration
    config = {
        'data': {
            'target_fips': args.fips,
            'state_fips': args.fips[:2],
            'county_fips': args.fips[2:5],
            'neighbor_tracts': args.neighbor_tracts
        },
        'model': {
            'epochs': args.epochs,
            'hidden_dim': 64,
            'dropout': 0.3
        },
        'training': {
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'enforce_constraints': not args.no_constraints
        },
        'processing': {
            'verbose': args.verbose,
            'enable_caching': not args.no_cache,
            'cache_dir': args.cache_dir,
            'skip_importance': args.skip_importance 
        },
        'validation': {
            'holdout_mode': args.holdout_validation,
            'evaluate_unconstrained': args.evaluate_unconstrained,
            'neighbor_fips': []  # Auto-discovered in pipeline
        }
    }
    
    # Override with config file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            file_config = yaml.safe_load(f)
            config.update(file_config)
    
    print(f"\n{'='*60}")
    print(f"GRANITE Simplified: Accessibility → SVI Prediction")
    print(f"{'='*60}")
    print(f"Target FIPS: {args.fips}")
    print(f"Epochs: {args.epochs}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")
    
    # Run the simplified pipeline
    from granite.disaggregation.pipeline import GRANITEPipeline
    
    pipeline = GRANITEPipeline(config, output_dir=args.output)
    results = pipeline.run()
    
    if results.get('success', False):
        print(f"\n{'='*60}")
        print(f"SUCCESS: Analysis completed!")
        print(f"{'='*60}")
        print(f"Addresses processed: {results['summary']['addresses_processed']}")
        print(f"Accessibility features: {results['summary']['accessibility_features']}")
        print(f"Spatial variation: {results['summary']['spatial_variation']:.4f}")
        print(f"Constraint error: {results['summary']['constraint_error']:.2f}%")
        print(f"Training epochs: {results['summary']['training_epochs']}")
        
        # Save results
        pipeline.save_results(results)
        print(f"\nResults saved to: {args.output}")
        
        # Print validation summary if available
        if 'validation_results' in results:
            val = results['validation_results']
            print(f"\nValidation Results:")
            print(f"  Constraint satisfaction: {val['quality_metrics']['constraint_satisfaction']}")
            print(f"  Spatial variation: {val['quality_metrics']['spatial_variation']}")
            if 'accessibility_svi_correlations' in val:
                corr = val['accessibility_svi_correlations'].get('overall', 'N/A')
                print(f"  Accessibility-SVI correlation: {corr}")
    else:
        print(f"\nERROR: {results.get('error', 'Unknown error')}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()