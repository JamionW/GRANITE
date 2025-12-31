"""
GRANITE CLI (Spatial Version)

Simplified entry point for spatial disaggregation.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from granite.models.gnn import set_random_seed
from granite.disaggregation.pipeline import GRANITEPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description='GRANITE: Spatial Disaggregation of Social Vulnerability'
    )
    
    parser.add_argument('--fips', type=str, required=True,
                        help='Target census tract FIPS code')
    parser.add_argument('--neighbor-tracts', type=int, default=0,
                        help='Number of neighboring tracts (0 for single-tract)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--hidden-dim', type=int, default=32,
                        help='GNN hidden dimension')
    parser.add_argument('--k-neighbors', type=int, default=8,
                        help='Graph k-NN neighbors')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output', type=str, default='./output',
                        help='Output directory')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = {
        'data': {
            'target_fips': args.fips,
            'state_fips': args.fips[:2],
            'county_fips': args.fips[2:5],
            'neighbor_tracts': args.neighbor_tracts
        },
        'model': {
            'hidden_dim': args.hidden_dim,
            'k_neighbors': args.k_neighbors,
            'dropout': 0.2
        },
        'training': {
            'epochs': args.epochs,
            'learning_rate': 0.001,
            'constraint_weight': 2.0
        },
        'processing': {
            'verbose': args.verbose,
            'random_seed': args.seed
        }
    }
    
    set_random_seed(args.seed)
    
    print(f"\n{'='*60}")
    print("GRANITE: Spatial Disaggregation")
    print(f"{'='*60}")
    print(f"Target: {args.fips}")
    print(f"Mode: {'Multi-tract' if args.neighbor_tracts > 0 else 'Single-tract'}")
    print(f"{'='*60}\n")
    
    pipeline = GRANITEPipeline(config, output_dir=args.output)
    results = pipeline.run()
    
    if results.get('success'):
        print(f"\n{'='*60}")
        print("Results Summary")
        print(f"{'='*60}")
        summary = results.get('summary', {})
        print(f"Addresses: {summary.get('addresses_processed', 'N/A')}")
        print(f"Spatial features: {summary.get('spatial_features', 'N/A')}")
        print(f"GNN spatial variation (std): {summary.get('spatial_variation', 0):.4f}")
        print(f"GNN constraint error: {summary.get('constraint_error', 0):.2f}%")
        
        # Baseline comparison
        baselines = results.get('baselines', {})
        if baselines:
            print(f"\n{'='*60}")
            print("Baseline Comparison")
            print(f"{'='*60}")
            print(f"{'Method':<12} {'Std Dev':<12} {'Constraint Err':<15}")
            print("-" * 40)
            print(f"{'GNN':<12} {summary.get('spatial_variation', 0):<12.4f} {summary.get('constraint_error', 0):.2f}%")
            
            if 'idw' in baselines and 'std' in baselines['idw']:
                print(f"{'IDW':<12} {baselines['idw']['std']:<12.4f} {baselines['idw'].get('constraint_error', 0):.2f}%")
            elif 'idw' in baselines:
                print(f"{'IDW':<12} {'FAILED':<12} {baselines['idw'].get('error', '')[:20]}")
            
            if 'kriging' in baselines and 'std' in baselines['kriging']:
                print(f"{'Kriging':<12} {baselines['kriging']['std']:<12.4f} {baselines['kriging'].get('constraint_error', 0):.2f}%")
            elif 'kriging' in baselines:
                print(f"{'Kriging':<12} {'FAILED':<12} {baselines['kriging'].get('error', '')[:20]}")
            
            print(f"{'Naive':<12} {'0.0000':<12} {'0.00'}%")
        
        # Generate plots
        print(f"\n{'='*60}")
        print("Generating Visualizations...")
        print(f"{'='*60}")
        pipeline.generate_plots(results)
        
        # Save CSV
        pipeline.save_results(results)
        
        return 0
    else:
        print(f"\nFailed: {results.get('error', 'Unknown error')}")
        return 1


if __name__ == '__main__':
    sys.exit(main())