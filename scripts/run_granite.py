#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main execution script for GRANITE framework

GRANITE: Graph-Refined Accessibility Network for Integrated Transit Equity

Usage:
    python run_granite.py [--roads_file PATH] [--epochs N] [--output_dir PATH]
"""
import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from granite.disaggregation.pipeline import GRANITEPipeline


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='GRANITE: SVI Disaggregation using GNN-MetricGraph Integration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        '--roads_file',
        type=str,
        default=None,
        help='Path to roads shapefile (e.g., tl_2023_47065_roads.shp)'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='Directory containing input data'
    )
    
    # Model arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of GNN training epochs'
    )
    
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=64,
        help='Hidden dimension for GNN'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Learning rate for GNN training'
    )
    
    # Output arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output',
        help='Directory for output files'
    )
    
    parser.add_argument(
        '--no_visualize',
        action='store_true',
        help='Skip visualization creation'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    return parser.parse_args()


def print_banner():
    """Print GRANITE banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                            GRANITE                                    ║
    ║  Graph-Refined Accessibility Network for Integrated Transit Equity   ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║  A framework for Social Vulnerability Index disaggregation using     ║
    ║  Graph Neural Networks integrated with MetricGraph spatial models    ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Print banner
    if not args.quiet:
        print_banner()
    
    # Start time
    start_time = datetime.now()
    
    print(f"\nStarting GRANITE pipeline at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  - Roads file: {args.roads_file or 'Download from Census'}")
    print(f"  - Data directory: {args.data_dir}")
    print(f"  - Output directory: {args.output_dir}")
    print(f"  - GNN epochs: {args.epochs}")
    print(f"  - Visualization: {'Yes' if not args.no_visualize else 'No'}")
    print()
    
    try:
        # Create pipeline
        pipeline = GRANITEPipeline(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            verbose=not args.quiet
        )
        
        # Run pipeline
        results = pipeline.run(
            roads_file=args.roads_file,
            epochs=args.epochs,
            visualize=not args.no_visualize
        )
        
        # Calculate runtime
        end_time = datetime.now()
        runtime = end_time - start_time
        
        print(f"\n{'='*70}")
        print(f"GRANITE pipeline completed successfully!")
        print(f"Total runtime: {runtime}")
        print(f"{'='*70}")
        
        # Print summary results
        if 'predictions' in results:
            print("\nSummary Results:")
            print(f"  - Addresses processed: {len(results['predictions'])}")
            print(f"  - Mean predicted SVI: {results['predictions']['mean'].mean():.3f}")
            print(f"  - Average uncertainty: {results['predictions']['sd'].mean():.3f}")
        
        if 'validation' in results:
            val_df = results['validation']
            print(f"\nValidation Results:")
            print(f"  - Tracts validated: {len(val_df)}")
            print(f"  - Mean absolute error: {val_df['error'].mean():.4f}")
            print(f"  - Correlation: {val_df['true_svi'].corr(val_df['predicted_avg']):.3f}")
        
        print(f"\nResults saved to: {args.output_dir}")
        print("  - granite_predictions.csv")
        print("  - gnn_features.npy")
        if 'validation' in results:
            print("  - validation_results.csv")
        if not args.no_visualize:
            print("  - granite_visualization.png")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()