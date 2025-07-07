#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main execution script for GRANITE framework

GRANITE: Graph-Refined Accessibility Network for Integrated Transit Equity

Usage:
    # Original county-wide processing
    python run_granite.py --epochs 5 --mode county
    
    # New FIPS-based processing
    python run_granite.py --list_fips
    python run_granite.py --fips 47065001100 --epochs 5
    python run_granite.py --mode fips --fips_count 5 --epochs 5
"""
import os
import sys
import argparse
from datetime import datetime
from typing import List

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
    
    # NEW FIPS-related arguments
    parser.add_argument(
        '--mode',
        type=str,
        choices=['county', 'fips', 'auto'],
        default='auto',
        help='Processing mode: county-wide (original), FIPS-based (memory-efficient), or auto-detect'
    )
    
    parser.add_argument(
        '--fips',
        type=str,
        default=None,
        help='FIPS code(s) to process. Single: "47065001100", Multiple: "47065001100,47065001200"'
    )
    
    parser.add_argument(
        '--fips_count',
        type=int,
        default=5,
        help='Number of FIPS codes to process if no specific codes given'
    )
    
    parser.add_argument(
        '--tract_buffer',
        type=float,
        default=0.01,
        help='Buffer around tract in degrees for road network (FIPS mode only)'
    )
    
    parser.add_argument(
        '--max_nodes',
        type=int,
        default=None,  
        help='Maximum nodes in road network graph (None=unlimited, recommended for disaggregation)'
    )

    parser.add_argument(
        '--preserve_network',
        action='store_true',
        default=True,  # Default to preserving network
        help='Preserve full network granularity (recommended for disaggregation)'
    )
    
    parser.add_argument(
        '--allow_simplification',
        action='store_true',
        default=False,
        help='Allow network simplification (reduces disaggregation granularity)'
    )
    
    parser.add_argument(
        '--list_fips',
        action='store_true',
        help='List available FIPS codes and exit'
    )
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Show what would be processed without running pipeline'
    )
    
    return parser.parse_args()


def parse_fips_codes(fips_string: str) -> List[str]:
    """Parse FIPS codes from command line string"""
    if not fips_string:
        return []
    
    # Handle comma-separated list
    fips_list = [fips.strip() for fips in fips_string.split(',')]
    
    # Validate FIPS codes (should be 11 digits for tract-level)
    valid_fips = []
    for fips in fips_list:
        if len(fips) == 11 and fips.isdigit():
            valid_fips.append(fips)
        else:
            print(f"Warning: Invalid FIPS code format: {fips} (should be 11 digits)")
    
    return valid_fips


def determine_processing_mode(args) -> tuple:
    """Determine processing mode and target FIPS codes"""
    target_fips = parse_fips_codes(args.fips) if args.fips else []
    
    if args.mode == 'auto':
        if target_fips:
            mode = 'fips'
        else:
            # Default to FIPS mode for memory efficiency
            print("Auto-detecting mode: Using FIPS mode for memory efficiency")
            mode = 'fips'
    else:
        mode = args.mode
    
    return mode, target_fips


def list_available_fips(args):
    """List available FIPS codes and exit"""
    from granite.data.loaders import DataLoader
    
    print("Loading available census tracts...")
    loader = DataLoader(args.data_dir, verbose=True)
    
    try:
        fips_codes = loader.get_available_fips_codes('47', '065')  # Hamilton County, TN
        
        print(f"\nAvailable FIPS codes in Hamilton County, TN:")
        print(f"{'FIPS Code':<15} {'Tract':<10}")
        print("-" * 25)
        
        for fips in fips_codes:
            tract_num = fips[-6:]  # Last 6 digits are tract number
            print(f"{fips:<15} {tract_num:<10}")
        
        print(f"\nTotal tracts: {len(fips_codes)}")
        print(f"\nExample usage:")
        print(f"  Single tract:    --fips {fips_codes[0]} --mode fips --preserve_network")
        print(f"  Multiple tracts: --fips {','.join(fips_codes[:3])} --mode fips --preserve_network")
        print(f"  First 5 tracts:  --mode fips --fips_count 5 --preserve_network")
        print(f"  Allow simplification: --mode fips --allow_simplification --max_nodes 5000")
            
    except Exception as e:
        print(f"Error loading FIPS codes: {str(e)}")


def update_config_for_fips(args):
    """Update configuration for FIPS processing with network preservation"""
    
    # Determine network preservation settings
    preserve_network = args.preserve_network and not args.allow_simplification
    
    # Create a configuration dictionary for FIPS mode
    config = {
        'data': {
            'state_fips': '47',
            'county_fips': '065',
            'processing_mode': 'fips',
            'fips_config': {
                'memory': {
                    'tract_buffer_degrees': args.tract_buffer,
                    'max_network_nodes': args.max_nodes,  # Can be None now
                    'max_network_edges': args.max_nodes * 2 if args.max_nodes else None,
                    'preserve_network': preserve_network  # NEW: Add preservation setting
                },
                'batch': {
                    'auto_select': {
                        'enabled': True,
                        'mode': 'range',
                        'range_start': 1,
                        'range_end': args.fips_count
                    }
                }
            }
        },
        'model': {
            'epochs': args.epochs,
            'hidden_dim': min(args.hidden_dim, 32),  # Use smaller hidden dim for FIPS mode
            'learning_rate': args.learning_rate,
            'input_dim': 5,
            'output_dim': 3,
            'dropout': 0.2
        },
        'metricgraph': {
            'alpha': 1.5,
            'mesh_resolution': 0.02,
            'formula': "y ~ gnn_kappa + gnn_alpha + gnn_tau"
        },
        'output': {
            'save_predictions': True,
            'save_features': True,
            'save_validation': True,
            'save_plots': True,
            'fips_output': {
                'individual_tract_folders': True,
                'batch_summary': True,
                'combined_predictions': True
            }
        },
        'processing': {
            'parallel_tracts': False,
            'max_workers': 4,
            'continue_on_error': True
        }
    }
    
    # Set specific FIPS list if provided
    if args.fips:
        target_fips = parse_fips_codes(args.fips)
        config['data']['fips_config']['batch']['target_list'] = target_fips
    
    return config

def print_banner():
    """Print GRANITE banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                            GRANITE                                   ║
    ║  Graph-Refined Accessibility Network for Integrated Transit Equity   ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║  A framework for Social Vulnerability Index disaggregation using     ║
    ║  Graph Neural Networks integrated with MetricGraph spatial models    ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_fips_results(results, mode, target_fips):
    """Print results for FIPS processing"""
    if results.get('mode') == 'batch':
        print(f"\nFIPS Batch Processing Results:")
        print(f"  - Total tracts: {results['total_tracts']}")
        print(f"  - Successful: {results['successful']}")
        print(f"  - Skipped: {results['skipped']}")
        print(f"  - Errors: {results['errors']}")
        print(f"  - Success rate: {results['success_rate']:.1%}")
        
        # Show sample of successful results
        successful_results = {k: v for k, v in results['results'].items() 
                            if v.get('status') == 'success'}
        if successful_results:
            sample_fips = list(successful_results.keys())[0]
            sample_result = successful_results[sample_fips]
            if 'predictions' in sample_result:
                print(f"\nSample Results (Tract {sample_fips}):")
                print(f"  - Address predictions: {len(sample_result['predictions'])}")
                mean_svi = sample_result['predictions']['predicted_svi'].mean()
                print(f"  - Mean predicted SVI: {mean_svi:.3f}")
    
    else:
        # Single tract results
        if results['status'] == 'success':
            print(f"\nSingle Tract Results:")
            print(f"  - FIPS: {results['fips_code']}")
            print(f"  - Network nodes: {results['network_stats']['nodes']}")
            print(f"  - Predictions: {len(results['predictions'])}")
            mean_svi = results['predictions']['predicted_svi'].mean()
            print(f"  - Mean predicted SVI: {mean_svi:.3f}")
        else:
            print(f"\nTract processing {results['status']}: {results.get('reason', 'Unknown')}")


def print_county_results(results):
    """Print results for county-wide processing"""
    if 'predictions' in results:
        print("\nCounty-wide Results:")
        print(f"  - Addresses processed: {len(results['predictions'])}")
        if 'mean' in results['predictions'].columns:
            print(f"  - Mean predicted SVI: {results['predictions']['mean'].mean():.3f}")
            print(f"  - Average uncertainty: {results['predictions']['sd'].mean():.3f}")
        else:
            print("  - Prediction details available in output files")
    
    if 'validation' in results:
        val_df = results['validation']
        print(f"\nValidation Results:")
        print(f"  - Tracts validated: {len(val_df)}")
        if 'error' in val_df.columns:
            print(f"  - Mean absolute error: {val_df['error'].mean():.4f}")
        if 'true_svi' in val_df.columns and 'predicted_avg' in val_df.columns:
            print(f"  - Correlation: {val_df['true_svi'].corr(val_df['predicted_avg']):.3f}")


def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Handle special modes
    if args.list_fips:
        list_available_fips(args)
        return
    
    # Print banner
    if not args.quiet:
        print_banner()
    
    # Determine processing mode
    mode, target_fips = determine_processing_mode(args)
    
    # Start time
    start_time = datetime.now()
    
    print(f"\nStarting GRANITE pipeline at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  - Processing mode: {mode}")
    if mode == 'fips':
        print(f"  - Target FIPS: {target_fips if target_fips else f'Auto-select {args.fips_count} tracts'}")
        print(f"  - Tract buffer: {args.tract_buffer} degrees")
        
        # REPLACE the max_nodes print with network preservation info:
        preserve_network = args.preserve_network and not args.allow_simplification
        if preserve_network:
            print(f"  - Network granularity: PRESERVED (recommended for disaggregation)")
        else:
            print(f"  - Max network nodes: {args.max_nodes}")
            print(f"  - Network granularity: LIMITED (may reduce disaggregation quality)")
            
    print(f"  - Roads file: {args.roads_file or 'Download from Census'}")
    print(f"  - Data directory: {args.data_dir}")
    print(f"  - Output directory: {args.output_dir}")
    print(f"  - GNN epochs: {args.epochs}")
    print(f"  - Visualization: {'Yes' if not args.no_visualize else 'No'}")
    print()
    
    # Dry run mode
    if args.dry_run:
        print("DRY RUN MODE - No actual processing will occur")
        if mode == 'fips':
            print(f"Would process {len(target_fips) if target_fips else args.fips_count} census tracts")
        else:
            print("Would process entire Hamilton County")
        return
    
    try:
        # Create pipeline
        pipeline = GRANITEPipeline(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            verbose=not args.quiet
        )
        
        if mode == 'fips':
            # Update configuration for FIPS processing
            config = update_config_for_fips(args)
            
            # Run in FIPS mode
            results = pipeline.run_fips_mode(
                config=config,
                fips_list=target_fips if target_fips else None,
                epochs=args.epochs,
                visualize=not args.no_visualize
            )
            
            # Print FIPS-specific results
            print_fips_results(results, mode, target_fips)
            
        else:
            # Original county-wide processing
            results = pipeline.run(
                roads_file=args.roads_file,
                epochs=args.epochs,
                visualize=not args.no_visualize,
                mode='county'
            )
            
            # Print original results
            print_county_results(results)
        
        # Calculate runtime
        end_time = datetime.now()
        runtime = end_time - start_time
        
        print(f"\n{'='*70}")
        print(f"GRANITE pipeline completed successfully!")
        print(f"Total runtime: {runtime}")
        print(f"Results saved to: {args.output_dir}")
        print(f"{'='*70}")
        
    except KeyboardInterrupt:
        print(f"\n\nProcessing interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()