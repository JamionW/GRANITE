#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-line entry point for GRANITE framework

This module provides the main() function that serves as the entry point
for the 'granite' command.
"""
import os
import sys
import argparse
import yaml
from datetime import datetime
from typing import Dict, List, Optional

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from granite.disaggregation.pipeline import GRANITEPipeline
from granite.data.loaders import DataLoader


def load_config(config_path: str = 'config/config.yaml') -> Dict:
    """Load configuration from YAML file"""
    # Try multiple possible config locations
    possible_paths = [
        config_path,
        os.path.join(os.getcwd(), config_path),
        os.path.join(parent_dir, config_path),
        os.path.join(parent_dir, '..', config_path)
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return yaml.safe_load(f)
    
    print(f"Warning: Config file not found. Searched in: {possible_paths}")
    print("Using default configuration.")
    
    # Return default config
    return {
        'data': {
            'state': 'Tennessee',
            'state_fips': '47',
            'county': 'Hamilton',
            'county_fips': '065',
            'processing_mode': 'fips',
            'fips_config': {
                'memory': {
                    'tract_buffer_degrees': 0.01,
                    'max_network_nodes': 10000,
                    'max_network_edges': 20000,
                    'preserve_network': True
                },
                'batch': {
                    'auto_select': {
                        'enabled': True,
                        'mode': 'range',
                        'range_start': 1,
                        'range_end': 5
                    }
                }
            }
        },
        'model': {
            'epochs': 100,
            'hidden_dim': 64,
            'learning_rate': 0.01,
            'output_dim': 3
        },
        'metricgraph': {
            'alpha': 1,
            'mesh_resolution': 0.05,
            'max_edges': 2000,
            'enable_sampling': False
        },
        'output': {
            'save_predictions': True,
            'save_features': True,
            'save_plots': True
        },
        'processing': {
            'continue_on_error': True,
            'verbose': True
        }
    }


def merge_config_with_args(config: Dict, args: argparse.Namespace) -> Dict:
    """Merge configuration with command-line arguments"""
    import copy
    merged = copy.deepcopy(config)
    
    # Processing mode
    if args.mode:
        merged['data']['processing_mode'] = args.mode
    
    # Model parameters
    if args.epochs is not None:
        merged['model']['epochs'] = args.epochs
    
    if args.hidden_dim is not None:
        merged['model']['hidden_dim'] = args.hidden_dim
    
    if args.learning_rate is not None:
        merged['model']['learning_rate'] = args.learning_rate
    
    # FIPS configuration
    if args.fips:
        fips_list = [f.strip() for f in args.fips.split(',')]
        merged['data']['fips_config']['batch']['target_list'] = fips_list
        merged['data']['fips_config']['batch']['auto_select']['enabled'] = False
    
    if args.fips_count is not None:
        merged['data']['fips_config']['batch']['auto_select']['range_end'] = args.fips_count
    
    # Memory configuration
    if args.tract_buffer is not None:
        merged['data']['fips_config']['memory']['tract_buffer_degrees'] = args.tract_buffer
    
    if args.max_nodes is not None:
        merged['data']['fips_config']['memory']['max_network_nodes'] = args.max_nodes
    
    # MetricGraph configuration
    if args.enable_sampling:
        merged['metricgraph']['enable_sampling'] = True
    
    # Output configuration
    if args.no_visualize:
        merged['output']['save_plots'] = False
    
    # File paths
    if args.roads_file:
        merged['data']['roads_file'] = args.roads_file
    
    return merged


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='GRANITE: SVI Disaggregation using GNN-MetricGraph Integration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    # Data arguments
    parser.add_argument(
        '--roads_file',
        type=str,
        default=None,
        help='Path to roads shapefile'
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
        default=None,
        help='Number of GNN training epochs'
    )
    
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=None,
        help='Hidden dimension for GNN'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='Learning rate for GNN training'
    )
    
    # Processing mode
    parser.add_argument(
        '--mode',
        type=str,
        choices=['county', 'fips'],
        default=None,
        help='Processing mode'
    )
    
    parser.add_argument(
        '--fips',
        type=str,
        default=None,
        help='FIPS code(s) to process'
    )
    
    parser.add_argument(
        '--fips_count',
        type=int,
        default=None,
        help='Number of FIPS codes to process'
    )
    
    parser.add_argument(
        '--tract_buffer',
        type=float,
        default=None,
        help='Buffer around tract in degrees'
    )
    
    parser.add_argument(
        '--max_nodes',
        type=int,
        default=None,
        help='Maximum nodes in road network'
    )
    
    parser.add_argument(
        '--enable_sampling',
        action='store_true',
        help='Enable smart network sampling'
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
    
    # Utility arguments
    parser.add_argument(
        '--list_fips',
        action='store_true',
        help='List available FIPS codes and exit'
    )
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Show configuration without running'
    )
    
    return parser.parse_args()


def list_available_fips(data_dir: str, state_fips: str, county_fips: str):
    """List available FIPS codes"""
    loader = DataLoader(data_dir, verbose=True)
    
    try:
        fips_codes = loader.get_available_fips_codes(state_fips, county_fips)
        
        print(f"\nAvailable FIPS codes in Hamilton County, TN:")
        print(f"{'FIPS Code':<15} {'Tract':<10}")
        print("-" * 25)
        
        for fips in fips_codes:
            tract_num = fips[-6:]
            print(f"{fips:<15} {tract_num:<10}")
        
        print(f"\nTotal tracts: {len(fips_codes)}")
        print("\nExample usage:")
        print(f"  granite --fips {fips_codes[0]}")
        print(f"  granite --fips {','.join(fips_codes[:3])}")
        print(f"  granite --fips_count 5")
            
    except Exception as e:
        print(f"Error loading FIPS codes: {str(e)}")
        sys.exit(1)


def main():
    """Main entry point for granite command"""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Merge with command-line arguments
    config = merge_config_with_args(config, args)
    
    # Handle list FIPS mode
    if args.list_fips:
        list_available_fips(
            args.data_dir,
            config['data']['state_fips'],
            config['data']['county_fips']
        )
        return
    
    # Start time
    start_time = datetime.now()
    
    print("\n" + "="*60)
    print("GRANITE: Graph-Refined Accessibility Network")
    print("         for Integrated Transit Equity")
    print("="*60)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Print configuration
    if not args.quiet:
        print("\nConfiguration:")
        print(f"  Processing mode: {config['data']['processing_mode']}")
        print(f"  Model epochs: {config['model']['epochs']}")
        print(f"  Output directory: {args.output_dir}")
        
        if config['data']['processing_mode'] == 'fips':
            if args.fips:
                print(f"  Target FIPS: {args.fips}")
            else:
                auto = config['data']['fips_config']['batch']['auto_select']
                if auto['mode'] == 'range':
                    print(f"  Auto-select: tracts {auto['range_start']} to {auto['range_end']}")
    
    # Dry run mode
    if args.dry_run:
        print("\nDRY RUN MODE - No processing will occur.")
        return
    
    try:
        # Create pipeline
        pipeline = GRANITEPipeline(
            config=config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            verbose=not args.quiet
        )
        
        # Run pipeline
        results = pipeline.run()
        
        # Calculate runtime
        end_time = datetime.now()
        runtime = end_time - start_time
        
        print("\n" + "="*60)
        print("GRANITE pipeline completed successfully!")
        print(f"Total runtime: {runtime}")
        print(f"Results saved to: {args.output_dir}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
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