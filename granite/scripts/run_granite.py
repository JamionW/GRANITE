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
    """Main entry point for GRANITE pipeline"""
    
    # Parse arguments FIRST
    parser = argparse.ArgumentParser(description='GRANITE Pipeline')
    parser.add_argument('--fips', type=str, help='Target FIPS code(s) - single or comma-separated')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    args = parser.parse_args()
    
    print(f"GRANITE Framework - Starting analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Load config
        config = load_config('config/config.yaml')
        
        # FIXED: Handle comma-separated FIPS codes
        if args.fips:
            if ',' in args.fips:
                # Multiple FIPS codes
                fips_list = [fips.strip() for fips in args.fips.split(',')]
                config['data']['processing_mode'] = 'multi_fips'
                config['data']['target_fips_list'] = fips_list
                print(f"Configuration:")
                print(f"  Processing mode: multi_fips")
                print(f"  Target FIPS list: {fips_list}")
            else:
                # Single FIPS code
                config['data']['target_fips'] = args.fips.strip()
                config['data']['processing_mode'] = 'fips'
                print(f"Configuration:")
                print(f"  Processing mode: fips")
                print(f"  Target FIPS: {args.fips}")
        else:
            # No FIPS specified - process all
            config['data']['processing_mode'] = 'county'
            print(f"Configuration:")
            print(f"  Processing mode: county")
        
        if args.epochs:
            config['model']['epochs'] = args.epochs
            print(f"  Model epochs: {args.epochs}")
        
        print(f"  Output directory: ./output")
        
        # Create pipeline
        from granite.disaggregation.pipeline import GRANITEPipeline
        
        pipeline = GRANITEPipeline(
            config=config,
            data_dir='./data',
            output_dir='./output',
            verbose=True
        )
        
        results = pipeline.run()
        if results.get('success', False):
            print("Analysis completed successfully")
            return {'success': True, 'message': 'Analysis completed'}
        else:
            print(f"Analysis failed: {results.get('error', 'Unknown error')}")
            return {'success': False}
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    main()