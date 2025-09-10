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

def parse_arguments():
    parser = argparse.ArgumentParser(description='GRANITE Accessibility Research')
    parser.add_argument('--fips', type=str, help='Target FIPS code')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found, using defaults")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing config file {config_path}: {e}")
        return {}

def main():
    args = parse_arguments()
    
    # Load base config from YAML file
    config = load_config(args.config)
    
    # Override/merge with command line arguments
    if args.fips:
        if 'data' not in config:
            config['data'] = {}
        config['data']['target_fips'] = args.fips
        config['data']['processing_mode'] = 'fips'
    
    # Override epochs if provided (for both stages)
    if args.epochs != 50:  # Only override if not default
        if 'model' not in config:
            config['model'] = {}
        config['model']['accessibility_epochs'] = args.epochs
        config['model']['svi_epochs'] = args.epochs
    
    # Set processing options
    if 'processing' not in config:
        config['processing'] = {}
    config['processing']['verbose'] = args.verbose
    
    print(f"Using config: FIPS={config.get('data', {}).get('target_fips')}, "
          f"Epochs={config.get('model', {}).get('accessibility_epochs', 50)}")
    
    pipeline = GRANITEPipeline(config, output_dir=args.output)
    results = pipeline.run()
    
    if results.get('success', False):
        print(f"Analysis completed successfully!")
        print(f"Processed {results['summary']['total_addresses']} addresses")
    else:
        print(f"Analysis failed: {results.get('error', 'Unknown error')}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()