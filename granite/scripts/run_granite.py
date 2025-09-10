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
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    config = {
        'data': {'target_fips': args.fips, 'processing_mode': 'fips'},
        'model': {'epochs': args.epochs},
        'processing': {'verbose': args.verbose}
    }
    
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