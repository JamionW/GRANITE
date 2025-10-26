#!/usr/bin/env python3
"""Test socioeconomic controls integration"""

import sys
import yaml
from granite.disaggregation.pipeline import GRANITEPipeline

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Override for single tract
config['data']['target_fips'] = '47065000600'
config['processing'] = {'verbose': True}

# Run pipeline
pipeline = GRANITEPipeline(
    config=config,
    data_dir='./data',
    output_dir='./output',
    verbose=True
)

results = pipeline.run()

# Check results
if results['success']:
    print("\n" + "="*60)
    print("SOCIOECONOMIC CONTROLS INTEGRATION TEST")
    print("="*60)
    
    # Check feature count
    feature_count = results['accessibility_features'].shape[1]
    print(f"Total features: {feature_count}")
    print(f"Expected: 34 (accessibility) + 9 (socioeconomic) = 43")
    
    if feature_count == 43:
        print("✓ Feature count correct!")
    else:
        print(f"✗ Expected 43 features, got {feature_count}")
    
    # Check for negative values
    neg_count = (results['accessibility_features'] < 0).sum()
    print(f"\nNegative values: {neg_count}")
    
    if neg_count == 0:
        print("✓ No negative values!")
    else:
        print(f"✗ Found {neg_count} negative values")
    
    print("\n" + "="*60)
else:
    print(f"Pipeline failed: {results.get('error', 'Unknown error')}")
    sys.exit(1)