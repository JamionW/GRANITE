#!/usr/bin/env python3
"""
Quick feature validation without full GNN training.
Tests correlation directions and identifies confounds.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from granite.data.accessibility_loader import AccessibilityLoader

def validate_feature_directions(features, svi_values, feature_names):
    """
    Validate that features correlate with SVI in expected directions.
    
    Expected correlations:
    - Time features (min/mean/median): POSITIVE (longer time → higher vulnerability)
    - Count features (5min/10min/15min): NEGATIVE (more destinations → lower vulnerability)
    - Drive advantage: POSITIVE (car-dependent → higher vulnerability)
    - Walkability: NEGATIVE (walkable → lower vulnerability)
    """
    
    results = []
    
    for i, name in enumerate(feature_names):
        feat_vals = features[:, i]
        
        # Skip if no variance
        if feat_vals.std() < 1e-6:
            continue
            
        corr, pval = pearsonr(feat_vals, svi_values)
        
        # Determine expected direction
        if any(x in name for x in ['min_time', 'mean_time', 'median_time', 'time_range']):
            expected = 'positive'
            correct = corr > 0
        elif any(x in name for x in ['count_5min', 'count_10min', 'count_15min']):
            expected = 'negative'
            correct = corr < 0
        elif 'drive_advantage' in name:
            expected = 'positive'
            correct = corr > 0
        elif 'walkability' in name or 'walk_competitive' in name:
            expected = 'negative'
            correct = corr < 0
        else:
            expected = 'unknown'
            correct = None
            
        results.append({
            'feature': name,
            'correlation': corr,
            'p_value': pval,
            'expected': expected,
            'correct': correct
        })
    
    df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "="*80)
    print("FEATURE CORRELATION VALIDATION")
    print("="*80)
    
    # Overall stats
    known_direction = df[df['expected'] != 'unknown']
    if len(known_direction) > 0:
        pct_correct = known_direction['correct'].sum() / len(known_direction) * 100
        print(f"\nOverall Correctness: {pct_correct:.1f}% ({known_direction['correct'].sum}/{len(known_direction)})")
        
        if pct_correct < 60:
            print("❌ FAIL: Less than 60% of features have correct direction")
            print("   → Feature encoding is flawed or confounding factors present")
        elif pct_correct < 80:
            print("⚠️  WARN: 60-80% correct - some issues remain")
        else:
            print("✅ PASS: Feature directions are mostly correct")
    
    # Show problematic features
    print("\n" + "-"*80)
    print("PROBLEMATIC FEATURES (wrong direction):")
    print("-"*80)
    wrong = df[(df['correct'] == False) & (df['p_value'] < 0.05)]
    for _, row in wrong.iterrows():
        print(f"  {row['feature']:40s} r={row['correlation']:+.3f} (expected {row['expected']})")
    
    # Check for perfect correlations
    print("\n" + "-"*80)
    print("REDUNDANCY CHECK:")
    print("-"*80)
    n_perfect = 0
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            corr = np.corrcoef(features[:, i], features[:, j])[0, 1]
            if abs(corr) > 0.99:
                print(f"  {feature_names[i]} <-> {feature_names[j]}: r={corr:.4f}")
                n_perfect += 1
    
    if n_perfect == 0:
        print("  ✅ No perfect correlations detected")
    else:
        print(f"  ❌ Found {n_perfect} perfect/near-perfect correlations")
    
    return df, pct_correct if len(known_direction) > 0 else 0

def test_single_tract(fips='47065000600'):
    """Test feature generation for a single tract."""
    
    print(f"\nLoading data for tract {fips}...")
    loader = AccessibilityLoader(
        fips=fips,
        neighbor_tracts=0,  # Single tract only for speed
        data_dir='./data'
    )
    
    # Load spatial data
    loader.load_spatial_data()
    loader.load_accessibility_destinations()
    
    print(f"Loaded {len(loader.addresses)} addresses")
    print(f"Target SVI: {loader.target_tract_svi:.4f}")
    
    # Generate features
    print("\nGenerating accessibility features...")
    features, feature_names = loader.compute_enhanced_accessibility_features()
    
    print(f"Generated {features.shape[1]} features:")
    print(f"  Feature matrix shape: {features.shape}")
    print(f"  Feature names: {len(feature_names)}")
    
    # Get SVI values (all addresses in tract have same tract-level SVI)
    svi_values = np.full(len(loader.addresses), loader.target_tract_svi)
    
    # Validate
    df, pct_correct = validate_feature_directions(features, svi_values, feature_names)
    
    # Save results
    output_dir = Path('./output/validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f'feature_validation_{fips}.csv', index=False)
    
    return pct_correct >= 80  # Pass threshold

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fips', default='47065000600', help='Target FIPS code')
    args = parser.parse_args()
    
    success = test_single_tract(args.fips)
    sys.exit(0 if success else 1)