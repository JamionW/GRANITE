#!/usr/bin/env python3
"""
Quick confounding analysis for GRANITE single-tract issue
Tests if within-tract accessibility-vulnerability inversion is due to confounding
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

def load_granite_results(output_dir='./output'):
    """Load the most recent GRANITE results"""
    output_path = Path(output_dir)
    
    # Look for results files
    predictions_file = output_path / 'granite_predictions.csv'
    
    if not predictions_file.exists():
        print(f"Error: Could not find {predictions_file}")
        print("Please run GRANITE first to generate results")
        return None, None
    
    # Load predictions
    predictions_df = pd.read_csv(predictions_file)
    
    print(f"\nLoaded {len(predictions_df)} addresses from {predictions_file}")
    print(f"Columns: {predictions_df.columns.tolist()}")
    
    # Try to load accessibility features if available
    features_file = output_path / 'accessibility_features.csv'
    if features_file.exists():
        features_df = pd.read_csv(features_file)
        print(f"\nLoaded accessibility features from {features_file}")
        print(f"Feature columns: {features_df.columns.tolist()}")
    else:
        print(f"\nNote: {features_file} not found")
        print("Will analyze spatial patterns only (without accessibility features)")
        features_df = None
    
    return predictions_df, features_df


def calculate_centrality_measures(df):
    """Calculate distance from tract centroid using x, y columns"""
    
    if 'x' not in df.columns or 'y' not in df.columns:
        print("Error: Cannot find x, y coordinate columns")
        return None
    
    # Calculate tract centroid
    centroid_x = df['x'].mean()
    centroid_y = df['y'].mean()
    
    print(f"\nTract centroid: (x={centroid_x:.6f}, y={centroid_y:.6f})")
    
    # Calculate Euclidean distance from centroid
    # Note: These are likely already in a projected coordinate system (meters or feet)
    df['dist_from_center'] = np.sqrt(
        (df['x'] - centroid_x)**2 + (df['y'] - centroid_y)**2
    )
    
    # Standardize distance for interpretation
    df['dist_from_center_normalized'] = (
        (df['dist_from_center'] - df['dist_from_center'].min()) / 
        (df['dist_from_center'].max() - df['dist_from_center'].min())
    )
    
    print(f"Distance range: {df['dist_from_center'].min():.1f} to {df['dist_from_center'].max():.1f} units")
    
    return df


def analyze_confounding(predictions_df, features_df, svi_col='mean'):
    """
    Test confounding hypothesis:
    - Is centrality correlated with SVI?
    - If accessibility features available, is centrality correlated with accessibility?
    """
    
    print("\n" + "="*80)
    print("CONFOUNDING ANALYSIS")
    print("="*80)
    
    # Basic centrality-SVI correlation
    corr_centrality_svi = predictions_df['dist_from_center_normalized'].corr(predictions_df[svi_col])
    
    print(f"\n{'Spatial Pattern Analysis':^80}")
    print("-"*80)
    print(f"Distance from center ↔ SVI:           {corr_centrality_svi:7.3f}")
    
    # If we have accessibility features, analyze those too
    if features_df is not None:
        # Find time-based features (should correlate positively with distance from center in typical cities)
        time_cols = [col for col in features_df.columns if 'time' in col.lower() and 'min' in col.lower()]
        count_cols = [col for col in features_df.columns if 'count' in col.lower()]
        
        if len(time_cols) > 0:
            # Use mean of first few time features as accessibility proxy
            features_df['avg_travel_time'] = features_df[time_cols[:3]].mean(axis=1)
            
            # Merge with predictions to get centrality
            merged_df = pd.merge(
                predictions_df[['address_id', 'dist_from_center_normalized', svi_col]],
                features_df[['address_id', 'avg_travel_time']],
                on='address_id',
                how='inner'
            )
            
            corr_centrality_access = merged_df['dist_from_center_normalized'].corr(merged_df['avg_travel_time'])
            corr_access_svi = merged_df['avg_travel_time'].corr(merged_df[svi_col])
            
            print(f"Distance from center ↔ Avg Travel Time: {corr_centrality_access:7.3f}")
            print(f"Avg Travel Time ↔ SVI:                  {corr_access_svi:7.3f}")
            print("-"*80)
            
            # Partial correlation
            r_xy = corr_access_svi
            r_xz = corr_centrality_access
            r_yz = corr_centrality_svi
            
            numerator = r_xy - (r_xz * r_yz)
            denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
            
            if denominator > 0.01:
                partial_corr = numerator / denominator
                print(f"\n{'Partial Correlation (controlling for centrality)':^80}")
                print("-"*80)
                print(f"Travel Time ↔ SVI (controlling for distance): {partial_corr:7.3f}")
            else:
                partial_corr = None
        else:
            print("\nNo travel time features found in accessibility data")
            corr_centrality_access = None
            corr_access_svi = None
            partial_corr = None
    else:
        print("\n(Accessibility features not available - analyzing spatial pattern only)")
        print("-"*80)
        corr_centrality_access = None
        corr_access_svi = None
        partial_corr = None
    
    # Interpretation
    print(f"\n{'Interpretation':^80}")
    print("="*80)
    
    if abs(corr_centrality_svi) > 0.3:
        print("\n⚠️  STRONG SPATIAL PATTERN DETECTED!")
        
        if corr_centrality_svi > 0:
            print("\n  • Addresses farther from center → Higher SVI")
            print("  • Pattern: Suburban/peripheral addresses are more vulnerable")
        else:
            print("\n  • Addresses farther from center → Lower SVI")
            print("  • Pattern: Urban/central addresses are more vulnerable")
            print("\n🚨 THIS CREATES THE BACKWARDS LEARNING PROBLEM!")
            print("\nExplanation:")
            print("  1. Central addresses have HIGH SVI (urban core, likely rentals)")
            print("  2. Central addresses have GOOD accessibility (short travel times)")
            print("  3. Model learns: good accessibility → high SVI (backwards!)")
            print("\n  The TRUE relationship:")
            print("  • Urban = vulnerable + accessible")
            print("  • Suburbs = affluent + car-dependent")
            print("  • Accessibility is IRRELEVANT - housing/wealth drives the pattern")
        
        if corr_centrality_access is not None and abs(corr_centrality_access) > 0.3:
            print("\n✓ Centrality also correlates with accessibility")
            print("  This confirms spatial confounding.")
            
            if partial_corr is not None and abs(partial_corr) < 0.2:
                print("\n✓ After controlling for centrality, accessibility-SVI correlation weakens")
                print("  → Centrality is the primary confounding variable")
    
    elif abs(corr_centrality_svi) < 0.1:
        print("\n✓ No strong spatial pattern detected")
        print("\nThe backwards learning is NOT due to simple centrality confounding.")
        print("\nOther potential causes:")
        print("  • Feature calculation errors")
        print("  • Spatial smoothing overwhelming accessibility signal")
        print("  • Other confounding variables (housing type, density)")
    
    else:
        print(f"\n→ Moderate spatial pattern (r={corr_centrality_svi:.3f})")
        print("  May contribute to but doesn't fully explain the backwards learning")
    
    # Quartile analysis for clarity
    print(f"\n{'Quartile Analysis':^80}")
    print("="*80)
    
    predictions_df['distance_quartile'] = pd.qcut(
        predictions_df['dist_from_center_normalized'], 
        q=4, 
        labels=['Q1_Center', 'Q2', 'Q3', 'Q4_Periphery']
    )
    
    quartile_stats = predictions_df.groupby('distance_quartile')[svi_col].agg(['mean', 'std', 'count'])
    print("\nSVI by Distance from Center:")
    print(quartile_stats)
    
    print("\n" + "="*80)
    
    return {
        'corr_centrality_svi': corr_centrality_svi,
        'corr_centrality_access': corr_centrality_access,
        'corr_access_svi': corr_access_svi,
        'partial_corr': partial_corr,
        'confounding_detected': abs(corr_centrality_svi) > 0.3
    }


def main():
    """Run confounding analysis on GRANITE results"""
    
    print("="*80)
    print("GRANITE CONFOUNDING ANALYSIS")
    print("="*80)
    
    # Load results
    predictions_df, features_df = load_granite_results()
    
    if predictions_df is None:
        sys.exit(1)
    
    # Calculate centrality
    predictions_df = calculate_centrality_measures(predictions_df)
    
    if predictions_df is None:
        sys.exit(1)
    
    # Analyze confounding
    results = analyze_confounding(predictions_df, features_df)
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if results['confounding_detected']:
        if results['corr_centrality_svi'] < 0:
            print("\n✓ CONFOUNDING PATTERN CONFIRMED")
            print("\nThe single-tract training creates a confounded learning problem where:")
            print("  • Central (urban) addresses = high SVI + good accessibility")
            print("  • Peripheral (suburban) addresses = low SVI + poor accessibility")
            print("  • Model incorrectly learns: good accessibility → high SVI")
        
        print("\n1. EXPAND TO MULTI-TRACT TRAINING (Recommended)")
        print("   • Train on 15-30 tracts with varying SVI and accessibility")
        print("   • Captures between-tract accessibility-vulnerability gradient")
        print("   • Should reverse the correlation to expected negative direction")
        
        print("\n2. Add spatial confounder controls")
        print("   • Include distance-to-center as explicit feature")
        print("   • Add housing density / land use features if available")
        print("   • Use hierarchical model: location → housing type → SVI")
        
        print("\n3. Reduce spatial smoothing")
        print("   • Decrease k-NN neighbors (try 3-5 instead of 10-15)")
        print("   • Use accessibility-weighted edges")
        print("   • Force model to rely more on node features")
    
    else:
        print("\n1. No strong centrality confounding detected")
        print("\n2. Investigate other potential issues:")
        print("   • Check for feature calculation errors")
        print("   • Test spatial smoothing dominance (train without spatial edges)")
        print("   • Look for other confounding variables (housing type, density)")
        
        print("\n3. Still recommend multi-tract training:")
        print("   • Even without obvious confounding, single-tract has limited variation")
        print("   • Between-tract gradient is where accessibility effects are strongest")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()