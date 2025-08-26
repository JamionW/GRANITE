#!/usr/bin/env python3
"""
Quick verification that GNN corrections are working properly
Run this after your GRANITE analysis completes
"""

import pandas as pd
import numpy as np
import os

def verify_gnn_performance(output_dir='./output'):
    """Verify GNN is generating meaningful spatial corrections"""
    
    print("=" * 60)
    print("GNN PERFORMANCE VERIFICATION")
    print("=" * 60)
    
    # Load predictions
    try:
        predictions = pd.read_csv(os.path.join(output_dir, 'granite_predictions.csv'))
        print(f"✅ Loaded {len(predictions)} predictions")
    except Exception as e:
        print(f"❌ Could not load predictions: {e}")
        return
    
    # Check if we have the expected columns
    required_columns = ['mean', 'fips', 'tract_svi']
    optional_columns = ['idm_baseline', 'gnn_correction', 'hybrid_final']
    
    print(f"\nColumn Analysis:")
    print(f"Available columns: {list(predictions.columns)}")
    
    missing_required = [col for col in required_columns if col not in predictions.columns]
    if missing_required:
        print(f"❌ Missing required columns: {missing_required}")
        return
    
    available_optional = [col for col in optional_columns if col in predictions.columns]
    print(f"✅ Available optional columns: {available_optional}")
    
    # Basic statistics
    gnn_predictions = predictions['mean']
    tract_svi = predictions['tract_svi'].iloc[0]
    
    print(f"\n📊 PREDICTION STATISTICS:")
    print(f"  Tract SVI target: {tract_svi:.6f}")
    print(f"  GNN prediction mean: {gnn_predictions.mean():.6f}")
    print(f"  Constraint error: {abs(gnn_predictions.mean() - tract_svi):.6f}")
    print(f"  GNN prediction std: {gnn_predictions.std():.6f}")
    print(f"  GNN prediction range: [{gnn_predictions.min():.6f}, {gnn_predictions.max():.6f}]")
    print(f"  Coefficient of variation: {gnn_predictions.std() / gnn_predictions.mean():.3f}")
    
    # Check for collapse
    if gnn_predictions.std() < 0.001:
        print("🚨 WARNING: Very low spatial variation - possible parameter collapse")
    elif gnn_predictions.std() < 0.01:
        print("⚠️  Caution: Low spatial variation")
    else:
        print("✅ Good spatial variation")
    
    # Analyze corrections if available
    if 'gnn_correction' in predictions.columns:
        corrections = predictions['gnn_correction']
        print(f"\n🔧 GNN CORRECTION ANALYSIS:")
        print(f"  Correction std: {corrections.std():.6f}")
        print(f"  Correction range: [{corrections.min():.6f}, {corrections.max():.6f}]")
        print(f"  Correction mean: {corrections.mean():.6f} (should be ~0)")
        
        if corrections.std() > 0.01:
            print("✅ Meaningful corrections generated")
        elif corrections.std() > 0.001:
            print("⚠️  Modest corrections generated")
        else:
            print("❌ Minimal corrections - GNN not learning much")
    
    # Compare methods if available
    if 'idm_baseline' in predictions.columns:
        idm_baseline = predictions['idm_baseline']
        print(f"\n🆚 GNN vs IDM COMPARISON:")
        print(f"  GNN std: {gnn_predictions.std():.6f}")
        print(f"  IDM std: {idm_baseline.std():.6f}")
        print(f"  Variation ratio: {idm_baseline.std() / gnn_predictions.std():.2f}:1 (IDM:GNN)")
        
        # Correlation
        correlation = np.corrcoef(gnn_predictions, idm_baseline)[0, 1]
        print(f"  Method correlation: {correlation:.3f}")
        
        if abs(correlation) > 0.7:
            print("✅ High method agreement")
        elif abs(correlation) > 0.3:
            print("⚠️  Moderate method agreement")
        else:
            print("❌ Low method agreement - investigate")
    
    # Success assessment
    print(f"\n🎯 OVERALL ASSESSMENT:")
    
    constraint_ok = abs(gnn_predictions.mean() - tract_svi) < 0.01
    variation_ok = gnn_predictions.std() > 0.005
    
    if constraint_ok and variation_ok:
        print("✅ GNN WORKING CORRECTLY")
        print("   - Tract constraint satisfied")
        print("   - Meaningful spatial variation generated")
        print("   - Ready for research analysis")
    elif constraint_ok:
        print("⚠️  GNN PARTIALLY WORKING")
        print("   - Tract constraint satisfied")
        print("   - Low spatial variation (check loss function)")
    else:
        print("❌ GNN NEEDS DEBUGGING")
        print("   - Constraint not satisfied or no spatial variation")
    
    return {
        'constraint_satisfied': constraint_ok,
        'spatial_variation': variation_ok,
        'prediction_std': gnn_predictions.std(),
        'constraint_error': abs(gnn_predictions.mean() - tract_svi)
    }

if __name__ == "__main__":
    verify_gnn_performance()