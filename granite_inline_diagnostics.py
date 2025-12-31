#!/usr/bin/env python3
"""
GRANITE Inline Diagnostics
==========================
Run this from your GRANITE directory. It hooks into the pipeline,
computes features for a tract, and runs diagnostics.

Usage:
    cd /workspaces/GRANITE
    python granite_inline_diagnostics.py

No arguments needed - it uses your existing config and data.
"""

import sys
import os
import numpy as np

# Ensure we're in GRANITE directory
if not os.path.exists('./granite'):
    print("ERROR: Run this from the GRANITE root directory")
    print("  cd /workspaces/GRANITE")
    print("  python granite_inline_diagnostics.py")
    sys.exit(1)

sys.path.insert(0, '.')

def main():
    print("\n" + "="*70)
    print(" GRANITE INLINE DIAGNOSTICS")
    print(" Running directly against your pipeline")
    print("="*70 + "\n")
    
    # =========================================================================
    # STEP 1: Load GRANITE modules
    # =========================================================================
    print("[1/5] Loading GRANITE modules...")
    
    try:
        from granite.disaggregation.pipeline import GRANITEPipeline
        from granite.data.loaders import DataLoader
        from granite.data.enhanced_accessibility import EnhancedAccessibilityComputer
        print("  ✓ Modules loaded successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import: {e}")
        print("  Make sure you're in the GRANITE directory with the simplify branch checked out")
        sys.exit(1)
    
    # =========================================================================
    # STEP 2: Initialize pipeline and load data
    # =========================================================================
    print("\n[2/5] Initializing pipeline and loading data...")
    
    config = {
        'data': {'data_dir': './data'},
        'processing': {'random_seed': 42},
        'model': {
            'hidden_dim': 64,
            'dropout': 0.3,
            'epochs': 100
        },
        'training': {
            'constraint_weight': 3.0,
            'enforce_constraints': True
        }
    }
    
    try:
        loader = DataLoader(data_dir='./data', config=config)
        
        # Load data piece by piece (matching actual API)
        census_tracts = loader.load_census_tracts('47', '065')
        svi = loader.load_svi_data('47', 'Hamilton')
        tracts = census_tracts.merge(svi, on='FIPS', how='inner')
        
        employment = loader.create_employment_destinations(use_real_data=True)
        healthcare = loader.create_healthcare_destinations(use_real_data=True)
        grocery = loader.create_grocery_destinations(use_real_data=True)
        
        data = {
            'loader': loader,
            'tracts': tracts,
            'svi': svi,
            'census_tracts': census_tracts,
            'employment_destinations': employment,
            'healthcare_destinations': healthcare,
            'grocery_destinations': grocery
        }
        
        print(f"  ✓ Loaded {len(tracts)} tracts with SVI data")
        print(f"  ✓ Employment destinations: {len(employment)}")
        print(f"  ✓ Healthcare destinations: {len(healthcare)}")
        print(f"  ✓ Grocery destinations: {len(grocery)}")
    except Exception as e:
        print(f"  ✗ Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # =========================================================================
    # STEP 3: Get a sample of tracts and compute features
    # =========================================================================
    print("\n[3/5] Computing features for sample tracts...")
    
    # Use CURATED tracts from run_holdout_validation.py - these are likely cached
    # Training tracts (known to have been processed):
    curated_tracts = [
        '47065012000',  # Very Low SVI, 592 addresses
        '47065000600',  # Low SVI, 2089 addresses (your main test tract)
        '47065012400',  # Medium SVI, 2316 addresses
        '47065003000',  # Very High SVI, 1276 addresses
    ]
    
    print(f"  Using curated tracts (likely cached): {curated_tracts}")
    
    all_features = []
    all_tract_ids = []
    
    pipeline = GRANITEPipeline(config=config, data_dir='./data', output_dir='./output', verbose=False)
    
    for fips in curated_tracts:
        fips_str = str(fips)  # Already correct format
        print(f"  Processing tract {fips_str}...")
        
        try:
            addresses = loader.get_addresses_for_tract(fips_str)
            if len(addresses) == 0:
                print(f"    Skipping - no addresses")
                continue
            
            # Compute features using pipeline method
            features = pipeline._compute_accessibility_features(addresses, data)
            
            if features is not None and len(features) > 0:
                all_features.append(features)
                all_tract_ids.extend([fips_str] * len(features))
                print(f"    ✓ {len(addresses)} addresses, {features.shape[1]} features")
            else:
                print(f"    ✗ Feature computation failed")
                
        except Exception as e:
            print(f"    ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    if len(all_features) == 0:
        print("\n  ✗ No features computed. Check OSRM servers and data paths.")
        sys.exit(1)
    
    # Combine features
    features = np.vstack(all_features)
    tract_assignments = np.array(all_tract_ids)
    
    print(f"\n  Combined: {features.shape[0]} addresses, {features.shape[1]} features")
    
    # =========================================================================
    # STEP 4: Run diagnostics
    # =========================================================================
    print("\n[4/5] Running diagnostics...\n")
    
    run_diagnostics(features, tract_assignments)
    
    # =========================================================================
    # STEP 5: Save for later analysis
    # =========================================================================
    print("\n[5/5] Saving diagnostic data...")
    
    os.makedirs('./diagnostic_output', exist_ok=True)
    np.save('./diagnostic_output/features.npy', features)
    np.save('./diagnostic_output/tract_assignments.npy', tract_assignments)
    
    print("  Saved to ./diagnostic_output/")
    print("  You can now run: python granite_quick_check.py --features ./diagnostic_output/features.npy --tracts ./diagnostic_output/tract_assignments.npy")


def run_diagnostics(features, tract_assignments):
    """Run all diagnostic checks."""
    
    n_samples, n_cols = features.shape
    unique_tracts = np.unique(tract_assignments)
    
    # Get feature names
    feature_names = get_feature_names(n_cols)
    
    issues = []
    
    # =========================================================================
    # CHECK 1: NaN/Inf values
    # =========================================================================
    print("="*60)
    print(" CHECK #1: NaN/Inf Values")
    print("="*60)
    
    nan_cols = []
    for col in range(n_cols):
        data = features[:, col]
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        
        if nan_count > 0 or inf_count > 0:
            name = feature_names[col] if col < len(feature_names) else f"col_{col}"
            nan_cols.append((col, name, nan_count, inf_count))
            print(f"  ✗ {name}: {nan_count} NaN, {inf_count} Inf")
    
    total_nan = sum(c[2] for c in nan_cols)
    total_inf = sum(c[3] for c in nan_cols)
    
    if nan_cols:
        print(f"\n  CRITICAL: {len(nan_cols)} columns have {total_nan} NaN and {total_inf} Inf values")
        print("  This WILL cause loss=nan during training!")
        issues.append("NaN/Inf values")
        
        # Show which rows have problems
        problem_rows = np.where(np.isnan(features).any(axis=1) | np.isinf(features).any(axis=1))[0]
        print(f"\n  Problem addresses: {len(problem_rows)} / {n_samples}")
        
        if len(problem_rows) <= 20:
            for row in problem_rows[:10]:
                tract = tract_assignments[row]
                bad_cols = np.where(np.isnan(features[row]) | np.isinf(features[row]))[0]
                col_names = [feature_names[c] if c < len(feature_names) else f"col_{c}" for c in bad_cols]
                print(f"    Row {row} (tract {tract}): {col_names}")
    else:
        print("  ✓ PASS: No NaN/Inf values found")
    
    # =========================================================================
    # CHECK 2: Within-tract variance
    # =========================================================================
    print("\n" + "="*60)
    print(" CHECK #2: Within-Tract Feature Variance")
    print("="*60)
    
    # Find features with zero variance in ALL tracts
    zero_var_features = []
    
    for col in range(n_cols):
        is_constant_everywhere = True
        
        for tract in unique_tracts:
            mask = tract_assignments == tract
            tract_data = features[mask, col]
            
            # Skip if not enough data
            if len(tract_data) < 2:
                continue
                
            # Check variance (ignoring NaN)
            valid_data = tract_data[~np.isnan(tract_data)]
            if len(valid_data) > 1 and np.var(valid_data) > 1e-10:
                is_constant_everywhere = False
                break
        
        if is_constant_everywhere:
            name = feature_names[col] if col < len(feature_names) else f"col_{col}"
            zero_var_features.append((col, name))
    
    pct = len(zero_var_features) / n_cols * 100
    
    print(f"  Features with NO within-tract variance: {len(zero_var_features)}/{n_cols} ({pct:.1f}%)")
    
    if zero_var_features:
        print("\n  These features are USELESS for within-tract learning:")
        for col, name in zero_var_features:
            print(f"    - {name}")
    
    if pct > 40:
        print(f"\n  CRITICAL: {pct:.0f}% of features have no within-tract signal!")
        print("  The GNN cannot learn meaningful variation from these features.")
        issues.append("Zero-variance features (>40%)")
    elif pct > 20:
        print(f"\n  WARNING: {pct:.0f}% of features have no within-tract signal")
    else:
        print("\n  ✓ PASS: Most features have within-tract variance")
    
    # =========================================================================
    # CHECK 3: Loss weight analysis
    # =========================================================================
    print("\n" + "="*60)
    print(" CHECK #3: Loss Function Weight Balance")
    print("="*60)
    
    # Default weights from codebase
    weights = {
        'constraint': 3.0,
        'variation': 0.5,
        'bounds': 1.0,
        'range': 0.3,
        'accessibility': 0.2
    }
    
    print("  Current weights (from granite/models/gnn.py):")
    for name, w in weights.items():
        print(f"    {name}: {w}")
    
    ratio = weights['constraint'] / weights['variation']
    print(f"\n  Constraint/Variation ratio: {ratio:.1f}x")
    
    if ratio > 4:
        print(f"\n  CRITICAL: Constraint is {ratio:.0f}x stronger than variation!")
        print("  Model will converge to uniform predictions (tract mean).")
        issues.append("Loss weight imbalance")
    elif ratio > 2:
        print(f"\n  WARNING: Constraint may dominate variation learning")
    else:
        print("  ✓ PASS: Weight balance looks reasonable")
    
    # =========================================================================
    # CHECK 4: Feature statistics
    # =========================================================================
    print("\n" + "="*60)
    print(" CHECK #4: Feature Statistics Summary")
    print("="*60)
    
    print(f"\n  {'Feature':<35} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("  " + "-"*75)
    
    for col in range(min(n_cols, 30)):  # Show first 30
        data = features[:, col]
        valid = data[~np.isnan(data) & ~np.isinf(data)]
        
        if len(valid) > 0:
            name = feature_names[col] if col < len(feature_names) else f"col_{col}"
            print(f"  {name:<35} {np.mean(valid):>10.3f} {np.std(valid):>10.3f} {np.min(valid):>10.3f} {np.max(valid):>10.3f}")
    
    if n_cols > 30:
        print(f"  ... and {n_cols - 30} more features")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*60)
    print(" DIAGNOSTIC SUMMARY")
    print("="*60)
    
    if issues:
        print(f"\n  ✗ CRITICAL ISSUES FOUND: {len(issues)}")
        for issue in issues:
            print(f"    - {issue}")
        print("\n  Fix these before concluding GRANITE is a negative result!")
        print("\n  Recommended fixes:")
        if "NaN/Inf values" in issues:
            print("    1. Check OSRM servers are running (docker ps)")
            print("    2. Add NaN handling in enhanced_accessibility.py (see pressure point #1)")
        if "Zero-variance features" in issues:
            print("    3. Remove tract-level features or compute them per-address")
        if "Loss weight imbalance" in issues:
            print("    4. Rebalance loss weights: constraint=1.0, variation=2.0")
    else:
        print("\n  ✓ All checks passed!")
        print("  If the model still underperforms, this may be a genuine negative result.")
    
    return issues


def get_feature_names(n_features):
    """Generate feature names based on expected structure."""
    
    # Base accessibility features (10 per destination type = 30)
    base = ['min_time', 'mean_time', 'median_time', 
            'count_5min', 'count_10min', 'count_15min',
            'drive_advantage', 'dispersion', 'time_range', 'percentile']
    
    dest_types = ['employment', 'healthcare', 'grocery']
    
    names = []
    
    # 30 base features
    for dest in dest_types:
        for feat in base:
            names.append(f"{dest}_{feat}")
    
    # If more than 30, add modal features (15)
    if n_features > 30:
        modal = ['avg_time', 'time_std', 'access_density', 'equity_gap', 'car_advantage']
        for dest in dest_types:
            for feat in modal:
                names.append(f"modal_{dest}_{feat}")
    
    # If more than 45, add socioeconomic (9)
    if n_features > 45:
        socio = ['no_vehicle', 'poverty', 'unemployment', 'no_highschool',
                 'age65_plus', 'age17_under', 'disability', 'single_parent', 'minority']
        names.extend(socio)
    
    # Pad if needed
    while len(names) < n_features:
        names.append(f"feature_{len(names)}")
    
    return names


if __name__ == '__main__':
    main()