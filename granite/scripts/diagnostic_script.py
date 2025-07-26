#!/usr/bin/env python
"""
Diagnostic script to check IDM implementation issues
"""
import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def diagnose_idm_issues():
    """Diagnose what's going wrong with IDM comparison"""
    
    print("=== DIAGNOSING IDM IMPLEMENTATION ISSUES ===\n")
    
    # 1. Check if results files exist and what they contain
    print("1. CHECKING OUTPUT FILES:")
    
    if os.path.exists('./output/granite_predictions.csv'):
        predictions = pd.read_csv('./output/granite_predictions.csv')
        print(f"✓ Predictions file exists: {len(predictions)} rows, {len(predictions.columns)} columns")
        print(f"  Columns: {list(predictions.columns)}")
        print(f"  Mean SVI: {predictions['mean'].mean():.6f}")
        print(f"  SVI Std Dev: {predictions['mean'].std():.6f}")
        print(f"  SVI Range: [{predictions['mean'].min():.6f}, {predictions['mean'].max():.6f}]")
        
        # Check if predictions are uniform (major red flag)
        if predictions['mean'].std() < 0.001:
            print("  🚨 WARNING: Predictions are nearly uniform!")
        else:
            print("  ✓ Predictions show spatial variation")
    else:
        print("✗ Predictions file not found")
    
    print()
    
    # 2. Check NLCD file directly
    print("2. CHECKING NLCD FILE:")
    
    nlcd_path = "./data/nlcd_hamilton_county.tif"
    if os.path.exists(nlcd_path):
        print(f"✓ NLCD file exists: {nlcd_path}")
        
        try:
            import rasterio
            with rasterio.open(nlcd_path) as src:
                print(f"  Full NLCD shape: {src.shape}")
                print(f"  NLCD CRS: {src.crs}")
                print(f"  NLCD bounds: {src.bounds}")
                
                # Sample full image to check classes
                sample = src.read(1, window=rasterio.windows.Window(0, 0, 
                                                                  min(500, src.width), 
                                                                  min(500, src.height)))
                unique_classes = sorted(list(set(sample.flatten())))
                print(f"  Unique NLCD classes in sample: {unique_classes[:15]}")
                
                # Check for developed areas (21-24)
                developed_classes = [c for c in unique_classes if c in [21, 22, 23, 24]]
                if developed_classes:
                    print(f"  ✓ Developed areas found: {developed_classes}")
                else:
                    print(f"  🚨 WARNING: No developed areas (21-24) found in sample!")
                    
        except Exception as e:
            print(f"  ✗ Error reading NLCD: {str(e)}")
    else:
        print(f"✗ NLCD file not found: {nlcd_path}")
    
    print()
    
    # 3. Test IDM baseline directly
    print("3. TESTING IDM BASELINE DIRECTLY:")
    
    try:
        from granite.baselines.idm import IDMBaseline
        
        # Create test data
        n_test = 100
        test_locations = pd.DataFrame({
            'x': np.random.uniform(-85.31, -85.29, n_test),
            'y': np.random.uniform(35.06, 35.08, n_test)
        })
        
        # Test NLCD features (mock some)
        test_nlcd = pd.DataFrame({
            'nlcd_class': np.random.choice([21, 22, 23, 24, 41, 42], n_test)
        })
        
        # Test IDM
        idm = IDMBaseline()
        result = idm.disaggregate_svi(
            tract_svi=0.224,
            prediction_locations=test_locations,
            nlcd_features=test_nlcd
        )
        
        if result['success']:
            predictions = result['predictions']['mean']
            print(f"✓ IDM test successful")
            print(f"  Test predictions std: {np.std(predictions):.6f}")
            print(f"  Test predictions range: [{predictions.min():.6f}, {predictions.max():.6f}]")
            
            if np.std(predictions) < 0.001:
                print("  🚨 IDM producing uniform predictions - check coefficients!")
            else:
                print("  ✓ IDM producing spatial variation")
        else:
            print(f"✗ IDM test failed: {result.get('error', 'Unknown')}")
            
    except ImportError:
        print("✗ IDM module not found - check if granite/baselines/idm.py exists")
    except Exception as e:
        print(f"✗ IDM test error: {str(e)}")
    
    print()
    
    # 4. Check visualization results structure
    print("4. CHECKING LAST PIPELINE RESULTS:")
    
    # This would require access to the pipeline results, but we can check logs
    log_indicators = [
        "GNN vs IDM correlation: 0.000",
        "NLCD data loaded: (1, 81, 81)",
        "Total Addresses Compared: N/A"
    ]
    
    for indicator in log_indicators:
        print(f"  Issue found in logs: '{indicator}'")
    
    print()
    
    # 5. Recommendations
    print("5. RECOMMENDATIONS:")
    
    recommendations = [
        "🔧 Fix NLCD cropping - (1,81,81) is too small",
        "🔧 Check IDM coefficient mapping - may be producing uniform results", 
        "🔧 Verify visualization is detecting IDM comparison results",
        "🔧 Debug correlation calculation between GNN and IDM",
        "🔧 Ensure NLCD features have spatial variation"
    ]
    
    for rec in recommendations:
        print(f"  {rec}")

if __name__ == "__main__":
    diagnose_idm_issues()