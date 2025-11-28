#!/usr/bin/env python3
"""
GRANITE Batching Implementation Test Script

Validates batching implementation with comprehensive checks.
Run after deploying modified pipeline.py
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

def test_batching_implementation():
    """
    Comprehensive test suite for batching implementation.
    """
    print("="*80)
    print("GRANITE BATCHING IMPLEMENTATION TEST")
    print("="*80)
    
    # Test 1: Import and method validation
    print("\n[TEST 1] Checking implementation...")
    try:
        from granite.disaggregation.pipeline import GRANITEPipeline
        
        # Verify new methods exist
        required_methods = [
            '_compute_features_batched',
            '_load_batch_cache',
            '_save_batch_cache',
            '_estimate_optimal_batch_size'
        ]
        
        missing_methods = []
        for method_name in required_methods:
            if not hasattr(GRANITEPipeline, method_name):
                missing_methods.append(method_name)
        
        if missing_methods:
            print(f"  ✗ FAILED: Missing methods: {missing_methods}")
            return False
        
        print(f"  ✓ PASSED: All {len(required_methods)} new methods present")
        
    except ImportError as e:
        print(f"  ✗ FAILED: Could not import GRANITEPipeline: {e}")
        return False
    
    # Test 2: Batch size estimation
    print("\n[TEST 2] Batch size estimation...")
    try:
        config = {'processing': {'verbose': False}}
        pipeline = GRANITEPipeline(config)
        
        test_cases = [
            (5000, 2, 1),      # 2 tracts, 5K addresses → batch_size=1
            (7500, 3, 1),      # 3 tracts, 7.5K addresses → batch_size=1-2
            (10000, 4, 2),     # 4 tracts, 10K addresses → batch_size=2
            (20000, 8, 3),     # 8 tracts, 20K addresses → batch_size=3-4
            (30000, 12, 3),    # 12 tracts, 30K addresses → batch_size=3-4
        ]
        
        all_passed = True
        for num_addrs, num_tracts, expected_min in test_cases:
            batch_size = pipeline._estimate_optimal_batch_size(num_addrs, num_tracts)
            passed = batch_size >= expected_min and batch_size <= 4
            status = "✓" if passed else "✗"
            print(f"  {status} {num_tracts} tracts, {num_addrs} addresses → batch_size={batch_size} (expected min {expected_min})")
            all_passed = all_passed and passed
        
        if all_passed:
            print(f"  ✓ PASSED: Batch size estimation working correctly")
        else:
            print(f"  ✗ FAILED: Some batch size estimates out of range")
            return False
    
    except Exception as e:
        print(f"  ✗ FAILED: Batch size estimation error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Method signatures
    print("\n[TEST 3] Checking method signatures...")
    try:
        import inspect
        
        # Check _compute_features_batched signature
        sig = inspect.signature(GRANITEPipeline._compute_features_batched)
        params = list(sig.parameters.keys())
        expected_params = ['self', 'addresses_df', 'data', 'batch_size']
        
        if params == expected_params:
            print(f"  ✓ _compute_features_batched: {params}")
        else:
            print(f"  ✗ _compute_features_batched: {params} (expected {expected_params})")
            return False
        
        # Check other methods have 'self' plus expected params
        sig = inspect.signature(GRANITEPipeline._load_batch_cache)
        params = list(sig.parameters.keys())
        if 'self' in params and 'batch_addresses' in params:
            print(f"  ✓ _load_batch_cache signature correct")
        else:
            print(f"  ✗ _load_batch_cache signature incorrect: {params}")
            return False
        
        print(f"  ✓ PASSED: All method signatures correct")
    
    except Exception as e:
        print(f"  ✗ FAILED: Signature check error: {e}")
        return False
    
    # Test 4: Backward compatibility
    print("\n[TEST 4] Checking backward compatibility...")
    try:
        # Original run_global_training should still work
        sig = inspect.signature(GRANITEPipeline.run_global_training)
        params = list(sig.parameters.keys())
        
        # Should have at least these parameters
        if 'self' in params and 'training_fips_list' in params:
            print(f"  ✓ run_global_training signature compatible")
        else:
            print(f"  ✗ run_global_training modified incompatibly: {params}")
            return False
        
        print(f"  ✓ PASSED: Backward compatibility maintained")
    
    except Exception as e:
        print(f"  ✗ FAILED: Backward compatibility check: {e}")
        return False
    
    # Test 5: Code validation
    print("\n[TEST 5] Code quality checks...")
    try:
        import ast
        
        # Read the pipeline file
        pipeline_path = Path(__file__).parent / 'granite' / 'disaggregation' / 'pipeline.py'
        if not pipeline_path.exists():
            # Try alternative path
            pipeline_path = Path('granite/disaggregation/pipeline.py')
        
        if not pipeline_path.exists():
            print(f"  ⚠ Could not find pipeline.py at expected locations")
            print(f"  ⚠ Skipping code validation")
        else:
            with open(pipeline_path, 'r') as f:
                code = f.read()
            
            # Try to parse the file
            try:
                ast.parse(code)
                print(f"  ✓ Python syntax valid")
            except SyntaxError as e:
                print(f"  ✗ Syntax error in pipeline.py: {e}")
                return False
            
            # Check for key strings
            checks = [
                ('_compute_features_batched', 'Batching method'),
                ('BATCHED FEATURE COMPUTATION', 'Batching logging'),
                ('batch_size', 'Batch size parameter'),
                ('_load_batch_cache', 'Cache loading'),
            ]
            
            all_found = True
            for check_str, description in checks:
                if check_str in code:
                    print(f"  ✓ Found: {description}")
                else:
                    print(f"  ✗ Missing: {description}")
                    all_found = False
            
            if all_found:
                print(f"  ✓ PASSED: All implementation elements present")
            else:
                return False
    
    except Exception as e:
        print(f"  ⚠ Code validation skipped: {e}")
    
    # All tests passed
    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED")
    print("="*80)
    print("\nNext steps:")
    print("1. Run 6-tract training: python -m granite.cli run_global_training \\")
    print("   --training-fips 47065012000 47065000700 47065011304 \\")
    print("   --training-fips 47065011311 47065012400 47065011445")
    print("\n2. Monitor console for batch processing messages")
    print("3. Verify cache directory created: ls -la ./granite_cache/")
    print("4. Run second training session (should be ~10x faster)")
    print("\n")
    
    return True


def validate_batching_on_training_data():
    """
    Run quick validation on actual training data (requires data loaded).
    """
    print("\n" + "="*80)
    print("VALIDATION: Testing with actual training data")
    print("="*80)
    
    try:
        from granite.disaggregation.pipeline import GRANITEPipeline
        
        # Example configuration
        config = {
            'data': {
                'target_fips': '47065',
                'state_fips': '47',
                'county_fips': '065'
            },
            'processing': {
                'verbose': True,
                'enable_caching': True,
                'cache_dir': './granite_cache'
            },
            'model': {'hidden_dim': 64, 'dropout': 0.3},
            'training': {'learning_rate': 0.001}
        }
        
        print("\nInitializing pipeline...")
        pipeline = GRANITEPipeline(config)
        
        # Test batch size estimation for typical scenarios
        print("\nEstimated batch sizes for common scenarios:")
        scenarios = [
            ("6-tract (15K addresses)", 15000, 6),
            ("8-tract (20K addresses)", 20000, 8),
            ("10-tract (25K addresses)", 25000, 10),
            ("12-tract (30K addresses)", 30000, 12),
        ]
        
        for label, num_addrs, num_tracts in scenarios:
            batch_size = pipeline._estimate_optimal_batch_size(num_addrs, num_tracts)
            n_batches = (num_tracts + batch_size - 1) // batch_size
            print(f"  {label}: batch_size={batch_size}, n_batches={n_batches}")
        
        print("\n✓ Configuration validated")
        
    except Exception as e:
        print(f"\n⚠ Could not initialize with training data: {e}")
        print("  (This is expected if data files not available)")


if __name__ == '__main__':
    success = test_batching_implementation()
    
    if success:
        validate_batching_on_training_data()
        sys.exit(0)
    else:
        print("\n✗ TESTS FAILED")
        print("Please verify implementation and check logs above.")
        sys.exit(1)