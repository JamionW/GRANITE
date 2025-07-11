#!/usr/bin/env python
"""
Verify clean code implementation
Checks for removed defaults and proper structure
"""
import ast
import os
import sys


def check_no_default_configs(filepath):
    """Check that Python files don't contain default configurations"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Parse AST
    tree = ast.parse(content)
    
    # Look for _default_config methods
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name == '_default_config' or node.name == 'default_config':
                return False, f"Found {node.name} method"
    
    # Look for hardcoded config dictionaries
    suspicious_patterns = [
        "'state_fips': '47'",
        '"state_fips": "47"',
        "'epochs': 100",
        '"epochs": 100',
        "'hidden_dim': 64",
        '"hidden_dim": 64'
    ]
    
    for pattern in suspicious_patterns:
        if pattern in content and 'config.yaml' not in content:
            return False, f"Found hardcoded config: {pattern}"
    
    return True, "No default configs found"


def check_spatial_disaggregation(filepath):
    """Check that interface implements spatial disaggregation"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    required_functions = [
        'create_spde_model',
        'disaggregate_with_constraint',
        '_kriging_baseline'
    ]
    
    missing = []
    for func in required_functions:
        if func not in content:
            missing.append(func)
    
    if missing:
        return False, f"Missing functions: {', '.join(missing)}"
    
    # Check for old regression functions
    regression_functions = [
        'fit_whittle_matern',
        'graph_lme'
    ]
    
    found = []
    for func in regression_functions:
        if func in content and 'create_spde_model' not in content:
            found.append(func)
    
    if found:
        return False, f"Found regression functions: {', '.join(found)}"
    
    return True, "Spatial disaggregation properly implemented"


def check_visualization_comparison(filepath):
    """Check that visualization supports comparison"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    required_methods = [
        '_plot_difference_map',
        '_plot_method_comparison',
        '_plot_uncertainty_comparison',
        '_plot_comparison_summary'
    ]
    
    missing = []
    for method in required_methods:
        if method not in content:
            missing.append(method)
    
    if missing:
        return False, f"Missing comparison methods: {', '.join(missing)}"
    
    # Check for comparison_results parameter
    if 'comparison_results' not in content:
        return False, "No comparison_results parameter in visualization"
    
    return True, "Visualization supports comparison"


def verify_clean_implementation():
    """Run all verification checks"""
    print("="*60)
    print("GRANITE Clean Code Verification")
    print("="*60)
    
    # Define files to check
    checks = [
        {
            'file': 'granite/disaggregation/pipeline.py',
            'check': check_no_default_configs,
            'description': 'Pipeline has no default configs'
        },
        {
            'file': 'granite/metricgraph/interface.py',
            'check': check_spatial_disaggregation,
            'description': 'Interface implements spatial disaggregation'
        },
        {
            'file': 'granite/visualization/plots.py',
            'check': check_visualization_comparison,
            'description': 'Visualization supports comparison'
        }
    ]
    
    all_passed = True
    
    for check_item in checks:
        filepath = check_item['file']
        check_func = check_item['check']
        description = check_item['description']
        
        print(f"\nChecking: {description}")
        print(f"  File: {filepath}")
        
        if not os.path.exists(filepath):
            print(f"  ✗ File not found")
            all_passed = False
            continue
        
        passed, message = check_func(filepath)
        
        if passed:
            print(f"  ✓ {message}")
        else:
            print(f"  ✗ {message}")
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All checks passed! Clean implementation verified.")
    else:
        print("✗ Some checks failed. Review implementation.")
    print("="*60)
    
    return all_passed


def check_config_usage():
    """Additional check for proper config usage"""
    print("\nChecking configuration usage patterns...")
    
    # Check that pipeline requires config
    pipeline_path = 'granite/disaggregation/pipeline.py'
    if os.path.exists(pipeline_path):
        with open(pipeline_path, 'r') as f:
            content = f.read()
        
        if 'config is None' in content and 'raise ValueError' in content:
            print("  ✓ Pipeline requires configuration")
        else:
            print("  ✗ Pipeline should require configuration")
    
    # Check for config.yaml
    config_path = 'config/config.yaml'
    if os.path.exists(config_path):
        print("  ✓ config.yaml exists")
    else:
        print("  ✗ config.yaml not found")


if __name__ == "__main__":
    # Run verification
    passed = verify_clean_implementation()
    
    # Additional checks
    check_config_usage()
    
    # Exit with appropriate code
    sys.exit(0 if passed else 1)