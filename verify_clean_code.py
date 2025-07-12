#!/usr/bin/env python
"""
Verify clean code implementation and check for syntax errors
"""
import ast
import os
import sys


def check_syntax_errors(filepath):
    """Check for Python syntax errors"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Check for common syntax issues
        if '*def*' in content or 'def *' in content:
            return False, "Found asterisks in function definition"
        
        if '.*' in content and 'self.*' in content:
            # Check if it's a method call with asterisks
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'self.*' in line and '*(' in line:
                    return False, f"Found asterisks in method call on line {i+1}"
        
        # Try to parse the file
        ast.parse(content)
        return True, "Syntax OK"
        
    except SyntaxError as e:
        return False, f"Syntax error on line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Error: {str(e)}"


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
    
    # First check for syntax errors
    print("\n1. Checking for syntax errors...")
    syntax_files = [
        'granite/__init__.py',
        'granite/metricgraph/__init__.py', 
        'granite/metricgraph/interface.py',
        'granite/disaggregation/pipeline.py',
        'granite/visualization/plots.py'
    ]
    
    syntax_ok = True
    for filepath in syntax_files:
        if os.path.exists(filepath):
            passed, message = check_syntax_errors(filepath)
            if passed:
                print(f"  ✓ {filepath} - {message}")
            else:
                print(f"  ✗ {filepath} - {message}")
                syntax_ok = False
        else:
            print(f"  ? {filepath} - File not found")
    
    if not syntax_ok:
        print("\n✗ Fix syntax errors before proceeding with other checks")
        return False
    
    print("\n2. Checking implementation requirements...")
    
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
        
        print(f"\n  Checking: {description}")
        print(f"    File: {filepath}")
        
        if not os.path.exists(filepath):
            print(f"    ✗ File not found")
            all_passed = False
            continue
        
        passed, message = check_func(filepath)
        
        if passed:
            print(f"    ✓ {message}")
        else:
            print(f"    ✗ {message}")
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