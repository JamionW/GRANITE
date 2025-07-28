#!/usr/bin/env python3
"""
Comprehensive Spatial Smoothing Configuration Debug Script
Searches codebase for spatial_weight, smoothing parameters, and loss function issues

This script will:
1. Find all spatial_weight parameter references
2. Identify GNN loss function implementations
3. Check config loading and parameter passing
4. Find hardcoded spatial parameters
5. Analyze MetricGraph interface parameter flow
6. Detect any parameter override issues
"""

import os
import re
import glob
from pathlib import Path
from collections import defaultdict

def find_spatial_smoothing_issues(root_dir="."):
    """
    Comprehensive analysis of spatial smoothing configuration issues
    """
    
    print("🔍 SPATIAL SMOOTHING CONFIGURATION DEBUG")
    print("=" * 80)
    print(f"📁 Analyzing directory: {os.path.abspath(root_dir)}")
    print("=" * 80)
    
    results = {
        'spatial_weight_refs': [],
        'loss_functions': [],
        'config_loading': [],
        'hardcoded_params': [],
        'metricgraph_interfaces': [],
        'gnn_training': [],
        'parameter_passing': [],
        'suspicious_patterns': []
    }
    
    # Define search patterns
    patterns = {
        'spatial_weight': [
            r'spatial_weight\s*=',
            r'spatial_weight\s*:',
            r'spatial_weight.*[0-9.]+',
            r'["\']spatial_weight["\']',
            r'config.*spatial_weight',
            r'self\.spatial_weight'
        ],
        'loss_functions': [
            r'def.*loss.*\(',
            r'class.*Loss.*\(',
            r'loss_function',
            r'spatial.*loss',
            r'smoothness.*loss',
            r'regularization.*loss',
            r'\.backward\(',
            r'loss\.item\('
        ],
        'config_loading': [
            r'config\[.*gnn.*\]',
            r'config\.get\(',
            r'yaml\.load',
            r'yaml\.safe_load',
            r'ConfigParser',
            r'load_config',
            r'parse_config'
        ],
        'hardcoded_params': [
            r'0\.[0-9]+.*#.*spatial',
            r'0\.[0-9]+.*#.*smooth',
            r'spatial.*=.*0\.[0-9]+',
            r'smooth.*=.*0\.[0-9]+',
            r'weight.*=.*0\.[1-9]',  # Looking for hardcoded weights > 0.1
            r'alpha.*=.*[0-9.]+',
            r'beta.*=.*[0-9.]+',
            r'lambda.*=.*[0-9.]+'
        ],
        'metricgraph_interfaces': [
            r'class.*MetricGraph',
            r'def.*metricgraph',
            r'mg_interface',
            r'metric_graph',
            r'disaggregate_svi',
            r'spatial.*parameter',
            r'whittle.*matérn',
            r'spde'
        ],
        'gnn_training': [
            r'class.*GNN',
            r'def.*train.*gnn',
            r'torch\.optim',
            r'model\.train\(',
            r'optimizer\.step',
            r'loss\.backward',
            r'train_gnn',
            r'gnn_model'
        ],
        'parameter_passing': [
            r'\*\*kwargs',
            r'\*args',
            r'def.*\(.*config.*\)',
            r'\.update\(',
            r'dict\(',
            r'params\[',
            r'parameters\[',
            r'settings\['
        ]
    }
    
    # Find all Python files
    python_files = []
    for ext in ['*.py', '*.yaml', '*.yml', '*.json']:
        python_files.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
    
    print(f"📄 Found {len(python_files)} files to analyze\n")
    
    # Analyze each file
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
            rel_path = os.path.relpath(file_path, root_dir)
            
            # Search for each pattern category
            for category, pattern_list in patterns.items():
                for pattern in pattern_list:
                    matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        line_content = lines[line_num - 1].strip()
                        
                        results[category].append({
                            'file': rel_path,
                            'line': line_num,
                            'pattern': pattern,
                            'content': line_content,
                            'match': match.group()
                        })
        
        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {str(e)}")
    
    # Analyze results and identify issues
    analyze_spatial_smoothing_issues(results)
    
    return results

def analyze_spatial_smoothing_issues(results):
    """
    Analyze the grep results to identify specific spatial smoothing issues
    """
    
    print("\n🎯 SPATIAL_WEIGHT PARAMETER ANALYSIS")
    print("-" * 60)
    
    spatial_refs = results['spatial_weight_refs']
    if spatial_refs:
        print(f"Found {len(spatial_refs)} spatial_weight references:")
        
        # Group by file
        by_file = defaultdict(list)
        for ref in spatial_refs:
            by_file[ref['file']].append(ref)
        
        for file_path, refs in by_file.items():
            print(f"\n📄 {file_path}:")
            for ref in refs:
                print(f"   Line {ref['line']:3d}: {ref['content']}")
                
                # Check for potential issues
                content = ref['content'].lower()
                if 'spatial_weight' in content:
                    if '0.5' in content or '0.6' in content or '0.7' in content:
                        print(f"      🚨 HIGH spatial_weight detected (>0.5)")
                    elif '0.1' in content:
                        print(f"      ✅ Reasonable spatial_weight (0.1)")
                    elif 'config' not in content:
                        print(f"      ⚠️  Hardcoded spatial_weight (not from config)")
    else:
        print("❌ NO spatial_weight references found - this could be the problem!")
    
    print("\n🧠 GNN LOSS FUNCTION ANALYSIS")
    print("-" * 60)
    
    loss_funcs = results['loss_functions']
    if loss_funcs:
        print(f"Found {len(loss_funcs)} loss function references:")
        
        # Look for spatial loss components
        spatial_loss_found = False
        for loss in loss_funcs:
            content = loss['content'].lower()
            if any(word in content for word in ['spatial', 'smooth', 'regulariz']):
                print(f"📄 {loss['file']}:{loss['line']} - {loss['content']}")
                spatial_loss_found = True
        
        if not spatial_loss_found:
            print("🚨 NO spatial loss components found in loss functions!")
            print("   This could explain why spatial_weight has no effect")
    else:
        print("❌ NO loss functions found")
    
    print("\n⚙️ CONFIG LOADING ANALYSIS")
    print("-" * 60)
    
    config_refs = results['config_loading']
    if config_refs:
        print(f"Found {len(config_refs)} config loading references:")
        
        # Look for GNN config access
        gnn_config_found = False
        for config in config_refs:
            content = config['content'].lower()
            if 'gnn' in content or 'spatial_weight' in content:
                print(f"📄 {config['file']}:{config['line']} - {config['content']}")
                gnn_config_found = True
        
        if not gnn_config_found:
            print("⚠️  No GNN-specific config loading found")
    else:
        print("❌ NO config loading found")
    
    print("\n🔧 HARDCODED PARAMETER ANALYSIS")
    print("-" * 60)
    
    hardcoded = results['hardcoded_params']
    if hardcoded:
        print(f"Found {len(hardcoded)} potentially hardcoded parameters:")
        
        for param in hardcoded:
            print(f"📄 {param['file']}:{param['line']} - {param['content']}")
            
            # Check if this might override config
            content = param['content'].lower()
            if any(word in content for word in ['=', 'weight', 'spatial', 'smooth']):
                print(f"   🚨 May override config spatial_weight!")
    else:
        print("✅ No obviously hardcoded spatial parameters")
    
    print("\n📊 METRICGRAPH INTERFACE ANALYSIS")
    print("-" * 60)
    
    mg_refs = results['metricgraph_interfaces']
    if mg_refs:
        print(f"Found {len(mg_refs)} MetricGraph interface references:")
        
        # Look for parameter passing to MetricGraph
        for mg in mg_refs[:10]:  # Show first 10
            print(f"📄 {mg['file']}:{mg['line']} - {mg['content']}")
    else:
        print("❌ NO MetricGraph interfaces found")
    
    print("\n🤖 GNN TRAINING ANALYSIS")
    print("-" * 60)
    
    gnn_training = results['gnn_training']
    if gnn_training:
        print(f"Found {len(gnn_training)} GNN training references:")
        
        # Look for training loops and parameter usage
        for train in gnn_training[:5]:  # Show first 5
            print(f"📄 {train['file']}:{train['line']} - {train['content']}")
    else:
        print("❌ NO GNN training code found")
    
    # CRITICAL ISSUE DETECTION
    print("\n🚨 CRITICAL ISSUE DETECTION")
    print("=" * 60)
    
    issues_found = []
    
    # Issue 1: No spatial_weight references
    if len(spatial_refs) == 0:
        issues_found.append("CRITICAL: No spatial_weight parameter found anywhere!")
    
    # Issue 2: spatial_weight found but not in loss functions
    elif len(spatial_refs) > 0 and len([l for l in loss_funcs if 'spatial' in l['content'].lower()]) == 0:
        issues_found.append("CRITICAL: spatial_weight parameter exists but no spatial loss functions!")
    
    # Issue 3: High spatial_weight values
    high_spatial_weights = [r for r in spatial_refs if any(val in r['content'] for val in ['0.5', '0.6', '0.7', '0.8', '0.9'])]
    if high_spatial_weights:
        issues_found.append(f"WARNING: {len(high_spatial_weights)} high spatial_weight values (>0.5) found")
    
    # Issue 4: Hardcoded overrides
    spatial_overrides = [h for h in hardcoded if 'spatial' in h['content'].lower() or 'smooth' in h['content'].lower()]
    if spatial_overrides:
        issues_found.append(f"WARNING: {len(spatial_overrides)} potential hardcoded spatial parameter overrides")
    
    # Issue 5: No config loading
    if len(config_refs) == 0:
        issues_found.append("CRITICAL: No config loading mechanism found!")
    
    if issues_found:
        print("🔴 ISSUES DETECTED:")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
    else:
        print("✅ No critical configuration issues detected")
    
    # SPECIFIC DEBUGGING RECOMMENDATIONS
    print("\n🛠️ DEBUGGING RECOMMENDATIONS")
    print("=" * 60)
    
    recommendations = []
    
    if len(spatial_refs) == 0:
        recommendations.append("1. Search for 'spatial_weight' manually - parameter may be named differently")
        recommendations.append("2. Check if spatial smoothness is implemented differently (e.g., 'regularization_weight')")
    
    if len(spatial_refs) > 0:
        # Find the most likely config file
        config_files = [r['file'] for r in spatial_refs if any(ext in r['file'] for ext in ['.yaml', '.yml', '.json'])]
        if config_files:
            recommendations.append(f"3. Verify spatial_weight value in config file: {config_files[0]}")
        
        # Find the most likely implementation file
        impl_files = [r['file'] for r in spatial_refs if r['file'].endswith('.py')]
        if impl_files:
            recommendations.append(f"4. Check spatial_weight usage in: {impl_files[0]}")
    
    recommendations.extend([
        "5. Add debug prints: print(f'spatial_weight = {spatial_weight}') in loss function",
        "6. Verify loss function actually uses spatial_weight parameter",
        "7. Check if GNN loss = task_loss + spatial_weight * spatial_loss",
        "8. Ensure config changes are reloaded (restart training after config changes)"
    ])
    
    for rec in recommendations:
        print(f"   {rec}")

def generate_debug_commands(results):
    """
    Generate specific grep commands for further investigation
    """
    
    print("\n💻 SPECIFIC DEBUG COMMANDS TO RUN")
    print("=" * 60)
    
    commands = [
        "# Find all spatial_weight references",
        "grep -r -n 'spatial_weight' . --include='*.py' --include='*.yaml'",
        "",
        "# Find loss function definitions", 
        "grep -r -n 'def.*loss' . --include='*.py'",
        "",
        "# Find spatial loss components",
        "grep -r -n -i 'spatial.*loss\\|smooth.*loss\\|regulariz.*loss' . --include='*.py'",
        "",
        "# Find config loading",
        "grep -r -n 'config\\[.*gnn.*\\]\\|spatial_weight' . --include='*.py'",
        "",
        "# Find hardcoded spatial parameters", 
        "grep -r -n '0\\.[0-9].*spatial\\|spatial.*0\\.[0-9]' . --include='*.py'",
        "",
        "# Find MetricGraph parameter passing",
        "grep -r -n -A 5 -B 5 'disaggregate_svi\\|mg_interface' . --include='*.py'",
        "",
        "# Check for parameter override patterns",
        "grep -r -n 'spatial_weight.*=' . --include='*.py' | grep -v 'config'"
    ]
    
    for cmd in commands:
        print(cmd)

if __name__ == "__main__":
    
    # Run the analysis
    results = find_spatial_smoothing_issues()
    
    # Generate additional debug commands
    generate_debug_commands(results)
    
    print(f"\n📋 Analysis complete. Check the results above for spatial smoothing configuration issues.")
    print(f"💡 Focus on files with spatial_weight references and loss function implementations.")