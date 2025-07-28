#!/usr/bin/env python3
"""
Automatic Spatial Smoothing Issue Detection
Automatically identifies the most likely cause of spatial_weight not working

This script specifically looks for the TOP 5 most common issues:
1. spatial_weight parameter not found anywhere
2. spatial_weight found but not used in loss functions  
3. Config loading issues
4. Hardcoded parameter overrides
5. GNN training not using spatial loss
"""

import os
import re
import glob
import yaml
from pathlib import Path

def detect_spatial_smoothing_issues():
    """
    Automatically detect and rank spatial smoothing configuration issues
    """
    
    print("🔍 AUTOMATIC SPATIAL SMOOTHING ISSUE DETECTION")
    print("=" * 70)
    
    issues = []
    confidence_scores = {}
    
    # Get all Python and config files
    py_files = glob.glob("**/*.py", recursive=True)
    config_files = glob.glob("**/*.yaml", recursive=True) + glob.glob("**/*.yml", recursive=True)
    
    print(f"📁 Analyzing {len(py_files)} Python files and {len(config_files)} config files")
    
    # ISSUE 1: Check if spatial_weight exists anywhere
    print("\n1️⃣ CHECKING FOR SPATIAL_WEIGHT PARAMETER...")
    spatial_weight_found = False
    spatial_weight_files = []
    
    for file_path in py_files + config_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if 'spatial_weight' in content:
                    spatial_weight_found = True
                    spatial_weight_files.append(file_path)
        except:
            continue
    
    if not spatial_weight_found:
        issues.append({
            'priority': 'CRITICAL',
            'issue': 'NO spatial_weight parameter found anywhere in codebase',
            'description': 'The spatial_weight parameter does not exist in any Python or config files',
            'solution': 'Add spatial_weight parameter to config.yaml under gnn: section',
            'confidence': 95
        })
        confidence_scores['missing_parameter'] = 95
    else:
        print(f"   ✅ spatial_weight found in {len(spatial_weight_files)} files")
        confidence_scores['missing_parameter'] = 0
    
    # ISSUE 2: Check if spatial_weight is used in loss functions
    print("\n2️⃣ CHECKING SPATIAL_WEIGHT USAGE IN LOSS FUNCTIONS...")
    loss_functions = []
    spatial_in_loss = False
    
    for file_path in py_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # Find loss function definitions
                loss_matches = re.findall(r'def\s+\w*loss\w*\([^)]*\):', content, re.IGNORECASE)
                if loss_matches:
                    loss_functions.extend([(file_path, match) for match in loss_matches])
                
                # Check if spatial_weight appears in same file as loss
                if 'spatial_weight' in content and any(word in content.lower() for word in ['loss', 'backward']):
                    spatial_in_loss = True
        except:
            continue
    
    if spatial_weight_found and not spatial_in_loss:
        issues.append({
            'priority': 'CRITICAL',
            'issue': 'spatial_weight parameter exists but NOT used in loss functions',
            'description': 'spatial_weight is defined but never used in loss calculation',
            'solution': 'Modify loss function to include: total_loss = task_loss + spatial_weight * spatial_loss',
            'confidence': 90
        })
        confidence_scores['loss_usage'] = 90
    else:
        if spatial_in_loss:
            print(f"   ✅ spatial_weight appears to be used in loss functions")
        confidence_scores['loss_usage'] = 0
    
    # ISSUE 3: Check config loading
    print("\n3️⃣ CHECKING CONFIG LOADING MECHANISMS...")
    config_loading_found = False
    config_loading_methods = []
    
    for file_path in py_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                if any(pattern in content for pattern in ['yaml.load', 'yaml.safe_load', 'load_config']):
                    config_loading_found = True
                    config_loading_methods.append(file_path)
        except:
            continue
    
    if not config_loading_found:
        issues.append({
            'priority': 'HIGH',
            'issue': 'NO config loading mechanism found',
            'description': 'No YAML config loading detected in Python files',
            'solution': 'Add config loading: config = yaml.safe_load(open("config.yaml"))',
            'confidence': 85
        })
        confidence_scores['config_loading'] = 85
    else:
        print(f"   ✅ Config loading found in {len(config_loading_methods)} files")
        confidence_scores['config_loading'] = 0
    
    # ISSUE 4: Check for hardcoded overrides
    print("\n4️⃣ CHECKING FOR HARDCODED PARAMETER OVERRIDES...")
    hardcoded_overrides = []
    
    for file_path in py_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
                for i, line in enumerate(lines):
                    # Look for hardcoded spatial parameters
                    if re.search(r'spatial_weight\s*=\s*0\.[0-9]+', line) and 'config' not in line:
                        hardcoded_overrides.append((file_path, i+1, line.strip()))
                    elif re.search(r'(spatial|smooth).*=.*0\.[0-9]+', line) and 'config' not in line:
                        hardcoded_overrides.append((file_path, i+1, line.strip()))
        except:
            continue
    
    if hardcoded_overrides:
        issues.append({
            'priority': 'HIGH', 
            'issue': f'{len(hardcoded_overrides)} hardcoded spatial parameters found',
            'description': 'Hardcoded spatial parameters may override config values',
            'solution': 'Replace hardcoded values with config.get("spatial_weight")',
            'confidence': 80,
            'details': hardcoded_overrides[:3]  # Show first 3
        })
        confidence_scores['hardcoded_overrides'] = 80
    else:
        print(f"   ✅ No hardcoded spatial parameter overrides detected")
        confidence_scores['hardcoded_overrides'] = 0
    
    # ISSUE 5: Check GNN training implementation
    print("\n5️⃣ CHECKING GNN TRAINING IMPLEMENTATION...")
    gnn_training_files = []
    spatial_loss_in_training = False
    
    for file_path in py_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # Look for GNN training
                if any(pattern in content for pattern in ['class.*GNN', 'train.*gnn', 'gnn.*train']):
                    gnn_training_files.append(file_path)
                    
                    # Check if spatial loss is implemented
                    if any(term in content.lower() for term in ['spatial_loss', 'smoothness_loss', 'regularization']):
                        spatial_loss_in_training = True
        except:
            continue
    
    if gnn_training_files and not spatial_loss_in_training:
        issues.append({
            'priority': 'MEDIUM',
            'issue': 'GNN training found but no spatial loss implementation detected',
            'description': 'GNN training code exists but spatial/smoothness loss not implemented',
            'solution': 'Add spatial loss component to GNN training loop',
            'confidence': 70
        })
        confidence_scores['training_implementation'] = 70
    else:
        if spatial_loss_in_training:
            print(f"   ✅ Spatial loss appears to be implemented in training")
        confidence_scores['training_implementation'] = 0
    
    # ISSUE 6: Check config file values
    print("\n6️⃣ CHECKING CONFIG FILE VALUES...")
    config_issues = []
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
                
                # Check for spatial_weight in config
                if isinstance(config_data, dict):
                    # Navigate through config structure
                    gnn_config = config_data.get('gnn', {})
                    if isinstance(gnn_config, dict):
                        spatial_weight = gnn_config.get('spatial_weight')
                        if spatial_weight is not None:
                            if spatial_weight > 0.5:
                                config_issues.append(f"High spatial_weight value: {spatial_weight} in {config_file}")
                            elif spatial_weight <= 0:
                                config_issues.append(f"Zero/negative spatial_weight: {spatial_weight} in {config_file}")
                            else:
                                print(f"   ✅ spatial_weight = {spatial_weight} in {config_file}")
        except:
            continue
    
    if config_issues:
        issues.append({
            'priority': 'MEDIUM',
            'issue': 'Config file spatial_weight value issues',
            'description': '; '.join(config_issues),
            'solution': 'Set spatial_weight to value between 0.001 and 0.1',
            'confidence': 60
        })
    
    # RANK AND DISPLAY ISSUES
    print("\n" + "=" * 70)
    print("🚨 DETECTED ISSUES (RANKED BY PRIORITY)")
    print("=" * 70)
    
    if not issues:
        print("✅ NO CRITICAL ISSUES DETECTED!")
        print("   Your spatial smoothing configuration appears to be correct.")
        print("   The issue may be in the implementation details or parameter values.")
        return
    
    # Sort by priority and confidence
    priority_order = {'CRITICAL': 3, 'HIGH': 2, 'MEDIUM': 1}
    issues.sort(key=lambda x: (priority_order.get(x['priority'], 0), x['confidence']), reverse=True)
    
    for i, issue in enumerate(issues, 1):
        priority_emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡'}.get(issue['priority'], '⚪')
        
        print(f"\n{priority_emoji} ISSUE #{i} - {issue['priority']} (Confidence: {issue['confidence']}%)")
        print(f"   Problem: {issue['issue']}")
        print(f"   Details: {issue['description']}")
        print(f"   Solution: {issue['solution']}")
        
        if 'details' in issue:
            print(f"   Examples:")
            for detail in issue['details']:
                if isinstance(detail, tuple):
                    print(f"      {detail[0]}:{detail[1]} - {detail[2]}")
    
    # PROVIDE SPECIFIC NEXT STEPS
    print("\n" + "=" * 70)
    print("🛠️ IMMEDIATE ACTION PLAN")
    print("=" * 70)
    
    # Recommend actions based on highest confidence issues
    if issues:
        top_issue = issues[0]
        print(f"🎯 START HERE: {top_issue['solution']}")
        
        if top_issue['priority'] == 'CRITICAL':
            print(f"⚡ This is a critical issue that will prevent spatial smoothing from working entirely.")
        
        print(f"\n📋 Additional recommended actions:")
        print(f"   1. Add debug print: print(f'spatial_weight = {{spatial_weight}}')")
        print(f"   2. Test with extreme values (0.001 vs 0.999) to confirm parameter takes effect")
        print(f"   3. Check loss function output before and after adding spatial component")
        print(f"   4. Verify config changes require restarting the training process")
    
    print(f"\n📊 Issue Summary:")
    print(f"   Total issues detected: {len(issues)}")
    print(f"   Critical issues: {len([i for i in issues if i['priority'] == 'CRITICAL'])}")
    print(f"   High priority issues: {len([i for i in issues if i['priority'] == 'HIGH'])}")

if __name__ == "__main__":
    detect_spatial_smoothing_issues()