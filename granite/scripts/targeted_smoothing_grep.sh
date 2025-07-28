#!/bin/bash

# Targeted Spatial Smoothing Debug Script
# Focuses on the most likely causes of spatial_weight not taking effect

echo "🎯 TARGETED SPATIAL SMOOTHING DEBUG"
echo "=================================="

echo ""
echo "1. 🔍 FINDING ALL SPATIAL_WEIGHT REFERENCES:"
echo "--------------------------------------------"
echo "Files containing 'spatial_weight':"
grep -r -l "spatial_weight" . --include="*.py" --include="*.yaml" --include="*.yml" 2>/dev/null || echo "❌ No spatial_weight found!"

echo ""
echo "All spatial_weight lines with context:"
grep -r -n -C 2 "spatial_weight" . --include="*.py" --include="*.yaml" --include="*.yml" 2>/dev/null || echo "❌ No spatial_weight references!"

echo ""
echo "2. 🧠 LOSS FUNCTION IMPLEMENTATIONS:"
echo "-----------------------------------"
echo "Loss function definitions:"
grep -r -n "def.*loss" . --include="*.py" | head -10

echo ""
echo "Loss function calls with spatial terms:"
grep -r -n -i "spatial.*loss\|smooth.*loss\|regularization" . --include="*.py" | head -10

echo ""
echo "3. ⚙️ CONFIG LOADING AND GNN PARAMETERS:"
echo "---------------------------------------"
echo "GNN config access patterns:"
grep -r -n "config.*gnn\|gnn.*config" . --include="*.py" | head -10

echo ""
echo "Config file spatial_weight values:"
grep -r -n "spatial_weight.*:" . --include="*.yaml" --include="*.yml"

echo ""
echo "4. 🚨 POTENTIAL OVERRIDES AND HARDCODED VALUES:"
echo "----------------------------------------------"
echo "Hardcoded spatial/smooth parameters:"
grep -r -n "spatial.*=.*0\.\|smooth.*=.*0\.\|weight.*=.*0\.[1-9]" . --include="*.py" | head -10

echo ""
echo "Parameter assignments that might override config:"
grep -r -n "spatial_weight.*=" . --include="*.py" | grep -v "config\|yaml"

echo ""
echo "5. 🔗 METRICGRAPH AND SPDE PARAMETER PASSING:"
echo "--------------------------------------------"
echo "MetricGraph interface calls:"
grep -r -n -A 3 -B 3 "disaggregate_svi\|mg_interface\|MetricGraph" . --include="*.py" | head -20

echo ""
echo "SPDE parameter passing:"
grep -r -n "alpha\|kappa\|tau\|sigma" . --include="*.py" | grep -E "(=|:)" | head -10

echo ""
echo "6. 🤖 GNN TRAINING AND MODEL CONFIGURATION:"
echo "------------------------------------------"
echo "GNN training function definitions:"
grep -r -n "def.*train.*gnn\|class.*GNN" . --include="*.py"

echo ""
echo "PyTorch loss backward calls:"
grep -r -n "loss\.backward\|\.backward()" . --include="*.py" | head -5

echo ""
echo "7. 🔧 CONFIGURATION FILE ANALYSIS:"
echo "---------------------------------"
echo "YAML/YML files in project:"
find . -name "*.yaml" -o -name "*.yml" 2>/dev/null

echo ""
echo "Config files with GNN sections:"
grep -l "gnn:\|GNN:" . -r --include="*.yaml" --include="*.yml" 2>/dev/null

echo ""
echo "8. 🎯 CRITICAL PATTERN DETECTION:"
echo "--------------------------------"

# Check for common issues
echo "Checking for critical issues..."

# Issue 1: Config not being loaded
config_loading=$(grep -r "yaml.load\|yaml.safe_load\|load_config" . --include="*.py" 2>/dev/null | wc -l)
if [ $config_loading -eq 0 ]; then
    echo "🚨 CRITICAL: No YAML config loading found!"
else
    echo "✅ Config loading detected ($config_loading references)"
fi

# Issue 2: spatial_weight parameter not used in loss
spatial_in_loss=$(grep -r "spatial_weight" . --include="*.py" 2>/dev/null | grep -i "loss" | wc -l)
if [ $spatial_in_loss -eq 0 ]; then
    echo "🚨 CRITICAL: spatial_weight not used in any loss function!"
else
    echo "✅ spatial_weight used in loss functions ($spatial_in_loss references)"
fi

# Issue 3: High spatial_weight values
high_spatial=$(grep -r "spatial_weight.*0\.[5-9]\|spatial_weight.*1\." . --include="*.py" --include="*.yaml" 2>/dev/null | wc -l)
if [ $high_spatial -gt 0 ]; then
    echo "⚠️  WARNING: High spatial_weight values detected ($high_spatial instances)"
    grep -r "spatial_weight.*0\.[5-9]\|spatial_weight.*1\." . --include="*.py" --include="*.yaml" 2>/dev/null
else
    echo "✅ No excessively high spatial_weight values found"
fi

# Issue 4: Multiple spatial_weight definitions
multiple_spatial=$(grep -r "spatial_weight.*=" . --include="*.py" 2>/dev/null | wc -l)
if [ $multiple_spatial -gt 1 ]; then
    echo "⚠️  WARNING: Multiple spatial_weight assignments found ($multiple_spatial instances)"
    echo "Check for conflicts:"
    grep -r "spatial_weight.*=" . --include="*.py" 2>/dev/null
else
    echo "✅ Single or no spatial_weight assignments found"
fi

echo ""
echo "9. 🛠️ SPECIFIC FILES TO INVESTIGATE:"
echo "----------------------------------"

# Find the most important files to check
echo "Most likely files containing spatial smoothing logic:"

# Config files
echo "📄 Config files:"
find . -name "*.yaml" -o -name "*.yml" | head -5

# GNN files
echo "📄 GNN implementation files:"
find . -name "*gnn*" -type f | head -5

# Training files  
echo "📄 Training/pipeline files:"
find . -name "*train*" -o -name "*pipeline*" | grep "\.py$" | head -5

# MetricGraph files
echo "📄 MetricGraph interface files:"
find . -name "*metric*" -o -name "*mg*" | grep "\.py$" | head -5

echo ""
echo "10. 📋 QUICK INVESTIGATION CHECKLIST:"
echo "-----------------------------------"
echo "□ 1. Check if spatial_weight appears in any Python files"
echo "□ 2. Verify spatial_weight is used in loss function calculation"  
echo "□ 3. Confirm config file contains spatial_weight parameter"
echo "□ 4. Check if config changes require restarting training"
echo "□ 5. Look for hardcoded spatial parameters overriding config"
echo "□ 6. Verify GNN loss = task_loss + spatial_weight * spatial_loss"
echo "□ 7. Add debug print: print(f'Using spatial_weight: {spatial_weight}')"
echo "□ 8. Test with extreme values (0.001 and 0.999) to confirm effect"

echo ""
echo "✅ Targeted analysis complete!"
echo "💡 Focus on files with spatial_weight references and loss implementations"