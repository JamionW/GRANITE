#!/usr/bin/env python3
"""
GRANITE Debug Script: Diagnose shape mismatches in hybrid accessibility-SVI pipeline
Run this to systematically identify the graph node vs address mapping issue
"""

import numpy as np
import torch
import sys
import os

def debug_shape_mismatch():
    """
    Debug the 3098 vs 2394 shape mismatch systematically
    """
    print("🔍 GRANITE SHAPE MISMATCH DIAGNOSIS")
    print("=" * 50)
    
    # Test 1: Verify the core issue
    print("\n1. REPRODUCING THE ISSUE")
    print("-" * 25)
    
    # Simulate the exact shapes from your log
    graph_nodes = 3098
    address_count = 2394
    feature_dim = 9
    
    print(f"Graph total nodes: {graph_nodes}")
    print(f"Address points: {address_count}")
    print(f"Missing nodes: {graph_nodes - address_count} (road network nodes)")
    
    # Simulate the prediction tensor shape mismatch
    predictions_all_nodes = torch.randn(graph_nodes, feature_dim)  # All graph nodes
    targets_addresses_only = torch.randn(address_count, feature_dim)  # Address targets only
    
    print(f"\nPrediction shape: {predictions_all_nodes.shape}")
    print(f"Target shape: {targets_addresses_only.shape}")
    print(f"Shape mismatch: {predictions_all_nodes.shape != targets_addresses_only.shape}")
    
    # Test 2: Identify the node mapping structure
    print("\n2. NODE MAPPING ANALYSIS")
    print("-" * 25)
    
    # Simulate node mapping - addresses should be first N nodes
    # This is the typical structure in graph construction
    address_node_indices = list(range(address_count))  # 0 to 2393
    road_node_indices = list(range(address_count, graph_nodes))  # 2394 to 3097
    
    print(f"Address node indices: {address_node_indices[:5]}...{address_node_indices[-5:]}")
    print(f"Road node indices: {road_node_indices[:5]}...{road_node_indices[-5:]}")
    print(f"Address nodes range: [0, {max(address_node_indices)}]")
    print(f"Road nodes range: [{min(road_node_indices)}, {max(road_node_indices)}]")
    
    # Test 3: Fix strategy - slice predictions to match targets
    print("\n3. FIX STRATEGY TESTING")
    print("-" * 25)
    
    # Strategy A: Slice predictions to address nodes only
    predictions_addresses_only = predictions_all_nodes[:address_count]
    print(f"Strategy A - Slice first N:")
    print(f"  Sliced predictions shape: {predictions_addresses_only.shape}")
    print(f"  Matches targets: {predictions_addresses_only.shape == targets_addresses_only.shape}")
    
    # Strategy B: Use node mapping to extract address predictions
    node_mapping = {i: i for i in range(address_count)}  # Identity mapping for addresses
    mapped_predictions = predictions_all_nodes[list(node_mapping.values())]
    print(f"Strategy B - Node mapping:")
    print(f"  Mapped predictions shape: {mapped_predictions.shape}")
    print(f"  Matches targets: {mapped_predictions.shape == targets_addresses_only.shape}")
    
    # Test 4: Concatenation fix for final results
    print("\n4. CONCATENATION FIX")
    print("-" * 20)
    
    # Simulate the final concatenation error
    # Array 1: predictions for all nodes (3098,)
    # Array 2: address data (2394,)
    
    all_node_data = np.random.randn(graph_nodes)
    address_data = np.random.randn(address_count)
    
    print(f"All node data shape: {all_node_data.shape}")
    print(f"Address data shape: {address_data.shape}")
    
    try:
        # This will fail - reproducing your error
        bad_concat = np.concatenate([all_node_data, address_data])
        print("❌ This shouldn't happen - concatenation succeeded unexpectedly")
    except ValueError as e:
        print(f"✅ Reproduced error: {str(e)}")
    
    # Fix: Slice all_node_data to match address_data
    fixed_node_data = all_node_data[:address_count]
    good_concat = np.concatenate([fixed_node_data, address_data])
    print(f"✅ Fixed concatenation shape: {good_concat.shape}")
    
    return {
        'graph_nodes': graph_nodes,
        'address_count': address_count,
        'address_node_indices': address_node_indices,
        'road_node_indices': road_node_indices,
        'fix_strategy': 'slice_predictions_to_addresses'
    }

def generate_fix_code():
    """
    Generate the actual fix code for the GRANITE pipeline
    """
    print("\n5. GENERATED FIX CODE")
    print("=" * 25)
    
    fix_code = """
# CRITICAL FIX: Address the 3098 vs 2394 shape mismatch

def fix_prediction_shapes(predictions_all_nodes, num_addresses):
    '''
    Fix shape mismatch by extracting address-only predictions
    
    Args:
        predictions_all_nodes: Tensor of shape (3098, features) 
        num_addresses: int, number of actual addresses (2394)
    
    Returns:
        Tensor of shape (2394, features) matching address targets
    '''
    if predictions_all_nodes.shape[0] > num_addresses:
        # Slice to address nodes only (assumes addresses are first N nodes)
        return predictions_all_nodes[:num_addresses]
    else:
        return predictions_all_nodes

# Apply this fix in the training loop:
# Before: pred shape (3098, 9), target shape (2394, 9) -> MISMATCH
predicted_accessibility = self.model(graph_data.x, graph_data.edge_index)
predicted_accessibility_fixed = predicted_accessibility[:num_addresses]  # Slice to (2394, 9)
# After: pred shape (2394, 9), target shape (2394, 9) -> MATCH

# Apply this fix in the final concatenation:
# Before: hybrid_predictions shape (3098,), other arrays shape (2394,) -> MISMATCH
hybrid_predictions_fixed = hybrid_predictions[:len(constrained_predictions)]  # Slice to (2394,)
# After: all arrays shape (2394,) -> concatenation works
"""
    
    print(fix_code)
    return fix_code

def test_node_mapping_assumption():
    """
    Test if addresses are indeed the first N nodes in the graph
    """
    print("\n6. NODE MAPPING VALIDATION")
    print("-" * 30)
    
    # This would need to be run with actual GRANITE data
    # For now, we'll show the validation approach
    
    validation_code = """
# Validation script to run in your GRANITE environment:

python -c "
import sys
sys.path.append('.')
from granite.data.data_loader import DataLoader

# Load your actual data
loader = DataLoader()
tract_data = loader.get_tract_data('47065000600')
graph_data, node_mapping = loader.prepare_graph_data(tract_data)

print(f'Graph nodes: {graph_data.x.shape[0]}')
print(f'Addresses: {len(tract_data[\"addresses\"])}')
print(f'Node mapping type: {type(node_mapping)}')
print(f'Node mapping sample: {dict(list(node_mapping.items())[:5])}')

# Check if addresses are first N nodes
addresses = tract_data['addresses']
address_count = len(addresses)
first_n_nodes = list(range(address_count))
mapped_nodes = list(node_mapping.values())[:address_count]

print(f'First N nodes: {first_n_nodes[:5]}')
print(f'Mapped address nodes: {mapped_nodes[:5]}')
print(f'Addresses are first N nodes: {first_n_nodes == mapped_nodes}')
"
"""
    
    print("Run this validation in your environment:")
    print(validation_code)

if __name__ == "__main__":
    # Run the complete diagnosis
    debug_info = debug_shape_mismatch()
    fix_code = generate_fix_code()
    test_node_mapping_assumption()
    
    print("\n🎯 SUMMARY & NEXT STEPS")
    print("=" * 30)
    print("1. Issue: GNN predicts on all 3098 graph nodes, targets exist for 2394 addresses only")
    print("2. Root cause: Graph includes road network nodes beyond address points") 
    print("3. Fix: Slice predictions to address-only nodes before loss computation")
    print("4. Apply fix in: training loop, prediction extraction, final concatenation")
    print("5. Validate with the node mapping test above")