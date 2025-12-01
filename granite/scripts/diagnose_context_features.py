#!/usr/bin/env python3
"""
Diagnostic: What's actually happening with context features?
"""

import sys
sys.path.insert(0, '/workspaces/GRANITE')

print("="*70)
print("GRANITE CONTEXT FEATURE DIAGNOSTIC")
print("="*70)

# Test 1: Check the default parameter
print("\n1. Checking default parameter value...")
from granite.data.loaders import DataLoader
import inspect

sig = inspect.signature(DataLoader.create_context_features_for_addresses)
default = sig.parameters['include_tract_svi'].default

print(f"   include_tract_svi default: {default}")
print(f"   Expected: False")
print(f"   Status: {'✓ CORRECT' if default == False else '✗ WRONG'}")

# Test 2: Check what the method actually creates
print("\n2. Testing actual context feature creation...")
import geopandas as gpd
import pandas as pd
import numpy as np

# Create minimal test data
addresses = gpd.GeoDataFrame({
    'tract_fips': ['47065010700'] * 5,
    'geometry': [None] * 5
})

svi_data = pd.DataFrame({
    'FIPS': ['47065010700'],
    'SVI': [0.5],
    'pct_no_vehicle': [10.0],
    'pct_poverty': [15.0],
    'pct_unemployed': [5.0],
    'pct_no_hs_diploma': [8.0],
    'population': [5000]
})

loader = DataLoader()

print("\n   Testing with default (include_tract_svi not specified):")
context_default = loader.create_context_features_for_addresses(addresses, svi_data)
print(f"   Created shape: {context_default.shape}")
print(f"   Expected: (5, 5)")
print(f"   Status: {'✓ CORRECT' if context_default.shape[1] == 5 else '✗ WRONG - Got ' + str(context_default.shape[1]) + ' dims'}")

print("\n   Testing with include_tract_svi=False (explicit):")
context_false = loader.create_context_features_for_addresses(addresses, svi_data, include_tract_svi=False)
print(f"   Created shape: {context_false.shape}")
print(f"   Expected: (5, 5)")
print(f"   Status: {'✓ CORRECT' if context_false.shape[1] == 5 else '✗ WRONG - Got ' + str(context_false.shape[1]) + ' dims'}")

print("\n   Testing with include_tract_svi=True (explicit):")
context_true = loader.create_context_features_for_addresses(addresses, svi_data, include_tract_svi=True)
print(f"   Created shape: {context_true.shape}")
print(f"   Expected: (5, 6) if working, (5, 5) if broken")
print(f"   Actual: {context_true.shape}")

if context_true.shape[1] == 5:
    print("   ✗ BUG CONFIRMED: include_tract_svi=True branch doesn't add tract_svi!")
elif context_true.shape[1] == 6:
    print("   ✓ include_tract_svi=True branch works correctly (adds tract_svi)")
else:
    print(f"   ??? Unexpected dimension: {context_true.shape[1]}")

# Test 3: Check if they're actually the same
print("\n3. Comparing False vs default:")
if np.array_equal(context_default, context_false):
    print("   ✓ Default uses include_tract_svi=False as expected")
else:
    print("   ✗ Default is NOT using include_tract_svi=False!")

# Test 4: Inspect the actual code
print("\n4. Inspecting deployed method code...")
import textwrap
source = inspect.getsource(DataLoader.create_context_features_for_addresses)

# Check for the if include_tract_svi line
if_line = [line for line in source.split('\n') if 'if include_tract_svi:' in line]
if if_line:
    print(f"   Found: {if_line[0].strip()}")
    
    # Get the next few lines after "if include_tract_svi:"
    lines = source.split('\n')
    for i, line in enumerate(lines):
        if 'if include_tract_svi:' in line:
            print("\n   Lines after 'if include_tract_svi:':")
            for j in range(i+1, min(i+10, len(lines))):
                if lines[j].strip() and not lines[j].strip().startswith('#'):
                    print(f"      {lines[j]}")
                    if 'else:' in lines[j]:
                        break
            break

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)