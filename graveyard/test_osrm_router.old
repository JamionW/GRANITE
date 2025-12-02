"""
Simple test to verify OSRMRouter actually calls OSRM servers
Run from workspace root: python test_osrm_router.py
"""
import sys
sys.path.insert(0, 'granite')

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from data.osrm_router import OSRMRouter

print("="*60)
print("OSRM Router Test - Verify Real Routing")
print("="*60)

# Create test origins (2 addresses in Chattanooga)
origins = gpd.GeoDataFrame({
    'address_id': [0, 1],
    'geometry': [
        Point(-85.3, 35.05),  # Downtown Chattanooga area
        Point(-85.25, 35.08)  # North of downtown
    ]
}, crs='EPSG:4326')

# Create test destinations (2 locations)
destinations = gpd.GeoDataFrame({
    'dest_id': [0, 1],
    'geometry': [
        Point(-85.28, 35.06),  # ~5km east
        Point(-85.32, 35.10)   # ~5km north-west
    ]
}, crs='EPSG:4326')

print(f"\nTest setup:")
print(f"  {len(origins)} origins × {len(destinations)} destinations = {len(origins) * len(destinations)} routes")
print()

# Initialize router
router = OSRMRouter(verbose=True)

# Compute travel times
results = router.compute_multimodal_travel_times(origins, destinations)

print("\n" + "="*60)
print("Results:")
print("="*60)
print(results.to_string(index=False))

print("\n" + "="*60)
print("Validation:")
print("="*60)

# Check if we have any valid results
has_valid_walk = results['walk_time'].notna().any()
has_valid_drive = results['drive_time'].notna().any()

if not has_valid_walk and not has_valid_drive:
    print("✗ FAIL: All results are NaN (routing failed)")
    print("  Possible causes:")
    print("    - Test coordinates are outside the Tennessee network")
    print("    - OSRM servers aren't processing requests correctly")
    print("    - Coordinate snapping failed")
    sys.exit(1)

if not has_valid_walk:
    print("✗ WARNING: All walk times are NaN")
elif not has_valid_drive:
    print("✗ WARNING: All drive times are NaN")

# Check if walk and drive times are different (for valid results)
valid_results = results[results['walk_time'].notna() & results['drive_time'].notna()]

if len(valid_results) == 0:
    print("✗ FAIL: No valid route pairs to compare")
    sys.exit(1)

walk_drive_equal = (valid_results['walk_time'] == valid_results['drive_time']).all()

if walk_drive_equal:
    print("✗ FAIL: All walk times equal drive times (fallback mode)")
    print("  This means OSRM is not being called correctly")
    sys.exit(1)
else:
    print("✓ PASS: Walk and drive times differ (real routing)")
    print("  OSRM servers are being called successfully")
    print(f"  Valid routes: {len(valid_results)}/{len(results)}")
    
    # Show the differences
    valid_results_copy = valid_results.copy()
    valid_results_copy['time_diff'] = abs(valid_results_copy['walk_time'] - valid_results_copy['drive_time'])
    avg_diff = valid_results_copy['time_diff'].mean()
    print(f"  Average time difference: {avg_diff:.2f} minutes")
    
    sys.exit(0)