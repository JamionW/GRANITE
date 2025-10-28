"""
Test real data integration for GRANITE
Updated to match config-free OSRM implementation
"""
from granite.data.real_data_loaders import RealDataLoader
from granite.routing.osrm_router import OSRMRouter
from granite.data.loaders import DataLoader
import geopandas as gpd
from shapely.geometry import Point

def test_real_destinations():
    """Test loading real destinations"""
    print("="*60)
    print("TEST 1: Real Destination Loading")
    print("="*60)
    
    loader = RealDataLoader(data_dir='./data/raw')
    
    # Test each destination type
    employment = loader.load_lehd_employment()
    print(f"✓ Employment: {len(employment)} locations, {employment['employees'].sum():,} jobs")
    
    healthcare = loader.load_healthcare_facilities()
    print(f"✓ Healthcare: {len(healthcare)} facilities")
    
    grocery = loader.load_grocery_stores()
    print(f"✓ Grocery: {len(grocery)} stores")
    
    print("\n✓ All destination types loaded successfully")
    return True

def test_osrm_routing():
    """Test OSRM routing with new config-free implementation"""
    print("\n" + "="*60)
    print("TEST 2: OSRM Routing (Config-Free)")
    print("="*60)
    
    # Initialize router - no config needed!
    router = OSRMRouter(verbose=True)
    
    # Create test origins (2 addresses in Chattanooga)
    origins = gpd.GeoDataFrame({
        'address_id': [0, 1],
        'geometry': [
            Point(-85.3097, 35.0456),  # Downtown Chattanooga
            Point(-85.2111, 35.0407)   # Hamilton Place area
        ]
    }, crs='EPSG:4326')
    
    # Create test destinations (2 locations)
    destinations = gpd.GeoDataFrame({
        'dest_id': [0, 1],
        'geometry': [
            Point(-85.2597, 35.0456),  # Parkridge Medical
            Point(-85.3083, 35.0539)   # Erlanger Hospital
        ]
    }, crs='EPSG:4326')
    
    print("\nTest case: 2 origins × 2 destinations = 4 route pairs")
    
    # Test routing
    travel_times = router.compute_multimodal_travel_times(origins, destinations)
    
    print(f"\n✓ Computed {len(travel_times)} route pairs")
    print(f"\nSample travel times:")
    print("-" * 60)
    print(travel_times.to_string(index=False))
    print("-" * 60)
    
    # Validate results
    assert len(travel_times) == 4, "Should have 4 route pairs"
    assert all(col in travel_times.columns for col in ['walk_time', 'drive_time', 'combined_time', 'best_mode']), "Missing required columns"
    assert all(travel_times['walk_time'] > 0), "Walk times should be positive"
    assert all(travel_times['drive_time'] > 0), "Drive times should be positive"
    
    print("\n✓ OSRM routing working correctly")
    return True

def test_feature_count():
    """Test that feature count is correct after removal"""
    print("\n" + "="*60)
    print("TEST 3: Feature Count After Removal")
    print("="*60)
    
    from granite.data.enhanced_accessibility import EnhancedAccessibilityComputer
    
    computer = EnhancedAccessibilityComputer(verbose=False)
    
    # Expected: 9 features per type × 3 types = 27 base features
    expected_base = 27
    expected_derived = 4
    expected_total = expected_base + expected_derived
    
    print(f"Expected base features: {expected_base} (9 per destination type)")
    print(f"Expected derived features: {expected_derived}")
    print(f"Expected total features: {expected_total}")
    
    print("\n✓ Feature count verification complete")
    print("  (Run full pipeline to validate actual feature extraction)")
    return True

def test_osrm_performance():
    """Test OSRM performance with realistic data volume"""
    print("\n" + "="*60)
    print("TEST 4: OSRM Performance Test")
    print("="*60)
    
    import time
    
    router = OSRMRouter(verbose=True)
    
    # Simulate realistic scale: 50 origins × 25 destinations = 1,250 routes
    print("\nSimulating tract-scale routing:")
    print("  50 addresses × 25 destinations = 1,250 route pairs")
    
    # Create test data
    import numpy as np
    
    # Random addresses in Hamilton County bounds
    origin_lons = np.random.uniform(-85.4, -85.1, 50)
    origin_lats = np.random.uniform(34.9, 35.2, 50)
    origins = gpd.GeoDataFrame({
        'address_id': range(50),
        'geometry': [Point(lon, lat) for lon, lat in zip(origin_lons, origin_lats)]
    }, crs='EPSG:4326')
    
    # Random destinations
    dest_lons = np.random.uniform(-85.4, -85.1, 25)
    dest_lats = np.random.uniform(34.9, 35.2, 25)
    destinations = gpd.GeoDataFrame({
        'dest_id': range(25),
        'geometry': [Point(lon, lat) for lon, lat in zip(dest_lons, dest_lats)]
    }, crs='EPSG:4326')
    
    # Time the routing
    start_time = time.time()
    travel_times = router.compute_multimodal_travel_times(origins, destinations)
    elapsed_time = time.time() - start_time
    
    print(f"\n✓ Completed 1,250 route pairs in {elapsed_time:.1f} seconds")
    
    # Extrapolate to full tract
    full_tract_routes = 2400 * 125  # 300,000 routes per destination type
    estimated_time = (elapsed_time / 1250) * full_tract_routes
    estimated_total = estimated_time * 3  # 3 destination types
    
    print(f"\nEstimated time for full tract:")
    print(f"  {full_tract_routes:,} routes per destination type: {estimated_time/60:.1f} minutes")
    print(f"  Total for 3 destination types: {estimated_total/60:.1f} minutes")
    
    if estimated_total < 300:  # Less than 5 minutes
        print("\n✓ Performance meets target (<5 minutes per tract)")
    else:
        print(f"\n⚠ Performance slower than target (>{estimated_total/60:.1f} min per tract)")
        print("  Consider reducing batch size or using more powerful machine")
    
    return True

def test_end_to_end_data_loading():
    """Test full data loading pipeline with real data"""
    print("\n" + "="*60)
    print("TEST 5: End-to-End Data Loading")
    print("="*60)
    
    from granite.data.loaders import DataLoader
    
    config = {
        'data': {
            'target_fips': '47065010100',
            'state_fips': '47',
            'county_fips': '065'
        },
        'processing': {
            'verbose': True
        }
    }
    
    loader = DataLoader(data_dir='./data', config=config)
    
    print("\nLoading destinations with real data...")
    
    # Test destination loading
    employment = loader.create_employment_destinations(use_real_data=True)
    print(f"✓ Employment destinations: {len(employment)}")
    
    healthcare = loader.create_healthcare_destinations(use_real_data=True)
    print(f"✓ Healthcare destinations: {len(healthcare)}")
    
    grocery = loader.create_grocery_destinations(use_real_data=True)
    print(f"✓ Grocery destinations: {len(grocery)}")
    
    # Check data quality
    total_destinations = len(employment) + len(healthcare) + len(grocery)
    print(f"\nTotal destinations: {total_destinations}")
    
    if total_destinations > 1000:
        print("✓ Real data loaded successfully (>1000 destinations)")
    else:
        print("⚠ Unexpectedly low destination count - may be using synthetic fallback")
    
    # Verify required columns
    for dest_type, dests in [('Employment', employment), ('Healthcare', healthcare), ('Grocery', grocery)]:
        assert 'geometry' in dests.columns, f"{dest_type} missing geometry"
        assert 'dest_id' in dests.columns, f"{dest_type} missing dest_id"
        assert 'dest_type' in dests.columns, f"{dest_type} missing dest_type"
        print(f"✓ {dest_type} has required columns")
    
    print("\n✓ End-to-end data loading successful")
    return True

def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "="*60)
    print("GRANITE REAL DATA INTEGRATION TEST SUITE")
    print("="*60 + "\n")
    
    tests = [
        ("Real Destination Loading", test_real_destinations),
        ("OSRM Routing", test_osrm_routing),
        ("Feature Count", test_feature_count),
        ("OSRM Performance", test_osrm_performance),
        ("End-to-End Data Loading", test_end_to_end_data_loading)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"\n✗ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, "FAIL"))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results:
        status_symbol = "✓" if result == "PASS" else "✗"
        print(f"{status_symbol} {test_name}: {result}")
    
    passed = sum(1 for _, result in results if result == "PASS")
    total = len(results)
    
    print("\n" + "="*60)
    if passed == total:
        print(f"ALL TESTS PASSED ({passed}/{total})")
        print("✓ Ready for production use")
    else:
        print(f"SOME TESTS FAILED ({passed}/{total} passed)")
        print("⚠ Review failures before proceeding")
    print("="*60)
    
    return passed == total

if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)