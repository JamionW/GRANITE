"""
Standalone Test Script for GRANITE Baseline Comparisons

Run this to validate that IDW and Kriging implementations work correctly
before integrating into your full pipeline.
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt


def create_test_data():
    """Create synthetic spatial data for testing."""
    
    print("Creating synthetic test data...")
    
    # Create 4 tracts in a 2x2 grid
    tract_polys = [
        Polygon([(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)]),  # Low SVI
        Polygon([(0.5, 0), (1, 0), (1, 0.5), (0.5, 0.5)]),   # Medium SVI
        Polygon([(0, 0.5), (0.5, 0.5), (0.5, 1), (0, 1)]),   # High SVI
        Polygon([(0.5, 0.5), (1, 0.5), (1, 1), (0.5, 1)])    # Very High SVI
    ]
    
    tract_gdf = gpd.GeoDataFrame({
        'geometry': tract_polys,
        'SVI': [0.2, 0.4, 0.7, 0.9],
        'tract_id': ['A', 'B', 'C', 'D']
    })
    
    # Create address points distributed across tracts
    np.random.seed(42)
    n_per_tract = 100
    
    addresses = []
    true_svi = []
    
    for poly, svi in zip(tract_polys, [0.2, 0.4, 0.7, 0.9]):
        bounds = poly.bounds
        for _ in range(n_per_tract):
            x = np.random.uniform(bounds[0], bounds[2])
            y = np.random.uniform(bounds[1], bounds[3])
            
            # Add spatial variation within tract
            svi_variation = svi + np.random.normal(0, 0.05)
            svi_variation = np.clip(svi_variation, 0, 1)
            
            addresses.append(Point(x, y))
            true_svi.append(svi_variation)
    
    address_gdf = gpd.GeoDataFrame({
        'geometry': addresses,
        'SVI': true_svi
    })
    
    return tract_gdf, address_gdf


def test_idw():
    """Test IDW implementation."""
    
    print("\n" + "="*80)
    print("TEST 1: Inverse Distance Weighting (IDW)")
    print("="*80)
    
    from granite.evaluation.baseline_comparisons import InverseDistanceWeighting
    
    tract_gdf, address_gdf = create_test_data()
    
    # Test with different power parameters
    for power in [1.0, 2.0, 3.0]:
        print(f"\nTesting IDW with power={power}...")
        
        idw = InverseDistanceWeighting(power=power, n_neighbors=4)
        idw.fit(tract_gdf, 'SVI')
        
        coords = np.array([[geom.x, geom.y] for geom in address_gdf.geometry])
        predictions = idw.predict(coords)
        
        true_svi = address_gdf['SVI'].values
        
        mae = np.mean(np.abs(predictions - true_svi))
        rmse = np.sqrt(np.mean((predictions - true_svi)**2))
        corr = np.corrcoef(predictions, true_svi)[0, 1]
        
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Corr: {corr:.4f}")
        
        # Sanity checks
        assert predictions.min() >= 0 and predictions.max() <= 1, \
            "Predictions outside valid SVI range [0, 1]"
        assert not np.any(np.isnan(predictions)), \
            "NaN values in predictions"
        
        print("  ✓ Passed validation checks")
    
    return tract_gdf, address_gdf, predictions


def test_kriging():
    """Test Kriging implementation."""
    
    print("\n" + "="*80)
    print("TEST 2: Ordinary Kriging")
    print("="*80)
    
    from granite.evaluation.baseline_comparisons import OrdinaryKriging
    
    tract_gdf, address_gdf = create_test_data()
    
    print("\nTesting Ordinary Kriging...")
    
    ok = OrdinaryKriging(
        variogram_range=0.3,  # Adjusted for 0-1 coordinate space
        sill=0.1,
        nugget=0.01
    )
    
    ok.fit(tract_gdf, 'SVI')
    
    coords = np.array([[geom.x, geom.y] for geom in address_gdf.geometry])
    predictions = ok.predict(coords)
    
    true_svi = address_gdf['SVI'].values
    
    mae = np.mean(np.abs(predictions - true_svi))
    rmse = np.sqrt(np.mean((predictions - true_svi)**2))
    corr = np.corrcoef(predictions, true_svi)[0, 1]
    
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Corr: {corr:.4f}")
    
    # Sanity checks
    assert predictions.min() >= -0.2 and predictions.max() <= 1.2, \
        "Predictions far outside reasonable range"
    assert not np.any(np.isnan(predictions)), \
        "NaN values in predictions"
    
    print("  ✓ Passed validation checks")
    
    return tract_gdf, address_gdf, predictions


def test_full_comparison():
    """Test the full BaselineComparison framework."""
    
    print("\n" + "="*80)
    print("TEST 3: Full Baseline Comparison Framework")
    print("="*80)
    
    from granite.evaluation.baseline_comparisons import BaselineComparison
    
    tract_gdf, address_gdf = create_test_data()
    
    # Simulate GNN predictions (assume GNN does better than baselines)
    true_svi = address_gdf['SVI'].values
    gnn_predictions = true_svi + np.random.normal(0, 0.04, len(true_svi))
    gnn_predictions = np.clip(gnn_predictions, 0, 1)
    
    print("\nRunning full comparison framework...")
    
    comparison = BaselineComparison()
    
    try:
        results = comparison.run_comparison(
            tract_gdf=tract_gdf,
            address_gdf=address_gdf,
            gnn_predictions=gnn_predictions,
            svi_column='SVI'
        )
        
        print("\n✓ Comparison completed successfully!")
        
        # Print summary
        comparison.print_summary()
        
        # Check that all methods produced results
        assert 'IDW_p2' in results['methods'], "IDW_p2 results missing"
        assert 'IDW_p3' in results['methods'], "IDW_p3 results missing"
        assert 'Kriging' in results['methods'], "Kriging results missing"
        
        print("\n✓ All baseline methods ran successfully!")
        
        # Verify metrics computed
        assert 'metrics' in results, "Metrics not computed"
        assert len(results['metrics']) == 4, "Expected 4 methods in metrics"
        
        print("✓ Metrics computed correctly!")
        
        # Check spatial stats
        if 'spatial_stats' in results:
            print("✓ Spatial statistics computed!")
        else:
            print("⚠ Spatial statistics skipped (libpysal not installed)")
        
        return results
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        raise


def test_visual_validation():
    """Create visual validation plots."""
    
    print("\n" + "="*80)
    print("TEST 4: Visual Validation")
    print("="*80)
    
    from granite.evaluation.baseline_comparisons import InverseDistanceWeighting
    
    tract_gdf, address_gdf = create_test_data()
    
    # Fit IDW
    idw = InverseDistanceWeighting(power=2.0, n_neighbors=4)
    idw.fit(tract_gdf, 'SVI')
    
    coords = np.array([[geom.x, geom.y] for geom in address_gdf.geometry])
    predictions = idw.predict(coords)
    true_svi = address_gdf['SVI'].values
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # True SVI
    sc1 = axes[0].scatter(coords[:, 0], coords[:, 1], 
                          c=true_svi, cmap='RdYlGn_r', s=20, alpha=0.6)
    axes[0].set_title('True SVI', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(sc1, ax=axes[0])
    
    # Predicted SVI
    sc2 = axes[1].scatter(coords[:, 0], coords[:, 1], 
                          c=predictions, cmap='RdYlGn_r', s=20, alpha=0.6)
    axes[1].set_title('IDW Predictions', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    plt.colorbar(sc2, ax=axes[1])
    
    # Prediction errors
    errors = predictions - true_svi
    sc3 = axes[2].scatter(coords[:, 0], coords[:, 1], 
                          c=errors, cmap='RdBu_r', s=20, alpha=0.6,
                          vmin=-0.2, vmax=0.2)
    axes[2].set_title('Prediction Errors', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Longitude')
    axes[2].set_ylabel('Latitude')
    plt.colorbar(sc3, ax=axes[2])
    
    # Add tract boundaries to all plots
    for ax in axes:
        tract_gdf.boundary.plot(ax=ax, color='black', linewidth=2, alpha=0.5)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('/workspaces/GRANITE/output/visualizations/baseline_visual_validation.png', dpi=150, bbox_inches='tight')
    
    print("✓ Visual validation plot saved to: baseline_visual_validation.png")
    print("  Check that:")
    print("  - True SVI shows clear spatial structure")
    print("  - Predictions capture general pattern")
    print("  - Errors are randomly distributed (no systematic patterns)")
    
    return fig


def run_all_tests():
    """Run complete test suite."""
    
    print("\n" + "="*80)
    print("GRANITE BASELINE COMPARISON - TEST SUITE")
    print("="*80)
    print("\nRunning comprehensive validation tests...")
    
    try:
        # Test 1: IDW
        test_idw()
        
        # Test 2: Kriging  
        test_kriging()
        
        # Test 3: Full comparison
        test_full_comparison()
        
        # Test 4: Visual validation
        test_visual_validation()
        
        # Success summary
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        print("\nThe baseline comparison module is working correctly.")
        print("You can now integrate it into your GRANITE pipeline.")
        print("\nNext steps:")
        print("1. Copy baseline_comparisons.py to your GRANITE repo")
        print("2. See baseline_integration_guide.py for integration instructions")
        print("3. Run your full pipeline with baseline comparisons enabled")
        print("="*80)
        
    except Exception as e:
        print("\n" + "="*80)
        print("✗ TEST FAILED")
        print("="*80)
        print(f"Error: {e}")
        print("\nPlease check:")
        print("- NumPy, GeoPandas, SciPy are installed")
        print("- baseline_comparisons.py is in the same directory")
        print("- Python version is 3.8+")
        raise


if __name__ == '__main__':
    run_all_tests()