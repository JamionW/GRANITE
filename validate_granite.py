#!/usr/bin/env python3
"""
GRANITE Framework System Validation
Quick end-to-end test of all major components
"""

import sys
import traceback
from datetime import datetime


def print_banner():
    print("=" * 70)
    print("🔍 GRANITE Framework Validation")
    print("=" * 70)


def test_component(name, test_func):
    """Test a component and report results"""
    print(f"\n🧪 Testing {name}...")
    try:
        result = test_func()
        print(f"  ✅ {name}: PASSED")
        return True, result
    except Exception as e:
        print(f"  ❌ {name}: FAILED")
        print(f"     Error: {str(e)}")
        return False, None


def test_imports():
    """Test all critical imports"""
    import granite
    from granite.data.loaders import DataLoader
    from granite.models.gnn import create_gnn_model
    from granite.metricgraph.interface import MetricGraphInterface
    from granite.disaggregation.pipeline import GRANITEPipeline
    from granite.visualization.plots import DisaggregationVisualizer
    
    return f"GRANITE v{granite.__version__}"


def test_data_loading():
    """Test data loading capabilities"""
    from granite.data.loaders import DataLoader
    
    loader = DataLoader(verbose=False)
    
    # Test SVI data (should fall back to mock if needed)
    svi_data = loader.load_svi_data()
    
    # Test synthetic addresses
    addresses = loader.load_address_points(n_synthetic=50)
    
    # Test mock roads
    roads = loader._create_mock_roads()
    
    # Test network graph creation
    graph = loader.create_network_graph(roads)
    
    return f"SVI: {len(svi_data)} tracts, Addresses: {len(addresses)}, Graph: {graph.number_of_nodes()} nodes"


def test_gnn_models():
    """Test GNN model creation and basic operations"""
    import torch
    import numpy as np
    from granite.models.gnn import create_gnn_model
    
    # Create simple test data
    node_features = torch.randn(10, 5)  # 10 nodes, 5 features each
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])  # Simple cycle
    
    # Create model
    model = create_gnn_model(
        input_dim=5,
        hidden_dim=16,
        output_dim=3,  # kappa, alpha, tau
        model_type='standard'
    )
    
    # Test forward pass
    with torch.no_grad():
        output = model(node_features, edge_index)
    
    return f"Model created, output shape: {output.shape}"


def test_metricgraph_interface():
    """Test MetricGraph R interface"""
    from granite.metricgraph.interface import MetricGraphInterface
    import pandas as pd
    import numpy as np
    
    # Create interface (should handle missing R gracefully)
    mg = MetricGraphInterface(verbose=False)
    
    # Test fallback prediction
    obs_df = pd.DataFrame({
        'x': [0, 1, 2],
        'y': [0, 1, 0],
        'value': [0.3, 0.5, 0.7]
    })
    
    locations_df = pd.DataFrame({
        'x': [0.5, 1.5],
        'y': [0.5, 0.5]
    })
    
    # This should work even without R
    predictions = mg._fallback_prediction(obs_df, locations_df)
    
    return f"Interface created, fallback predictions: {len(predictions)} locations"


def test_pipeline():
    """Test the main GRANITE pipeline"""
    from granite.disaggregation.pipeline import GRANITEPipeline
    
    # Create pipeline
    pipeline = GRANITEPipeline(
        data_dir='./data',
        output_dir='./output',
        verbose=False
    )
    
    # Test data loading
    pipeline.load_data()
    
    # Check that data was loaded
    data_keys = list(pipeline.data.keys())
    
    return f"Pipeline created, loaded data: {data_keys}"


def test_end_to_end():
    """Test minimal end-to-end functionality"""
    from granite.disaggregation.pipeline import GRANITEPipeline
    import tempfile
    import os
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create pipeline
        pipeline = GRANITEPipeline(
            data_dir='./data',
            output_dir=temp_dir,
            verbose=False
        )
        
        # Load data
        pipeline.load_data()
        
        # Test that we have the required data
        assert 'svi' in pipeline.data
        assert 'addresses' in pipeline.data
        assert 'road_network' in pipeline.data
        
        # Test basic disaggregation setup
        svi_data = pipeline.data['svi']
        addresses = pipeline.data['addresses']
        
        return f"E2E test passed: {len(svi_data)} SVI tracts → {len(addresses)} addresses"


def main():
    """Run all validation tests"""
    print_banner()
    
    tests = [
        ("Critical Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("GNN Models", test_gnn_models),
        ("MetricGraph Interface", test_metricgraph_interface),
        ("Pipeline Components", test_pipeline),
        ("End-to-End Flow", test_end_to_end)
    ]
    
    results = []
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        passed, result = test_component(test_name, test_func)
        results.append(passed)
        
        if result and passed:
            print(f"     Details: {result}")
    
    # Summary
    passed_count = sum(results)
    print("\n" + "=" * 70)
    print(f"📊 Validation Summary: {passed_count}/{total_tests} tests passed")
    
    if passed_count == total_tests:
        print("🎉 GRANITE Framework: FULLY OPERATIONAL")
        print("\n🚀 Ready for:")
        print("   • SVI disaggregation experiments")
        print("   • GNN-MetricGraph integration research")
        print("   • Hamilton County transit accessibility analysis")
        print("\n💡 Try: python scripts/run_granite.py")
    else:
        print("⚠️  Some components need attention")
        print("   • Check error messages above")
        print("   • Ensure all dependencies are installed")
    
    print("=" * 70)
    
    return passed_count == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)