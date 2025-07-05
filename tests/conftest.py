"""
Test configuration for GRANITE framework
Save this as tests/conftest.py
"""
import pytest
import warnings
import subprocess
import sys
import os


def pytest_configure(config):
    """Configure pytest with custom markers and settings"""
    
    # Register custom markers
    config.addinivalue_line(
        "markers", "requires_r: mark test as requiring R installation"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (>5 seconds)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    
    # Suppress specific warnings during testing
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="rpy2")
    warnings.filterwarnings("ignore", category=UserWarning, module="rpy2")
    warnings.filterwarnings("ignore", message=".*R home.*")
    warnings.filterwarnings("ignore", message=".*Environment variable.*")


def check_r_available():
    """Check if R is available"""
    try:
        result = subprocess.run(['R', '--version'], 
                              capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_rpy2_working():
    """Check if rpy2 can connect to R"""
    try:
        import rpy2.robjects as ro
        ro.r('1+1')  # Simple test
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def r_available():
    """Session-scoped fixture indicating if R is available"""
    return check_r_available()


@pytest.fixture(scope="session") 
def rpy2_working():
    """Session-scoped fixture indicating if rpy2 is working"""
    return check_rpy2_working()


@pytest.fixture(scope="session")
def metricgraph_available(rpy2_working):
    """Check if MetricGraph package is available in R"""
    if not rpy2_working:
        return False
        
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        importr('MetricGraph')
        return True
    except Exception:
        return False


def pytest_runtest_setup(item):
    """Skip tests based on available dependencies"""
    
    # Skip R-dependent tests if R is not available
    if item.get_closest_marker("requires_r"):
        if not check_r_available():
            pytest.skip("R is not installed or not accessible")
        
        if not check_rpy2_working():
            pytest.skip("rpy2 cannot connect to R")


@pytest.fixture
def sample_graph_data():
    """Sample graph data for testing"""
    import pandas as pd
    import numpy as np
    
    # Create simple test graph
    nodes_df = pd.DataFrame({
        'node_id': range(4),
        'x': [0, 1, 1, 0],
        'y': [0, 0, 1, 1]
    })
    
    edges_df = pd.DataFrame({
        'from': [1, 2, 3, 4],  # 1-indexed for R
        'to': [2, 3, 4, 1]
    })
    
    return nodes_df, edges_df


@pytest.fixture
def sample_observations():
    """Sample observation data for testing"""
    import pandas as pd
    import numpy as np
    
    obs_df = pd.DataFrame({
        'x': [0.25, 0.75, 0.75, 0.25],
        'y': [0.25, 0.25, 0.75, 0.75],
        'value': [0.2, 0.4, 0.6, 0.8]
    })
    
    return obs_df


@pytest.fixture
def sample_gnn_features():
    """Sample GNN features for testing"""
    import numpy as np
    
    # Random features for 10 nodes, 3 features each (kappa, alpha, tau)
    np.random.seed(42)  # Reproducible tests
    features = np.random.randn(10, 3)
    
    return features


@pytest.fixture
def sample_locations():
    """Sample prediction locations"""
    import pandas as pd
    
    locations_df = pd.DataFrame({
        'x': [0.1, 0.5, 0.9],
        'y': [0.1, 0.5, 0.9]
    })
    
    return locations_df


# Custom pytest hooks for better error reporting
def pytest_exception_interact(node, call, report):
    """Custom error reporting for common issues"""
    
    if "rpy2" in str(call.excinfo.value):
        print("\n" + "="*60)
        print("🔧 rpy2 Error Detected!")
        print("This usually means R is not properly installed or configured.")
        print("\nTo fix this issue:")
        print("1. Install R: sudo apt-get install r-base r-base-dev")
        print("2. Install rpy2: pip install rpy2")
        print("3. Or skip R tests: pytest -m 'not requires_r'")
        print("="*60)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle missing dependencies gracefully"""
    
    # Add skip markers for missing dependencies
    r_available = check_r_available()
    rpy2_working = check_rpy2_working()
    
    for item in items:
        # Skip R tests if R not available
        if "metricgraph" in item.nodeid.lower() and not r_available:
            item.add_marker(pytest.mark.skip(reason="R not available"))
        
        # Skip rpy2 tests if rpy2 not working  
        if "rpy2" in str(item.function).lower() and not rpy2_working:
            item.add_marker(pytest.mark.skip(reason="rpy2 not working"))


# Test data cleanup
@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically clean up temporary files after each test"""
    yield
    
    # Cleanup common temp files
    temp_files = [
        'test_output.csv',
        'test_model.pth',
        'test_features.npy'
    ]
    
    for file in temp_files:
        if os.path.exists(file):
            os.remove(file)