"""
Test data loading functionality for GRANITE
"""
import unittest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from granite.data.loaders import DataLoader


class TestDataLoader(unittest.TestCase):
    """Test DataLoader class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.loader = DataLoader(verbose=False)
    
    def test_load_svi_data(self):
        """Test SVI data loading"""
        # Test with default parameters
        svi_data = self.loader.load_svi_data()
        
        # Check data type
        self.assertIsInstance(svi_data, pd.DataFrame)
        
        # Check required columns
        required_cols = ['FIPS', 'RPL_THEMES']
        for col in required_cols:
            self.assertIn(col, svi_data.columns)
        
        # Check data validity
        self.assertGreater(len(svi_data), 0)
        
        # Check SVI range
        valid_svi = svi_data['RPL_THEMES'].dropna()
        if len(valid_svi) > 0:
            self.assertTrue(all(0 <= v <= 1 for v in valid_svi))
    
    def test_create_network_graph(self):
        """Test network graph creation"""
        # Create simple test road data
        roads_data = {
            'geometry': [
                LineString([(0, 0), (1, 0)]),
                LineString([(1, 0), (1, 1)]),
                LineString([(0, 0), (0, 1)])
            ],
            'MTFCC': ['S1100', 'S1200', 'S1200']
        }
        roads_gdf = gpd.GeoDataFrame(roads_data)
        
        # Create graph
        G = self.loader.create_network_graph(roads_gdf)
        
        # Check graph properties
        self.assertGreater(G.number_of_nodes(), 0)
        self.assertGreater(G.number_of_edges(), 0)
        
        # Check node attributes
        for node in G.nodes():
            self.assertIn('x', G.nodes[node])
            self.assertIn('y', G.nodes[node])
    
    def test_load_address_points(self):
        """Test address point generation"""
        n_addresses = 100
        addresses = self.loader.load_address_points(n_synthetic=n_addresses)
        
        # Check type
        self.assertIsInstance(addresses, gpd.GeoDataFrame)
        
        # Check size
        self.assertEqual(len(addresses), n_addresses)
        
        # Check columns
        required_cols = ['address_id', 'longitude', 'latitude', 'geometry']
        for col in required_cols:
            self.assertIn(col, addresses.columns)
        
        # Check geometry
        self.assertTrue(all(isinstance(geom, Point) for geom in addresses.geometry))
    
    def test_bbox_validation(self):
        """Test bounding box constraints"""
        bbox = (-85.5, 35.0, -85.0, 35.5)
        addresses = self.loader.load_address_points(n_synthetic=100)
        
        # Check all points are within bbox
        for _, addr in addresses.iterrows():
            self.assertGreaterEqual(addr['longitude'], bbox[0])
            self.assertLessEqual(addr['longitude'], bbox[2])
            self.assertGreaterEqual(addr['latitude'], bbox[1])
            self.assertLessEqual(addr['latitude'], bbox[3])


class TestDataIntegration(unittest.TestCase):
    """Test data integration functionality"""
    
    def test_load_hamilton_county_data(self):
        """Test integrated data loading function"""
        from granite.data.loaders import load_hamilton_county_data
        
        # Load all data
        data = load_hamilton_county_data()
        
        # Check all components are present
        expected_keys = ['svi', 'census_tracts', 'roads', 
                        'transit_stops', 'addresses', 'road_network']
        
        for key in expected_keys:
            self.assertIn(key, data)
            self.assertIsNotNone(data[key])


if __name__ == '__main__':
    unittest.main()