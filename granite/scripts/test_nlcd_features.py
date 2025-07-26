#!/usr/bin/env python
"""
Simplified test script to verify NLCD feature loading
"""
import sys
import os

# Add parent directory to path  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from granite.data.loaders import DataLoader

def test_nlcd_simple():
    """Test NLCD data loading with existing DataLoader methods"""
    
    loader = DataLoader('./data', verbose=True)
    
    print("Testing NLCD data loading...")
    
    try:
        # Check if NLCD file exists
        nlcd_path = "./data/nlcd_hamilton_county.tif"
        
        if not os.path.exists(nlcd_path):
            print(f"✗ NLCD file not found at {nlcd_path}")
            return False
        
        print(f"✓ NLCD file found")
        
        # Test basic file reading
        import rasterio
        with rasterio.open(nlcd_path) as src:
            print(f"✓ NLCD file readable: {src.shape}")
            sample = src.read(1, window=rasterio.windows.Window(0, 0, 100, 100))
            unique_values = sorted(list(set(sample.flatten())))[:10]
            print(f"✓ Sample NLCD classes: {unique_values}")
        
        # Test tract loading
        tracts = loader.load_census_tracts('47', '065')
        print(f"✓ Loaded {len(tracts)} tracts")
        
        # Test address loading  
        addresses = loader.load_address_points('47', '065')
        print(f"✓ Loaded {len(addresses)} addresses")
        
        print(f"\n🎉 Basic components working!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_nlcd_simple()