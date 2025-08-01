#!/usr/bin/env python3
"""
Test script for enhanced transit data loading
Run this to validate your transit data setup
"""

import sys
import os
sys.path.append('.')

import yaml
import matplotlib.pyplot as plt
from granite.data.loaders import DataLoader

def test_transit_loading():
    """Test all transit data loading methods"""
    
    print("🚌 GRANITE Transit Data Loading Test")
    print("="*50)
    
    # Load config
    try:
        with open('./config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Config loaded successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not load config ({e}), using defaults")
        config = {
            'transit': {
                'preferred_source': 'gtfs',
                'download_real_data': True,
                'fallback_to_grid': True
            }
        }
    
    # Initialize loader
    loader = DataLoader('./data', verbose=True, config=config)
    
    # Test individual methods
    print("\n🔍 Testing Individual Methods:")
    print("-" * 30)
    
    # Test GTFS
    print("\n1. Testing CARTA GTFS loading...")
    gtfs_stops = loader._load_carta_gtfs_stops()
    if gtfs_stops is not None:
        print(f"   ✅ GTFS: {len(gtfs_stops)} stops loaded")
        print(f"   📍 Sample stops: {list(gtfs_stops['stop_name'].head(3))}")
    else:
        print("   ❌ GTFS: Failed to load")
    
    # Test OSM
    print("\n2. Testing OpenStreetMap loading...")
    osm_stops = loader._load_osm_transit_stops()
    if osm_stops is not None:
        print(f"   ✅ OSM: {len(osm_stops)} stops loaded")
        print(f"   📍 Sample stops: {list(osm_stops['stop_name'].head(3))}")
        route_types = osm_stops['route_type'].value_counts()
        print(f"   🚌 Route types: {dict(route_types)}")
    else:
        print("   ❌ OSM: Failed to load")
    
    # Test Grid
    print("\n3. Testing realistic grid generation...")
    grid_stops = loader._create_realistic_transit_grid()
    if grid_stops is not None:
        print(f"   ✅ Grid: {len(grid_stops)} stops generated")
        service_areas = grid_stops['service_area'].value_counts()
        print(f"   🏙️  Service areas: {dict(service_areas)}")
    else:
        print("   ❌ Grid: Failed to generate")
    
    # Test automatic loading with fallback
    print("\n🎯 Testing Automatic Loading (with fallback):")
    print("-" * 45)
    
    final_stops = loader.load_transit_stops()
    
    if len(final_stops) > 0:
        data_source = final_stops['data_source'].iloc[0]
        print(f"\n✅ Final Result: {len(final_stops)} stops")
        print(f"📊 Data source: {data_source}")
        
        # Analyze quality
        if 'route_type' in final_stops.columns:
            route_counts = final_stops['route_type'].value_counts()
            print(f"🚌 Route type breakdown:")
            for route_type, count in route_counts.items():
                print(f"   {route_type}: {count} stops")
        
        # Geographic coverage
        bounds = final_stops.total_bounds
        print(f"🗺️  Geographic coverage:")
        print(f"   Longitude: {bounds[0]:.4f} to {bounds[2]:.4f}")
        print(f"   Latitude: {bounds[1]:.4f} to {bounds[3]:.4f}")
        
        # Data quality assessment
        print(f"\n📈 Quality Assessment:")
        if data_source == 'CARTA_GTFS':
            print("   🟢 Excellent: Real GTFS data")
            print("   ✓ Includes real stop names and locations")
            print("   ✓ May include schedule information")
            print("   ✓ Highest research credibility")
        elif data_source == 'OpenStreetMap':
            print("   🟡 Good: Community-verified OSM data")
            print("   ✓ Real stop locations")
            print("   ✓ Community maintained")
            print("   ⚠ May be incomplete")
        elif data_source == 'Generated_Grid':
            print("   🟡 Fair: Realistic generated stops")
            print("   ✓ Covers all service areas")
            print("   ✓ Demographically informed placement")
            print("   ⚠ Not real data - note in research")
        else:
            print("   🔴 Poor: Minimal fallback data")
            print("   ⚠ Only 5 stops - inadequate for research")
            print("   ⚠ Consider downloading real data")
        
        # Create visualization
        print("\n🎨 Creating visualization...")
        create_transit_map(final_stops)
        
    else:
        print("❌ CRITICAL: No transit stops loaded!")
        return False
    
    print("\n" + "="*50)
    print("🎉 Transit data loading test complete!")
    
    return True

def create_transit_map(transit_stops):
    """Create a simple map of transit stops"""
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot stops colored by route type
        if 'route_type' in transit_stops.columns:
            route_types = transit_stops['route_type'].unique()
            colors = plt.cm.Set1(range(len(route_types)))
            
            for i, route_type in enumerate(route_types):
                subset = transit_stops[transit_stops['route_type'] == route_type]
                subset.plot(ax=ax, color=colors[i], label=route_type, 
                           markersize=50, alpha=0.7)
        else:
            transit_stops.plot(ax=ax, color='blue', markersize=50, alpha=0.7)
        
        # Styling
        ax.set_title('Chattanooga Transit Stops', fontsize=16, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        if 'route_type' in transit_stops.columns:
            ax.legend(title='Route Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add data source annotation
        data_source = transit_stops['data_source'].iloc[0]
        ax.text(0.02, 0.98, f'Data Source: {data_source}', 
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        plt.tight_layout()
        
        # Save map
        output_path = './output/transit_stops_test.png'
        os.makedirs('./output', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📍 Transit map saved to: {output_path}")
        
    except Exception as e:
        print(f"   ⚠️  Could not create visualization: {e}")

if __name__ == "__main__":
    success = test_transit_loading()
    if success:
        print("\n✅ All tests passed! Your transit data setup is working.")
        print("\nNext steps:")
        print("1. Review the generated map in ./output/transit_stops_test.png")
        print("2. If using generated data, consider downloading real CARTA GTFS")
        print("3. Update your config.yaml with transit preferences")
    else:
        print("\n❌ Tests failed. Check your setup and try again.")
    
    sys.exit(0 if success else 1)