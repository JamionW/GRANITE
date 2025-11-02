"""
Real Data Loaders for GRANITE - Corrected for Actual File Structure
Loads LEHD employment, healthcare facilities, and grocery stores
"""
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')


class RealDataLoader:
    """
    Loader for real-world data sources
    Handles LEHD employment, healthcare facilities, and grocery stores
    File paths corrected to match actual directory structure
    """
    
    def __init__(self, data_dir='./data/raw', verbose=True):
        self.data_dir = data_dir
        self.verbose = verbose
        
    def _log(self, message):
        if self.verbose:
            print(f"[RealDataLoader] {message}")
    
    def load_lehd_employment(self, state_fips='47', county_fips='065'):
        """
        Load LEHD employment data from WAC files
        Returns GeoDataFrame with employment locations and job counts
        
        Uses actual file structure:
        - Employment data: data/raw/lehd/tn_wac_S000_JT00_2021.csv
        - Block geometries: data/raw/lehd/tl_2021_47_tabblock20.shp
        """
        self._log(f"Loading LEHD employment data for {state_fips}{county_fips}...")
        
        # Load LEHD WAC file
        lehd_file = os.path.join(self.data_dir, 'lehd', 'tn_wac_S000_JT00_2021.csv')
        
        if not os.path.exists(lehd_file):
            raise FileNotFoundError(
                f"LEHD data not found: {lehd_file}"
            )
        
        self._log(f"Reading LEHD data from {lehd_file}")
        lehd_data = pd.read_csv(lehd_file, dtype={'w_geocode': str})
        
        # Filter to target county (first 5 digits of w_geocode = state+county FIPS)
        target_prefix = f"{state_fips}{county_fips}"
        county_data = lehd_data[lehd_data['w_geocode'].str.startswith(target_prefix)].copy()
        
        if len(county_data) == 0:
            raise ValueError(f"No LEHD data found for county {state_fips}{county_fips}")
        
        self._log(f"Filtered to {len(county_data)} blocks in target county")
        
        # Aggregate to block group level (first 12 digits)
        county_data['block_group'] = county_data['w_geocode'].str[:12]
        block_group_employment = county_data.groupby('block_group')['C000'].sum().reset_index()
        block_group_employment.columns = ['block_group', 'employees']
        
        self._log(f"Aggregated to {len(block_group_employment)} block groups")
        
        # Load block geometries from LEHD directory
        block_file = os.path.join(self.data_dir, 'lehd', 'tl_2021_47_tabblock20.shp')
        
        if not os.path.exists(block_file):
            raise FileNotFoundError(
                f"Census block geometries not found: {block_file}\n"
                f"Required for LEHD employment location mapping"
            )
        
        self._log(f"Loading block geometries from {block_file}")
        blocks = gpd.read_file(block_file)
        
        # Create block group ID from block GEOID (first 12 characters)
        blocks['block_group'] = blocks['GEOID20'].str[:12]
        
        # Get centroids for each block group
        block_groups = blocks.dissolve(by='block_group')
        block_groups = block_groups.reset_index()
        block_groups['geometry'] = block_groups.geometry.centroid
        
        # Merge employment data with geometries
        employment_gdf = block_group_employment.merge(
            block_groups[['block_group', 'geometry']], 
            on='block_group'
        )
        employment_gdf = gpd.GeoDataFrame(employment_gdf, geometry='geometry', crs='EPSG:4326')
        
        # Filter to county (in case block file has broader coverage)
        if employment_gdf.crs != 'EPSG:4326':
            employment_gdf = employment_gdf.to_crs('EPSG:4326')
        
        # Filter to meaningful employment locations (>10 employees)
        employment_gdf = employment_gdf[employment_gdf['employees'] > 10].copy()
        
        # Add required columns
        employment_gdf['dest_id'] = range(len(employment_gdf))
        employment_gdf['dest_type'] = 'employment'
        employment_gdf['name'] = employment_gdf.apply(
            lambda x: f"Employment BG {x['block_group'][-4:]} ({int(x['employees'])} jobs)", 
            axis=1
        )
        
        self._log(f"✓ Loaded {len(employment_gdf)} employment locations with {employment_gdf['employees'].sum():,} total jobs")
        
        return employment_gdf[['dest_id', 'name', 'employees', 'dest_type', 'geometry']]
    
    def load_healthcare_facilities(self):
        """
        Load healthcare facilities from CSV
        Returns GeoDataFrame with facility locations
        
        Uses actual file structure:
        - data/raw/healthcare/hamilton_county_healthcare.csv
        """
        self._log("Loading healthcare facilities...")
        
        healthcare_file = os.path.join(self.data_dir, 'healthcare', 'hamilton_county_healthcare.csv')
        
        if not os.path.exists(healthcare_file):
            raise FileNotFoundError(
                f"Healthcare data not found: {healthcare_file}"
            )
        
        # Load healthcare data
        self._log(f"Reading healthcare data from {healthcare_file}")
        healthcare_data = pd.read_csv(healthcare_file)
        
        # Validate required columns
        required_cols = ['name', 'lat', 'lon']
        missing_cols = [col for col in required_cols if col not in healthcare_data.columns]
        if missing_cols:
            raise ValueError(f"Healthcare CSV missing required columns: {missing_cols}")
        
        # Create geometry from lat/lon
        geometry = [Point(xy) for xy in zip(healthcare_data['lon'], healthcare_data['lat'])]
        healthcare_gdf = gpd.GeoDataFrame(
            healthcare_data,
            geometry=geometry,
            crs='EPSG:4326'
        )
        
        # Add required columns
        healthcare_gdf['dest_id'] = range(len(healthcare_gdf))
        healthcare_gdf['dest_type'] = 'healthcare'
        
        self._log(f"✓ Loaded {len(healthcare_gdf)} healthcare facilities")
        
        return healthcare_gdf[['dest_id', 'name', 'dest_type', 'geometry']]
    
    def load_grocery_stores(self):
        """
        Load grocery stores from OSM export CSV
        Returns GeoDataFrame with store locations
        
        Uses actual file structure:
        - data/raw/osm_grocery/hamilton_county_grocery_stores.csv
        """
        self._log("Loading grocery stores...")
        
        grocery_file = os.path.join(self.data_dir, 'osm_grocery', 'hamilton_county_grocery_stores.csv')
        
        if not os.path.exists(grocery_file):
            raise FileNotFoundError(
                f"Grocery data not found: {grocery_file}"
            )
        
        # Load grocery data
        self._log(f"Reading grocery data from {grocery_file}")
        grocery_data = pd.read_csv(grocery_file)
        
        # Validate required columns
        required_cols = ['name', 'lat', 'lon']
        missing_cols = [col for col in required_cols if col not in grocery_data.columns]
        if missing_cols:
            raise ValueError(f"Grocery CSV missing required columns: {missing_cols}")
        
        # Create geometry from lat/lon
        geometry = [Point(xy) for xy in zip(grocery_data['lon'], grocery_data['lat'])]
        grocery_gdf = gpd.GeoDataFrame(
            grocery_data,
            geometry=geometry,
            crs='EPSG:4326'
        )
        
        # Add required columns
        grocery_gdf['dest_id'] = range(len(grocery_gdf))
        grocery_gdf['dest_type'] = 'grocery'
        
        self._log(f"✓ Loaded {len(grocery_gdf)} grocery stores")
        
        return grocery_gdf[['dest_id', 'name', 'dest_type', 'geometry']]


# Quick validation function
def validate_data_files(data_dir='./data/raw'):
    """
    Validate that all required data files exist
    Returns dict with validation results
    """
    results = {
        'employment': {
            'wac_file': os.path.exists(os.path.join(data_dir, 'lehd', 'tn_wac_S000_JT00_2021.csv')),
            'block_file': os.path.exists(os.path.join(data_dir, 'lehd', 'tl_2021_47_tabblock20.shp'))
        },
        'healthcare': {
            'facility_file': os.path.exists(os.path.join(data_dir, 'healthcare', 'hamilton_county_healthcare.csv'))
        },
        'grocery': {
            'store_file': os.path.exists(os.path.join(data_dir, 'osm_grocery', 'hamilton_county_grocery_stores.csv'))
        }
    }
    
    all_present = all([
        results['employment']['wac_file'],
        results['employment']['block_file'],
        results['healthcare']['facility_file'],
        results['grocery']['store_file']
    ])
    
    return results, all_present


if __name__ == '__main__':
    # Quick test
    print("Validating data file structure...")
    results, all_present = validate_data_files()
    
    print("\nEmployment Data:")
    print(f"  WAC file: {'✓' if results['employment']['wac_file'] else '✗'}")
    print(f"  Block geometries: {'✓' if results['employment']['block_file'] else '✗'}")
    
    print("\nHealthcare Data:")
    print(f"  Facility file: {'✓' if results['healthcare']['facility_file'] else '✗'}")
    
    print("\nGrocery Data:")
    print(f"  Store file: {'✓' if results['grocery']['store_file'] else '✗'}")
    
    if all_present:
        print("\n✓ All required data files present!")
        print("\nAttempting to load data...")
        
        loader = RealDataLoader(data_dir='./data/raw')
        
        try:
            emp = loader.load_lehd_employment()
            print(f"✓ Employment: {len(emp)} locations")
        except Exception as e:
            print(f"✗ Employment: {e}")
        
        try:
            health = loader.load_healthcare_facilities()
            print(f"✓ Healthcare: {len(health)} facilities")
        except Exception as e:
            print(f"✗ Healthcare: {e}")
        
        try:
            grocery = loader.load_grocery_stores()
            print(f"✓ Grocery: {len(grocery)} stores")
        except Exception as e:
            print(f"✗ Grocery: {e}")
    else:
        print("\n✗ Some required data files are missing")
        print("See above for details")