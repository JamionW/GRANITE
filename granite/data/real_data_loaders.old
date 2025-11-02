"""
Real Data Loaders for GRANITE
Replaces synthetic destination generation with real data sources
"""
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

class RealDataLoader:
    """
    Load real-world destination data from acquired sources
    """
    
    def __init__(self, data_dir='./data/raw', verbose=True):
        self.data_dir = Path(data_dir)
        self.verbose = verbose
    
    def log(self, message):
        if self.verbose:
            print(f"[RealDataLoader] {message}")
    
    def load_lehd_employment(self, state_fips='47', county_fips='065'):
        """
        Load employment destinations from LEHD data
        
        Returns:
            GeoDataFrame with employment locations at census block centroids
        """
        self.log(f"Loading LEHD employment data for {state_fips}{county_fips}...")
        
        # Load LEHD workplace data
        lehd_file = self.data_dir / 'lehd' / 'tn_wac_S000_JT00_2021.csv'
        if not lehd_file.exists():
            raise FileNotFoundError(f"LEHD data not found: {lehd_file}")
        
        lehd_data = pd.read_csv(lehd_file, dtype={'w_geocode': str})
        
        # Filter to target county
        county_filter = state_fips + county_fips
        lehd_data = lehd_data[lehd_data['w_geocode'].str.startswith(county_filter)].copy()
        
        # Filter to blocks with significant employment (>10 jobs)
        lehd_data = lehd_data[lehd_data['C000'] > 10].copy()
        
        self.log(f"  Found {len(lehd_data)} employment blocks")
        self.log(f"  Total jobs: {lehd_data['C000'].sum():,}")
        
        # Load census block shapefiles
        block_shp = self.data_dir / 'lehd' / 'tl_2021_47_tabblock20.shp'
        if not block_shp.exists():
            raise FileNotFoundError(f"Census block shapefile not found: {block_shp}")
        
        blocks_gdf = gpd.read_file(block_shp)
        blocks_gdf = blocks_gdf[blocks_gdf['GEOID20'].str.startswith(county_filter)].copy()
        
        # Merge employment data with block geometries
        employment_blocks = blocks_gdf.merge(
            lehd_data,
            left_on='GEOID20',
            right_on='w_geocode',
            how='inner'
        )
        
        # Get block centroids as employment destinations
        employment_blocks['geometry'] = employment_blocks.geometry.centroid
        
        # Create standardized destination format
        employment_destinations = employment_blocks[['GEOID20', 'C000', 'geometry']].copy()
        employment_destinations.rename(columns={
            'GEOID20': 'dest_id',
            'C000': 'employees'
        }, inplace=True)
        
        employment_destinations['dest_type'] = 'employment'
        employment_destinations['name'] = 'Block ' + employment_destinations['dest_id']
        
        # Add importance weighting based on job count
        max_jobs = employment_destinations['employees'].max()
        employment_destinations['importance'] = employment_destinations['employees'] / max_jobs
        
        self.log(f"  Created {len(employment_destinations)} employment destinations")
        
        return employment_destinations
    
    def load_healthcare_facilities(self, county_filter='hamilton'):
        """
        Load healthcare facilities from curated CSV
        
        Returns:
            GeoDataFrame with healthcare facility locations
        """
        self.log("Loading healthcare facilities...")
        
        healthcare_file = self.data_dir / 'healthcare' / 'hamilton_county_healthcare.csv'
        if not healthcare_file.exists():
            raise FileNotFoundError(f"Healthcare data not found: {healthcare_file}")
        
        healthcare_df = pd.read_csv(healthcare_file)
        
        # Create geometry from lat/lon
        geometry = [Point(lon, lat) for lon, lat in zip(healthcare_df['lon'], healthcare_df['lat'])]
        healthcare_gdf = gpd.GeoDataFrame(healthcare_df, geometry=geometry, crs='EPSG:4326')
        
        # Standardize format
        healthcare_gdf['dest_id'] = range(len(healthcare_gdf))
        healthcare_gdf['dest_type'] = 'healthcare'
        
        # Calculate importance based on beds (hospitals) or set to 0.5 (clinics)
        healthcare_gdf['importance'] = healthcare_gdf.apply(
            lambda row: min(1.0, row['beds'] / 400) if row['beds'] > 0 else 0.5,
            axis=1
        )
        
        self.log(f"  Loaded {len(healthcare_gdf)} healthcare facilities")
        self.log(f"    Hospitals: {len(healthcare_gdf[healthcare_gdf['type'] == 'hospital'])}")
        self.log(f"    Clinics: {len(healthcare_gdf[healthcare_gdf['type'] == 'clinic'])}")
        
        return healthcare_gdf
    
    def load_grocery_stores(self):
        """
        Load grocery stores from OSM data
        
        Returns:
            GeoDataFrame with grocery store locations
        """
        self.log("Loading grocery stores...")
        
        grocery_file = self.data_dir / 'osm_grocery' / 'hamilton_county_grocery_stores.csv'
        if not grocery_file.exists():
            raise FileNotFoundError(f"Grocery data not found: {grocery_file}")
        
        grocery_df = pd.read_csv(grocery_file)
        
        # Create geometry from lat/lon
        geometry = [Point(lon, lat) for lon, lat in zip(grocery_df['lon'], grocery_df['lat'])]
        grocery_gdf = gpd.GeoDataFrame(grocery_df, geometry=geometry, crs='EPSG:4326')
        
        # Standardize format
        grocery_gdf['dest_id'] = range(len(grocery_gdf))
        grocery_gdf['dest_type'] = 'grocery'
        
        # Calculate importance based on store type
        store_type_importance = {
            'supermarket': 1.0,
            'convenience': 0.4,
            'grocery': 0.7
        }
        grocery_gdf['importance'] = grocery_gdf['type'].map(store_type_importance).fillna(0.6)
        
        self.log(f"  Loaded {len(grocery_gdf)} grocery stores")
        
        return grocery_gdf
    
    def load_all_destinations(self, state_fips='47', county_fips='065'):
        """
        Load all destination types
        
        Returns:
            Dictionary with 'employment', 'healthcare', 'grocery' GeoDataFrames
        """
        self.log("Loading all real destinations...")
        
        destinations = {
            'employment': self.load_lehd_employment(state_fips, county_fips),
            'healthcare': self.load_healthcare_facilities(),
            'grocery': self.load_grocery_stores()
        }
        
        total_destinations = sum(len(dests) for dests in destinations.values())
        self.log(f"Total destinations loaded: {total_destinations}")
        
        return destinations