"""
GRANITE Data Loaders

Provides data loading and preprocessing for census, SVI, addresses,
destinations, and graph construction.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import networkx as nx
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import types
import warnings
warnings.filterwarnings('ignore')

from .block_group_loader import BlockGroupLoader
from .enhanced_accessibility import EnhancedAccessibilityComputer


class DataLoader:
    """
    Streamlined data loader focused on accessibility feature computation
    Uses road network topology for graph creation instead of simple KNN
    """
    
    def __init__(self, data_dir: str = '/workspaces/GRANITE/data', config: dict = None):
        self.data_dir = data_dir
        self.config = config or {}
        self.verbose = config.get('processing', {}).get('verbose', False) if config else False
        
        # Initialize enhanced accessibility computer
        self.accessibility_computer = EnhancedAccessibilityComputer(verbose=self.verbose)
        
        # Simplified caching
        self._address_cache = None
        self._svi_cache = None
        self._transit_cache = None

        # Load census data for validation
        self.block_group_loader = BlockGroupLoader(data_dir, verbose=self.verbose)
        
        os.makedirs(data_dir, exist_ok=True)

        # Bind enhanced destination methods
        self.bind_enhanced_destination_methods()
    
    def _log(self, message: str):
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] DataLoader: {message}")

    # =========================================================================
    # CENSUS AND SVI DATA LOADING
    # =========================================================================

    def load_census_tracts(self, state_fips: str = '47', county_fips: str = '065') -> gpd.GeoDataFrame:
        """Load census tract geometries"""
        self._log(f"Loading census tracts for {state_fips}-{county_fips}...")
        
        local_file = os.path.join(self.data_dir, 'raw', f'tl_2020_{state_fips}_tract.shp')
        
        try:
            if os.path.exists(local_file):
                tracts = gpd.read_file(local_file)
                self._log(f"Loaded {len(tracts)} tracts from local file")
            else:
                raise FileNotFoundError(
                    f"Census tracts file not found at {local_file}. "
                    f"Download from Census TIGER/Line: https://www.census.gov/geo/maps-data/data/tiger-line.html"
                )
            
            # Filter to county
            county_tracts = tracts[tracts['COUNTYFP'] == county_fips].copy()
            county_tracts['FIPS'] = county_tracts['GEOID'].astype(str).str.strip()
            
            if county_tracts.crs is None:
                county_tracts.set_crs(epsg=4326, inplace=True)
                
            self._log(f"Filtered to {len(county_tracts)} tracts")
            return county_tracts
            
        except Exception as e:
            self._log(f"Error loading census tracts: {str(e)}")
            raise

    def load_svi_data(self, state_fips: str = '47', county_name: str = 'Hamilton') -> pd.DataFrame:
        """Load Social Vulnerability Index data with socioeconomic controls"""
        self._log(f"Loading SVI data for {county_name} County, {state_fips}...")
        
        svi_file = os.path.join(self.data_dir, 'raw', f'SVI_2020_US.csv')
        
        try:
            if os.path.exists(svi_file):
                svi_data = pd.read_csv(svi_file, dtype={'FIPS': str, 'ST': str})
                self._log(f"Loaded local SVI data ({len(svi_data)} records)")
            else:
                raise FileNotFoundError(
                    f"SVI data file not found at {svi_file}. "
                    f"Download from CDC SVI: https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html"
                )
            
            # Filter to county
            county_svi = svi_data[
                (svi_data['ST'] == state_fips) & 
                (svi_data['COUNTY'] == county_name)
            ].copy()
            
            county_svi['FIPS'] = county_svi['FIPS'].astype(str)
            county_svi['RPL_THEMES'] = county_svi['RPL_THEMES'].replace(-999, np.nan)
            
            # Extract socioeconomic control variables
            desired_columns = [
                'FIPS', 'LOCATION', 'RPL_THEMES', 'E_TOTPOP', 'E_HU',
                'E_NOVEH', 'EP_NOVEH',
                'E_POV', 'EP_POV150', 'E_UNEMP', 'EP_UNEMP',
                'E_NOHSDP', 'EP_NOHSDP',
                'E_UNINSUR', 'EP_UNINSUR',
                'E_MOBILE', 'EP_MOBILE', 'E_CROWD', 'EP_CROWD'
            ]
            
            available_cols = [col for col in desired_columns if col in county_svi.columns]
            county_svi = county_svi[available_cols].copy()
            
            for col in county_svi.columns:
                if col not in ['FIPS', 'LOCATION']:
                    county_svi[col] = county_svi[col].replace(-999, np.nan)
            
            valid_count = county_svi['RPL_THEMES'].notna().sum()
            self._log(f"Loaded {len(county_svi)} tracts ({valid_count} with valid SVI)")
            
            return county_svi
            
        except Exception as e:
            self._log(f"Error loading SVI data: {str(e)}")
            raise

    def get_tract_socioeconomic_features(self, tract_fips: str, svi_data: pd.DataFrame) -> Dict:
        """Extract socioeconomic control features for a tract"""
        
        tract_fips = str(tract_fips).strip()
        tract_data = svi_data[svi_data['FIPS'] == tract_fips]
        
        if len(tract_data) == 0:
            raise ValueError(
                f"No SVI data found for tract {tract_fips}. "
                f"Ensure the tract exists in the SVI dataset."
            )
        
        tract = tract_data.iloc[0]
        
        def safe_get(column, default):
            val = tract.get(column, default)
            return float(val) if pd.notna(val) else float(default)
        
        features = {
            'pct_no_vehicle': safe_get('EP_NOVEH', 10.0),
            'pct_poverty': safe_get('EP_POV150', 15.0),
            'pct_unemployed': safe_get('EP_UNEMP', 5.0),
            'pct_no_hs_diploma': safe_get('EP_NOHSDP', 10.0),
            'pct_uninsured': safe_get('EP_UNINSUR', 12.0),
            'pct_mobile_homes': safe_get('EP_MOBILE', 5.0),
            'pct_crowded': safe_get('EP_CROWD', 2.0),
            'population': safe_get('E_TOTPOP', 2000),
            'housing_units': safe_get('E_HU', 800)
        }
        
        return features

    def _default_socioeconomic_features(self) -> Dict:
        """Default socioeconomic features when data is missing"""
        return {
            'pct_no_vehicle': 10.0,
            'pct_poverty': 15.0,
            'pct_unemployed': 5.0,
            'pct_no_hs_diploma': 10.0,
            'pct_uninsured': 12.0,
            'pct_mobile_homes': 5.0,
            'pct_crowded': 2.0,
            'population': 2000.0,
            'housing_units': 800.0
        }

    def _get_tract_svi(self, tract_fips: str, svi_data: pd.DataFrame) -> float:
        """Get tract-level SVI value for a given FIPS code."""
        tract_fips = str(tract_fips).strip()
        tract_data = svi_data[svi_data['FIPS'].astype(str).str.strip() == tract_fips]
        
        if len(tract_data) == 0:
            self._log(f"WARNING: No SVI data for tract {tract_fips}, using default 0.5")
            return 0.5
        
        svi = tract_data.iloc[0].get('RPL_THEMES', 0.5)
        if pd.isna(svi) or svi < 0:
            return 0.5
        
        return float(svi)

    def _get_county_name(self, state_fips: str, county_fips: str) -> str:
        """Map FIPS codes to county name"""
        county_map = {
            ('47', '065'): 'Hamilton'
        }
        return county_map.get((state_fips, county_fips), 'Unknown')

    # =========================================================================
    # CONTEXT FEATURES
    # =========================================================================

    def create_context_features_for_addresses(self, addresses: gpd.GeoDataFrame, 
                                            svi_data: pd.DataFrame,
                                            include_tract_svi: bool = False) -> np.ndarray:
        """Create context feature matrix for addresses."""
        n_addresses = len(addresses)
        context_feature_list = []
        
        for idx, address in addresses.iterrows():
            tract_fips = str(address['tract_fips']).strip()
            tract_features = self.get_tract_socioeconomic_features(tract_fips, svi_data)
            tract_svi = self._get_tract_svi(tract_fips, svi_data)
            
            if include_tract_svi:
                context_vec = [
                    tract_svi,
                    tract_features['pct_no_vehicle'],
                    tract_features['pct_poverty'],
                    tract_features['pct_unemployed'],
                    tract_features['pct_no_hs_diploma'],
                    tract_features['population'] / 10000.0,
                ]
            else:
                context_vec = [
                    tract_features['pct_no_vehicle'],
                    tract_features['pct_poverty'],
                    tract_features['pct_unemployed'],
                    tract_features['pct_no_hs_diploma'],
                    tract_features['population'] / 10000.0,
                ]
            
            context_feature_list.append(context_vec)
        
        context_features = np.array(context_feature_list, dtype=np.float32)
        
        if self.verbose:
            self._log(f"Created context features: {context_features.shape}")
        
        return context_features

    def normalize_context_features(self, context_features: np.ndarray) -> Tuple[np.ndarray, object]:
        """Normalize context features using robust scaling."""
        from sklearn.preprocessing import RobustScaler
        
        scaler = RobustScaler()
        normalized = scaler.fit_transform(context_features)
        
        if self.verbose:
            self._log(f"Normalized context features: {normalized.shape}")
        
        return normalized, scaler

    def _load_lehd_employment(self, state_fips: str = '47', county_fips: str = '065') -> gpd.GeoDataFrame:
        """Load LEHD employment data from WAC files"""
        self._log(f"Loading LEHD employment data for {state_fips}{county_fips}...")
        
        lehd_file = os.path.join(self.data_dir, 'raw', 'lehd', 'tn_wac_S000_JT00_2021.csv')
        block_file = os.path.join(self.data_dir, 'raw', 'lehd', 'tl_2021_47_tabblock20.shp')
        
        if not os.path.exists(lehd_file):
            raise FileNotFoundError(f"LEHD data not found: {lehd_file}")
        
        lehd_data = pd.read_csv(lehd_file, dtype={'w_geocode': str})
        
        target_prefix = f"{state_fips}{county_fips}"
        county_data = lehd_data[lehd_data['w_geocode'].str.startswith(target_prefix)].copy()
        
        if len(county_data) == 0:
            raise ValueError(f"No employment data found for county {target_prefix}")
        
        self._log(f"Found {len(county_data)} employment blocks in county")
        
        # Aggregate by block
        block_employment = county_data.groupby('w_geocode').agg({
            'C000': 'sum',
            'CE01': 'sum',
            'CE02': 'sum',
            'CE03': 'sum',
        }).reset_index()
        
        block_employment.columns = ['block_geoid', 'total_jobs', 'low_wage', 'mid_wage', 'high_wage']
        
        if not os.path.exists(block_file):
            raise FileNotFoundError(f"Block geometries not found: {block_file}")
        
        blocks = gpd.read_file(block_file)
        blocks = blocks[blocks['COUNTYFP20'] == county_fips].copy()
        blocks['block_geoid'] = blocks['GEOID20'].astype(str)
        
        employment_gdf = blocks.merge(block_employment, on='block_geoid', how='inner')
        
        min_jobs = 10
        employment_gdf = employment_gdf[employment_gdf['total_jobs'] >= min_jobs].copy()
        
        employment_gdf['geometry'] = employment_gdf.geometry.centroid
        employment_gdf = employment_gdf.to_crs('EPSG:4326')
        
        employment_gdf['dest_id'] = range(len(employment_gdf))
        employment_gdf['dest_type'] = 'employment'
        employment_gdf['name'] = 'Employment Block ' + employment_gdf['block_geoid'].str[-4:]
        employment_gdf['employees'] = employment_gdf['total_jobs']
        
        self._log(f"Loaded {len(employment_gdf)} employment locations")
        
        return employment_gdf[['dest_id', 'name', 'dest_type', 'employees', 'geometry']]

    def _load_healthcare_facilities(self) -> gpd.GeoDataFrame:
        """Load healthcare facilities from CMS/local data"""
        self._log("Loading healthcare facilities...")
        
        healthcare_file = os.path.join(self.data_dir, 'raw', 'healthcare', 'hamilton_county_healthcare.csv')
        
        if not os.path.exists(healthcare_file):
            raise FileNotFoundError(f"Healthcare data not found: {healthcare_file}")
        
        healthcare_df = pd.read_csv(healthcare_file)

        lon_col = 'longitude' if 'longitude' in healthcare_df.columns else 'lon'
        lat_col = 'latitude' if 'latitude' in healthcare_df.columns else 'lat'
        
        if lon_col not in healthcare_df.columns or lat_col not in healthcare_df.columns:
            raise ValueError(f"Healthcare CSV missing coordinate columns. Found: {healthcare_df.columns.tolist()}")
        
        geometry = [Point(xy) for xy in zip(healthcare_df[lon_col], healthcare_df[lat_col])]
        healthcare_gdf = gpd.GeoDataFrame(healthcare_df, geometry=geometry, crs='EPSG:4326')
        
        healthcare_gdf['dest_id'] = range(len(healthcare_gdf))
        healthcare_gdf['dest_type'] = 'healthcare'
        
        if 'facility_name' in healthcare_gdf.columns:
            healthcare_gdf['name'] = healthcare_gdf['facility_name']
        elif 'name' not in healthcare_gdf.columns:
            healthcare_gdf['name'] = 'Healthcare Facility ' + healthcare_gdf['dest_id'].astype(str)
        
        self._log(f"Loaded {len(healthcare_gdf)} healthcare facilities")
        
        return healthcare_gdf[['dest_id', 'name', 'dest_type', 'geometry']]

    def _load_grocery_stores(self) -> gpd.GeoDataFrame:
        """Load grocery stores from OSM data"""
        self._log("Loading grocery stores...")
        
        grocery_file = os.path.join(self.data_dir, 'raw', 'osm_grocery', 'hamilton_county_grocery_stores.csv')
        
        if not os.path.exists(grocery_file):
            raise FileNotFoundError(f"Grocery data not found: {grocery_file}")
        
        grocery_df = pd.read_csv(grocery_file)
        
        lon_col = 'longitude' if 'longitude' in grocery_df.columns else 'lon'
        lat_col = 'latitude' if 'latitude' in grocery_df.columns else 'lat'
        
        if lon_col not in grocery_df.columns or lat_col not in grocery_df.columns:
            raise ValueError(f"Grocery CSV missing coordinate columns. Found: {grocery_df.columns.tolist()}")
        
        geometry = [Point(xy) for xy in zip(grocery_df[lon_col], grocery_df[lat_col])]
        grocery_gdf = gpd.GeoDataFrame(grocery_df, geometry=geometry, crs='EPSG:4326')
        
        grocery_gdf['dest_id'] = range(len(grocery_gdf))
        grocery_gdf['dest_type'] = 'grocery'
        
        if 'name' not in grocery_gdf.columns:
            grocery_gdf['name'] = 'Grocery Store ' + grocery_gdf['dest_id'].astype(str)
        
        self._log(f"Loaded {len(grocery_gdf)} grocery stores")
        
        return grocery_gdf[['dest_id', 'name', 'dest_type', 'geometry']]

    # =========================================================================
    # DESTINATION CREATION
    # =========================================================================

    def create_employment_destinations(self, use_real_data: bool = False) -> gpd.GeoDataFrame:
        """Create employment destinations for accessibility analysis"""
        if use_real_data:
            try:
                self._log("Attempting to load REAL employment data...")
                employment_gdf = self._load_lehd_employment()
                self._log(f"Loaded {len(employment_gdf)} REAL employment locations")
                return employment_gdf
            except Exception as e:
                self._log(f"Error loading real employment data: {e}")
                self._log("Falling back to synthetic data")
        
        # Synthetic data fallback
        employers = [
            {'name': 'Downtown Chattanooga', 'lat': 35.0456, 'lon': -85.3097, 'employees': 5000, 'type': 'mixed'},
            {'name': 'Volkswagen Chattanooga', 'lat': 35.0614, 'lon': -85.1580, 'employees': 4000, 'type': 'manufacturing'},
            {'name': 'BlueCross BlueShield TN', 'lat': 35.0456, 'lon': -85.3097, 'employees': 3500, 'type': 'insurance'},
            {'name': 'Erlanger Health System', 'lat': 35.0539, 'lon': -85.3083, 'employees': 8000, 'type': 'healthcare'},
            {'name': 'University of Tennessee Chattanooga', 'lat': 35.0456, 'lon': -85.3011, 'employees': 2500, 'type': 'education'},
            {'name': 'Tennessee Valley Authority', 'lat': 35.0398, 'lon': -85.3062, 'employees': 1500, 'type': 'utilities'},
            {'name': 'Hamilton County Government', 'lat': 35.0456, 'lon': -85.3097, 'employees': 2000, 'type': 'government'},
            {'name': 'Hamilton Place Mall Area', 'lat': 35.0407, 'lon': -85.2111, 'employees': 3000, 'type': 'retail'},
            {'name': 'East Brainerd Business District', 'lat': 35.0156, 'lon': -85.2180, 'employees': 1500, 'type': 'mixed'},
            {'name': 'Northshore Business District', 'lat': 35.0722, 'lon': -85.2967, 'employees': 1200, 'type': 'mixed'}
        ]
        
        geometries = [Point(emp['lon'], emp['lat']) for emp in employers]
        employment_gdf = gpd.GeoDataFrame(employers, geometry=geometries, crs='EPSG:4326')
        employment_gdf['dest_id'] = range(len(employment_gdf))
        employment_gdf['dest_type'] = 'employment'
        
        self._log(f"Created {len(employment_gdf)} synthetic employment destinations")
        return employment_gdf

    def create_healthcare_destinations(self, use_real_data: bool = False) -> gpd.GeoDataFrame:
        """Create healthcare destinations for accessibility analysis"""
        if use_real_data:
            try:
                self._log("Attempting to load REAL healthcare data...")
                healthcare_gdf = self._load_healthcare_facilities()
                self._log(f"Loaded {len(healthcare_gdf)} REAL healthcare facilities")
                return healthcare_gdf
            except Exception as e:
                self._log(f"Error loading real healthcare data: {e}")
                self._log("Falling back to synthetic data")
        
        # Synthetic data fallback
        hospitals = [
            {'name': 'Erlanger Baroness Hospital', 'lat': 35.0539, 'lon': -85.3083, 'beds': 400, 'type': 'General'},
            {'name': 'CHI Memorial Hospital', 'lat': 35.0627, 'lon': -85.2985, 'beds': 300, 'type': 'General'},
            {'name': 'Parkridge Medical Center', 'lat': 35.0456, 'lon': -85.2597, 'beds': 368, 'type': 'General'},
            {'name': 'TriStar StoneCrest Medical Center', 'lat': 35.1156, 'lon': -85.2441, 'beds': 101, 'type': 'General'},
            {'name': 'Erlanger East Hospital', 'lat': 35.0407, 'lon': -85.2111, 'beds': 140, 'type': 'General'},
            {'name': 'Siskin Hospital', 'lat': 35.0456, 'lon': -85.3097, 'beds': 79, 'type': 'Rehabilitation'},
            {'name': 'Moccasin Bend Mental Health', 'lat': 35.0722, 'lon': -85.3365, 'beds': 150, 'type': 'Psychiatric'},
            {'name': 'Parkridge Valley Hospital', 'lat': 35.0175, 'lon': -85.3365, 'beds': 60, 'type': 'General'}
        ]
        
        geometries = [Point(hosp['lon'], hosp['lat']) for hosp in hospitals]
        healthcare_gdf = gpd.GeoDataFrame(hospitals, geometry=geometries, crs='EPSG:4326')
        healthcare_gdf['dest_id'] = range(len(healthcare_gdf))
        healthcare_gdf['dest_type'] = 'healthcare'
        
        self._log(f"Created {len(healthcare_gdf)} synthetic healthcare destinations")
        return healthcare_gdf

    def create_grocery_destinations(self, use_real_data: bool = False) -> gpd.GeoDataFrame:
        """Create grocery destinations for accessibility analysis"""
        if use_real_data:
            try:
                self._log("Attempting to load REAL grocery data...")
                grocery_gdf = self._load_grocery_stores()
                self._log(f"Loaded {len(grocery_gdf)} REAL grocery stores")
                return grocery_gdf
            except Exception as e:
                self._log(f"Error loading real grocery data: {e}")
                self._log("Falling back to synthetic data")
        
        # Synthetic data fallback
        stores = [
            {'name': 'Walmart Supercenter - Hamilton Place', 'lat': 35.0407, 'lon': -85.2111, 'type': 'supermarket'},
            {'name': 'Kroger - East Brainerd', 'lat': 35.0156, 'lon': -85.2180, 'type': 'supermarket'},
            {'name': 'Publix - Signal Mountain', 'lat': 35.1456, 'lon': -85.3456, 'type': 'supermarket'},
            {'name': 'Food City - Northgate', 'lat': 35.0722, 'lon': -85.2967, 'type': 'supermarket'},
            {'name': 'IGA - Downtown', 'lat': 35.0456, 'lon': -85.3097, 'type': 'grocery'},
            {'name': 'Walmart Neighborhood Market - Hixson', 'lat': 35.1256, 'lon': -85.2441, 'type': 'grocery'},
            {'name': 'Fresh Market - Northshore', 'lat': 35.0627, 'lon': -85.2985, 'type': 'grocery'},
            {'name': 'Bi-Lo - East Ridge', 'lat': 35.0495, 'lon': -85.1938, 'type': 'supermarket'},
            {'name': 'Save-A-Lot - South Chattanooga', 'lat': 35.0175, 'lon': -85.3365, 'type': 'discount'},
            {'name': 'Food Lion - East Chattanooga', 'lat': 35.0456, 'lon': -85.2580, 'type': 'supermarket'}
        ]
        
        geometries = [Point(store['lon'], store['lat']) for store in stores]
        grocery_gdf = gpd.GeoDataFrame(stores, geometry=geometries, crs='EPSG:4326')
        grocery_gdf['dest_id'] = range(len(grocery_gdf))
        grocery_gdf['dest_type'] = 'grocery'
        
        self._log(f"Created {len(grocery_gdf)} synthetic grocery destinations")
        return grocery_gdf

    # =========================================================================
    # ADDRESS LOADING
    # =========================================================================

    def load_address_points(self, state_fips: str = '47', county_fips: str = '065') -> gpd.GeoDataFrame:
        """Load real Chattanooga address points"""
        
        if self._address_cache is not None:
            self._log(f"Using cached address data ({len(self._address_cache)} addresses)")
            return self._address_cache
        
        self._log("Loading Chattanooga address data...")
        
        address_files = [
            os.path.join(self.data_dir, 'chattanooga.geojson'),
            os.path.join(self.data_dir, 'raw', 'chattanooga.geojson'),
            './chattanooga.geojson'
        ]
        
        addresses_gdf = None
        for address_file in address_files:
            if os.path.exists(address_file):
                try:
                    addresses_gdf = gpd.read_file(address_file)
                    
                    if len(addresses_gdf) > 0:
                        addresses_gdf = addresses_gdf.copy()
                        
                        if 'address_id' not in addresses_gdf.columns:
                            addresses_gdf['address_id'] = range(len(addresses_gdf))
                        
                        if addresses_gdf.crs is None:
                            addresses_gdf.set_crs(epsg=4326, inplace=True)
                        elif addresses_gdf.crs != 'EPSG:4326':
                            addresses_gdf = addresses_gdf.to_crs('EPSG:4326')
                        
                        keep_cols = ['address_id', 'geometry']
                        if 'full_address' in addresses_gdf.columns:
                            keep_cols.append('full_address')
                        elif 'street' in addresses_gdf.columns:
                            addresses_gdf['full_address'] = addresses_gdf['street'].fillna('Unknown Address')
                            keep_cols.append('full_address')
                        else:
                            addresses_gdf['full_address'] = 'Address ' + addresses_gdf['address_id'].astype(str)
                            keep_cols.append('full_address')
                        
                        addresses_gdf = addresses_gdf[keep_cols]
                        self._log(f"Loaded {len(addresses_gdf)} real addresses from {address_file}")
                        break
                        
                except Exception as e:
                    self._log(f"Error loading {address_file}: {str(e)}")
                    continue
        
        if addresses_gdf is None or len(addresses_gdf) == 0:
            self._log("WARNING: No real address data found, creating synthetic addresses")
            addresses_gdf = self._create_synthetic_addresses(state_fips, county_fips)
        
        self._address_cache = addresses_gdf
        self._log(f"Final address count: {len(addresses_gdf)}")
        return addresses_gdf

    def get_addresses_for_tract(self, fips_code: str) -> gpd.GeoDataFrame:
        """Get addresses within a specific census tract"""
        
        try:
            all_addresses = self.load_address_points()
            
            if len(all_addresses) == 0:
                return gpd.GeoDataFrame(columns=['address_id', 'geometry'])
            
            state_fips = fips_code[:2]
            county_fips = fips_code[2:5]
            tracts = self.load_census_tracts(state_fips, county_fips)
            
            tract = tracts[tracts['FIPS'] == fips_code]
            if len(tract) == 0:
                self._log(f"Tract {fips_code} not found")
                return gpd.GeoDataFrame(columns=['address_id', 'geometry'])
            
            tract_geom = tract.iloc[0].geometry
            
            tract_addresses = all_addresses[all_addresses.geometry.within(tract_geom)].copy()
            tract_addresses['tract_fips'] = fips_code
            
            self._log(f"Found {len(tract_addresses)} addresses in tract {fips_code}")
            return tract_addresses
            
        except Exception as e:
            self._log(f"Error getting addresses for tract {fips_code}: {str(e)}")
            return gpd.GeoDataFrame(columns=['address_id', 'geometry'])

    def _create_synthetic_addresses(self, state_fips: str, county_fips: str, density: int = 200) -> gpd.GeoDataFrame:
        """Create synthetic addresses as fallback"""
        
        try:
            tracts = self.load_census_tracts(state_fips, county_fips)
            county_boundary = tracts.geometry.unary_union
            
            bounds = county_boundary.bounds
            addresses = []
            address_id = 0
            
            attempts = 0
            max_attempts = density * 10
            
            while len(addresses) < density and attempts < max_attempts:
                x = np.random.uniform(bounds[0], bounds[2])
                y = np.random.uniform(bounds[1], bounds[3])
                point = Point(x, y)
                
                if county_boundary.contains(point):
                    addresses.append({
                        'address_id': address_id,
                        'geometry': point,
                        'full_address': f'Synthetic Address {address_id}'
                    })
                    address_id += 1
                
                attempts += 1
            
            addresses_gdf = gpd.GeoDataFrame(addresses, crs='EPSG:4326')
            self._log(f"Created {len(addresses_gdf)} synthetic addresses")
            
            return addresses_gdf
            
        except Exception as e:
            self._log(f"Error creating synthetic addresses: {str(e)}")
            return gpd.GeoDataFrame(columns=['address_id', 'geometry', 'full_address'])

    def get_neighboring_tracts(self, target_fips: str, n_neighbors: int = 4) -> List[str]:
        """Get neighboring tracts by geographic proximity"""
        
        if n_neighbors == 0:
            return [target_fips]
        
        state_fips = target_fips[:2]
        county_fips = target_fips[2:5]
        all_tracts = self.load_census_tracts(state_fips, county_fips)
        
        target_tract = all_tracts[all_tracts['FIPS'] == target_fips]
        if len(target_tract) == 0:
            self._log(f"Target tract {target_fips} not found")
            return [target_fips]
        
        target_geom = target_tract.geometry.iloc[0]
        all_tracts['distance'] = all_tracts.geometry.distance(target_geom)
        neighbors = all_tracts[all_tracts['FIPS'] != target_fips].nsmallest(n_neighbors, 'distance')
        
        tract_list = [target_fips] + neighbors['FIPS'].tolist()
        
        self._log(f"Multi-tract training with {len(tract_list)} tracts:")
        for fips in tract_list:
            self._log(f"  {fips}")
        
        return tract_list

    def select_diverse_training_tracts(self, state_fips: str, county_fips: str, 
                                    n_tracts: int = 12, seed: int = 42,
                                    exclude_fips: List[str] = None) -> List[str]:
        """Select diverse tracts spanning SVI spectrum for global training."""
        np.random.seed(seed)
        
        all_tracts = self.load_census_tracts(state_fips, county_fips)
        county_name = self._get_county_name(state_fips, county_fips)
        svi_data = self.load_svi_data(state_fips, county_name)
        
        tracts_with_svi = all_tracts.merge(svi_data, on='FIPS', how='inner')
        tracts_with_svi = tracts_with_svi[tracts_with_svi['RPL_THEMES'].notna()].copy()
        
        if exclude_fips:
            tracts_with_svi = tracts_with_svi[~tracts_with_svi['FIPS'].isin(exclude_fips)]
        
        tracts_with_svi['svi_quintile'] = pd.qcut(
            tracts_with_svi['RPL_THEMES'],
            q=5,
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
            duplicates='drop'
        )
        
        selected_tracts = []
        tracts_per_quintile = max(2, n_tracts // 5)
        
        for quintile in ['Very Low', 'Low', 'Medium', 'High', 'Very High']:
            quintile_tracts = tracts_with_svi[tracts_with_svi['svi_quintile'] == quintile]
            
            if len(quintile_tracts) > 0:
                n_sample = min(tracts_per_quintile, len(quintile_tracts))
                sample = quintile_tracts.sample(n=n_sample, random_state=seed)
                selected_tracts.extend(sample['FIPS'].tolist())
        
        if len(selected_tracts) < n_tracts:
            remaining = tracts_with_svi[~tracts_with_svi['FIPS'].isin(selected_tracts)]
            additional = remaining.sample(n=n_tracts - len(selected_tracts), random_state=seed)
            selected_tracts.extend(additional['FIPS'].tolist())
        
        selected_tracts = selected_tracts[:n_tracts]
        
        self._log(f"Selected {len(selected_tracts)} diverse training tracts")
        return selected_tracts

    # =========================================================================
    # ROAD NETWORK AND GRAPH CONSTRUCTION
    # =========================================================================

    def load_road_network(self, roads_file: Optional[str] = None, 
                        state_fips: str = '47', county_fips: str = '065') -> gpd.GeoDataFrame:
        """Load road network data"""
        self._log("Loading road network...")
        
        try:
            if roads_file and os.path.exists(roads_file):
                roads = gpd.read_file(roads_file)
                self._log(f"Loaded {len(roads)} road segments from {roads_file}")
            else:
                default_file = os.path.join(
                    self.data_dir, 'raw', 
                    f'tl_2023_{state_fips}{county_fips}_roads.shp'
                )
                
                if os.path.exists(default_file):
                    roads = gpd.read_file(default_file)
                    self._log(f"Loaded {len(roads)} road segments")
                else:
                    self._log("Road file not found, will use geographic fallback")
                    return gpd.GeoDataFrame(geometry=[])
            
            if roads.crs is None:
                roads.set_crs(epsg=4326, inplace=True)
            
            return roads
            
        except Exception as e:
            self._log(f"Error loading roads: {str(e)}")
            return gpd.GeoDataFrame(geometry=[])

    def create_spatial_accessibility_graph(self, addresses, accessibility_features, 
                                            state_fips='47', county_fips='065',
                                            context_features=None):
        """Create graph with context features for context-aware learning"""
        import torch
        from torch_geometric.data import Data
        
        n_addresses = len(addresses)
        self._log(f"Creating road network graph for {n_addresses} addresses...")
        
        roads = self.load_road_network(state_fips=state_fips, county_fips=county_fips)
        
        if len(roads) > 0:
            edge_index, edge_weight = self._create_road_network_graph(addresses, roads)
        else:
            self._log("No road network available, using geographic connectivity")
            edge_index, edge_weight = self._create_geographic_fallback_graph(addresses)
        
        if edge_index.shape[1] == 0:
            self._log("WARNING: No edges created, adding minimal connectivity")
            edge_index, edge_weight = self._create_minimal_connectivity(n_addresses)
        
        node_features = torch.FloatTensor(accessibility_features)
        
        if context_features is not None:
            context_tensor = torch.FloatTensor(context_features)
            self._log(f"Added context features: {context_tensor.shape}")
            
            graph_data = Data(
                x=node_features,
                context=context_tensor,
                edge_index=edge_index,
                edge_attr=edge_weight
            )
        else:
            graph_data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_weight
            )
        
        self._log(f"Created graph: {n_addresses} nodes, {edge_index.shape[1]//2} undirected edges")
        
        return graph_data
    
    def _create_minimal_connectivity(self, n_addresses):
        """Create minimal ring connectivity as absolute fallback"""
        import torch
        
        if n_addresses < 2:
            edge_index = torch.LongTensor([[0], [0]])
            edge_weight = torch.FloatTensor([1.0])
            return edge_index, edge_weight
        
        edge_list = []
        edge_weights = []
        
        for i in range(n_addresses):
            next_i = (i + 1) % n_addresses
            edge_list.extend([[i, next_i], [next_i, i]])
            edge_weights.extend([1.0, 1.0])
        
        edge_index = torch.LongTensor(edge_list).t().contiguous()
        edge_weight = torch.FloatTensor(edge_weights)
        
        return edge_index, edge_weight

    def _create_road_network_graph(self, addresses, roads):
        """Create graph with road network-based connectivity"""
        import torch
        from sklearn.neighbors import NearestNeighbors
        
        n_addresses = len(addresses)
        
        try:
            road_graph, address_to_road_mapping = self._create_road_connectivity(addresses, roads)
            network_edges = self._extract_network_edges(road_graph, address_to_road_mapping)
        except Exception as e:
            self._log(f"Road network creation failed: {str(e)}")
            network_edges = []
        
        geographic_edges = self._create_geographic_edges(addresses)
        all_edges = network_edges + geographic_edges
        
        edge_set = set()
        edge_list = []
        edge_weights = []
        edge_types = []
        
        for edge in all_edges:
            if (edge['from'] < 0 or edge['from'] >= n_addresses or 
                edge['to'] < 0 or edge['to'] >= n_addresses or
                edge['from'] == edge['to']):
                continue
                
            edge_key = tuple(sorted([edge['from'], edge['to']]))
            if edge_key not in edge_set:
                edge_set.add(edge_key)
                edge_list.extend([[edge['from'], edge['to']], [edge['to'], edge['from']]])
                edge_weights.extend([edge['weight'], edge['weight']])
                edge_types.extend([edge['type'], edge['type']])
        
        if edge_list and len(edge_list) > 0:
            edge_index = torch.LongTensor(edge_list).t().contiguous()
            edge_weight = torch.FloatTensor(edge_weights)
            
            if edge_index.max() >= n_addresses:
                self._log("ERROR: Edge indices exceed number of nodes")
                edge_index, edge_weight = self._create_minimal_connectivity(n_addresses)
        else:
            self._log("No valid edges created, using minimal connectivity")
            edge_index, edge_weight = self._create_minimal_connectivity(n_addresses)
        
        network_count = sum(1 for t in edge_types if t == 'network')
        geographic_count = sum(1 for t in edge_types if t == 'geographic')
        
        self._log(f"  Network edges: {network_count//2}")
        self._log(f"  Geographic edges: {geographic_count//2}")
        
        return edge_index, edge_weight

    def _create_road_connectivity(self, addresses, roads):
        """Create NetworkX graph from road segments and map addresses to roads"""
        from sklearn.neighbors import BallTree
        
        self._log("Building road network graph...")
        
        road_graph = nx.Graph()
        
        for idx, road in roads.iterrows():
            if road.geometry.geom_type == 'LineString':
                coords = list(road.geometry.coords)
                
                for i in range(len(coords) - 1):
                    u, v = coords[i], coords[i + 1]
                    length = ((u[0] - v[0])**2 + (u[1] - v[1])**2)**0.5 * 111000
                    
                    road_graph.add_edge(
                        u, v, 
                        length=length,
                        road_id=idx,
                        road_type=road.get('RTTYP', 'unknown')
                    )
        
        if road_graph.number_of_nodes() == 0:
            self._log("No road nodes found")
            return road_graph, {}
        
        road_nodes = np.array(list(road_graph.nodes()))
        tree = BallTree(np.radians(road_nodes), metric='haversine')
        
        address_to_road_mapping = {}
        address_coords = np.array([[addr.geometry.x, addr.geometry.y] for _, addr in addresses.iterrows()])
        
        distances, indices = tree.query(np.radians(address_coords), k=1)
        distances = distances.flatten() * 6371000
        
        for i, (addr_idx, addr) in enumerate(addresses.iterrows()):
            if distances[i] < 500:
                nearest_road_node = tuple(road_nodes[indices[i][0]])
                address_to_road_mapping[i] = nearest_road_node
        
        self._log(f"Mapped {len(address_to_road_mapping)}/{len(addresses)} addresses to road network")
        
        return road_graph, address_to_road_mapping

    def _extract_network_edges(self, road_graph, address_to_road_mapping):
        """Extract address-to-address edges via road network connectivity"""
        from sklearn.neighbors import NearestNeighbors
        
        network_edges = []
        road_connected_addresses = list(address_to_road_mapping.keys())
        
        if len(road_connected_addresses) < 2:
            return network_edges
        
        self._log(f"Computing network connectivity for {len(road_connected_addresses)} road-connected addresses...")
        
        address_coords = []
        for addr_idx in road_connected_addresses:
            road_node = address_to_road_mapping[addr_idx]
            address_coords.append([road_node[0], road_node[1]])
        
        address_coords = np.array(address_coords)
        
        max_neighbors = min(20, len(road_connected_addresses) - 1)
        nbrs = NearestNeighbors(n_neighbors=max_neighbors, metric='euclidean').fit(address_coords)
        distances, indices = nbrs.kneighbors(address_coords)
        
        for i, addr1_idx in enumerate(road_connected_addresses):
            road_node1 = address_to_road_mapping[addr1_idx]
            
            for j_idx in range(1, min(10, len(indices[i]))):
                j = indices[i][j_idx]
                addr2_idx = road_connected_addresses[j]
                road_node2 = address_to_road_mapping[addr2_idx]
                
                if road_node1 == road_node2:
                    continue
                
                geo_distance = distances[i][j_idx] * 111000
                
                if geo_distance < 1000:
                    try:
                        path_length = nx.shortest_path_length(
                            road_graph, road_node1, road_node2, 
                            weight='length'
                        )
                        
                        if path_length > 1500:
                            continue
                        
                        weight = 1.0 / (1.0 + path_length / 500.0)
                        
                        network_edges.append({
                            'from': addr1_idx,
                            'to': addr2_idx,
                            'weight': weight,
                            'type': 'network',
                            'road_distance': path_length
                        })
                        
                    except (nx.NetworkXNoPath, nx.NetworkXError):
                        continue
            
            if (i + 1) % 100 == 0:
                self._log(f"  Processed {i + 1}/{len(road_connected_addresses)} addresses...")
        
        self._log(f"Created {len(network_edges)} network edges")
        return network_edges

    def _create_geographic_edges(self, addresses, max_neighbors=6):
        """Create edges based on geographic proximity as fallback"""
        from sklearn.neighbors import NearestNeighbors
        
        coords = np.array([[addr.geometry.x, addr.geometry.y] for _, addr in addresses.iterrows()])
        n_addresses = len(addresses)
        
        k_neighbors = min(max_neighbors, n_addresses - 1)
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        geographic_edges = []
        
        for i in range(n_addresses):
            for j_idx in range(1, len(indices[i])):
                j = indices[i][j_idx]
                distance_deg = distances[i][j_idx]
                distance_m = distance_deg * 111000
                
                if distance_m < 1000:
                    weight = np.exp(-distance_m / 300.0)
                    
                    geographic_edges.append({
                        'from': i,
                        'to': j,
                        'weight': weight,
                        'type': 'geographic',
                        'distance': distance_m
                    })
        
        return geographic_edges

    def _create_geographic_fallback_graph(self, addresses):
        """Fallback to geographic connectivity when no road network is available"""
        import torch
        from sklearn.neighbors import NearestNeighbors
        
        n_addresses = len(addresses)
        coords = np.array([[addr.geometry.x, addr.geometry.y] for _, addr in addresses.iterrows()])
        
        if n_addresses < 100:
            k_neighbors = min(8, n_addresses - 1)
        elif n_addresses < 500:
            k_neighbors = min(12, n_addresses - 1)
        else:
            k_neighbors = min(16, n_addresses - 1)
        
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='euclidean').fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        edge_list = []
        edge_weights = []
        
        for i in range(n_addresses):
            for j_idx in range(1, len(indices[i])):
                j = indices[i][j_idx]
                distance_deg = distances[i][j_idx]
                distance_m = distance_deg * 111000
                
                if distance_m < 1000:
                    weight = np.exp(-distance_m / 300.0)
                    edge_list.extend([[i, j], [j, i]])
                    edge_weights.extend([weight, weight])
        
        if not edge_list:
            for i in range(n_addresses):
                if len(indices[i]) > 1:
                    j = indices[i][1]
                    edge_list.extend([[i, j], [j, i]])
                    edge_weights.extend([1.0, 1.0])
        
        edge_index = torch.LongTensor(edge_list).t().contiguous()
        edge_weight = torch.FloatTensor(edge_weights)
        
        return edge_index, edge_weight

    # =========================================================================
    # TRAVEL TIME AND ACCESSIBILITY COMPUTATION
    # =========================================================================

    def calculate_multimodal_travel_times_batch(self, origins: gpd.GeoDataFrame, 
                                            destinations: gpd.GeoDataFrame,
                                            time_periods: list = ['morning']) -> pd.DataFrame:
        """Network-based travel time calculation using enhanced computation"""
        self._log("Using enhanced accessibility computation...")
        
        time_period = time_periods[0] if time_periods else 'morning'
        
        travel_times = self.accessibility_computer.calculate_realistic_travel_times(
            origins=origins,
            destinations=destinations, 
            time_period=time_period
        )
        
        return travel_times

    def compute_accessibility_features(self, addresses: gpd.GeoDataFrame) -> np.ndarray:
        """Compute comprehensive accessibility features for all addresses"""
        self._log(f"Computing accessibility features for {len(addresses)} addresses...")
        
        destinations = {
            'employment': self.create_employment_destinations(),
            'healthcare': self.create_healthcare_destinations(),
            'grocery': self.create_grocery_destinations()
        }
        
        all_features = []
        
        for dest_type, dest_gdf in destinations.items():
            self._log(f"  Computing {dest_type} accessibility...")
            
            travel_times = self._calculate_simple_travel_times(addresses, dest_gdf)
            
            features = self._extract_accessibility_features_from_times(
                addresses, dest_gdf, travel_times, dest_type
            )
            
            all_features.append(features)
        
        accessibility_matrix = np.column_stack(all_features)
        derived_features = self._compute_derived_features(accessibility_matrix)
        final_features = np.column_stack([accessibility_matrix, derived_features])
        
        self._log(f"Final accessibility features: {final_features.shape}")
        return final_features

    def _calculate_simple_travel_times(self, addresses: gpd.GeoDataFrame, 
                                     destinations: gpd.GeoDataFrame) -> pd.DataFrame:
        """Simplified travel time calculation using distance approximation"""
        results = []
        
        for addr_idx, address in addresses.iterrows():
            addr_id = address.get('address_id', addr_idx)
            addr_point = address.geometry
            
            for dest_idx, destination in destinations.iterrows():
                dest_id = destination.get('dest_id', dest_idx)
                dest_point = destination.geometry
                
                distance_deg = addr_point.distance(dest_point)
                distance_km = distance_deg * 111
                
                walk_time = distance_km / 5.0 * 60
                drive_time = distance_km / 30.0 * 60
                
                if 2 <= distance_km <= 15:
                    transit_base_time = distance_km / 15.0 * 60
                    transit_wait_time = 10
                    transit_time = transit_base_time + transit_wait_time
                else:
                    transit_time = walk_time * 1.5
                
                times = {'walk': walk_time, 'drive': drive_time, 'transit': transit_time}
                best_mode = min(times.keys(), key=lambda k: times[k])
                combined_time = times[best_mode]
                
                results.append({
                    'origin_id': addr_id,
                    'destination_id': dest_id,
                    'destination_type': destination.get('dest_type', 'unknown'),
                    'walk_time': walk_time,
                    'drive_time': drive_time,
                    'transit_time': transit_time,
                    'combined_time': combined_time,
                    'best_mode': best_mode
                })
        
        return pd.DataFrame(results)

    def _extract_accessibility_features_from_times(self, addresses: gpd.GeoDataFrame,
                                                 destinations: gpd.GeoDataFrame,
                                                 travel_times: pd.DataFrame,
                                                 dest_type: str) -> np.ndarray:
        """Extract 8 accessibility features for one destination type"""
        
        features = []
        
        for addr_idx, address in addresses.iterrows():
            addr_id = address.get('address_id', addr_idx)
            addr_times = travel_times[travel_times['origin_id'] == addr_id]
            
            if len(addr_times) > 0:
                combined_times = addr_times['combined_time'].values
                
                min_time = float(np.min(combined_times))
                mean_time = float(np.mean(combined_times))
                percentile_90 = float(np.percentile(combined_times, 90))
                
                count_5min = int(np.sum(combined_times <= 5))
                count_10min = int(np.sum(combined_times <= 10))
                count_15min = int(np.sum(combined_times <= 15))
                
                transit_trips = addr_times['best_mode'] == 'transit'
                transit_share = float(transit_trips.mean())
                
                accessibility_score = float(np.sum(1.0 / np.maximum(combined_times, 1.0)))
                
            else:
                min_time = mean_time = percentile_90 = 120.0
                count_10min = count_5min = count_15min = 0
                transit_share = accessibility_score = 0.0
            
            features.append([
                min_time, mean_time, percentile_90,
                count_5min, count_10min, count_15min,
                transit_share, accessibility_score
            ])
        
        return np.array(features, dtype=np.float64)

    def _compute_derived_features(self, base_features: np.ndarray) -> np.ndarray:
        """Compute derived accessibility features from base metrics"""
        
        n_addresses = base_features.shape[0]
        
        if base_features.shape[1] < 24:
            return np.zeros((n_addresses, 4), dtype=np.float64)
        
        derived = []
        
        for i in range(n_addresses):
            emp_score = base_features[i, 7]
            health_score = base_features[i, 15]
            grocery_score = base_features[i, 23]
            
            total_accessibility = emp_score + health_score + grocery_score
            
            if total_accessibility > 0:
                scores = np.array([emp_score, health_score, grocery_score]) / total_accessibility
                scores = np.maximum(scores, 1e-8)
                balance = -np.sum(scores * np.log(scores))
            else:
                balance = 0.0
            
            emp_transit = base_features[i, 6]
            health_transit = base_features[i, 14]
            grocery_transit = base_features[i, 22]
            avg_transit_dependence = (emp_transit + health_transit + grocery_transit) / 3
            
            emp_min, emp_mean = base_features[i, 0], base_features[i, 1]
            health_min, health_mean = base_features[i, 8], base_features[i, 9]
            grocery_min, grocery_mean = base_features[i, 16], base_features[i, 17]
            
            all_mins = [emp_min, health_min, grocery_min]
            all_means = [emp_mean, health_mean, grocery_mean]
            
            min_avg = np.mean(all_mins)
            mean_avg = np.mean(all_means)
            
            time_efficiency = (mean_avg - min_avg) / mean_avg if mean_avg > 0 else 0
            
            derived.append([
                total_accessibility,
                balance,
                avg_transit_dependence,
                time_efficiency
            ])
        
        return np.array(derived, dtype=np.float64)

    def load_block_groups_for_validation(self, state_fips: str, county_fips: str):
        '''
        Load block group geometries and demographics for validation.
        
        Returns:
            Tuple of (geometries_gdf, demographics_df) or None if not available
        '''
        from granite.data.block_group_loader import BlockGroupLoader
        
        try:
            loader = BlockGroupLoader(
                data_dir=self.data_dir,
                verbose=self.verbose
            )
            
            geometries = loader.load_block_group_geometries(state_fips, county_fips)
            demographics = loader.fetch_acs_demographics(state_fips, county_fips)
            
            if geometries is not None and demographics is not None:
                return (geometries, demographics)
            else:
                return None
                
        except Exception as e:
            if self.verbose:
                print(f"Could not load block group data: {e}")
            return None

    # =========================================================================
    # ENHANCED DESTINATION METHODS (bound at init)
    # =========================================================================

    def bind_enhanced_destination_methods(self):
        """Bind the enhanced destination methods to DataLoader instance"""
        
        def create_enhanced_destinations_for_tract(self, tract_addresses, existing_destinations):
            """Create enhanced destination set appropriate for intra-tract analysis"""
            self._log("Creating enhanced destinations for intra-tract analysis...")
            
            enhanced_destinations = {}
            
            tract_bounds = tract_addresses.geometry.total_bounds
            tract_center = Point(
                (tract_bounds[0] + tract_bounds[2]) / 2,
                (tract_bounds[1] + tract_bounds[3]) / 2
            )
            
            tract_width = tract_bounds[2] - tract_bounds[0]
            tract_height = tract_bounds[3] - tract_bounds[1]
            
            for dest_type, existing_dests in existing_destinations.items():
                self._log(f"  Enhancing {dest_type} destinations...")
                
                enhanced_dests = []
                max_distance_deg = max(tract_width, tract_height) * 3
                
                for _, dest in existing_dests.iterrows():
                    dist_to_center = tract_center.distance(dest.geometry)
                    if dist_to_center <= max_distance_deg:
                        enhanced_dests.append({
                            'name': dest.get('name', f'{dest_type}_existing'),
                            'geometry': dest.geometry,
                            'dest_id': len(enhanced_dests),
                            'dest_type': dest_type,
                            'scale': 'regional',
                            'importance': 1.0
                        })
                
                local_dests = self._create_local_destinations(
                    tract_center, tract_width, tract_height, dest_type
                )
                enhanced_dests.extend(local_dests)
                
                edge_dests = self._create_edge_destinations(tract_bounds, dest_type)
                enhanced_dests.extend(edge_dests)
                
                if enhanced_dests:
                    enhanced_gdf = gpd.GeoDataFrame(enhanced_dests, crs='EPSG:4326')
                    enhanced_gdf['dest_id'] = range(len(enhanced_gdf))
                else:
                    enhanced_gdf = existing_dests.copy()
                    enhanced_gdf['scale'] = 'regional'
                    enhanced_gdf['importance'] = 1.0
                
                enhanced_destinations[dest_type] = enhanced_gdf
                self._log(f"    {dest_type}: {len(enhanced_gdf)} destinations")
            
            for dest_type, enhanced_gdf in enhanced_destinations.items():
                enhanced_gdf = self._remove_spatial_duplicates(enhanced_gdf, min_distance_m=150)
                if len(enhanced_gdf) > 20:
                    enhanced_gdf = enhanced_gdf.head(20)
                enhanced_destinations[dest_type] = enhanced_gdf
                self._log(f"    Final {dest_type}: {len(enhanced_gdf)} destinations")

            return enhanced_destinations

        def _create_local_destinations(self, tract_center, tract_width, tract_height, dest_type):
            """Create local destinations within and near the tract"""
            
            dest_params = {
                'employment': {
                    'names': ['Local Business District', 'Commercial Center', 'Office Complex', 'Retail Cluster'],
                    'count': 4,
                    'spread_factor': 0.8
                },
                'healthcare': {
                    'names': ['Neighborhood Clinic', 'Urgent Care', 'Medical Center', 'Health Services'],
                    'count': 3,
                    'spread_factor': 1.0
                },
                'grocery': {
                    'names': ['Neighborhood Market', 'Corner Store', 'Shopping Center', 'Local Grocery'],
                    'count': 4,
                    'spread_factor': 0.6
                }
            }
            
            params = dest_params.get(dest_type, {
                'names': [f'Local {dest_type.title()}'],
                'count': 2,
                'spread_factor': 0.8
            })
            
            local_destinations = []
            angles = np.linspace(0, 2*np.pi, params['count'], endpoint=False)
            
            for i, angle in enumerate(angles):
                base_distance = max(tract_width, tract_height) * params['spread_factor']
                distance_variation = np.random.uniform(0.3, 1.2)
                distance = base_distance * distance_variation
                
                x_offset = distance * np.cos(angle)
                y_offset = distance * np.sin(angle)
                
                dest_point = Point(
                    tract_center.x + x_offset,
                    tract_center.y + y_offset
                )
                
                name = params['names'][i % len(params['names'])]
                if len(params['names']) <= i:
                    name += f" #{i+1}"
                
                local_destinations.append({
                    'name': name,
                    'geometry': dest_point,
                    'dest_type': dest_type,
                    'scale': 'local',
                    'importance': np.random.uniform(0.6, 1.0)
                })
            
            return local_destinations

        def _create_edge_destinations(self, tract_bounds, dest_type):
            """Create destinations at tract edges for accessibility gradients"""
            
            minx, miny, maxx, maxy = tract_bounds
            width = maxx - minx
            height = maxy - miny
            
            edge_positions = [
                Point(minx + width * 0.5, maxy + height * 0.3),
                Point(minx + width * 0.5, miny - height * 0.3),
                Point(maxx + width * 0.3, miny + height * 0.5),
                Point(minx - width * 0.3, miny + height * 0.5)
            ]
            
            edge_names = [
                f'North {dest_type.title()}', f'South {dest_type.title()}', 
                f'East {dest_type.title()}', f'West {dest_type.title()}'
            ]
            
            edge_destinations = []
            for i, (position, name) in enumerate(zip(edge_positions, edge_names)):
                edge_destinations.append({
                    'name': name,
                    'geometry': position,
                    'dest_type': dest_type,
                    'scale': 'edge',
                    'importance': np.random.uniform(0.4, 0.8)
                })
            
            return edge_destinations

        def create_tract_appropriate_destinations(self, tract_fips):
            """Create tract-appropriate destinations"""
            
            tract_addresses = self.get_addresses_for_tract(tract_fips)
            
            if len(tract_addresses) == 0:
                self._log(f"No addresses found for tract {tract_fips}, using default destinations")
                return {
                    'employment': self.create_employment_destinations(),
                    'healthcare': self.create_healthcare_destinations(),
                    'grocery': self.create_grocery_destinations()
                }
            
            existing_destinations = {
                'employment': self.create_employment_destinations(),
                'healthcare': self.create_healthcare_destinations(),
                'grocery': self.create_grocery_destinations()
            }
            
            enhanced_destinations = self.create_enhanced_destinations_for_tract(
                tract_addresses, existing_destinations
            )
            
            return enhanced_destinations

        # Bind all methods to the instance
        self.create_enhanced_destinations_for_tract = types.MethodType(
            create_enhanced_destinations_for_tract, self
        )
        self._create_local_destinations = types.MethodType(
            _create_local_destinations, self
        )
        self._create_edge_destinations = types.MethodType(
            _create_edge_destinations, self
        )
        self.create_tract_appropriate_destinations = types.MethodType(
            create_tract_appropriate_destinations, self
        )

    def _remove_spatial_duplicates(self, destinations_gdf, min_distance_m=100):
        """Remove spatially duplicate destinations"""
        if len(destinations_gdf) <= 1:
            return destinations_gdf
        
        dest_projected = destinations_gdf.to_crs('EPSG:3857')
        
        keep_indices = []
        for i, dest1 in dest_projected.iterrows():
            is_duplicate = False
            for j in keep_indices:
                dest2 = dest_projected.iloc[j]
                distance = dest1.geometry.distance(dest2.geometry)
                if distance < min_distance_m:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                keep_indices.append(i)
        
        return destinations_gdf.iloc[keep_indices].reset_index(drop=True)