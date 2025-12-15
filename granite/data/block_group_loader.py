"""
Block Group Data Loader for GRANITE Validation

Loads Census block group geometries and ACS demographic indicators
for validating address-level predictions against known sub-tract aggregates.
"""
import os
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, Optional, Tuple
from pathlib import Path


class BlockGroupLoader:
    """
    Loads and processes Census block group data for validation.
    
    Block groups are the smallest geography with reliable ACS estimates,
    sitting between tracts (~4,000 people) and blocks (~100 people).
    Hamilton County has approximately 150-200 block groups.
    """
    
    # ACS variables for validation (2020 5-year estimates)
    ACS_VARIABLES = {
        # Poverty (B17001)
        'B17001_001E': 'total_poverty_universe',
        'B17001_002E': 'below_poverty',
        
        # Vehicles (B25044)
        'B25044_001E': 'total_occupied_units',
        'B25044_003E': 'no_vehicle_renter',
        'B25044_010E': 'no_vehicle_owner',
        
        # Education (B15003) - population 25+
        'B15003_001E': 'total_25plus',
        'B15003_002E': 'no_schooling',
        'B15003_003E': 'nursery_school',
        'B15003_004E': 'kindergarten',
        'B15003_005E': 'grade_1',
        'B15003_006E': 'grade_2',
        'B15003_007E': 'grade_3',
        'B15003_008E': 'grade_4',
        'B15003_009E': 'grade_5',
        'B15003_010E': 'grade_6',
        'B15003_011E': 'grade_7',
        'B15003_012E': 'grade_8',
        'B15003_013E': 'grade_9',
        'B15003_014E': 'grade_10',
        'B15003_015E': 'grade_11',
        'B15003_016E': 'grade_12_no_diploma',
        'B15003_017E': 'hs_diploma',
        
        # Total population
        'B01001_001E': 'total_population',
    }
    
    def __init__(self, data_dir: str = './data', api_key: Optional[str] = None,
                 verbose: bool = False):
        self.data_dir = Path(data_dir)
        self.api_key = api_key or os.environ.get('CENSUS_API_KEY')
        self.verbose = verbose
        
        # Cache
        self._bg_geometries = None
        self._bg_demographics = None
    
    def _log(self, message: str):
        if self.verbose:
            print(f"[BlockGroupLoader] {message}")
    
    def load_block_group_geometries(self, state_fips: str = '47', 
                                     county_fips: str = '065') -> gpd.GeoDataFrame:
        """
        Load block group geometries from TIGER/Line shapefile.
        
        Returns:
            GeoDataFrame with columns: GEOID, TRACTCE, geometry
        """
        if self._bg_geometries is not None:
            return self._bg_geometries
        
        bg_file = self.data_dir / 'raw' / f'tl_2020_{state_fips}_bg.shp'
        
        if not bg_file.exists():
            raise FileNotFoundError(
                f"Block group shapefile not found: {bg_file}\n"
                f"Download from Census TIGER/Line: "
                f"https://www.census.gov/cgi-bin/geo/shapefiles/index.php"
            )
        
        self._log(f"Loading block groups from {bg_file}")
        bg_gdf = gpd.read_file(bg_file)
        
        # Filter to county
        county_bg = bg_gdf[bg_gdf['COUNTYFP'] == county_fips].copy()
        
        # Standardize columns
        county_bg['GEOID'] = county_bg['GEOID'].astype(str)
        county_bg['TRACTCE'] = county_bg['TRACTCE'].astype(str)
        county_bg['tract_fips'] = state_fips + county_fips + county_bg['TRACTCE']
        
        # Compute centroids for spatial joins
        county_bg['centroid'] = county_bg.geometry.centroid
        
        if county_bg.crs is None:
            county_bg.set_crs(epsg=4326, inplace=True)
        elif county_bg.crs != 'EPSG:4326':
            county_bg = county_bg.to_crs(epsg=4326)
        
        self._log(f"Loaded {len(county_bg)} block groups")
        self._bg_geometries = county_bg
        
        return county_bg
    
    def fetch_acs_demographics(self, state_fips: str = '47',
                                county_fips: str = '065',
                                year: int = 2020) -> pd.DataFrame:
        """
        Fetch ACS 5-year estimates for block groups via Census API.
        
        Returns:
            DataFrame with computed demographic rates per block group
        """
        if self._bg_demographics is not None:
            return self._bg_demographics
        
        if not self.api_key:
            # Try loading from cached file
            cache_file = self.data_dir / 'processed' / 'acs_block_groups.csv'
            if cache_file.exists():
                self._log(f"Loading cached ACS data from {cache_file}")
                self._bg_demographics = pd.read_csv(cache_file, dtype={'GEOID': str})
                return self._bg_demographics
            
            raise ValueError(
                "Census API key required. Set CENSUS_API_KEY environment variable "
                "or pass api_key to constructor.\n"
                "Get a key at: https://api.census.gov/data/key_signup.html"
            )
        
        self._log(f"Fetching ACS {year} 5-year estimates from Census API...")
        
        # Build API request
        variables = ','.join(self.ACS_VARIABLES.keys())
        url = (
            f"https://api.census.gov/data/{year}/acs/acs5"
            f"?get=NAME,{variables}"
            f"&for=block%20group:*"
            f"&in=state:{state_fips}%20county:{county_fips}"
            f"&key={self.api_key}"
        )
        
        response = requests.get(url, timeout=60)
        
        if response.status_code != 200:
            raise RuntimeError(
                f"Census API error {response.status_code}: {response.text}"
            )
        
        data = response.json()
        
        # Parse response: first row is headers, rest is data
        headers = data[0]
        rows = data[1:]
        
        df = pd.DataFrame(rows, columns=headers)
        
        # Create GEOID from components
        df['GEOID'] = df['state'] + df['county'] + df['tract'] + df['block group']
        
        # Convert numeric columns
        for var in self.ACS_VARIABLES.keys():
            df[var] = pd.to_numeric(df[var], errors='coerce')
        
        # Compute derived rates
        demographics = self._compute_demographic_rates(df)
        
        # Cache results
        cache_dir = self.data_dir / 'processed'
        cache_dir.mkdir(exist_ok=True)
        demographics.to_csv(cache_dir / 'acs_block_groups.csv', index=False)
        
        self._log(f"Fetched demographics for {len(demographics)} block groups")
        self._bg_demographics = demographics
        
        return demographics
    
    def _compute_demographic_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute vulnerability-related rates from raw ACS counts."""
        
        result = pd.DataFrame()
        result['GEOID'] = df['GEOID']
        
        # Poverty rate
        result['poverty_rate'] = np.where(
            df['B17001_001E'] > 0,
            df['B17001_002E'] / df['B17001_001E'] * 100,
            np.nan
        )
        
        # No vehicle rate (combine renter + owner)
        total_units = df['B25044_001E']
        no_vehicle = df['B25044_003E'] + df['B25044_010E']
        result['no_vehicle_rate'] = np.where(
            total_units > 0,
            no_vehicle / total_units * 100,
            np.nan
        )
        
        # No high school diploma rate (sum grades 0-12 without diploma)
        no_hs_cols = [f'B15003_{str(i).zfill(3)}E' for i in range(2, 17)]
        no_hs = df[no_hs_cols].sum(axis=1)
        total_25plus = df['B15003_001E']
        result['no_hs_rate'] = np.where(
            total_25plus > 0,
            no_hs / total_25plus * 100,
            np.nan
        )
        
        # Total population (for weighting)
        result['population'] = df['B01001_001E']
        
        return result
    
    def get_block_groups_with_demographics(self, state_fips: str = '47',
                                            county_fips: str = '065') -> gpd.GeoDataFrame:
        """
        Load block groups with both geometries and demographic indicators.
        
        Returns:
            GeoDataFrame with geometry and demographic rates
        """
        geometries = self.load_block_group_geometries(state_fips, county_fips)
        demographics = self.fetch_acs_demographics(state_fips, county_fips)
        
        # Merge
        merged = geometries.merge(demographics, on='GEOID', how='left')
        
        self._log(f"Merged {len(merged)} block groups with demographics")
        
        return merged
    
    def assign_addresses_to_block_groups(self, 
                                          addresses: gpd.GeoDataFrame,
                                          block_groups: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Spatially join addresses to their containing block groups.
        
        Args:
            addresses: GeoDataFrame with address points
            block_groups: GeoDataFrame with block group polygons
            
        Returns:
            addresses with added 'block_group_id' column
        """
        self._log(f"Assigning {len(addresses)} addresses to block groups...")
        
        # Ensure consistent CRS
        if addresses.crs != block_groups.crs:
            addresses = addresses.to_crs(block_groups.crs)
        
        # Spatial join
        joined = gpd.sjoin(
            addresses,
            block_groups[['GEOID', 'geometry']],
            how='left',
            predicate='within'
        )
        
        joined = joined.rename(columns={'GEOID': 'block_group_id'})
        
        # Handle addresses outside block groups (rare edge cases)
        n_unmatched = joined['block_group_id'].isna().sum()
        if n_unmatched > 0:
            self._log(f"Warning: {n_unmatched} addresses not in any block group")
        
        # Drop duplicate index from sjoin
        if 'index_right' in joined.columns:
            joined = joined.drop(columns=['index_right'])
        
        return joined
    
    def aggregate_predictions_to_block_groups(self,
                                               addresses: gpd.GeoDataFrame,
                                               predictions: np.ndarray,
                                               block_groups: gpd.GeoDataFrame,
                                               method: str = 'mean') -> pd.DataFrame:
        """
        Aggregate address-level predictions to block group level.
        
        Args:
            addresses: GeoDataFrame with address points (must have block_group_id)
            predictions: Array of address-level predictions
            block_groups: GeoDataFrame with block group polygons
            method: Aggregation method ('mean', 'median', 'weighted_mean')
            
        Returns:
            DataFrame with block group GEOID and aggregated predictions
        """
        if 'block_group_id' not in addresses.columns:
            addresses = self.assign_addresses_to_block_groups(addresses, block_groups)
        
        # Create working dataframe
        df = pd.DataFrame({
            'block_group_id': addresses['block_group_id'],
            'prediction': predictions
        })
        
        # Remove addresses without block group assignment
        df = df.dropna(subset=['block_group_id'])
        
        # Aggregate
        if method == 'mean':
            agg = df.groupby('block_group_id')['prediction'].agg(['mean', 'std', 'count'])
        elif method == 'median':
            agg = df.groupby('block_group_id')['prediction'].agg(
                mean='median', std='std', count='count'
            )
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        agg = agg.reset_index()
        agg.columns = ['GEOID', 'predicted_vulnerability', 'prediction_std', 'n_addresses']
        
        self._log(f"Aggregated to {len(agg)} block groups")
        
        return agg


def load_block_group_validation_data(data_dir: str = './data',
                                      api_key: Optional[str] = None,
                                      verbose: bool = False) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Convenience function to load all block group validation data.
    
    Returns:
        Tuple of (block_groups_gdf, demographics_df)
    """
    loader = BlockGroupLoader(data_dir, api_key, verbose)
    
    bg_gdf = loader.get_block_groups_with_demographics()
    demographics = loader.fetch_acs_demographics()
    
    return bg_gdf, demographics