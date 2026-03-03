"""
Block Group Data Loader for GRANITE Validation

Loads Census block group geometries and ACS demographic indicators
for validating address-level predictions against known sub-tract aggregates.

Extended to compute block-group-level SVI using CDC methodology.
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
    
    # ACS variables for block-group SVI computation (2020 5-year estimates)
    # Note: Poverty (B17001) is suppressed at BG level. Using income as proxy.
    ACS_VARIABLES = {
        # === THEME 1: Socioeconomic Status ===
        # Income measures (poverty is suppressed at BG level)
        'B19013_001E': 'median_household_income',
        'B19301_001E': 'per_capita_income',
        
        # Unemployment - B23025
        'B23025_003E': 'civilian_labor_force',
        'B23025_005E': 'unemployed',
        
        # Education (no HS diploma) - B15003
        'B15003_001E': 'edu_universe_25plus',
        'B15003_002E': 'edu_no_school',
        'B15003_003E': 'edu_nursery',
        'B15003_004E': 'edu_kindergarten',
        'B15003_005E': 'edu_grade_1',
        'B15003_006E': 'edu_grade_2',
        'B15003_007E': 'edu_grade_3',
        'B15003_008E': 'edu_grade_4',
        'B15003_009E': 'edu_grade_5',
        'B15003_010E': 'edu_grade_6',
        'B15003_011E': 'edu_grade_7',
        'B15003_012E': 'edu_grade_8',
        'B15003_013E': 'edu_grade_9',
        'B15003_014E': 'edu_grade_10',
        'B15003_015E': 'edu_grade_11',
        'B15003_016E': 'edu_grade_12_no_diploma',
        
        # === THEME 2: Household Characteristics ===
        # Age from B01001 (by sex)
        'B01001_001E': 'total_population',
        # Male 65+
        'B01001_020E': 'male_65_66',
        'B01001_021E': 'male_67_69',
        'B01001_022E': 'male_70_74',
        'B01001_023E': 'male_75_79',
        'B01001_024E': 'male_80_84',
        'B01001_025E': 'male_85_plus',
        # Female 65+
        'B01001_044E': 'female_65_66',
        'B01001_045E': 'female_67_69',
        'B01001_046E': 'female_70_74',
        'B01001_047E': 'female_75_79',
        'B01001_048E': 'female_80_84',
        'B01001_049E': 'female_85_plus',
        # Male under 18
        'B01001_003E': 'male_under_5',
        'B01001_004E': 'male_5_9',
        'B01001_005E': 'male_10_14',
        'B01001_006E': 'male_15_17',
        # Female under 18
        'B01001_027E': 'female_under_5',
        'B01001_028E': 'female_5_9',
        'B01001_029E': 'female_10_14',
        'B01001_030E': 'female_15_17',
        
        # Single parent households - B09002
        'B09002_001E': 'children_universe',
        'B09002_015E': 'children_single_parent',
        
        # === THEME 3: Minority Status / Language ===
        # Race - B02001
        'B02001_001E': 'race_total',
        'B02001_002E': 'race_white_alone',
        
        # Hispanic origin - B03003
        'B03003_001E': 'hispanic_universe',
        'B03003_003E': 'hispanic_or_latino',
        
        # === THEME 4: Housing Type / Transportation ===
        # Housing units in structure - B25024
        'B25024_001E': 'housing_units_total',
        'B25024_007E': 'units_10_to_19',
        'B25024_008E': 'units_20_to_49',
        'B25024_009E': 'units_50_plus',
        'B25024_010E': 'mobile_homes',
        
        # Crowding - B25014
        'B25014_001E': 'occupancy_universe',
        'B25014_005E': 'owner_1_to_1.5_per_room',
        'B25014_006E': 'owner_1.5_to_2_per_room',
        'B25014_007E': 'owner_2_plus_per_room',
        'B25014_011E': 'renter_1_to_1.5_per_room',
        'B25014_012E': 'renter_1.5_to_2_per_room',
        'B25014_013E': 'renter_2_plus_per_room',
        
        # No vehicle - B25044
        'B25044_001E': 'vehicle_universe',
        'B25044_003E': 'no_vehicle_owner',
        'B25044_010E': 'no_vehicle_renter',
    }
    
    def __init__(self, data_dir: str = './data', api_key: Optional[str] = None,
                 verbose: bool = False):
        self.data_dir = Path(data_dir)
        self.api_key = api_key or os.environ.get('CENSUS_API_KEY')
        self.verbose = verbose
        
        # Cache
        self._bg_geometries = None
        self._bg_demographics = None
        self._raw_acs_data = None
    
    def _log(self, message: str):
        if self.verbose:
            print(f"[BlockGroupLoader] {message}")
    
    def load_block_group_geometries(self, state_fips: str = '47', 
                                     county_fips: str = '065') -> gpd.GeoDataFrame:
        """Load block group geometries from TIGER/Line shapefile."""
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
            DataFrame with computed demographic rates and SVI per block group
        """
        if self._bg_demographics is not None:
            return self._bg_demographics
        
        # Check for cached processed file
        cache_file = self.data_dir / 'processed' / 'acs_block_groups_svi.csv'
        if cache_file.exists():
            self._log(f"Loading cached ACS+SVI data from {cache_file}")
            self._bg_demographics = pd.read_csv(cache_file, dtype={'GEOID': str})
            return self._bg_demographics
        
        if not self.api_key:
            raise ValueError(
                "Census API key required. Set CENSUS_API_KEY environment variable "
                "or pass api_key to constructor.\n"
                "Get a key at: https://api.census.gov/data/key_signup.html"
            )
        
        # Fetch raw data
        raw_df = self._fetch_raw_acs_data(state_fips, county_fips, year)
        
        # Compute rates
        rates_df = self._compute_demographic_rates(raw_df)
        
        # Compute SVI using CDC methodology
        svi_df = self._compute_block_group_svi(rates_df)
        
        # Cache results
        cache_dir = self.data_dir / 'processed'
        cache_dir.mkdir(exist_ok=True)
        svi_df.to_csv(cache_file, index=False)
        
        self._log(f"Computed SVI for {len(svi_df)} block groups")
        self._bg_demographics = svi_df
        
        return svi_df
    
    def _fetch_raw_acs_data(self, state_fips: str, county_fips: str, 
                            year: int) -> pd.DataFrame:
        """Fetch raw ACS counts from Census API."""
        
        self._log(f"Fetching ACS {year} 5-year estimates from Census API...")
        
        # API has variable limits per request, so we chunk
        var_list = list(self.ACS_VARIABLES.keys())
        chunk_size = 48  # Safe limit under 50-variable cap
        
        all_data = None
        
        for i in range(0, len(var_list), chunk_size):
            chunk_vars = var_list[i:i + chunk_size]
            variables = ','.join(chunk_vars)
            
            url = (
                f"https://api.census.gov/data/{year}/acs/acs5"
                f"?get=NAME,{variables}"
                f"&for=block%20group:*"
                f"&in=state:{state_fips}%20county:{county_fips}"
                f"&key={self.api_key}"
            )
            
            self._log(f"  Fetching variables {i+1}-{min(i+chunk_size, len(var_list))}...")
            response = requests.get(url, timeout=60)
            
            if response.status_code != 200:
                raise RuntimeError(
                    f"Census API error {response.status_code}: {response.text}"
                )
            
            data = response.json()
            headers = data[0]
            rows = data[1:]
            
            chunk_df = pd.DataFrame(rows, columns=headers)
            
            if all_data is None:
                all_data = chunk_df
            else:
                # Merge on geography columns
                geo_cols = ['NAME', 'state', 'county', 'tract', 'block group']
                new_cols = [c for c in chunk_df.columns if c not in geo_cols]
                all_data = all_data.merge(
                    chunk_df[geo_cols + new_cols],
                    on=geo_cols,
                    how='outer'
                )
        
        # Create GEOID
        all_data['GEOID'] = (all_data['state'] + all_data['county'] + 
                            all_data['tract'] + all_data['block group'])
        
        # Convert numeric columns
        for var in self.ACS_VARIABLES.keys():
            if var in all_data.columns:
                all_data[var] = pd.to_numeric(all_data[var], errors='coerce')
        
        self._log(f"Fetched {len(all_data)} block groups")
        self._raw_acs_data = all_data
        
        return all_data
    
    def _compute_demographic_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute SVI component rates from raw ACS counts.
        
        Note: This is a simplified SVI using variables available at block group level.
        Poverty rate (B17001) is suppressed at BG level, so we use income as proxy.
        """
        
        result = pd.DataFrame()
        result['GEOID'] = df['GEOID']
        result['population'] = df['B01001_001E']
        
        # === THEME 1: Socioeconomic ===
        
        # EP_MHI: Median household income (lower = more vulnerable)
        # This replaces poverty rate which is suppressed at BG level
        result['EP_MHI'] = df['B19013_001E']
        
        # EP_PCI: Per capita income (lower = more vulnerable)
        result['EP_PCI'] = df['B19301_001E']
        
        # EP_UNEMP: Unemployed
        result['EP_UNEMP'] = np.where(
            df['B23025_003E'] > 0,
            df['B23025_005E'] / df['B23025_003E'] * 100,
            np.nan
        )
        
        # EP_NOHSDP: No high school diploma
        no_hs = (df['B15003_002E'] + df['B15003_003E'] + df['B15003_004E'] +
                 df['B15003_005E'] + df['B15003_006E'] + df['B15003_007E'] +
                 df['B15003_008E'] + df['B15003_009E'] + df['B15003_010E'] +
                 df['B15003_011E'] + df['B15003_012E'] + df['B15003_013E'] +
                 df['B15003_014E'] + df['B15003_015E'] + df['B15003_016E'])
        result['EP_NOHSDP'] = np.where(
            df['B15003_001E'] > 0,
            no_hs / df['B15003_001E'] * 100,
            np.nan
        )
        
        # === THEME 2: Household Characteristics ===
        
        # EP_AGE65: Aged 65 and older
        age_65_plus = (df['B01001_020E'] + df['B01001_021E'] + df['B01001_022E'] +
                       df['B01001_023E'] + df['B01001_024E'] + df['B01001_025E'] +
                       df['B01001_044E'] + df['B01001_045E'] + df['B01001_046E'] +
                       df['B01001_047E'] + df['B01001_048E'] + df['B01001_049E'])
        result['EP_AGE65'] = np.where(
            df['B01001_001E'] > 0,
            age_65_plus / df['B01001_001E'] * 100,
            np.nan
        )
        
        # EP_AGE17: Aged 17 and younger
        age_under_18 = (df['B01001_003E'] + df['B01001_004E'] + df['B01001_005E'] +
                        df['B01001_006E'] + df['B01001_027E'] + df['B01001_028E'] +
                        df['B01001_029E'] + df['B01001_030E'])
        result['EP_AGE17'] = np.where(
            df['B01001_001E'] > 0,
            age_under_18 / df['B01001_001E'] * 100,
            np.nan
        )
        
        # EP_SNGPNT: Children in single parent households (using B09002)
        result['EP_SNGPNT'] = np.where(
            df['B09002_001E'] > 0,
            df['B09002_015E'] / df['B09002_001E'] * 100,
            np.nan
        )
        
        # === THEME 3: Minority Status / Language ===
        
        # EP_MINRTY: Minority (all except white non-Hispanic)
        result['EP_MINRTY'] = np.where(
            df['B02001_001E'] > 0,
            (1 - df['B02001_002E'] / df['B02001_001E']) * 100 + 
            (df['B03003_003E'] / df['B03003_001E'].replace(0, np.nan) * 100).fillna(0),
            np.nan
        )
        result['EP_MINRTY'] = result['EP_MINRTY'].clip(upper=100)
        
        # === THEME 4: Housing Type / Transportation ===
        
        # EP_MUNIT: Multi-unit structures (10+ units)
        multi_unit = df['B25024_007E'] + df['B25024_008E'] + df['B25024_009E']
        result['EP_MUNIT'] = np.where(
            df['B25024_001E'] > 0,
            multi_unit / df['B25024_001E'] * 100,
            np.nan
        )
        
        # EP_MOBILE: Mobile homes
        result['EP_MOBILE'] = np.where(
            df['B25024_001E'] > 0,
            df['B25024_010E'] / df['B25024_001E'] * 100,
            np.nan
        )
        
        # EP_CROWD: More than 1 person per room
        crowded = (df['B25014_005E'] + df['B25014_006E'] + df['B25014_007E'] +
                   df['B25014_011E'] + df['B25014_012E'] + df['B25014_013E'])
        result['EP_CROWD'] = np.where(
            df['B25014_001E'] > 0,
            crowded / df['B25014_001E'] * 100,
            np.nan
        )
        
        # EP_NOVEH: No vehicle
        no_vehicle = df['B25044_003E'] + df['B25044_010E']
        result['EP_NOVEH'] = np.where(
            df['B25044_001E'] > 0,
            no_vehicle / df['B25044_001E'] * 100,
            np.nan
        )
        
        return result
    
    def _compute_block_group_svi(self, rates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute SVI using CDC percentile ranking methodology.
        
        Simplified version using 11 variables available at block group level:
        - Theme 1: MHI, PCI, Unemployment, Education (4 vars) [income replaces poverty]
        - Theme 2: Age 65+, Age 17-, Single Parent (3 vars)
        - Theme 3: Minority (1 var)
        - Theme 4: Multi-unit, Mobile, Crowding, No Vehicle (4 vars)
        
        CDC Method:
        1. Rank each variable across all block groups (percentile 0-1)
        2. For income vars, invert the percentile (lower income = higher vulnerability)
        3. Sum percentiles within each theme
        4. Rank theme sums to get theme percentiles
        5. Sum theme percentiles and rank for overall SVI
        """
        
        result = rates_df.copy()
        n = len(result)
        
        # Define SVI variables by theme
        # Income variables (EP_MHI, EP_PCI) are inverted: lower = more vulnerable
        theme_vars = {
            'theme1_socioeconomic': ['EP_MHI', 'EP_PCI', 'EP_UNEMP', 'EP_NOHSDP'],
            'theme2_household': ['EP_AGE65', 'EP_AGE17', 'EP_SNGPNT'],
            'theme3_minority': ['EP_MINRTY'],
            'theme4_housing': ['EP_MUNIT', 'EP_MOBILE', 'EP_CROWD', 'EP_NOVEH'],
        }
        
        # Variables where lower value = higher vulnerability (need inversion)
        invert_vars = ['EP_MHI', 'EP_PCI']
        
        # Step 1: Compute percentile rank for each variable
        for theme, variables in theme_vars.items():
            for var in variables:
                if var not in result.columns:
                    self._log(f"Warning: {var} not found, skipping")
                    continue
                
                # For income vars, invert: lower income = higher percentile = more vulnerable
                if var in invert_vars:
                    result[f'{var}_pctl'] = result[var].rank(ascending=False, pct=True, na_option='keep')
                else:
                    result[f'{var}_pctl'] = result[var].rank(ascending=True, pct=True, na_option='keep')
        
        # Step 2: Sum percentiles within each theme
        for theme, variables in theme_vars.items():
            pctl_cols = [f'{var}_pctl' for var in variables if f'{var}_pctl' in result.columns]
            if pctl_cols:
                result[f'{theme}_sum'] = result[pctl_cols].sum(axis=1, skipna=False)
        
        # Step 3: Rank theme sums to get theme percentiles
        for theme in theme_vars.keys():
            sum_col = f'{theme}_sum'
            if sum_col in result.columns:
                result[f'{theme}_pctl'] = result[sum_col].rank(ascending=True, pct=True, na_option='keep')
        
        # Step 4: Sum theme percentiles
        theme_pctl_cols = [f'{theme}_pctl' for theme in theme_vars.keys() 
                          if f'{theme}_pctl' in result.columns]
        result['themes_sum'] = result[theme_pctl_cols].sum(axis=1, skipna=False)
        
        # Step 5: Final SVI percentile
        result['SVI'] = result['themes_sum'].rank(ascending=True, pct=True, na_option='keep')
        
        # Also compute individual theme SVIs for analysis
        result['SVI_theme1'] = result.get('theme1_socioeconomic_pctl', np.nan)
        result['SVI_theme2'] = result.get('theme2_household_pctl', np.nan)
        result['SVI_theme3'] = result.get('theme3_minority_pctl', np.nan)
        result['SVI_theme4'] = result.get('theme4_housing_pctl', np.nan)
        
        # Flag block groups with missing data (using available vars)
        core_vars = ['EP_MHI', 'EP_UNEMP', 'EP_NOHSDP', 'EP_NOVEH']
        result['svi_complete'] = ~result[core_vars].isna().any(axis=1)
        
        self._log(f"Computed SVI: {result['svi_complete'].sum()}/{n} block groups complete")
        
        return result
    
    def get_block_groups_with_demographics(self, state_fips: str = '47',
                                            county_fips: str = '065') -> gpd.GeoDataFrame:
        """Load block groups with both geometries and demographic indicators."""
        geometries = self.load_block_group_geometries(state_fips, county_fips)
        demographics = self.fetch_acs_demographics(state_fips, county_fips)
        
        # Merge
        merged = geometries.merge(demographics, on='GEOID', how='left')
        
        self._log(f"Merged {len(merged)} block groups with demographics")
        
        return merged
    
    def assign_addresses_to_block_groups(self, 
                                          addresses: gpd.GeoDataFrame,
                                          block_groups: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Spatially join addresses to their containing block groups."""
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
        
        # Handle addresses outside block groups
        n_unmatched = joined['block_group_id'].isna().sum()
        if n_unmatched > 0:
            self._log(f"Warning: {n_unmatched} addresses not in any block group")
        
        if 'index_right' in joined.columns:
            joined = joined.drop(columns=['index_right'])
        
        return joined
    
    def aggregate_predictions_to_block_groups(self,
                                               addresses: gpd.GeoDataFrame,
                                               predictions: np.ndarray,
                                               block_groups: gpd.GeoDataFrame,
                                               method: str = 'mean') -> pd.DataFrame:
        """Aggregate address-level predictions to block group level."""
        if 'block_group_id' not in addresses.columns:
            addresses = self.assign_addresses_to_block_groups(addresses, block_groups)
        
        df = pd.DataFrame({
            'block_group_id': addresses['block_group_id'],
            'prediction': predictions
        })
        
        df = df.dropna(subset=['block_group_id'])
        
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


# Quick test function
def test_svi_computation(api_key: str, data_dir: str = './data'):
    """Test the SVI computation with real data."""
    
    print("Testing block group SVI computation...")
    print("="*60)
    
    loader = BlockGroupLoader(data_dir=data_dir, api_key=api_key, verbose=True)
    
    # Clear any cached file to force fresh computation
    cache_file = Path(data_dir) / 'processed' / 'acs_block_groups_svi.csv'
    if cache_file.exists():
        print(f"Removing cached file: {cache_file}")
        cache_file.unlink()
    
    # Fetch raw data first to debug
    raw_df = loader._fetch_raw_acs_data('47', '065', 2020)
    
    print(f"\nRaw data columns ({len(raw_df.columns)}):")
    acs_cols = [c for c in raw_df.columns if c.startswith('B')]
    print(f"  ACS columns: {len(acs_cols)}")
    
    # Check income columns
    print(f"\nIncome columns check:")
    for col in ['B19013_001E', 'B19301_001E']:
        if col in raw_df.columns:
            non_null = raw_df[col].notna().sum()
            print(f"  {col}: {non_null} non-null, sample: {raw_df[col].head(3).tolist()}")
        else:
            print(f"  {col}: NOT FOUND")
    
    # Check vehicle columns
    print(f"\nVehicle columns check:")
    for col in ['B25044_001E', 'B25044_003E', 'B25044_010E']:
        if col in raw_df.columns:
            non_null = raw_df[col].notna().sum()
            print(f"  {col}: {non_null} non-null, sample: {raw_df[col].head(3).tolist()}")
        else:
            print(f"  {col}: NOT FOUND")
    
    # Reset cache to force recomputation
    loader._bg_demographics = None
    loader._raw_acs_data = None
    
    # Now compute demographics
    demographics = loader.fetch_acs_demographics()
    
    print(f"\nResults:")
    print(f"  Block groups: {len(demographics)}")
    print(f"  Complete SVI: {demographics['svi_complete'].sum()}")
    
    # Check which core vars are null
    print(f"\nCore variable coverage:")
    for var in ['EP_MHI', 'EP_PCI', 'EP_UNEMP', 'EP_NOHSDP', 'EP_NOVEH']:
        non_null = demographics[var].notna().sum()
        print(f"  {var}: {non_null}/{len(demographics)} non-null")
    
    if demographics['SVI'].notna().any():
        print(f"\n  SVI range: {demographics['SVI'].min():.3f} - {demographics['SVI'].max():.3f}")
        print(f"  SVI mean: {demographics['SVI'].mean():.3f}")
        
        # Show distribution
        print(f"\nSVI Distribution:")
        for threshold in [0.25, 0.50, 0.75]:
            count = (demographics['SVI'] <= threshold).sum()
            print(f"  <= {threshold}: {count} ({count/len(demographics)*100:.1f}%)")
    else:
        print("\n  SVI computation failed - all values NaN")
    
    return demographics


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
        test_svi_computation(api_key)
    else:
        print("Usage: python block_group_loader.py <CENSUS_API_KEY>")