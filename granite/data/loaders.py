"""
Updated data loading functions for GRANITE framework with robust URL handling
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import fiona
from shapely.geometry import Point, LineString
import networkx as nx
from datetime import datetime
import urllib.request
import urllib.error


class DataLoader:
    """Main data loader class for GRANITE framework with robust error handling"""
    
    def __init__(self, data_dir='./data', verbose=True):
        self.data_dir = data_dir
        self.verbose = verbose
        
        # Updated SVI URLs to try (in order of preference)
        self.svi_urls = [
            "https://www.atsdr.cdc.gov/placeandhealth/svi/data/SviDataCsv/SVI_2020_US.csv",
            "https://svi.cdc.gov/Documents/Data/2020/csv/SVI_2020_US.csv",
            "https://www.atsdr.cdc.gov/placeandhealth/svi/data/SviDataCsv/SVI2020_US.csv"
        ]
        
    def _log(self, message):
        """Logging with timestamp"""
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def _try_download_svi(self):
        """Try multiple SVI URLs until one works"""
        for i, url in enumerate(self.svi_urls):
            try:
                self._log(f"  Trying SVI URL {i+1}/{len(self.svi_urls)}: {url[:50]}...")
                
                # Test if URL is accessible
                with urllib.request.urlopen(url, timeout=10) as response:
                    if response.status == 200:
                        self._log(f"  ✓ URL {i+1} is accessible")
                        return pd.read_csv(url)
                        
            except (urllib.error.HTTPError, urllib.error.URLError, Exception) as e:
                self._log(f"  ✗ URL {i+1} failed: {str(e)}")
                continue
        
        return None
    
    def load_svi_data(self, state='Tennessee', county='Hamilton'):
        """
        Load Social Vulnerability Index data from CDC with robust URL handling
        
        Parameters:
        -----------
        state : str
            State name for SVI data
        county : str
            County name to filter
            
        Returns:
        --------
        pd.DataFrame
            SVI data for specified county
        """
        self._log(f"Loading SVI data for {county} County, {state}...")
        
        try:
            # Try to download SVI data
            svi_df = self._try_download_svi()
            
            if svi_df is None:
                # If all URLs fail, create mock data for testing
                self._log("  ⚠️  All SVI URLs failed. Creating mock data for testing...")
                return self._create_mock_svi_data(state, county)
            
            self._log(f"  ✓ Downloaded SVI data ({len(svi_df)} total records)")
            
            # Filter for specific state and county
            # Try different column name variations
            state_cols = ['STATE', 'ST', 'ST_ABBR', 'STUSPS']
            county_cols = ['COUNTY', 'CNTY', 'COUNTY_NAME']
            
            state_col = None
            county_col = None
            
            for col in state_cols:
                if col in svi_df.columns:
                    state_col = col
                    break
            
            for col in county_cols:
                if col in svi_df.columns:
                    county_col = col
                    break
            
            if state_col and county_col:
                # Filter data
                county_svi = svi_df[
                    (svi_df[state_col].str.contains(state, case=False, na=False)) &
                    (svi_df[county_col].str.contains(county, case=False, na=False))
                ].copy()
            else:
                # Fallback: use FIPS code filtering
                # Hamilton County, TN FIPS: 47065
                if 'FIPS' in svi_df.columns:
                    county_svi = svi_df[svi_df['FIPS'].astype(str).str.startswith('47065')].copy()
                else:
                    self._log("  ⚠️  Cannot filter by state/county. Using mock data...")
                    return self._create_mock_svi_data(state, county)
            
            if len(county_svi) == 0:
                self._log("  ⚠️  No data found for specified county. Using mock data...")
                return self._create_mock_svi_data(state, county)
            
            # Select relevant columns (flexible column selection)
            desired_columns = ['FIPS', 'LOCATION', 'RPL_THEMES', 'RPL_THEME1', 
                              'RPL_THEME2', 'RPL_THEME3', 'RPL_THEME4', 
                              'E_TOTPOP', 'E_HU', 'E_POV', 'E_UNEMP', 'E_NOHSDP',
                              'E_AGE65', 'E_AGE17', 'E_DISABL', 'E_SNGPNT', 
                              'E_MINRTY', 'E_LIMENG', 'E_MUNIT', 'E_MOBILE',
                              'E_CROWD', 'E_NOVEH', 'E_GROUPQ']
            
            available_cols = [col for col in desired_columns if col in county_svi.columns]
            county_svi = county_svi[available_cols]
            
            # Handle missing values
            if 'RPL_THEMES' in county_svi.columns:
                county_svi['RPL_THEMES'] = county_svi['RPL_THEMES'].replace(-999, np.nan)
                
                # Statistics
                valid_svi = county_svi['RPL_THEMES'].dropna()
                self._log(f"  ✓ Loaded SVI data for {len(county_svi)} census tracts")
                self._log(f"    - Valid SVI scores: {len(valid_svi)}")
                if len(valid_svi) > 0:
                    self._log(f"    - Mean SVI: {valid_svi.mean():.3f}")
                    self._log(f"    - SVI Range: [{valid_svi.min():.3f}, {valid_svi.max():.3f}]")
            else:
                self._log(f"  ✓ Loaded {len(county_svi)} records (RPL_THEMES column not found)")
            
            return county_svi
            
        except Exception as e:
            self._log(f"  ✗ Error loading SVI data: {str(e)}")
            self._log("  ⚠️  Falling back to mock data...")
            return self._create_mock_svi_data(state, county)
    
    def _create_mock_svi_data(self, state='Tennessee', county='Hamilton'):
        """Create mock SVI data for testing when real data is unavailable"""
        self._log("  Creating mock SVI data for testing...")
        
        # Hamilton County, TN tract FIPS codes (simplified)
        fips_codes = [
            '47065000100', '47065000200', '47065000300', '47065000400', '47065000500',
            '47065000600', '47065000700', '47065000800', '47065000900', '47065001000',
            '47065001100', '47065001200', '47065001300', '47065001400', '47065001500'
        ]
        
        np.random.seed(42)  # Reproducible mock data
        
        mock_data = pd.DataFrame({
            'FIPS': fips_codes,
            'LOCATION': [f"{county} County, {state}, Census Tract {i+1}" for i in range(len(fips_codes))],
            'RPL_THEMES': np.random.uniform(0.1, 0.9, len(fips_codes)),
            'RPL_THEME1': np.random.uniform(0.0, 1.0, len(fips_codes)),
            'RPL_THEME2': np.random.uniform(0.0, 1.0, len(fips_codes)),
            'RPL_THEME3': np.random.uniform(0.0, 1.0, len(fips_codes)),
            'RPL_THEME4': np.random.uniform(0.0, 1.0, len(fips_codes)),
            'E_TOTPOP': np.random.randint(1000, 8000, len(fips_codes)),
            'E_POV': np.random.randint(50, 800, len(fips_codes))
        })
        
        self._log(f"  ✓ Created mock SVI data for {len(mock_data)} census tracts")
        
        return mock_data
    
    def load_census_tracts(self, state_fips='47', county_fips='065', year=2020):
        """
        Load census tract geometries from Census TIGER files
        
        Parameters:
        -----------
        state_fips : str
            State FIPS code (e.g., '47' for Tennessee)
        county_fips : str
            County FIPS code (e.g., '065' for Hamilton)
        year : int
            Census year
            
        Returns:
        --------
        gpd.GeoDataFrame
            Census tract geometries with FIPS codes
        """
        self._log(f"Loading census tracts for state {state_fips}, county {county_fips}...")
        
        # Census TIGER URL
        tiger_url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state_fips}_tract.zip"
        
        try:
            # Load tract geometries
            tracts_gdf = gpd.read_file(tiger_url)
            self._log(f"  ✓ Downloaded {len(tracts_gdf)} census tracts for state {state_fips}")
            
            # Filter for specific county
            county_tracts = tracts_gdf[tracts_gdf['COUNTYFP'] == county_fips].copy()
            
            # Create full FIPS code
            county_tracts['FIPS'] = county_tracts['STATEFP'] + county_tracts['COUNTYFP'] + county_tracts['TRACTCE']
            
            self._log(f"  ✓ Filtered to {len(county_tracts)} tracts for county {county_fips}")
            
            return county_tracts
            
        except Exception as e:
            self._log(f"  ✗ Error loading census tracts: {str(e)}")
            
            # Create mock tract geometries
            self._log("  Creating mock census tract geometries...")
            return self._create_mock_census_tracts(state_fips, county_fips)
    
    def _create_mock_census_tracts(self, state_fips='47', county_fips='065'):
        """Create mock census tract geometries for testing"""
        from shapely.geometry import Polygon
        
        # Create simple rectangular tracts
        fips_codes = [
            f'{state_fips}{county_fips}000100', f'{state_fips}{county_fips}000200', 
            f'{state_fips}{county_fips}000300', f'{state_fips}{county_fips}000400'
        ]
        
        geometries = [
            Polygon([(-85.4, 35.0), (-85.3, 35.0), (-85.3, 35.1), (-85.4, 35.1)]),
            Polygon([(-85.3, 35.0), (-85.2, 35.0), (-85.2, 35.1), (-85.3, 35.1)]),
            Polygon([(-85.4, 35.1), (-85.3, 35.1), (-85.3, 35.2), (-85.4, 35.2)]),
            Polygon([(-85.3, 35.1), (-85.2, 35.1), (-85.2, 35.2), (-85.3, 35.2)])
        ]
        
        mock_tracts = gpd.GeoDataFrame({
            'STATEFP': [state_fips] * len(fips_codes),
            'COUNTYFP': [county_fips] * len(fips_codes),
            'TRACTCE': ['000100', '000200', '000300', '000400'],
            'FIPS': fips_codes,
            'geometry': geometries
        })
        
        self._log(f"  ✓ Created {len(mock_tracts)} mock census tracts")
        
        return mock_tracts
    
    # ... (rest of the methods remain the same)
    
    def load_road_network(self, roads_file=None, state_fips='47', county_fips='065', year=2023):
        """Load road network data - same as before but with better error handling"""
        if roads_file and os.path.exists(roads_file):
            self._log(f"Loading road network from {roads_file}...")
            try:
                roads_gdf = gpd.read_file(roads_file)
                self._log(f"  ✓ Loaded {len(roads_gdf)} road segments")
                return roads_gdf
            except Exception as e:
                self._log(f"  ✗ Error loading roads file: {e}")
        
        # Download from Census TIGER
        self._log(f"Downloading road network for county {state_fips}{county_fips}...")
        tiger_roads_url = f"https://www2.census.gov/geo/tiger/TIGER{year}/ROADS/tl_{year}_{state_fips}{county_fips}_roads.zip"
        
        try:
            roads_gdf = gpd.read_file(tiger_roads_url)
            self._log(f"  ✓ Downloaded {len(roads_gdf)} road segments")
            return roads_gdf
        except Exception as e:
            self._log(f"  ✗ Error downloading roads: {e}")
            return self._create_mock_roads()
    
    def _create_mock_roads(self):
        """Create mock road network for testing"""
        mock_roads = gpd.GeoDataFrame({
            'LINEARID': ['110001', '110002', '110003'],
            'FULLNAME': ['Main St', 'Oak Ave', 'Park Rd'],
            'MTFCC': ['S1100', 'S1200', 'S1200'],
            'geometry': [
                LineString([(-85.35, 35.05), (-85.25, 35.05)]),
                LineString([(-85.30, 35.0), (-85.30, 35.1)]),
                LineString([(-85.35, 35.1), (-85.25, 35.1)])
            ]
        })
        self._log(f"  ✓ Created {len(mock_roads)} mock road segments")
        return mock_roads
    
    def load_transit_stops(self):
        """Load mock transit stops for testing"""
        self._log("Creating mock transit stops...")
        
        mock_stops = gpd.GeoDataFrame({
            'stop_id': ['1001', '1002', '1003'],
            'stop_name': ['Downtown Station', 'University Stop', 'Mall Transit Center'],
            'geometry': [
                Point(-85.30, 35.05),
                Point(-85.28, 35.08),
                Point(-85.32, 35.12)
            ]
        })
        
        self._log(f"  ✓ Created {len(mock_stops)} mock transit stops")
        return mock_stops
    
    def load_address_points(self, n_synthetic=1000, bbox=(-85.5, 35.0, -85.0, 35.5)):
        """Generate synthetic address points - same as before"""
        self._log(f"Generating {n_synthetic} synthetic address points...")
        
        np.random.seed(42)
        
        # Generate random points within bounding box
        lons = np.random.uniform(bbox[0], bbox[2], n_synthetic)
        lats = np.random.uniform(bbox[1], bbox[3], n_synthetic)
        
        # Create GeoDataFrame
        addresses = gpd.GeoDataFrame({
            'address_id': range(n_synthetic),
            'longitude': lons,
            'latitude': lats,
            'geometry': [Point(lon, lat) for lon, lat in zip(lons, lats)]
        })
        
        self._log(f"  ✓ Generated {len(addresses)} address points")
        
        return addresses
    
    def create_network_graph(self, roads_gdf, directed=False):
        """Create NetworkX graph from road geometries - same as before"""
        self._log("Creating network graph from road geometries...")
        
        G = nx.DiGraph() if directed else nx.Graph()
        
        for idx, road in roads_gdf.iterrows():
            if hasattr(road.geometry, 'coords'):
                coords = list(road.geometry.coords)
                
                for i in range(len(coords) - 1):
                    u = coords[i]
                    v = coords[i + 1]
                    
                    length = np.sqrt((u[0] - v[0])**2 + (u[1] - v[1])**2)
                    G.add_edge(u, v, length=length, road_id=idx)
                    
                    if 'MTFCC' in road:
                        G[u][v]['road_type'] = road['MTFCC']
        
        for node in G.nodes():
            G.nodes[node]['x'] = node[0]
            G.nodes[node]['y'] = node[1]
        
        self._log(f"  ✓ Created network graph:")
        self._log(f"    - Nodes: {G.number_of_nodes()}")
        self._log(f"    - Edges: {G.number_of_edges()}")
        
        return G


# Updated convenience function
def load_hamilton_county_data(data_dir='./data', roads_file=None):
    """Load all data for Hamilton County analysis with robust error handling"""
    loader = DataLoader(data_dir=data_dir)
    
    print("\n" + "="*60)
    print("GRANITE: Loading Hamilton County Data")
    print("="*60 + "\n")
    
    try:
        data = {
            'svi': loader.load_svi_data(),
            'census_tracts': loader.load_census_tracts(),
            'roads': loader.load_road_network(roads_file=roads_file),
            'transit_stops': loader.load_transit_stops(),
            'addresses': loader.load_address_points()
        }
        
        data['road_network'] = loader.create_network_graph(data['roads'])
        
        print("\n" + "="*60)
        print("Data Loading Complete!")
        print("="*60)
        
        return data
        
    except Exception as e:
        print(f"\n⚠️  Error in data loading: {e}")
        print("Creating minimal mock dataset for testing...")
        
        # Return minimal mock data
        loader._log("Creating minimal mock dataset...")
        return {
            'svi': loader._create_mock_svi_data(),
            'census_tracts': loader._create_mock_census_tracts(),
            'roads': loader._create_mock_roads(),
            'transit_stops': loader.load_transit_stops(),
            'addresses': loader.load_address_points(n_synthetic=100),
            'road_network': loader.create_network_graph(loader._create_mock_roads())
        }