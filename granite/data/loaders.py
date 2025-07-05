"""
Data loading functions for GRANITE framework
Updated with robust URL handling, local file support, and better error handling
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import fiona
from shapely.geometry import Point, LineString, Polygon
import networkx as nx
from datetime import datetime
import urllib.request
import urllib.error
import requests


class DataLoader:
    """Main data loader class for GRANITE framework with robust error handling"""
    
    def __init__(self, data_dir='./data', verbose=True):
        self.data_dir = data_dir
        self.verbose = verbose
        
        # Ensure data directories exist
        os.makedirs(os.path.join(self.data_dir, 'raw'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'processed'), exist_ok=True)
        
        # Updated SVI URLs (current as of July 2025)
        self.svi_urls = [
            "https://svi.cdc.gov/Documents/Data/2020/csv/SVI2020_US_tract.csv",
            "https://svi.cdc.gov/Documents/Data/2020/csv/SVI_2020_US.csv",
            "https://www.atsdr.cdc.gov/placeandhealth/svi/data/SviDataCsv/SVI_2020_US.csv"
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
                self._log(f"  Trying SVI URL {i+1}/{len(self.svi_urls)}: {url[:60]}{'...' if len(url) > 60 else ''}")
                
                # Use requests with timeout
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    # Try to parse as CSV
                    from io import StringIO
                    df = pd.read_csv(StringIO(response.text), nrows=5)
                    if len(df.columns) > 10:  # SVI should have many columns
                        self._log(f"  ✓ URL {i+1} is working - CSV with {len(df.columns)} columns")
                        return pd.read_csv(StringIO(response.text))
                    else:
                        self._log(f"  ✗ URL {i+1} returned invalid CSV - only {len(df.columns)} columns")
                else:
                    self._log(f"  ✗ URL {i+1} failed: HTTP {response.status_code}")
                    
            except Exception as e:
                self._log(f"  ✗ URL {i+1} failed: {str(e)[:50]}")
                continue
        
        return None
    
    def load_svi_data(self, state='Tennessee', county='Hamilton', svi_file=None):
        """
        Load Social Vulnerability Index data from local file or CDC with robust URL handling
        
        Parameters:
        -----------
        state : str
            State name for SVI data
        county : str
            County name to filter
        svi_file : str, optional
            Path to local SVI CSV file
            
        Returns:
        --------
        pd.DataFrame
            SVI data for specified county
        """
        self._log(f"Loading SVI data for {county} County, {state}...")
        
        # Check for local file first
        if svi_file and os.path.exists(svi_file):
            self._log(f"  Loading SVI data from local file: {svi_file}")
            try:
                svi_df = pd.read_csv(svi_file)
                self._log(f"  ✓ Loaded local SVI data ({len(svi_df)} total records)")
            except Exception as e:
                self._log(f"  ✗ Error loading local SVI file: {e}")
                self._log("  Falling back to URL download...")
                svi_df = self._try_download_svi()
        else:
            # Check for file in data/raw directory
            local_svi_file = os.path.join(self.data_dir, 'raw', 'SVI2020_US_tract.csv')
            alt_svi_file = os.path.join(self.data_dir, 'raw', 'SVI_2020_US.csv')
            
            if os.path.exists(local_svi_file):
                self._log(f"  Loading SVI data from: {local_svi_file}")
                try:
                    svi_df = pd.read_csv(local_svi_file)
                    self._log(f"  ✓ Loaded local SVI data ({len(svi_df)} total records)")
                except Exception as e:
                    self._log(f"  ✗ Error loading local SVI file: {e}")
                    svi_df = self._try_download_svi()
            elif os.path.exists(alt_svi_file):
                self._log(f"  Loading SVI data from: {alt_svi_file}")
                try:
                    svi_df = pd.read_csv(alt_svi_file)
                    self._log(f"  ✓ Loaded local SVI data ({len(svi_df)} total records)")
                except Exception as e:
                    self._log(f"  ✗ Error loading local SVI file: {e}")
                    svi_df = self._try_download_svi()
            else:
                # Try to download SVI data
                svi_df = self._try_download_svi()
        
        if svi_df is None:
            # If all methods fail, create mock data for testing
            self._log("  ⚠️  All SVI sources failed. Creating mock data for testing...")
            return self._create_mock_svi_data(state, county)
        
        try:
            # Filter for specific state and county
            # Try different column name variations
            state_cols = ['STATE', 'ST', 'ST_ABBR', 'STUSPS', 'STNAME']
            county_cols = ['COUNTY', 'CNTY', 'COUNTY_NAME', 'LOCATION']
            
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
                # Filter data by state and county
                state_filter = svi_df[state_col].astype(str).str.contains(state, case=False, na=False)
                county_filter = svi_df[county_col].astype(str).str.contains(county, case=False, na=False)
                county_svi = svi_df[state_filter & county_filter].copy()
            elif 'FIPS' in svi_df.columns:
                # Fallback: use FIPS code filtering
                # Hamilton County, TN FIPS: 47065
                fips_pattern = '47065'
                county_svi = svi_df[svi_df['FIPS'].astype(str).str.startswith(fips_pattern)].copy()
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
            self._log(f"  ✗ Error processing SVI data: {str(e)}")
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
    
    def load_census_tracts(self, state_fips='47', county_fips='065', year=2020, census_tracts_file=None):
        """
        Load census tract geometries from local file or Census TIGER files
        
        Parameters:
        -----------
        state_fips : str
            State FIPS code (e.g., '47' for Tennessee)
        county_fips : str
            County FIPS code (e.g., '065' for Hamilton)
        year : int
            Census year
        census_tracts_file : str, optional
            Path to local census tracts shapefile
            
        Returns:
        --------
        gpd.GeoDataFrame
            Census tract geometries with FIPS codes
        """
        self._log(f"Loading census tracts for state {state_fips}, county {county_fips}...")
        
        # Check for local file first
        if census_tracts_file and os.path.exists(census_tracts_file):
            self._log(f"  Loading census tracts from local file: {census_tracts_file}")
            try:
                tracts_gdf = gpd.read_file(census_tracts_file)
                self._log(f"  ✓ Loaded {len(tracts_gdf)} census tracts from local file")
            except Exception as e:
                self._log(f"  ✗ Error loading local tracts file: {e}")
                tracts_gdf = None
        else:
            # Check for file in data/raw directory
            local_tracts_file = os.path.join(self.data_dir, 'raw', f'tl_{year}_{state_fips}_tract.shp')
            
            if os.path.exists(local_tracts_file):
                self._log(f"  Loading census tracts from: {local_tracts_file}")
                try:
                    tracts_gdf = gpd.read_file(local_tracts_file)
                    self._log(f"  ✓ Loaded {len(tracts_gdf)} census tracts from local file")
                except Exception as e:
                    self._log(f"  ✗ Error loading local tracts file: {e}")
                    tracts_gdf = None
            else:
                # Try to download from Census TIGER
                tiger_url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state_fips}_tract.zip"
                
                try:
                    self._log(f"  Downloading from: {tiger_url}")
                    tracts_gdf = gpd.read_file(tiger_url)
                    self._log(f"  ✓ Downloaded {len(tracts_gdf)} census tracts for state {state_fips}")
                except Exception as e:
                    self._log(f"  ✗ Error downloading census tracts: {str(e)}")
                    tracts_gdf = None
        
        if tracts_gdf is None:
            # Create mock tract geometries
            self._log("  Creating mock census tract geometries...")
            return self._create_mock_census_tracts(state_fips, county_fips)
        
        # Filter for specific county
        if 'COUNTYFP' in tracts_gdf.columns:
            county_tracts = tracts_gdf[tracts_gdf['COUNTYFP'] == county_fips].copy()
        else:
            # If no county column, return all tracts (for debugging)
            county_tracts = tracts_gdf.copy()
        
        # Create full FIPS code
        if 'STATEFP' in county_tracts.columns and 'COUNTYFP' in county_tracts.columns and 'TRACTCE' in county_tracts.columns:
            county_tracts['FIPS'] = county_tracts['STATEFP'] + county_tracts['COUNTYFP'] + county_tracts['TRACTCE']
        
        self._log(f"  ✓ Filtered to {len(county_tracts)} tracts for county {county_fips}")
        
        return county_tracts
    
    def _create_mock_census_tracts(self, state_fips='47', county_fips='065'):
        """Create mock census tract geometries for testing"""
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
    
    def load_road_network(self, roads_file=None, state_fips='47', county_fips='065', year=2023):
        """
        Load road network data from local file or Census TIGER with better error handling
        
        Parameters:
        -----------
        roads_file : str, optional
            Path to roads shapefile
        state_fips : str
            State FIPS code
        county_fips : str
            County FIPS code
        year : int
            Year for Census TIGER data
            
        Returns:
        --------
        gpd.GeoDataFrame
            Road network data
        """
        self._log("Loading road network...")
        
        # Check for explicitly provided file
        if roads_file and os.path.exists(roads_file):
            self._log(f"  Loading road network from: {roads_file}")
            try:
                roads_gdf = gpd.read_file(roads_file)
                self._log(f"  ✓ Loaded {len(roads_gdf)} road segments from local file")
                return roads_gdf
            except Exception as e:
                self._log(f"  ✗ Error loading roads file: {e}")
        
        # Check for file in data/raw directory
        local_roads_file = os.path.join(self.data_dir, 'raw', f'tl_{year}_{state_fips}{county_fips}_roads.shp')
        
        if os.path.exists(local_roads_file):
            self._log(f"  Loading road network from: {local_roads_file}")
            try:
                roads_gdf = gpd.read_file(local_roads_file)
                self._log(f"  ✓ Loaded {len(roads_gdf)} road segments from local file")
                return roads_gdf
            except Exception as e:
                self._log(f"  ✗ Error loading local roads file: {e}")
        
        # Try to download from Census TIGER
        self._log(f"  Downloading road network for county {state_fips}{county_fips}...")
        tiger_roads_url = f"https://www2.census.gov/geo/tiger/TIGER{year}/ROADS/tl_{year}_{state_fips}{county_fips}_roads.zip"
        
        try:
            roads_gdf = gpd.read_file(tiger_roads_url)
            self._log(f"  ✓ Downloaded {len(roads_gdf)} road segments")
            return roads_gdf
        except Exception as e:
            self._log(f"  ✗ Error downloading roads: {e}")
            self._log("  Creating mock road network...")
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
    
    def load_transit_stops(self, transit_file=None):
        """
        Load transit stops from local file or create mock data
        
        Parameters:
        -----------
        transit_file : str, optional
            Path to transit stops file
            
        Returns:
        --------
        gpd.GeoDataFrame
            Transit stop locations
        """
        if transit_file and os.path.exists(transit_file):
            self._log(f"Loading transit stops from: {transit_file}")
            try:
                transit_gdf = gpd.read_file(transit_file)
                self._log(f"  ✓ Loaded {len(transit_gdf)} transit stops")
                return transit_gdf
            except Exception as e:
                self._log(f"  ✗ Error loading transit file: {e}")
        
        # Check for file in data/raw directory
        local_transit_file = os.path.join(self.data_dir, 'raw', 'transit_stops.shp')
        
        if os.path.exists(local_transit_file):
            self._log(f"Loading transit stops from: {local_transit_file}")
            try:
                transit_gdf = gpd.read_file(local_transit_file)
                self._log(f"  ✓ Loaded {len(transit_gdf)} transit stops from local file")
                return transit_gdf
            except Exception as e:
                self._log(f"  ✗ Error loading local transit file: {e}")
        
        # Create mock transit stops
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
    
    def load_address_points(self, addresses_file=None, n_synthetic=1000, bbox=(-85.5, 35.0, -85.0, 35.5)):
        """
        Load address points from local file or generate synthetic points
        
        Parameters:
        -----------
        addresses_file : str, optional
            Path to address points file
        n_synthetic : int
            Number of synthetic addresses to generate if no file
        bbox : tuple
            Bounding box for synthetic addresses (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
        --------
        gpd.GeoDataFrame
            Address point locations
        """
        if addresses_file and os.path.exists(addresses_file):
            self._log(f"Loading address points from: {addresses_file}")
            try:
                addresses_gdf = gpd.read_file(addresses_file)
                self._log(f"  ✓ Loaded {len(addresses_gdf)} address points")
                return addresses_gdf
            except Exception as e:
                self._log(f"  ✗ Error loading addresses file: {e}")
        
        # Check for file in data/raw directory
        local_addresses_file = os.path.join(self.data_dir, 'raw', 'address_points.shp')
        
        if os.path.exists(local_addresses_file):
            self._log(f"Loading address points from: {local_addresses_file}")
            try:
                addresses_gdf = gpd.read_file(local_addresses_file)
                self._log(f"  ✓ Loaded {len(addresses_gdf)} address points from local file")
                return addresses_gdf
            except Exception as e:
                self._log(f"  ✗ Error loading local addresses file: {e}")
        
        # Generate synthetic address points
        self._log(f"Generating {n_synthetic} synthetic address points...")
        
        np.random.seed(42)  # Reproducible addresses
        
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
        """
        Create NetworkX graph from road geometries
        
        Parameters:
        -----------
        roads_gdf : gpd.GeoDataFrame
            Road network data
        directed : bool
            Whether to create directed graph
            
        Returns:
        --------
        nx.Graph or nx.DiGraph
            Network graph
        """
        self._log("Creating network graph from road geometries...")
        
        G = nx.DiGraph() if directed else nx.Graph()
        
        for idx, road in roads_gdf.iterrows():
            if hasattr(road.geometry, 'coords'):
                coords = list(road.geometry.coords)
                
                # Add edges between consecutive coordinates
                for i in range(len(coords) - 1):
                    u = coords[i]
                    v = coords[i + 1]
                    
                    # Calculate length
                    length = np.sqrt((u[0] - v[0])**2 + (u[1] - v[1])**2)
                    
                    # Add edge with attributes
                    G.add_edge(u, v, length=length, road_id=idx)
                    
                    # Add road type if available
                    if 'MTFCC' in road:
                        G[u][v]['road_type'] = road['MTFCC']
        
        # Add node attributes
        for node in G.nodes():
            G.nodes[node]['x'] = node[0]
            G.nodes[node]['y'] = node[1]
        
        self._log(f"  ✓ Created network graph:")
        self._log(f"    - Nodes: {G.number_of_nodes()}")
        self._log(f"    - Edges: {G.number_of_edges()}")
        if not directed:
            self._log(f"    - Connected components: {nx.number_connected_components(G)}")
        
        return G


# Convenience functions for direct imports
def load_hamilton_county_data(data_dir='./data', roads_file=None, svi_file=None, census_tracts_file=None):
    """
    Load all data for Hamilton County analysis with robust error handling
    
    Parameters:
    -----------
    data_dir : str
        Directory containing data files
    roads_file : str, optional
        Path to roads shapefile
    svi_file : str, optional
        Path to SVI CSV file
    census_tracts_file : str, optional
        Path to census tracts shapefile
        
    Returns:
    --------
    dict
        Dictionary containing all loaded datasets
    """
    loader = DataLoader(data_dir=data_dir)
    
    print("\n" + "="*60)
    print("GRANITE: Loading Hamilton County Data")
    print("="*60 + "\n")
    
    try:
        # Load all datasets
        data = {
            'svi': loader.load_svi_data(svi_file=svi_file),
            'census_tracts': loader.load_census_tracts(census_tracts_file=census_tracts_file),
            'roads': loader.load_road_network(roads_file=roads_file),
            'transit_stops': loader.load_transit_stops(),
            'addresses': loader.load_address_points()
        }
        
        # Create network graph
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