"""
Data loading functions for GRANITE framework
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import fiona
from shapely.geometry import Point, LineString
import networkx as nx
from datetime import datetime

class DataLoader:
    """Main data loader class for GRANITE framework"""
    
    def __init__(self, data_dir='./data', verbose=True):
        self.data_dir = data_dir
        self.verbose = verbose
        
    def _log(self, message):
        """Logging with timestamp"""
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def load_svi_data(self, state_fips='47', county_name='Hamilton', year=2020):
        """
        Load Social Vulnerability Index data for Hamilton County
        
        Parameters:
        -----------
        state_fips : str
            State FIPS code 
        county_name : str
            County name (e.g., 'Hamilton')
        year : int
            SVI data year
            
        Returns:
        --------
        pd.DataFrame
            SVI data with selected columns
        """
        self._log("Loading SVI data for Hamilton County, Tennessee...")
        
        # Local file path
        svi_file = os.path.join(self.data_dir, 'raw', f'SVI_{year}_US.csv')
        self._log(f"  Loading SVI data from: {svi_file}")
        
        try:
            # Load SVI data
            if os.path.exists(svi_file):
                # CRITICAL FIX: Force FIPS to be read as string
                svi_data = pd.read_csv(svi_file, dtype={'FIPS': str, 'ST': str, 'COUNTY': str})
                self._log(f"  ✓ Loaded local SVI data ({len(svi_data)} total records)")
            else:
                # Download if not available locally
                svi_url = f"https://www.atsdr.cdc.gov/placeandhealth/centerforsvi/csv/SVI{year}_US.csv"
                svi_data = pd.read_csv(svi_url, dtype={'FIPS': str, 'ST': str, 'COUNTY': str})
                self._log(f"  ✓ Downloaded SVI data ({len(svi_data)} total records)")
                
                # Save locally for future use
                os.makedirs(os.path.dirname(svi_file), exist_ok=True)
                svi_data.to_csv(svi_file, index=False)
                self._log(f"  ✓ Saved SVI data to {svi_file}")
            
            # **CRITICAL FIX**: Filter by county NAME, not FIPS code
            county_svi = svi_data[
                (svi_data['ST'] == state_fips) & 
                (svi_data['COUNTY'] == county_name)  # Use 'Hamilton' not '065'
            ].copy()
            
            # **CRITICAL FIX**: Ensure FIPS is string type to match census tracts
            county_svi['FIPS'] = county_svi['FIPS'].astype(str)
            
            # Select relevant columns
            columns = ['FIPS', 'LOCATION', 'RPL_THEMES', 'RPL_THEME1', 
                    'RPL_THEME2', 'RPL_THEME3', 'RPL_THEME4', 
                    'E_TOTPOP', 'E_HU', 'E_POV', 'E_UNEMP', 'E_NOHSDP',
                    'E_AGE65', 'E_AGE17', 'E_DISABL', 'E_SNGPNT', 
                    'E_MINRTY', 'E_LIMENG', 'E_MUNIT', 'E_MOBILE',
                    'E_CROWD', 'E_NOVEH', 'E_GROUPQ']
            
            available_cols = [col for col in columns if col in county_svi.columns]
            county_svi = county_svi[available_cols]
            
            # Handle missing values
            county_svi['RPL_THEMES'] = county_svi['RPL_THEMES'].replace(-999, np.nan)
            
            # Statistics
            valid_svi = county_svi['RPL_THEMES'].dropna()
            self._log(f"  ✓ Loaded SVI data for {len(county_svi)} census tracts")
            self._log(f"    - Valid SVI scores: {len(valid_svi)}")
            self._log(f"    - Mean SVI: {valid_svi.mean():.3f}")
            self._log(f"    - SVI Range: [{valid_svi.min():.3f}, {valid_svi.max():.3f}]")
            
            return county_svi
            
        except Exception as e:
            self._log(f"  ✗ Error loading SVI data: {str(e)}")
            raise
    
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
            Census tract geometries
        """
        self._log(f"Loading census tracts for state {state_fips}, county {county_fips}...")
        
        # Local file path
        local_file = os.path.join(self.data_dir, 'raw', f'tl_{year}_{state_fips}_tract.shp')
        
        try:
            if os.path.exists(local_file):
                # Load from local file
                tracts = gpd.read_file(local_file)
                self._log(f"  ✓ Loaded {len(tracts)} census tracts from local file")
            else:
                # Download from Census TIGER
                census_url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state_fips}_tract.zip"
                tracts = gpd.read_file(census_url)
                self._log(f"  ✓ Downloaded {len(tracts)} tracts for state {state_fips}")
                
                # Save locally for future use
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                tracts.to_file(local_file)
                self._log(f"  ✓ Saved census tracts to {local_file}")
            
            # Filter for specific county
            county_tracts = tracts[tracts['COUNTYFP'] == county_fips].copy()
            county_tracts['FIPS'] = county_tracts['GEOID'].astype(str)
            
            # Set CRS
            if county_tracts.crs is None:
                county_tracts.set_crs(epsg=4326, inplace=True)
                
            self._log(f"  ✓ Filtered to {len(county_tracts)} tracts for county {county_fips}")
            
            return county_tracts
            
        except Exception as e:
            self._log(f"  ✗ Error loading census tracts: {str(e)}")
            raise
    
    def load_road_network(self, roads_file=None, state_fips='47', county_fips='065'):
        """
        Load road network from TIGER/Line shapefiles
        
        Parameters:
        -----------
        roads_file : str, optional
            Path to roads shapefile. If None, downloads from Census.
        state_fips : str
            State FIPS code
        county_fips : str
            County FIPS code
            
        Returns:
        --------
        gpd.GeoDataFrame
            Road network geometries
        """
        self._log("Loading road network...")
        
        try:
            if roads_file and os.path.exists(roads_file):
                # Load from provided file
                roads = gpd.read_file(roads_file)
                self._log(f"  ✓ Loaded {len(roads)} road segments from {roads_file}")
            else:
                # Load from Census TIGER
                local_file = os.path.join(self.data_dir, 'raw', f'tl_2023_{state_fips}{county_fips}_roads.shp')
                
                if os.path.exists(local_file):
                    roads = gpd.read_file(local_file)
                    self._log(f"  ✓ Loaded {len(roads)} road segments from local file")
                else:
                    # Download roads
                    roads_url = f"https://www2.census.gov/geo/tiger/TIGER2023/ROADS/tl_2023_{state_fips}{county_fips}_roads.zip"
                    roads = gpd.read_file(roads_url)
                    self._log(f"  ✓ Downloaded {len(roads)} road segments")
                    
                    # Save locally
                    os.makedirs(os.path.dirname(local_file), exist_ok=True)
                    roads.to_file(local_file)
                    self._log(f"  ✓ Saved roads to {local_file}")
            
            # Set CRS if not present
            if roads.crs is None:
                roads.set_crs(epsg=4326, inplace=True)
            
            return roads
            
        except Exception as e:
            self._log(f"  ✗ Error loading road network: {str(e)}")
            # Return empty GeoDataFrame as fallback
            return gpd.GeoDataFrame(geometry=[])
    
    def load_transit_stops(self, n_mock=3):
        """
        Load transit stops (currently creates mock data)
        
        Parameters:
        -----------
        n_mock : int
            Number of mock transit stops to create
            
        Returns:
        --------
        gpd.GeoDataFrame
            Transit stop locations
        """
        self._log("Creating mock transit stops...")
        
        # Hamilton County approximate bounds
        bbox = [-85.5, 35.0, -85.0, 35.5]  # [min_lon, min_lat, max_lon, max_lat]
        
        # Generate random points within bbox
        np.random.seed(42)
        lons = np.random.uniform(bbox[0], bbox[2], n_mock)
        lats = np.random.uniform(bbox[1], bbox[3], n_mock)
        
        # Create GeoDataFrame
        stops = gpd.GeoDataFrame({
            'stop_id': [f'stop_{i+1}' for i in range(n_mock)],
            'stop_name': [f'Transit Stop {i+1}' for i in range(n_mock)],
            'geometry': [Point(lon, lat) for lon, lat in zip(lons, lats)]
        }, crs='EPSG:4326')
        
        self._log(f"  ✓ Created {len(stops)} mock transit stops")
        
        return stops
    
    def load_address_points(self, n_synthetic=1000):
        """
        Generate synthetic address points within Hamilton County
        
        Parameters:
        -----------
        n_synthetic : int
            Number of synthetic addresses to generate
            
        Returns:
        --------
        gpd.GeoDataFrame
            Address point locations
        """
        self._log(f"Generating {n_synthetic} synthetic address points...")
        
        # Hamilton County approximate bounds
        bbox = [-85.5, 35.0, -85.0, 35.5]  # [min_lon, min_lat, max_lon, max_lat]
        
        # Generate random points within bbox
        np.random.seed(42)
        lons = np.random.uniform(bbox[0], bbox[2], n_synthetic)
        lats = np.random.uniform(bbox[1], bbox[3], n_synthetic)
        
        # Create GeoDataFrame
        addresses = gpd.GeoDataFrame({
            'address_id': [f'addr_{i+1}' for i in range(n_synthetic)],
            'longitude': lons,
            'latitude': lats,
            'geometry': [Point(lon, lat) for lon, lat in zip(lons, lats)]
        }, crs='EPSG:4326')
        
        self._log(f"  ✓ Generated {len(addresses)} address points")
        
        return addresses
    
    def create_network_graph(self, roads_gdf, directed=False):
        """
        Create NetworkX graph from road geometries
        
        Parameters:
        -----------
        roads_gdf : gpd.GeoDataFrame
            Road network geometries
        directed : bool
            Whether to create directed graph
            
        Returns:
        --------
        nx.Graph or nx.DiGraph
            Network graph representation
        """
        self._log("Creating network graph from road geometries...")
        
        if len(roads_gdf) == 0:
            self._log("  ⚠️  No road data available, creating empty graph")
            return nx.Graph() if not directed else nx.DiGraph()
        
        # Create graph
        G = nx.DiGraph() if directed else nx.Graph()
        
        # Process each road segment
        for idx, road in roads_gdf.iterrows():
            if road.geometry.geom_type == 'LineString':
                # Extract coordinates
                coords = list(road.geometry.coords)
                
                # Add edges between consecutive points
                for i in range(len(coords) - 1):
                    u, v = coords[i], coords[i + 1]
                    
                    # Calculate edge length
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
        self._log(f"    - Connected components: {nx.number_connected_components(G) if not directed else 'N/A'}")
        
        return G


# Convenience functions for direct imports
def load_hamilton_county_data(data_dir='./data', roads_file=None):
    """
    Load all data for Hamilton County analysis
    
    Parameters:
    -----------
    data_dir : str
        Directory containing data files
    roads_file : str, optional
        Path to roads shapefile (e.g., 'tl_2023_47065_roads.shp')
        
    Returns:
    --------
    dict
        Dictionary containing all loaded datasets
    """
    loader = DataLoader(data_dir=data_dir)
    
    print("\n" + "="*60)
    print("GRANITE: Loading Hamilton County Data")
    print("="*60 + "\n")
    
    # Load all datasets
    data = {
        'svi': loader.load_svi_data(),
        'census_tracts': loader.load_census_tracts(),
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