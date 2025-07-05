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
    
    def load_svi_data(self, state='Tennessee', county='Hamilton'):
        """
        Load Social Vulnerability Index data from CDC
        
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
        
        # CDC SVI data URL
        svi_url = f"https://svi.cdc.gov/data/csv/2020/States/{state}.csv"
        
        try:
            # Load data
            svi_df = pd.read_csv(svi_url)
            self._log(f"  ✓ Downloaded SVI data for {state}")
            
            # Filter for specific county
            county_svi = svi_df[svi_df['COUNTY'] == county].copy()
            
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
        self._log(f"Loading census tract geometries for {state_fips}-{county_fips}...")
        
        # Census TIGER URL
        census_url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state_fips}_tract.zip"
        
        try:
            # Load tract data
            tracts = gpd.read_file(census_url)
            self._log(f"  ✓ Downloaded {len(tracts)} tracts for state {state_fips}")
            
            # Filter for specific county
            county_tracts = tracts[tracts['COUNTYFP'] == county_fips].copy()
            county_tracts['FIPS'] = county_tracts['GEOID'].astype(str)
            
            # Set CRS
            if county_tracts.crs is None:
                county_tracts.set_crs(epsg=4326, inplace=True)
                
            self._log(f"  ✓ Filtered to {len(county_tracts)} tracts in county {county_fips}")
            self._log(f"    - CRS: {county_tracts.crs}")
            self._log(f"    - Total area: {county_tracts.geometry.area.sum():.2f} sq degrees")
            
            return county_tracts
            
        except Exception as e:
            self._log(f"  ✗ Error loading census tracts: {str(e)}")
            raise
    
    def load_road_network(self, roads_file=None, state_fips='47', county_fips='065'):
        """
        Load road network from TIGER/Line shapefiles
        
        This incorporates your existing road loading code!
        
        Parameters:
        -----------
        roads_file : str, optional
            Path to roads shapefile. If None, downloads from Census
        state_fips : str
            State FIPS code
        county_fips : str  
            County FIPS code
            
        Returns:
        --------
        gpd.GeoDataFrame
            Road network data
        """
        self._log("Loading road network...")
        
        # Use provided file or download
        if roads_file and os.path.exists(roads_file):
            self._log(f"  Loading from local file: {roads_file}")
            
            # Your existing code for loading roads
            with fiona.open(roads_file) as collection:
                roads_df = gpd.GeoDataFrame.from_features(collection)
                roads_df.set_crs(epsg=4326, inplace=True)  # Set initial CRS to WGS84
            
            self._log(f"  ✓ Imported {len(roads_df)} road segments")
            
        else:
            # Download from Census if no local file
            self._log("  Downloading from Census TIGER/Line...")
            year = 2023
            roads_url = f"https://www2.census.gov/geo/tiger/TIGER{year}/ROADS/tl_{year}_{state_fips}{county_fips}_roads.zip"
            
            try:
                roads_df = gpd.read_file(roads_url)
                roads_df.set_crs(epsg=4326, inplace=True)
                self._log(f"  ✓ Downloaded {len(roads_df)} road segments")
            except:
                # Fallback for different year
                year = 2022
                roads_url = f"https://www2.census.gov/geo/tiger/TIGER{year}/ROADS/tl_{year}_{state_fips}{county_fips}_roads.zip"
                roads_df = gpd.read_file(roads_url)
                roads_df.set_crs(epsg=4326, inplace=True)
                self._log(f"  ✓ Downloaded {len(roads_df)} road segments (using {year} data)")
        
        # Add useful attributes
        roads_df['length_m'] = roads_df.geometry.to_crs('EPSG:3857').length
        
        # Road type statistics
        if 'MTFCC' in roads_df.columns:
            road_types = roads_df['MTFCC'].value_counts()
            self._log("  Road type distribution:")
            for rtype, count in road_types.head(5).items():
                self._log(f"    - {rtype}: {count} segments")
        
        # Filter to major roads if needed (optional)
        # major_roads = roads_df[roads_df['MTFCC'].isin(['S1100', 'S1200'])]  # Primary and secondary roads
        
        return roads_df
    
    def load_transit_stops(self, transit_file=None):
        """
        Load transit stop data (bus stops, rail stations)
        
        Parameters:
        -----------
        transit_file : str, optional
            Path to transit stops file (GTFS or shapefile)
            
        Returns:
        --------
        gpd.GeoDataFrame
            Transit stop locations
        """
        self._log("Loading transit stops...")
        
        if transit_file and os.path.exists(transit_file):
            # Load from file
            if transit_file.endswith('.txt'):
                # GTFS format
                stops_df = pd.read_csv(transit_file)
                geometry = [Point(xy) for xy in zip(stops_df.stop_lon, stops_df.stop_lat)]
                stops_gdf = gpd.GeoDataFrame(stops_df, geometry=geometry, crs='EPSG:4326')
            else:
                # Shapefile
                stops_gdf = gpd.read_file(transit_file)
            
            self._log(f"  ✓ Loaded {len(stops_gdf)} transit stops")
            
        else:
            # Generate synthetic stops for demo
            self._log("  ⚠ No transit file provided, generating synthetic stops")
            
            n_stops = 50
            bbox = (-85.5, 35.0, -85.0, 35.5)  # Hamilton County approximate bounds
            
            stops_data = {
                'stop_id': range(n_stops),
                'stop_name': [f'Stop {i}' for i in range(n_stops)],
                'stop_lon': np.random.uniform(bbox[0], bbox[2], n_stops),
                'stop_lat': np.random.uniform(bbox[1], bbox[3], n_stops),
                'stop_type': np.random.choice(['bus', 'rail'], n_stops, p=[0.9, 0.1])
            }
            
            stops_df = pd.DataFrame(stops_data)
            geometry = [Point(xy) for xy in zip(stops_df.stop_lon, stops_df.stop_lat)]
            stops_gdf = gpd.GeoDataFrame(stops_df, geometry=geometry, crs='EPSG:4326')
            
            self._log(f"  ✓ Generated {len(stops_gdf)} synthetic transit stops")
        
        return stops_gdf
    
    def load_address_points(self, addresses_file=None, n_synthetic=1000):
        """
        Load address points for disaggregation
        
        Parameters:
        -----------
        addresses_file : str, optional
            Path to address points file
        n_synthetic : int
            Number of synthetic addresses to generate if no file provided
            
        Returns:
        --------
        gpd.GeoDataFrame
            Address point locations
        """
        self._log("Loading address points...")
        
        if addresses_file and os.path.exists(addresses_file):
            # Load from file
            addresses_gdf = gpd.read_file(addresses_file)
            self._log(f"  ✓ Loaded {len(addresses_gdf)} address points")
            
        else:
            # Generate synthetic addresses
            self._log(f"  ⚠ No address file provided, generating {n_synthetic} synthetic addresses")
            
            bbox = (-85.5, 35.0, -85.0, 35.5)  # Hamilton County approximate bounds
            
            addresses_data = {
                'address_id': range(n_synthetic),
                'longitude': np.random.uniform(bbox[0], bbox[2], n_synthetic),
                'latitude': np.random.uniform(bbox[1], bbox[3], n_synthetic)
            }
            
            # Add synthetic demographic features
            addresses_data['population_density'] = np.random.lognormal(7, 1.5, n_synthetic)
            addresses_data['median_income'] = np.random.lognormal(10.5, 0.7, n_synthetic)
            addresses_data['pct_minority'] = np.random.beta(2, 5, n_synthetic)
            addresses_data['distance_to_hospital'] = np.random.exponential(2, n_synthetic)
            
            addresses_df = pd.DataFrame(addresses_data)
            geometry = [Point(xy) for xy in zip(addresses_df.longitude, addresses_df.latitude)]
            addresses_gdf = gpd.GeoDataFrame(addresses_df, geometry=geometry, crs='EPSG:4326')
            
            self._log(f"  ✓ Generated {len(addresses_gdf)} synthetic addresses")
        
        return addresses_gdf
    
    def create_network_graph(self, roads_gdf, directed=False):
        """
        Convert road GeoDataFrame to NetworkX graph
        
        Parameters:
        -----------
        roads_gdf : gpd.GeoDataFrame
            Road network data
        directed : bool
            Whether to create directed graph
            
        Returns:
        --------
        nx.Graph or nx.DiGraph
            Road network as graph
        """
        self._log("Converting roads to network graph...")
        
        # Create graph
        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        
        # Add edges from road segments
        for idx, road in roads_gdf.iterrows():
            if road.geometry.geom_type == 'LineString':
                coords = list(road.geometry.coords)
                
                # Add edges between consecutive points
                for i in range(len(coords) - 1):
                    u = coords[i]
                    v = coords[i + 1]
                    
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