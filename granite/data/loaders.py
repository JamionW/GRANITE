"""
Data loading functions for GRANITE framework

This module handles all data loading operations including SVI data,
census tracts, road networks, and address generation.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import networkx as nx
from datetime import datetime
from typing import List, Dict, Optional, Tuple


class DataLoader:
    """Main data loader class for GRANITE framework"""
    
    def __init__(self, data_dir: str = './data', verbose: bool = True):
        """
        Initialize data loader
        
        Parameters:
        -----------
        data_dir : str
            Base directory for data files
        verbose : bool
            Enable verbose logging
        """
        self.data_dir = data_dir
        self.verbose = verbose
        
        # Create data directories if needed
        os.makedirs(os.path.join(data_dir, 'raw'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'processed'), exist_ok=True)
    
    def _log(self, message: str):
        """Log message with timestamp"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] DataLoader: {message}")
    
    def load_svi_data(self, state_fips: str = '47', county_name: str = 'Hamilton', 
                     year: int = 2020) -> pd.DataFrame:
        """
        Load Social Vulnerability Index data
        
        Parameters:
        -----------
        state_fips : str
            State FIPS code
        county_name : str
            County name (not FIPS)
        year : int
            SVI data year
            
        Returns:
        --------
        pd.DataFrame
            SVI data for specified county
        """
        self._log(f"Loading SVI data for {county_name} County, {state_fips}...")
        
        svi_file = os.path.join(self.data_dir, 'raw', f'SVI_{year}_US.csv')
        
        try:
            # Load SVI data with proper types
            if os.path.exists(svi_file):
                svi_data = pd.read_csv(svi_file, dtype={'FIPS': str, 'ST': str})
                self._log(f"Loaded local SVI data ({len(svi_data)} records)")
            else:
                # Download from CDC
                url = f"https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download/{year}.html"
                self._log(f"Local file not found. Please download SVI data from: {url}")
                raise FileNotFoundError(f"SVI data not found at {svi_file}")
            
            # Filter to county
            county_svi = svi_data[
                (svi_data['ST'] == state_fips) & 
                (svi_data['COUNTY'] == county_name)
            ].copy()
            
            # Ensure FIPS is string
            county_svi['FIPS'] = county_svi['FIPS'].astype(str)
            
            # Select relevant columns
            columns = ['FIPS', 'LOCATION', 'RPL_THEMES', 'E_TOTPOP']
            county_svi = county_svi[columns]
            
            # Handle missing values
            county_svi['RPL_THEMES'] = county_svi['RPL_THEMES'].replace(-999, np.nan)
            
            valid_count = county_svi['RPL_THEMES'].notna().sum()
            self._log(f"Loaded {len(county_svi)} tracts ({valid_count} with valid SVI)")
            
            return county_svi
            
        except Exception as e:
            self._log(f"Error loading SVI data: {str(e)}")
            raise
    
    def load_census_tracts(self, state_fips: str = '47', county_fips: str = '065', 
                          year: int = 2020) -> gpd.GeoDataFrame:
        """
        Load census tract geometries
        
        Parameters:
        -----------
        state_fips : str
            State FIPS code
        county_fips : str
            County FIPS code
        year : int
            Census year
            
        Returns:
        --------
        gpd.GeoDataFrame
            Census tract geometries
        """
        self._log(f"Loading census tracts for {state_fips}-{county_fips}...")
        
        local_file = os.path.join(self.data_dir, 'raw', f'tl_{year}_{state_fips}_tract.shp')
        
        try:
            if os.path.exists(local_file):
                tracts = gpd.read_file(local_file)
                self._log(f"Loaded {len(tracts)} tracts from local file")
            else:
                # User needs to download from Census
                url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/"
                self._log(f"Local file not found. Please download from: {url}")
                raise FileNotFoundError(f"Census tracts not found at {local_file}")
            
            # Filter to county
            county_tracts = tracts[tracts['COUNTYFP'] == county_fips].copy()
            county_tracts['FIPS'] = county_tracts['GEOID'].astype(str)
            
            # Ensure CRS
            if county_tracts.crs is None:
                county_tracts.set_crs(epsg=4326, inplace=True)
                
            self._log(f"Filtered to {len(county_tracts)} tracts")
            
            return county_tracts
            
        except Exception as e:
            self._log(f"Error loading census tracts: {str(e)}")
            raise
    
    def load_road_network(self, roads_file: Optional[str] = None, 
                         state_fips: str = '47', county_fips: str = '065') -> gpd.GeoDataFrame:
        """
        Load road network data
        
        Parameters:
        -----------
        roads_file : str, optional
            Path to roads shapefile
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
                roads = gpd.read_file(roads_file)
                self._log(f"Loaded {len(roads)} road segments from {roads_file}")
            else:
                # Try default location
                default_file = os.path.join(
                    self.data_dir, 'raw', 
                    f'tl_2023_{state_fips}{county_fips}_roads.shp'
                )
                
                if os.path.exists(default_file):
                    roads = gpd.read_file(default_file)
                    self._log(f"Loaded {len(roads)} road segments")
                else:
                    url = f"https://www2.census.gov/geo/tiger/TIGER2023/ROADS/"
                    self._log(f"Road file not found. Please download from: {url}")
                    raise FileNotFoundError(f"Roads not found")
            
            # Ensure CRS
            if roads.crs is None:
                roads.set_crs(epsg=4326, inplace=True)
            
            return roads
            
        except Exception as e:
            self._log(f"Error loading roads: {str(e)}")
            # Return empty GeoDataFrame
            return gpd.GeoDataFrame(geometry=[])
    
    def create_network_graph(self, roads: gpd.GeoDataFrame) -> nx.Graph:
        """
        Create NetworkX graph from road geometries
        
        Parameters:
        -----------
        roads : gpd.GeoDataFrame
            Road geometries
            
        Returns:
        --------
        nx.Graph
            Network graph
        """
        if len(roads) == 0:
            self._log("No roads provided for network creation")
            return nx.Graph()
        
        self._log("Creating network graph...")
        
        G = nx.Graph()
        
        # Process each road segment
        for idx, road in roads.iterrows():
            if road.geometry.geom_type == 'LineString':
                coords = list(road.geometry.coords)
                
                # Add edges between consecutive points
                for i in range(len(coords) - 1):
                    u, v = coords[i], coords[i + 1]
                    
                    # Calculate edge length
                    length = Point(u).distance(Point(v))
                    
                    # Add edge
                    G.add_edge(u, v, length=length, road_id=idx)
        
        # Add node attributes
        for node in G.nodes():
            G.nodes[node]['x'] = node[0]
            G.nodes[node]['y'] = node[1]
        
        self._log(f"Created network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def get_available_fips_codes(self, state_fips: str = '47', 
                                county_fips: str = '065') -> List[str]:
        """
        Get list of available FIPS codes
        
        Parameters:
        -----------
        state_fips : str
            State FIPS code
        county_fips : str
            County FIPS code
            
        Returns:
        --------
        List[str]
            Available FIPS codes
        """
        tracts = self.load_census_tracts(state_fips, county_fips)
        return sorted(tracts['FIPS'].tolist())
    
    def load_single_tract_data(self, fips_code: str, buffer_degrees: float = 0.01,
                              max_nodes: Optional[int] = None, 
                              max_edges: Optional[int] = None) -> Dict:
        """
        Load data for a single census tract
        
        This is the key function for FIPS-based processing. It loads only
        the data needed for a single tract, with proper spatial filtering.
        
        Parameters:
        -----------
        fips_code : str
            11-digit FIPS code
        buffer_degrees : float
            Buffer around tract boundary
        max_nodes : int, optional
            Maximum nodes (for memory management)
        max_edges : int, optional
            Maximum edges (for memory management)
            
        Returns:
        --------
        Dict
            Tract-specific data
        """
        self._log(f"Loading data for tract {fips_code}")
        
        try:
            # Parse FIPS components
            state_fips = fips_code[:2]
            county_fips = fips_code[2:5]
            
            # Load tract geometry
            all_tracts = self.load_census_tracts(state_fips, county_fips)
            tract = all_tracts[all_tracts['FIPS'] == fips_code]
            
            if len(tract) == 0:
                raise ValueError(f"Tract {fips_code} not found")
            
            tract_geom = tract.geometry.iloc[0]
            
            # Create buffered boundary
            buffered_geom = tract_geom.buffer(buffer_degrees)
            bbox = buffered_geom.bounds  # (minx, miny, maxx, maxy)
            
            # Load and filter roads
            all_roads = self.load_road_network(None, state_fips, county_fips)
            
            # Spatial filter - roads intersecting buffered tract
            roads_filtered = all_roads[all_roads.geometry.intersects(buffered_geom)]
            self._log(f"Filtered to {len(roads_filtered)} road segments")
            
            # Create network from filtered roads
            road_network = self.create_network_graph(roads_filtered)
            
            # Apply size limits if specified
            if max_nodes and road_network.number_of_nodes() > max_nodes:
                self._log(f"Network exceeds {max_nodes} nodes, may need simplification")
            
            # Load SVI data for tract
            all_svi = self.load_svi_data(state_fips, self._get_county_name(state_fips, county_fips))
            tract_svi = all_svi[all_svi['FIPS'] == fips_code]
            
            # Generate addresses within tract
            addresses = self._generate_tract_addresses(tract_geom, bbox, n_addresses=100)
            
            return {
                'fips_code': fips_code,
                'tract_geometry': tract_geom,
                'road_network': road_network,
                'roads_gdf': roads_filtered,
                'svi_data': tract_svi,
                'addresses': addresses,
                'bbox': bbox
            }
            
        except Exception as e:
            self._log(f"Error loading tract data: {str(e)}")
            raise
    
    def _generate_tract_addresses(self, tract_geom, bbox: Tuple, 
                                 n_addresses: int = 100) -> gpd.GeoDataFrame:
        """Generate synthetic addresses within tract"""
        np.random.seed(42)  # For reproducibility
        
        minx, miny, maxx, maxy = bbox
        addresses = []
        
        # Generate random points within bbox, keep those in tract
        attempts = 0
        while len(addresses) < n_addresses and attempts < n_addresses * 10:
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            point = Point(x, y)
            
            if tract_geom.contains(point):
                addresses.append({
                    'address_id': len(addresses),
                    'geometry': point
                })
            
            attempts += 1
        
        if not addresses:
            # Fallback: use tract centroid
            centroid = tract_geom.centroid
            addresses = [{
                'address_id': 0,
                'geometry': centroid
            }]
        
        gdf = gpd.GeoDataFrame(addresses, crs='EPSG:4326')
        self._log(f"Generated {len(gdf)} addresses")
        
        return gdf
    
    def load_transit_stops(self) -> gpd.GeoDataFrame:
        """
        Load or create transit stop locations
        
        Returns:
        --------
        gpd.GeoDataFrame
            Transit stop locations
        """
        self._log("Loading transit stops...")
        
        # For now, create mock transit stops
        # In production, load from GTFS or other transit data source
        stops = [
            Point(-85.3096, 35.0456),  # Downtown Chattanooga
            Point(-85.2967, 35.0722),  # North Chattanooga
            Point(-85.2580, 35.0456),  # East Chattanooga
            Point(-85.3365, 35.0175),  # South Chattanooga
            Point(-85.1938, 35.0495),  # East Ridge
        ]
        
        transit_stops = gpd.GeoDataFrame(
            {'stop_id': range(len(stops))},
            geometry=stops,
            crs='EPSG:4326'
        )
        
        self._log(f"Created {len(transit_stops)} transit stops")
        
        return transit_stops
    
    def load_address_points(self, n_points: int = 1000) -> gpd.GeoDataFrame:
        """
        Generate or load address point locations
        
        Parameters:
        -----------
        n_points : int
            Number of address points to generate
            
        Returns:
        --------
        gpd.GeoDataFrame
            Address point locations
        """
        self._log(f"Generating {n_points} address points...")
        
        # Hamilton County bounding box
        bbox = [-85.5, 35.0, -85.0, 35.5]
        
        # Generate random points
        np.random.seed(42)  # For reproducibility
        lons = np.random.uniform(bbox[0], bbox[2], n_points)
        lats = np.random.uniform(bbox[1], bbox[3], n_points)
        
        points = [Point(lon, lat) for lon, lat in zip(lons, lats)]
        
        addresses = gpd.GeoDataFrame(
            {'address_id': range(n_points)},
            geometry=points,
            crs='EPSG:4326'
        )
        
        self._log(f"Generated {len(addresses)} address points")
        
        return addresses
    
    def _get_county_name(self, state_fips: str, county_fips: str) -> str:
        """Map FIPS codes to county name"""
        # This should be expanded based on your needs
        county_map = {
            ('47', '065'): 'Hamilton'
        }
        return county_map.get((state_fips, county_fips), 'Unknown')
    
    def resolve_fips_list(self, fips_config: Dict, state_fips: str, 
                         county_fips: str) -> List[str]:
        """
        Resolve FIPS configuration to list of codes
        
        Parameters:
        -----------
        fips_config : Dict
            FIPS configuration from config file
        state_fips : str
            State FIPS code
        county_fips : str
            County FIPS code
            
        Returns:
        --------
        List[str]
            Resolved FIPS codes
        """
        # Check for explicit list
        target_list = fips_config.get('batch', {}).get('target_list', [])
        if target_list:
            return target_list
        
        # Check for single FIPS
        if fips_config.get('single_fips'):
            return [fips_config['single_fips']]
        
        # Auto-selection
        auto_config = fips_config.get('batch', {}).get('auto_select', {})
        if auto_config.get('enabled', True):
            all_fips = self.get_available_fips_codes(state_fips, county_fips)
            
            mode = auto_config.get('mode', 'range')
            
            if mode == 'all':
                return all_fips
            elif mode == 'range':
                start = auto_config.get('range_start', 1) - 1
                end = auto_config.get('range_end', 5)
                return all_fips[start:end]
            elif mode == 'sample':
                size = min(auto_config.get('sample_size', 10), len(all_fips))
                return np.random.choice(all_fips, size=size, replace=False).tolist()
        
        # Default: first 5 tracts
        all_fips = self.get_available_fips_codes(state_fips, county_fips)
        return all_fips[:5]


# Convenience function for backward compatibility
def load_hamilton_county_data(data_dir: str = './data', 
                            roads_file: Optional[str] = None) -> Dict:
    """
    Load all data for Hamilton County analysis
    
    This function is used by other modules and must be maintained
    for backward compatibility.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing data files
    roads_file : str, optional
        Path to roads shapefile
        
    Returns:
    --------
    Dict
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
    
    # Merge SVI with census tracts
    data['tracts_with_svi'] = data['census_tracts'].merge(
        data['svi'],
        on='FIPS',
        how='inner'
    )
    
    print("\n" + "="*60)
    print("Data Loading Complete!")
    print("="*60)
    
    return data