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

        self._address_cache = None
        
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
            county_tracts['FIPS'] = county_tracts['GEOID'].astype(str).str.strip()
            
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
        UPDATED: Load data for a single census tract using real addresses
        
        Parameters:
        -----------
        fips_code : str
            11-digit FIPS code
        buffer_degrees : float
            Buffer around tract in degrees
        max_nodes : int, optional
            Maximum nodes in road network
        max_edges : int, optional  
            Maximum edges in road network
            
        Returns:
        --------
        Dict
            Complete tract data including real addresses
        """
        self._log(f"Loading data for tract {fips_code}")
        
        try:
            # Parse FIPS code
            state_fips = fips_code[:2]
            county_fips = fips_code[2:5]
            
            # 1. Load tract geometry and SVI
            tracts = self.load_census_tracts(state_fips, county_fips)
            tract = tracts[tracts['FIPS'] == fips_code]
            
            if len(tract) == 0:
                raise ValueError(f"Tract {fips_code} not found")
            
            tract_geom = tract.iloc[0].geometry
            
            # Load SVI data
            county_name = self._get_county_name(state_fips, county_fips)
            svi_data = self.load_svi_data(state_fips, county_name)
            tract_svi = svi_data[svi_data['FIPS'] == fips_code]
            
            if len(tract_svi) == 0:
                raise ValueError(f"No SVI data for tract {fips_code}")
            
            # 2. Create buffered bounding box
            bounds = tract_geom.bounds
            buffered_bbox = (
                bounds[0] - buffer_degrees, bounds[1] - buffer_degrees,
                bounds[2] + buffer_degrees, bounds[3] + buffer_degrees
            )
            
            # 3. Load roads within buffered area
            roads = self._load_roads_for_bbox(state_fips, county_fips, buffered_bbox)
            
            # 4. Create road network
            road_network = self.create_network_graph(roads)
            
            # 5. UPDATED: Get real addresses for this tract
            addresses = self.get_addresses_for_tract(fips_code, buffer_meters=200)
            
            # If no real addresses found, generate synthetic ones as fallback
            if len(addresses) == 0:
                self._log(f"No real addresses found for tract {fips_code}, generating synthetic ones")
                addresses = self._generate_tract_addresses(tract_geom, buffered_bbox, n_addresses=100)
                addresses['tract_fips'] = fips_code
            
            return {
                'fips_code': fips_code,
                'tract_geometry': tract_geom,
                'svi_data': tract_svi.iloc[0],
                'roads': roads,
                'road_network': road_network,
                'addresses': addresses,  # NOW CONTAINS REAL ADDRESSES
                'bbox': buffered_bbox,
                'network_stats': {
                    'nodes': road_network.number_of_nodes(),
                    'edges': road_network.number_of_edges(),
                    'real_addresses': len(addresses),
                    'address_source': 'real' if 'full_address' in addresses.columns else 'synthetic'
                }
            }
            
        except Exception as e:
            self._log(f"Error loading tract data: {str(e)}")
            raise
    
    def _generate_tract_addresses(self, tract_geom, bbox: Tuple, 
                                 n_addresses: int = 100) -> gpd.GeoDataFrame:
        """
        UPDATED: Generate synthetic addresses within tract (fallback only)
        """
        np.random.seed(123)  # Loader-specific seed (different from kriging)
        
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
                    'geometry': point,
                    'full_address': f"Synthetic Address {len(addresses)}"
                })
            
            attempts += 1
        
        if not addresses:
            # Fallback: use tract centroid
            centroid = tract_geom.centroid
            addresses = [{
                'address_id': 0,
                'geometry': centroid,
                'full_address': 'Tract Centroid Address'
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
    
    def load_address_points(self, state_fips: str = '47', county_fips: str = '065') -> gpd.GeoDataFrame:
        """
        Load real Chattanooga address point locations
        
        UPDATED TO USE REAL GEOJSON DATA
        
        Parameters:
        -----------
        state_fips : str
            State FIPS code (47 for Tennessee)
        county_fips : str  
            County FIPS code (065 for Hamilton County)
            
        Returns:
        --------
        gpd.GeoDataFrame
            Real address point locations from chattanooga.geojson
        """
        # Check cache first
        if self._address_cache is not None:
            self._log(f"Using cached address data ({len(self._address_cache)} addresses)")
            return self._address_cache
        
        self._log("Loading real Chattanooga address data...")
        
        # Try multiple file locations
        address_files = [
            os.path.join(self.data_dir, 'chattanooga.geojson'),
            os.path.join(self.data_dir, 'raw', 'chattanooga.geojson'),
            os.path.join(self.data_dir, 'addresses', 'chattanooga.geojson'),
            './chattanooga.geojson',  # Current directory
            'chattanooga.geojson'     # Direct filename
        ]
        
        addresses_gdf = None
        for address_file in address_files:
            if os.path.exists(address_file):
                addresses_gdf = self._load_chattanooga_geojson(address_file)
                if len(addresses_gdf) > 0:
                    self._log(f"Loaded {len(addresses_gdf)} real addresses from {address_file}")
                    break
        
        if addresses_gdf is None or len(addresses_gdf) == 0:
            self._log("WARNING: Real address data not found, using fallback synthetic generation")
            return self._generate_tract_constrained_addresses(state_fips, county_fips)
        
        # Filter to Hamilton County bounds if needed
        addresses_gdf = self._filter_to_hamilton_county(addresses_gdf, state_fips, county_fips)
        
        # Cache the result
        self._address_cache = addresses_gdf
        
        self._log(f"Loaded {len(addresses_gdf)} Hamilton County addresses")
        return addresses_gdf
    
    def _load_chattanooga_geojson(self, file_path: str) -> gpd.GeoDataFrame:
        """
        Load and standardize the chattanooga.geojson file
        
        Parameters:
        -----------
        file_path : str
            Path to chattanooga.geojson file
            
        Returns:
        --------
        gpd.GeoDataFrame
            Standardized address data
        """
        try:
            # Load GeoJSON
            addresses = gpd.read_file(file_path)
            
            if len(addresses) == 0:
                self._log(f"No addresses found in {file_path}")
                return gpd.GeoDataFrame(columns=['address_id', 'geometry'])
            
            # Standardize columns to match expected format
            addresses = addresses.copy()
            
            # Create standardized address_id
            if 'address_id' not in addresses.columns:
                addresses['address_id'] = range(len(addresses))
            
            # Create full address field for reference
            addresses['full_address'] = addresses.apply(self._create_full_address, axis=1)
            
            # Extract relevant fields from properties
            if 'number' in addresses.columns:
                addresses['house_number'] = addresses['number']
            if 'street' in addresses.columns:
                addresses['street_name'] = addresses['street']
            if 'city' in addresses.columns:
                addresses['city_name'] = addresses['city']
            if 'postcode' in addresses.columns:
                addresses['zipcode'] = addresses['postcode']
            
            # Ensure proper CRS (GeoJSON is typically WGS84)
            if addresses.crs is None:
                addresses.set_crs(epsg=4326, inplace=True)
            elif addresses.crs != 'EPSG:4326':
                addresses = addresses.to_crs('EPSG:4326')
            
            # Keep essential columns
            essential_columns = ['address_id', 'geometry', 'full_address']
            optional_columns = ['house_number', 'street_name', 'city_name', 'zipcode', 'hash']
            
            keep_columns = essential_columns + [col for col in optional_columns if col in addresses.columns]
            addresses = addresses[keep_columns]
            
            return addresses
            
        except Exception as e:
            self._log(f"Error loading {file_path}: {str(e)}")
            return gpd.GeoDataFrame(columns=['address_id', 'geometry'])
    
    def _create_full_address(self, row) -> str:
        """Create full address string from components"""
        parts = []
        
        if pd.notna(row.get('number', '')) and row.get('number', '') != '':
            parts.append(str(row['number']))
        
        if pd.notna(row.get('street', '')) and row.get('street', '') != '':
            parts.append(str(row['street']))
        
        if pd.notna(row.get('unit', '')) and row.get('unit', '') != '':
            parts.append(f"Unit {row['unit']}")
        
        if pd.notna(row.get('city', '')) and row.get('city', '') != '':
            parts.append(str(row['city']))
        
        if pd.notna(row.get('postcode', '')) and row.get('postcode', '') != '':
            parts.append(str(row['postcode']))
        
        return ', '.join(parts) if parts else 'Unknown Address'
    
    def _filter_to_hamilton_county(self, addresses: gpd.GeoDataFrame, 
                                  state_fips: str, county_fips: str) -> gpd.GeoDataFrame:
        """
        Filter addresses to Hamilton County boundaries
        
        Parameters:
        -----------
        addresses : gpd.GeoDataFrame
            Address data to filter
        state_fips : str
            State FIPS code
        county_fips : str
            County FIPS code
            
        Returns:
        --------
        gpd.GeoDataFrame
            Addresses within Hamilton County
        """
        try:
            # Load Hamilton County boundary
            county_tracts = self.load_census_tracts(state_fips, county_fips)
            if len(county_tracts) == 0:
                self._log("Warning: Could not load county boundary for filtering")
                return addresses
            
            # Create county boundary from tract union
            county_boundary = county_tracts.geometry.unary_union
            
            # Spatial filter
            within_county = addresses[addresses.geometry.within(county_boundary)]
            
            self._log(f"Filtered {len(addresses)} addresses to {len(within_county)} within Hamilton County")
            
            return within_county
            
        except Exception as e:
            self._log(f"Error filtering to county boundary: {str(e)}")
            return addresses
    
    def get_addresses_for_tract(self, fips_code: str, 
                              buffer_meters: float = 100) -> gpd.GeoDataFrame:
        """
        Get real addresses within a specific census tract
        
        NEW METHOD FOR TRACT-SPECIFIC ADDRESS LOADING
        
        Parameters:
        -----------
        fips_code : str
            11-digit FIPS code for census tract
        buffer_meters : float
            Buffer around tract boundary in meters
            
        Returns:
        --------
        gpd.GeoDataFrame
            Addresses within the specified tract
        """
        self._log(f"Getting addresses for tract {fips_code}")
        
        try:
            # Load all addresses
            all_addresses = self.load_address_points()
            
            if len(all_addresses) == 0:
                self._log(f"No addresses available for tract {fips_code}")
                return gpd.GeoDataFrame(columns=['address_id', 'geometry'])
            
            # Load tract geometry
            state_fips = fips_code[:2]
            county_fips = fips_code[2:5]
            tracts = self.load_census_tracts(state_fips, county_fips)
            
            tract = tracts[tracts['FIPS'] == fips_code]
            if len(tract) == 0:
                self._log(f"Tract {fips_code} not found")
                return gpd.GeoDataFrame(columns=['address_id', 'geometry'])
            
            tract_geom = tract.iloc[0].geometry
            
            # Apply buffer if specified
            if buffer_meters > 0:
                # Convert to projected CRS for accurate buffering
                tract_proj = tract.to_crs('EPSG:3857')  # Web Mercator
                buffered = tract_proj.geometry.buffer(buffer_meters)
                tract_geom = buffered.to_crs('EPSG:4326').iloc[0]
            
            # Spatial filter
            tract_addresses = all_addresses[all_addresses.geometry.within(tract_geom)].copy()
            
            # Add tract FIPS to addresses
            tract_addresses['tract_fips'] = fips_code
            
            self._log(f"Found {len(tract_addresses)} addresses in tract {fips_code}")
            
            return tract_addresses
            
        except Exception as e:
            self._log(f"Error getting addresses for tract {fips_code}: {str(e)}")
            return gpd.GeoDataFrame(columns=['address_id', 'geometry'])
    
    def _generate_tract_constrained_addresses(self, state_fips: str, county_fips: str, 
                                            density_per_sq_km: int = 500) -> gpd.GeoDataFrame:
        """
        FALLBACK: Generate addresses constrained to tract boundaries
        Only used if real address data is unavailable
        """
        self._log("Generating tract-constrained synthetic addresses as fallback...")
        
        try:
            # Load census tracts
            tracts = self.load_census_tracts(state_fips, county_fips)
            
            all_addresses = []
            address_id = 0
            
            for _, tract in tracts.iterrows():
                # Calculate number of addresses for this tract based on area
                tract_area_sq_km = tract.geometry.area * 111**2  # Rough conversion to sq km
                n_addresses = max(10, int(tract_area_sq_km * density_per_sq_km))
                
                # Generate addresses within this specific tract
                tract_addresses = self._generate_tract_addresses(
                    tract.geometry, 
                    tract.geometry.bounds, 
                    n_addresses=n_addresses
                )
                
                # Update address IDs
                tract_addresses['address_id'] = range(address_id, address_id + len(tract_addresses))
                tract_addresses['tract_fips'] = tract['FIPS']
                tract_addresses['full_address'] = f"Synthetic Address in {tract['FIPS']}"
                
                all_addresses.append(tract_addresses)
                address_id += len(tract_addresses)
            
            if all_addresses:
                combined = gpd.GeoDataFrame(pd.concat(all_addresses, ignore_index=True))
                self._log(f"Generated {len(combined)} tract-constrained synthetic addresses")
                return combined[['address_id', 'geometry', 'full_address', 'tract_fips']]
            else:
                self._log("Failed to generate any addresses")
                return gpd.GeoDataFrame(columns=['address_id', 'geometry'])
                
        except Exception as e:
            self._log(f"Error generating tract-constrained addresses: {str(e)}")
            return gpd.GeoDataFrame(columns=['address_id', 'geometry'])
    
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

# Helper function for addresses validation
def analyze_address_coverage(data_loader, state_fips='47', county_fips='065'):
    """
    Analyze coverage of real address data across census tracts
    """
    print("Analyzing address coverage across Hamilton County...")
    
    # Load tracts and addresses
    tracts = data_loader.load_census_tracts(state_fips, county_fips)
    addresses = data_loader.load_address_points(state_fips, county_fips)
    
    coverage_stats = []
    
    for _, tract in tracts.iterrows():
        fips_code = tract['FIPS']
        tract_addresses = data_loader.get_addresses_for_tract(fips_code)
        
        coverage_stats.append({
            'fips': fips_code,
            'address_count': len(tract_addresses),
            'tract_area_sq_km': tract.geometry.area * 111**2,
            'address_density': len(tract_addresses) / (tract.geometry.area * 111**2) if tract.geometry.area > 0 else 0
        })
    
    coverage_df = pd.DataFrame(coverage_stats)
    
    print(f"\nAddress Coverage Summary:")
    print(f"Total tracts: {len(coverage_df)}")
    print(f"Tracts with addresses: {sum(coverage_df['address_count'] > 0)}")
    print(f"Total addresses: {coverage_df['address_count'].sum()}")
    print(f"Mean addresses per tract: {coverage_df['address_count'].mean():.1f}")
    print(f"Median addresses per tract: {coverage_df['address_count'].median():.1f}")
    
    return coverage_df

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