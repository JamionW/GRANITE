"""
Streamlined Data Loaders for Simplified GRANITE
Focus on robust accessibility feature computation with road network graphs
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import networkx as nx
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """
    Streamlined data loader focused on accessibility feature computation
    Uses road network topology for graph creation instead of simple KNN
    """
    
    def __init__(self, data_dir: str = './data', config: dict = None):
        self.data_dir = data_dir
        self.config = config or {}
        self.verbose = config.get('processing', {}).get('verbose', False) if config else False
        
        # Simplified caching
        self._address_cache = None
        self._svi_cache = None
        self._transit_cache = None
        
        os.makedirs(data_dir, exist_ok=True)
    
    def _log(self, message: str):
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] AccessibilityLoader: {message}")

    def load_census_tracts(self, state_fips: str = '47', county_fips: str = '065') -> gpd.GeoDataFrame:
        """Load census tract geometries"""
        self._log(f"Loading census tracts for {state_fips}-{county_fips}...")
        
        local_file = os.path.join(self.data_dir, 'raw', f'tl_2020_{state_fips}_tract.shp')
        
        try:
            if os.path.exists(local_file):
                tracts = gpd.read_file(local_file)
                self._log(f"Loaded {len(tracts)} tracts from local file")
            else:
                raise FileNotFoundError(f"Census tracts not found at {local_file}")
            
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
        """Load Social Vulnerability Index data"""
        self._log(f"Loading SVI data for {county_name} County, {state_fips}...")
        
        svi_file = os.path.join(self.data_dir, 'raw', f'SVI_2020_US.csv')
        
        try:
            if os.path.exists(svi_file):
                svi_data = pd.read_csv(svi_file, dtype={'FIPS': str, 'ST': str})
                self._log(f"Loaded local SVI data ({len(svi_data)} records)")
            else:
                raise FileNotFoundError(f"SVI data not found at {svi_file}")
            
            # Filter to county
            county_svi = svi_data[
                (svi_data['ST'] == state_fips) & 
                (svi_data['COUNTY'] == county_name)
            ].copy()
            
            county_svi['FIPS'] = county_svi['FIPS'].astype(str)
            county_svi['RPL_THEMES'] = county_svi['RPL_THEMES'].replace(-999, np.nan)
            
            columns = ['FIPS', 'LOCATION', 'RPL_THEMES', 'E_TOTPOP']
            county_svi = county_svi[columns]
            
            valid_count = county_svi['RPL_THEMES'].notna().sum()
            self._log(f"Loaded {len(county_svi)} tracts ({valid_count} with valid SVI)")
            
            return county_svi
            
        except Exception as e:
            self._log(f"Error loading SVI data: {str(e)}")
            raise

    def calculate_multimodal_travel_times_batch(self, origins: gpd.GeoDataFrame, 
                                            destinations: gpd.GeoDataFrame,
                                            time_periods: list = ['morning']) -> pd.DataFrame:
        """
        Network-based travel time calculation for accessibility analysis
        """
        from geopy.distance import geodesic
        
        results = []
        
        for orig_idx, origin in origins.iterrows():
            orig_id = origin.get('address_id', orig_idx)
            orig_coord = (origin.geometry.y, origin.geometry.x)
            
            for dest_idx, destination in destinations.iterrows():
                dest_id = destination.get('dest_id', dest_idx)
                dest_coord = (destination.geometry.y, destination.geometry.x)
                
                # Use geodesic distance with network factor
                straight_distance = geodesic(orig_coord, dest_coord).kilometers
                network_distance = straight_distance * 1.3  # Network routing factor
                
                # Calculate multi-modal times
                walk_time = network_distance / 5.0 * 60  # 5 km/h walking
                drive_time = network_distance / 25.0 * 60  # 25 km/h urban driving
                
                # Transit time estimation
                if 1 <= network_distance <= 15:
                    transit_base = network_distance / 12.0 * 60  # 12 km/h transit
                    transit_wait = 8  # Average wait time
                    transit_time = transit_base + transit_wait
                else:
                    transit_time = walk_time * 1.2
                
                # Best mode
                times = {'walk': walk_time, 'drive': drive_time, 'transit': transit_time}
                best_mode = min(times.keys(), key=lambda k: times[k])
                combined_time = times[best_mode]
                
                results.append({
                    'origin_id': orig_id,
                    'destination_id': dest_id,
                    'destination_type': destination.get('dest_type', 'unknown'),
                    'walk_time': walk_time,
                    'drive_time': drive_time,
                    'transit_time': transit_time,
                    'combined_time': combined_time,
                    'best_mode': best_mode
                })
        
        return pd.DataFrame(results)

    def create_spatial_accessibility_graph(self, addresses, accessibility_features, 
                                        state_fips='47', county_fips='065'):
        """
        Create graph based on actual road network connectivity
        Falls back to geographic proximity when road network is unavailable
        """
        import torch
        from torch_geometric.data import Data
        
        n_addresses = len(addresses)
        self._log(f"Creating road network graph for {n_addresses} addresses...")
        
        # Load road network
        roads = self.load_road_network(state_fips=state_fips, county_fips=county_fips)
        
        if len(roads) > 0:
            # Primary approach: Road network-based connectivity
            edge_index, edge_weight = self._create_road_network_graph(addresses, roads)
        else:
            self._log("No road network available, using geographic connectivity")
            edge_index, edge_weight = self._create_geographic_fallback_graph(addresses)
        
        # Node features are the accessibility features
        node_features = torch.FloatTensor(accessibility_features)
        
        # Create graph data
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_weight
        )
        
        self._log(f"Created graph: {n_addresses} nodes, {edge_index.shape[1]//2} undirected edges")
        
        return graph_data

    def _create_road_network_graph(self, addresses, roads):
        """
        Create graph based on actual road network connectivity
        Integrates road network topology with geographic fallback
        """
        import torch
        from sklearn.neighbors import NearestNeighbors
        
        n_addresses = len(addresses)
        
        # Method 1: Road network-based connectivity
        road_graph, address_to_road_mapping = self._create_road_connectivity(addresses, roads)
        network_edges = self._extract_network_edges(road_graph, address_to_road_mapping)
        
        # Method 2: Geographic proximity (for disconnected areas)
        geographic_edges = self._create_geographic_edges(addresses)
        
        # Combine edges
        all_edges = network_edges + geographic_edges
        
        # Remove duplicates and create tensors
        edge_set = set()
        edge_list = []
        edge_weights = []
        edge_types = []
        
        for edge in all_edges:
            edge_key = tuple(sorted([edge['from'], edge['to']]))
            if edge_key not in edge_set and edge['from'] != edge['to']:
                edge_set.add(edge_key)
                edge_list.extend([[edge['from'], edge['to']], [edge['to'], edge['from']]])  # Undirected
                edge_weights.extend([edge['weight'], edge['weight']])
                edge_types.extend([edge['type'], edge['type']])
        
        # Convert to tensors
        if edge_list:
            edge_index = torch.LongTensor(edge_list).t().contiguous()
            edge_weight = torch.FloatTensor(edge_weights)
        else:
            # Fallback: connect each address to nearest neighbor
            edge_index, edge_weight = self._create_fallback_edges(n_addresses)
            edge_types = ['fallback'] * edge_index.shape[1]
        
        # Log edge statistics
        network_count = sum(1 for t in edge_types if t == 'network')
        geographic_count = sum(1 for t in edge_types if t == 'geographic')
        
        self._log(f"  Network edges: {network_count//2}")
        self._log(f"  Geographic edges: {geographic_count//2}")
        
        return edge_index, edge_weight

    def _create_road_connectivity(self, addresses, roads):
        """Create NetworkX graph from road segments and map addresses to roads"""
        import networkx as nx
        from sklearn.neighbors import BallTree
        import numpy as np
        
        self._log("Building road network graph...")
        
        # Create NetworkX graph from roads
        road_graph = nx.Graph()
        
        # Add road segments as edges
        for idx, road in roads.iterrows():
            if road.geometry.geom_type == 'LineString':
                coords = list(road.geometry.coords)
                
                # Add edges between consecutive points along road
                for i in range(len(coords) - 1):
                    u, v = coords[i], coords[i + 1]
                    
                    # Calculate edge length (approximate)
                    length = ((u[0] - v[0])**2 + (u[1] - v[1])**2)**0.5 * 111000  # Convert to meters
                    
                    # Add edge with road properties
                    road_graph.add_edge(
                        u, v, 
                        length=length,
                        road_id=idx,
                        road_type=road.get('RTTYP', 'unknown')
                    )
        
        # Map each address to nearest road node
        if road_graph.number_of_nodes() == 0:
            self._log("No road nodes found")
            return road_graph, {}
        
        # Get all road nodes
        road_nodes = np.array(list(road_graph.nodes()))
        
        # Build spatial index for road nodes
        tree = BallTree(np.radians(road_nodes), metric='haversine')
        
        # Map addresses to nearest road nodes
        address_to_road_mapping = {}
        address_coords = np.array([[addr.geometry.x, addr.geometry.y] for _, addr in addresses.iterrows()])
        
        # Find nearest road node for each address (within 500m)
        distances, indices = tree.query(np.radians(address_coords), k=1)
        distances = distances.flatten() * 6371000  # Convert to meters
        
        for i, (addr_idx, addr) in enumerate(addresses.iterrows()):
            if distances[i] < 500:  # Within 500m of a road
                nearest_road_node = tuple(road_nodes[indices[i][0]])
                address_to_road_mapping[i] = nearest_road_node
        
        self._log(f"Mapped {len(address_to_road_mapping)}/{len(addresses)} addresses to road network")
        
        return road_graph, address_to_road_mapping

    def _extract_network_edges(self, road_graph, address_to_road_mapping):
        """
        Extract address-to-address edges via road network connectivity
        Uses distance limits and local connectivity for efficiency
        """
        import networkx as nx
        from sklearn.neighbors import NearestNeighbors
        import numpy as np
        
        network_edges = []
        
        # Get road-connected addresses
        road_connected_addresses = list(address_to_road_mapping.keys())
        
        if len(road_connected_addresses) < 2:
            return network_edges
        
        self._log(f"Computing network connectivity for {len(road_connected_addresses)} road-connected addresses...")
        
        # Use geographic proximity to limit road network queries
        address_coords = []
        for addr_idx in road_connected_addresses:
            road_node = address_to_road_mapping[addr_idx]
            address_coords.append([road_node[0], road_node[1]])  # lon, lat
        
        address_coords = np.array(address_coords)
        
        # Find geographic neighbors
        max_neighbors = min(20, len(road_connected_addresses) - 1)
        nbrs = NearestNeighbors(n_neighbors=max_neighbors, metric='euclidean').fit(address_coords)
        distances, indices = nbrs.kneighbors(address_coords)
        
        # Only compute road paths for geographic neighbors
        for i, addr1_idx in enumerate(road_connected_addresses):
            road_node1 = address_to_road_mapping[addr1_idx]
            
            # Check only geographic neighbors
            for j_idx in range(1, min(10, len(indices[i]))):
                j = indices[i][j_idx]
                addr2_idx = road_connected_addresses[j]
                road_node2 = address_to_road_mapping[addr2_idx]
                
                # Skip if same node
                if road_node1 == road_node2:
                    continue
                
                # Geographic distance check
                geo_distance = distances[i][j_idx] * 111000  # rough conversion to meters
                
                # Only compute road path for nearby addresses (within 1km)
                if geo_distance < 1000:
                    try:
                        # Compute shortest path length without cutoff parameter
                        path_length = nx.shortest_path_length(
                            road_graph, road_node1, road_node2, 
                            weight='length'
                        )
                        
                        # Apply manual cutoff check
                        if path_length > 1500:  # Don't use paths longer than 1.5km
                            continue
                        
                        # Weight inversely proportional to road distance
                        weight = 1.0 / (1.0 + path_length / 500.0)
                        
                        network_edges.append({
                            'from': addr1_idx,
                            'to': addr2_idx,
                            'weight': weight,
                            'type': 'network',
                            'road_distance': path_length
                        })
                        
                    except (nx.NetworkXNoPath, nx.NetworkXError):
                        # No road connection
                        continue
            
            # Progress update every 100 addresses
            if (i + 1) % 100 == 0:
                self._log(f"  Processed {i + 1}/{len(road_connected_addresses)} addresses...")
        
        self._log(f"Created {len(network_edges)} network edges")
        return network_edges

    def _create_geographic_edges(self, addresses, max_neighbors=6):
        """Create edges based on geographic proximity as fallback"""
        from sklearn.neighbors import NearestNeighbors
        import numpy as np
        
        coords = np.array([[addr.geometry.x, addr.geometry.y] for _, addr in addresses.iterrows()])
        n_addresses = len(addresses)
        
        # Find geographic neighbors
        k_neighbors = min(max_neighbors, n_addresses - 1)
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        geographic_edges = []
        
        for i in range(n_addresses):
            for j_idx in range(1, len(indices[i])):  # Skip self (index 0)
                j = indices[i][j_idx]
                distance_deg = distances[i][j_idx]
                distance_m = distance_deg * 111000  # Approximate conversion to meters
                
                # Only connect nearby addresses (within 1km)
                if distance_m < 1000:
                    weight = np.exp(-distance_m / 300.0)  # Exponential decay
                    
                    geographic_edges.append({
                        'from': i,
                        'to': j,
                        'weight': weight,
                        'type': 'geographic',
                        'distance': distance_m
                    })
        
        return geographic_edges

    def _create_fallback_edges(self, n_addresses):
        """Create minimal connectivity as absolute fallback"""
        import torch
        
        # Create simple ring connectivity as absolute fallback
        edge_list = []
        edge_weights = []
        
        for i in range(n_addresses):
            next_i = (i + 1) % n_addresses
            edge_list.extend([[i, next_i], [next_i, i]])
            edge_weights.extend([1.0, 1.0])
        
        edge_index = torch.LongTensor(edge_list).t().contiguous()
        edge_weight = torch.FloatTensor(edge_weights)
        
        return edge_index, edge_weight

    def _create_geographic_fallback_graph(self, addresses):
        """
        Fallback to geographic connectivity when no road network is available
        """
        import torch
        from sklearn.neighbors import NearestNeighbors
        import numpy as np
        
        n_addresses = len(addresses)
        coords = np.array([[addr.geometry.x, addr.geometry.y] for _, addr in addresses.iterrows()])
        
        # Adaptive number of neighbors based on density
        if n_addresses < 100:
            k_neighbors = min(8, n_addresses - 1)
        elif n_addresses < 500:
            k_neighbors = min(12, n_addresses - 1)
        else:
            k_neighbors = min(16, n_addresses - 1)
        
        # Find k-nearest geographic neighbors
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='euclidean').fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        edge_list = []
        edge_weights = []
        
        for i in range(n_addresses):
            for j_idx in range(1, len(indices[i])):  # Skip self (index 0)
                j = indices[i][j_idx]
                distance_deg = distances[i][j_idx]
                distance_m = distance_deg * 111000  # Convert to meters
                
                # Distance-based weight with reasonable cutoff
                if distance_m < 1000:  # Within 1km
                    weight = np.exp(-distance_m / 300.0)  # Exponential decay
                    
                    # Add both directions for undirected graph
                    edge_list.extend([[i, j], [j, i]])
                    edge_weights.extend([weight, weight])
        
        # Ensure we have connectivity
        if not edge_list:
            # Fallback: connect each node to nearest neighbor
            for i in range(n_addresses):
                if len(indices[i]) > 1:
                    j = indices[i][1]  # Nearest neighbor
                    edge_list.extend([[i, j], [j, i]])
                    edge_weights.extend([1.0, 1.0])
        
        # Convert to tensors
        edge_index = torch.LongTensor(edge_list).t().contiguous()
        edge_weight = torch.FloatTensor(edge_weights)
        
        return edge_index, edge_weight

    def load_road_network(self, roads_file: Optional[str] = None, 
                        state_fips: str = '47', county_fips: str = '065') -> gpd.GeoDataFrame:
        """
        Load road network data
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
                    self._log("Returning empty road network - will use geographic fallback")
                    return gpd.GeoDataFrame(geometry=[])
            
            # Ensure CRS
            if roads.crs is None:
                roads.set_crs(epsg=4326, inplace=True)
            
            return roads
            
        except Exception as e:
            self._log(f"Error loading roads: {str(e)}")
            return gpd.GeoDataFrame(geometry=[])

    # Rest of the methods remain the same as in your new file...
    def load_address_points(self, state_fips: str = '47', county_fips: str = '065') -> gpd.GeoDataFrame:
        """Load real Chattanooga address points (simplified)"""
        
        if self._address_cache is not None:
            self._log(f"Using cached address data ({len(self._address_cache)} addresses)")
            return self._address_cache
        
        self._log("Loading Chattanooga address data...")
        
        # Try multiple locations for address file
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
                        # Standardize columns
                        addresses_gdf = addresses_gdf.copy()
                        
                        if 'address_id' not in addresses_gdf.columns:
                            addresses_gdf['address_id'] = range(len(addresses_gdf))
                        
                        # Ensure proper CRS
                        if addresses_gdf.crs is None:
                            addresses_gdf.set_crs(epsg=4326, inplace=True)
                        elif addresses_gdf.crs != 'EPSG:4326':
                            addresses_gdf = addresses_gdf.to_crs('EPSG:4326')
                        
                        # Keep essential columns
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
        
        # Cache the result
        self._address_cache = addresses_gdf
        
        self._log(f"Final address count: {len(addresses_gdf)}")
        return addresses_gdf

    def get_addresses_for_tract(self, fips_code: str) -> gpd.GeoDataFrame:
        """Get addresses within a specific census tract"""
        
        try:
            # Load all addresses
            all_addresses = self.load_address_points()
            
            if len(all_addresses) == 0:
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
            
            # Spatial filter
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
            # Load county boundary from tracts
            tracts = self.load_census_tracts(state_fips, county_fips)
            county_boundary = tracts.geometry.unary_union
            
            bounds = county_boundary.bounds
            addresses = []
            address_id = 0
            
            # Generate points within county bounds
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

    def _get_county_name(self, state_fips: str, county_fips: str) -> str:
        """Map FIPS codes to county name"""
        county_map = {
            ('47', '065'): 'Hamilton'
        }
        return county_map.get((state_fips, county_fips), 'Unknown')

    # Destination creation methods (core accessibility destinations)
    def create_employment_destinations(self) -> gpd.GeoDataFrame:
        """Create employment destinations for accessibility analysis"""
        
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
        
        self._log(f"Created {len(employment_gdf)} employment destinations")
        return employment_gdf

    def create_healthcare_destinations(self) -> gpd.GeoDataFrame:
        """Create healthcare destinations for accessibility analysis"""
        
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
        
        self._log(f"Created {len(healthcare_gdf)} healthcare destinations")
        return healthcare_gdf

    def create_grocery_destinations(self) -> gpd.GeoDataFrame:
        """Create grocery destinations for accessibility analysis"""
        
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
        
        self._log(f"Created {len(grocery_gdf)} grocery destinations")
        return grocery_gdf

    def compute_accessibility_features(self, addresses: gpd.GeoDataFrame) -> np.ndarray:
        """
        Compute comprehensive accessibility features for all addresses
        """
        self._log(f"Computing accessibility features for {len(addresses)} addresses...")
        
        # Create destinations
        destinations = {
            'employment': self.create_employment_destinations(),
            'healthcare': self.create_healthcare_destinations(),
            'grocery': self.create_grocery_destinations()
        }
        
        # Compute features for each destination type
        all_features = []
        
        for dest_type, dest_gdf in destinations.items():
            self._log(f"  Computing {dest_type} accessibility...")
            
            # Calculate travel times
            travel_times = self._calculate_simple_travel_times(addresses, dest_gdf)
            
            # Extract accessibility features
            features = self._extract_accessibility_features_from_times(
                addresses, dest_gdf, travel_times, dest_type
            )
            
            all_features.append(features)
        
        # Combine all destination features
        accessibility_matrix = np.column_stack(all_features)
        
        # Add derived/summary features
        derived_features = self._compute_derived_features(accessibility_matrix)
        
        final_features = np.column_stack([accessibility_matrix, derived_features])
        
        self._log(f"Final accessibility features: {final_features.shape}")
        return final_features

    def _calculate_simple_travel_times(self, addresses: gpd.GeoDataFrame, 
                                     destinations: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Simplified travel time calculation using distance approximation
        """
        results = []
        
        for addr_idx, address in addresses.iterrows():
            addr_id = address.get('address_id', addr_idx)
            addr_point = address.geometry
            
            for dest_idx, destination in destinations.iterrows():
                dest_id = destination.get('dest_id', dest_idx)
                dest_point = destination.geometry
                
                # Calculate great circle distance
                distance_deg = addr_point.distance(dest_point)
                distance_km = distance_deg * 111  # Rough conversion to km
                
                # Estimate travel times for different modes
                walk_time = distance_km / 5.0 * 60  # 5 km/h walking speed → minutes
                drive_time = distance_km / 30.0 * 60  # 30 km/h average urban speed → minutes
                
                # Simple transit time estimation
                if 2 <= distance_km <= 15:
                    transit_base_time = distance_km / 15.0 * 60  # 15 km/h average transit speed
                    transit_wait_time = 10  # Average wait time
                    transit_time = transit_base_time + transit_wait_time
                else:
                    transit_time = walk_time * 1.5  # Transit not competitive
                
                # Best mode and combined time
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
            
            # Get travel times for this address to all destinations of this type
            addr_times = travel_times[travel_times['origin_id'] == addr_id]
            
            if len(addr_times) > 0:
                combined_times = addr_times['combined_time'].values
                
                # Time-based features
                min_time = float(np.min(combined_times))
                mean_time = float(np.mean(combined_times))
                percentile_90 = float(np.percentile(combined_times, 90))
                
                # Count-based features (destinations within time thresholds)
                count_30min = int(np.sum(combined_times <= 30))
                count_60min = int(np.sum(combined_times <= 60))
                count_90min = int(np.sum(combined_times <= 90))
                
                # Transit accessibility
                transit_trips = addr_times['best_mode'] == 'transit'
                transit_share = float(transit_trips.mean())
                
                # Overall accessibility score (gravity-style)
                accessibility_score = float(np.sum(1.0 / np.maximum(combined_times, 1.0)))
                
            else:
                # No destinations accessible
                min_time = mean_time = percentile_90 = 120.0
                count_30min = count_60min = count_90min = 0
                transit_share = accessibility_score = 0.0
            
            features.append([
                min_time, mean_time, percentile_90,
                count_30min, count_60min, count_90min,
                transit_share, accessibility_score
            ])
        
        return np.array(features, dtype=np.float64)

    def _compute_derived_features(self, base_features: np.ndarray) -> np.ndarray:
        """Compute derived accessibility features from base metrics"""
        
        n_addresses = base_features.shape[0]
        
        # Assuming base_features has shape [n_addresses, 24] (3 dest types × 8 features each)
        if base_features.shape[1] < 24:
            # Return minimal derived features
            return np.zeros((n_addresses, 4), dtype=np.float64)
        
        derived = []
        
        for i in range(n_addresses):
            # Extract accessibility scores for each destination type (8th feature)
            emp_score = base_features[i, 7]    # Employment accessibility score
            health_score = base_features[i, 15]  # Healthcare accessibility score
            grocery_score = base_features[i, 23] # Grocery accessibility score
            
            # Overall accessibility
            total_accessibility = emp_score + health_score + grocery_score
            
            # Accessibility balance (entropy measure)
            if total_accessibility > 0:
                scores = np.array([emp_score, health_score, grocery_score]) / total_accessibility
                scores = np.maximum(scores, 1e-8)  # Avoid log(0)
                balance = -np.sum(scores * np.log(scores))  # Shannon entropy
            else:
                balance = 0.0
            
            # Transit dependence
            emp_transit = base_features[i, 6]
            health_transit = base_features[i, 14]
            grocery_transit = base_features[i, 22]
            avg_transit_dependence = (emp_transit + health_transit + grocery_transit) / 3
            
            # Time efficiency (min vs mean time performance)
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