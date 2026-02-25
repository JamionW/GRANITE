"""
GRANITE Data Loaders (Spatial Version)

Provides data loading for census, SVI, addresses, and graph construction.
Removes all accessibility/routing dependencies.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
from torch_geometric.data import Data
import torch
from typing import List, Dict, Optional
from datetime import datetime
import networkx as nx
from sklearn.neighbors import BallTree
import warnings
warnings.filterwarnings('ignore')

from .block_group_loader import BlockGroupLoader


class DataLoader:
    """
    Streamlined data loader for spatial disaggregation.
    No routing or destination dependencies.
    """
    
    def __init__(self, data_dir: str = '/workspaces/GRANITE/data', config: dict = None):
        self.data_dir = data_dir
        self.config = config or {}
        self.verbose = config.get('processing', {}).get('verbose', False) if config else False
        
        # Caches
        self._address_cache = None
        self._tract_cache = None
        self._svi_cache = None
        
        # Block group loader for validation
        self.block_group_loader = BlockGroupLoader(data_dir, verbose=self.verbose)
        
        os.makedirs(data_dir, exist_ok=True)
    
    def _log(self, message: str):
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] DataLoader: {message}")
    
    # =========================================================================
    # CENSUS AND SVI DATA
    # =========================================================================
    
    def load_census_tracts(self, state_fips: str = '47', county_fips: str = '065') -> gpd.GeoDataFrame:
        """Load census tract geometries."""
        cache_key = f"{state_fips}_{county_fips}"
        if self._tract_cache is not None and hasattr(self, '_tract_cache_key') and self._tract_cache_key == cache_key:
            return self._tract_cache
        
        self._log(f"Loading census tracts for {state_fips}-{county_fips}")
        
        local_file = os.path.join(self.data_dir, 'raw', f'tl_2020_{state_fips}_tract.shp')
        
        if not os.path.exists(local_file):
            raise FileNotFoundError(f"Census tracts file not found: {local_file}")
        
        tracts = gpd.read_file(local_file)
        county_tracts = tracts[tracts['COUNTYFP'] == county_fips].copy()
        county_tracts['FIPS'] = county_tracts['GEOID'].astype(str).str.strip()
        
        if county_tracts.crs is None:
            county_tracts.set_crs(epsg=4326, inplace=True)
        
        self._log(f"Loaded {len(county_tracts)} tracts")
        self._tract_cache = county_tracts
        self._tract_cache_key = cache_key
        
        return county_tracts
    
    def load_svi_data(self, state_fips: str = '47', county_name: str = 'Hamilton') -> pd.DataFrame:
        """Load Social Vulnerability Index data."""
        if self._svi_cache is not None:
            return self._svi_cache
        
        self._log(f"Loading SVI data for {county_name} County")
        
        svi_file = os.path.join(self.data_dir, 'raw', 'SVI_2020_US.csv')
        
        if not os.path.exists(svi_file):
            raise FileNotFoundError(f"SVI file not found: {svi_file}")
        
        svi_data = pd.read_csv(svi_file, dtype={'FIPS': str, 'ST': str})
        
        county_svi = svi_data[
            (svi_data['ST'] == state_fips) & 
            (svi_data['COUNTY'] == county_name)
        ].copy()
        
        county_svi['FIPS'] = county_svi['FIPS'].astype(str)
        county_svi['RPL_THEMES'] = county_svi['RPL_THEMES'].replace(-999, np.nan)
        
        # Keep essential columns
        keep_cols = ['FIPS', 'LOCATION', 'RPL_THEMES', 'E_TOTPOP', 'E_HU',
                     'EP_NOVEH', 'EP_POV150', 'EP_UNEMP', 'EP_NOHSDP']
        available = [c for c in keep_cols if c in county_svi.columns]
        county_svi = county_svi[available].copy()
        
        self._log(f"Loaded SVI for {len(county_svi)} tracts")
        self._svi_cache = county_svi
        
        return county_svi
    
    def _get_county_name(self, state_fips: str, county_fips: str) -> str:
        """Get county name from FIPS codes."""
        county_map = {'47065': 'Hamilton'}
        return county_map.get(f"{state_fips}{county_fips}", 'Hamilton')
    
    # =========================================================================
    # ADDRESS LOADING
    # =========================================================================
    
    def load_address_points(self, state_fips: str = '47', county_fips: str = '065') -> gpd.GeoDataFrame:
        """Load address points from GeoJSON."""
        if self._address_cache is not None:
            return self._address_cache
        
        self._log("Loading address data...")
        
        address_files = [
            os.path.join(self.data_dir, 'raw', 'chattanooga.geojson'),
            os.path.join(self.data_dir, 'chattanooga.geojson'),
        ]
        
        for addr_file in address_files:
            if os.path.exists(addr_file):
                addresses = gpd.read_file(addr_file)
                
                if 'address_id' not in addresses.columns:
                    addresses['address_id'] = range(len(addresses))
                
                if addresses.crs is None:
                    addresses.set_crs(epsg=4326, inplace=True)
                elif addresses.crs != 'EPSG:4326':
                    addresses = addresses.to_crs('EPSG:4326')
                
                self._log(f"Loaded {len(addresses)} addresses")
                self._address_cache = addresses
                return addresses
        
        raise FileNotFoundError("No address file found")
    
    def get_addresses_for_tract(self, fips_code: str) -> gpd.GeoDataFrame:
        """Get addresses within a specific census tract."""
        all_addresses = self.load_address_points()
        
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
    
    def get_neighboring_tracts(self, target_fips: str, n_neighbors: int = 4) -> List[str]:
        """Get neighboring tracts by geographic proximity."""
        if n_neighbors == 0:
            return [target_fips]
        
        state_fips = target_fips[:2]
        county_fips = target_fips[2:5]
        all_tracts = self.load_census_tracts(state_fips, county_fips)
        
        target_tract = all_tracts[all_tracts['FIPS'] == target_fips]
        if len(target_tract) == 0:
            return [target_fips]
        
        target_geom = target_tract.geometry.iloc[0]
        all_tracts['distance'] = all_tracts.geometry.distance(target_geom)
        neighbors = all_tracts[all_tracts['FIPS'] != target_fips].nsmallest(n_neighbors, 'distance')
        
        return [target_fips] + neighbors['FIPS'].tolist()
    
    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================
    
    def create_spatial_graph(self,
                            addresses: gpd.GeoDataFrame,
                            features: np.ndarray,
                            k_neighbors: int = 8) -> Data:
        """
        Create PyG graph from addresses using k-NN connectivity.
        
        Args:
            addresses: GeoDataFrame with address points
            features: Spatial features [n_addresses, n_features]
            k_neighbors: Number of neighbors for graph edges
        
        Returns:
            PyG Data object with x (features) and edge_index
        """
        n_addresses = len(addresses)
        self._log(f"Building spatial graph: {n_addresses} nodes, k={k_neighbors}")
        
        # Extract coordinates
        coords = np.column_stack([
            addresses.geometry.x.values,
            addresses.geometry.y.values
        ])
        
        # Build k-NN graph
        tree = cKDTree(coords)
        
        # Query k+1 neighbors (includes self)
        _, indices = tree.query(coords, k=min(k_neighbors + 1, n_addresses))
        
        # Build edge list (exclude self-loops)
        edge_list = []
        for i in range(n_addresses):
            for j in indices[i, 1:]:  # Skip first (self)
                if j < n_addresses:  # Safety check
                    edge_list.append([i, j])
                    edge_list.append([j, i])  # Undirected
        
        # Remove duplicates
        edge_set = set(tuple(e) for e in edge_list)
        edge_list = list(edge_set)
        
        edge_index = torch.LongTensor(edge_list).t().contiguous()
        x = torch.FloatTensor(features)
        
        self._log(f"Graph: {n_addresses} nodes, {edge_index.shape[1]} edges")
        
        return Data(x=x, edge_index=edge_index)

    def load_road_network(self, state_fips: str = '47', county_fips: str = '065') -> gpd.GeoDataFrame:
        """Load road network shapefile."""
        self._log(f"Loading road network for {state_fips}-{county_fips}")
        
        # Try different filename patterns
        road_files = [
            os.path.join(self.data_dir, 'raw', f'tl_2023_{state_fips}{county_fips}_roads.shp'),
            os.path.join(self.data_dir, 'raw', f'tl_2022_{state_fips}{county_fips}_roads.shp'),
            os.path.join(self.data_dir, 'raw', f'tl_2020_{state_fips}{county_fips}_roads.shp'),
        ]
        
        for road_file in road_files:
            if os.path.exists(road_file):
                roads = gpd.read_file(road_file)
                if roads.crs is None:
                    roads.set_crs(epsg=4326, inplace=True)
                elif roads.crs != 'EPSG:4326':
                    roads = roads.to_crs('EPSG:4326')
                self._log(f"Loaded {len(roads)} road segments")
                return roads
        
        self._log("Road network file not found, will use geographic fallback")
        return gpd.GeoDataFrame(geometry=[])
    
    def _build_road_graph(self, roads: gpd.GeoDataFrame) -> nx.Graph:
        """Build NetworkX graph from road LineStrings."""
        road_graph = nx.Graph()
        
        for idx, road in roads.iterrows():
            if road.geometry is None:
                continue
            if road.geometry.geom_type == 'LineString':
                coords = list(road.geometry.coords)
                for i in range(len(coords) - 1):
                    u, v = coords[i], coords[i + 1]
                    # Length in meters (approximate)
                    length = ((u[0] - v[0])**2 + (u[1] - v[1])**2)**0.5 * 111000
                    road_graph.add_edge(u, v, length=length)
            elif road.geometry.geom_type == 'MultiLineString':
                for line in road.geometry.geoms:
                    coords = list(line.coords)
                    for i in range(len(coords) - 1):
                        u, v = coords[i], coords[i + 1]
                        length = ((u[0] - v[0])**2 + (u[1] - v[1])**2)**0.5 * 111000
                        road_graph.add_edge(u, v, length=length)
        
        self._log(f"Road graph: {road_graph.number_of_nodes()} nodes, {road_graph.number_of_edges()} edges")
        return road_graph


    def _map_addresses_to_roads(self, 
                                addresses: gpd.GeoDataFrame, 
                                road_graph: nx.Graph,
                                max_snap_distance: float = 500.0) -> Dict[int, tuple]:
        """
        Map each address to its nearest road network node.
        
        Args:
            addresses: GeoDataFrame with address points
            road_graph: NetworkX graph of road network
            max_snap_distance: Maximum distance (meters) to snap to road
        
        Returns:
            Dict mapping address index -> road node (x, y) tuple
        """
        if road_graph.number_of_nodes() == 0:
            return {}
        
        road_nodes = np.array(list(road_graph.nodes()))
        tree = BallTree(np.radians(road_nodes), metric='haversine')
        
        address_coords = np.array([
            [addr.geometry.x, addr.geometry.y] 
            for _, addr in addresses.iterrows()
        ])
        
        # Query nearest road node for each address
        distances, indices = tree.query(np.radians(address_coords), k=1)
        distances_m = distances.flatten() * 6371000  # Earth radius in meters
        
        mapping = {}
        for i in range(len(addresses)):
            if distances_m[i] <= max_snap_distance:
                mapping[i] = tuple(road_nodes[indices[i][0]])
        
        self._log(f"Mapped {len(mapping)}/{len(addresses)} addresses to road network")
        return mapping

    def compute_simple_accessibility_features(self,
                                            addresses: gpd.GeoDataFrame,
                                            state_fips: str = '47',
                                            county_fips: str = '065') -> np.ndarray:
        """
        Compute accessibility features using simple distance-based travel times.
        No OSRM required.
        
        Returns: np.ndarray of shape [n_addresses, 18] (6 features × 3 dest types)
        """
        self._log("Computing accessibility features (distance-based)...")
        
        # Load destinations
        destinations = {
            'employment': self.create_employment_destinations(use_real_data=True),
            'healthcare': self.create_healthcare_destinations(use_real_data=True),
            'grocery': self.create_grocery_destinations(use_real_data=True)
        }
        
        all_features = []
        
        for dest_type, dest_gdf in destinations.items():
            self._log(f"  {dest_type}: {len(dest_gdf)} destinations")
            features = self._compute_dest_type_features(addresses, dest_gdf)
            all_features.append(features)
        
        accessibility_matrix = np.column_stack(all_features)
        self._log(f"Accessibility features: {accessibility_matrix.shape}")
        
        return accessibility_matrix


    def _compute_dest_type_features(self,
                                    addresses: gpd.GeoDataFrame,
                                    destinations: gpd.GeoDataFrame) -> np.ndarray:
        """
        Compute 6 accessibility features for one destination type.
        Uses simple Euclidean distance → travel time conversion.
        """
        n_addr = len(addresses)
        
        # Extract coordinates
        addr_coords = np.column_stack([
            addresses.geometry.x.values,
            addresses.geometry.y.values
        ])
        dest_coords = np.column_stack([
            destinations.geometry.x.values,
            destinations.geometry.y.values
        ])
        
        # Compute all pairwise distances (in degrees)
        # Shape: [n_addresses, n_destinations]
        dx = addr_coords[:, 0:1] - dest_coords[:, 0].reshape(1, -1)
        dy = addr_coords[:, 1:2] - dest_coords[:, 1].reshape(1, -1)
        distances_deg = np.sqrt(dx**2 + dy**2)
        
        # Convert to km (approximate)
        distances_km = distances_deg * 111.0
        
        # Convert to drive time (minutes) assuming 30 km/h average
        drive_times = (distances_km / 30.0) * 60.0
        
        # Compute features for each address
        features = np.zeros((n_addr, 6))
        
        for i in range(n_addr):
            times = drive_times[i, :]
            
            # Time-based features
            features[i, 0] = np.min(times)                    # min_time
            features[i, 1] = np.mean(times)                   # mean_time
            features[i, 2] = np.median(times)                 # median_time
            
            # Count-based features
            features[i, 3] = np.sum(times <= 5)               # count_5min
            features[i, 4] = np.sum(times <= 10)              # count_10min
            features[i, 5] = np.sum(times <= 15)              # count_15min
        
        return features

    def _compute_network_edges(self,
                            road_graph: nx.Graph,
                            address_mapping: Dict[int, tuple],
                            max_path_length: float = 1500.0,
                            max_neighbors: int = 10) -> List[Dict]:
        """
        Compute edges between addresses via shortest road network paths.
        
        Args:
            road_graph: NetworkX road graph
            address_mapping: Dict mapping address idx -> road node
            max_path_length: Maximum path length (meters) for edges
            max_neighbors: Maximum neighbors to check per address
        
        Returns:
            List of edge dicts with 'from', 'to', 'weight', 'road_distance'
        """
        network_edges = []
        mapped_addresses = list(address_mapping.keys())
        
        if len(mapped_addresses) < 2:
            return network_edges
        
        # Get coordinates of mapped addresses for k-NN pre-filtering
        mapped_coords = np.array([address_mapping[i] for i in mapped_addresses])
        
        from sklearn.neighbors import NearestNeighbors
        k = min(max_neighbors + 1, len(mapped_addresses))
        nbrs = NearestNeighbors(n_neighbors=k).fit(mapped_coords)
        _, indices = nbrs.kneighbors(mapped_coords)
        
        self._log(f"Computing network paths for {len(mapped_addresses)} addresses...")
        
        for i, addr1_idx in enumerate(mapped_addresses):
            node1 = address_mapping[addr1_idx]
            
            for j_idx in range(1, len(indices[i])):
                addr2_idx = mapped_addresses[indices[i][j_idx]]
                node2 = address_mapping[addr2_idx]
                
                if node1 == node2:
                    continue
                
                try:
                    path_length = nx.shortest_path_length(
                        road_graph, node1, node2, weight='length'
                    )
                    
                    if path_length <= max_path_length:
                        # Weight decays with road distance
                        weight = 1.0 / (1.0 + path_length / 500.0)
                        
                        network_edges.append({
                            'from': addr1_idx,
                            'to': addr2_idx,
                            'weight': weight,
                            'road_distance': path_length
                        })
                except (nx.NetworkXNoPath, nx.NetworkXError):
                    continue
            
            if (i + 1) % 200 == 0:
                self._log(f"  Processed {i + 1}/{len(mapped_addresses)} addresses")
        
        self._log(f"Created {len(network_edges)} network edges")
        return network_edges

    def create_road_network_graph(self,
                                addresses: gpd.GeoDataFrame,
                                features: np.ndarray,
                                state_fips: str = '47',
                                county_fips: str = '065',
                                k_neighbors: int = 8,
                                max_path_length: float = 1500.0) -> Data:
        """
        Create PyG graph using road network connectivity.
        
        Addresses connected if reachable via road network within max_path_length.
        Falls back to geographic k-NN for addresses not mappable to roads.
        
        Args:
            addresses: GeoDataFrame with address points
            features: Node features [n_addresses, n_features]
            state_fips: State FIPS code
            county_fips: County FIPS code  
            k_neighbors: k for geographic fallback edges
            max_path_length: Max road distance (meters) for network edges
        
        Returns:
            PyG Data object
        """
        n_addresses = len(addresses)
        self._log(f"Building road network graph: {n_addresses} addresses")
        
        # Load and build road graph
        roads = self.load_road_network(state_fips, county_fips)
        
        if len(roads) == 0:
            self._log("No roads available, falling back to k-NN")
            return self.create_spatial_graph(addresses, features, k_neighbors)
        
        road_graph = self._build_road_graph(roads)
        
        if road_graph.number_of_nodes() == 0:
            self._log("Empty road graph, falling back to k-NN")
            return self.create_spatial_graph(addresses, features, k_neighbors)
        
        # Map addresses to road nodes
        address_mapping = self._map_addresses_to_roads(addresses, road_graph)
        
        # Compute network-based edges
        network_edges = self._compute_network_edges(
            road_graph, address_mapping, max_path_length
        )
        
        # Compute geographic fallback edges for unmapped addresses
        geographic_edges = self._create_geographic_fallback_edges(
            addresses, address_mapping, k_neighbors
        )
        
        # Combine edges
        all_edges = network_edges + geographic_edges
        
        # Build edge index
        edge_set = set()
        edge_list = []
        
        for edge in all_edges:
            key = tuple(sorted([edge['from'], edge['to']]))
            if key not in edge_set and edge['from'] != edge['to']:
                edge_set.add(key)
                edge_list.append([edge['from'], edge['to']])
                edge_list.append([edge['to'], edge['from']])
        
        if len(edge_list) == 0:
            self._log("No edges created, falling back to k-NN")
            return self.create_spatial_graph(addresses, features, k_neighbors)
        
        edge_index = torch.LongTensor(edge_list).t().contiguous()
        x = torch.FloatTensor(features)
        
        network_count = len(network_edges)
        geo_count = len(geographic_edges)
        self._log(f"Graph: {n_addresses} nodes, {edge_index.shape[1]} edges "
                f"(network: {network_count}, geographic: {geo_count})")
        
        return Data(x=x, edge_index=edge_index)


    def _create_geographic_fallback_edges(self,
                                        addresses: gpd.GeoDataFrame,
                                        address_mapping: Dict[int, tuple],
                                        k_neighbors: int) -> List[Dict]:
        """
        Create k-NN edges for addresses not mapped to road network.
        Also ensures minimum connectivity for all addresses.
        """
        n_addresses = len(addresses)
        mapped_set = set(address_mapping.keys())
        
        coords = np.column_stack([
            addresses.geometry.x.values,
            addresses.geometry.y.values
        ])
        
        tree = cKDTree(coords)
        _, indices = tree.query(coords, k=min(k_neighbors + 1, n_addresses))
        
        geographic_edges = []
        
        for i in range(n_addresses):
            # Add geographic edges for unmapped addresses
            # or to ensure minimum connectivity
            if i not in mapped_set:
                for j in indices[i, 1:]:
                    if j < n_addresses:
                        dist_deg = np.sqrt((coords[i, 0] - coords[j, 0])**2 + 
                                        (coords[i, 1] - coords[j, 1])**2)
                        dist_m = dist_deg * 111000
                        
                        if dist_m < 1000:
                            weight = np.exp(-dist_m / 300.0)
                            geographic_edges.append({
                                'from': i,
                                'to': j,
                                'weight': weight,
                                'road_distance': None
                            })
        
        return geographic_edges

    def create_employment_destinations(self, use_real_data: bool = True) -> gpd.GeoDataFrame:
        """Load or create employment destinations."""
        if use_real_data:
            try:
                return self._load_lehd_employment()
            except Exception as e:
                self._log(f"LEHD load failed: {e}, using synthetic")
        
        # Synthetic fallback - major Chattanooga employers
        employers = [
            {'name': 'Downtown Chattanooga', 'lat': 35.0456, 'lon': -85.3097},
            {'name': 'Volkswagen', 'lat': 35.0614, 'lon': -85.1580},
            {'name': 'Erlanger Hospital', 'lat': 35.0539, 'lon': -85.3083},
            {'name': 'Hamilton Place', 'lat': 35.0407, 'lon': -85.2111},
            {'name': 'UTC', 'lat': 35.0456, 'lon': -85.3011},
            {'name': 'Enterprise South', 'lat': 35.0156, 'lon': -85.1580},
            {'name': 'Northgate Mall', 'lat': 35.0989, 'lon': -85.2847},
            {'name': 'Eastgate', 'lat': 35.0407, 'lon': -85.2300},
        ]
        
        gdf = gpd.GeoDataFrame(
            employers,
            geometry=[Point(e['lon'], e['lat']) for e in employers],
            crs='EPSG:4326'
        )
        gdf['dest_id'] = range(len(gdf))
        gdf['dest_type'] = 'employment'
        return gdf


    def create_healthcare_destinations(self, use_real_data: bool = True) -> gpd.GeoDataFrame:
        """Load or create healthcare destinations."""
        if use_real_data:
            try:
                return self._load_healthcare_facilities()
            except Exception as e:
                self._log(f"Healthcare load failed: {e}, using synthetic")
        
        hospitals = [
            {'name': 'Erlanger Baroness', 'lat': 35.0539, 'lon': -85.3083},
            {'name': 'CHI Memorial', 'lat': 35.0627, 'lon': -85.2985},
            {'name': 'Parkridge Medical', 'lat': 35.0456, 'lon': -85.2597},
            {'name': 'Parkridge East', 'lat': 35.0407, 'lon': -85.2111},
            {'name': 'Erlanger East', 'lat': 35.0407, 'lon': -85.2111},
            {'name': 'Erlanger North', 'lat': 35.1156, 'lon': -85.2700},
        ]
        
        gdf = gpd.GeoDataFrame(
            hospitals,
            geometry=[Point(h['lon'], h['lat']) for h in hospitals],
            crs='EPSG:4326'
        )
        gdf['dest_id'] = range(len(gdf))
        gdf['dest_type'] = 'healthcare'
        return gdf


    def create_grocery_destinations(self, use_real_data: bool = True) -> gpd.GeoDataFrame:
        """Load or create grocery destinations."""
        if use_real_data:
            try:
                return self._load_grocery_stores()
            except Exception as e:
                self._log(f"Grocery load failed: {e}, using synthetic")
        
        groceries = [
            {'name': 'Publix - Gunbarrel', 'lat': 35.0350, 'lon': -85.1950},
            {'name': 'Publix - Hixson', 'lat': 35.1100, 'lon': -85.2400},
            {'name': 'Walmart - Hamilton Place', 'lat': 35.0380, 'lon': -85.2080},
            {'name': 'Walmart - Hixson', 'lat': 35.1050, 'lon': -85.2350},
            {'name': 'Kroger - Brainerd', 'lat': 35.0200, 'lon': -85.2600},
            {'name': 'Kroger - Signal Mtn', 'lat': 35.1200, 'lon': -85.3100},
            {'name': 'Food City - Rossville', 'lat': 34.9830, 'lon': -85.2860},
            {'name': 'Aldi - East Brainerd', 'lat': 35.0250, 'lon': -85.2000},
            {'name': 'Bi-Lo - Red Bank', 'lat': 35.0900, 'lon': -85.2950},
        ]
        
        gdf = gpd.GeoDataFrame(
            groceries,
            geometry=[Point(g['lon'], g['lat']) for g in groceries],
            crs='EPSG:4326'
        )
        gdf['dest_id'] = range(len(gdf))
        gdf['dest_type'] = 'grocery'
        return gdf

    # =========================================================================
    # VALIDATION HELPERS
    # =========================================================================
    
    def get_block_groups_for_tract(self, tract_fips: str) -> gpd.GeoDataFrame:
        """Get block groups within a tract for validation."""
        state_fips = tract_fips[:2]
        county_fips = tract_fips[2:5]
        
        bg_data = self.block_group_loader.get_block_groups_with_demographics(
            state_fips, county_fips
        )
        
        # Filter to tract
        tract_bg = bg_data[bg_data['tract_fips'] == tract_fips].copy()
        return tract_bg