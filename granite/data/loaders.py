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
    
    # ==========================================
    # ORIGINAL DATA LOADING METHODS
    # ==========================================
    
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
    
    def load_transit_stops(self):
        """
        Load or create transit stop locations
        
        Returns:
        --------
        gpd.GeoDataFrame
            Transit stop locations
        """
        self._log("Creating mock transit stops...")
        
        # Create mock transit stops for Hamilton County
        # In practice, load from GTFS or other transit data
        stops = [
            Point(-85.3096, 35.0456),  # Downtown Chattanooga
            Point(-85.2967, 35.0722),  # North Chattanooga
            Point(-85.2580, 35.0456),  # East Chattanooga
        ]
        
        transit_stops = gpd.GeoDataFrame(
            {'stop_id': range(len(stops))},
            geometry=stops,
            crs='EPSG:4326'
        )
        
        self._log(f"  ✓ Created {len(transit_stops)} mock transit stops")
        
        return transit_stops
    
    def load_address_points(self, n_points=1000):
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
        self._log(f"Generating {n_points} synthetic address points...")
        
        # Bounding box for Hamilton County, TN
        bbox = [-85.5, 35.0, -85.0, 35.5]
        
        # Generate random points
        lons = np.random.uniform(bbox[0], bbox[2], n_points)
        lats = np.random.uniform(bbox[1], bbox[3], n_points)
        
        points = [Point(lon, lat) for lon, lat in zip(lons, lats)]
        
        addresses = gpd.GeoDataFrame(
            {'address_id': range(n_points)},
            geometry=points,
            crs='EPSG:4326'
        )
        
        self._log(f"  ✓ Generated {len(addresses)} address points")
        
        return addresses
    
    def create_network_graph(self, roads, directed=False):
        """
        Create NetworkX graph from road geometries
        
        Parameters:
        -----------
        roads : gpd.GeoDataFrame
            Road network geometries
        directed : bool
            Whether to create directed graph
            
        Returns:
        --------
        nx.Graph or nx.DiGraph
            Network graph
        """
        self._log("Creating network graph from road geometries...")
        
        if len(roads) == 0:
            self._log("  ✗ No road data provided")
            return nx.Graph()
        
        # Create graph
        G = nx.DiGraph() if directed else nx.Graph()
        
        # Add edges from road geometries
        for idx, road in roads.iterrows():
            if road.geometry.geom_type == 'LineString':
                coords = list(road.geometry.coords)
                
                # Add edges between consecutive points
                for i in range(len(coords) - 1):
                    u, v = coords[i], coords[i + 1]
                    
                    # Calculate edge length
                    length = Point(u).distance(Point(v))
                    
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
    
    # ==========================================
    # NEW FIPS-BASED PROCESSING METHODS
    # ==========================================
    
    def get_available_fips_codes(self, state_fips='47', county_fips='065'):
        """
        Get list of available FIPS codes for the county
        
        Parameters:
        -----------
        state_fips : str
            State FIPS code
        county_fips : str  
            County FIPS code
            
        Returns:
        --------
        List[str]
            List of available FIPS codes
        """
        tracts = self.load_census_tracts(state_fips, county_fips)
        return sorted(tracts['FIPS'].tolist())
    
    def _create_preserved_network(self, roads, max_nodes=None, max_edges=None, preserve_network=True):
        """
        Create network graph with optional granularity preservation for disaggregation
        
        Parameters:
        -----------
        roads : gpd.GeoDataFrame
            Road network geometries
        max_nodes : int, optional
            Maximum nodes (None = unlimited)
        max_edges : int, optional
            Maximum edges (None = unlimited)
        preserve_network : bool
            If True, preserves full network granularity
            
        Returns:
        --------
        nx.Graph
            Network graph
        """
        if len(roads) == 0:
            return nx.Graph()
        
        # Build initial network
        G = self.create_network_graph(roads)
        
        original_nodes = G.number_of_nodes()
        original_edges = G.number_of_edges()
        
        # CRITICAL FIX: Preserve network granularity for disaggregation
        if preserve_network:
            if max_nodes is not None and original_nodes > max_nodes:
                self._log(f"    ⚠️  Network has {original_nodes} nodes (>{max_nodes} limit)")
                self._log(f"    ✓ PRESERVING full network for disaggregation granularity")
                self._log(f"    (Use preserve_network=False to enable simplification)")
            
            # Return full network without simplification
            return G
        
        else:
            # Legacy behavior with simplification (NOT recommended for disaggregation)
            if (max_nodes is not None and original_nodes > max_nodes) or \
            (max_edges is not None and original_edges > max_edges):
                
                import warnings
                warnings.warn(
                    f"Simplifying network from {original_nodes} nodes to {max_nodes} nodes. "
                    "This reduces granularity needed for spatial disaggregation. "
                    "Consider using preserve_network=True or removing node limits.",
                    UserWarning
                )
                
                self._log(f"    ⚠️  Network too large ({original_nodes} nodes, {original_edges} edges), simplifying...")
                G = self._simplify_network(G, max_nodes or original_nodes, max_edges or original_edges)
                self._log(f"    Simplified to {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                self._log(f"    ⚠️  GRANULARITY REDUCED - May impact disaggregation quality!")
            
            return G

    def load_single_tract_data(self, fips_code, buffer_degrees=0.01, 
                          max_nodes=None, max_edges=None, 
                          preserve_network=True):  # NEW PARAMETER
        """
        Load all data for a single census tract with optional memory management
        
        Parameters:
        -----------
        fips_code : str
            Census tract FIPS code (11 digits)
        buffer_degrees : float
            Buffer around tract for road network (degrees)
        max_nodes : int, optional
            Maximum nodes in road network (None = unlimited, recommended for disaggregation)
        max_edges : int, optional
            Maximum edges in road network (None = unlimited, recommended for disaggregation)
        preserve_network : bool
            If True, preserves full network granularity (recommended for spatial disaggregation)
            
        Returns:
        --------
        dict
            Dictionary containing tract data
        """
        self._log(f"Loading data for tract {fips_code}")
        
        # Extract state and county from FIPS
        state_fips = fips_code[:2]
        county_fips = fips_code[2:5]
        
        # 1. Load tract geometry
        tract_geometry = self._load_single_tract_geometry(fips_code)
        if tract_geometry is None:
            raise ValueError(f"Tract {fips_code} not found")
        
        # 2. Load SVI data for this tract
        tract_svi = self._load_single_tract_svi(fips_code)
        
        # 3. Create buffered bbox for road network
        bounds = tract_geometry.bounds  # Returns tuple: (minx, miny, maxx, maxy)
        buffered_bbox = [
            bounds[0] - buffer_degrees,  # minx
            bounds[1] - buffer_degrees,  # miny
            bounds[2] + buffer_degrees,  # maxx
            bounds[3] + buffer_degrees   # maxy
        ]
        
        # 4. Load roads within buffered area
        roads = self._load_roads_for_bbox(
            state_fips, county_fips, buffered_bbox
        )
        
        # 5. Create network with granularity preservation
        road_network = self._create_preserved_network(
            roads, max_nodes, max_edges, preserve_network
        )
        
        # 6. Generate address points within tract
        addresses = self._generate_tract_addresses(
            tract_geometry, n_points=min(200, max(50, road_network.number_of_nodes() // 50))
        )
        
        # 7. Create transit stops
        transit_stops = self._create_tract_transit_stops(tract_geometry)
        
        return {
            'fips_code': fips_code,
            'tract_geometry': tract_geometry,
            'svi_data': tract_svi,
            'roads': roads,
            'road_network': road_network,
            'addresses': addresses,
            'transit_stops': transit_stops,
            'network_stats': {
                'nodes': road_network.number_of_nodes(),
                'edges': road_network.number_of_edges(),
                'buffer_km': buffer_degrees * 111,  # Rough conversion
                'preserved': preserve_network,
            }
        }

    def load_batch_fips_data(self, fips_list, preserve_network=True, **kwargs):
        """
        Load data for multiple FIPS codes with granularity preservation
        
        Parameters:
        -----------
        fips_list : List[str]
            List of FIPS codes to load
        preserve_network : bool
            If True, preserves full network granularity (recommended for disaggregation)
        **kwargs
            Arguments passed to load_single_tract_data
            
        Returns:
        --------
        dict
            Dictionary with FIPS codes as keys, tract data as values
        """
        self._log(f"Loading data for {len(fips_list)} census tracts")
        
        # Set preserve_network parameter
        kwargs['preserve_network'] = preserve_network
        
        batch_data = {}
        successful = 0
        failed = []
        
        for i, fips_code in enumerate(fips_list, 1):
            self._log(f"  [{i}/{len(fips_list)}] Loading tract {fips_code}")
            
            try:
                tract_data = self.load_single_tract_data(fips_code, **kwargs)
                batch_data[fips_code] = tract_data
                successful += 1
                
            except Exception as e:
                self._log(f"    ✗ Failed to load {fips_code}: {str(e)}")
                failed.append((fips_code, str(e)))
                
                # Store error info
                batch_data[fips_code] = {
                    'status': 'error',
                    'error': str(e),
                    'fips_code': fips_code
                }
        
        self._log(f"Batch loading complete: {successful} successful, {len(failed)} failed")
        
        return batch_data

    def _load_single_tract_geometry(self, fips_code):
        """Load geometry for a single tract"""
        state_fips = fips_code[:2]
        county_fips = fips_code[2:5]
        
        tracts = self.load_census_tracts(state_fips, county_fips)
        tract = tracts[tracts['FIPS'] == fips_code]
        
        if len(tract) == 0:
            return None
        
        return tract.iloc[0].geometry

    def _load_single_tract_svi(self, fips_code):
        """Load SVI data for a single tract"""
        state_fips = fips_code[:2]
        county_name = self._get_county_name_from_fips(state_fips, fips_code[2:5])
        
        # Load county SVI data
        county_svi = self.load_svi_data(state_fips, county_name)
        
        # Filter to specific tract
        tract_svi = county_svi[county_svi['FIPS'] == fips_code]
        
        return tract_svi

    def _load_roads_for_bbox(self, state_fips, county_fips, bbox):
        """Load roads within bounding box"""
        # Load full county roads
        roads = self.load_road_network(None, state_fips, county_fips)
        
        if len(roads) == 0:
            return roads
        
        # Create bbox polygon for spatial filtering
        minx, miny, maxx, maxy = bbox
        from shapely.geometry import box
        bbox_poly = box(minx, miny, maxx, maxy)
        
        # Spatial filter
        roads_clipped = roads[roads.geometry.intersects(bbox_poly)].copy()
        
        return roads_clipped

    def _create_limited_network(self, roads, max_nodes, max_edges):
        """Create network graph with size limits"""
        if len(roads) == 0:
            return nx.Graph()
        
        # Build initial network
        G = self.create_network_graph(roads)
        
        # Apply size limits
        original_nodes = G.number_of_nodes()
        original_edges = G.number_of_edges()
        
        if original_nodes > max_nodes or original_edges > max_edges:
            self._log(f"    Network too large ({original_nodes} nodes, {original_edges} edges), simplifying...")
            G = self._simplify_network(G, max_nodes, max_edges)
            self._log(f"    Simplified to {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G

    def _simplify_network(self, G, max_nodes, max_edges):
        """Simplify network by removing low-importance nodes and edges"""
        import random
        
        # Strategy 1: Remove degree-2 nodes (pass-through intersections)
        nodes_to_remove = []
        for node in list(G.nodes()):
            if G.degree(node) == 2:
                neighbors = list(G.neighbors(node))
                if len(neighbors) == 2:
                    # Connect neighbors directly
                    n1, n2 = neighbors
                    # Combine edge weights
                    w1 = G[node][n1].get('weight', 1)
                    w2 = G[node][n2].get('weight', 1)
                    G.add_edge(n1, n2, weight=w1 + w2)
                    nodes_to_remove.append(node)
        
        G.remove_nodes_from(nodes_to_remove)
        
        # Strategy 2: If still too large, randomly sample nodes
        if G.number_of_nodes() > max_nodes:
            # Sample nodes to keep (prefer high-degree nodes)
            node_degrees = dict(G.degree())
            sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
            
            # Keep top max_nodes nodes
            nodes_to_keep = [node for node, degree in sorted_nodes[:max_nodes]]
            G = G.subgraph(nodes_to_keep).copy()
        
        # Strategy 3: If too many edges, remove lowest-weight edges
        if G.number_of_edges() > max_edges:
            # Get edges with weights
            edge_weights = []
            for u, v, data in G.edges(data=True):
                weight = data.get('weight', 1)
                edge_weights.append((weight, u, v))
            
            # Sort by weight (keep highest weight edges)
            edge_weights.sort(reverse=True)
            
            # Keep only top max_edges edges
            edges_to_keep = edge_weights[:max_edges]
            
            # Create new graph with selected edges
            G_simplified = nx.Graph()
            G_simplified.add_nodes_from(G.nodes(data=True))
            for weight, u, v in edges_to_keep:
                G_simplified.add_edge(u, v, weight=weight)
            
            G = G_simplified
        
        return G

    def _generate_tract_addresses(self, tract_geometry, n_points=200):
        """Generate random address points within tract"""
        import random
        
        bounds = tract_geometry.bounds  # Returns tuple: (minx, miny, maxx, maxy)
        points = []
        attempts = 0
        max_attempts = n_points * 10
        
        while len(points) < n_points and attempts < max_attempts:
            x = random.uniform(bounds[0], bounds[2])  # minx to maxx
            y = random.uniform(bounds[1], bounds[3])  # miny to maxy
            point = Point(x, y)
            
            if tract_geometry.contains(point):
                points.append(point)
            attempts += 1
        
        return gpd.GeoDataFrame(
            {'address_id': range(len(points))},
            geometry=points,
            crs='EPSG:4326'
        )

    def _create_tract_transit_stops(self, tract_geometry, n_stops=3):
        """Create mock transit stops for tract"""
        centroid = tract_geometry.centroid
        
        stops = []
        for i in range(n_stops):
            # Random offset around centroid
            offset_x = np.random.uniform(-0.005, 0.005)
            offset_y = np.random.uniform(-0.005, 0.005)
            stop_point = Point(centroid.x + offset_x, centroid.y + offset_y)
            
            # Ensure stop is within tract
            if tract_geometry.contains(stop_point):
                stops.append(stop_point)
            else:
                # Fallback to centroid
                stops.append(centroid)
        
        return gpd.GeoDataFrame(
            {'stop_id': range(len(stops))},
            geometry=stops,
            crs='EPSG:4326'
        )

    def _get_county_name_from_fips(self, state_fips, county_fips):
        """Get county name from FIPS codes"""
        # Simple mapping - extend as needed
        county_map = {
            ('47', '065'): 'Hamilton',
            # Add more counties as needed
        }
        
        return county_map.get((state_fips, county_fips), 'Hamilton')  # Default to Hamilton

    def resolve_fips_list(self, fips_config, state_fips='47', county_fips='065'):
        """
        Resolve FIPS configuration to actual list of FIPS codes
        
        Parameters:
        -----------
        fips_config : dict
            FIPS configuration from config file
        state_fips : str
            State FIPS code
        county_fips : str
            County FIPS code
            
        Returns:
        --------
        List[str]
            Resolved list of FIPS codes
        """
        # Handle single FIPS
        if fips_config.get('single_fips'):
            return [fips_config['single_fips']]
        
        # Handle explicit list
        target_list = fips_config.get('batch', {}).get('target_list', [])
        if target_list:
            return target_list
        
        # Handle auto-selection
        auto_config = fips_config.get('batch', {}).get('auto_select', {})
        if auto_config.get('enabled', True):
            all_fips = self.get_available_fips_codes(state_fips, county_fips)
            
            mode = auto_config.get('mode', 'range')
            
            if mode == 'all':
                return all_fips
            elif mode == 'range':
                start = auto_config.get('range_start', 1) - 1  # 0-indexed
                end = auto_config.get('range_end', 5)
                return all_fips[start:end]
            elif mode == 'sample':
                sample_size = min(auto_config.get('sample_size', 10), len(all_fips))
                return np.random.choice(all_fips, size=sample_size, replace=False).tolist()
        
        # Default fallback
        all_fips = self.get_available_fips_codes(state_fips, county_fips)
        return all_fips[:5]  # First 5 tracts


# Convenience functions for direct imports
def load_hamilton_county_data(data_dir='./data', roads_file=None, preserve_network=True):
    """
    Load all data for Hamilton County analysis with granularity preservation
    
    Parameters:
    -----------
    data_dir : str
        Directory containing data files
    roads_file : str, optional
        Path to roads shapefile (e.g., 'tl_2023_47065_roads.shp')
    preserve_network : bool
        If True, preserves full network granularity (recommended for disaggregation)
        
    Returns:
    --------
    dict
        Dictionary containing all loaded datasets
    """
    loader = DataLoader(data_dir=data_dir)
    
    print("\n" + "="*60)
    print("GRANITE: Loading Hamilton County Data")
    if preserve_network:
        print("Network Granularity: PRESERVED for disaggregation")
    else:
        print("Network Granularity: SIMPLIFIED (not recommended)")
    print("="*60 + "\n")
    
    # Load all datasets
    data = {
        'svi': loader.load_svi_data(),
        'census_tracts': loader.load_census_tracts(),
        'roads': loader.load_road_network(roads_file=roads_file),
        'transit_stops': loader.load_transit_stops(),
        'addresses': loader.load_address_points()
    }
    
    # Create network graph with granularity preservation
    if preserve_network:
        data['road_network'] = loader.create_network_graph(data['roads'])
    else:
        # Use the old limited approach
        data['road_network'] = loader._create_preserved_network(
            data['roads'], max_nodes=10000, max_edges=20000, preserve_network=False
        )
    
    print("\n" + "="*60)
    print("Data Loading Complete!")
    print("="*60)
    
    return data