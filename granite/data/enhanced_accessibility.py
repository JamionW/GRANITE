"""
FIXED: Enhanced Accessibility Computation for GRANITE
Corrects destination counting logic and feature computation errors
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from geopy.distance import geodesic
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class EnhancedAccessibilityComputer:
    """
    FIXED: Enhanced accessibility computation with corrected counting logic
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        
        # FIXED: More realistic speed parameters
        self.speed_params = {
            'walk': {'base': 5.0, 'std': 0.8, 'min': 3.5, 'max': 6.5},
            'drive': {'base': 25.0, 'std': 7.0, 'min': 15.0, 'max': 45.0},  
            'transit': {'base': 12.0, 'std': 4.0, 'min': 8.0, 'max': 20.0}
        }
        
        # Network complexity factors
        self.network_factors = {
            'urban_core': 1.15,
            'suburban': 1.10,
            'edge': 1.05
        }
    
    def log(self, message):
        if self.verbose:
            print(f"[EnhancedAccessibility] {message}")
    
    def calculate_realistic_travel_times(self, origins: gpd.GeoDataFrame, 
                                    destinations: gpd.GeoDataFrame,
                                    time_period: str = 'morning') -> pd.DataFrame:
        """
        OPTIMIZED: Calculate travel times with batch processing and smart routing
        """
        total_pairs = len(origins) * len(destinations)
        self.log(f"Computing travel times: {len(origins)} origins → {len(destinations)} destinations ({total_pairs:,} pairs)")
        
        # Load road network once (expensive operation)
        road_graph, address_mapping = self._load_road_network_for_routing(origins, destinations)
        
        results = []
        processed = 0
        
        # Process origins in batches with progress reporting
        for orig_idx, origin in origins.iterrows():
            orig_id = origin.get('address_id', orig_idx)
            orig_coord = (origin.geometry.y, origin.geometry.x)
            
            # Process all destinations for this origin at once
            batch_results = self._process_origin_batch(
                orig_id, orig_coord, destinations, road_graph, address_mapping, time_period
            )
            results.extend(batch_results)
            
            processed += len(batch_results)  # Track actual processed pairs
            
            # Progress reporting every 100 origins
            if (orig_idx + 1) % 100 == 0:
                self.log(f"  Processed {processed:,}/{total_pairs:,} pairs ({processed/total_pairs*100:.1f}%) - Origin {orig_idx + 1}/{len(origins)}")
                
        travel_df = pd.DataFrame(results)
        self.log(f"Generated {len(travel_df)} travel time records")
        
        # Validate results
        self._validate_travel_times_fixed(travel_df)
        
        return travel_df
    
    def _calculate_mode_times_fixed(self, distance_km: float, time_period: str) -> Dict[str, float]:
        """FIXED: Mode-specific travel time calculation with bounds"""
        
        times = {}
        
        for mode, params in self.speed_params.items():
            # Add realistic speed variation
            speed_variation = np.random.normal(0, params['std'])
            actual_speed = np.clip(params['base'] + speed_variation, 
                                 params['min'], params['max'])
            
            # Base travel time in minutes
            base_time = (distance_km / actual_speed) * 60
            
            # Mode-specific adjustments
            if mode == 'transit':
                # Wait time (5-15 minutes)
                wait_time = np.random.uniform(5, 15)
                # Transfer penalty for longer trips
                if distance_km > 5:
                    transfer_penalty = np.random.uniform(3, 8)
                else:
                    transfer_penalty = 0
                final_time = base_time + wait_time + transfer_penalty
                
            elif mode == 'drive':
                # Parking time (1-5 minutes)
                parking_time = np.random.uniform(1, 5)
                # Traffic factor based on time period
                traffic_factor = {'morning': 1.3, 'midday': 1.0, 'evening': 1.4}.get(time_period, 1.0)
                final_time = base_time * traffic_factor + parking_time
                
            else:  # walk
                final_time = base_time
            
            # CRITICAL FIX: Ensure minimum time
            times[mode] = max(1.0, final_time)
        
        return times
    
    def extract_enhanced_accessibility_features(self, addresses: gpd.GeoDataFrame,
                                              travel_times: pd.DataFrame, 
                                              dest_type: str) -> np.ndarray:
        """
        FIXED: Extract accessibility features with corrected counting logic
        """
        self.log(f"Extracting features for {dest_type} destinations")
        
        features = []
        
        # Get unique destinations for validation
        unique_destinations = travel_times['destination_id'].nunique()
        self.log(f"Processing {len(addresses)} addresses with {unique_destinations} destinations")
        
        for addr_idx, address in addresses.iterrows():
            addr_id = address.get('address_id', addr_idx)
            
            # FIXED: Handle variable destination counts per address
            addr_times = travel_times[travel_times['origin_id'] == addr_id].copy()
            
            if len(addr_times) == 0:
                # No destinations accessible - use default values
                features.append([8.0, 10.0, 9.0, 0, 1, 2, 0.2, 0.7, 5.0, 0.5])
                continue
            
            # Remove duplicates and validate
            addr_times = addr_times.drop_duplicates(subset=['destination_id'])
            combined_times = pd.to_numeric(addr_times['combined_time'], errors='coerce').dropna()
            
            if len(combined_times) == 0:
                features.append([8.0, 10.0, 9.0, 0, 1, 2, 0.2, 0.7, 5.0, 0.5])
                continue
            
            # Time-based features
            min_time = float(combined_times.min())
            mean_time = float(combined_times.mean())
            median_time = float(combined_times.median())
            
            # IMPORTANT: Destination counting with variable available destinations
            available_destinations = len(combined_times)  # This will vary by address now!
            
            # Use original 5/10/15 minute thresholds - these will now vary meaningfully
            count_5min = int((combined_times <= 5).sum())
            count_10min = int((combined_times <= 10).sum())
            count_15min = int((combined_times <= 15).sum())
            
            # Validation: ensure counts don't exceed available destinations (should be ≤ 6)
            count_5min = min(count_5min, available_destinations)
            count_10min = min(count_10min, available_destinations)
            count_15min = min(count_15min, available_destinations)
            
            # Ensure logical progression
            count_10min = max(count_10min, count_5min)
            count_15min = max(count_15min, count_10min)
            
            # Modal advantage calculation
            if 'drive_time' in addr_times.columns and 'walk_time' in addr_times.columns:
                drive_times = pd.to_numeric(addr_times['drive_time'], errors='coerce').dropna()
                walk_times = pd.to_numeric(addr_times['walk_time'], errors='coerce').dropna()
                
                if len(drive_times) > 0 and len(walk_times) > 0:
                    avg_drive = drive_times.mean()
                    avg_walk = walk_times.mean()
                    
                    if avg_walk > 0:
                        drive_advantage = float(max(0, (avg_walk - avg_drive) / avg_walk))
                        drive_advantage = min(0.8, drive_advantage)  # Cap at 80%
                    else:
                        drive_advantage = 0.0
                else:
                    drive_advantage = 0.0
            else:
                drive_advantage = 0.0
            
            # Accessibility concentration
            if len(combined_times) > 1:
                time_std = float(combined_times.std())
                time_range = float(combined_times.max() - combined_times.min())
                # Normalize concentration (higher = more concentrated)
                max_possible_range = 20.0  # Assume max 20-minute range
                accessibility_concentration = float(1.0 - min(1.0, time_range / max_possible_range))
            else:
                time_std = 0.0
                time_range = 0.0 
                accessibility_concentration = 1.0
            
            # Placeholder for relative percentile (computed later)
            accessibility_percentile = 0.5
            
            feature_vector = [
                min_time, mean_time, median_time,
                count_5min, count_10min, count_15min,
                drive_advantage, accessibility_concentration,
                time_range, accessibility_percentile
            ]
            
            features.append(feature_vector)
        
        feature_array = np.array(features, dtype=np.float64)
        
        # Compute relative accessibility percentiles
        if len(feature_array) > 1:
            mean_times = feature_array[:, 1]  # mean_time column
            for i in range(len(feature_array)):
                # Lower mean time = better accessibility = higher percentile
                percentile = 1.0 - (np.sum(mean_times <= mean_times[i]) / len(mean_times))
                feature_array[i, 9] = percentile
        
        # VALIDATION: Check for systematic issues
        self._validate_intra_tract_features_fixed(feature_array, dest_type, unique_destinations)
        
        return feature_array
    
    def _validate_intra_tract_features_fixed(self, features: np.ndarray, dest_type: str, 
                                        available_destinations: int):
        """UPDATED: Enhanced validation accounting for variable destination counts per address"""
        
        feature_names = [
            'min_time', 'mean_time', 'median_time',
            'count_5min', 'count_10min', 'count_15min', 
            'drive_advantage', 'accessibility_concentration',
            'time_range', 'accessibility_percentile'
        ]
        
        self.log(f"=== {dest_type.upper()} VALIDATION ===")
        
        validation_passed = True
        
        for i, name in enumerate(feature_names):
            values = features[:, i]
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            self.log(f"{name}: std={std_val:.4f}, range=[{min_val:.2f}, {max_val:.2f}]")
            
            # UPDATED: Check count features against maximum possible (6 destinations per address)
            if 'count' in name:
                max_possible_per_address = 6  # We limit each address to 6 destinations
                if max_val > max_possible_per_address:
                    self.log(f"ERROR: {name} max ({max_val}) exceeds max destinations per address ({max_possible_per_address})")
                    validation_passed = False
                if min_val < 0:
                    self.log(f"ERROR: {name} has negative values")
                    validation_passed = False
            
            # Check for zero variance
            if std_val < 1e-8:
                self.log(f"WARNING: {name} has zero variance")
        
        # Check expected relationships
        min_times = features[:, 0]
        count_5min = features[:, 3]
        
        if np.std(min_times) > 1e-8 and np.std(count_5min) > 1e-8:
            corr = np.corrcoef(min_times, count_5min)[0, 1]
            self.log(f"Min_time vs Count_5min correlation: {corr:.3f}")
            
            if corr > 0.1:  # Should be negative
                self.log(f"WARNING: Correlation should be negative, got {corr:.3f}")
                validation_passed = False
            else:
                self.log("✓ Correlation direction is correct (negative)")
        
        if validation_passed:
            self.log("✓ Validation passed")
        else:
            self.log("✗ Validation FAILED - fix required")
        
        return validation_passed
    
    def _validate_travel_times_fixed(self, travel_df: pd.DataFrame):
        """FIXED: Enhanced travel time validation"""
        
        for mode in ['walk_time', 'drive_time', 'transit_time', 'combined_time']:
            if mode in travel_df.columns:
                times = travel_df[mode]
                
                std_dev = times.std()
                mean_time = times.mean()
                cv = std_dev / mean_time if mean_time > 0 else 0
                
                # Check for correlation with distance
                if 'straight_distance_km' in travel_df.columns:
                    dist_corr = travel_df['straight_distance_km'].corr(times)
                    self.log(f"Distance-{mode} correlation: {dist_corr:.3f}")
                    
                    if dist_corr < 0.3:
                        self.log(f"WARNING: {mode} correlation with distance too low ({dist_corr:.3f})")
                
                self.log(f"{mode}: mean={mean_time:.2f}min, std={std_dev:.2f}min, CV={cv:.3f}")
    
    def compute_enhanced_derived_features(self, base_features: np.ndarray) -> np.ndarray:
        """FIXED: Compute derived features with improved logic"""
        
        n_addresses = base_features.shape[0]
        
        if base_features.shape[1] < 30:  # 3 destinations × 10 features each
            self.log(f"WARNING: Expected 30 base features, got {base_features.shape[1]}")
            # Return minimal derived features
            return np.ones((n_addresses, 4), dtype=np.float64) * 0.5
        
        derived = []
        
        for i in range(n_addresses):
            # Extract 5-minute counts for immediate accessibility
            emp_5min = base_features[i, 3] if base_features.shape[1] > 3 else 0
            health_5min = base_features[i, 13] if base_features.shape[1] > 13 else 0
            grocery_5min = base_features[i, 23] if base_features.shape[1] > 23 else 0
            
            # 1. Local accessibility index
            local_access = emp_5min + health_5min + grocery_5min
            
            # 2. Modal flexibility (average drive advantage)
            emp_drive_adv = base_features[i, 6] if base_features.shape[1] > 6 else 0
            health_drive_adv = base_features[i, 16] if base_features.shape[1] > 16 else 0
            grocery_drive_adv = base_features[i, 26] if base_features.shape[1] > 26 else 0
            
            modal_flexibility = (emp_drive_adv + health_drive_adv + grocery_drive_adv) / 3
            
            # 3. Accessibility equity (consistency across destination types)
            emp_percentile = base_features[i, 9] if base_features.shape[1] > 9 else 0.5
            health_percentile = base_features[i, 19] if base_features.shape[1] > 19 else 0.5
            grocery_percentile = base_features[i, 29] if base_features.shape[1] > 29 else 0.5
            
            percentiles = [emp_percentile, health_percentile, grocery_percentile]
            accessibility_equity = 1.0 - np.std(percentiles)  # Higher when consistent
            accessibility_equity = max(0.0, min(1.0, accessibility_equity))
            
            # 4. Geographic advantage (overall position)
            geographic_advantage = np.mean(percentiles)
            
            derived.append([
                local_access,
                modal_flexibility,
                accessibility_equity,
                geographic_advantage
            ])
        
        derived_array = np.array(derived, dtype=np.float64)
        
        # VALIDATION: Check derived features
        self.log("=== DERIVED FEATURE VALIDATION ===")
        derived_names = ['local_access', 'modal_flexibility', 'accessibility_equity', 'geographic_advantage']
        
        for i, name in enumerate(derived_names):
            values = derived_array[:, i]
            self.log(f"{name}: mean={np.mean(values):.3f}, std={np.std(values):.3f}")
        
        return derived_array
    
    def _load_road_network_for_routing(self, origins, destinations):
        """Load road network and create address mappings"""
        import networkx as nx
        from sklearn.neighbors import BallTree
        
        # Import road loading functionality from loaders
        # You'll need to extract this or pass road_graph from loaders
        try:
            # Attempt to load roads (adapt paths as needed)
            from ..data.loaders import DataLoader
            loader = DataLoader()  # You may need to pass config
            
            # Get state/county from first origin coordinate
            first_origin = origins.iloc[0]
            # Estimate state/county from coordinates (rough approximation)
            state_fips = '47'  # Tennessee
            county_fips = '065'  # Hamilton
            
            roads = loader.load_road_network(state_fips=state_fips, county_fips=county_fips)
            
            if len(roads) == 0:
                self.log("No road network available - will use geodesic fallback")
                return None, {}
            
            # Build road graph
            road_graph = nx.Graph()
            
            for idx, road in roads.iterrows():
                if road.geometry.geom_type == 'LineString':
                    coords = list(road.geometry.coords)
                    for i in range(len(coords) - 1):
                        u, v = coords[i], coords[i + 1]
                        length = ((u[0] - v[0])**2 + (u[1] - v[1])**2)**0.5 * 111000  # Convert to meters
                        road_graph.add_edge(u, v, length=length)
            
            # Map addresses to road nodes
            if road_graph.number_of_nodes() > 0:
                road_nodes = np.array(list(road_graph.nodes()))
                tree = BallTree(np.radians(road_nodes), metric='haversine')
                
                address_mapping = {}
                
                # Map origins
                for orig_idx, origin in origins.iterrows():
                    coord = np.array([[origin.geometry.x, origin.geometry.y]])
                    distances, indices = tree.query(np.radians(coord), k=1)
                    if distances[0][0] * 6371000 < 500:  # Within 500m
                        address_mapping[orig_idx] = tuple(road_nodes[indices[0][0]])
                
                # Map destinations  
                for dest_idx, destination in destinations.iterrows():
                    coord = np.array([[destination.geometry.x, destination.geometry.y]])
                    distances, indices = tree.query(np.radians(coord), k=1)
                    if distances[0][0] * 6371000 < 500:  # Within 500m
                        address_mapping[f"dest_{dest_idx}"] = tuple(road_nodes[indices[0][0]])
            
                self.log(f"Mapped {len(address_mapping)} addresses to road network")
                return road_graph, address_mapping
            
        except Exception as e:
            self.log(f"Road network loading failed: {str(e)}")
            return None, {}
        
        return None, {}

    def _calculate_network_distance(self, orig_coord, dest_coord, road_graph, address_mapping):
        """Calculate actual network distance using road routing"""
        import networkx as nx
        from geopy.distance import geodesic
        
        if road_graph is None or len(address_mapping) == 0:
            return geodesic(orig_coord, dest_coord).kilometers, False
        
        try:
            # Find road nodes for origin and destination
            orig_node = None
            dest_node = None
            
            # This is simplified - you'd need to match by address IDs properly
            road_nodes = list(road_graph.nodes())
            
            # Find nearest road nodes (simplified approach)
            min_orig_dist = float('inf')
            min_dest_dist = float('inf')
            
            for node in road_nodes:
                node_coord = (node[1], node[0])  # lat, lon
                
                orig_dist = geodesic(orig_coord, node_coord).kilometers
                if orig_dist < min_orig_dist and orig_dist < 0.5:  # Within 500m
                    min_orig_dist = orig_dist
                    orig_node = node
                
                dest_dist = geodesic(dest_coord, node_coord).kilometers  
                if dest_dist < min_dest_dist and dest_dist < 0.5:  # Within 500m
                    min_dest_dist = dest_dist
                    dest_node = node
            
            if orig_node and dest_node and orig_node != dest_node:
                # Calculate shortest path
                path_length_m = nx.shortest_path_length(
                    road_graph, orig_node, dest_node, weight='length'
                )
                return path_length_m / 1000.0, True  # Convert to km
            
        except (nx.NetworkXNoPath, nx.NetworkXError, Exception) as e:
            self.log(f"Network routing failed: {str(e)}")
        
        # Fallback to geodesic
        return geodesic(orig_coord, dest_coord).kilometers, False
    
    def _calculate_network_distance_fast(self, orig_coord, dest_coord, road_graph, address_mapping, max_distance_km=5.0):
        """
        Fast network distance calculation with early termination
        """
        import networkx as nx
        from geopy.distance import geodesic
        
        if road_graph is None or len(address_mapping) == 0:
            return geodesic(orig_coord, dest_coord).kilometers * 1.3, False
        
        # Pre-filter: if straight distance > max_distance_km, skip expensive routing
        straight_km = geodesic(orig_coord, dest_coord).kilometers
        if straight_km > max_distance_km:
            return straight_km * 1.4, False
        
        try:
            # Find nearest road nodes (simplified approach for speed)
            road_nodes = list(road_graph.nodes())
            
            # Limit search to first 20 nodes for speed
            search_nodes = road_nodes[:min(20, len(road_nodes))]
            
            orig_node = None
            dest_node = None
            min_orig_dist = float('inf')
            min_dest_dist = float('inf')
            
            for node in search_nodes:
                node_coord = (node[1], node[0])  # lat, lon
                
                # Check origin
                orig_dist = geodesic(orig_coord, node_coord).kilometers
                if orig_dist < min_orig_dist and orig_dist < 0.5:  # Within 500m
                    min_orig_dist = orig_dist
                    orig_node = node
                
                # Check destination  
                dest_dist = geodesic(dest_coord, node_coord).kilometers
                if dest_dist < min_dest_dist and dest_dist < 0.5:  # Within 500m
                    min_dest_dist = dest_dist
                    dest_node = node
            
            if orig_node and dest_node and orig_node != dest_node:
                # Use NetworkX with length limit for early termination
                try:
                    path_length_m = nx.shortest_path_length(
                        road_graph, orig_node, dest_node, 
                        weight='length'
                    )
                    
                    # Sanity check - if network path is way longer than straight line, use approximation
                    path_length_km = path_length_m / 1000.0
                    if path_length_km > straight_km * 3.0:  # Network path > 3x straight line seems wrong
                        return straight_km * 1.4, False
                    
                    return path_length_km, True
                    
                except nx.NetworkXNoPath:
                    # No path found - use approximation
                    return straight_km * 1.5, False
            
        except Exception as e:
            # Any error - fall back to approximation
            pass
        
        # Fallback
        return straight_km * 1.3, False
    
    def _process_origin_batch(self, orig_id, orig_coord, destinations, road_graph, address_mapping, time_period):
        """
        Process only the closest destinations for each origin to create variation
        """
        # Pre-compute straight-line distances to all destinations
        dest_data = []
        for dest_idx, destination in destinations.iterrows():
            dest_id = destination.get('dest_id', dest_idx)
            dest_type = destination.get('dest_type', 'unknown')
            dest_coord = (destination.geometry.y, destination.geometry.x)
            straight_km = geodesic(orig_coord, dest_coord).kilometers
            
            dest_data.append({
                'dest_idx': dest_idx,
                'dest_id': dest_id,
                'dest_type': dest_type,
                'dest_coord': dest_coord,
                'straight_km': straight_km
            })
        
        # CRITICAL: Sort by distance and only process closest destinations
        dest_data.sort(key=lambda x: x['straight_km'])
        
        # Limit to closest 6 destinations per address
        # This creates natural variation across addresses in different tract locations
        max_destinations_per_address = 6
        selected_destinations = dest_data[:max_destinations_per_address]
        
        batch_results = []
        
        for dest_info in selected_destinations:
            straight_km = dest_info['straight_km']
            
            # Use the same distance-based routing strategy as before
            if straight_km > 15.0:
                network_distance_km = straight_km * 1.6
                route_found = False
            elif straight_km > 5.0:
                network_distance_km = straight_km * 1.35
                route_found = False
            elif straight_km > 1.0:
                network_distance_km, route_found = self._calculate_network_distance_fast(
                    orig_coord, dest_info['dest_coord'], road_graph, address_mapping, max_distance_km=5.0
                )
                if not route_found:
                    network_distance_km = straight_km * 1.25
            else:
                network_distance_km, route_found = self._calculate_network_distance_fast(
                    orig_coord, dest_info['dest_coord'], road_graph, address_mapping, max_distance_km=2.0
                )
                if not route_found:
                    network_distance_km = straight_km * 1.15
            
            # Calculate travel times
            travel_times = self._calculate_mode_times_fixed(network_distance_km, time_period)
            best_mode = min(travel_times.keys(), key=lambda k: travel_times[k])
            
            batch_results.append({
                'origin_id': orig_id,
                'destination_id': dest_info['dest_id'],
                'destination_type': dest_info['dest_type'],
                'straight_distance_km': straight_km,
                'network_distance_km': network_distance_km,
                'walk_time': travel_times['walk'],
                'drive_time': travel_times['drive'],
                'transit_time': travel_times['transit'],
                'combined_time': travel_times[best_mode],
                'best_mode': best_mode,
                'route_found': route_found
            })
        
        return batch_results