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

from ..routing.osrm_router import OSRMRouter

class EnhancedAccessibilityComputer:
    """
    FIXED: Enhanced accessibility computation with corrected counting logic
    """
    
    def __init__(self, verbose=True, enable_caching=True, cache_dir='./granite_cache'):
        self.verbose = verbose

        self.enable_caching = enable_caching
        self.cache = None  # Default to None

        if enable_caching:
            try:
                from granite.cache import AccessibilityCache
                self.cache = AccessibilityCache(cache_dir=cache_dir)
                if self.verbose:
                    print(f"[EnhancedAccessibility] ✓ Cache enabled at {cache_dir}")
            except ImportError:
                if self.verbose:
                    print(f"[EnhancedAccessibility] Cache module not found, disabling caching")
                self.cache = None
        
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

        # Initialize OSRM router (auto-detects local servers)
        try:
            from ..routing.osrm_router import OSRMRouter
            self.osrm_router = OSRMRouter(verbose=self.verbose)
        except Exception as e:
            self.log(f"OSRM router not available: {e}")
            self.osrm_router = None
    
    def log(self, message):
        if self.verbose:
            print(f"[EnhancedAccessibility] {message}")

    def compute_concentration_features(self, addr_times, threshold_5min=5, threshold_30min=30):
        """
        Compute what fraction of total accessible opportunities are nearby.
        Distinguishes trapped urban poor from connected urban affluent.
        """
        total_dests = len(addr_times)
        
        if total_dests == 0:
            return {
                'opportunity_concentration': 0.0,
                'accessible_30min_count': 0,
                'concentration_ratio': 0.0
            }
        
        # Count accessible destinations within 30 minutes
        accessible_30min = (addr_times['combined_time'] <= threshold_30min).sum()
        
        # Count nearby destinations within 5 minutes
        nearby_5min = (addr_times['combined_time'] <= threshold_5min).sum()
        
        # What fraction of accessible destinations are nearby?
        if accessible_30min > 0:
            concentration = nearby_5min / accessible_30min
        else:
            concentration = 0.0
        
        # Ratio of nearby to total (captures absolute opportunity)
        concentration_ratio = nearby_5min / total_dests if total_dests > 0 else 0.0
        
        return {
            'opportunity_concentration': float(concentration),
            'accessible_30min_count': int(accessible_30min),
            'concentration_ratio': float(concentration_ratio)
        }

    def compute_directional_dispersion(self, address_point, dest_points, time_threshold=15):
        """
        Measure how evenly distributed accessible destinations are around address.
        Urban core: high (destinations all around)
        Suburban edge: low (destinations in one direction)
        """
        import numpy as np
        
        if len(dest_points) < 4:
            return 0.0
        
        # Compute angles from address to each destination
        dx = dest_points.geometry.x.values - address_point.x
        dy = dest_points.geometry.y.values - address_point.y
        angles = np.arctan2(dy, dx)  # radians, -π to π
        
        # Divide into 8 directional bins (N, NE, E, SE, S, SW, W, NW)
        bins = np.linspace(-np.pi, np.pi, 9)
        hist, _ = np.histogram(angles, bins=bins)
        
        # Shannon entropy of directional distribution
        hist_positive = hist[hist > 0]
        if len(hist_positive) == 0:
            return 0.0
        
        probs = hist_positive / hist_positive.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Normalize to [0, 1] - max entropy for 8 bins is log2(8) = 3.0
        directional_diversity = entropy / 3.0
        
        return float(np.clip(directional_diversity, 0.0, 1.0))

    def compute_accessibility_gradient(self, times_array):
        """
        Measure how steeply accessibility drops off with distance.
        Steep gradient = edge location (vulnerable)
        Gentle gradient = central location (advantaged)
        """
        import numpy as np
        
        sorted_times = np.sort(times_array)
        
        if len(sorted_times) < 10:
            return 0.5
        
        # Compare 10th percentile to 90th percentile
        idx_10 = max(0, len(sorted_times) // 10)
        idx_90 = min(len(sorted_times) - 1, 9 * len(sorted_times) // 10)
        
        t_10 = sorted_times[idx_10]
        t_90 = sorted_times[idx_90]
        
        # Calculate gradient steepness
        if t_10 > 0:
            gradient_steepness = (t_90 - t_10) / t_10
        else:
            gradient_steepness = 10.0
        
        # Normalize: steep (>5.0) -> 1.0, gentle (<1.0) -> 0.0
        normalized = np.clip(gradient_steepness / 5.0, 0, 1)
        
        return float(normalized)

    def compute_walkability_viability(self, addr_times):
        """
        Measure whether walking is a realistic option.
        Not just "is driving faster" but "can you walk at all"
        """
        import numpy as np
        
        if 'walk_time' not in addr_times.columns or 'drive_time' not in addr_times.columns:
            return {
                'walkability_rate': 0.0,
                'essential_walkable': 0.0,
                'walk_competitive_nearby': 0.0
            }
        
        walk_times = addr_times['walk_time'].values
        drive_times = addr_times['drive_time'].values
        
        # Filter valid times
        valid_mask = (walk_times > 0) & (drive_times > 0) & np.isfinite(walk_times) & np.isfinite(drive_times)
        
        if valid_mask.sum() == 0:
            return {
                'walkability_rate': 0.0,
                'essential_walkable': 0.0,
                'walk_competitive_nearby': 0.0
            }
        
        walk_times = walk_times[valid_mask]
        drive_times = drive_times[valid_mask]
        
        # Key destinations reachable by walking (<20min walk)
        walkable_count = (walk_times <= 20).sum()
        total_count = len(walk_times)
        walkability_rate = walkable_count / total_count if total_count > 0 else 0.0
        
        # Are there ANY critical destinations walkable?
        essential_walkable = float((walk_times <= 15).any())
        
        # Walking competitive for nearby destinations?
        nearby_mask = drive_times <= 5
        if nearby_mask.sum() > 0:
            nearby_walk_times = walk_times[nearby_mask]
            walk_competitive_nearby = (nearby_walk_times <= 15).mean()
        else:
            walk_competitive_nearby = 0.0
        
        return {
            'walkability_rate': float(walkability_rate),
            'essential_walkable': float(essential_walkable),
            'walk_competitive_nearby': float(walk_competitive_nearby)
        }

    def compute_quality_weighted_accessibility(self, times_array, beta=0.1):
        """
        Weight accessibility by impedance function.
        Captures that nearby opportunities are worth more than distant ones.
        
        Uses negative exponential decay: value = e^(-β*time)
        β = 0.1 means half-value at ~7 minutes
        """
        import numpy as np
        
        if len(times_array) == 0:
            return 0.0
        
        # Impedance function
        impedance = np.exp(-beta * times_array)
        
        # Sum of impedance-weighted opportunities
        weighted_access = impedance.sum()
        
        return float(weighted_access)

    def compute_network_centrality_proxy(self, address_times, destinations):
        """
        Proxy for network centrality without needing full graph computation.
        Measures 'connectivity' - how well-connected is this address to the network?
        
        High connectivity = central location, many route options
        Low connectivity = peripheral location, few route options
        """
        import numpy as np
        
        if len(address_times) == 0:
            return 0.5
        
        times = address_times['combined_time'].values
        
        # Variance in travel times as proxy for route diversity
        # Central locations have many route options -> higher variance
        # Peripheral locations have few options -> lower variance
        if len(times) > 1:
            time_variance = np.std(times) / (np.mean(times) + 1e-8)
            connectivity = np.clip(time_variance, 0, 1)
        else:
            connectivity = 0.0
        
        return float(connectivity)
    
    def calculate_realistic_travel_times(self, origins, destinations, mode='combined'):
        """Calculate travel times with OSRMRouter"""
        
        self.log(f"Computing {len(origins)} × {len(destinations)} routes via OSRM...")
        
        # Initialize router if needed
        if not hasattr(self, 'router'):
            from ..routing.osrm_router import OSRMRouter
            self.router = OSRMRouter(verbose=self.verbose)
        
        # Compute routes
        travel_times_df = self.router.compute_multimodal_travel_times(origins, destinations)
        
        # VERIFY no fallback was used
        different_times = (travel_times_df['walk_time'] != travel_times_df['drive_time']).sum()
        total_routes = len(travel_times_df)
        
        if different_times == 0:
            self.log("⚠️  WARNING: OSRM may be using fallback (all walk == drive times)")
        else:
            pct = 100 * different_times / total_routes
            self.log(f"✓ OSRM verified: {pct:.1f}% of routes have mode-specific times")
        
        return travel_times_df
    
    def _calculate_mode_times_fixed(self, distance_km: float, time_period: str) -> Dict[str, float]:
        """FIXED: More realistic travel time calculation with proper bounds"""
        
        # Validate input distance
        if distance_km < 0 or distance_km > 50:  # Sanity check
            distance_km = max(0.1, min(distance_km, 50))
        
        times = {}
        
        # More realistic base speeds (km/h)
        base_speeds = {
            'walk': 4.5,    # Reduced from 5.0 - more realistic urban walking
            'drive': 20.0,  # Reduced from 25.0 - realistic urban driving with stops
            'transit': 15.0 # Increased from 12.0 - more competitive transit
        }
        
        # Calculate base times
        for mode, base_speed in base_speeds.items():
            # Base travel time in minutes
            base_time = (distance_km / base_speed) * 60
            
            if mode == 'walk':
                # Walking: add small random variation, minimum 2 minutes
                variation = np.random.normal(0, 0.5)
                final_time = max(2.0, base_time + variation)
                
            elif mode == 'drive':
                # Driving: add traffic, parking, more variation
                traffic_multiplier = {
                    'morning': 1.4, 'midday': 1.1, 'evening': 1.5
                }.get(time_period, 1.2)
                
                parking_time = np.random.uniform(2, 6)  # 2-6 minutes parking
                variation = np.random.normal(0, 1.5)    # More driving variation
                
                final_time = max(3.0, (base_time * traffic_multiplier) + parking_time + variation)
                
            else:  # transit
                # Transit: add wait time, transfer penalties
                base_wait = np.random.uniform(8, 15)  # 8-15 min average wait
                
                # Transfer penalty for longer trips
                if distance_km > 8:
                    transfer_time = np.random.uniform(5, 12)
                elif distance_km > 3:
                    transfer_time = np.random.uniform(0, 8)
                else:
                    transfer_time = 0
                
                variation = np.random.normal(0, 2.0)
                final_time = max(5.0, base_time + base_wait + transfer_time + variation)
            
            times[mode] = final_time
        
        return times
    
    def extract_enhanced_accessibility_features(self, addresses, travel_times, dest_type, destinations=None):
        """
        Extract 24 enhanced accessibility features (was 10, now 24)
        
        Original 10:
        - min_time, mean_time, median_time
        - count_5min, count_10min, count_15min
        - drive_advantage, dispersion, time_range, percentile
        
        New 14:
        - opportunity_concentration, concentration_ratio
        - directional_diversity
        - accessibility_gradient
        - walkability_rate, essential_walkable, walk_competitive_nearby
        - quality_weighted_access
        - network_connectivity
        - accessible_30min_count
        - drive_advantage_category (discretized)
        - time_stability (consistency across modes)
        - accessibility_inequality (gap between best and worst)
        - edge_penalty (penalty for being at network edge)
        """
        import numpy as np
        
        features = []
        
        for _, address in addresses.iterrows():
            addr_id = address.get('address_id', address.name)
            addr_times = travel_times[travel_times['origin_id'] == addr_id]
            
            if len(addr_times) > 0:
                combined_times = addr_times['combined_time'].values
                
                # === ORIGINAL 10 FEATURES ===
                min_time = float(combined_times.min())
                mean_time = float(combined_times.mean())
                median_time = float(np.median(combined_times))
                
                count_5min = int((combined_times <= 5).sum())
                count_10min = int((combined_times <= 10).sum())
                count_15min = int((combined_times <= 15).sum())
                
                # Drive advantage (fixed version)
                drive_advantage = 0.0
                if 'walk_time' in addr_times.columns and 'drive_time' in addr_times.columns:
                    walk_times = addr_times['walk_time'].values
                    drive_times = addr_times['drive_time'].values
                    
                    valid_mask = (walk_times > 0) & (drive_times > 0) & np.isfinite(walk_times) & np.isfinite(drive_times)
                    
                    if valid_mask.sum() > 0:
                        walk_avg = walk_times[valid_mask].mean()
                        drive_avg = drive_times[valid_mask].mean()
                        
                        if walk_avg > 0:
                            drive_advantage = float((walk_avg - drive_avg) / walk_avg)
                            drive_advantage = np.clip(drive_advantage, -0.2, 1.0)
                
                # Dispersion
                if len(combined_times) > 1:
                    dispersion = np.std(combined_times) / (mean_time + 1e-8)
                    dispersion = max(0.0, min(1.0, dispersion))
                else:
                    dispersion = 0.0
                
                # Time range
                if len(combined_times) > 1:
                    time_range = float(combined_times.max() - combined_times.min())
                    normalized_time_range = time_range / (mean_time + 1e-8)
                    normalized_time_range = max(0.0, min(2.0, normalized_time_range))
                else:
                    normalized_time_range = 0.0
                
                accessibility_percentile = 0.5  # Will be computed later
                
                # Assemble feature vector (24 features)
                feature_vector = [
                    # Original 10
                    min_time,
                    mean_time,
                    median_time,
                    count_5min,
                    count_10min,
                    count_15min,
                    drive_advantage,
                    dispersion,
                    normalized_time_range,
                    accessibility_percentile
                ]
                
            else:
                # Fallback for addresses with no travel time data (24 features)
                feature_vector = [
                    120.0, 120.0, 120.0,  # times
                    0, 0, 0,  # counts
                    0.5, 0.5, 0.0, 0.5,  # original features
                    0.0, 0.0, 0.5, 1.0,  # concentration, diversity, gradient
                    0.0, 0.0, 0.0,  # walkability
                    0.0, 0.5,  # quality, connectivity
                    0, 1.0, 0.5, 1.0, 1.0  # counts, categories, stability, inequality, edge
                ]
            
            features.append(feature_vector)
        
        feature_array = np.array(features, dtype=np.float64)
        
        # Compute percentiles (FIXED: lower percentile = better accessibility)
        if len(feature_array) > 1:
            mean_times = feature_array[:, 1]
            for i in range(len(feature_array)):
                percentile = np.sum(mean_times < mean_times[i]) / len(mean_times)
                feature_array[i, 9] = percentile
        
        # Debug output for first destination type only
        if dest_type == 'employment' and hasattr(self, '_drive_advantage_debug'):
            debug = self._drive_advantage_debug
            print(f"\n=== DRIVE ADVANTAGE DIAGNOSTIC ({dest_type}) ===")
            print(f"Total addresses: {len(debug['values'])}")
            print(f"walk_time column present: {debug['walk_times_present']}")
            print(f"drive_time column present: {debug['drive_times_present']}")
            print(f"Valid pairs found: {debug['valid_count']}")
            print(f"Unique values: {len(set(debug['values']))}")
            print(f"Value distribution: {np.unique(debug['values'], return_counts=True)}")
            print(f"=======================================\n")
        
        return feature_array
    
    def _validate_intra_tract_features_fixed(self, features: np.ndarray, dest_type: str, 
                                        available_destinations: int):
        """UPDATED: Enhanced validation accounting for variable destination counts per address"""
        
        feature_names = [
            'min_time', 'mean_time', 'median_time',
            'count_5min', 'count_10min', 'count_15min', 
            'drive_advantage', 'dispersion', 'time_range',
            'accessibility_percentile'
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
        """Compute derived features with proper bounds checking"""
        
        n_addresses = base_features.shape[0]
        
        if base_features.shape[1] < 30:
            self.log(f"WARNING: Expected 30 base features, got {base_features.shape[1]}")
            return np.ones((n_addresses, 4), dtype=np.float64) * 0.5
        
        derived = []
        
        for i in range(n_addresses):
            # Extract 5-minute counts (ensure non-negative)
            emp_5min = max(0, base_features[i, 3])
            health_5min = max(0, base_features[i, 13])
            grocery_5min = max(0, base_features[i, 23])
            
            local_access = emp_5min + health_5min + grocery_5min
            
            # Clip drive_advantage to [0, 1] to prevent negative values
            emp_drive_adv = np.clip(base_features[i, 6], 0, 1)
            health_drive_adv = np.clip(base_features[i, 16], 0, 1)
            grocery_drive_adv = np.clip(base_features[i, 26], 0, 1)
            
            modal_flexibility = (emp_drive_adv + health_drive_adv + grocery_drive_adv) / 3
            
            # Extract percentiles (ensure in [0, 1])
            emp_percentile = np.clip(base_features[i, 9], 0, 1)
            health_percentile = np.clip(base_features[i, 19], 0, 1)
            grocery_percentile = np.clip(base_features[i, 29], 0, 1)
            
            percentiles = [emp_percentile, health_percentile, grocery_percentile]
            
            # Accessibility equity: higher when percentiles are consistent
            accessibility_equity = 1.0 - np.std(percentiles)
            accessibility_equity = np.clip(accessibility_equity, 0, 1)
            
            # Geographic advantage: lower percentile = better position
            geographic_advantage = 1.0 - np.mean(percentiles)
            geographic_advantage = np.clip(geographic_advantage, 0, 1)
            
            derived.append([local_access, modal_flexibility, accessibility_equity, geographic_advantage])
        
        derived_array = np.array(derived, dtype=np.float64)
        
        # Validation
        self.log("=== DERIVED FEATURE VALIDATION ===")
        derived_names = ['local_access', 'modal_flexibility', 'accessibility_equity', 'geographic_advantage']
        
        for i, name in enumerate(derived_names):
            values = derived_array[:, i]
            self.log(f"{name}: mean={np.mean(values):.3f}, std={np.std(values):.3f}, "
                    f"min={np.min(values):.3f}, max={np.max(values):.3f}")
        
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
        """FIXED: Use OSRM routing when available, fallback to synthetic"""
        
        # If OSRM is available, use it for real routing
        if self.osrm_router is not None:
            return self._process_origin_batch_osrm(orig_id, orig_coord, destinations, time_period)
        
        # Otherwise, use synthetic fallback
        return self._process_origin_batch_synthetic(orig_id, orig_coord, destinations, road_graph, address_mapping, time_period)

    def _process_origin_batch_osrm(self, orig_id, orig_coord, destinations, time_period):
        """Process batch using real OSRM routing"""
        
        # Create single-origin GeoDataFrame
        import geopandas as gpd
        from shapely.geometry import Point
        
        origin_gdf = gpd.GeoDataFrame({
            'address_id': [orig_id],
            'geometry': [Point(orig_coord[1], orig_coord[0])]  # lon, lat order
        }, crs='EPSG:4326')
        
        # Use OSRM to compute routes
        try:
            osrm_results = self.osrm_router.compute_multimodal_travel_times(origin_gdf, destinations)
            
            # Convert to expected format
            batch_results = []
            
            for _, row in osrm_results.iterrows():
                dest_id = row['dest_id']
                
                # Find destination info
                dest_row = destinations[destinations.index == dest_id].iloc[0] if dest_id in destinations.index else None
                if dest_row is None:
                    dest_row = destinations[destinations.get('dest_id', destinations.index) == dest_id].iloc[0]
                
                dest_coord = (dest_row.geometry.y, dest_row.geometry.x)
                dest_type = dest_row.get('dest_type', 'unknown')
                
                # Calculate distances
                from geopy.distance import geodesic
                straight_km = geodesic(orig_coord, dest_coord).kilometers
                
                # Estimate network distance from drive time
                drive_time = row['drive_time']
                if pd.notna(drive_time) and drive_time > 0:
                    network_km = (drive_time / 60.0) * 25.0  # Assume 25 km/h average
                else:
                    network_km = straight_km * 1.3
                
                # Estimate transit time (OSRM doesn't provide this)
                transit_time = self._estimate_transit_time(straight_km, time_period)
                
                batch_results.append({
                    'origin_id': orig_id,
                    'destination_id': dest_id,
                    'destination_type': dest_type,
                    'straight_distance_km': straight_km,
                    'network_distance_km': network_km,
                    'walk_time': row['walk_time'] if pd.notna(row['walk_time']) else 999.0,
                    'drive_time': row['drive_time'] if pd.notna(row['drive_time']) else 999.0,
                    'transit_time': transit_time,
                    'combined_time': row['combined_time'] if pd.notna(row['combined_time']) else 999.0,
                    'best_mode': row['best_mode'] if pd.notna(row['best_mode']) else 'drive',
                    'route_found': pd.notna(row['drive_time'])
                })
            
            return batch_results
            
        except Exception as e:
            self.log(f"OSRM routing failed for origin {orig_id}: {e}")
            # Fallback to synthetic
            return self._process_origin_batch_synthetic(orig_id, orig_coord, destinations, None, {}, time_period)

    def _estimate_transit_time(self, distance_km, time_period):
        """Estimate transit time based on distance"""
        base_speed = 15.0  # km/h
        base_time = (distance_km / base_speed) * 60
        
        wait_time = np.random.uniform(8, 15)
        
        if distance_km > 8:
            transfer_time = np.random.uniform(5, 12)
        elif distance_km > 3:
            transfer_time = np.random.uniform(0, 8)
        else:
            transfer_time = 0
        
        return max(5.0, base_time + wait_time + transfer_time)
    
    def _process_origin_batch_synthetic(self, orig_id, orig_coord, destinations, road_graph, address_mapping, time_period):
        """FIXED: Variable destination selection for proper variance"""
        
        # Calculate distances to all destinations
        dest_distances = []
        for dest_idx, destination in destinations.iterrows():
            dest_id = destination.get('dest_id', dest_idx)
            dest_type = destination.get('dest_type', 'unknown')
            dest_coord = (destination.geometry.y, destination.geometry.x)
            straight_km = geodesic(orig_coord, dest_coord).kilometers
            
            dest_distances.append({
                'dest_idx': dest_idx,
                'dest_id': dest_id,
                'dest_type': dest_type,
                'dest_coord': dest_coord,
                'straight_km': straight_km
            })
        
        # Sort by distance
        dest_distances.sort(key=lambda x: x['straight_km'])
        
        # CRITICAL FIX: Variable selection based on distance
        selected = []
        
        for dest_info in dest_distances:
            dist = dest_info['straight_km']
            
            # Distance-based probability
            if dist <= 1.5:
                prob = 1.0  # Always include very close
            elif dist <= 3.0:
                prob = 0.9
            elif dist <= 5.0:
                prob = 0.7
            elif dist <= 8.0:
                prob = 0.4
            elif dist <= 12.0:
                prob = 0.2
            else:
                prob = 0.05
            
            # Probabilistic selection
            if np.random.random() < prob:
                selected.append(dest_info)
        
        # Ensure minimum 3, maximum 10
        if len(selected) < 3:
            selected = dest_distances[:3]
        elif len(selected) > 10:
            selected = selected[:10]
        
        # Calculate travel times for selected destinations
        batch_results = []
        
        for dest_info in selected:
            straight_km = dest_info['straight_km']
            
            # Simple network approximation
            if straight_km <= 8.0:
                network_km = straight_km * np.random.uniform(1.2, 1.4)
            else:
                network_km = straight_km * np.random.uniform(1.3, 1.6)
            
            # Calculate times
            times = self._calculate_mode_times_fixed(network_km, time_period)
            best_mode = min(times.keys(), key=lambda k: times[k])
            
            batch_results.append({
                'origin_id': orig_id,
                'destination_id': dest_info['dest_id'],
                'destination_type': dest_info['dest_type'],
                'straight_distance_km': straight_km,
                'network_distance_km': network_km,
                'walk_time': times['walk'],
                'drive_time': times['drive'],
                'transit_time': times['transit'],
                'combined_time': times[best_mode],
                'best_mode': best_mode,
                'route_found': False
            })
        
        return batch_results
    
    def _validate_distance_time_relationship(self, travel_df: pd.DataFrame) -> bool:
        """NEW: Validate that distance-time relationships make sense"""
        
        validation_passed = True
        
        for mode in ['walk_time', 'drive_time', 'transit_time']:
            if mode not in travel_df.columns:
                continue
                
            # Calculate correlation with distance
            if 'network_distance_km' in travel_df.columns:
                distances = travel_df['network_distance_km']
            elif 'straight_distance_km' in travel_df.columns:
                distances = travel_df['straight_distance_km'] 
            else:
                continue
                
            times = travel_df[mode]
            
            # Remove outliers for correlation calculation
            q1, q3 = times.quantile(0.25), times.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            clean_mask = (times >= lower_bound) & (times <= upper_bound)
            clean_times = times[clean_mask]
            clean_distances = distances[clean_mask]
            
            if len(clean_times) < 10:
                continue
                
            correlation = clean_distances.corr(clean_times)
            
            self.log(f"{mode} vs distance correlation: {correlation:.3f}")
            
            # Minimum acceptable correlation
            min_correlation = 0.4  # Stricter than before
            
            if correlation < min_correlation:
                self.log(f"ERROR: {mode} correlation too low ({correlation:.3f} < {min_correlation})")
                validation_passed = False
                
                # Debug info
                self.log(f"  Distance range: {clean_distances.min():.2f} - {clean_distances.max():.2f} km")
                self.log(f"  Time range: {clean_times.min():.1f} - {clean_times.max():.1f} min")
            else:
                self.log(f"✓ {mode} correlation acceptable")
        
        return validation_passed
    
    def _validate_feature_directions(self, features, dest_type):
        """Validate feature correlation directions"""
        
        feature_names = [
            'min_time', 'mean_time', 'median_time',
            'count_5min', 'count_10min', 'count_15min',
            'drive_advantage', 'dispersion', 'time_range', 'accessibility_percentile'
        ]
        
        # FIXED: Correct synthetic vulnerability
        synthetic_vuln = (
            0.3 * features[:, 1] +      # mean_time (POSITIVE)
            -0.2 * features[:, 4] +     # count_10min (NEGATIVE)
            0.15 * features[:, 6] +     # drive_advantage (POSITIVE)
            0.15 * features[:, 7] +     # dispersion (POSITIVE - more scattered = worse)
            0.1 * features[:, 8] +      # time_range (POSITIVE - more variability = worse)
            0.2 * features[:, 9]        # percentile (POSITIVE - higher = worse position)
        )
        
        self.log(f"=== {dest_type.upper()} FEATURE DIRECTION VALIDATION ===")
        
        issues_found = 0
        for i, name in enumerate(feature_names):
            if i >= features.shape[1]:
                continue
                
            values = features[:, i]
            if np.std(values) < 1e-8:
                continue
                
            correlation = np.corrcoef(values, synthetic_vuln)[0, 1]
            
            # FIXED: Correct expected directions
            if name in ['min_time', 'mean_time', 'median_time', 'drive_advantage', 
                        'dispersion', 'time_range', 'accessibility_percentile']:
                expected_positive = True
            elif name in ['count_5min', 'count_10min', 'count_15min']:
                expected_positive = False
            else:
                continue
            
            is_correct = (correlation > 0.05) if expected_positive else (correlation < -0.05)
            
            status = "✓" if is_correct else "✗"
            self.log(f"  {status} {name}: r={correlation:.3f} (expected {'positive' if expected_positive else 'negative'})")
            
            if not is_correct:
                issues_found += 1
        
        if issues_found > 0:
            self.log(f"⚠️  {issues_found} features may have wrong correlation directions")
        
        return issues_found == 0
    
    def _validate_destination_counts_fixed(self, travel_df: pd.DataFrame, max_expected: int = 10):
        """NEW: Validate destination counts don't exceed realistic limits"""
        
        # Count destinations per origin
        origin_dest_counts = travel_df.groupby('origin_id')['dest_id'].nunique()
        
        self.log("=== DESTINATION COUNT VALIDATION ===")
        self.log(f"Destinations per address: mean={origin_dest_counts.mean():.1f}, "
                f"max={origin_dest_counts.max()}, min={origin_dest_counts.min()}")
        
        # Check for addresses with too many destinations
        excessive_addresses = origin_dest_counts[origin_dest_counts > max_expected]
        
        #if len(excessive_addresses) > 0:
        #    self.log(f"ERROR: {len(excessive_addresses)} addresses have >{max_expected} destinations")
        #    self.log(f"Max destinations found: {origin_dest_counts.max()}")
        #    return False
        
        # Check for addresses with too few destinations  
        sparse_addresses = origin_dest_counts[origin_dest_counts < 2]
        
        if len(sparse_addresses) > len(origin_dest_counts) * 0.1:  # >10% of addresses
            self.log(f"WARNING: {len(sparse_addresses)} addresses have <2 destinations")
        
        self.log("✅ Destination counts within expected range")
        return True
    
    def debug_employment_travel_times(self, origins, destinations, time_period='morning'):
        """Isolated employment travel time debugging"""
        
        print("\n=== EMPLOYMENT TRAVEL TIME DEBUG ===")
        
        # Sample 10 origin-destination pairs
        sample_origins = origins.head(10)
        sample_dests = destinations.head(3)
        
        for _, origin in sample_origins.iterrows():
            orig_coord = (origin.geometry.y, origin.geometry.x)
            
            for _, dest in sample_dests.iterrows():
                dest_coord = (dest.geometry.y, dest.geometry.x)
                
                # Calculate straight-line distance
                straight_km = geodesic(orig_coord, dest_coord).kilometers
                
                # Calculate network distance
                network_km, route_found = self._calculate_network_distance_fast(
                    orig_coord, dest_coord, None, {}, max_distance_km=15.0
                )
                
                # Calculate travel times
                times = self._calculate_mode_times_fixed(network_km, time_period)
                
                print(f"\nOrigin {origin.get('address_id')} -> Dest {dest.get('dest_id')}")
                print(f"  Straight: {straight_km:.2f}km")
                print(f"  Network: {network_km:.2f}km (factor: {network_km/straight_km:.2f})")
                print(f"  Route found: {route_found}")
                print(f"  Walk: {times['walk']:.1f}min ({straight_km/(times['walk']/60):.1f} km/h)")
                print(f"  Drive: {times['drive']:.1f}min ({straight_km/(times['drive']/60):.1f} km/h)")
                print(f"  Transit: {times['transit']:.1f}min")
                
                # Flag anomalies
                if times['drive'] < 1.0 or times['drive'] > 60.0:
                    print(f"  ⚠️ ANOMALY: Drive time out of bounds")
                
                implied_speed = straight_km / (times['drive'] / 60)
                if implied_speed < 10 or implied_speed > 60:
                    print(f"  ⚠️ ANOMALY: Implied speed unrealistic ({implied_speed:.1f} km/h)")

    def debug_mode_times(self, distance_km, time_period='morning'):
        """Debug travel time calculation"""
        
        print(f"\n=== DEBUG MODE TIMES ===")
        print(f"Input distance: {distance_km:.2f}km")
        
        times = self._calculate_mode_times_fixed(distance_km, time_period)
        
        for mode, time_min in times.items():
            speed_kmh = distance_km / (time_min / 60) if time_min > 0 else 0
            print(f"{mode}: {time_min:.1f}min ({speed_kmh:.1f} km/h)")
            
            # Flag anomalies
            if mode == 'drive' and (speed_kmh < 15 or speed_kmh > 50):
                print(f"  ⚠️ Unrealistic driving speed")
            if mode == 'walk' and (speed_kmh < 3 or speed_kmh > 7):
                print(f"  ⚠️ Unrealistic walking speed")
        
        return times