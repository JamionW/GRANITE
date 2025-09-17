"""
Enhanced Accessibility Computation for GRANITE
Fixes systematic issues in travel time calculation and feature engineering
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
    Enhanced accessibility computation with realistic variability and proper validation
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        
        # Realistic speed parameters with variability
        self.speed_params = {
            'walk': {'base': 5.0, 'std': 0.8, 'min': 3.5, 'max': 6.5},      # km/h
            'drive': {'base': 25.0, 'std': 8.0, 'min': 15.0, 'max': 45.0},  # km/h urban
            'transit': {'base': 12.0, 'std': 4.0, 'min': 8.0, 'max': 20.0}  # km/h average
        }
        
        # Network complexity factors
        self.network_factors = {
            'urban_core': 1.4,      # High density, complex routing
            'suburban': 1.3,        # Moderate complexity
            'edge': 1.2            # Simpler routing
        }
        
        # Time-of-day factors
        self.time_factors = {
            'morning': {'drive': 1.3, 'transit': 1.2, 'walk': 1.0},
            'midday': {'drive': 1.0, 'transit': 1.0, 'walk': 1.0},
            'evening': {'drive': 1.4, 'transit': 1.1, 'walk': 1.0}
        }
    
    def log(self, message):
        if self.verbose:
            print(f"[EnhancedAccessibility] {message}")
    
    def calculate_realistic_travel_times(self, origins: gpd.GeoDataFrame, 
                                       destinations: gpd.GeoDataFrame,
                                       time_period: str = 'morning') -> pd.DataFrame:
        """
        Calculate realistic travel times with proper variability
        """
        self.log(f"Computing enhanced travel times: {len(origins)} origins → {len(destinations)} destinations")
        
        results = []
        
        # Determine urban context for network complexity
        origin_coords = np.array([[orig.geometry.x, orig.geometry.y] for _, orig in origins.iterrows()])
        urban_context = self._classify_urban_context(origin_coords)
        
        for i, (orig_idx, origin) in enumerate(origins.iterrows()):
            orig_id = origin.get('address_id', orig_idx)
            orig_coord = (origin.geometry.y, origin.geometry.x)
            orig_context = urban_context[i]  # Use enumeration index, not DataFrame index
            
            for dest_idx, destination in destinations.iterrows():
                dest_id = destination.get('dest_id', dest_idx)
                dest_coord = (destination.geometry.y, destination.geometry.x)
                dest_type = destination.get('dest_type', 'unknown')
                
                # Calculate base distance
                straight_distance = geodesic(orig_coord, dest_coord).kilometers
                
                # Apply network complexity based on urban context
                network_factor = self.network_factors[orig_context]
                network_distance = straight_distance * network_factor
                
                # Add distance-based variability (real networks aren't perfectly predictable)
                distance_noise = np.random.normal(0, 0.1 * network_distance)
                network_distance = max(straight_distance, network_distance + distance_noise)
                
                # Calculate mode-specific times with variability
                travel_times = self._calculate_mode_times(
                    network_distance, time_period, dest_type
                )
                
                # Determine best mode
                best_mode = min(travel_times.keys(), key=lambda k: travel_times[k])
                combined_time = travel_times[best_mode]
                
                results.append({
                    'origin_id': orig_id,
                    'destination_id': dest_id,
                    'destination_type': dest_type,
                    'straight_distance_km': straight_distance,
                    'network_distance_km': network_distance,
                    'walk_time': travel_times['walk'],
                    'drive_time': travel_times['drive'],
                    'transit_time': travel_times['transit'],
                    'combined_time': combined_time,
                    'best_mode': best_mode,
                    'urban_context': orig_context,
                    'time_period': time_period
                })
        
        travel_df = pd.DataFrame(results)
        self.log(f"Generated {len(travel_df)} travel time records")
        self._validate_travel_times(travel_df)
        
        return travel_df
    
    def _classify_urban_context(self, coords: np.ndarray) -> List[str]:
        """
        Classify origins by urban context for network complexity
        """
        # Simple classification based on coordinate density
        # In a real implementation, this could use land use data
        
        from sklearn.neighbors import NearestNeighbors
        
        if len(coords) < 10:
            return ['suburban'] * len(coords)
        
        # Calculate local density
        nbrs = NearestNeighbors(n_neighbors=min(10, len(coords))).fit(coords)
        distances, _ = nbrs.kneighbors(coords)
        
        # Mean distance to 5 nearest neighbors (excluding self)
        avg_distances = np.mean(distances[:, 1:6], axis=1)
        
        # Classify based on density thresholds
        contexts = []
        for dist in avg_distances:
            if dist < 0.002:  # High density
                contexts.append('urban_core')
            elif dist < 0.005:  # Medium density
                contexts.append('suburban')
            else:  # Low density
                contexts.append('edge')
        
        return contexts
    
    def _calculate_mode_times(self, distance_km: float, time_period: str, 
                            dest_type: str) -> Dict[str, float]:
        """
        Calculate mode-specific travel times with realistic variability
        """
        times = {}
        time_multipliers = self.time_factors.get(time_period, self.time_factors['midday'])
        
        for mode, params in self.speed_params.items():
            # Base time calculation
            base_speed = params['base']
            
            # Add speed variability based on conditions
            if mode == 'drive' and distance_km > 10:
                # Longer drives can use highways
                speed_variance = np.random.normal(0, params['std'] * 0.5)
                actual_speed = base_speed + 10 + speed_variance  # Highway speeds
            elif mode == 'transit' and distance_km < 2:
                # Transit penalty for short distances (wait time dominates)
                speed_variance = np.random.normal(0, params['std'])
                actual_speed = max(params['min'], base_speed * 0.6 + speed_variance)
            else:
                # Normal variability
                speed_variance = np.random.normal(0, params['std'])
                actual_speed = base_speed + speed_variance
            
            # Clamp to realistic bounds
            actual_speed = np.clip(actual_speed, params['min'], params['max'])
            
            # Calculate time
            base_time = distance_km / actual_speed * 60  # Convert to minutes
            
            # Apply time-of-day factors
            time_factor = time_multipliers.get(mode, 1.0)
            final_time = base_time * time_factor
            
            # Add mode-specific penalties
            if mode == 'transit':
                # Add wait time (varies by time of day and distance)
                wait_time = np.random.normal(8, 3) if distance_km > 3 else np.random.normal(12, 4)
                wait_time = max(2, wait_time)  # Minimum 2 min wait
                final_time += wait_time
                
                # Add transfer penalty for longer trips
                if distance_km > 8:
                    transfer_penalty = np.random.normal(5, 2)  # Transfer time
                    final_time += max(0, transfer_penalty)
            
            elif mode == 'drive':
                # Add parking time for certain destinations
                if dest_type in ['healthcare', 'employment']:
                    parking_time = np.random.normal(3, 1)
                    final_time += max(1, parking_time)
            
            times[mode] = max(1.0, final_time)  # Minimum 1 minute
        
        return times
    
    def _validate_travel_times(self, travel_df: pd.DataFrame):
        """
        Validate that travel times have realistic properties
        """
        # Check for variability
        for mode in ['walk_time', 'drive_time', 'transit_time', 'combined_time']:
            std_dev = travel_df[mode].std()
            mean_time = travel_df[mode].mean()
            cv = std_dev / mean_time if mean_time > 0 else 0
            
            self.log(f"{mode}: mean={mean_time:.2f}min, std={std_dev:.2f}min, CV={cv:.3f}")
            
            if cv < 0.1:
                self.log(f"WARNING: {mode} has low variability (CV={cv:.3f})")
        
        # Check distance-time correlations
        for mode in ['walk_time', 'drive_time', 'transit_time']:
            corr = travel_df['straight_distance_km'].corr(travel_df[mode])
            self.log(f"Distance-{mode} correlation: {corr:.3f}")
            
            if abs(corr) > 0.95:
                self.log(f"WARNING: {mode} correlation too high ({corr:.3f})")
    
    def extract_enhanced_accessibility_features(self, addresses: gpd.GeoDataFrame,
                                            travel_times: pd.DataFrame,
                                            dest_type: str) -> np.ndarray:
        """
        UPDATED: Extract accessibility features designed for intra-tract variation analysis
        """
        self.log(f"Extracting intra-tract features for {dest_type} destinations")
        
        features = []
        
        for addr_idx, address in addresses.iterrows():
            addr_id = address.get('address_id', addr_idx)
            
            # Get travel times for this address
            addr_times = travel_times[
                travel_times['origin_id'].astype(str) == str(addr_id)
            ]
            
            if len(addr_times) == 0:
                # Default values for no access
                features.append([15.0, 20.0, 20.0, 0, 1, 2, 0.0, 1.0, 5.0, 0.5])
                continue
            
            combined_times = pd.to_numeric(addr_times['combined_time'], errors='coerce').dropna()
            
            if len(combined_times) == 0:
                features.append([15.0, 20.0, 20.0, 0, 1, 2, 0.0, 1.0, 5.0, 0.5])
                continue
            
            # INTRA-TRACT FEATURES (designed for census tract scale):
            
            # 1. Time-based features (more granular)
            min_time = float(combined_times.min())
            mean_time = float(combined_times.mean())
            median_time = float(combined_times.median())
            
            # 2. Fine-grained count features (meaningful within tract)
            count_5min = int((combined_times <= 5).sum())    # Walking distance
            count_10min = int((combined_times <= 10).sum())  # Short drive/bike
            count_15min = int((combined_times <= 15).sum())  # Neighborhood access
            
            # 3. Modal accessibility 
            drive_times = pd.to_numeric(addr_times['drive_time'], errors='coerce').dropna()
            walk_times = pd.to_numeric(addr_times['walk_time'], errors='coerce').dropna()
            
            if len(drive_times) > 0 and len(walk_times) > 0:
                drive_advantage = float(walk_times.min() - drive_times.min())
            else:
                drive_advantage = 0.0
            
            # 4. Accessibility clustering (captures local geography)
            if len(combined_times) > 1:
                time_range = float(combined_times.max() - combined_times.min())
                accessibility_concentration = float(1.0 / (1.0 + time_range / 5.0))
            else:
                time_range = 0.0
                accessibility_concentration = 1.0
            
            # 5. Placeholder for relative accessibility (computed later)
            accessibility_percentile = 0.5
            
            feature_vector = [
                min_time,                    # 0: Best access time
                mean_time,                   # 1: Average access time  
                median_time,                 # 2: Median access time
                count_5min,                  # 3: Immediate access count
                count_10min,                 # 4: Local access count
                count_15min,                 # 5: Neighborhood access count
                drive_advantage,             # 6: Car vs walk advantage
                accessibility_concentration, # 7: Geographic clustering
                time_range,                  # 8: Access time spread
                accessibility_percentile     # 9: Relative position (filled later)
            ]
            
            features.append(feature_vector)
        
        feature_array = np.array(features, dtype=np.float64)
        
        # Post-process: Compute relative accessibility percentiles
        if len(feature_array) > 1:
            for i in range(len(feature_array)):
                mean_access_time = feature_array[i, 1]  # mean_time
                # Percentile rank (lower time = better access = higher percentile)
                percentile = float(np.sum(feature_array[:, 1] >= mean_access_time) / len(feature_array))
                feature_array[i, 9] = percentile
        
        # Validate features
        self._validate_intra_tract_features(feature_array, dest_type)
        
        return feature_array
    
    def _validate_intra_tract_features(self, features: np.ndarray, dest_type: str):
        """UPDATED: Validate features for intra-tract analysis"""
        
        feature_names = [
            'min_time', 'mean_time', 'median_time',
            'count_5min', 'count_10min', 'count_15min', 
            'drive_advantage', 'accessibility_concentration',
            'time_range', 'accessibility_percentile'
        ]
        
        self.log(f"=== {dest_type.upper()} INTRA-TRACT VALIDATION ===")
        
        zero_var_count = 0
        for i, name in enumerate(feature_names):
            values = features[:, i]
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            self.log(f"{name}: std={std_val:.4f}, range=[{min_val:.2f}, {max_val:.2f}]")
            
            if std_val < 1e-6:
                self.log(f"WARNING: {name} has minimal variation")
                zero_var_count += 1
        
        if zero_var_count == 0:
            self.log("✓ All features have meaningful variation for intra-tract analysis")
        else:
            self.log(f"INFO: {zero_var_count} features have limited variation (may be acceptable for local analysis)")
        
        # Check for expected relationships
        min_times = features[:, 0]
        count_5min = features[:, 3]
        
        if np.std(min_times) > 0 and np.std(count_5min) > 0:
            corr = np.corrcoef(min_times, count_5min)[0, 1]
            self.log(f"Min_time vs Count_5min correlation: {corr:.3f}")
    
    def _validate_feature_relationships(self, features: np.ndarray, 
                                      feature_names: List[str], dest_type: str):
        """
        Validate expected relationships between features
        """
        min_times = features[:, 0]  # min_time
        mean_times = features[:, 1]  # mean_time
        count_60 = features[:, 4]   # count_60min
        
        # Check time ordering
        time_order_violations = np.sum(min_times > mean_times)
        if time_order_violations > 0:
            self.log(f"WARNING: {time_order_violations} addresses with min_time > mean_time")
        
        # Check time-count relationship (should be negative correlation)
        if np.std(min_times) > 0 and np.std(count_60) > 0:
            time_count_corr = np.corrcoef(min_times, count_60)[0, 1]
            self.log(f"Min_time vs Count_60min correlation: {time_count_corr:.3f}")
            
            if time_count_corr > 0.1:
                self.log(f"WARNING: Positive time-count correlation ({time_count_corr:.3f})")
        
        # Check for perfect correlations
        corr_matrix = np.corrcoef(features.T)
        perfect_corrs = np.sum((np.abs(corr_matrix) > 0.99) & (corr_matrix != 1.0))
        if perfect_corrs > 0:
            self.log(f"WARNING: {perfect_corrs} perfect feature correlations detected")
    
    def compute_enhanced_derived_features(self, base_features: np.ndarray) -> np.ndarray:
        """
        UPDATED: Compute derived features focused on intra-tract patterns
        """
        n_addresses = base_features.shape[0]
        
        if base_features.shape[1] < 30:  # 3 destinations × 10 features each
            self.log(f"WARNING: Expected 30 base features, got {base_features.shape[1]}")
            return np.zeros((n_addresses, 4), dtype=np.float64)
        
        derived = []
        
        for i in range(n_addresses):
            # Extract key measures for each destination type (features 3, 13, 23 = count_5min)
            emp_5min = base_features[i, 3]   # employment_count_5min
            health_5min = base_features[i, 13] if base_features.shape[1] > 13 else 0
            grocery_5min = base_features[i, 23] if base_features.shape[1] > 23 else 0
            
            # 1. Local accessibility index (immediate access)
            local_access = emp_5min + health_5min + grocery_5min
            
            # 2. Modal flexibility (driving advantage) - features 6, 16, 26
            emp_drive_adv = base_features[i, 6]
            health_drive_adv = base_features[i, 16] if base_features.shape[1] > 16 else 0
            grocery_drive_adv = base_features[i, 26] if base_features.shape[1] > 26 else 0
            
            modal_flexibility = (emp_drive_adv + health_drive_adv + grocery_drive_adv) / 3
            
            # 3. Accessibility equity - features 9, 19, 29 (percentiles)
            emp_percentile = base_features[i, 9]
            health_percentile = base_features[i, 19] if base_features.shape[1] > 19 else 0.5
            grocery_percentile = base_features[i, 29] if base_features.shape[1] > 29 else 0.5
            
            percentiles = [emp_percentile, health_percentile, grocery_percentile]
            accessibility_equity = 1.0 - np.std(percentiles)  # Higher when access is consistent
            
            # 4. Geographic advantage (overall position)
            geographic_advantage = np.mean(percentiles)
            
            derived.append([
                local_access,
                modal_flexibility, 
                accessibility_equity,
                geographic_advantage
            ])
        
        derived_array = np.array(derived, dtype=np.float64)
        
        # Validate derived features
        self.log("=== DERIVED INTRA-TRACT FEATURE VALIDATION ===")
        derived_names = ['local_access', 'modal_flexibility', 'accessibility_equity', 'geographic_advantage']
        
        for i, name in enumerate(derived_names):
            values = derived_array[:, i]
            self.log(f"{name}: mean={np.mean(values):.3f}, std={np.std(values):.3f}")
        
        return derived_array

def debug_accessibility_computation(addresses: gpd.GeoDataFrame,
                                  destinations: Dict[str, gpd.GeoDataFrame],
                                  sample_size: int = 5) -> Dict:
    """
    Debug function to trace accessibility computation step by step
    """
    print("=== DEBUGGING ACCESSIBILITY COMPUTATION ===")
    
    # Sample addresses for detailed analysis
    sample_addresses = addresses.head(sample_size)
    print(f"Analyzing {len(sample_addresses)} sample addresses...")
    
    computer = EnhancedAccessibilityComputer(verbose=True)
    debug_results = {}
    
    for dest_type, dest_gdf in destinations.items():
        print(f"\n--- {dest_type.upper()} DESTINATIONS ---")
        print(f"Destination count: {len(dest_gdf)}")
        
        # Calculate travel times
        travel_times = computer.calculate_realistic_travel_times(
            sample_addresses, dest_gdf, time_period='morning'
        )
        
        # Show sample travel times
        print("\nSample travel times:")
        print(travel_times[['origin_id', 'destination_id', 'straight_distance_km', 
                           'walk_time', 'drive_time', 'transit_time', 'best_mode']].head(10))
        
        # Extract features
        features = computer.extract_enhanced_accessibility_features(
            sample_addresses, travel_times, dest_type
        )
        
        debug_results[dest_type] = {
            'travel_times': travel_times,
            'features': features,
            'feature_stats': {
                'mean': np.mean(features, axis=0),
                'std': np.std(features, axis=0),
                'min': np.min(features, axis=0),
                'max': np.max(features, axis=0)
            }
        }
    
    return debug_results