"""
Spatial Feature Computation for GRANITE

Computes minimal spatial features for disaggregation:
- Normalized coordinates (position within tract)
- Distance to tract boundary (edge vs interior)
- Local address density (urban vs rural patterns)
"""
import numpy as np
import geopandas as gpd
from typing import Dict, List, Tuple, Optional
from scipy.spatial import cKDTree


class SpatialFeatureComputer:
    """
    Computes spatial features for address-level disaggregation.
    
    Features are designed to capture spatial position and context
    without requiring routing or destination data.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.feature_names = []
    
    def _log(self, message: str):
        if self.verbose:
            print(f"[SpatialFeatures] {message}")
    
    def compute_features(self, 
                        addresses: gpd.GeoDataFrame,
                        tract_geometry,
                        include_density: bool = True,
                        include_boundary: bool = True,
                        data_loader=None) -> Tuple[np.ndarray, List[str]]:
        """
        Compute spatial features for all addresses.
        
        Args:
            addresses: GeoDataFrame with address points
            tract_geometry: Shapely geometry of the tract boundary
            include_density: Include local density feature
            include_boundary: Include distance to boundary feature
        
        Returns:
            Tuple of (features array [n_addresses, n_features], feature_names list)
        """
        n_addresses = len(addresses)
        self._log(f"Computing spatial features for {n_addresses} addresses")
        
        # Extract coordinates
        coords = np.column_stack([
            addresses.geometry.x.values,
            addresses.geometry.y.values
        ])
        
        # Get tract bounds for normalization
        bounds = tract_geometry.bounds  # (minx, miny, maxx, maxy)
        tract_center = tract_geometry.centroid
        tract_width = bounds[2] - bounds[0]
        tract_height = bounds[3] - bounds[1]
        
        features = []
        feature_names = []
        
        # 1. Normalized coordinates (always included)
        x_norm = (coords[:, 0] - tract_center.x) / max(tract_width, 1e-6)
        y_norm = (coords[:, 1] - tract_center.y) / max(tract_height, 1e-6)
        
        features.append(x_norm)
        features.append(y_norm)
        feature_names.extend(['x_normalized', 'y_normalized'])
        
        # 2. Distance to tract centroid (radial position)
        dist_to_centroid = np.sqrt(x_norm**2 + y_norm**2)
        features.append(dist_to_centroid)
        feature_names.append('dist_to_centroid')
        
        # 3. Distance to tract boundary (edge detection)
        if include_boundary:
            boundary_dist = self._compute_boundary_distances(
                addresses, tract_geometry, tract_width, tract_height
            )
            features.append(boundary_dist)
            feature_names.append('dist_to_boundary')
        
        # 4. Local address density
        if include_density:
            density = self._compute_local_density(coords, tract_width)
            features.append(density)
            feature_names.append('local_density')
        
        # 5. Spatial percentile (rank-based position)
        # Addresses sorted by distance to centroid, normalized to [0, 1]
        percentile = np.argsort(np.argsort(dist_to_centroid)) / max(n_addresses - 1, 1)
        features.append(percentile)
        feature_names.append('spatial_percentile')
        
        # Stack into feature matrix
        feature_matrix = np.column_stack(features)
        
        # 6. Accessibility features (if data_loader provided)
        if data_loader is not None:
            try:
                self._log("Computing accessibility features...")
                accessibility_features = data_loader.compute_simple_accessibility_features(addresses)
                
                accessibility_names = [
                    'employment_min_time', 'employment_mean_time', 'employment_median_time',
                    'employment_count_5min', 'employment_count_10min', 'employment_count_15min',
                    'healthcare_min_time', 'healthcare_mean_time', 'healthcare_median_time',
                    'healthcare_count_5min', 'healthcare_count_10min', 'healthcare_count_15min',
                    'grocery_min_time', 'grocery_mean_time', 'grocery_median_time',
                    'grocery_count_5min', 'grocery_count_10min', 'grocery_count_15min',
                ]
                
                feature_matrix = np.column_stack([feature_matrix, accessibility_features])
                feature_names.extend(accessibility_names)
                
                self._log(f"Added {len(accessibility_names)} accessibility features")
            except Exception as e:
                self._log(f"Accessibility features failed: {e}, using spatial only")
        
        self._log(f"Computed {len(feature_names)} features: {feature_names}")
        self._log(f"Feature matrix shape: {feature_matrix.shape}")
        
        # Validate
        if np.any(np.isnan(feature_matrix)):
            nan_count = np.sum(np.isnan(feature_matrix))
            self._log(f"WARNING: {nan_count} NaN values in features, replacing with 0")
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)

        self.feature_names = feature_names
        return feature_matrix, feature_names
    
    def _compute_boundary_distances(self, 
                                    addresses: gpd.GeoDataFrame,
                                    tract_geometry,
                                    tract_width: float,
                                    tract_height: float) -> np.ndarray:
        """
        Compute normalized distance from each address to tract boundary.
        
        Returns values in [0, 1] where:
        - 0 = on boundary (edge)
        - 1 = at centroid (interior)
        """
        boundary = tract_geometry.boundary
        
        distances = []
        for point in addresses.geometry:
            dist = point.distance(boundary)
            distances.append(dist)
        
        distances = np.array(distances)
        
        # Normalize by tract size
        max_possible_dist = min(tract_width, tract_height) / 2
        normalized = distances / max(max_possible_dist, 1e-6)
        normalized = np.clip(normalized, 0, 1)
        
        return normalized
    
    def _compute_local_density(self, 
                               coords: np.ndarray,
                               tract_width: float,
                               k_neighbors: int = 10) -> np.ndarray:
        """
        Compute local address density using k-nearest neighbors.
        
        Higher values = more addresses nearby = more urban/dense.
        """
        n_addresses = len(coords)
        
        if n_addresses < k_neighbors + 1:
            # Not enough addresses for meaningful density
            return np.ones(n_addresses) * 0.5
        
        # Build KD-tree for efficient neighbor search
        tree = cKDTree(coords)
        
        # Query k+1 neighbors (includes self)
        distances, _ = tree.query(coords, k=min(k_neighbors + 1, n_addresses))
        
        # Use mean distance to k neighbors as inverse density proxy
        # Exclude self (first column)
        mean_distances = distances[:, 1:].mean(axis=1)
        
        # Convert to density (inverse, normalized)
        # Smaller distance = higher density
        max_dist = mean_distances.max()
        if max_dist > 0:
            density = 1.0 - (mean_distances / max_dist)
        else:
            density = np.ones(n_addresses) * 0.5
        
        return density
    
    def compute_multi_tract_features(self,
                                     addresses: gpd.GeoDataFrame,
                                     tracts: gpd.GeoDataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Compute features for addresses spanning multiple tracts.
        
        Adds tract-relative positioning to handle cross-tract patterns.
        """
        # Get tract assignment for each address
        if 'tract_fips' not in addresses.columns:
            raise ValueError("Addresses must have 'tract_fips' column for multi-tract mode")
        
        all_features = []
        
        # Compute per-tract features
        for fips in addresses['tract_fips'].unique():
            tract_mask = addresses['tract_fips'] == fips
            tract_addresses = addresses[tract_mask]
            
            tract_geom = tracts[tracts['FIPS'] == fips].geometry.iloc[0]
            
            tract_features, feature_names = self.compute_features(
                tract_addresses, tract_geom
            )
            
            # Store with original indices
            for i, idx in enumerate(tract_addresses.index):
                all_features.append((idx, tract_features[i]))
        
        # Sort by original index and stack
        all_features.sort(key=lambda x: x[0])
        feature_matrix = np.array([f[1] for f in all_features])
        
        return feature_matrix, feature_names


def normalize_spatial_features(features: np.ndarray, 
                               method: str = 'robust') -> Tuple[np.ndarray, object]:
    """
    Normalize spatial features for GNN input.
    
    Args:
        features: Raw feature matrix [n_addresses, n_features]
        method: 'robust' (default) or 'standard'
    
    Returns:
        Tuple of (normalized_features, scaler)
    """
    if method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    
    normalized = scaler.fit_transform(features)
    
    # Handle edge cases
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    
    return normalized, scaler