"""
GRANITE Disaggregation Baselines
Implements IDW, Kriging, and naive baselines for comparison with GNN disaggregation.

Key insight: For disaggregation, we're not predicting unknown values - we're
allocating a KNOWN aggregate (tract SVI) to addresses using spatial patterns.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class DisaggregationBaseline:
    """Base class for disaggregation methods."""
    
    def __init__(self, name: str):
        self.name = name
        self.fitted = False
    
    def fit(self, tract_gdf: gpd.GeoDataFrame, svi_column: str = 'RPL_THEMES'):
        """Fit baseline to tract-level data."""
        raise NotImplementedError
    
    def disaggregate(self, address_coords: np.ndarray, tract_fips: str, 
                     tract_svi: float) -> np.ndarray:
        """
        Disaggregate tract SVI to address level.
        
        Args:
            address_coords: Nx2 array of (lon, lat) coordinates
            tract_fips: FIPS code of tract being disaggregated
            tract_svi: Known tract-level SVI value
            
        Returns:
            Array of address-level SVI estimates (mean should equal tract_svi)
        """
        raise NotImplementedError


class NaiveUniformBaseline(DisaggregationBaseline):
    """
    Naive baseline: assign tract SVI uniformly to all addresses.
    
    This represents the "null hypothesis" - no within-tract variation.
    Any method claiming to add value must outperform this on spatial coherence.
    """
    
    def __init__(self):
        super().__init__('Naive_Uniform')
    
    def fit(self, tract_gdf: gpd.GeoDataFrame, svi_column: str = 'RPL_THEMES'):
        self.fitted = True
        return self
    
    def disaggregate(self, address_coords: np.ndarray, tract_fips: str,
                     tract_svi: float) -> np.ndarray:
        n_addresses = len(address_coords)
        return np.full(n_addresses, tract_svi)


class IDWDisaggregation(DisaggregationBaseline):
    """
    Inverse Distance Weighting disaggregation.
    
    Uses neighboring tract centroids to create spatial gradients within
    the target tract, then adjusts to satisfy constraint.
    
    Constraint-preserving: final predictions are scaled to ensure
    mean equals known tract SVI.
    """
    
    def __init__(self, power: float = 2.0, n_neighbors: int = 8):
        super().__init__(f'IDW_p{power}')
        self.power = power
        self.n_neighbors = n_neighbors
        self.tract_centroids = None
        self.tract_svi = None
        self.tract_fips = None
        self.kdtree = None
    
    def fit(self, tract_gdf: gpd.GeoDataFrame, svi_column: str = 'RPL_THEMES'):
        """Fit IDW using all tract centroids and SVI values."""
        
        # Store tract data
        self.tract_fips = tract_gdf['FIPS'].values
        self.tract_svi = tract_gdf[svi_column].values
        
        # Compute centroids
        self.tract_centroids = np.array([
            [geom.centroid.x, geom.centroid.y] 
            for geom in tract_gdf.geometry
        ])
        
        # Build KD-tree for fast neighbor lookup
        self.kdtree = cKDTree(self.tract_centroids)
        self.fitted = True
        
        return self
    
    def disaggregate(self, address_coords: np.ndarray, tract_fips: str,
                     tract_svi: float) -> np.ndarray:
        """
        Disaggregate using IDW interpolation from neighboring tracts.
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        n_addresses = len(address_coords)
        
        # Find neighbors (excluding target tract itself)
        target_idx = np.where(self.tract_fips == tract_fips)[0]
        
        # Get all neighboring tract data
        distances, indices = self.kdtree.query(
            address_coords, 
            k=min(self.n_neighbors + 1, len(self.tract_svi))
        )
        
        # Handle edge case of point exactly on centroid
        min_distance = 1e-10
        distances = np.maximum(distances, min_distance)
        
        # Compute IDW weights
        weights = 1.0 / (distances ** self.power)
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # Weighted average of neighbor SVI values
        raw_predictions = (weights * self.tract_svi[indices]).sum(axis=1)
        
        # Constraint enforcement: adjust to ensure mean equals tract SVI
        current_mean = np.mean(raw_predictions)
        if abs(current_mean) > 1e-8:
            # Scale to match tract constraint
            scaling_factor = tract_svi / current_mean
            adjusted_predictions = raw_predictions * scaling_factor
        else:
            adjusted_predictions = np.full(n_addresses, tract_svi)
        
        # Clip to valid SVI range [0, 1]
        adjusted_predictions = np.clip(adjusted_predictions, 0, 1)
        
        # Final mean adjustment (clipping may have shifted mean)
        mean_shift = tract_svi - np.mean(adjusted_predictions)
        adjusted_predictions += mean_shift
        adjusted_predictions = np.clip(adjusted_predictions, 0, 1)
        
        return adjusted_predictions


class OrdinaryKrigingDisaggregation(DisaggregationBaseline):
    """
    Ordinary Kriging disaggregation with exponential variogram.
    
    Uses spatial correlation structure from neighboring tracts
    to create spatially coherent within-tract variation.
    """
    
    def __init__(self, variogram_range: float = 5000.0, sill: float = 0.1, 
                 nugget: float = 0.01):
        super().__init__('Kriging')
        self.variogram_range = variogram_range # meters
        self.sill = sill
        self.nugget = nugget
        self.tract_centroids = None
        self.tract_svi = None
    
    def _exponential_variogram(self, h: np.ndarray) -> np.ndarray:
        """Exponential variogram model."""
        return self.nugget + self.sill * (1 - np.exp(-3 * h / self.variogram_range))
    
    def fit(self, tract_gdf: gpd.GeoDataFrame, svi_column: str = 'RPL_THEMES'):
        """Fit kriging model to tract data."""
        
        self.tract_centroids = np.array([
            [geom.centroid.x, geom.centroid.y] 
            for geom in tract_gdf.geometry
        ])
        self.tract_svi = tract_gdf[svi_column].values
        self.fitted = True
        
        return self
    
    def disaggregate(self, address_coords: np.ndarray, tract_fips: str,
                     tract_svi: float) -> np.ndarray:
        """Disaggregate using ordinary kriging."""
        
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        n_addresses = len(address_coords)
        n_tracts = len(self.tract_centroids)
        
        # Compute distance matrices
        # Convert coordinates to approximate meters (rough conversion)
        lat_center = np.mean(address_coords[:, 1])
        meters_per_deg_lon = 111320 * np.cos(np.radians(lat_center))
        meters_per_deg_lat = 110540
        
        # Scale coordinates to meters
        address_coords_m = address_coords.copy()
        address_coords_m[:, 0] *= meters_per_deg_lon
        address_coords_m[:, 1] *= meters_per_deg_lat
        
        tract_coords_m = self.tract_centroids.copy()
        tract_coords_m[:, 0] *= meters_per_deg_lon
        tract_coords_m[:, 1] *= meters_per_deg_lat
        
        # Tract-to-tract covariance matrix
        tract_distances = cdist(tract_coords_m, tract_coords_m)
        C = self.sill - self._exponential_variogram(tract_distances)
        
        # Address-to-tract covariance matrix
        addr_tract_distances = cdist(address_coords_m, tract_coords_m)
        c = self.sill - self._exponential_variogram(addr_tract_distances)
        
        # Set up kriging system (with Lagrange multiplier for unbiasedness)
        n = n_tracts
        K = np.zeros((n + 1, n + 1))
        K[:n, :n] = C
        K[n, :n] = 1
        K[:n, n] = 1
        K[n, n] = 0
        
        # Regularization for numerical stability
        K[:n, :n] += np.eye(n) * 1e-6
        
        predictions = np.zeros(n_addresses)
        
        for i in range(n_addresses):
            k = np.zeros(n + 1)
            k[:n] = c[i, :]
            k[n] = 1
            
            try:
                weights = np.linalg.solve(K, k)
                predictions[i] = np.dot(weights[:n], self.tract_svi)
            except np.linalg.LinAlgError:
                # Fallback to IDW-like behavior
                predictions[i] = tract_svi
        
        # Constraint enforcement
        current_mean = np.mean(predictions)
        if abs(current_mean) > 1e-8:
            scaling_factor = tract_svi / current_mean
            predictions = predictions * scaling_factor
        else:
            predictions = np.full(n_addresses, tract_svi)
        
        # Clip to valid range
        predictions = np.clip(predictions, 0, 1)
        
        # Final mean adjustment
        mean_shift = tract_svi - np.mean(predictions)
        predictions += mean_shift
        predictions = np.clip(predictions, 0, 1)
        
        return predictions


class DisaggregationComparison:
    """
    Compare GNN disaggregation against traditional baselines.
    
    Key metrics for disaggregation (not prediction):
    - Constraint satisfaction: |mean(predictions) - tract_svi|
    - Spatial coherence: Moran's I of residuals from uniform
    - Variation appropriateness: std of predictions
    - Pattern validity: correlation with accessibility features
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.baselines = {}
        self.results = {}
    
    def add_baseline(self, baseline: DisaggregationBaseline):
        """Register a baseline method."""
        self.baselines[baseline.name] = baseline
    
    def run_comparison(self, 
                       tract_gdf: gpd.GeoDataFrame,
                       address_gdf: gpd.GeoDataFrame,
                       gnn_predictions: np.ndarray,
                       tract_fips: str,
                       tract_svi: float,
                       accessibility_features: np.ndarray = None,
                       svi_column: str = 'RPL_THEMES') -> Dict:
        """
        Run comparison of GNN against baselines.
        
        Args:
            tract_gdf: All tract geometries with SVI
            address_gdf: Address points within target tract
            gnn_predictions: GNN-predicted SVI values
            tract_fips: Target tract FIPS code
            tract_svi: Known tract-level SVI
            accessibility_features: Optional accessibility features for correlation
            svi_column: Column name for SVI values
            
        Returns:
            Dictionary of comparison results
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print("DISAGGREGATION BASELINE COMPARISON")
            print(f"{'='*60}")
            print(f"Target tract: {tract_fips}")
            print(f"Known tract SVI: {tract_svi:.4f}")
            print(f"Addresses: {len(address_gdf)}")
        
        # Extract coordinates
        address_coords = np.array([
            [geom.x, geom.y] for geom in address_gdf.geometry
        ])
        
        results = {
            'tract_fips': tract_fips,
            'tract_svi': tract_svi,
            'n_addresses': len(address_gdf),
            'methods': {}
        }
        
        # Add default baselines if none registered
        if not self.baselines:
            self.add_baseline(NaiveUniformBaseline())
            self.add_baseline(IDWDisaggregation(power=2.0, n_neighbors=8))
            self.add_baseline(OrdinaryKrigingDisaggregation())
        
        # Fit all baselines
        for name, baseline in self.baselines.items():
            baseline.fit(tract_gdf, svi_column)
        
        # GNN results
        results['methods']['GNN'] = self._evaluate_method(
            'GNN', gnn_predictions, tract_svi, accessibility_features
        )
        
        # Baseline results
        for name, baseline in self.baselines.items():
            if self.verbose:
                print(f"\n Running {name}...")
            
            predictions = baseline.disaggregate(address_coords, tract_fips, tract_svi)
            results['methods'][name] = self._evaluate_method(
                name, predictions, tract_svi, accessibility_features
            )
        
        # Compute comparative metrics
        results['comparison'] = self._compute_comparison_metrics(results['methods'])
        
        self.results = results
        return results
    
    def _evaluate_method(self, name: str, predictions: np.ndarray, 
                        tract_svi: float, 
                        accessibility_features: np.ndarray = None) -> Dict:
        """Evaluate a single disaggregation method."""
        
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        pred_range = np.ptp(predictions)
        
        # Constraint satisfaction
        constraint_error_abs = abs(pred_mean - tract_svi)
        constraint_error_pct = constraint_error_abs / tract_svi * 100 if tract_svi > 0 else 0
        
        # Spatial variation metrics
        cv = pred_std / pred_mean if pred_mean > 0 else 0 # Coefficient of variation
        
        # Accessibility correlation (if available)
        access_corr = None
        if accessibility_features is not None and len(accessibility_features) == len(predictions):
            # Use mean accessibility as summary
            access_summary = np.mean(accessibility_features, axis=1)
            if np.std(access_summary) > 0 and np.std(predictions) > 0:
                access_corr = float(np.corrcoef(access_summary, predictions)[0, 1])
        
        result = {
            'predictions': predictions,
            'mean': pred_mean,
            'std': pred_std,
            'range': pred_range,
            'cv': cv,
            'constraint_error_abs': constraint_error_abs,
            'constraint_error_pct': constraint_error_pct,
            'accessibility_correlation': access_corr
        }
        
        if self.verbose:
            print(f"    {name}: mean={pred_mean:.4f}, std={pred_std:.4f}, "
                  f"constraint_err={constraint_error_pct:.2f}%")
        
        return result
    
    def _compute_comparison_metrics(self, methods: Dict) -> Dict:
        """Compute comparative metrics across methods."""
        
        # Rankings
        constraint_ranking = sorted(
            methods.keys(), 
            key=lambda k: methods[k]['constraint_error_pct']
        )
        
        variation_ranking = sorted(
            methods.keys(),
            key=lambda k: -methods[k]['std'] # Higher variation is better for disaggregation
        )
        
        # GNN vs baselines improvement
        gnn_results = methods.get('GNN', {})
        naive_results = methods.get('Naive_Uniform', {})
        idw_results = methods.get('IDW_p2.0', methods.get('IDW_p2', {}))
        
        comparison = {
            'constraint_ranking': constraint_ranking,
            'variation_ranking': variation_ranking,
            'gnn_variation_vs_naive': gnn_results.get('std', 0) - naive_results.get('std', 0),
            'gnn_variation_vs_idw': gnn_results.get('std', 0) - idw_results.get('std', 0) if idw_results else None,
        }
        
        # Summary statistics
        gnn_access_corr = gnn_results.get('accessibility_correlation')
        idw_access_corr = idw_results.get('accessibility_correlation') if idw_results else None
        
        if gnn_access_corr is not None and idw_access_corr is not None:
            comparison['gnn_accessibility_advantage'] = abs(gnn_access_corr) - abs(idw_access_corr)
        
        return comparison
    
    def print_summary(self):
        """Print formatted summary of comparison results."""
        
        if not self.results:
            print("No results available. Run comparison first.")
            return
        
        print(f"\n{'='*70}")
        print("DISAGGREGATION COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"Target Tract: {self.results['tract_fips']}")
        print(f"Known SVI: {self.results['tract_svi']:.4f}")
        print(f"Addresses: {self.results['n_addresses']}")
        
        print(f"\n{'Method':<20} {'Mean':<10} {'Std':<10} {'Err %':<10} {'Access r':<10}")
        print("-" * 60)
        
        methods = self.results['methods']
        for name in ['GNN', 'Naive_Uniform', 'IDW_p2.0', 'IDW_p2', 'Kriging']:
            if name in methods:
                m = methods[name]
                access_r = f"{m['accessibility_correlation']:.3f}" if m['accessibility_correlation'] else "N/A"
                print(f"{name:<20} {m['mean']:<10.4f} {m['std']:<10.4f} "
                      f"{m['constraint_error_pct']:<10.2f} {access_r:<10}")
        
        print(f"\n{'='*70}")
        print("INTERPRETATION")
        print("-" * 70)
        
        comparison = self.results.get('comparison', {})
        
        # Variation advantage
        var_vs_naive = comparison.get('gnn_variation_vs_naive', 0)
        if var_vs_naive > 0.01:
            print(f"+ GNN produces {var_vs_naive:.4f} MORE spatial variation than naive uniform")
            print(" (Evidence of meaningful within-tract disaggregation)")
        elif var_vs_naive < -0.01:
            print(f"- GNN produces {-var_vs_naive:.4f} LESS variation than naive")
            print(" (Consider: is the tract truly homogeneous?)")
        else:
            print("~ GNN variation similar to naive baseline")
        
        # Accessibility correlation
        gnn_corr = methods.get('GNN', {}).get('accessibility_correlation')
        if gnn_corr is not None:
            if abs(gnn_corr) > 0.3:
                direction = "negative" if gnn_corr < 0 else "positive"
                print(f"+ Strong {direction} accessibility-SVI correlation (r={gnn_corr:.3f})")
                print(" (GNN disaggregation reflects accessibility patterns)")
            elif abs(gnn_corr) > 0.1:
                print(f"~ Moderate accessibility correlation (r={gnn_corr:.3f})")
            else:
                print(f"- Weak accessibility correlation (r={gnn_corr:.3f})")
        
        print(f"{'='*70}")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for further analysis."""
        
        if not self.results:
            return pd.DataFrame()
        
        rows = []
        for method_name, method_results in self.results['methods'].items():
            row = {
                'method': method_name,
                'tract_fips': self.results['tract_fips'],
                'tract_svi': self.results['tract_svi'],
                'pred_mean': method_results['mean'],
                'pred_std': method_results['std'],
                'pred_range': method_results['range'],
                'constraint_error_pct': method_results['constraint_error_pct'],
                'accessibility_correlation': method_results['accessibility_correlation']
            }
            rows.append(row)
        
        return pd.DataFrame(rows)


def run_disaggregation_baselines(
    tract_gdf: gpd.GeoDataFrame,
    address_gdf: gpd.GeoDataFrame, 
    gnn_predictions: np.ndarray,
    tract_fips: str,
    tract_svi: float,
    accessibility_features: np.ndarray = None,
    verbose: bool = True
) -> Dict:
    """
    Convenience function to run full disaggregation comparison.
    
    Returns:
        Dict with comparison results and summary metrics
    """
    comparison = DisaggregationComparison(verbose=verbose)
    
    # Register baselines
    comparison.add_baseline(NaiveUniformBaseline())
    comparison.add_baseline(IDWDisaggregation(power=2.0, n_neighbors=8))
    comparison.add_baseline(IDWDisaggregation(power=3.0, n_neighbors=8))
    comparison.add_baseline(OrdinaryKrigingDisaggregation())
    
    # Run comparison
    results = comparison.run_comparison(
        tract_gdf=tract_gdf,
        address_gdf=address_gdf,
        gnn_predictions=gnn_predictions,
        tract_fips=tract_fips,
        tract_svi=tract_svi,
        accessibility_features=accessibility_features
    )
    
    if verbose:
        comparison.print_summary()
    
    return results