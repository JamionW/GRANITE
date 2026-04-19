"""
GRANITE Disaggregation Baselines
Mass-preserving disaggregation baselines (dasymetric, pycnophylactic, naive uniform).

Key insight: For disaggregation, we're not predicting unknown values - we're
allocating a KNOWN aggregate (tract SVI) to addresses using spatial patterns.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


def _enforce_constraint(predictions: np.ndarray, target_mean: float,
                        max_iterations: int = 20, tol: float = 1e-6) -> np.ndarray:
    """Iteratively enforce mean constraint while keeping values in [0, 1].

    Alternates shift-to-mean and clip-to-bounds until convergence.
    """
    preds = predictions.copy()
    for _ in range(max_iterations):
        current_mean = np.mean(preds)
        if abs(current_mean - target_mean) < tol:
            break
        preds += (target_mean - current_mean)
        preds = np.clip(preds, 0, 1)
    return preds


class DisaggregationBaseline:
    """Base class for disaggregation methods."""

    def __init__(self, name: str):
        self.name = name
        self.fitted = False

    def fit(self, tract_gdf: gpd.GeoDataFrame, svi_column: str = 'RPL_THEMES'):
        """Fit baseline to tract-level data."""
        raise NotImplementedError

    def disaggregate(self, address_coords: np.ndarray, tract_fips: str,
                     tract_svi: float, address_gdf: gpd.GeoDataFrame = None) -> np.ndarray:
        """
        Disaggregate tract SVI to address level.

        Args:
            address_coords: Nx2 array of (lon, lat) coordinates
            tract_fips: FIPS code of tract being disaggregated
            tract_svi: Known tract-level SVI value
            address_gdf: optional address GeoDataFrame for ancillary columns

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
                     tract_svi: float, address_gdf: gpd.GeoDataFrame = None) -> np.ndarray:
        n_addresses = len(address_coords)
        return np.full(n_addresses, tract_svi)


class DasymetricDisaggregation(DisaggregationBaseline):
    """
    Additive dasymetric disaggregation using an ancillary surface variable.

    Allocates a known tract-level rate (e.g. SVI) to addresses proportional
    to deviation from the within-tract mean of the ancillary column
    (default: NLCD impervious surface percentage). Alpha is chosen as the
    largest multiplier that keeps all predictions in [0, 1].

    Mass preservation: mean(predictions) == tract_svi by construction
    since mean(dev) == 0.
    """

    def __init__(self, ancillary_column: str = 'nlcd_impervious_pct'):
        super().__init__('Dasymetric')
        self.ancillary_column = ancillary_column

    def fit(self, tract_gdf: gpd.GeoDataFrame, svi_column: str = 'RPL_THEMES'):
        self.fitted = True
        return self

    def disaggregate(self, address_coords: np.ndarray, tract_fips: str,
                     tract_svi: float, address_gdf: gpd.GeoDataFrame = None) -> np.ndarray:
        n = len(address_coords)

        if address_gdf is None or self.ancillary_column not in address_gdf.columns:
            # uniform fallback when ancillary data unavailable
            return np.full(n, tract_svi)

        a = pd.to_numeric(address_gdf[self.ancillary_column], errors='coerce').fillna(0.0).values
        dev = a - a.mean()

        if np.ptp(dev) < 1e-9 or a.sum() < 1e-9:
            predictions = np.full(n, tract_svi)
        else:
            max_up = (1.0 - tract_svi) / dev.max() if dev.max() > 0 else np.inf
            max_down = tract_svi / abs(dev.min()) if dev.min() < 0 else np.inf
            alpha = min(max_up, max_down)
            predictions = tract_svi + alpha * dev

        return np.clip(predictions, 0, 1)


class PycnophylacticDisaggregation(DisaggregationBaseline):
    """
    Pycnophylactic interpolation (Tobler 1979).

    Iterative spatial smoothing under hard aggregate constraint. No features
    consulted; pure geometric smoothing using k-nearest-neighbor adjacency
    within the tract. Tract boundaries act as discontinuities.
    """

    def __init__(self, n_iterations: int = 50, k_neighbors: int = 8):
        super().__init__('Pycnophylactic')
        self.n_iterations = n_iterations
        self.k_neighbors = k_neighbors

    def fit(self, tract_gdf: gpd.GeoDataFrame, svi_column: str = 'RPL_THEMES'):
        # store multi-tract context for seeding spatial gradient (drop NaN SVIs)
        svi_vals = tract_gdf[svi_column].values.astype(float)
        valid = ~np.isnan(svi_vals)
        self._tract_centroids = np.array([
            [g.centroid.x, g.centroid.y]
            for g, v in zip(tract_gdf.geometry, valid) if v
        ])
        self._tract_svi_values = svi_vals[valid]
        self.fitted = True
        return self

    def disaggregate(self, address_coords: np.ndarray, tract_fips: str,
                     tract_svi: float, address_gdf: gpd.GeoDataFrame = None) -> np.ndarray:
        n = len(address_coords)
        if n <= 1:
            return np.full(n, tract_svi)

        # seed initial surface from neighboring tract gradient
        if hasattr(self, '_tract_centroids') and len(self._tract_centroids) > 1:
            diffs = address_coords[:, np.newaxis, :] - self._tract_centroids[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=2))
            weights = 1.0 / np.maximum(dists, 1e-10) ** 2
            weights = weights / weights.sum(axis=1, keepdims=True)
            predictions = (weights * self._tract_svi_values).sum(axis=1)
            predictions = predictions - predictions.mean() + tract_svi
        else:
            predictions = np.full(n, tract_svi)

        # build k-NN adjacency (within-tract only by construction)
        k = min(self.k_neighbors, n - 1)
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(address_coords)
        _, indices = nn.kneighbors(address_coords)
        neighbor_idx = indices[:, 1:]  # exclude self

        for _ in range(self.n_iterations):
            smoothed = np.mean(predictions[neighbor_idx], axis=1)
            # re-center to preserve tract aggregate
            smoothed = smoothed - smoothed.mean() + tract_svi
            predictions = smoothed

        predictions = np.clip(predictions, 0, 1)
        # final mean-preservation pass after clipping
        predictions = predictions - predictions.mean() + tract_svi
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
            self.add_baseline(DasymetricDisaggregation())
            self.add_baseline(PycnophylacticDisaggregation())
        
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
                print(f"\n  Running {name}...")
            
            predictions = baseline.disaggregate(address_coords, tract_fips, tract_svi, address_gdf=address_gdf)
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
        cv = pred_std / pred_mean if pred_mean > 0 else 0  # Coefficient of variation
        
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
            key=lambda k: -methods[k]['std']  # Higher variation is better for disaggregation
        )
        
        # GNN vs baselines improvement
        gnn_results = methods.get('GNN', {})
        naive_results = methods.get('Naive_Uniform', {})
        dasymetric_results = methods.get('Dasymetric', {})

        comparison = {
            'constraint_ranking': constraint_ranking,
            'variation_ranking': variation_ranking,
            'gnn_variation_vs_naive': gnn_results.get('std', 0) - naive_results.get('std', 0),
            'gnn_variation_vs_dasymetric': gnn_results.get('std', 0) - dasymetric_results.get('std', 0) if dasymetric_results else None,
        }

        # Summary statistics
        gnn_access_corr = gnn_results.get('accessibility_correlation')
        dasy_access_corr = dasymetric_results.get('accessibility_correlation') if dasymetric_results else None

        if gnn_access_corr is not None and dasy_access_corr is not None:
            comparison['gnn_accessibility_advantage'] = abs(gnn_access_corr) - abs(dasy_access_corr)
        
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
        for name in ['GNN', 'Naive_Uniform', 'Dasymetric', 'Pycnophylactic']:
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
            print("  (Evidence of meaningful within-tract disaggregation)")
        elif var_vs_naive < -0.01:
            print(f"- GNN produces {-var_vs_naive:.4f} LESS variation than naive")
            print("  (Consider: is the tract truly homogeneous?)")
        else:
            print("~ GNN variation similar to naive baseline")
        
        # Accessibility correlation
        gnn_corr = methods.get('GNN', {}).get('accessibility_correlation')
        if gnn_corr is not None:
            if abs(gnn_corr) > 0.3:
                direction = "negative" if gnn_corr < 0 else "positive"
                print(f"+ Strong {direction} accessibility-SVI correlation (r={gnn_corr:.3f})")
                print("  (GNN disaggregation reflects accessibility patterns)")
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
    comparison.add_baseline(DasymetricDisaggregation())
    comparison.add_baseline(PycnophylacticDisaggregation())
    
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