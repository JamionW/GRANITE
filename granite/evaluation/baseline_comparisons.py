"""
Baseline Comparison Module for GRANITE
Implements traditional spatial interpolation methods (IDW, Kriging) for comparison with GNN approach.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class InverseDistanceWeighting:
    """
    Inverse Distance Weighting (IDW) spatial interpolation.
    Predicts address-level SVI from tract-level SVI using distance weighting.
    """
    
    def __init__(self, power: float = 2.0, n_neighbors: int = 8):
        """
        Args:
            power: Exponent for distance weighting (typical: 2.0)
            n_neighbors: Number of nearest neighbors to consider
        """
        self.power = power
        self.n_neighbors = n_neighbors
        self.tract_centroids = None
        self.tract_svi = None
        self.kdtree = None
        
    def fit(self, tract_gdf: gpd.GeoDataFrame, svi_column: str = 'SVI'):
        """
        Fit IDW model using tract-level SVI data.
        
        Args:
            tract_gdf: GeoDataFrame with tract geometries and SVI values
            svi_column: Column name for SVI values
        """
        self.tract_centroids = np.array([
            [geom.centroid.x, geom.centroid.y] 
            for geom in tract_gdf.geometry
        ])
        self.tract_svi = tract_gdf[svi_column].values
        self.kdtree = cKDTree(self.tract_centroids)
        
    def predict(self, address_coords: np.ndarray) -> np.ndarray:
        """
        Predict SVI for address points using IDW.
        
        Args:
            address_coords: Nx2 array of (lon, lat) coordinates
            
        Returns:
            Array of predicted SVI values
        """
        if self.kdtree is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        distances, indices = self.kdtree.query(
            address_coords, 
            k=min(self.n_neighbors, len(self.tract_svi))
        )
        
        # Handle points exactly on tract centroids (avoid division by zero)
        min_distance = 1e-10
        distances = np.maximum(distances, min_distance)
        
        # Compute IDW weights
        weights = 1.0 / (distances ** self.power)
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # Weighted average of neighbor SVI values
        predictions = (weights * self.tract_svi[indices]).sum(axis=1)
        
        return predictions


class OrdinaryKriging:
    """
    Ordinary Kriging spatial interpolation with accessibility covariates.
    Simple implementation using exponential variogram model.
    """
    
    def __init__(self, variogram_range: float = 5000.0, sill: float = 1.0, 
                 nugget: float = 0.1, use_covariates: bool = False):
        """
        Args:
            variogram_range: Range parameter for exponential variogram (meters)
            sill: Sill parameter (total variance)
            nugget: Nugget effect (measurement error variance)
            use_covariates: Whether to use accessibility features as covariates
        """
        self.variogram_range = variogram_range
        self.sill = sill
        self.nugget = nugget
        self.use_covariates = use_covariates
        
        self.tract_centroids = None
        self.tract_svi = None
        self.tract_features = None
        
    def _variogram(self, distances: np.ndarray) -> np.ndarray:
        """Exponential variogram model."""
        return self.nugget + self.sill * (
            1 - np.exp(-3 * distances / self.variogram_range)
        )
    
    def fit(self, tract_gdf: gpd.GeoDataFrame, svi_column: str = 'SVI',
            feature_columns: List[str] = None):
        """
        Fit kriging model using tract-level data.
        
        Args:
            tract_gdf: GeoDataFrame with tract geometries and SVI values
            svi_column: Column name for SVI values
            feature_columns: Accessibility feature columns (if use_covariates=True)
        """
        self.tract_centroids = np.array([
            [geom.centroid.x, geom.centroid.y] 
            for geom in tract_gdf.geometry
        ])
        self.tract_svi = tract_gdf[svi_column].values
        
        if self.use_covariates and feature_columns:
            self.tract_features = tract_gdf[feature_columns].values
        
    def predict(self, address_coords: np.ndarray, 
                address_features: np.ndarray = None) -> np.ndarray:
        """
        Predict SVI for address points using ordinary kriging.
        
        Args:
            address_coords: Nx2 array of (lon, lat) coordinates
            address_features: NxF array of accessibility features (if use_covariates)
            
        Returns:
            Array of predicted SVI values
        """
        if self.tract_centroids is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        n_addresses = len(address_coords)
        n_tracts = len(self.tract_centroids)
        predictions = np.zeros(n_addresses)
        
        # Compute distance matrices
        tract_distances = cdist(self.tract_centroids, self.tract_centroids)
        
        # Build kriging system for each address
        for i, addr_coord in enumerate(address_coords):
            # Distances from address to all tract centroids
            addr_distances = cdist([addr_coord], self.tract_centroids)[0]
            
            # Build kriging matrix (tract-to-tract variogram + Lagrange multiplier)
            K = np.zeros((n_tracts + 1, n_tracts + 1))
            K[:n_tracts, :n_tracts] = self._variogram(tract_distances)
            K[n_tracts, :n_tracts] = 1
            K[:n_tracts, n_tracts] = 1
            K[n_tracts, n_tracts] = 0
            
            # Right-hand side (address-to-tract variogram + Lagrange constraint)
            k = np.zeros(n_tracts + 1)
            k[:n_tracts] = self._variogram(addr_distances)
            k[n_tracts] = 1
            
            # Solve kriging system
            try:
                weights = np.linalg.solve(K, k)
                predictions[i] = np.dot(weights[:n_tracts], self.tract_svi)
            except np.linalg.LinAlgError:
                # Fallback to simple inverse distance if kriging fails
                distances = np.maximum(addr_distances, 1e-10)
                weights = 1.0 / (distances ** 2)
                weights = weights / weights.sum()
                predictions[i] = np.dot(weights, self.tract_svi)
        
        # Apply covariate adjustment if enabled
        if self.use_covariates and address_features is not None:
            # Simple linear adjustment based on feature differences
            # This is a simplified version - full kriging with covariates is more complex
            pass
            
        return predictions


class BaselineComparison:
    """
    Comprehensive comparison framework for GRANITE vs traditional methods.
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def run_comparison(self, 
                      tract_gdf: gpd.GeoDataFrame,
                      address_gdf: gpd.GeoDataFrame,
                      gnn_predictions: np.ndarray,
                      svi_column: str = 'SVI',
                      accessibility_features: pd.DataFrame = None) -> Dict:
        """
        Run comprehensive comparison of GNN against baseline methods.
        
        Args:
            tract_gdf: GeoDataFrame with tract boundaries and SVI
            address_gdf: GeoDataFrame with address points and true SVI
            gnn_predictions: Array of GNN-predicted SVI values
            svi_column: Column name for SVI values
            accessibility_features: DataFrame with accessibility features for addresses
            
        Returns:
            Dictionary of comparison results and metrics
        """
        print("Running baseline comparisons...")
        
        # Extract coordinates and true SVI
        address_coords = np.array([
            [geom.x, geom.y] for geom in address_gdf.geometry
        ])
        true_svi = address_gdf[svi_column].values
        
        results = {
            'true_svi': true_svi,
            'gnn_predictions': gnn_predictions,
            'methods': {}
        }
        
        # IDW with different parameters
        print("  IDW (power=2.0)...")
        idw_2 = InverseDistanceWeighting(power=2.0, n_neighbors=8)
        idw_2.fit(tract_gdf, svi_column)
        idw_2_pred = idw_2.predict(address_coords)
        results['methods']['IDW_p2'] = {
            'predictions': idw_2_pred,
            'params': {'power': 2.0, 'n_neighbors': 8}
        }
        
        print("  IDW (power=3.0)...")
        idw_3 = InverseDistanceWeighting(power=3.0, n_neighbors=8)
        idw_3.fit(tract_gdf, svi_column)
        idw_3_pred = idw_3.predict(address_coords)
        results['methods']['IDW_p3'] = {
            'predictions': idw_3_pred,
            'params': {'power': 3.0, 'n_neighbors': 8}
        }
        
        # Ordinary Kriging
        print("  Ordinary Kriging...")
        ok = OrdinaryKriging(variogram_range=5000.0, sill=1.0, nugget=0.1)
        ok.fit(tract_gdf, svi_column)
        ok_pred = ok.predict(address_coords)
        results['methods']['Kriging'] = {
            'predictions': ok_pred,
            'params': {'range': 5000.0, 'sill': 1.0, 'nugget': 0.1}
        }
        
        # Compute metrics for all methods
        print("\nComputing metrics...")
        results['metrics'] = self._compute_metrics(results)
        
        # Compute spatial statistics
        print("Computing spatial statistics...")
        results['spatial_stats'] = self._compute_spatial_stats(
            address_gdf, results
        )
        
        self.results = results
        return results
    
    def _compute_metrics(self, results: Dict) -> pd.DataFrame:
        """Compute comparison metrics for all methods."""
        true_svi = results['true_svi']
        
        metrics_data = []
        
        # GNN metrics
        gnn_pred = results['gnn_predictions']
        
        # Check for NaN in GNN predictions
        gnn_valid_mask = ~np.isnan(gnn_pred) & ~np.isnan(true_svi)
        if np.sum(gnn_valid_mask) < len(gnn_pred) * 0.5:
            print(f"Warning: GNN has {np.sum(~gnn_valid_mask)} NaN predictions")
        
        metrics_data.append({
            'Method': 'GNN (GRANITE)',
            'MAE': mean_absolute_error(true_svi[gnn_valid_mask], gnn_pred[gnn_valid_mask]),
            'RMSE': np.sqrt(mean_squared_error(true_svi[gnn_valid_mask], gnn_pred[gnn_valid_mask])),
            'R²': r2_score(true_svi[gnn_valid_mask], gnn_pred[gnn_valid_mask]),
            'Correlation': np.corrcoef(true_svi[gnn_valid_mask], gnn_pred[gnn_valid_mask])[0, 1],
            'Valid_Predictions': np.sum(gnn_valid_mask)
        })
        
        # Baseline methods metrics
        for method_name, method_data in results['methods'].items():
            pred = method_data['predictions']
            
            # Handle NaN values
            valid_mask = ~np.isnan(pred) & ~np.isnan(true_svi)
            n_valid = np.sum(valid_mask)
            
            if n_valid < len(pred) * 0.5:
                print(f"Warning: {method_name} has {np.sum(~valid_mask)} NaN/invalid predictions")
            
            if n_valid > 0:
                metrics_data.append({
                    'Method': method_name,
                    'MAE': mean_absolute_error(true_svi[valid_mask], pred[valid_mask]),
                    'RMSE': np.sqrt(mean_squared_error(true_svi[valid_mask], pred[valid_mask])),
                    'R²': r2_score(true_svi[valid_mask], pred[valid_mask]),
                    'Correlation': np.corrcoef(true_svi[valid_mask], pred[valid_mask])[0, 1],
                    'Valid_Predictions': n_valid
                })
            else:
                metrics_data.append({
                    'Method': method_name,
                    'MAE': np.nan,
                    'RMSE': np.nan,
                    'R²': np.nan,
                    'Correlation': np.nan,
                    'Valid_Predictions': 0
                })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df = metrics_df.sort_values('RMSE')
        
        return metrics_df
    
    def _compute_spatial_stats(self, address_gdf: gpd.GeoDataFrame, 
                               results: Dict) -> Dict:
        """Compute spatial autocorrelation statistics for predictions."""
        from libpysal.weights import KNN
        from esda.moran import Moran
        
        spatial_stats = {}
        
        # Build spatial weights matrix
        coords = np.array([[geom.x, geom.y] for geom in address_gdf.geometry])
        w = KNN.from_array(coords, k=8)
        
        # Compute Moran's I for all methods
        true_svi = results['true_svi']
        
        moran_true = Moran(true_svi, w)
        spatial_stats['True SVI'] = {
            'morans_i': moran_true.I,
            'p_value': moran_true.p_sim
        }
        
        gnn_pred = results['gnn_predictions']
        moran_gnn = Moran(gnn_pred, w)
        spatial_stats['GNN'] = {
            'morans_i': moran_gnn.I,
            'p_value': moran_gnn.p_sim
        }
        
        for method_name, method_data in results['methods'].items():
            pred = method_data['predictions']
            moran = Moran(pred, w)
            spatial_stats[method_name] = {
                'morans_i': moran.I,
                'p_value': moran.p_sim
            }
        
        return spatial_stats
    
    def print_summary(self):
        """Print comparison summary to console."""
        if not self.results:
            print("No results available. Run run_comparison() first.")
            return
            
        print("\n" + "="*80)
        print("BASELINE COMPARISON RESULTS")
        print("="*80)
        
        print("\nPrediction Accuracy Metrics:")
        print(self.results['metrics'].to_string(index=False))
        
        print("\n\nSpatial Autocorrelation (Moran's I):")
        for method, stats in self.results['spatial_stats'].items():
            print(f"  {method:20s}: I = {stats['morans_i']:.4f} "
                  f"(p = {stats['p_value']:.4f})")
        
        # Compute improvement percentages
        print("\n\nGNN Performance vs Baselines:")
        gnn_rmse = self.results['metrics'][
            self.results['metrics']['Method'] == 'GNN (GRANITE)'
        ]['RMSE'].values[0]
        
        for _, row in self.results['metrics'].iterrows():
            if row['Method'] != 'GNN (GRANITE)':
                improvement = ((row['RMSE'] - gnn_rmse) / row['RMSE']) * 100
                print(f"  vs {row['Method']:15s}: {improvement:+.1f}% RMSE improvement")
    
    def plot_comparison(self, save_path: str = None):
        """Create comprehensive comparison visualization."""
        if not self.results:
            print("No results available. Run run_comparison() first.")
            return
            
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        true_svi = self.results['true_svi']
        gnn_pred = self.results['gnn_predictions']
        
        # 1. Scatter plots comparing each method to true SVI
        methods_to_plot = ['GNN (GRANITE)', 'IDW_p2', 'IDW_p3', 'Kriging']
        for i, method in enumerate(methods_to_plot):
            ax = fig.add_subplot(gs[i // 2, i % 2])
            
            if method == 'GNN (GRANITE)':
                pred = gnn_pred
            else:
                pred = self.results['methods'][method]['predictions']
            
            ax.scatter(true_svi, pred, alpha=0.3, s=10)
            ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect prediction')
            
            r2 = r2_score(true_svi, pred)
            rmse = np.sqrt(mean_squared_error(true_svi, pred))
            
            ax.set_xlabel('True SVI', fontsize=10)
            ax.set_ylabel('Predicted SVI', fontsize=10)
            ax.set_title(f'{method}\nR² = {r2:.3f}, RMSE = {rmse:.3f}', 
                        fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # 2. Error distribution comparison
        ax = fig.add_subplot(gs[0, 2])
        for method in methods_to_plot:
            if method == 'GNN (GRANITE)':
                pred = gnn_pred
            else:
                pred = self.results['methods'][method]['predictions']
            
            errors = pred - true_svi
            ax.hist(errors, bins=30, alpha=0.5, label=method, density=True)
        
        ax.set_xlabel('Prediction Error', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title('Error Distributions', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        
        # 3. Metrics bar chart
        ax = fig.add_subplot(gs[1, 2])
        metrics_df = self.results['metrics']
        x_pos = np.arange(len(metrics_df))
        
        ax.barh(x_pos, metrics_df['RMSE'], alpha=0.7)
        ax.set_yticks(x_pos)
        ax.set_yticklabels(metrics_df['Method'], fontsize=9)
        ax.set_xlabel('RMSE', fontsize=10)
        ax.set_title('RMSE Comparison', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        
        # 4. Spatial autocorrelation comparison
        ax = fig.add_subplot(gs[2, :])
        spatial_stats = self.results['spatial_stats']
        methods = list(spatial_stats.keys())
        morans_i = [spatial_stats[m]['morans_i'] for m in methods]
        
        bars = ax.bar(methods, morans_i, alpha=0.7)
        ax.set_ylabel("Moran's I", fontsize=10)
        ax.set_title("Spatial Autocorrelation Comparison", 
                    fontsize=11, fontweight='bold')
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
        
        # Color GNN bar differently
        bars[1].set_color('darkgreen')
        bars[1].set_alpha(0.9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nComparison plot saved to: {save_path}")
        
        return fig


def main():
    """
    Example usage demonstrating how to integrate baseline comparisons into GRANITE pipeline.
    """
    print("Baseline Comparison Module for GRANITE")
    print("="*80)
    print("\nThis module provides IDW and Kriging baseline implementations.")
    print("Integrate into your pipeline as follows:")
    print("""
    from baseline_comparisons import BaselineComparison
    
    # After training your GNN and generating predictions:
    comparison = BaselineComparison()
    results = comparison.run_comparison(
        tract_gdf=your_tract_geodataframe,
        address_gdf=your_address_geodataframe,
        gnn_predictions=your_gnn_predictions,
        svi_column='SVI'
    )
    
    comparison.print_summary()
    comparison.plot_comparison(save_path='baseline_comparison.png')
    """)


if __name__ == '__main__':
    main()