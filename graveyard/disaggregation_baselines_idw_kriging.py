# retired 2026-04-18
# replacement classes: DasymetricDisaggregation, PycnophylacticDisaggregation
# reason: point-interpolation methods collapse to tract mean under single-centroid
# interpolation; replaced by mass-preserving disaggregation baselines.

import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


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


class IDWDisaggregation:
    """
    Inverse Distance Weighting disaggregation.

    Uses neighboring tract centroids to create spatial gradients within
    the target tract, then adjusts to satisfy constraint.

    Constraint-preserving: final predictions are scaled to ensure
    mean equals known tract SVI.
    """

    def __init__(self, power: float = 2.0, n_neighbors: int = 8):
        self.name = f'IDW_p{power}'
        self.power = power
        self.n_neighbors = n_neighbors
        self.tract_centroids = None
        self.tract_svi = None
        self.tract_fips = None
        self.kdtree = None
        self.fitted = False

    def fit(self, tract_gdf: gpd.GeoDataFrame, svi_column: str = 'RPL_THEMES'):
        """Fit IDW using all tract centroids and SVI values."""

        # Store tract data
        self.tract_fips = tract_gdf['FIPS'].values
        self.tract_svi = tract_gdf[svi_column].values

        # Compute centroids in lon/lat
        centroids_lonlat = np.array([
            [geom.centroid.x, geom.centroid.y]
            for geom in tract_gdf.geometry
        ])

        # Convert to approximate meters for isotropic distance computation
        lat_center = np.mean(centroids_lonlat[:, 1])
        self._meters_per_deg_lon = 111320 * np.cos(np.radians(lat_center))
        self._meters_per_deg_lat = 110540

        self.tract_centroids = centroids_lonlat.copy()
        self.tract_centroids[:, 0] *= self._meters_per_deg_lon
        self.tract_centroids[:, 1] *= self._meters_per_deg_lat

        # Build KD-tree on meter-scale coordinates
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

        # Convert address coords to meters (same scale as fitted KD-tree)
        address_coords_m = address_coords.copy()
        address_coords_m[:, 0] *= self._meters_per_deg_lon
        address_coords_m[:, 1] *= self._meters_per_deg_lat

        # Find target tract index so we can exclude it from neighbors
        target_idx = np.where(self.tract_fips == tract_fips)[0]

        # Query one extra neighbor to allow filtering out the target tract
        k_query = min(self.n_neighbors + 1, len(self.tract_svi))
        distances, indices = self.kdtree.query(address_coords_m, k=k_query)

        # Exclude the target tract from each address's neighbor set
        if len(target_idx) > 0:
            filtered_distances = []
            filtered_indices = []
            for i in range(n_addresses):
                mask = ~np.isin(indices[i], target_idx)
                d = distances[i][mask][:self.n_neighbors]
                idx = indices[i][mask][:self.n_neighbors]
                # if filtering removed too many, pad with last valid
                while len(d) < min(self.n_neighbors, k_query - len(target_idx)):
                    d = np.append(d, d[-1])
                    idx = np.append(idx, idx[-1])
                filtered_distances.append(d)
                filtered_indices.append(idx)
            distances = np.array(filtered_distances)
            indices = np.array(filtered_indices)

        # Handle edge case of point exactly on centroid
        min_distance = 1e-10
        distances = np.maximum(distances, min_distance)

        # Compute IDW weights
        weights = 1.0 / (distances ** self.power)
        weights = weights / weights.sum(axis=1, keepdims=True)

        # Weighted average of neighbor SVI values
        raw_predictions = (weights * self.tract_svi[indices]).sum(axis=1)

        # Constraint enforcement: iteratively adjust so mean == tract_svi
        # while keeping all values in [0, 1]
        current_mean = np.mean(raw_predictions)
        if abs(current_mean) > 1e-8:
            adjusted_predictions = raw_predictions * (tract_svi / current_mean)
        else:
            adjusted_predictions = np.full(n_addresses, tract_svi)

        return _enforce_constraint(adjusted_predictions, tract_svi)


class OrdinaryKrigingDisaggregation:
    """
    Ordinary Kriging disaggregation with exponential variogram.

    Uses spatial correlation structure from neighboring tracts
    to create spatially coherent within-tract variation.
    """

    def __init__(self, variogram_range: float = 5000.0, sill: float = 0.1,
                 nugget: float = 0.01):
        self.name = 'Kriging'
        self.variogram_range = variogram_range  # meters
        self.sill = sill
        self.nugget = nugget
        self.tract_centroids = None
        self.tract_svi = None
        self.fitted = False

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

        # Constraint enforcement: iteratively adjust so mean == tract_svi
        # while keeping all values in [0, 1]
        current_mean = np.mean(predictions)
        if abs(current_mean) > 1e-8:
            predictions = predictions * (tract_svi / current_mean)
        else:
            predictions = np.full(n_addresses, tract_svi)

        return _enforce_constraint(predictions, tract_svi)
