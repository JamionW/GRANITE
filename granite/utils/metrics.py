"""
Evaluation metrics for GRANITE framework
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_disaggregation_metrics(true_values, predicted_values, uncertainties=None):
    """
    Calculate comprehensive metrics for disaggregation evaluation
    
    Parameters:
    -----------
    true_values : array-like
        True SVI values
    predicted_values : array-like
        Predicted SVI values
    uncertainties : array-like, optional
        Prediction uncertainties (standard deviations)
        
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['mae'] = mean_absolute_error(true_values, predicted_values)
    metrics['rmse'] = np.sqrt(mean_squared_error(true_values, predicted_values))
    metrics['r2'] = r2_score(true_values, predicted_values)
    metrics['correlation'] = np.corrcoef(true_values, predicted_values)[0, 1]
    
    # Bias metrics
    errors = predicted_values - true_values
    metrics['bias'] = np.mean(errors)
    metrics['abs_bias'] = np.abs(metrics['bias'])
    
    # Distribution metrics
    metrics['std_error'] = np.std(errors)
    metrics['iqr_error'] = np.percentile(errors, 75) - np.percentile(errors, 25)
    
    # Percentile errors
    for p in [10, 25, 50, 75, 90]:
        metrics[f'p{p}_error'] = np.percentile(np.abs(errors), p)
    
    # Uncertainty metrics if provided
    if uncertainties is not None:
        metrics.update(calculate_uncertainty_metrics(
            true_values, predicted_values, uncertainties
        ))
    
    return metrics


def calculate_uncertainty_metrics(true_values, predicted_values, uncertainties):
    """
    Calculate uncertainty calibration metrics
    
    Parameters:
    -----------
    true_values : array-like
        True values
    predicted_values : array-like
        Predicted values
    uncertainties : array-like
        Prediction uncertainties (standard deviations)
        
    Returns:
    --------
    dict
        Uncertainty metrics
    """
    metrics = {}
    
    # Standardized errors
    z_scores = (true_values - predicted_values) / (uncertainties + 1e-8)
    
    # Coverage metrics
    for conf_level in [0.68, 0.90, 0.95]:
        z_crit = stats.norm.ppf((1 + conf_level) / 2)
        coverage = np.mean(np.abs(z_scores) <= z_crit)
        metrics[f'coverage_{int(conf_level*100)}'] = coverage
    
    # Calibration metrics
    metrics['mean_z_score'] = np.mean(z_scores)
    metrics['std_z_score'] = np.std(z_scores)
    metrics['avg_uncertainty'] = np.mean(uncertainties)
    
    # Sharpness (lower is better)
    metrics['sharpness'] = np.mean(uncertainties)
    
    # Interval score
    metrics['interval_score'] = calculate_interval_score(
        true_values, predicted_values, uncertainties
    )
    
    # Calibration error
    metrics['calibration_error'] = np.abs(metrics['std_z_score'] - 1.0)
    
    return metrics


def calculate_interval_score(true_values, predicted_values, uncertainties, alpha=0.05):
    """
    Calculate interval score for probabilistic predictions
    
    Parameters:
    -----------
    true_values : array-like
        True values
    predicted_values : array-like
        Predicted values
    uncertainties : array-like
        Standard deviations
    alpha : float
        Significance level
        
    Returns:
    --------
    float
        Average interval score
    """
    z_crit = stats.norm.ppf(1 - alpha/2)
    
    lower = predicted_values - z_crit * uncertainties
    upper = predicted_values + z_crit * uncertainties
    
    # Width of prediction interval
    width = upper - lower
    
    # Penalties for values outside interval
    lower_penalty = (2/alpha) * np.maximum(0, lower - true_values)
    upper_penalty = (2/alpha) * np.maximum(0, true_values - upper)
    
    # Interval score
    scores = width + lower_penalty + upper_penalty
    
    return np.mean(scores)


def calculate_spatial_metrics(predictions_gdf, k_neighbors=10):
    """
    Calculate spatial autocorrelation metrics
    
    Parameters:
    -----------
    predictions_gdf : gpd.GeoDataFrame
        Predictions with geometry
    k_neighbors : int
        Number of neighbors for spatial metrics
        
    Returns:
    --------
    dict
        Spatial metrics
    """
    from sklearn.neighbors import NearestNeighbors
    
    metrics = {}
    
    # Extract coordinates
    coords = np.array([[geom.x, geom.y] for geom in predictions_gdf.geometry])
    
    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    
    # Calculate spatial autocorrelation of predictions
    svi_values = predictions_gdf['svi_predicted'].values
    
    # Local Moran's I
    spatial_lag = np.array([
        np.mean(svi_values[indices[i][1:]])  # Exclude self
        for i in range(len(svi_values))
    ])
    
    # Global Moran's I
    n = len(svi_values)
    z = svi_values - np.mean(svi_values)
    W = np.zeros((n, n))
    
    for i in range(n):
        W[i, indices[i][1:]] = 1 / distances[i][1:]
    
    W = W / W.sum(axis=1, keepdims=True)
    
    morans_i = (n / np.sum(z**2)) * np.sum(z * (W @ z))
    metrics['morans_i'] = morans_i
    
    # Geary's C
    c_num = 0
    for i in range(n):
        for j in indices[i][1:]:
            c_num += (svi_values[i] - svi_values[j])**2
    
    gearys_c = ((n - 1) * c_num) / (2 * W.sum() * np.sum(z**2))
    metrics['gearys_c'] = gearys_c
    
    # Average neighbor correlation
    neighbor_corrs = []
    for i in range(n):
        if len(indices[i]) > 1:
            corr = np.corrcoef(
                svi_values[i], 
                svi_values[indices[i][1:]]
            )[0, 1]
            if not np.isnan(corr):
                neighbor_corrs.append(corr)
    
    metrics['avg_neighbor_corr'] = np.mean(neighbor_corrs)
    
    return metrics


def calculate_mass_preservation_error(tract_true, tract_predicted, tract_counts):
    """
    Calculate mass preservation error
    
    Parameters:
    -----------
    tract_true : array-like
        True tract-level values
    tract_predicted : array-like
        Average of predicted values in tract
    tract_counts : array-like
        Number of predictions in each tract
        
    Returns:
    --------
    dict
        Mass preservation metrics
    """
    metrics = {}
    
    # Weight by number of predictions
    weights = tract_counts / np.sum(tract_counts)
    
    # Weighted metrics
    metrics['weighted_mae'] = np.sum(weights * np.abs(tract_true - tract_predicted))
    metrics['weighted_rmse'] = np.sqrt(np.sum(weights * (tract_true - tract_predicted)**2))
    
    # Relative errors
    rel_errors = np.abs(tract_true - tract_predicted) / (tract_true + 1e-8)
    metrics['mean_relative_error'] = np.mean(rel_errors)
    metrics['max_relative_error'] = np.max(rel_errors)
    
    # Conservation metric (should be close to 0)
    total_true = np.sum(tract_true * tract_counts)
    total_pred = np.sum(tract_predicted * tract_counts)
    metrics['conservation_error'] = (total_pred - total_true) / total_true
    
    return metrics


def create_metrics_report(all_metrics):
    """
    Create formatted metrics report
    
    Parameters:
    -----------
    all_metrics : dict
        Dictionary of all metrics
        
    Returns:
    --------
    str
        Formatted report
    """
    report = []
    report.append("="*60)
    report.append("GRANITE Evaluation Metrics Report")
    report.append("="*60)
    
    # Group metrics
    groups = {
        'Accuracy Metrics': ['mae', 'rmse', 'r2', 'correlation', 'bias'],
        'Error Distribution': ['std_error', 'iqr_error', 'p50_error', 'p90_error'],
        'Uncertainty Calibration': ['coverage_95', 'calibration_error', 'sharpness'],
        'Spatial Metrics': ['morans_i', 'gearys_c', 'avg_neighbor_corr'],
        'Mass Preservation': ['weighted_mae', 'conservation_error']
    }
    
    for group_name, metric_names in groups.items():
        report.append(f"\n{group_name}:")
        report.append("-" * len(group_name))
        
        for metric in metric_names:
            if metric in all_metrics:
                value = all_metrics[metric]
                if isinstance(value, float):
                    report.append(f"  {metric:.<30} {value:.4f}")
                else:
                    report.append(f"  {metric:.<30} {value}")
    
    return "\n".join(report)