"""
Intelligent Dasymetric Mapping (IDM) baseline for GRANITE
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, Tuple

class IDMBaseline:
    """IDM baseline with FIXED mapping for your NLCD format"""
    
    def __init__(self):
        self.svi_coefficients = {
            # Your actual NLCD values → SVI coefficients
            0: 0.1,    # Likely water/undeveloped → low vulnerability
            1: 0.6,    # Likely low-medium development → moderate vulnerability  
            2: 1.2,    # Likely high development → high vulnerability
            250: 0.0,  # No data → zero vulnerability
            
            # Keep standard mapping as fallback
            21: 0.2, 22: 0.6, 23: 1.0, 24: 1.5,
            11: 0.0, 12: 0.0, 31: 0.0, 41: 0.0, 42: 0.0, 43: 0.0,
            51: 0.0, 52: 0.0, 71: 0.0, 72: 0.0, 73: 0.0, 74: 0.0,
            81: 0.1, 82: 0.1, 90: 0.0, 95: 0.0
        }
        
        print(f"IDM initialized with coefficient mapping: {self.svi_coefficients}")
    
    def disaggregate_svi(self, tract_svi: float, 
                        prediction_locations: pd.DataFrame,
                        nlcd_features: pd.DataFrame = None) -> Dict:
        """
        Apply IDM using FIXED coefficients
        
        Parameters:
        -----------
        tract_svi : float
            Known tract-level SVI value
        prediction_locations : pd.DataFrame
            Address locations for prediction
        nlcd_features : pd.DataFrame, optional
            NLCD features (if None, uses uniform distribution)
            
        Returns:
        --------
        Dict
            IDM results with predictions and diagnostics
        """
        n_addresses = len(prediction_locations)
        
        if nlcd_features is not None and len(nlcd_features) > 0:
            # Use NLCD-based IDM with FIXED coefficients
            coefficients = nlcd_features['nlcd_class'].map(
                self.svi_coefficients
            ).fillna(0.0)
            
            # Calculate weighted distribution
            total_weight = coefficients.sum()
            if total_weight > 0:
                # Normalize weights and distribute tract SVI
                normalized_weights = coefficients / total_weight
                predictions = normalized_weights * tract_svi * n_addresses
            else:
                # Fallback: uniform distribution
                predictions = pd.Series([tract_svi] * n_addresses)
        else:
            # Fallback: uniform distribution (no NLCD data)
            predictions = pd.Series([tract_svi] * n_addresses)
            coefficients = pd.Series([1.0] * n_addresses)
        
        # Simple uncertainty estimate (IDM doesn't provide uncertainty)
        prediction_std = np.std(predictions)
        uncertainty = np.ones_like(predictions) * max(0.01, prediction_std * 0.1)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'x': prediction_locations['x'] if 'x' in prediction_locations.columns else prediction_locations.iloc[:, 0],
            'y': prediction_locations['y'] if 'y' in prediction_locations.columns else prediction_locations.iloc[:, 1], 
            'mean': predictions.values,
            'sd': uncertainty,
            'q025': predictions.values - 1.96 * uncertainty,
            'q975': predictions.values + 1.96 * uncertainty
        })
        
        # Diagnostics
        predicted_mean = predictions.mean()
        constraint_error = abs(predicted_mean - tract_svi) / tract_svi if tract_svi > 0 else 0
        
        return {
            'success': True,
            'predictions': results_df,
            'diagnostics': {
                'method': 'IDM_fixed_coefficients',
                'constraint_satisfied': constraint_error < 0.01,
                'constraint_error': constraint_error,
                'mean_prediction': predicted_mean,
                'std_prediction': np.std(predictions),
                'mean_uncertainty': np.mean(uncertainty),
                'total_addresses': n_addresses
            }
        }
