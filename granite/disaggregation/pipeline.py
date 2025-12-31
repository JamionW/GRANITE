"""
GRANITE Pipeline (Spatial Version)

Simplified pipeline for spatial disaggregation of tract-level SVI.
"""
import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from datetime import datetime
from typing import Dict, Optional

from ..models.gnn import (
    set_random_seed, 
    SpatialDisaggregationGNN, 
    SpatialGNNTrainer,
    MultiTractGNNTrainer
)
from ..data.loaders import DataLoader
from ..features.spatial_features import SpatialFeatureComputer, normalize_spatial_features
from ..evaluation.baselines import IDWDisaggregation, OrdinaryKrigingDisaggregation
from ..visualization.plots import SpatialVisualizer, generate_all_plots


class GRANITEPipeline:
    """
    GRANITE: Spatial disaggregation of tract-level SVI.
    
    Pipeline:
        1. Load spatial data (addresses, tracts, SVI)
        2. Compute spatial features (coordinates, density, boundary)
        3. Build k-NN graph
        4. Train GNN with tract mean constraint
        5. Validate against baselines and block groups
    """
    
    def __init__(self, config: dict, data_dir: str = './data', output_dir: str = './output'):
        self.config = config
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.verbose = config.get('processing', {}).get('verbose', False)
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.data_loader = DataLoader(data_dir, config=config)
        self.feature_computer = SpatialFeatureComputer(verbose=self.verbose)
    
    def _log(self, message: str, level: str = 'INFO'):
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    def run(self) -> Dict:
        """Main pipeline execution."""
        start_time = time.time()
        
        seed = self.config.get('processing', {}).get('random_seed', 42)
        set_random_seed(seed)
        
        self._log("Starting GRANITE Spatial Pipeline...")
        
        # Load data
        data = self._load_data()
        
        # Get target tract
        target_fips = self.config.get('data', {}).get('target_fips')
        if not target_fips:
            return {'success': False, 'error': 'FIPS code required'}
        
        # Check for multi-tract mode
        n_neighbors = self.config.get('data', {}).get('neighbor_tracts', 0)
        
        if n_neighbors > 0:
            result = self._run_multi_tract(target_fips, n_neighbors, data)
        else:
            result = self._run_single_tract(target_fips, data)
        
        elapsed = time.time() - start_time
        self._log(f"Pipeline completed in {elapsed:.1f} seconds")
        
        result['elapsed_time'] = elapsed
        return result
    
    def _load_data(self) -> Dict:
        """Load all required data."""
        self._log("Loading data...")
        
        state_fips = self.config.get('data', {}).get('state_fips', '47')
        county_fips = self.config.get('data', {}).get('county_fips', '065')
        
        tracts = self.data_loader.load_census_tracts(state_fips, county_fips)
        county_name = self.data_loader._get_county_name(state_fips, county_fips)
        svi = self.data_loader.load_svi_data(state_fips, county_name)
        
        data = {
            'tracts': tracts.merge(svi, on='FIPS', how='inner'),
            'svi': svi
        }
        
        self._log(f"Loaded {len(data['tracts'])} tracts with SVI data")
        return data
    
    def _run_single_tract(self, target_fips: str, data: Dict) -> Dict:
        """Process a single tract."""
        self._log(f"Processing tract {target_fips}")
        
        # Get tract info
        tract_data = data['tracts'][data['tracts']['FIPS'] == target_fips]
        if len(tract_data) == 0:
            return {'success': False, 'error': f'Tract {target_fips} not found'}
        
        tract_svi = float(tract_data.iloc[0]['RPL_THEMES'])
        tract_geom = tract_data.iloc[0].geometry
        
        # Get addresses
        addresses = self.data_loader.get_addresses_for_tract(target_fips)
        if len(addresses) == 0:
            return {'success': False, 'error': f'No addresses in tract {target_fips}'}
        
        self._log(f"Found {len(addresses)} addresses, tract SVI: {tract_svi:.4f}")
        
        # Compute spatial features
        features, feature_names = self.feature_computer.compute_features(
            addresses, tract_geom
        )
        normalized_features, scaler = normalize_spatial_features(features)
        
        # Build graph
        k_neighbors = self.config.get('model', {}).get('k_neighbors', 8)
        graph_data = self.data_loader.create_spatial_graph(
            addresses, normalized_features, k_neighbors
        )
        
        # Train GNN
        model = SpatialDisaggregationGNN(
            input_dim=normalized_features.shape[1],
            hidden_dim=self.config.get('model', {}).get('hidden_dim', 32),
            dropout=self.config.get('model', {}).get('dropout', 0.2)
        )
        
        trainer = SpatialGNNTrainer(
            model,
            learning_rate=self.config.get('training', {}).get('learning_rate', 0.001),
            constraint_weight=self.config.get('training', {}).get('constraint_weight', 2.0)
        )
        
        epochs = self.config.get('training', {}).get('epochs', 100)
        training_result = trainer.train(graph_data, tract_svi, epochs, self.verbose)
        
        if not training_result['success']:
            return training_result
        
        # Apply constraint correction
        raw_predictions = training_result['raw_predictions']
        final_predictions = self._apply_constraint_correction(raw_predictions, tract_svi)
        
        # Compute baselines
        baseline_results = self._compute_baselines(addresses, data['tracts'], target_fips, tract_svi)
        
        # Validation
        validation = self._validate_predictions(
            final_predictions, tract_svi, addresses, feature_names, features
        )
        
        return {
            'success': True,
            'predictions': final_predictions,
            'raw_predictions': raw_predictions,
            'addresses': addresses,
            'tract_svi': tract_svi,
            'feature_names': feature_names,
            'training_history': training_result['training_history'],
            'baselines': baseline_results,
            'validation': validation,
            'summary': {
                'addresses_processed': len(addresses),
                'spatial_features': len(feature_names),
                'spatial_variation': float(np.std(final_predictions)),
                'constraint_error': float(abs(np.mean(final_predictions) - tract_svi) / tract_svi * 100)
            }
        }
    
    def _run_multi_tract(self, target_fips: str, n_neighbors: int, data: Dict) -> Dict:
        """Process multiple tracts together."""
        tract_list = self.data_loader.get_neighboring_tracts(target_fips, n_neighbors)
        self._log(f"Multi-tract mode: {len(tract_list)} tracts")
        
        # Collect addresses from all tracts
        all_addresses = []
        tract_svis = {}
        
        for fips in tract_list:
            tract_data = data['tracts'][data['tracts']['FIPS'] == fips]
            if len(tract_data) == 0:
                continue
            
            tract_svis[fips] = float(tract_data.iloc[0]['RPL_THEMES'])
            addresses = self.data_loader.get_addresses_for_tract(fips)
            addresses['tract_fips'] = fips
            all_addresses.append(addresses)
        
        combined_addresses = pd.concat(all_addresses, ignore_index=True)
        combined_addresses = gpd.GeoDataFrame(combined_addresses, crs='EPSG:4326')
        
        self._log(f"Combined {len(combined_addresses)} addresses from {len(tract_list)} tracts")
        
        # Compute features per tract
        all_features = []
        for fips in tract_list:
            tract_mask = combined_addresses['tract_fips'] == fips
            tract_addresses = combined_addresses[tract_mask]
            tract_geom = data['tracts'][data['tracts']['FIPS'] == fips].geometry.iloc[0]
            
            features, feature_names = self.feature_computer.compute_features(
                tract_addresses, tract_geom
            )
            all_features.append(features)
        
        combined_features = np.vstack(all_features)
        normalized_features, _ = normalize_spatial_features(combined_features)
        
        # Build combined graph
        graph_data = self.data_loader.create_spatial_graph(
            combined_addresses, normalized_features
        )
        
        # Create tract masks
        tract_masks = {}
        for fips in tract_list:
            tract_masks[fips] = (combined_addresses['tract_fips'] == fips).values
        
        # Train
        model = SpatialDisaggregationGNN(
            input_dim=normalized_features.shape[1],
            hidden_dim=self.config.get('model', {}).get('hidden_dim', 32)
        )
        
        trainer = MultiTractGNNTrainer(model)
        epochs = self.config.get('training', {}).get('epochs', 100)
        
        training_result = trainer.train_multi_tract(
            graph_data, tract_svis, tract_masks, epochs, self.verbose
        )
        
        # Extract target tract predictions
        target_mask = combined_addresses['tract_fips'] == target_fips
        target_predictions = training_result['raw_predictions'][target_mask]
        target_addresses = combined_addresses[target_mask].copy()
        target_svi = tract_svis[target_fips]
        
        final_predictions = self._apply_constraint_correction(target_predictions, target_svi)
        
        return {
            'success': True,
            'predictions': final_predictions,
            'addresses': target_addresses,
            'tract_svi': target_svi,
            'tracts_used': tract_list,
            'summary': {
                'addresses_processed': len(target_addresses),
                'spatial_variation': float(np.std(final_predictions)),
                'constraint_error': float(abs(np.mean(final_predictions) - target_svi) / target_svi * 100)
            }
        }
    
    def _apply_constraint_correction(self, predictions: np.ndarray, target_svi: float) -> np.ndarray:
        """Apply additive correction to satisfy tract mean constraint."""
        current_mean = np.mean(predictions)
        correction = target_svi - current_mean
        corrected = predictions + correction
        
        # Clip to valid range
        corrected = np.clip(corrected, 0.0, 1.0)
        
        return corrected
    
    def _compute_baselines(self, addresses: gpd.GeoDataFrame, 
                        tracts: gpd.GeoDataFrame,
                        tract_fips: str,
                        tract_svi: float) -> Dict:
        """Compute IDW and Kriging baselines."""
        results = {}
        
        # Extract address coordinates as numpy array [lon, lat]
        address_coords = np.column_stack([
            addresses.geometry.x.values,
            addresses.geometry.y.values
        ])
        
        # Filter tracts with valid SVI
        tracts_valid = tracts[tracts['RPL_THEMES'].notna()].copy()
        
        try:
            idw = IDWDisaggregation(power=2.0, n_neighbors=8)
            idw.fit(tracts_valid, svi_column='RPL_THEMES')
            idw_predictions = idw.disaggregate(address_coords, tract_fips, tract_svi)
            results['idw'] = {
                'predictions': idw_predictions,
                'std': float(np.std(idw_predictions)),
                'constraint_error': float(abs(np.mean(idw_predictions) - tract_svi) / tract_svi * 100)
            }
            self._log(f"IDW baseline: std={results['idw']['std']:.4f}")
        except Exception as e:
            self._log(f"IDW baseline failed: {e}", level='WARN')
            results['idw'] = {'error': str(e)}
        
        try:
            kriging = OrdinaryKrigingDisaggregation()
            kriging.fit(tracts_valid, svi_column='RPL_THEMES')
            kriging_predictions = kriging.disaggregate(address_coords, tract_fips, tract_svi)
            results['kriging'] = {
                'predictions': kriging_predictions,
                'std': float(np.std(kriging_predictions)),
                'constraint_error': float(abs(np.mean(kriging_predictions) - tract_svi) / tract_svi * 100)
            }
            self._log(f"Kriging baseline: std={results['kriging']['std']:.4f}")
        except Exception as e:
            self._log(f"Kriging baseline failed: {e}", level='WARN')
            results['kriging'] = {'error': str(e)}
        
        return results
    
    def _validate_predictions(self, predictions: np.ndarray, tract_svi: float,
                             addresses: gpd.GeoDataFrame, 
                             feature_names: list,
                             features: np.ndarray) -> Dict:
        """Validate predictions."""
        validation = {
            'constraint_satisfied': abs(np.mean(predictions) - tract_svi) / tract_svi < 0.01,
            'prediction_range': [float(predictions.min()), float(predictions.max())],
            'prediction_std': float(np.std(predictions)),
            'prediction_mean': float(np.mean(predictions))
        }
        
        # Feature correlations with predictions
        correlations = {}
        for i, name in enumerate(feature_names):
            r = np.corrcoef(features[:, i], predictions)[0, 1]
            correlations[name] = float(r) if not np.isnan(r) else 0.0
        
        validation['feature_correlations'] = correlations
        
        return validation
    
    def save_results(self, results: Dict, filename: str = 'granite_results.csv'):
        """Save predictions to CSV."""
        if not results.get('success'):
            return
        
        output_path = os.path.join(self.output_dir, filename)
        
        df = pd.DataFrame({
            'address_id': results['addresses']['address_id'].values,
            'x': results['addresses'].geometry.x.values,
            'y': results['addresses'].geometry.y.values,
            'predicted_svi': results['predictions']
        })
        
        df.to_csv(output_path, index=False)
        self._log(f"Results saved to {output_path}")

    def generate_plots(self, results: Dict):
        """Generate visualization plots from results."""
        if not results.get('success'):
            self._log("Cannot generate plots - pipeline failed", level='WARN')
            return
        
        self._log("Generating visualization plots...")
        
        try:
            from ..visualization.plots import generate_all_plots
            generate_all_plots(results, output_dir=self.output_dir)
        except Exception as e:
            self._log(f"Plot generation failed: {e}", level='WARN')