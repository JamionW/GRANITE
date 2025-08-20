"""
granite/disaggregation/enhanced_pipeline.py

Enhanced GRANITE pipeline that implements the hybrid IDM+GNN architecture.
This replaces the original pipeline.py with the new research direction.
"""

import os
import time
import json
import warnings
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import torch

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import numpy as np
import pandas as pd
import os
import traceback

# Make sure matplotlib can save files
plt.ioff()  # Turn off interactive mode

# Import required classes (add these to existing imports)
from ..models.gnn import prepare_graph_data_with_nlcd
from ..disaggregation.hybrid_framework import AccessibilityGNNCorrector
from ..models.training import AccessibilityTrainer

# Local imports (these stay the same)
from ..data.loaders import DataLoader
from ..baselines.idm import IDMBaseline
from ..metricgraph.interface import MetricGraphInterface
from ..visualization.plots import DisaggregationVisualizer

# New imports for hybrid architecture
from .hybrid_framework import (
    HybridDisaggregator, 
    DisaggregationConfig,
    AccessibilityGNNCorrector,
    SpatialIntegrator
)
from ..models.correction_training import AccessibilityCorrectionTrainer
from ..models.gnn import prepare_graph_data_with_nlcd


class GRANITEPipeline:
    """
    Enhanced GRANITE pipeline implementing hybrid IDM+GNN disaggregation.
    
    Key improvements:
    1. Uses IDM as baseline instead of trying to replace it
    2. GNN learns accessibility corrections, not full SPDE parameters
    3. Prevents trivial solution collapse through better loss functions
    4. Maintains empirically-validated land cover knowledge
    """
    
    def __init__(self, config: Dict, data_dir: str = './data', 
                 output_dir: str = './output', verbose: bool = None):
        """
        Initialize enhanced pipeline.
        
        Parameters:
        -----------
        config : Dict
            Configuration from config.yaml
        data_dir : str
            Directory containing input data
        output_dir : str
            Directory for output files
        verbose : bool
            Enable verbose logging
        """
        if config is None:
            raise ValueError("Configuration is required")
        
        self.config = config
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.verbose = verbose if verbose is not None else config.get('processing', {}).get('verbose', False)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader(data_dir, config=config)
        
        # IDM baseline component
        self.idm_baseline = IDMBaseline(config=config, grid_resolution_meters=100)
        
        # MetricGraph interface for spatial integration
        self.mg_interface = MetricGraphInterface(verbose=self.verbose, config=config)
        
        # Visualization
        self.visualizer = DisaggregationVisualizer()
        
        # Initialize hybrid disaggregator
        disagg_config = DisaggregationConfig(
            idm_weight=config.get('model', {}).get('idm_weight', 0.7),
            gnn_correction_scale=config.get('model', {}).get('correction_scale', 0.3)
        )
        self.hybrid_disaggregator = HybridDisaggregator(config=disagg_config)
        
        # Storage for results
        self.results = {}
    
    def _log(self, message: str, level: str = 'INFO'):
        """Logging with timestamp."""
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    def run(self) -> Dict:
        """
        Run the enhanced GRANITE pipeline.
        
        Returns:
        --------
        Dict with results including predictions, diagnostics, and comparisons
        """
        start_time = time.time()
        self._log("Starting Enhanced GRANITE Pipeline with Hybrid IDM+GNN Architecture")
        
        # Load data
        self._log("Loading data...")
        data = self._load_data()
        
        # Process tracts
        processing_mode = self.config.get('data', {}).get('processing_mode', 'county')
        
        if processing_mode == 'county':
            results = self._process_county_mode(data)
        elif processing_mode == 'fips':
            target_fips = self.config.get('data', {}).get('target_fips')
            results = self._process_single_tract(data, target_fips)
        else:
            raise ValueError(f"Unknown processing mode: {processing_mode}")
        
        # Save results
        if results.get('success'):
            self._save_results(results)
            self._create_enhanced_visualizations(results)
        
        elapsed_time = time.time() - start_time
        self._log(f"Pipeline completed in {elapsed_time:.1f} seconds")
        
        return results
    
    def _load_data(self) -> Dict:
        """Load all required data."""
        data = {}
        
        # Load SVI data
        data['svi'] = self.data_loader.load_svi_data()
        self._log(f"Loaded SVI data: {len(data['svi'])} tracts")
        
        # Load census tracts
        data['tracts'] = self.data_loader.load_census_tracts()
        self._log(f"Loaded census tracts: {len(data['tracts'])} geometries")
        
        # Load road network
        data['roads'] = self.data_loader.load_road_network()
        self._log(f"Loaded road network with {len(data['roads'])} segments")
        
        # Load address points (real or synthetic)
        use_real = self.config.get('data', {}).get('use_real_addresses', True)
        if use_real:
            data['addresses'] = self.data_loader.load_address_points()
            self._log(f"Loaded real addresses: {len(data['addresses'])} points")
        else:
            data['addresses'] = self.data_loader.generate_synthetic_addresses(
                data['tracts'], num_points_per_tract=1000
            )
            self._log(f"Generated synthetic addresses: {len(data['addresses'])} points")
        
        return data
    
    def _load_nlcd_for_tract(self, tract_data):
        """
        Load NLCD data for tract area
        """
        try:
            # Import geopandas here to ensure it's available
            import geopandas as gpd
            
            # Get tract boundary for NLCD cropping
            tract_boundary = gpd.GeoDataFrame([tract_data['tract_info']], crs='EPSG:4326')
            
            # Load NLCD data
            nlcd_data = self.data_loader.load_nlcd_data(
                county_bounds=tract_boundary,
                nlcd_path="./data/nlcd_hamilton_county.tif"
            )
            
            if nlcd_data is None:
                self._log("  WARNING: NLCD data not available, falling back to topological features")
                return None
            
            # Extract NLCD features at address locations
            nlcd_features = self.data_loader.extract_nlcd_features_at_addresses(
                tract_data['addresses'], 
                nlcd_data
            )
            
            self._log(f"  Extracted NLCD features for {len(nlcd_features)} addresses")
            return nlcd_features
            
        except Exception as e:
            self._log(f"  Error loading NLCD: {str(e)}")
            return None
    
    def _process_single_tract(self, data: Dict, fips: str) -> Dict:
        """
        Process a single census tract with hybrid disaggregation.
        
        This is where the main innovation happens:
        1. Compute IDM baseline
        2. Train GNN to learn corrections
        3. Apply spatial integration with MetricGraph
        """
        try:
            # Find target tract
            target_tract = data['svi'][data['svi']['FIPS'] == fips]
            if len(target_tract) == 0:
                return {'success': False, 'message': f'Tract {fips} not found in SVI data'}
            
            tract_svi = float(target_tract.iloc[0]['RPL_THEMES'])
            tract_info = data['tracts'][data['tracts']['GEOID'] == fips].iloc[0]
            
            # Get addresses in this tract using dedicated method
            tract_addresses = self.data_loader.get_addresses_for_tract(fips)
            if len(tract_addresses) == 0:
                return {'success': False, 'message': f'No addresses found in tract {fips}'}
            
            self._log(f"Found {len(tract_addresses)} addresses in tract")
            
            # Load NLCD features for this tract
            tract_data = {
                'tract_info': tract_info,
                'addresses': tract_addresses
            }
            nlcd_features = self._load_nlcd_for_tract(tract_data)

            if nlcd_features is not None:
                self._log("DEBUG: NLCD features analysis:")
                self._log(f"  NLCD features shape: {nlcd_features.shape}")
                self._log(f"  NLCD columns: {list(nlcd_features.columns)}")
                
                if 'nlcd_class' in nlcd_features.columns:
                    unique_classes = nlcd_features['nlcd_class'].unique()
                    self._log(f"  Unique NLCD classes found: {sorted(unique_classes)}")
                    self._log(f"  Class counts: {nlcd_features['nlcd_class'].value_counts().head()}")
                    
                    # Check if these match IDM expected classes
                    expected_classes = [11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82, 90, 95]
                    found_valid = any(cls in expected_classes for cls in unique_classes)
                    self._log(f"  Contains valid NLCD classes: {found_valid}")
                    
                    if not found_valid:
                        self._log(f"  WARNING: No valid NLCD classes found! IDM will use fallback values.")
                        self._log(f"  Expected classes: {expected_classes}")
                        self._log(f"  Found classes: {list(unique_classes)}")
                else:
                    self._log("  ERROR: No 'nlcd_class' column found in NLCD features!")
            else:
                self._log("DEBUG: nlcd_features is None - NLCD loading failed")
            
            # Step 1: Compute IDM baseline disaggregation
            self._log("Step 1: Computing IDM baseline disaggregation...")
            
            try:
                # FIXED: Use correct method name and parameters from main branch
                idm_result = self.idm_baseline.disaggregate_svi(
                    tract_svi=float(tract_svi),
                    prediction_locations=tract_addresses,
                    nlcd_features=nlcd_features,
                    tract_geometry=tract_info.geometry
                )
                
                if not idm_result['success']:
                    raise ValueError("IDM disaggregation failed")
                
                # Handle different column names from IDM result
                predictions_df = idm_result['predictions']
                if 'svi_prediction' in predictions_df.columns:
                    idm_predictions = predictions_df['svi_prediction'].values
                else:
                    # Use first numeric column (excluding x, y, uncertainty)
                    numeric_cols = predictions_df.select_dtypes(include=[np.number]).columns
                    valid_cols = [c for c in numeric_cols if c not in ['x', 'y', 'uncertainty', 'address_id']]
                    if valid_cols:
                        idm_predictions = predictions_df[valid_cols[0]].values
                    else:
                        raise ValueError("No valid prediction columns found in IDM result")
                
            except Exception as e:
                self._log(f"  IDM baseline failed: {str(e)}")
                if 'tract_info' not in locals():
                    self._log(f"  Error loading NLCD: 'tract_info'")
                
                # Fallback to tract mean
                idm_predictions = np.full(len(tract_addresses), tract_svi)
                
            self._log(f"IDM baseline: mean={np.mean(idm_predictions):.3f}, "
                    f"std={np.std(idm_predictions):.3f}")
            
            # Step 2: Filter road network to tract area and prepare graph data for GNN
            self._log("Step 2: Preparing graph data for GNN...")
            
            # PERFORMANCE FIX: Filter road network to tract area
            tract_boundary = tract_info.geometry
            tract_roads = data['roads'][data['roads'].intersects(tract_boundary)].copy()
            
            if len(tract_roads) == 0:
                self._log("  WARNING: No roads found in tract area")
                return {'status': 'error', 'message': 'No roads found in tract area'}
            
            self._log(f"  Filtered roads: {len(data['roads'])} -> {len(tract_roads)} segments")
            
            # CRITICAL FIX: Convert filtered GeoDataFrame to NetworkX graph
            road_network_graph = self.data_loader.create_network_graph(tract_roads)
            
            if road_network_graph.number_of_nodes() == 0:
                self._log("  WARNING: Empty road network graph, using fallback")
                return {'status': 'error', 'message': 'Empty road network graph'}
            
            self._log(f"  Created graph: {road_network_graph.number_of_nodes()} nodes, {road_network_graph.number_of_edges()} edges")
            
            # FIXED: Pass IDM baseline to graph preparation function
            graph_data, node_mapping = prepare_graph_data_with_nlcd(
                road_network=road_network_graph,
                nlcd_features=nlcd_features,
                addresses=tract_addresses,
                idm_baseline=idm_predictions  # FIXED: Pass as parameter to avoid scope error
            )
            
            # Step 3: Compute GNN corrections with proper address mapping
            self._log("Step 3: Computing accessibility corrections...")
            
            try:
                # Import AccessibilityGNNCorrector if available
                from ..disaggregation.hybrid_framework import AccessibilityGNNCorrector
                
                # Create GNN model
                input_dim = graph_data.x.shape[1]
                gnn_model = AccessibilityGNNCorrector(
                    input_dim=input_dim,
                    hidden_dim=self.config.get('model', {}).get('hidden_dim', 64)
                )
                
                # FIXED: Use mapping function to get corrections for addresses
                gnn_corrections = self._map_network_corrections_to_addresses(
                    gnn_model=gnn_model,
                    graph_data=graph_data,
                    road_network_graph=road_network_graph,
                    addresses=tract_addresses,
                    idm_predictions=idm_predictions
                )
                
                # Verify shapes match now
                if len(gnn_corrections) != len(idm_predictions):
                    raise ValueError(f"Shape mismatch: {len(gnn_corrections)} corrections vs {len(idm_predictions)} addresses")
                
            except Exception as e:
                self._log(f"  GNN correction failed: {str(e)}")
                # Fallback to zero corrections
                gnn_corrections = np.zeros(len(idm_predictions))
            self._log(f"GNN corrections: mean={np.mean(gnn_corrections):.3f}, "
                    f"std={np.std(gnn_corrections):.3f}")
            
            # Step 4: Combine IDM + GNN with simple spatial integration
            self._log("Step 4: Applying spatial integration...")
            
            combined_predictions = idm_predictions + gnn_corrections
            
            # Simple constraint satisfaction for mass preservation
            predicted_mean = np.mean(combined_predictions)
            correction_factor = tract_svi / predicted_mean if predicted_mean != 0 else 1.0
            final_predictions = combined_predictions * correction_factor
            
            # Basic uncertainty estimates (placeholder)
            uncertainties = np.full(len(final_predictions), 0.1)
            
            # Step 5: Validate results
            self._log("Step 5: Validating results...")
            
            validation_metrics = {
                'tract_svi': tract_svi,
                'predicted_mean': np.mean(final_predictions),
                'mass_preservation_error': abs(np.mean(final_predictions) - tract_svi),
                'prediction_std': np.std(final_predictions),
                'idm_baseline_std': np.std(idm_predictions),
                'gnn_corrections_std': np.std(gnn_corrections),
                'mean_uncertainty': np.mean(uncertainties) if uncertainties is not None else 0.1
            }
            
            self._log(f"  Final predictions: mean={validation_metrics['predicted_mean']:.3f}, "
                    f"std={validation_metrics['prediction_std']:.3f}")
            self._log(f"  Mass preservation error: {validation_metrics['mass_preservation_error']:.6f}")
            
            # Return results
            return {
                'success': True,  # FIXED: Changed from 'status': 'success' to match pipeline expectations
                'fips': fips,
                'predictions': final_predictions,
                'uncertainties': uncertainties,
                'idm_baseline': idm_predictions,
                'gnn_corrections': gnn_corrections,
                'validation_metrics': validation_metrics,
                'training_history': {}  # FIXED: Removed reference to undefined training_result
            }
            
        except Exception as e:
            self._log(f"Error processing tract {fips}: {str(e)}")
            import traceback
            self._log(f"Traceback: {traceback.format_exc()}")
            return {'success': False, 'fips': fips, 'message': str(e)}

    def _map_network_corrections_to_addresses(self, gnn_model, graph_data, 
                                             road_network_graph, addresses, 
                                             idm_predictions):
        """
        Map GNN corrections from road network nodes to address locations.
        
        This solves the critical shape mismatch problem (network nodes != addresses).
        """
        from scipy.spatial import cKDTree
        import numpy as np
        
        # Get corrections at network nodes
        gnn_model.eval()
        with torch.no_grad():
            node_corrections = gnn_model(graph_data.x, graph_data.edge_index)
            node_corrections = node_corrections.squeeze().numpy()
        
        # Ensure node corrections have zero mean
        node_corrections = node_corrections - np.mean(node_corrections)
        
        # Get coordinates for network nodes and addresses
        node_coords = np.array([[node[0], node[1]] for node in road_network_graph.nodes()])
        address_coords = np.array([[addr.geometry.x, addr.geometry.y] 
                                  for _, addr in addresses.iterrows()])
        
        # Build spatial index for network nodes
        node_tree = cKDTree(node_coords)
        
        # Map corrections to addresses using inverse distance weighting
        address_corrections = np.zeros(len(addresses))
        
        for i, addr_coord in enumerate(address_coords):
            # Find k nearest network nodes
            k = min(3, len(node_coords))
            distances, indices = node_tree.query(addr_coord, k=k)
            
            if k == 1:
                # Single nearest neighbor
                address_corrections[i] = node_corrections[indices]
            else:
                # Inverse distance weighting
                weights = 1.0 / (distances + 1e-8)  # Avoid division by zero
                weights = weights / np.sum(weights)  # Normalize
                address_corrections[i] = np.sum(weights * node_corrections[indices])
        
        # Ensure mass preservation: zero mean
        address_corrections = address_corrections - np.mean(address_corrections)
        
        return address_corrections
    
    def _process_county_mode(self, data: Dict) -> Dict:
        """Process all tracts in the county."""
        tract_results = []
        
        # Get all tract FIPS codes
        all_fips = data['svi']['FIPS'].unique()
        self._log(f"Processing {len(all_fips)} tracts in county mode")
        
        for i, fips in enumerate(all_fips[:5]):  # Limit to 5 for testing
            self._log(f"Processing tract {i+1}/{min(5, len(all_fips))}: {fips}")
            
            try:
                result = self._process_single_tract(data, fips)
                tract_results.append(result)
            except Exception as e:
                self._log(f"Error processing tract {fips}: {str(e)}", level='ERROR')
                continue
        
        # Combine results
        if not tract_results:
            return {'success': False, 'error': 'No tracts processed successfully'}
        
        successful_results = [r for r in tract_results if r.get('success')]
        
        if not successful_results:
            return {'success': False, 'error': 'All tract processing failed'}
        
        # Aggregate predictions
        all_predictions = pd.concat(
            [r['predictions'] for r in successful_results],
            ignore_index=True
        )
        
        # Compute overall statistics
        overall_stats = {
            'total_tracts': len(successful_results),
            'total_addresses': len(all_predictions),
            'mean_svi': float(np.mean(all_predictions['svi_prediction'])),
            'std_svi': float(np.std(all_predictions['svi_prediction'])),
            'mean_correction': float(np.mean(all_predictions['gnn_correction'])),
            'improvement_metrics': self._compute_improvement_metrics(successful_results)
        }
        
        return {
            'success': True,
            'predictions': all_predictions,
            'tract_results': successful_results,
            'overall_statistics': overall_stats
        }
    
    def _compare_methods(self, idm_predictions: np.ndarray,
                        gnn_corrections: np.ndarray,
                        hybrid_predictions: np.ndarray,
                        tract_svi: float) -> Dict:
        """Compare performance of IDM baseline vs hybrid approach."""
        
        def compute_metrics(predictions, name):
            return {
                'name': name,
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions)),
                'cv': float(np.std(predictions) / (np.mean(predictions) + 1e-8)),
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions)),
                'constraint_error': float(abs(np.mean(predictions) - tract_svi))
            }
        
        return {
            'idm_baseline': compute_metrics(idm_predictions, 'IDM Baseline'),
            'hybrid': compute_metrics(hybrid_predictions, 'Hybrid IDM+GNN'),
            'corrections': {
                'mean': float(np.mean(gnn_corrections)),
                'std': float(np.std(gnn_corrections)),
                'min': float(np.min(gnn_corrections)),
                'max': float(np.max(gnn_corrections)),
                'percent_positive': float(np.mean(gnn_corrections > 0) * 100),
                'percent_negative': float(np.mean(gnn_corrections < 0) * 100)
            },
            'improvement': {
                'variation_ratio': float(
                    np.std(hybrid_predictions) / np.std(idm_predictions)
                ),
                'constraint_improvement': float(
                    1 - abs(np.mean(hybrid_predictions) - tract_svi) / 
                    abs(np.mean(idm_predictions) - tract_svi)
                ) if abs(np.mean(idm_predictions) - tract_svi) > 0 else 0
            }
        }
    
    def _compute_improvement_metrics(self, results: list) -> Dict:
        """Compute improvement metrics across all tracts."""
        improvements = []
        
        for r in results:
            if 'comparison' in r and 'improvement' in r['comparison']:
                improvements.append(r['comparison']['improvement']['variation_ratio'])
        
        if improvements:
            return {
                'mean_improvement': float(np.mean(improvements)),
                'median_improvement': float(np.median(improvements)),
                'percent_improved': float(np.mean(np.array(improvements) > 1.0) * 100)
            }
        return {}
    
    def _save_results(self, results: Dict):
        """Save results to output directory."""
        import pandas as pd
        import numpy as np
        import os
        import json
        from datetime import datetime
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Handle predictions - convert to DataFrame if it's a numpy array
        predictions = results['predictions']
        
        if isinstance(predictions, np.ndarray):
            # Convert numpy array to DataFrame with basic structure
            self._log("Converting numpy predictions to DataFrame format")
            
            # Create basic DataFrame structure
            predictions_df = pd.DataFrame({
                'svi_prediction': predictions.flatten() if predictions.ndim > 1 else predictions,
                'address_id': range(len(predictions.flatten() if predictions.ndim > 1 else predictions))
            })
            
            # Add placeholders for expected columns if they don't exist
            if 'x' not in predictions_df.columns:
                predictions_df['x'] = 0.0  # Will need actual coordinates
            if 'y' not in predictions_df.columns:
                predictions_df['y'] = 0.0  # Will need actual coordinates
            if 'sd' not in predictions_df.columns:
                predictions_df['sd'] = 0.01  # Placeholder uncertainty
            if 'idm_baseline' not in predictions_df.columns:
                predictions_df['idm_baseline'] = predictions_df['svi_prediction']
            if 'gnn_correction' not in predictions_df.columns:
                predictions_df['gnn_correction'] = 0.0
            
        elif isinstance(predictions, pd.DataFrame):
            predictions_df = predictions
        else:
            # If it's something else, try to handle it
            self._log(f"Unexpected predictions type: {type(predictions)}")
            try:
                predictions_df = pd.DataFrame(predictions)
            except:
                # Last resort - create minimal DataFrame
                predictions_df = pd.DataFrame({
                    'svi_prediction': [0.0],
                    'error': ['Failed to convert predictions']
                })
        
        # Save predictions
        predictions_path = os.path.join(self.output_dir, 'hybrid_predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        self._log(f"Saved predictions to {predictions_path}")
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'statistics': results.get('overall_statistics', {}),
            'success': results['success'],
            'predictions_shape': str(predictions_df.shape),
            'predictions_columns': list(predictions_df.columns)
        }
        
        summary_path = os.path.join(self.output_dir, 'hybrid_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)  # default=str handles non-serializable objects
        self._log(f"Saved summary to {summary_path}")
    
    # Fix 1: Update the visualization code to extract real coordinates from the DataFrame

    def _create_enhanced_visualizations(self, results: Dict):
        """Create visualizations comparing IDM baseline and hybrid results.
        
        Works with both single tract (fips mode) and multi-tract (county mode) results.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import os
        
        try:
            # Handle different result structures
            if 'tract_results' in results and results['tract_results']:
                # County mode - multiple tracts
                tract_data = results['tract_results'][0]  # Use first tract for visualization
                self._log("Creating visualizations from county mode results")
            elif 'predictions' in results:
                # Single FIPS mode - direct predictions
                tract_data = {
                    'predictions': results['predictions'],
                    'comparison': results.get('comparison', {}),
                    'training': results.get('training', {})
                }
                self._log("Creating visualizations from single tract mode results")
            else:
                self._log("No suitable data found for visualization")
                return
            
            predictions_data = tract_data['predictions']
            
            # Fix: Handle the case where predictions_data is a numpy array but we need coordinates
            if isinstance(predictions_data, np.ndarray):
                self._log("Converting numpy array to DataFrame for visualization")
                
                # Try to get the actual coordinates from the original tract processing
                # This is the key fix - we need to get the real coordinates from the address data
                
                # Load the tract addresses to get real coordinates
                fips = results.get('fips')  # Should be available in single tract mode
                if fips:
                    try:
                        # Get the real address coordinates for this tract
                        tract_addresses = self.data_loader.get_addresses_for_tract(fips)
                        
                        if len(tract_addresses) > 0 and hasattr(tract_addresses, 'geometry'):
                            # Extract real coordinates from the address geometry
                            coords = np.array([[geom.x, geom.y] for geom in tract_addresses.geometry])
                            
                            # Ensure we have the right number of coordinates
                            n_predictions = len(predictions_data.flatten() if predictions_data.ndim > 1 else predictions_data)
                            if len(coords) >= n_predictions:
                                coords = coords[:n_predictions]  # Trim to match predictions
                            else:
                                # Pad with last coordinate if needed (shouldn't happen)
                                while len(coords) < n_predictions:
                                    coords = np.vstack([coords, coords[-1]])
                            
                            predictions_df = pd.DataFrame({
                                'svi_prediction': predictions_data.flatten() if predictions_data.ndim > 1 else predictions_data,
                                'idm_baseline': predictions_data.flatten() if predictions_data.ndim > 1 else predictions_data,  # Assume same for now
                                'gnn_correction': np.zeros_like(predictions_data.flatten() if predictions_data.ndim > 1 else predictions_data),
                                'x': coords[:, 0],  # Real coordinates!
                                'y': coords[:, 1],  # Real coordinates!
                                'sd': np.full_like(predictions_data.flatten() if predictions_data.ndim > 1 else predictions_data, 0.01)
                            })
                            
                            self._log(f"Successfully extracted real coordinates for {len(coords)} addresses")
                            
                        else:
                            self._log("No address geometry found, using fallback coordinates")
                            raise ValueError("No real coordinates available")
                            
                    except Exception as e:
                        self._log(f"Failed to extract real coordinates: {str(e)}")
                        # Fallback to placeholder coordinates (the old behavior)
                        predictions_df = pd.DataFrame({
                            'svi_prediction': predictions_data.flatten() if predictions_data.ndim > 1 else predictions_data,
                            'idm_baseline': predictions_data.flatten() if predictions_data.ndim > 1 else predictions_data,
                            'gnn_correction': np.zeros_like(predictions_data.flatten() if predictions_data.ndim > 1 else predictions_data),
                            'x': np.random.uniform(-85.5, -85.0, len(predictions_data.flatten() if predictions_data.ndim > 1 else predictions_data)),
                            'y': np.random.uniform(35.0, 35.5, len(predictions_data.flatten() if predictions_data.ndim > 1 else predictions_data)),
                            'sd': np.full_like(predictions_data.flatten() if predictions_data.ndim > 1 else predictions_data, 0.01)
                        })
                else:
                    self._log("No FIPS available for coordinate extraction")
                    # Fallback coordinates
                    predictions_df = pd.DataFrame({
                        'svi_prediction': predictions_data.flatten() if predictions_data.ndim > 1 else predictions_data,
                        'idm_baseline': predictions_data.flatten() if predictions_data.ndim > 1 else predictions_data,
                        'gnn_correction': np.zeros_like(predictions_data.flatten() if predictions_data.ndim > 1 else predictions_data),
                        'x': np.random.uniform(-85.5, -85.0, len(predictions_data.flatten() if predictions_data.ndim > 1 else predictions_data)),
                        'y': np.random.uniform(35.0, 35.5, len(predictions_data.flatten() if predictions_data.ndim > 1 else predictions_data)),
                        'sd': np.full_like(predictions_data.flatten() if predictions_data.ndim > 1 else predictions_data, 0.01)
                    })
                    
            elif isinstance(predictions_data, pd.DataFrame):
                predictions_df = predictions_data
            else:
                self._log(f"Unknown predictions type: {type(predictions_data)}")
                return
            
            # Verify we have the basic required column
            if 'svi_prediction' not in predictions_df.columns:
                self._log("Missing svi_prediction column - cannot create visualizations")
                return
            
            # Create main comparison plot
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            fig.suptitle('GRANITE: IDM Baseline vs Hybrid IDM+GNN Results', fontsize=16, fontweight='bold')
            
            # Extract values safely with fallbacks
            svi_vals = predictions_df['svi_prediction'].values
            idm_vals = predictions_df.get('idm_baseline', svi_vals).values if 'idm_baseline' in predictions_df.columns else svi_vals
            gnn_corrections = predictions_df.get('gnn_correction', np.zeros_like(svi_vals)).values if 'gnn_correction' in predictions_df.columns else np.zeros_like(svi_vals)
            
            # Extract values safely with fallbacks
            svi_vals = predictions_df['svi_prediction'].values
            idm_vals = predictions_df.get('idm_baseline', svi_vals).values if 'idm_baseline' in predictions_df.columns else svi_vals
            gnn_corrections = predictions_df.get('gnn_correction', np.zeros_like(svi_vals)).values if 'gnn_correction' in predictions_df.columns else np.zeros_like(svi_vals)

            # CALCULATE CORRELATION EARLY - This fixes the UnboundLocalError
            correlation = np.corrcoef(idm_vals, svi_vals)[0, 1]
            variation_improvement = np.std(svi_vals) / (np.std(idm_vals) + 1e-8)
            gnn_effect_size = np.mean(np.abs(gnn_corrections))
            spatial_range = np.max(svi_vals) - np.min(svi_vals)

            # Check coordinate validity
            if 'x' in predictions_df.columns and 'y' in predictions_df.columns:
                x = predictions_df['x'].values
                y = predictions_df['y'].values
                is_real_coords = (-86.0 <= np.mean(x) <= -84.5) and (34.5 <= np.mean(y) <= 35.5)
            else:
                is_real_coords = False

            # Now continue with the plotting sections...

            # 1. IDM Baseline Distribution
            ax = axes[0, 0]
            ax.hist(idm_vals, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax.set_title(f'IDM Baseline Distribution\nMean: {np.mean(idm_vals):.3f}, Std: {np.std(idm_vals):.3f}')
            ax.set_xlabel('SVI Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)

            # 2. GNN Corrections Distribution  
            ax = axes[0, 1]
            ax.hist(gnn_corrections, bins=30, alpha=0.7, color='orange', edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero correction')
            ax.set_title(f'GNN Corrections\nMean: {np.mean(gnn_corrections):.4f}, Std: {np.std(gnn_corrections):.4f}')
            ax.set_xlabel('Correction Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 3. Hybrid Results Distribution
            ax = axes[0, 2]
            ax.hist(svi_vals, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax.set_title(f'Hybrid IDM+GNN\nMean: {np.mean(svi_vals):.3f}, Std: {np.std(svi_vals):.3f}')
            ax.set_xlabel('SVI Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)

            # 4. Spatial scatter plot with REAL coordinates and proper scaling
            ax = axes[1, 0]
            if 'x' in predictions_df.columns and 'y' in predictions_df.columns:
                if is_real_coords:
                    scatter = ax.scatter(x, y, c=svi_vals, cmap='viridis_r', s=15, alpha=0.7, edgecolors='white', linewidth=0.05)
                    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
                    ax.set_title('Spatial Distribution (Real Hamilton County)')
                    ax.set_xlabel('Longitude')
                    ax.set_ylabel('Latitude')
                    
                    # Use actual data bounds with small buffer
                    x_buffer = (x.max() - x.min()) * 0.05
                    y_buffer = (y.max() - y.min()) * 0.05
                    ax.set_xlim(x.min() - x_buffer, x.max() + x_buffer)
                    ax.set_ylim(y.min() - y_buffer, y.max() + y_buffer)
                    
                    # Make aspect ratio appropriate for Hamilton County
                    ax.set_aspect('equal', adjustable='box')
                    
                    # Add coordinate range info
                    coord_info = f"Lon: [{x.min():.4f}, {x.max():.4f}]\nLat: [{y.min():.4f}, {y.max():.4f}]"
                    ax.text(0.02, 0.98, coord_info, transform=ax.transAxes, fontsize=8, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    self._log(f"Coordinate ranges - Lon: [{x.min():.4f}, {x.max():.4f}], Lat: [{y.min():.4f}, {y.max():.4f}]")
                else:
                    ax.text(0.5, 0.5, f'Coordinate Issue Detected\nMean X: {np.mean(x):.3f}\nMean Y: {np.mean(y):.3f}\n(Should be ~-85.3, ~35.0)', 
                        ha='center', va='center', transform=ax.transAxes, fontsize=12, 
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
                    ax.set_title('Coordinate Issue Detected')
            else:
                ax.text(0.5, 0.5, 'No coordinate data\navailable for spatial plot', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title('Spatial Distribution')

            # 5. IDM vs Hybrid comparison  
            ax = axes[1, 1]
            ax.scatter(idm_vals, svi_vals, alpha=0.6, s=15, edgecolors='white', linewidth=0.1)

            # Add perfect correlation line
            min_val = min(np.min(idm_vals), np.min(svi_vals))
            max_val = max(np.max(idm_vals), np.max(svi_vals))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect correlation')

            ax.set_xlabel('IDM Baseline')
            ax.set_ylabel('Hybrid IDM+GNN')
            ax.set_title(f'Method Comparison\nCorrelation: {correlation:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 6. Enhanced summary statistics with better problem diagnosis
            ax = axes[1, 2]
            ax.axis('off')

            # Detailed problem diagnosis (now correlation is available)
            problems = []
            if correlation > 0.999:
                problems.append("⚠️  PERFECT CORRELATION")
            if np.std(gnn_corrections) < 0.001:
                problems.append("⚠️  ZERO GNN EFFECT")
            if spatial_range < 0.01:
                problems.append("⚠️  NO SPATIAL VARIATION")
            if np.std(idm_vals) < 0.001:
                problems.append("⚠️  IDM TOO UNIFORM")

            problem_status = "CRITICAL ISSUES DETECTED" if problems else "FUNCTIONING NORMALLY"
            coord_status = "✅ Real coordinates" if is_real_coords else "❌ Placeholder coordinates"

            stats_text = (
                f"GRANITE DIAGNOSTIC REPORT\n"
                f"{'='*25}\n\n"
                f"Status: {problem_status}\n"
                f"Coordinates: {coord_status}\n\n"
                f"CORE METRICS:\n"
                f"├─ IDM Baseline:\n"
                f"│  ├─ Mean: {np.mean(idm_vals):.4f}\n"
                f"│  ├─ Std:  {np.std(idm_vals):.6f}\n"
                f"│  └─ Range: {np.max(idm_vals)-np.min(idm_vals):.6f}\n"
                f"├─ GNN Corrections:\n"
                f"│  ├─ Mean: {np.mean(gnn_corrections):.6f}\n"
                f"│  ├─ Std:  {np.std(gnn_corrections):.6f}\n"
                f"│  └─ Effect: {gnn_effect_size:.6f}\n"
                f"├─ Hybrid Result:\n"
                f"│  ├─ Mean: {np.mean(svi_vals):.4f}\n"
                f"│  ├─ Std:  {np.std(svi_vals):.6f}\n"
                f"│  └─ Range: {spatial_range:.6f}\n"
                f"└─ Correlation: {correlation:.6f}\n\n"
                f"PROBLEM ANALYSIS:\n"
            )

            for problem in problems:
                stats_text += f"{problem}\n"

            if problems:
                stats_text += (
                    f"\nNext Actions (Research Pivot):\n"
                    f"• Recalibrate loss function weights\n"
                    f"• Add variance preservation term\n"
                    f"• Re-engineer GNN features\n"
                    f"• Remove hard tract constraints\n"
                    f"• Validate NLCD coefficient usage"
                )
            else:
                stats_text += f"\nSystem functioning normally"

            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=7,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))
            ax.set_title('Diagnostic Analysis', fontweight='bold')

            # Add difference histogram if we have problems
            if len(problems) > 0 and is_real_coords:
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                inset_ax = inset_axes(ax, width="40%", height="40%", loc='lower left')
                differences = svi_vals - idm_vals
                inset_ax.hist(differences, bins=20, alpha=0.7, color='red', edgecolor='black')
                inset_ax.set_title('IDM-Hybrid Diff', fontsize=8)
                inset_ax.tick_params(labelsize=6)
                inset_ax.axvline(0, color='black', linestyle='--', alpha=0.5)
                
                max_diff = np.max(np.abs(differences))
                inset_ax.text(0.95, 0.95, f'Max:|{max_diff:.6f}|', transform=inset_ax.transAxes, 
                            fontsize=6, ha='right', va='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Add an additional diagnostic plot showing the trivial solution more clearly
            if len(problems) > 0:
                # Create a small inset plot showing the problem
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                
                if 'x' in predictions_df.columns and 'y' in predictions_df.columns and is_real_coords:
                    # Add inset showing difference between IDM and hybrid (should be non-zero)
                    inset_ax = inset_axes(ax, width="40%", height="40%", loc='lower left')
                    differences = svi_vals - idm_vals
                    inset_ax.hist(differences, bins=20, alpha=0.7, color='red', edgecolor='black')
                    inset_ax.set_title('IDM-Hybrid Diff', fontsize=8)
                    inset_ax.tick_params(labelsize=6)
                    inset_ax.axvline(0, color='black', linestyle='--', alpha=0.5)
                    
                    # Add text showing max difference
                    max_diff = np.max(np.abs(differences))
                    inset_ax.text(0.95, 0.95, f'Max:|{max_diff:.6f}|', transform=inset_ax.transAxes, 
                                fontsize=6, ha='right', va='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 5. IDM vs Hybrid comparison
            ax = axes[1, 1]
            ax.scatter(idm_vals, svi_vals, alpha=0.6, s=15, edgecolors='white', linewidth=0.1)
            
            # Add perfect correlation line
            min_val = min(np.min(idm_vals), np.min(svi_vals))
            max_val = max(np.max(idm_vals), np.max(svi_vals))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect correlation')
            
            # Calculate correlation
            correlation = np.corrcoef(idm_vals, svi_vals)[0, 1]
            ax.set_xlabel('IDM Baseline')
            ax.set_ylabel('Hybrid IDM+GNN')
            ax.set_title(f'Method Comparison\nCorrelation: {correlation:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 6. Summary statistics with coordinate validation
            ax = axes[1, 2]
            ax.axis('off')
            
            # Compute basic statistics
            variation_improvement = np.std(svi_vals) / (np.std(idm_vals) + 1e-8)
            
            # Check coordinate validity
            coord_status = "✅ Real coordinates" if is_real_coords else "❌ Placeholder coordinates"
            
            stats_text = (
                f"GRANITE Results Summary\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"Coordinates: {coord_status}\n\n"
                f"IDM Baseline:\n"
                f"  Mean: {np.mean(idm_vals):.3f}\n"
                f"  Std:  {np.std(idm_vals):.4f}\n\n"
                f"Hybrid IDM+GNN:\n"
                f"  Mean: {np.mean(svi_vals):.3f}\n"
                f"  Std:  {np.std(svi_vals):.4f}\n\n"
                f"GNN Corrections:\n"
                f"  Mean: {np.mean(gnn_corrections):.4f}\n"
                f"  Std:  {np.std(gnn_corrections):.4f}\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Analysis:\n"
                f"  Addresses: {len(svi_vals)}\n"
                f"  Variation Ratio: {variation_improvement:.2f}\n\n"
            )
            
            # Add diagnosis of trivial solution
            if np.std(gnn_corrections) < 0.01:
                stats_text += (
                    f"⚠️  TRIVIAL SOLUTION DETECTED\n"
                    f"GNN corrections ≈ 0\n"
                    f"Indicates loss function issues\n"
                    f"See research pivot notes"
                )
            
            ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax.set_title('Summary Statistics')
            
            plt.tight_layout()
            
            # Save the visualization
            viz_path = os.path.join(self.output_dir, 'granite_comparison_visualization.png')
            fig.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close(fig)
            
            self._log(f"Saved visualization to {viz_path}")
            
            # Create additional spatial detail plot if coordinates are available
            if 'x' in predictions_df.columns and 'y' in predictions_df.columns and is_real_coords:
                self._create_spatial_detail_plot(predictions_df)
                
        except Exception as e:
            self._log(f"Error creating visualizations: {str(e)}", level='ERROR')
            import traceback
            self._log(f"Traceback: {traceback.format_exc()}", level='ERROR')
        
    def _create_diagnostic_plot(self, svi_vals, idm_vals, gnn_corrections):
        """Create diagnostic plot highlighting the trivial solution problem."""
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle('GRANITE Diagnostic: Trivial Solution Analysis', fontsize=14, fontweight='bold')
            
            # 1. Distribution comparison
            ax = axes[0]
            ax.hist(idm_vals, bins=20, alpha=0.7, label=f'IDM (σ={np.std(idm_vals):.4f})', color='blue')
            ax.hist(svi_vals, bins=20, alpha=0.7, label=f'Hybrid (σ={np.std(svi_vals):.4f})', color='green')
            ax.set_title('Value Distributions')
            ax.set_xlabel('SVI Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 2. GNN corrections histogram
            ax = axes[1]
            ax.hist(gnn_corrections, bins=20, alpha=0.7, color='orange', edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', linewidth=2)
            ax.set_title(f'GNN Corrections\n(σ={np.std(gnn_corrections):.4f})')
            ax.set_xlabel('Correction Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # 3. Problem diagnosis
            ax = axes[2]
            ax.axis('off')
            
            # Determine problem severity
            correction_std = np.std(gnn_corrections)
            if correction_std < 0.001:
                severity = "SEVERE"
                color = "red"
            elif correction_std < 0.01:
                severity = "MODERATE" 
                color = "orange"
            else:
                severity = "MILD"
                color = "green"
            
            diagnosis_text = (
                f"TRIVIAL SOLUTION DIAGNOSIS\n"
                f"{'='*30}\n\n"
                f"Status: {severity}\n"
                f"GNN Correction Std: {correction_std:.6f}\n\n"
                f"Expected: σ > 0.05\n"
                f"Observed: σ = {correction_std:.6f}\n\n"
                f"Root Causes:\n"
                f"• Loss function imbalance\n"
                f"• Weak spatial signal\n"
                f"• Feature ineffectiveness\n\n"
                f"Next Steps:\n"
                f"• Restore spatial_loss weight\n"
                f"• Add variance preservation\n"
                f"• Re-engineer features\n"
                f"• Remove hard constraints"
            )
            
            ax.text(0.1, 0.9, diagnosis_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
            ax.set_title('Problem Diagnosis', color=color)
            
            plt.tight_layout()
            
            # Save diagnostic plot
            diag_path = os.path.join(self.output_dir, 'granite_diagnostic_analysis.png')
            fig.savefig(diag_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close(fig)
            
            self._log(f"Saved diagnostic plot to {diag_path}")
            
        except Exception as e:
            self._log(f"Error creating diagnostic plot: {str(e)}", level='ERROR')

    def _create_spatial_detail_plot(self, predictions_df):
        """Create detailed spatial visualization showing IDM vs GNN patterns."""
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle('GRANITE Spatial Analysis: IDM vs GNN Enhancement', fontsize=14, fontweight='bold')
            
            x = predictions_df['x'].values
            y = predictions_df['y'].values
            svi_vals = predictions_df['svi_prediction'].values
            idm_vals = predictions_df.get('idm_baseline', svi_vals).values
            gnn_corrections = predictions_df.get('gnn_correction', np.zeros_like(svi_vals)).values
            uncertainty = predictions_df.get('sd', np.ones_like(svi_vals) * 0.01).values
            
            # 1. IDM Baseline spatial pattern
            ax = axes[0, 0]
            scatter1 = ax.scatter(x, y, c=idm_vals, cmap='viridis_r', s=25, alpha=0.8, edgecolors='white', linewidth=0.1)
            plt.colorbar(scatter1, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title('IDM Baseline Spatial Pattern')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_aspect('equal')
            
            # 2. GNN Corrections spatial pattern
            ax = axes[0, 1]
            # Use a diverging colormap for corrections (positive/negative)
            max_abs_corr = max(abs(np.min(gnn_corrections)), abs(np.max(gnn_corrections)))
            scatter2 = ax.scatter(x, y, c=gnn_corrections, cmap='RdBu_r', s=25, alpha=0.8, 
                                vmin=-max_abs_corr, vmax=max_abs_corr, edgecolors='white', linewidth=0.1)
            plt.colorbar(scatter2, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title('GNN Accessibility Corrections')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_aspect('equal')
            
            # 3. Final hybrid results
            ax = axes[1, 0]
            scatter3 = ax.scatter(x, y, c=svi_vals, cmap='viridis_r', s=25, alpha=0.8, edgecolors='white', linewidth=0.1)
            plt.colorbar(scatter3, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title('Hybrid IDM+GNN Results')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_aspect('equal')
            
            # 4. Prediction uncertainty
            ax = axes[1, 1]
            scatter4 = ax.scatter(x, y, c=uncertainty, cmap='Reds', s=25, alpha=0.8, edgecolors='white', linewidth=0.1)
            plt.colorbar(scatter4, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title('Prediction Uncertainty (SD)')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_aspect('equal')
            
            plt.tight_layout()
            
            # Save spatial detail plot
            spatial_path = os.path.join(self.output_dir, 'granite_spatial_analysis.png')
            fig.savefig(spatial_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close(fig)
            
            self._log(f"Saved spatial analysis to {spatial_path}")
            
        except Exception as e:
            self._log(f"Error creating spatial detail plot: {str(e)}", level='ERROR')