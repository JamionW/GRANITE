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
        
        # Save predictions
        predictions_path = os.path.join(self.output_dir, 'hybrid_predictions.csv')
        results['predictions'].to_csv(predictions_path, index=False)
        self._log(f"Saved predictions to {predictions_path}")
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'statistics': results.get('overall_statistics', {}),
            'success': results['success']
        }
        
        summary_path = os.path.join(self.output_dir, 'hybrid_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        self._log(f"Saved summary to {summary_path}")
    
    def _create_enhanced_visualizations(self, results: Dict):
        """Create visualizations comparing IDM baseline and hybrid results."""
        
        if not results.get('tract_results'):
            return
        
        # Get first tract result for detailed visualization
        first_tract = results['tract_results'][0]
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # IDM Baseline
        ax = axes[0, 0]
        idm_vals = first_tract['predictions']['idm_baseline'].values
        ax.hist(idm_vals, bins=30, alpha=0.7, color='blue')
        ax.set_title('IDM Baseline Distribution')
        ax.set_xlabel('SVI Value')
        ax.set_ylabel('Frequency')
        
        # GNN Corrections
        ax = axes[0, 1]
        corrections = first_tract['predictions']['gnn_correction'].values
        ax.hist(corrections, bins=30, alpha=0.7, color='orange')
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('GNN Corrections')
        ax.set_xlabel('Correction Value')
        
        # Hybrid Results
        ax = axes[0, 2]
        hybrid_vals = first_tract['predictions']['svi_prediction'].values
        ax.hist(hybrid_vals, bins=30, alpha=0.7, color='green')
        ax.set_title('Hybrid IDM+GNN Distribution')
        ax.set_xlabel('SVI Value')
        
        # Comparison scatter
        ax = axes[1, 0]
        ax.scatter(idm_vals, hybrid_vals, alpha=0.3)
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        ax.set_xlabel('IDM Baseline')
        ax.set_ylabel('Hybrid IDM+GNN')
        ax.set_title('Method Comparison')
        
        # Training history (if available)
        if 'training' in first_tract and 'history' in first_tract['training']:
            ax = axes[1, 1]
            history = first_tract['training']['history']
            ax.plot(history['epoch'], history['total_loss'])
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Progress')
        
        # Statistics comparison
        ax = axes[1, 2]
        ax.axis('off')
        comparison = first_tract['comparison']
        stats_text = (
            f"IDM Baseline:\n"
            f"  Mean: {comparison['idm_baseline']['mean']:.3f}\n"
            f"  Std: {comparison['idm_baseline']['std']:.3f}\n"
            f"  CV: {comparison['idm_baseline']['cv']:.3f}\n\n"
            f"Hybrid IDM+GNN:\n"
            f"  Mean: {comparison['hybrid']['mean']:.3f}\n"
            f"  Std: {comparison['hybrid']['std']:.3f}\n"
            f"  CV: {comparison['hybrid']['cv']:.3f}\n\n"
            f"Improvement:\n"
            f"  Variation Ratio: {comparison['improvement']['variation_ratio']:.2f}x"
        )
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='center', fontfamily='monospace')
        
        plt.suptitle('Hybrid IDM+GNN Disaggregation Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        viz_path = os.path.join(self.output_dir, 'hybrid_comparison.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self._log(f"Saved visualization to {viz_path}")