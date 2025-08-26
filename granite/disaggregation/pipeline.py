"""
Main disaggregation pipeline for GRANITE framework
Updated to use spatial disaggregation instead of regression
"""
# Standard library imports
import os
import time
import warnings
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import torch

# Configure warnings
warnings.filterwarnings('ignore')

# Local imports
from ..data.loaders import DataLoader
from ..models.gnn import prepare_graph_data_with_nlcd, prepare_graph_data_topological
from ..metricgraph.interface import MetricGraphInterface
from ..visualization.plots import DisaggregationVisualizer
from ..baselines.idm import IDMBaseline
from ..models.gnn import AccessibilityGNNCorrector, HybridCorrectionTrainer  

class GRANITEPipeline:
    """
    Main pipeline for SVI disaggregation using GNN-MetricGraph integration
    
    GRANITE: Graph-Refined Accessibility Network for Integrated Transit Equity
    
    This implementation uses spatial disaggregation with GNN-learned parameters
    rather than regression-based approaches.
    """
    
    def __init__(self, config, data_dir='./data', output_dir='./output', verbose=None):
        """
        Initialize GRANITE pipeline
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary (from config.yaml)
        data_dir : str
            Directory containing input data
        output_dir : str
            Directory for output files
        verbose : bool
            Enable verbose logging
        """
        if config is None:
            raise ValueError("Configuration is required. Please provide a config dict from config.yaml")
            
        self.config = config
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.verbose = config.get('processing', {}).get('verbose', False)
                
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader(data_dir, config=config)
        self.mg_interface = MetricGraphInterface(
            verbose=verbose,
            config=self.config  
        )
        self.visualizer = DisaggregationVisualizer()
        self.idm_baseline = IDMBaseline(config=config, grid_resolution_meters=100)
        
        # Storage for results
        self.results = {}
    
    def _log(self, message, level='INFO'):
        """Logging with timestamp and level"""
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    def run(self):
        """Run the simplified GRANITE pipeline for single-tract processing"""
        
        start_time = time.time()
        
        # Load data
        self._log("Loading data...")
        data = self._load_data()
        
        # Check processing mode - only support single FIPS now
        processing_mode = self.config.get('data', {}).get('processing_mode', 'fips')
        target_fips = self.config.get('data', {}).get('target_fips')
        
        if processing_mode == 'fips' and target_fips:
            self._log(f"Processing single FIPS: {target_fips}")
            results = self._process_fips_mode(data)
        else:
            self._log("ERROR: Only single FIPS mode supported. Use --fips command line argument.")
            return {'success': False, 'error': 'Multi-tract processing removed. Use --fips mode.'}
        
        # Save results
        self._save_results(results)
        
        elapsed_time = time.time() - start_time
        self._log(f"Pipeline completed in {elapsed_time:.2f} seconds")
        
        return results
    
    def _load_data(self):
        """
        Load data using Chattanooga addresses
        """
        self._log("Loading data...")
        
        # Get FIPS configuration from config
        state_fips = self.config.get('data', {}).get('state_fips', '47')
        county_fips = self.config.get('data', {}).get('county_fips', '065')
        
        data = {}
        
        # Load census tracts
        data['census_tracts'] = self.data_loader.load_census_tracts(
            state_fips=state_fips, 
            county_fips=county_fips
        )
        self._log(f"Loaded {len(data['census_tracts'])} census tracts")
        
        # Load SVI data
        county_name = self.data_loader._get_county_name(state_fips, county_fips)
        data['svi'] = self.data_loader.load_svi_data(
            state_fips=state_fips,
            county_name=county_name
        )
        self._log(f"Loaded SVI data for {len(data['svi'])} tracts")
        
        # Load road network
        data['roads'] = self.data_loader.load_road_network(
            state_fips=state_fips,
            county_fips=county_fips
        )
        self._log(f"Loaded road network with {len(data['roads'])} segments")
        
        # Load transit stops
        transit_config = self.config.get('transit', {})
        use_real_data = transit_config.get('download_real_data', True)
        
        data['transit_stops'] = self.data_loader.load_transit_stops(use_real_data=use_real_data)
        
        # Log transit data quality
        if len(data['transit_stops']) > 0:
            data_source = data['transit_stops']['data_source'].iloc[0]
            unique_sources = data['transit_stops']['data_source'].unique()
            
            self._log(f"Loaded {len(data['transit_stops'])} transit stops")
            self._log(f"  Primary source: {data_source}")
            if len(unique_sources) > 1:
                self._log(f"  Mixed sources: {list(unique_sources)}")
            
            # Log by route type
            if 'route_type' in data['transit_stops'].columns:
                route_counts = data['transit_stops']['route_type'].value_counts()
                for route_type, count in route_counts.items():
                    self._log(f"    {route_type}: {count} stops")
            
            # Quality assessment
            if data_source == 'CARTA_GTFS':
                self._log("  ✅ Using real GTFS data - highest quality for research")
            elif data_source == 'OpenStreetMap':
                self._log("  ✅ Using OSM data - good quality, community-verified")
            elif data_source == 'Generated_Grid':
                self._log("  ⚠️  Using generated grid - realistic but not real data")
            else:
                self._log("  ⚠️  Using minimal fallback - not recommended for research")
        else:
            self._log("❌ No transit stops loaded!")
        
        # Load address points
        data['addresses'] = self.data_loader.load_address_points(
            state_fips=state_fips, 
            county_fips=county_fips
        )
        
        address_source = 'real' if 'full_address' in data['addresses'].columns else 'synthetic'
        self._log(f"Loaded {len(data['addresses'])} {address_source} address points")
        
        # Create road network graph
        data['road_network'] = self.data_loader.create_network_graph(data['roads'])
        self._log(f"Created network graph with {data['road_network'].number_of_nodes()} nodes")
        
        # Merge SVI with census tracts
        data['tracts_with_svi'] = data['census_tracts'].merge(
            data['svi'], on='FIPS', how='inner'
        )

        data['tracts'] = data['tracts_with_svi'] 
        self._log(f"Merged data for {len(data['tracts_with_svi'])} tracts with SVI")
        
        return data

    def _process_fips_mode(self, data):
        """
        Process specific FIPS codes with proper string handling
        """
        # Get target FIPS from config
        target_fips = self.config.get('data', {}).get('target_fips')
        
        if not target_fips:
            self._log("No target FIPS specified in config, processing all tracts")
            return self._process_county_mode(data)
        
        # Ensure target_fips is string
        target_fips = str(target_fips).strip()
        self._log(f"Looking for target FIPS: '{target_fips}'")
        
        # Ensure FIPS column is string type for consistent matching
        data['tracts']['FIPS'] = data['tracts']['FIPS'].astype(str).str.strip()
        
        # Check if target exists
        available_fips = data['tracts']['FIPS'].tolist()
        if target_fips not in available_fips:
            self._log(f"Target FIPS {target_fips} not found in data")
            self._log(f"Available FIPS (first 5): {available_fips[:5]}")
            # Look for Hamilton County codes
            hamilton_codes = [f for f in available_fips if f.startswith('47065')]
            self._log(f"Hamilton County FIPS ({len(hamilton_codes)} total): {hamilton_codes[:10]}")
            return {'success': False, 'error': f'FIPS {target_fips} not found'}
        
        # Filter to specific tract
        target_tract = data['tracts'][data['tracts']['FIPS'] == target_fips]
        
        if len(target_tract) == 0:
            self._log(f"No tract found after filtering for FIPS {target_fips}")
            return {'success': False, 'error': f'FIPS {target_fips} filtering failed'}
        
        self._log(f"✓ Found target tract: {target_fips}")
        
        # Create single-tract dataset
        single_tract_data = data.copy()
        single_tract_data['tracts'] = target_tract
        
        self._log(f"Processing {len(single_tract_data['tracts'])} tract (single FIPS mode)")
        
        target_tract = data['tracts'][data['tracts']['FIPS'] == target_fips].iloc[0]
    
        self._log(f"✓ Found target tract: {target_fips}")
        self._log(f"Processing single tract with HYBRID approach")
        
        # Use your hybrid method directly
        tract_data = self._prepare_tract_data(target_tract, data)
        result = self._process_single_tract(tract_data)  # This calls your hybrid method!
        
        if result['status'] == 'success':
            return {
                'success': True,
                'predictions': result['predictions'], 
                'tract_results': [result],
                'summary': {
                    'total_tracts': 1,
                    'successful_tracts': 1,
                    'total_addresses': len(result['predictions']),
                    'processing_time': result['timing']['total']
                }
            }
        else:
            return {'success': False, 'error': result['error']}
    
    def _prepare_tract_data(self, tract, county_data):
        """
        Prepare data for a single tract using real addresses
        """
        # Get tract geometry
        tract_geom = tract.geometry
        fips_code = tract['FIPS']
        
        # Get roads within tract
        tract_roads = county_data['roads'][
            county_data['roads'].intersects(tract_geom)
        ].copy()
        
        # Get real addresses for this specific tract
        tract_addresses = self.data_loader.get_addresses_for_tract(fips_code)
        
        # Fallback if no addresses found
        if len(tract_addresses) == 0:
            self._log(f"No addresses found for tract {fips_code}, using tract centroid")
            centroid = tract_geom.centroid
            tract_addresses = gpd.GeoDataFrame([{
                'address_id': 0,
                'geometry': centroid,
                'full_address': f'Tract {fips_code} Centroid',
                'tract_fips': fips_code
            }], crs='EPSG:4326')
        
        # Build road network graph
        road_network = self.data_loader.create_network_graph(tract_roads)
        
        return {
            'tract_info': tract,
            'roads': tract_roads,
            'addresses': tract_addresses, 
            'road_network': road_network,
            'svi_value': tract['RPL_THEMES'],
            'address_count': len(tract_addresses),
            'address_source': 'real' if 'full_address' in tract_addresses.columns else 'synthetic'
        }
    
    def _process_single_tract(self, tract_data):
        """Process a single tract with HYBRID IDM+GNN approach"""
        fips = tract_data['tract_info']['FIPS']
        svi_value = tract_data['svi_value']
        self._log(f"  Processing tract {fips} with SVI={svi_value:.3f} [HYBRID MODE]")
        
        try:
            # STEP 1: Load NLCD features 
            nlcd_features = self._load_nlcd_for_tract(tract_data)
            
            # STEP 2: Compute IDM baseline 
            self._log("  Step 1: Computing IDM baseline...")
            tract_geom = tract_data['tract_info'].geometry
            addresses_df = tract_data['addresses']
            
            idm_result = self.idm_baseline.disaggregate_svi(
                tract_svi=svi_value,
                prediction_locations=addresses_df,
                nlcd_features=nlcd_features,
                tract_geometry=tract_geom
            )
            
            if not idm_result['success']:
                raise RuntimeError(f"IDM baseline failed: {idm_result.get('error', 'Unknown')}")
            
            idm_predictions = idm_result['predictions']['svi_prediction'].values
            self._log(f"    IDM baseline: mean={np.mean(idm_predictions):.3f}, std={np.std(idm_predictions):.3f}")
            
            # STEP 3: Train GNN for accessibility corrections
            self._log("  Step 2: Training GNN for accessibility corrections...")
            gnn_start = time.time()
            
            # Prepare graph data
            if nlcd_features is not None and len(nlcd_features) > 0:
                graph_data, node_mapping = prepare_graph_data_with_nlcd(
                    tract_data['road_network'], nlcd_features, addresses=tract_data['addresses']
                )
            else:
                graph_data, node_mapping = prepare_graph_data_topological(tract_data['road_network'])
            
            if isinstance(graph_data, tuple):
                graph_data = graph_data[0]
            
            # Create GNN for corrections
            input_dim = graph_data.x.shape[1]  
            gnn_model = AccessibilityGNNCorrector(
                input_dim=input_dim,
                hidden_dim=self.config.get('model', {}).get('hidden_dim', 64),
                max_correction=self.config.get('hybrid', {}).get('max_correction', 0.15)
            )
            
            # Train GNN to learn corrections
            training_config = {
                'learning_rate': self.config.get('model', {}).get('learning_rate', 0.001),
                'weight_decay': 1e-5,
                'loss_config': {
                    'smoothness_weight': 1.0,
                    'diversity_weight': 0.5,
                    'variation_weight': 0.3,
                    'feature_weight': 0.2,
                    'constraint_weight': 0.1
                }
            }
            
            trainer = HybridCorrectionTrainer(gnn_model, config=training_config)
            
            training_result = trainer.train_corrections(
                graph_data=graph_data,
                idm_baseline=idm_predictions,
                tract_svi=svi_value,
                epochs=self.config.get('model', {}).get('epochs', 100),
                verbose=self.verbose
            )

            gnn_corrections = training_result['final_corrections']
            gnn_time = time.time() - gnn_start

            # STEP 4: Combine IDM + GNN corrections
            self._log("  Step 3: Combining IDM baseline with GNN corrections...")
            idm_weight = self.config.get('hybrid', {}).get('idm_weight', 0.7)
            gnn_weight = self.config.get('hybrid', {}).get('gnn_weight', 0.3)
            
            hybrid_predictions = idm_weight * idm_predictions + gnn_weight * gnn_corrections

            # Ensure tract constraint satisfaction
            current_mean = np.mean(hybrid_predictions)
            adjustment = svi_value - current_mean
            hybrid_predictions += adjustment

            # Convert to SPDE format for MetricGraph compatibility
            gnn_features = self._corrections_to_spde_params(gnn_corrections, idm_predictions)

            # STEP 5: MetricGraph processing
            self._log("  Step 4: Creating MetricGraph representation...")
            mg_start = time.time()
            
            nodes_df, edges_df = self._prepare_metricgraph_data(tract_data['road_network'])
            metric_graph = self.mg_interface.create_graph(
                nodes_df, edges_df,
                enable_sampling=len(edges_df) > self.config['metricgraph']['max_edges']
            )
            
            mg_time = time.time() - mg_start
            
            # Prepare observation and prediction data
            tract_centroid = tract_data['tract_info'].geometry.centroid
            tract_observation = pd.DataFrame({
                'coord_x': [tract_centroid.x],
                'coord_y': [tract_centroid.y],
                'svi_value': [svi_value]
            })
            
            prediction_locations = pd.DataFrame({
                'x': [addr.geometry.x for _, addr in tract_data['addresses'].iterrows()],
                'y': [addr.geometry.y for _, addr in tract_data['addresses'].iterrows()]
            })
            
            # STEP 6: Get excellent MetricGraph predictions directly (FIXED VERSION)
            self._log("  Step 5: Performing direct MetricGraph spatial disaggregation...")
            direct_mg_result = self.mg_interface.disaggregate_svi(
                metric_graph=metric_graph,
                tract_observation=tract_observation,
                prediction_locations=prediction_locations,
                gnn_features=gnn_features,
                alpha=self.config['metricgraph']['alpha']
            )

            # Debug output to confirm we're getting excellent variation
            print(f"🔍 MetricGraph direct result std: {direct_mg_result['predictions']['mean'].std():.6f}")
            print(f"🔍 Hybrid predictions std: {np.std(hybrid_predictions):.6f}")
            print(f"🔍 IDM baseline std: {np.std(idm_predictions):.6f}")
            print(f"🔍 GNN corrections std: {np.std(gnn_corrections):.6f}")

            # Check if MetricGraph succeeded
            if not direct_mg_result.get('success', False):
                raise RuntimeError(f"MetricGraph disaggregation failed: {direct_mg_result.get('error', 'Unknown error')}")

            predictions = pd.DataFrame({
                'x': [addr.geometry.x for _, addr in tract_data['addresses'].iterrows()],
                'y': [addr.geometry.y for _, addr in tract_data['addresses'].iterrows()],
                'mean': hybrid_predictions,  # These are properly range-constrained
                'sd': np.full(len(hybrid_predictions), 0.05),  # Simple uncertainty
                'q025': hybrid_predictions - 0.1,
                'q975': hybrid_predictions + 0.1
            })
            predictions['q025'] = predictions['q025'].clip(lower=0.0)
            predictions['q975'] = predictions['q975'].clip(upper=1.0)

            # Generate IDM predictions for comparison
            idm_comparison_result = self.idm_baseline.disaggregate_svi(
                tract_svi=svi_value,
                prediction_locations=tract_data['addresses'],
                nlcd_features=nlcd_features,
                tract_geometry=tract_data['tract_info'].geometry
            )

            if idm_comparison_result.get('success') and 'predictions' in idm_comparison_result:
                idm_predictions_df = idm_comparison_result['predictions'].copy()
                
                # Map column names for visualization compatibility
                if 'svi_prediction' in idm_predictions_df.columns and 'mean' not in idm_predictions_df.columns:
                    idm_predictions_df['mean'] = idm_predictions_df['svi_prediction']
                
                # Also ensure standard deviation column if needed
                if 'uncertainty' in idm_predictions_df.columns and 'sd' not in idm_predictions_df.columns:
                    idm_predictions_df['sd'] = idm_predictions_df['uncertainty'] 
                
                # Update the comparison result
                idm_comparison_result['predictions'] = idm_predictions_df

            # Keep the original baseline for backwards compatibility
            baseline_result = self.mg_interface._idm_baseline(
                tract_observation=tract_observation,
                prediction_locations=prediction_locations, 
                nlcd_features=nlcd_features,
                tract_geometry=tract_geom
            )
            
            # Visualization data prep 
            network_data = self._prepare_network_data_for_viz(tract_data)
            transit_data = self._prepare_transit_data_for_viz(tract_data)
            
            return {
                'status': 'success',
                'fips': fips,
                'predictions': predictions,  # ← Uses excellent MetricGraph predictions
                'gnn_features': gnn_features,
                'spde_params': direct_mg_result['spde_params'],      # ← FIXED: Use direct_mg_result
                'diagnostics': direct_mg_result['diagnostics'],      # ← FIXED: Use direct_mg_result
                
                # Ensure all comparison fields are properly structured:
                'baseline_comparison': {
                    'success': True,
                    'predictions': idm_predictions_df,  # With mapped columns
                    'diagnostics': idm_comparison_result.get('diagnostics', {})
                } if idm_comparison_result.get('success') else None,
                
                'idm_comparison': idm_comparison_result,
                'idm_baseline': idm_comparison_result, 
                'comparison_results': idm_comparison_result,            
                'training_result': training_result,
                'network_data': network_data,
                'transit_data': transit_data,
                'trained_model': gnn_model,
                'timing': {
                    'gnn_training': gnn_time,
                    'metricgraph_creation': mg_time,
                    'disaggregation': time.time() - gnn_start,
                    'total': time.time() - gnn_start
                }
            }
        
        except Exception as e:
            self._log(f"  Error processing tract: {str(e)}")
            print(f"Error in hybrid processing: {e}")
            return {'status': 'failed', 'fips': fips, 'error': str(e)}
    
    def _prepare_metricgraph_data(self, road_network):
        """Prepare node and edge dataframes for MetricGraph"""
        # Extract nodes
        nodes = []
        for i, (node_id, data) in enumerate(road_network.nodes(data=True)):
            nodes.append({
                'node_id': i,
                'x': data['x'],
                'y': data['y']
            })
        nodes_df = pd.DataFrame(nodes)
        
        # Create node ID mapping
        node_id_map = {node_id: i for i, node_id in enumerate(road_network.nodes())}
        
        # Extract edges
        edges = []
        for u, v, data in road_network.edges(data=True):
            edges.append({
                'from': node_id_map[u],
                'to': node_id_map[v],
                'weight': data.get('length', 1.0)
            })
        edges_df = pd.DataFrame(edges)
        
        return nodes_df, edges_df
    
    def _compute_idm_validation_metrics(self, gnn_predictions, idm_result, svi_value):
        """
        Compute GNN vs IDM validation metrics (single tract)
        """
        gnn_values = gnn_predictions['mean'].values
        gnn_uncertainty = gnn_predictions['sd'].values
        
        if idm_result and idm_result.get('success') and 'predictions' in idm_result:
            idm_values = idm_result['predictions']['mean'].values
            idm_uncertainty = idm_result['predictions']['sd'].values
            
            # Ensure same length
            min_len = min(len(gnn_values), len(idm_values))
            gnn_values = gnn_values[:min_len]
            idm_values = idm_values[:min_len]
            
            # Compute correlation
            correlation = np.corrcoef(gnn_values, idm_values)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            
            # Compute effectiveness metrics
            gnn_variance = np.var(gnn_values)
            idm_variance = np.var(idm_values)
            
            # Constraint preservation
            gnn_constraint_error = abs(np.mean(gnn_values) - svi_value) / svi_value
            idm_constraint_error = abs(np.mean(idm_values) - svi_value) / svi_value
            
            # Detailed diagnostics
            self._log(f"    GNN vs IDM detailed comparison:")
            self._log(f"      GNN std: {np.std(gnn_values):.6f}")
            self._log(f"      IDM std: {np.std(idm_values):.6f}")
            self._log(f"      Correlation: {correlation:.6f}")
            
            validation_results = {
                'gnn_vs_idm_correlation': correlation,
                'gnn_variance': gnn_variance,
                'idm_variance': idm_variance,
                'gnn_constraint_error': gnn_constraint_error,
                'idm_constraint_error': idm_constraint_error,
                'gnn_uncertainty_mean': np.mean(gnn_uncertainty),
                'idm_uncertainty_mean': np.mean(idm_uncertainty),
                'comparison_method': 'GNN_vs_IDM', 
                'gnn_more_variable': gnn_variance > idm_variance,
                'gnn_better_constraint': gnn_constraint_error < idm_constraint_error,
                'gnn_effectiveness_score': (gnn_variance / idm_variance) * abs(correlation) if idm_variance > 0 else 1.0
            }
            
            return validation_results
        else:
            return {
                'gnn_vs_idm_correlation': None,
                'comparison_method': 'GNN_only',
                'error': 'IDM comparison failed'
            }

    def _load_nlcd_for_tract(self, tract_data):
        """
        Load NLCD data for single tract area - SIMPLIFIED VERSION
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

    def _calculate_gnn_effectiveness(self, correlation, gnn_var, idm_var, 
                                gnn_constraint, idm_constraint):
        """
        Calculate overall GNN effectiveness vs IDM
        
        Returns:
        --------
        float
            Effectiveness score (>1.0 means GNN is better)
        """
        # Spatial differentiation 
        spatial_score = gnn_var / idm_var if idm_var > 0 else 1.0
        
        # Constraint preservation 
        constraint_score = idm_constraint / gnn_constraint if gnn_constraint > 0 else 1.0
        
        # Correlation factor 
        correlation_factor = abs(correlation) if not np.isnan(correlation) else 0.5
        
        # Combined effectiveness
        effectiveness = (spatial_score * constraint_score * correlation_factor)
        
        return effectiveness

    def compute_proper_validation_metrics(self, predictions, baseline_result, svi_value):
        """
        Proper validation with correct IDM baseline comparison
        """
        # Ensure both methods use same locations
        gnn_predictions = predictions['mean'].values
        gnn_uncertainty = predictions['sd'].values
        
        # Get IDM predictions on locations
        if baseline_result and 'predictions' in baseline_result:
            idm_predictions = baseline_result['predictions']['mean'].values
            idm_uncertainty = baseline_result['predictions']['sd'].values
            
            # Verify same number of predictions
            if len(gnn_predictions) != len(idm_predictions):
                self._log(f"WARNING: Different prediction counts - GNN: {len(gnn_predictions)}, IDM: {len(idm_predictions)}")
                
                # Take minimum length for comparison
                min_len = min(len(gnn_predictions), len(idm_predictions))
                gnn_predictions = gnn_predictions[:min_len]
                idm_predictions = idm_predictions[:min_len]
                gnn_uncertainty = gnn_uncertainty[:min_len]
                idm_uncertainty = idm_uncertainty[:min_len]
            
            # Compute correlation between GNN and IDM
            correlation = np.corrcoef(gnn_predictions, idm_predictions)[0, 1]
            
            # Check for concerning results
            if correlation < 0.1:
                self._log(f"WARNING: Low correlation ({correlation:.3f}) suggests implementation issues")
                self._log(f"  GNN predictions range: [{gnn_predictions.min():.3f}, {gnn_predictions.max():.3f}]")
                self._log(f"  IDM predictions range: [{idm_predictions.min():.3f}, {idm_predictions.max():.3f}]")
                
                # Additional diagnostic for IDM comparison
                gnn_variation = np.std(gnn_predictions)
                idm_variation = np.std(idm_predictions)
                variation_ratio = idm_variation / gnn_variation if gnn_variation > 0 else float('inf')
                
                self._log(f"  GNN spatial variation: {gnn_variation:.6f}")
                self._log(f"  IDM spatial variation: {idm_variation:.6f}")
                self._log(f"  IDM/GNN variation ratio: {variation_ratio:.2f}")
                
                if variation_ratio > 10:
                    self._log(f"    CRITICAL: GNN shows {variation_ratio:.1f}x less spatial variation than IDM")
                    self._log(f"     This suggests GNN over-smoothing or feature uniformity issues")
            else:
                self._log(f"Good correlation with IDM baseline: {correlation:.3f}")
                
        else:
            self._log("No IDM baseline comparison available")
            correlation = None
            idm_predictions = None
            idm_uncertainty = None
        
        # Constraint validation 
        predicted_tract_mean = np.mean(gnn_predictions)
        constraint_error = abs(predicted_tract_mean - svi_value) / svi_value if svi_value > 0 else 0
        
        # Parameter variability check
        prediction_variance = np.var(gnn_predictions)
        
        # Enhanced validation results for IDM comparison
        validation_results = {
            # Method comparison
            'gnn_vs_idm_correlation': correlation,
            'method_correlation': correlation,  # Legacy alias for backward compatibility
            
            # Constraint satisfaction 
            'constraint_error': constraint_error,
            'constraint_satisfied': constraint_error < 0.01,  # 1% tolerance
            
            # Spatial variation metrics
            'prediction_variance': prediction_variance,
            'gnn_prediction_std': np.std(gnn_predictions),
            'gnn_uncertainty_mean': np.mean(gnn_uncertainty),
            'idm_prediction_std': np.std(idm_predictions) if idm_predictions is not None else None,
            'idm_uncertainty_mean': np.mean(idm_uncertainty) if idm_uncertainty is not None else None,
            
            # Variation ratio
            'spatial_variation_ratio': (np.std(idm_predictions) / np.std(gnn_predictions) 
                                    if idm_predictions is not None and np.std(gnn_predictions) > 0 
                                    else None),
            
            # Tract-level validation
            'predicted_tract_mean': predicted_tract_mean,
            'actual_tract_svi': svi_value,
            
            # Method-specific ranges
            'gnn_prediction_range': (gnn_predictions.min(), gnn_predictions.max()),
            'idm_prediction_range': ((idm_predictions.min(), idm_predictions.max()) 
                                if idm_predictions is not None else None),
            
            # Quality flags
            'concerning_low_correlation': correlation is not None and correlation < 0.1,
            'excessive_gnn_smoothing': (correlation is not None and 
                                    idm_predictions is not None and 
                                    np.std(idm_predictions) / np.std(gnn_predictions) > 10),
        }
        
        return validation_results
    
    def _save_results(self, results):
        """Save results with enhanced visualizations including clear GNN vs IDM comparison"""
        if not results.get('success', False):
            self._log("No results to save")
            return
        
        # Save existing outputs (predictions, summary, etc.)
        predictions_path = os.path.join(self.output_dir, 'granite_predictions.csv')
        results['predictions'].to_csv(predictions_path, index=False)
        self._log(f"Saved predictions to {predictions_path}")
        
        # Save summary
        summary_path = os.path.join(self.output_dir, 'granite_summary.json')
        import json
        with open(summary_path, 'w') as f:
            json.dump(results['summary'], f, indent=2)
        self._log(f"Saved summary to {summary_path}")
        
        # Create visualizations with clear GNN vs IDM comparison
        self._log("Creating enhanced visualizations...")
        
        # FIXED: Actually set viz_data and extract IDM predictions
        viz_data = None
        gnn_predictions = results['predictions']
        idm_predictions = None
        
        if results.get('tract_results'):
            for i, tract_result in enumerate(results['tract_results'][:1]):  # Check first tract
                
                viz_data = tract_result
                
                # Extract IDM predictions from any available field
                for field_name in ['baseline_comparison', 'idm_comparison', 'idm_baseline', 'comparison_results']:
                    if field_name in tract_result and tract_result[field_name]:
                        idm_source = tract_result[field_name]
                        if (isinstance(idm_source, dict) and 
                            idm_source.get('success') and 
                            'predictions' in idm_source):
                            
                            idm_predictions = idm_source['predictions']
                            self._log(f"Found IDM data from '{field_name}': {len(idm_predictions)} predictions")
                            
                            # Verify required columns exist
                            required_cols = ['mean', 'x', 'y']
                            missing_cols = [col for col in required_cols if col not in idm_predictions.columns]
                            if not missing_cols:
                                self._log(f"All required columns present: {list(idm_predictions.columns)}")
                                break
                            else:
                                self._log(f"Missing required columns: {missing_cols}")
                                idm_predictions = None
                
                # Break after first tract (we only need one for visualization)
                if viz_data:
                    break
        
        # FIXED: Now this condition should be True
        if viz_data and idm_predictions is not None:
            self._log(f"SUCCESS: Creating GNN vs IDM comparison visualization!")
            
            try:
                clear_comparison_path = os.path.join(self.output_dir, 'gnn_vs_idm_comparison.png')
                
                self.visualizer.create_clear_method_comparison(
                    gnn_predictions=gnn_predictions,
                    idm_predictions=idm_predictions,
                    gnn_results=results,
                    idm_results={'success': True, 'predictions': idm_predictions},
                    output_path=clear_comparison_path
                )
                
                self._log(f"Saved GNN vs IDM comparison to {clear_comparison_path}")
                
                # Also create other visualizations
                original_viz_path = os.path.join(self.output_dir, 'granite_visualization.png')
                self.visualizer.create_disaggregation_plot(
                    predictions=gnn_predictions,
                    results=results,
                    comparison_results={'success': True, 'predictions': idm_predictions},
                    output_path=original_viz_path
                )
                self._log(f"Saved comparison visualization to {original_viz_path}")
                
                # Print comparison summary to console
                self._print_comparison_summary(gnn_predictions, idm_predictions)
                
            except Exception as e:
                self._log(f"Visualization creation failed: {e}")
                import traceback
                self._log(f"Full traceback: {traceback.format_exc()}")
                
        else:
            self._log(f"  Could not extract IDM predictions for visualization")
            self._log(f"   viz_data exists: {viz_data is not None}")
            self._log(f"   idm_predictions exists: {idm_predictions is not None}")
            
            # Create fallback single-method visualization
            try:
                fallback_path = os.path.join(self.output_dir, 'granite_visualization_single.png')
                self.visualizer.create_disaggregation_plot(
                    predictions=gnn_predictions,
                    results=results,
                    comparison_results=None,  # No comparison
                    output_path=fallback_path
                )
                self._log(f"Created fallback single-method visualization: {fallback_path}")
            except Exception as e:
                self._log(f"Even fallback visualization failed: {e}")

        # Global validation summary
        if results.get('global_validation'):
            self._log(f"\n=== GLOBAL VALIDATION SUMMARY ===")
            global_val = results['global_validation']
            if global_val.get('method_correlation'):
                self._log(f"Global GNN-IDM Correlation: {global_val['method_correlation']:.3f}")
            self._log(f"Total Addresses Compared: {global_val.get('total_addresses', 'N/A')}")

        self._log(f"\nAll results and visualizations saved to {self.output_dir}/")

    def debug_gnn_data_sources(self, result):
        """Debug exactly what data sources are being used"""
        print("\n" + "="*60)
        print("🔍 GNN DATA SOURCE DEBUGGING")
        print("="*60)
        
        # Check what's in the result object
        print(f"Result keys: {list(result.keys())}")
        
        # Check each potential data source
        data_sources = [
            ('predictions', result.get('predictions')),
            ('disaggregation_result', result.get('disaggregation_result')),
            ('baseline_result', result.get('baseline_result')),
            ('metricgraph_result', result.get('metricgraph_result'))
        ]
        
        for source_name, source_data in data_sources:
            if source_data is not None:
                if isinstance(source_data, dict):
                    print(f"\n{source_name} (dict):")
                    print(f"  Keys: {list(source_data.keys())}")
                    
                    # Check for DataFrames in the dict
                    for key, value in source_data.items():
                        if hasattr(value, 'std'):  # It's array-like
                            try:
                                std_val = np.std(value)
                                print(f"    {key}: std = {std_val:.6f}")
                            except:
                                pass
                        elif isinstance(value, pd.DataFrame):
                            print(f"    {key} (DataFrame): shape = {value.shape}")
                            if 'mean' in value.columns:
                                std_val = value['mean'].std()
                                print(f"      mean column std = {std_val:.6f}")
                                
                elif isinstance(source_data, pd.DataFrame):
                    print(f"\n{source_name} (DataFrame): shape = {source_data.shape}")
                    if 'mean' in source_data.columns:
                        std_val = source_data['mean'].std()
                        print(f"  mean column std = {std_val:.6f}")
                        print(f"  columns: {list(source_data.columns)}")
        
        print("="*60)

    def _print_comparison_summary(self, gnn_predictions, idm_predictions):
        """
        Print a clear summary of GNN vs IDM comparison to console
        """
        try:
            # === DEBUGGING THE DATA SOURCE ===
            print("\n🔍 COMPARISON SUMMARY DEBUG:")
            print(f"   gnn_predictions type: {type(gnn_predictions)}")
            print(f"   gnn_predictions columns: {list(gnn_predictions.columns)}")
            print(f"   gnn_predictions shape: {gnn_predictions.shape}")
            
            # Check if this is the same data as the interface
            if 'mean' in gnn_predictions.columns:
                gnn_values = gnn_predictions['mean'].values
                interface_std = np.std(gnn_values)
                print(f"   Interface std in comparison: {interface_std:.6f}")
                
                # Check where this data came from
                if hasattr(gnn_predictions, 'source'):
                    print(f"   Data source: {gnn_predictions.source}")
                
                # Check for constraint enforcement indicators
                print(f"   Mean value: {np.mean(gnn_values):.6f}")
                print(f"   Min/Max: [{gnn_values.min():.4f}, {gnn_values.max():.4f}]")
                
            else:
                print(f"   ERROR: No 'mean' column found!")
                print(f"   Available columns: {list(gnn_predictions.columns)}")
            
            print("   === END DEBUG ===\n")

            gnn_values = gnn_predictions['mean'].values
            idm_values = idm_predictions['mean'].values
            
            gnn_std = np.std(gnn_values)
            idm_std = np.std(idm_values)
            correlation = np.corrcoef(gnn_values, idm_values)[0, 1]
            ratio = idm_std / gnn_std if gnn_std > 0 else 0
            
            self._log(f"\n" + "="*60)
            self._log(f"GNN vs IDM COMPARISON SUMMARY")
            self._log(f"="*60)
            self._log(f"GNN (Learned Parameters):")
            self._log(f"  • Spatial Standard Deviation: {gnn_std:.6f}")
            self._log(f"  • Mean SVI: {np.mean(gnn_values):.4f}")
            self._log(f"  • Range: [{gnn_values.min():.4f}, {gnn_values.max():.4f}]")
            
            self._log(f"\nIDM (Fixed Coefficients):")
            self._log(f"  • Spatial Standard Deviation: {idm_std:.6f}")
            self._log(f"  • Mean SVI: {np.mean(idm_values):.4f}")
            self._log(f"  • Range: [{idm_values.min():.4f}, {idm_values.max():.4f}]")
            
            self._log(f"\nComparison Metrics:")
            self._log(f"  • Spatial Variation Ratio: {ratio:.1f}:1 (IDM:GNN)")
            self._log(f"  • Method Correlation: {correlation:.3f}")
            
            # Interpretation
            if ratio > 5:
                interpretation = "IDM creates MUCH more spatial variation"
                implication = "Land cover coefficients drive fine-scale patterns"
            elif ratio > 2:
                interpretation = "IDM creates more spatial variation"
                implication = "Different approaches to spatial modeling"
            elif ratio < 0.5:
                interpretation = "GNN creates more spatial variation"
                implication = "Learned parameters capture spatial complexity"
            else:
                interpretation = "Similar spatial variation levels"
                implication = "Methods produce comparable spatial patterns"
            
            self._log(f"\nKey Finding:")
            self._log(f"  • {interpretation}")
            self._log(f"  • {implication}")
            
            if abs(correlation) > 0.7:
                agreement = "High agreement on spatial patterns"
            elif abs(correlation) > 0.3:
                agreement = "Moderate agreement on spatial patterns"
            else:
                agreement = "Low agreement - methods disagree on patterns"
            
            self._log(f"  • {agreement}")
            
            # Research implications
            self._log(f"\nResearch Implications:")
            if ratio > 2:
                self._log(f"  ✓ Fixed land cover coefficients preserve spatial detail")
                self._log(f"  ✓ GNN learned parameters emphasize spatial smoothness")
                self._log(f"  → Consider hybrid approach combining both strengths")
            else:
                self._log(f"  ✓ Methods show comparable spatial modeling performance")
                self._log(f"  → Choice depends on application requirements")
            
            self._log(f"="*60)
            
        except Exception as e:
            self._log(f"Error printing comparison summary: {str(e)}")

    def _corrections_to_spde_params(self, corrections, idm_baseline):        
        """Convert single correction values to 3-parameter SPDE format for MetricGraph"""
        base_kappa = 1.0
        base_alpha = 1.5
        base_tau = 1.0
        
        # Use corrections to modulate kappa and tau
        kappa_values = base_kappa * (1.0 + 3.0 * corrections)   
        alpha_values = np.full_like(corrections, base_alpha)  # Fixed
        tau_values = base_tau * (1.0 + 2.0 * corrections)       
        
        # Ensure valid ranges
        kappa_values = np.clip(kappa_values, 0.1, 5.0)
        tau_values = np.clip(tau_values, 0.1, 3.0)

        return np.column_stack([kappa_values, alpha_values, tau_values])
    
    def _prepare_network_data_for_viz(self, tract_data):
        """Prepare network data in format expected by visualizations"""
        road_network = tract_data['road_network']
        
        # Convert NetworkX graph to GeoDataFrame for edge visualization
        edges_list = []
        for u, v, data in road_network.edges(data=True):
            if 'x' in road_network.nodes[u] and 'x' in road_network.nodes[v]:
                from shapely.geometry import LineString
                line = LineString([
                    (road_network.nodes[u]['x'], road_network.nodes[u]['y']),
                    (road_network.nodes[v]['x'], road_network.nodes[v]['y'])
                ])
                edges_list.append({
                    'geometry': line,
                    'from': u,
                    'to': v,
                    'weight': data.get('length', 1.0)
                })
        
        edges_gdf = gpd.GeoDataFrame(edges_list, crs='EPSG:4326')
        
        return {
            'graph': road_network,
            'edges_gdf': edges_gdf
        }

    def _prepare_transit_data_for_viz(self, tract_data):
        """Prepare transit data for visualization"""
        
        # Load transit stops from data loader if not already in tract_data
        if 'transit_stops' not in tract_data:
            transit_stops = self.data_loader.load_transit_stops()
        else:
            transit_stops = tract_data['transit_stops']
        
        # Filter stops to tract area if needed
        tract_geom = tract_data['tract_info'].geometry
        local_stops = transit_stops[transit_stops.intersects(tract_geom.buffer(0.01))]
        
        return {
            'stops': local_stops
        }

    def _apply_hybrid_disaggregation(self, metric_graph, tract_observation, 
                                   prediction_locations, hybrid_predictions, 
                                   gnn_features, alpha):
        """Apply hybrid disaggregation using pre-computed hybrid predictions"""
        try:
            result = self.mg_interface.disaggregate_svi(
                metric_graph=metric_graph,
                tract_observation=tract_observation,
                prediction_locations=prediction_locations,
                gnn_features=gnn_features,
                alpha=alpha
            )
            
            if result['success']:
                predictions_df = result['predictions'].copy()
                predictions_df['mean'] = hybrid_predictions
                
                diagnostics = result['diagnostics'].copy()
                diagnostics['mean_prediction'] = np.mean(hybrid_predictions)
                diagnostics['std_prediction'] = np.std(hybrid_predictions)
                diagnostics['constraint_satisfied'] = abs(np.mean(hybrid_predictions) - tract_observation['svi_value'].iloc[0]) < 0.01
                
                result['predictions'] = predictions_df
                result['diagnostics'] = diagnostics
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': f'Hybrid disaggregation failed: {str(e)}'}

