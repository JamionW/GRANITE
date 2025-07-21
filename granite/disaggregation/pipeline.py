"""
Main disaggregation pipeline for GRANITE framework
Updated to use spatial disaggregation instead of regression
"""
import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime
import torch
import warnings
warnings.filterwarnings('ignore')

from ..data.loaders import DataLoader
from ..models.gnn import prepare_graph_data, create_gnn_model
from ..models.training import train_accessibility_gnn
from ..metricgraph.interface import MetricGraphInterface
from ..visualization.plots import DisaggregationVisualizer



class GRANITEPipeline:
    """
    Main pipeline for SVI disaggregation using GNN-MetricGraph integration
    
    GRANITE: Graph-Refined Accessibility Network for Integrated Transit Equity
    
    This implementation uses spatial disaggregation with GNN-learned parameters
    rather than regression-based approaches.
    """
    
    def __init__(self, config, data_dir='./data', output_dir='./output', verbose=True):
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
        self.verbose = verbose
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader(data_dir, verbose)
        self.mg_interface = MetricGraphInterface(
            verbose=verbose,
            config=self.config.get('metricgraph', {})
        )
        self.visualizer = DisaggregationVisualizer()
        
        # Storage for results
        self.results = {}
    
    def _log(self, message, level='INFO'):
        """Logging with timestamp and level"""
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    def run(self):
        """Run the complete GRANITE pipeline"""
        self._log("="*60)
        self._log("GRANITE Pipeline - Spatial Disaggregation Mode") 
        self._log("="*60)
        
        start_time = time.time()
        
        # Load data
        self._log("Loading data...")
        data = self._load_data()
        
        # Check processing mode
        processing_mode = self.config.get('data', {}).get('processing_mode', 'county')
        target_fips = self.config.get('data', {}).get('target_fips')
        
        # CRITICAL DEBUG OUTPUT
        self._log(f"Processing mode: {processing_mode}")
        self._log(f"Target FIPS: {target_fips}")
        
        if processing_mode == 'fips' and target_fips:
            self._log(f"Processing single FIPS: {target_fips}")
            results = self._process_fips_mode(data)
        else:
            self._log("No FIPS codes specified, processing all tracts")
            results = self._process_county_mode(data)
        
        # Save results
        self._save_results(results)
        
        elapsed_time = time.time() - start_time
        self._log(f"Pipeline completed in {elapsed_time:.2f} seconds")
        
        return results
    
    def _load_data(self):
        """
        UPDATED: Load data using real Chattanooga addresses
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
        data['transit_stops'] = self.data_loader.load_transit_stops()
        self._log(f"Loaded {len(data['transit_stops'])} transit stops")
        
        # UPDATED: Load real address points
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
    
    def _process_county_mode(self, data):
        """Process entire county with tract-by-tract disaggregation"""
        self._log("Processing in county mode")
        
        total_tracts = len(data['tracts'])
        all_results = []
        successful = 0
        failed = 0
        
        # Process each tract
        for idx, tract in data['tracts'].iterrows():
            fips = tract['FIPS']
            
            tract_number = idx
            self._log(f"\nProcessing tract {fips} ({tract_number}/{total_tracts})")

            
            # Skip if no SVI data
            if pd.isna(tract['RPL_THEMES']):
                self._log(f"  Skipping {fips} - no SVI data")
                continue
            
            try:
                # Get tract-specific data
                tract_data = self._prepare_tract_data(tract, data)
                
                if tract_data['addresses'].empty:
                    self._log(f"  Skipping {fips} - no addresses")
                    continue
                
                # Process tract
                result = self._process_single_tract(tract_data)
                
                if result['status'] == 'success':
                    all_results.append(result)
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                self._log(f"  Error processing {fips}: {str(e)}")
                failed += 1
        
        self._log(f"\nCounty processing complete: {successful} successful, {failed} failed")
        
        # Combine all results
        return self._combine_results(all_results)
    
    def _process_fips_mode(self, data):
        """
        FIXED: Process specific FIPS codes with proper string handling
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
        
        # Process single tract
        return self._process_county_mode(single_tract_data)
    
    def _prepare_tract_data(self, tract, county_data):
        """
        UPDATED: Prepare data for a single tract using real addresses
        """
        # Get tract geometry
        tract_geom = tract.geometry
        fips_code = tract['FIPS']
        
        # Get roads within tract
        tract_roads = county_data['roads'][
            county_data['roads'].intersects(tract_geom)
        ].copy()
        
        # UPDATED: Get real addresses for this specific tract
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
            'addresses': tract_addresses,  # NOW REAL ADDRESSES
            'road_network': road_network,
            'svi_value': tract['RPL_THEMES'],
            'address_count': len(tract_addresses),
            'address_source': 'real' if 'full_address' in tract_addresses.columns else 'synthetic'
        }
    
    def _process_single_tract(self, tract_data):
        """
        Process a single tract using spatial disaggregation
        
        This is the core innovation: using GNN-learned parameters
        for constrained spatial disaggregation
        """
        fips = tract_data['tract_info']['FIPS']
        svi_value = tract_data['svi_value']
        
        self._log(f"  Processing tract {fips} with SVI={svi_value:.3f}")
        
        try:
            # Step 1: Train GNN for accessibility features
            self._log("  Training GNN for accessibility feature learning...")
            gnn_start = time.time()
            
            # Prepare graph data
            graph_data, node_mapping = prepare_graph_data(tract_data['road_network'])
            
            # Create and train GNN model
            model = create_gnn_model(
                input_dim=graph_data.x.shape[1],
                hidden_dim=self.config['model']['hidden_dim'],
                output_dim=self.config['model']['output_dim']
            )
            
            feature_history = [] 

            trained_model, gnn_features, training_metrics = train_accessibility_gnn(
                graph_data,
                model,
                epochs=self.config['model']['epochs'],
                lr=self.config['model']['learning_rate'],
                spatial_weight=self.config['model']['spatial_weight'],
                reg_weight=self.config['model']['regularization_weight'],
                collect_history=True, 
                feature_history=feature_history 
            )
            
            attention_weights = None
            if hasattr(trained_model, 'attention_weights'):
                with torch.no_grad():
                    trained_model.eval()
                    _, attention_weights = trained_model(graph_data.x, graph_data.edge_index, 
                                                    return_attention=True)
                    attention_weights = attention_weights.cpu().numpy()
        
            gnn_time = time.time() - gnn_start
            self._log(f"    GNN training completed in {gnn_time:.2f}s")
            self._log(f"    Learned parameters - mean kappa: {gnn_features[:, 0].mean():.3f}, "
                     f"mean tau: {gnn_features[:, 2].mean():.3f}")
            
            # Step 2: Create MetricGraph
            self._log("  Creating MetricGraph representation...")
            mg_start = time.time()
            
            nodes_df, edges_df = self._prepare_metricgraph_data(tract_data['road_network'])
            
            metric_graph = self.mg_interface.create_graph(
                nodes_df,
                edges_df,
                enable_sampling=len(edges_df) > self.config['metricgraph']['max_edges']
            )
            
            if metric_graph is None:
                raise RuntimeError("Failed to create MetricGraph")
            
            mg_time = time.time() - mg_start
            self._log(f"    MetricGraph created in {mg_time:.2f}s")
            
            # Step 3: Perform spatial disaggregation with GNN-Whittle-Matérn
            self._log("  Performing GNN-informed spatial disaggregation...")
            disagg_start = time.time()
            
            # Prepare tract observation (centroid and SVI value)
            tract_centroid = tract_data['tract_info'].geometry.centroid
            tract_observation = pd.DataFrame({
                'coord_x': [tract_centroid.x],
                'coord_y': [tract_centroid.y],
                'svi_value': [svi_value]
            })
            
            # Prepare prediction locations (addresses)
            prediction_locations = pd.DataFrame({
                'x': [addr.geometry.x for _, addr in tract_data['addresses'].iterrows()],
                'y': [addr.geometry.y for _, addr in tract_data['addresses'].iterrows()]
            })
            
            # Perform GNN-Whittle-Matérn disaggregation
            disagg_result = self.mg_interface.disaggregate_svi(
                metric_graph=metric_graph,
                tract_observation=tract_observation,
                prediction_locations=prediction_locations,
                gnn_features=gnn_features,
                alpha=self.config['metricgraph']['alpha']
            )
            
            # Also run kriging baseline for comparison
            self._log("  Running kriging baseline for comparison...")
            baseline_result = self.mg_interface._kriging_baseline(
                tract_observation=tract_observation,
                prediction_locations=prediction_locations
            )
            
            disagg_time = time.time() - disagg_start
            self._log(f"    Disaggregation completed in {disagg_time:.2f}s")
            
            network_data = self._prepare_network_data_for_viz(tract_data)
        
            transit_data = self._prepare_transit_data_for_viz(tract_data)
        

            if disagg_result['success']:
                predictions = disagg_result['predictions']
                diagnostics = disagg_result['diagnostics']
                
                self._log(f"    Successfully disaggregated to {len(predictions)} addresses")
                self._log(f"    Constraint satisfied: {diagnostics['constraint_satisfied']}")
                self._log(f"    Mean prediction: {diagnostics['mean_prediction']:.3f} "
                         f"(+/- {diagnostics['std_prediction']:.3f})")
                self._log(f"    Mean uncertainty: {diagnostics['mean_uncertainty']:.3f}")
                
                # Add metadata to predictions
                predictions['fips'] = fips
                predictions['tract_svi'] = svi_value
                
                return {
                    'status': 'success',
                    'fips': fips,
                    'predictions': predictions,
                    'gnn_features': gnn_features,
                    'spde_params': disagg_result['spde_params'],
                    'diagnostics': diagnostics,
                    'baseline_comparison': baseline_result,
                    'feature_history': feature_history,
                    'attention_weights': attention_weights,
                    'network_data': network_data,
                    'transit_data': transit_data,
                    'trained_model': trained_model,
                    'timing': {
                        'gnn_training': gnn_time,
                        'metricgraph_creation': mg_time,
                        'disaggregation': disagg_time,
                        'total': gnn_time + mg_time + disagg_time
                    }
                }
            else:
                raise RuntimeError(f"Disaggregation failed: {disagg_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self._log(f"  Error processing tract: {str(e)}")
            return {
                'status': 'failed',
                'fips': fips,
                'error': str(e)
            }
    
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
        # You might need to load transit routes if available
        # For now, just use transit stops
        
        # Load transit stops from data loader if not already in tract_data
        if 'transit_stops' not in tract_data:
            transit_stops = self.data_loader.load_transit_stops()
        else:
            transit_stops = tract_data['transit_stops']
        
        # Filter stops to tract area if needed
        tract_geom = tract_data['tract_info'].geometry
        local_stops = transit_stops[transit_stops.intersects(tract_geom.buffer(0.01))]
        
        return {
            'stops': local_stops,
            'routes': None  # Add if you have route data
        }

    def _combine_results(self, tract_results):
        """Combine results from all tracts"""
        if not tract_results:
            return {
                'success': False,
                'message': 'No tracts successfully processed'
            }
        
        # Combine all predictions
        all_predictions = []
        for result in tract_results:
            if result['status'] == 'success':
                all_predictions.append(result['predictions'])
        
        if all_predictions:
            combined_predictions = pd.concat(all_predictions, ignore_index=True)
            
            # Calculate summary statistics
            summary_stats = {
                'total_tracts': len(tract_results),
                'successful_tracts': sum(1 for r in tract_results if r['status'] == 'success'),
                'total_addresses': len(combined_predictions),
                'mean_svi': combined_predictions['mean'].mean(),
                'std_svi': combined_predictions['mean'].std(),
                'mean_uncertainty': combined_predictions['sd'].mean(),
                'processing_time': sum(r['timing']['total'] for r in tract_results if 'timing' in r)
            }
            
            return {
                'success': True,
                'predictions': combined_predictions,
                'tract_results': tract_results,
                'summary': summary_stats
            }
        else:
            return {
                'success': False,
                'message': 'No predictions generated'
            }
    
    def _save_results(self, results):
        """Save results with enhanced visualizations"""
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
        
        # ENHANCED: Create comprehensive visualizations
        self._log("Creating enhanced visualizations...")
        
        # Get visualization data from first successful tract
        viz_data = None
        if results.get('tract_results'):
            for tract_result in results['tract_results']:
                if (tract_result.get('status') == 'success' and 
                    tract_result.get('network_data')):
                    viz_data = tract_result
                    break
        
        if viz_data:
            # 1. Main visualization (existing)
            viz_path = os.path.join(self.output_dir, 'granite_visualization.png')
            self.visualizer.create_disaggregation_plot(
                predictions=results['predictions'],
                results=results,
                comparison_results=viz_data.get('baseline_comparison'),
                output_path=viz_path
            )
            self._log(f"Saved main visualization to {viz_path}")
            
            # 2. NEW: GNN feature evolution plot
            if viz_data.get('feature_history') and len(viz_data['feature_history']) > 1:
                evolution_path = os.path.join(self.output_dir, 'gnn_feature_evolution.png')
                self.visualizer.plot_gnn_feature_evolution(
                    viz_data['feature_history'],
                    output_path=evolution_path
                )
                self._log(f"Saved feature evolution plot to {evolution_path}")
            
            # 3. NEW: Accessibility gradients
            gradients_path = os.path.join(self.output_dir, 'accessibility_gradients.png')
            self.visualizer.plot_accessibility_gradients(
                predictions=results['predictions'],
                network_data=viz_data['network_data'],
                transit_data=viz_data.get('transit_data'),
                output_path=gradients_path
            )
            self._log(f"Saved accessibility gradients to {gradients_path}")
            
            # 4. NEW: GNN attention maps (if available)
            if viz_data.get('attention_weights') is not None:
                attention_path = os.path.join(self.output_dir, 'gnn_attention_maps.png')
                self.visualizer.plot_gnn_attention_maps(
                    attention_weights=viz_data['attention_weights'],
                    network_data=viz_data['network_data'],
                    predictions=results['predictions'],
                    output_path=attention_path
                )
                self._log(f"Saved attention maps to {attention_path}")
            
            # 5. NEW: Uncertainty sources analysis
            uncertainty_path = os.path.join(self.output_dir, 'uncertainty_analysis.png')
            self.visualizer.plot_uncertainty_sources(
                predictions=results['predictions'],
                network_data=viz_data.get('network_data'),
                output_path=uncertainty_path
            )
            self._log(f"Saved uncertainty analysis to {uncertainty_path}")
            
            # 6. NEW: Model interpretability dashboard
            if viz_data.get('gnn_features') is not None:
                interpret_path = os.path.join(self.output_dir, 'model_interpretability.png')
                self.visualizer.plot_model_interpretability(
                    predictions=results['predictions'],
                    gnn_features=viz_data['gnn_features'],
                    network_data=viz_data.get('network_data'),
                    output_path=interpret_path
                )
                self._log(f"Saved interpretability dashboard to {interpret_path}")
        
        self._log(f"\nAll results and visualizations saved to {self.output_dir}/")