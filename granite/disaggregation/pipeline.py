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
        """
        Run the complete GRANITE pipeline with spatial disaggregation
        """
        self._log("="*60)
        self._log("GRANITE Pipeline - Spatial Disaggregation Mode")
        self._log("="*60)
        
        start_time = time.time()
        
        # Step 1: Load and prepare data
        self._log("Loading data...")
        data = self._load_data()
        
        # Step 2: Process by tract
        if self.config['data']['processing_mode'] == 'fips':
            results = self._process_fips_mode(data)
        else:
            results = self._process_county_mode(data)
        
        # Step 3: Save results
        self._save_results(results)
        
        elapsed_time = time.time() - start_time
        self._log(f"Pipeline completed in {elapsed_time:.2f} seconds")
        
        return results
    
    def _load_data(self):
        """Load required datasets"""
        state_fips = self.config['data']['state_fips']
        county_fips = self.config['data']['county_fips']
        
        # Load census tracts
        tracts = self.data_loader.load_census_tracts(state_fips, county_fips)
        self._log(f"Loaded {len(tracts)} census tracts")
        
        # Load SVI data
        svi = self.data_loader.load_svi_data()
        svi_filtered = svi[svi['FIPS'].str.startswith(f"{state_fips}{county_fips}")]
        self._log(f"Loaded SVI data for {len(svi_filtered)} tracts")
        
        # Merge tracts with SVI
        tracts_with_svi = tracts.merge(
            svi_filtered[['FIPS', 'RPL_THEMES']], 
            on='FIPS', 
            how='inner'
        )
        
        # Load road network
        roads = self.data_loader.load_road_network(state_fips, county_fips)
        self._log(f"Loaded road network with {len(roads)} segments")
        
        # Load addresses
        addresses = self.data_loader.load_address_points(state_fips, county_fips)
        if addresses is None:
            self._log("No address points found, generating synthetic addresses...")
            addresses = self.data_loader.generate_synthetic_addresses(
                roads, tracts, density_per_km=50
            )
        self._log(f"Using {len(addresses)} address points")
        
        return {
            'tracts': tracts_with_svi,
            'roads': roads,
            'addresses': addresses,
            'svi_data': svi_filtered
        }
    
    def _process_county_mode(self, data):
        """Process entire county with tract-by-tract disaggregation"""
        self._log("Processing in county mode")
        
        all_results = []
        successful = 0
        failed = 0
        
        # Process each tract
        for idx, tract in data['tracts'].iterrows():
            fips = tract['FIPS']
            self._log(f"\nProcessing tract {fips} ({idx+1}/{len(data['tracts'])})")
            
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
        """Process specific FIPS codes"""
        fips_list = self.config['data'].get('fips_list', [])
        if not fips_list:
            self._log("No FIPS codes specified, processing all tracts")
            return self._process_county_mode(data)
        
        self._log(f"Processing {len(fips_list)} specified FIPS codes")
        
        # Filter to requested tracts
        data['tracts'] = data['tracts'][data['tracts']['FIPS'].isin(fips_list)]
        
        return self._process_county_mode(data)
    
    def _prepare_tract_data(self, tract, county_data):
        """Prepare data for a single tract"""
        # Get tract geometry
        tract_geom = tract.geometry
        
        # Get roads within tract
        tract_roads = county_data['roads'][
            county_data['roads'].intersects(tract_geom)
        ].copy()
        
        # Get addresses within tract  
        tract_addresses = county_data['addresses'][
            county_data['addresses'].within(tract_geom)
        ].copy()
        
        # Build road network graph
        road_network = self.data_loader.build_road_network_graph(tract_roads)
        
        return {
            'tract_info': tract,
            'roads': tract_roads,
            'addresses': tract_addresses,
            'road_network': road_network,
            'svi_value': tract['RPL_THEMES']
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
            
            trained_model, gnn_features, training_metrics = train_accessibility_gnn(
                graph_data,
                model,
                epochs=self.config['model']['epochs'],
                lr=self.config['model']['learning_rate'],
                spatial_weight=self.config['model']['spatial_weight'],
                reg_weight=self.config['model']['regularization_weight']
            )
            
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
        """Save results to output directory"""
        if not results.get('success', False):
            self._log("No results to save")
            return
        
        # Save predictions
        predictions_path = os.path.join(self.output_dir, 'granite_predictions.csv')
        results['predictions'].to_csv(predictions_path, index=False)
        self._log(f"Saved predictions to {predictions_path}")
        
        # Save summary
        summary_path = os.path.join(self.output_dir, 'granite_summary.json')
        import json
        with open(summary_path, 'w') as f:
            json.dump(results['summary'], f, indent=2)
        self._log(f"Saved summary to {summary_path}")
        
        # Create visualization
        viz_path = os.path.join(self.output_dir, 'granite_visualization.png')
        
        # Extract baseline comparison if available
        comparison_results = None
        if results.get('tract_results'):
            # Get first tract with baseline comparison
            for tract_result in results['tract_results']:
                if tract_result.get('baseline_comparison'):
                    comparison_results = tract_result['baseline_comparison']
                    break
        
        self.visualizer.create_disaggregation_plot(
            predictions=results['predictions'],
            results=results,
            comparison_results=comparison_results,
            output_path=viz_path
        )
        self._log(f"Saved visualization to {viz_path}")
        
        self._log(f"\nResults saved to {self.output_dir}/")