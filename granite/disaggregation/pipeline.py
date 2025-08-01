"""
Main disaggregation pipeline for GRANITE framework
Updated to use spatial disaggregation instead of regression
"""
import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import warnings
warnings.filterwarnings('ignore')

from ..data.loaders import DataLoader
from ..models.gnn import create_gnn_model, prepare_graph_data_with_nlcd, prepare_graph_data_topological
from ..models.training import train_accessibility_gnn
from ..metricgraph.interface import MetricGraphInterface
from ..visualization.plots import DisaggregationVisualizer
from ..diagnostics.comparison_diagnostics import diagnose_comparison_issues, create_diagnostic_plots



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
        self.data_loader = DataLoader(data_dir, verbose, config=config)
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
        
        start_time = time.time()
        
        # Load data
        self._log("Loading data...")
        data = self._load_data()
        
        # Check processing mode
        processing_mode = self.config.get('data', {}).get('processing_mode', 'county')
        target_fips = self.config.get('data', {}).get('target_fips')
        target_fips_list = self.config.get('data', {}).get('target_fips_list', [])
        
        self._log(f"Processing mode: {processing_mode}")
        
        if processing_mode == 'fips' and target_fips:
            self._log(f"Processing single FIPS: {target_fips}")
            results = self._process_fips_mode(data)
        elif processing_mode == 'multi_fips' and target_fips_list:
            self._log(f"Processing multiple FIPS: {target_fips_list}")
            results = self._process_multi_fips_mode(data, target_fips_list)
        else:
            self._log("Processing all tracts")
            results = self._process_county_mode(data)
        
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
    
    def _process_county_mode(self, data):
        self._log("Processing in county mode with SHARED GNN training")
        
        # Step 1: Global GNN training on all tracts
        self._log("\n=== PHASE 1: Global GNN Training on All Tracts ===")
        global_gnn_model = self._train_global_gnn(data)
        
        # Step 2: Apply trained model to each tract
        self._log("\n=== PHASE 2: Applying Trained GNN to Individual Tracts ===")
        all_results = []
        
        for idx, tract in data['tracts'].iterrows():
            fips = tract['FIPS']
            self._log(f"\nApplying trained GNN to tract {fips}")
            
            try:
                tract_data = self._prepare_tract_data(tract, data)
                if tract_data['addresses'].empty:
                    continue
                    
                result = self._apply_trained_gnn_to_tract(tract_data, global_gnn_model)
                
                if result['status'] == 'success':
                    all_results.append(result)
                    
            except Exception as e:
                self._log(f"Error processing {fips}: {str(e)}")
        
        return self._combine_results(all_results)
    
    def _train_global_gnn(self, data):
        """Train one GNN on all tract networks combined with NLCD features"""
        self._log("Building combined network from all tracts...")
        
        # Combine networks and NLCD features from all tracts
        combined_networks = []
        combined_svi_values = []
        combined_nlcd_features = []
        combined_addresses = []
        
        for idx, tract in data['tracts'].iterrows():
            if pd.isna(tract['RPL_THEMES']):
                continue
                
            tract_data = self._prepare_tract_data(tract, data)
            if not tract_data['addresses'].empty:
                combined_networks.append(tract_data['road_network'])
                combined_svi_values.append(tract['RPL_THEMES'])
                
                # Collect NLCD features for this tract
                try:
                    tract_nlcd_features = self._load_nlcd_for_tract(tract_data)
                    if tract_nlcd_features is not None and len(tract_nlcd_features) > 0:
                        # Add tract identifier to avoid address_id conflicts
                        tract_nlcd_features = tract_nlcd_features.copy()
                        tract_nlcd_features['tract_id'] = tract['FIPS']
                        tract_nlcd_features['address_id'] = (
                            tract_nlcd_features['address_id'].astype(str) + 
                            '_' + tract['FIPS']
                        )
                        combined_nlcd_features.append(tract_nlcd_features)
                        
                        # Also combine addresses with tract identifier
                        tract_addresses = tract_data['addresses'].copy()
                        tract_addresses['address_id'] = (
                            tract_addresses.get('address_id', range(len(tract_addresses))).astype(str) + 
                            '_' + tract['FIPS']
                        )
                        combined_addresses.append(tract_addresses)
                        
                except Exception as e:
                    self._log(f"Warning: Could not load NLCD for tract {tract['FIPS']}: {e}")
        
        # Create single graph from all tracts
        mega_graph = self._combine_networks(combined_networks)
        
        # Combine all NLCD features if available
        if combined_nlcd_features:
            mega_nlcd_features = pd.concat(combined_nlcd_features, ignore_index=True)
            mega_addresses = pd.concat(combined_addresses, ignore_index=True)
            
            self._log(f"Combined NLCD features from {len(combined_nlcd_features)} tracts:")
            self._log(f"  Total features: {len(mega_nlcd_features)}")
            self._log(f"  Total addresses: {len(mega_addresses)}")
            
            # Prepare graph data with NLCD features
            graph_data, node_mapping = prepare_graph_data_with_nlcd(
                mega_graph, 
                mega_nlcd_features,
                addresses=mega_addresses
            )
            feature_type = "NLCD-based"
            
        else:
            self._log("No NLCD features available, falling back to topological features")
            # Fall back to topological features
            graph_data, node_mapping = prepare_graph_data_topological(mega_graph)
            feature_type = "topological"
        
        # Train GNN on full spatial diversity
        model = create_gnn_model(
            input_dim=graph_data.x.shape[1],
            hidden_dim=self.config['model']['hidden_dim'],
            output_dim=self.config['model']['output_dim']
        )
        
        self._log(f"Training GNN on combined network:")
        self._log(f"  Nodes: {mega_graph.number_of_nodes()}")
        self._log(f"  Features: {feature_type}")
        self._log(f"  Input dimensions: {graph_data.x.shape[1]}")
        
        trained_model, _, training_metrics = train_accessibility_gnn(
            graph_data,
            model,
            epochs=self.config['model']['epochs'],
            lr=self.config['model']['learning_rate'],
            spatial_weight=self.config['model']['spatial_weight'],
            reg_weight=self.config['model']['regularization_weight']
        )
        
        self._log(f"Global GNN training complete: {training_metrics['final_loss']:.4f}")
        return trained_model
    
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

    def _combine_networks(self, networks):
        """Combine multiple NetworkX graphs into one mega-graph"""
        import networkx as nx
        
        mega_graph = nx.Graph()
        node_offset = 0
        
        for net in networks:
            # Add nodes with offset to avoid conflicts
            for node, data in net.nodes(data=True):
                new_node = (node[0] + node_offset * 0.001, node[1])  # Slight offset
                mega_graph.add_node(new_node, **data)
            
            # Add edges
            for u, v, data in net.edges(data=True):
                new_u = (u[0] + node_offset * 0.001, u[1])
                new_v = (v[0] + node_offset * 0.001, v[1])
                mega_graph.add_edge(new_u, new_v, **data)
            
            node_offset += 1
        
        return mega_graph
    
    def _apply_trained_gnn_to_tract(self, tract_data, trained_model):
        """Apply pre-trained GNN to individual tract with IDM comparison"""
        fips = tract_data['tract_info']['FIPS']
        svi_value = tract_data['svi_value']
        
        self._log(f"  Applying trained GNN to tract {fips} with SVI={svi_value:.3f}")
            
        try:
            # Step 1: Prepare graph data
            # Load NLCD features for this tract
            nlcd_features = self._load_nlcd_for_tract(tract_data)

            if nlcd_features is not None and len(nlcd_features) > 0:
                graph_data, _ = prepare_graph_data_with_nlcd(
                    tract_data['road_network'], 
                    nlcd_features,
                    addresses=tract_data['addresses']
                )
            else:
                graph_data, _ = prepare_graph_data_topological(tract_data['road_network'])
            
            # Step 2: Extract features using pre-trained model
            trained_model.eval()
            with torch.no_grad():
                gnn_features = trained_model(graph_data.x, graph_data.edge_index).cpu().numpy()
            
            self._log(f"  Applied trained GNN: κ range [{gnn_features[:, 0].min():.3f}, {gnn_features[:, 0].max():.3f}]")
            
            # Step 3: Create MetricGraph
            self._log("  Creating MetricGraph representation...")
            mg_start = time.time()
            
            nodes_df, edges_df = self._prepare_metricgraph_data(tract_data['road_network'])
            
            metric_graph = self.mg_interface.create_graph(
                nodes_df, edges_df,
                enable_sampling=len(edges_df) > self.config['metricgraph']['max_edges']
            )
            
            if metric_graph is None:
                raise RuntimeError("Failed to create MetricGraph")
            
            mg_time = time.time() - mg_start
            self._log(f"  MetricGraph created in {mg_time:.2f}s")
            
            # Step 4: Prepare observation and prediction data
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
            
            # Step 5: Perform disaggregation with error checking
            self._log("  Performing GNN-informed spatial disaggregation...")
            disagg_result = self.mg_interface.disaggregate_svi(
                metric_graph=metric_graph,
                tract_observation=tract_observation,
                prediction_locations=prediction_locations,
                gnn_features=gnn_features,
                alpha=self.config['metricgraph']['alpha']
            )
            
            if disagg_result is None:
                raise RuntimeError("Disaggregation returned None")
            
            if not disagg_result.get('success', False):
                raise RuntimeError(f"Disaggregation failed: {disagg_result.get('error', 'Unknown error')}")
            
            # Step 6: Run IDM baseline 
            self._log("  Running IDM baseline for comparison...")
            
            # Load NLCD features 
            nlcd_features = self._load_nlcd_for_tract(tract_data)
            
            idm_result = self.mg_interface._idm_baseline(
                tract_observation=tract_observation,
                prediction_locations=prediction_locations,
                nlcd_features=nlcd_features,  
                tract_geometry=tract_data['tract_info'].geometry  
            )
            
            if idm_result is None:
                self._log("  WARNING: IDM baseline failed, continuing without it")
                idm_result = {'success': False, 'error': 'IDM baseline failed'}
        
            # Step 7: Process results with IDM comparison
            predictions = disagg_result['predictions']
            predictions['fips'] = fips
            predictions['tract_svi'] = svi_value
            
            # Compute validation metrics (GNN vs IDM)
            validation_metrics = self._compute_idm_validation_metrics(
                predictions, idm_result, svi_value
            )
            
            self._log(f"  Successfully disaggregated to {len(predictions)} addresses")
            self._log(f"  Constraint satisfied: {disagg_result['diagnostics']['constraint_satisfied']}")
            
            if validation_metrics.get('gnn_vs_idm_correlation') is not None:
                self._log(f"  GNN vs IDM correlation: {validation_metrics['gnn_vs_idm_correlation']:.3f}")
            
            if disagg_result['success'] and idm_result.get('success'):
                self._log("🔍 Running diagnostic analysis...")
                
                try:
                    gnn_predictions = disagg_result['predictions']
                    idm_predictions = idm_result['predictions']
                    
                    # Run diagnostics
                    diagnostic_results = diagnose_comparison_issues(
                        gnn_predictions, idm_predictions, svi_value
                    )
                    
                    # Create diagnostic plots
                    diagnostic_fig = create_diagnostic_plots(
                        gnn_predictions, idm_predictions, svi_value
                    )
                    
                    # Save diagnostic plot
                    diagnostic_path = os.path.join(self.output_dir, f'diagnostics_tract_{fips}.png')
                    diagnostic_fig.savefig(diagnostic_path, dpi=300, bbox_inches='tight')
                    plt.close(diagnostic_fig)
                    
                    self._log(f"Diagnostic plots saved to {diagnostic_path}")
                    
                    # Log key findings
                    if diagnostic_results['issues_found']:
                        self._log("DIAGNOSTIC ISSUES FOUND:")
                        for issue in diagnostic_results['issues_found']:
                            self._log(f"  - {issue}")
                    else:
                        self._log("No major diagnostic issues detected")
                        
                    # Add diagnostics to return data
                    return {
                        'status': 'success',
                        'fips': fips,
                        'predictions': predictions,
                        'gnn_features': gnn_features,
                        'spde_params': disagg_result['spde_params'],
                        'diagnostics': disagg_result['diagnostics'],
                        'idm_comparison': idm_result,
                        'validation_metrics': validation_metrics,
                        'diagnostic_results': diagnostic_results,  
                        'diagnostic_plot_path': diagnostic_path,  
                        'network_data': self._prepare_network_data_for_viz(tract_data),
                        'timing': {'total': mg_time}
                    }
                    
                except Exception as e:
                    self._log(f"Diagnostic analysis failed: {str(e)}")
                    # Continue without diagnostics

            return {
                'status': 'success',
                'fips': fips,
                'predictions': predictions,
                'gnn_features': gnn_features,
                'spde_params': disagg_result['spde_params'],
                'diagnostics': disagg_result['diagnostics'],
                'idm_comparison': idm_result, 
                'validation_metrics': validation_metrics,
                'network_data': self._prepare_network_data_for_viz(tract_data),
                'timing': {'total': mg_time}
            }
            
        except Exception as e:
            self._log(f"  Error applying GNN to tract: {str(e)}")
            return {'status': 'failed', 'fips': fips, 'error': str(e)}

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
        
        # Process single tract
        return self._process_county_mode(single_tract_data)
    
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
        """
        Process a single tract with NLCD features
        """
        fips = tract_data['tract_info']['FIPS']
        svi_value = tract_data['svi_value']
        
        self._log(f"  Processing tract {fips} with SVI={svi_value:.3f}")
        
        try:
            # Load NLCD features
            nlcd_features = self._load_nlcd_for_tract(tract_data)
            
            # Step 1: Train GNN with NLCD or topological features
            self._log("  Training GNN for accessibility feature learning...")
            gnn_start = time.time()
            
            # Choose feature preparation method based on NLCD availability
            if nlcd_features is not None and len(nlcd_features) > 0:
                self._log("  Using NLCD-based features")
                graph_data, node_mapping = prepare_graph_data_with_nlcd(
                    tract_data['road_network'], 
                    nlcd_features,
                    addresses=tract_data['addresses']  
                )
            else:
                self._log("  WARNING: Falling back to topological features")
                graph_data, node_mapping = prepare_graph_data_topological(
                    tract_data['road_network']
                )
            
            self._log(f"DEBUG: graph_data type = {type(graph_data)}")
            if isinstance(graph_data, tuple):
                self._log(f"DEBUG: graph_data is tuple with {len(graph_data)} elements")
                graph_data = graph_data[0]  # Take first element
                self._log(f"DEBUG: After unpacking, graph_data type = {type(graph_data)}")

            # Verify it has the expected attributes
            if not hasattr(graph_data, 'x'):
                raise RuntimeError(f"graph_data missing 'x' attribute. Type: {type(graph_data)}, Attributes: {dir(graph_data)}")

            self._log(f"DEBUG: graph_data type = {type(graph_data)}")
            if isinstance(graph_data, tuple):
                self._log(f"DEBUG: graph_data is tuple with {len(graph_data)} elements")
                graph_data = graph_data[0]  # Take first element
                self._log(f"DEBUG: After unpacking, graph_data type = {type(graph_data)}")

            # Verify it has the expected attributes
            if not hasattr(graph_data, 'x'):
                raise RuntimeError(f"graph_data missing 'x' attribute. Type: {type(graph_data)}, Attributes: {dir(graph_data)}")


            # Create and train GNN model
            model = create_gnn_model(
                input_dim=graph_data.x.shape[1],
                hidden_dim=self.config['model']['hidden_dim'],
                output_dim=self.config['model']['output_dim']
            )
            
            feature_history = [] 

            trained_model, gnn_features = train_accessibility_gnn(
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
            #baseline_result = self.mg_interface._random_baseline_test(
            baseline_result = self.mg_interface._idm_baseline(
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

                # Compute proper validation metrics
                validation_metrics = self._compute_proper_validation_metrics(
                    predictions, baseline_result, svi_value
                )
                
                # Log validation results
                if validation_metrics['method_correlation'] is not None:
                    self._log(f"    Method correlation: {validation_metrics['method_correlation']:.3f}")
                    if validation_metrics['method_correlation'] < 0.1:
                        self._log("    LOW CORRELATION - CHECK IMPLEMENTATION")
                
                self._log(f"    Constraint error: {validation_metrics['constraint_error']:.1%}")
                self._log(f"    Prediction variance: {validation_metrics['prediction_variance']:.6f}")
                
                return {
                    'status': 'success',
                    'fips': fips,
                    'predictions': predictions,
                    'gnn_features': gnn_features,
                    'spde_params': disagg_result['spde_params'],
                    'diagnostics': diagnostics,
                    'baseline_comparison': baseline_result,
                    'validation_metrics': validation_metrics, 
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
    
    def _process_multi_fips_mode(self, data, target_fips_list):
        """Process specific list of FIPS codes with global GNN training"""
        self._log(f"Processing {len(target_fips_list)} specific FIPS codes with shared GNN training")
        
        # Convert target_fips_list to strings for consistent matching
        target_fips_list = [str(fips).strip() for fips in target_fips_list]
        data['tracts']['FIPS'] = data['tracts']['FIPS'].astype(str).str.strip()
        
        self._log(f"Looking for FIPS codes: {target_fips_list}")
        
        # Filter to target tracts only
        target_tracts = data['tracts'][data['tracts']['FIPS'].isin(target_fips_list)]
        
        if len(target_tracts) == 0:
            self._log("No target tracts found in data")
            available_fips = data['tracts']['FIPS'].tolist()[:10]
            self._log(f"Available FIPS (first 10): {available_fips}")
            return {'success': False, 'error': 'No target FIPS found'}
        
        self._log(f"✓ Found {len(target_tracts)} target tracts out of {len(target_fips_list)} requested:")
        for _, tract in target_tracts.iterrows():
            self._log(f"  - {tract['FIPS']}: SVI = {tract.get('RPL_THEMES', 'N/A')}")
        
        # Create filtered dataset
        filtered_data = data.copy()
        filtered_data['tracts'] = target_tracts
        
        # Use the county mode processing (which does global training)
        return self._process_county_mode(filtered_data)

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

    def _combine_results(self, tract_results):
        """
        Combine results with proper IDM baseline aggregation
        """
        if not tract_results:
            return {
                'success': False,
                'message': 'No tracts successfully processed'
            }
        
        # Combine all predictions
        all_predictions = []
        all_idm_predictions = [] 
        
        for result in tract_results:
            if result['status'] == 'success':
                # Add GNN predictions
                all_predictions.append(result['predictions'])
                
                # Add IDM predictions if available
                if (result.get('idm_comparison') and 
                    result['idm_comparison'].get('success') and
                    'predictions' in result['idm_comparison']):
                    
                    idm_pred = result['idm_comparison']['predictions'].copy()
                    # Add tract metadata to IDM predictions
                    idm_pred['fips'] = result['fips']
                    idm_pred['tract_svi'] = result['predictions']['tract_svi'].iloc[0]
                    all_idm_predictions.append(idm_pred)
        
        if all_predictions:
            combined_predictions = pd.concat(all_predictions, ignore_index=True)
            
            # Create combined IDM comparison with proper naming
            combined_idm = None
            if all_idm_predictions:  
                combined_idm_predictions = pd.concat(all_idm_predictions, ignore_index=True)
                
                if len(combined_predictions) == len(combined_idm_predictions):
                    combined_idm = {
                        'success': True,
                        'predictions': combined_idm_predictions,
                        'diagnostics': {
                            'constraint_satisfied': True,
                            'constraint_error': 0.0,
                            'mean_prediction': combined_idm_predictions['mean'].mean(),
                            'std_prediction': combined_idm_predictions['mean'].std(),
                            'mean_uncertainty': combined_idm_predictions['sd'].mean(),
                            'method': 'combined_IDM'
                        }
                    }
                    
                    self._log(f"Combined IDM baseline: {len(combined_idm_predictions)} predictions")
            
            # Compute global validation
            global_validation = self._compute_global_idm_validation_metrics(
                combined_predictions, combined_idm
            )

            summary_stats = {
                'total_tracts': len(tract_results),
                'successful_tracts': sum(1 for r in tract_results if r['status'] == 'success'),
                'total_addresses': len(combined_predictions),
                'mean_svi': combined_predictions['mean'].mean(),
                'std_svi': combined_predictions['mean'].std(),
                'mean_uncertainty': combined_predictions['sd'].mean(),
                'processing_time': sum(r.get('timing', {}).get('total', 0) for r in tract_results),
                'global_correlation': global_validation.get('method_correlation')
            }
            
            return {
                'success': True,
                'predictions': combined_predictions,
                'combined_idm': combined_idm,  
                'tract_results': tract_results,
                'summary': summary_stats,
                'global_validation': global_validation
            }
    
    def _compute_global_idm_validation_metrics(self, combined_predictions, combined_idm):
        """
        Compute validation metrics across all tracts (GNN vs IDM)
        """
        if not combined_idm or not combined_idm.get('success'):
            return {'method_correlation': None, 'error': 'No IDM baseline available'}
        
        gnn_predictions = combined_predictions['mean'].values
        idm_predictions = combined_idm['predictions']['mean'].values
        
        # Verify same length
        if len(gnn_predictions) != len(idm_predictions):
            self._log(f"GLOBAL WARNING: Prediction length mismatch - "
                    f"GNN: {len(gnn_predictions)}, IDM: {len(idm_predictions)}")
            return {'method_correlation': None, 'error': 'Length mismatch'}
        
        # Compute global correlation
        try:
            correlation = np.corrcoef(gnn_predictions, idm_predictions)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        except Exception as e:
            self._log(f"Error computing global correlation: {str(e)}")
            correlation = None
        
        # Additional global metrics
        gnn_uncertainty = combined_predictions['sd'].values
        idm_uncertainty = combined_idm['predictions']['sd'].values
        
        # Effectiveness Calculations
        gnn_variance = np.var(gnn_predictions)
        idm_variance = np.var(idm_predictions)
        
        # Spatial differentiation score (higher variance often better)
        spatial_score = gnn_variance / idm_variance if idm_variance > 0 else 1.0
        
        # Overall effectiveness
        effectiveness = spatial_score * abs(correlation) if correlation is not None else spatial_score
        
        global_metrics = {
            'method_correlation': correlation,
            'gnn_prediction_range': [gnn_predictions.min(), gnn_predictions.max()],
            'idm_prediction_range': [idm_predictions.min(), idm_predictions.max()],
            'gnn_prediction_std': np.std(gnn_predictions),
            'idm_prediction_std': np.std(idm_predictions),
            'gnn_uncertainty_mean': np.mean(gnn_uncertainty),
            'idm_uncertainty_mean': np.mean(idm_uncertainty),
            'total_addresses': len(gnn_predictions),
            'spatial_score': spatial_score,
            'gnn_effectiveness': effectiveness
        }
        
        self._log(f"GLOBAL GNN vs IDM CORRELATION: {correlation:.3f}")
        self._log(f"  GNN spatial std: {global_metrics['gnn_prediction_std']:.6f}")
        self._log(f"  IDM spatial std: {global_metrics['idm_prediction_std']:.6f}")
        self._log(f"  GNN effectiveness score: {effectiveness:.3f}")
        
        return global_metrics

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
                    self._log(f"  🚨 CRITICAL: GNN shows {variation_ratio:.1f}x less spatial variation than IDM")
                    self._log(f"     This suggests GNN over-smoothing or feature uniformity issues")
            else:
                self._log(f"✅ Good correlation with IDM baseline: {correlation:.3f}")
                
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
        
        # Get visualization data from first successful tract
        viz_data = None
        gnn_predictions = results['predictions']
        idm_predictions = None
        
        if results.get('tract_results'):
            for tract_result in results['tract_results']:
                if (tract_result.get('status') == 'success' and 
                    tract_result.get('network_data')):
                    viz_data = tract_result
                    
                    # EXTRACT IDM predictions for clear comparison
                    if (tract_result.get('idm_comparison') and 
                        tract_result['idm_comparison'].get('success')):
                        idm_predictions = tract_result['idm_comparison']['predictions']
                        self._log("✓ Found IDM comparison data for visualization")
                    break
        
        if viz_data and idm_predictions is not None:
            # 1. Create GNN vs IDM comparison
            clear_comparison_path = os.path.join(self.output_dir, 'gnn_vs_idm_comparison.png')
            self.visualizer.create_clear_method_comparison(
                gnn_predictions=gnn_predictions,
                idm_predictions=idm_predictions,
                gnn_results=results,
                idm_results=viz_data.get('idm_comparison'),
                output_path=clear_comparison_path
            )
            self._log(f"Saved GNN vs IDM comparison to {clear_comparison_path}")
            
            # 2. Keep original visualization as backup
            original_viz_path = os.path.join(self.output_dir, 'granite_visualization.png')
            self.visualizer.create_disaggregation_plot(
                predictions=results['predictions'],
                results=results,
                comparison_results=results.get('combined_idm'),  # Use IDM instead of baseline
                output_path=original_viz_path
            )
            self._log(f"Saved original visualization to {original_viz_path}")
            
            # 3. Create additional specialized visualizations
            if viz_data.get('feature_history') and len(viz_data['feature_history']) > 1:
                evolution_path = os.path.join(self.output_dir, 'gnn_feature_evolution.png')
                self.visualizer.plot_gnn_feature_evolution(
                    viz_data['feature_history'],
                    output_path=evolution_path
                )
                self._log(f"Saved feature evolution plot to {evolution_path}")
            
            # 4. Accessibility gradients
            gradients_path = os.path.join(self.output_dir, 'accessibility_gradients.png')
            self.visualizer.plot_accessibility_gradients(
                predictions=results['predictions'],
                network_data=viz_data['network_data'],
                transit_data=viz_data.get('transit_data'),
                output_path=gradients_path
            )
            self._log(f"Saved accessibility gradients to {gradients_path}")
            
            # 5. GNN attention maps 
            if viz_data.get('attention_weights') is not None:
                attention_path = os.path.join(self.output_dir, 'gnn_attention_maps.png')
                self.visualizer.plot_gnn_attention_maps(
                    attention_weights=viz_data['attention_weights'],
                    network_data=viz_data['network_data'],
                    predictions=results['predictions'],
                    output_path=attention_path
                )
                self._log(f"Saved attention maps to {attention_path}")
            
            # 6. Uncertainty sources analysis
            uncertainty_path = os.path.join(self.output_dir, 'uncertainty_analysis.png')
            self.visualizer.plot_uncertainty_sources(
                predictions=results['predictions'],
                network_data=viz_data.get('network_data'),
                output_path=uncertainty_path
            )
            self._log(f"Saved uncertainty analysis to {uncertainty_path}")
            
            # 7. Model interpretability dashboard
            if viz_data.get('gnn_features') is not None:
                interpret_path = os.path.join(self.output_dir, 'model_interpretability.png')
                self.visualizer.plot_model_interpretability(
                    predictions=results['predictions'],
                    gnn_features=viz_data['gnn_features'],
                    network_data=viz_data.get('network_data'),
                    output_path=interpret_path
                )
                self._log(f"Saved interpretability dashboard to {interpret_path}")
            
            # 8. Print comparison summary to console
            self._print_comparison_summary(gnn_predictions, idm_predictions)
        
        else:
            # Fallback: No IDM comparison available
            self._log("No IDM comparison data available - creating single-method visualization")
            
            fallback_path = os.path.join(self.output_dir, 'granite_visualization.png')
            self.visualizer.create_disaggregation_plot(
                predictions=results['predictions'],
                results=results,
                output_path=fallback_path
            )
            self._log(f"Saved fallback visualization to {fallback_path}")
        
        # Global validation summary
        if results.get('global_validation'):
            self._log(f"\n=== GLOBAL VALIDATION SUMMARY ===")
            global_val = results['global_validation']
            if global_val.get('method_correlation'):
                self._log(f"Global GNN-IDM Correlation: {global_val['method_correlation']:.3f}")
            self._log(f"Total Addresses Compared: {global_val.get('total_addresses', 'N/A')}")

        self._log(f"\nAll results and visualizations saved to {self.output_dir}/")

    def _print_comparison_summary(self, gnn_predictions, idm_predictions):
        """
        Print a clear summary of GNN vs IDM comparison to console
        """
        try:
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
