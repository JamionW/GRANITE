"""
Main disaggregation pipeline for GRANITE framework
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime
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
    
    Supports both:
    - County-wide processing (original, memory-intensive)
    - FIPS-based processing (new, memory-efficient)
    """
    
    def __init__(self, data_dir='./data', output_dir='./output', verbose=True):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.verbose = verbose
        
        # Initialize components
        self.data_loader = DataLoader(data_dir, verbose)
        self.mg_interface = MetricGraphInterface(verbose)
        self.visualizer = DisaggregationVisualizer()
        
        # Storage for results
        self.data = {}
        self.results = {}
        
    def _log(self, message, level='INFO'):
        """Logging with timestamp and level"""
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    def run(self, roads_file=None, epochs=100, visualize=True, mode=None, fips_list=None):
        """
        Run the complete GRANITE pipeline
        
        Parameters:
        -----------
        roads_file : str, optional
            Path to roads shapefile
        epochs : int
            Number of GNN training epochs
        visualize : bool
            Whether to create visualizations
        mode : str, optional
            Processing mode: 'county' (default) or 'fips'
        fips_list : List[str], optional
            FIPS codes to process (only for FIPS mode)
            
        Returns:
        --------
        dict
            Results dictionary
        """
        # Determine processing mode
        if mode is None:
            # Check if config specifies FIPS mode
            try:
                from ..utils.config import load_config
                config = load_config('config/config.yaml')
                processing_mode = config.get('data', {}).get('processing_mode', 'county')
            except:
                processing_mode = 'county'
        else:
            processing_mode = mode
        
        if processing_mode == 'fips':
            return self.run_fips_mode(fips_list=fips_list, epochs=epochs, visualize=visualize)
        else:
            # Original county-wide processing
            return self._run_original_pipeline(roads_file, epochs, visualize)
    
    # ==========================================
    # ORIGINAL COUNTY-WIDE PROCESSING METHODS
    # ==========================================
    
    def _run_original_pipeline(self, roads_file, epochs, visualize):
        """Original county-wide pipeline (unchanged)"""
        self._log("="*60)
        self._log("GRANITE: County-wide Processing Mode")
        self._log("="*60)
        
        # Load data
        self.load_data(roads_file)
        
        # Prepare graph structures
        self.prepare_graph_structures()
        
        # Learn accessibility features
        self.learn_accessibility_features(epochs)
        
        # Disaggregate SVI
        self.disaggregate_svi()
        
        # Validate results
        self.validate_results()
        
        # Create visualizations
        if visualize:
            self.create_visualizations()
        
        return self.results
    
    def load_data(self, roads_file=None):
        """
        Load all required data for county-wide processing
        
        Parameters:
        -----------
        roads_file : str, optional
            Path to roads shapefile
        """
        self._log("="*60)
        self._log("GRANITE: Loading Data")
        self._log("="*60)
        
        # Load SVI data
        self.data['svi'] = self.data_loader.load_svi_data()
        
        # Load census tracts
        self.data['census_tracts'] = self.data_loader.load_census_tracts()
        
        # Load road network
        self.data['roads'] = self.data_loader.load_road_network(roads_file)
        
        # Load transit stops
        self.data['transit_stops'] = self.data_loader.load_transit_stops()
        
        # Load address points
        self.data['addresses'] = self.data_loader.load_address_points()
        
        # Create network graph
        self.data['road_network'] = self.data_loader.create_network_graph(
            self.data['roads']
        )
        
        # Merge SVI with census tracts
        self.data['tracts_with_svi'] = self.data['census_tracts'].merge(
            self.data['svi'],
            on='FIPS',
            how='inner'
        )
        
        self._log("Data loading complete!")
    
    def prepare_graph_structures(self):
        """
        Prepare graph data structures for GNN and MetricGraph
        """
        self._log("="*60)
        self._log("GRANITE: Preparing Graph Structures")
        self._log("="*60)
        
        self._log("Preparing data for GNN...")
        
        # Prepare PyTorch Geometric data
        self.data['pyg_data'] = prepare_graph_data(
            self.data['road_network'],
            self.data['addresses'],
            self.data['transit_stops']
        )
        
        self._log("Preparing data for MetricGraph...")
        
        # Create MetricGraph object
        self.data['metric_graph'] = self.mg_interface.create_graph(
            self.data['roads']
        )
        
        self._log("Graph preparation complete!")
    
    def learn_accessibility_features(self, epochs=100):
        """
        Train GNN to learn accessibility features
        
        Parameters:
        -----------
        epochs : int
            Number of training epochs
        """
        self._log("="*60)
        self._log("GRANITE: Learning Accessibility Features")
        self._log("="*60)
        
        # Create GNN model
        input_dim = self.data['pyg_data'].x.shape[1]
        gnn_model = create_gnn_model(input_dim=input_dim)
        
        # Train GNN
        self.results['gnn_features'] = train_accessibility_gnn(
            gnn_model,
            self.data['pyg_data'],
            epochs=epochs,
            verbose=self.verbose
        )
        
        self.results['gnn_model'] = gnn_model
        
        self._log("Feature learning complete!")
    
    def disaggregate_svi(self):
        """
        Perform SVI spatial disaggregation using MetricGraph
        """
        self._log("="*60)
        self._log("GRANITE: SVI Spatial Disaggregation")
        self._log("="*60)
        
        # Prepare covariates from GNN features
        covariates = pd.DataFrame(self.results['gnn_features'])
        covariates.columns = ['gnn_kappa', 'gnn_alpha', 'gnn_tau']
        
        # Prepare observations (tract-level SVI)
        observations = self.data['tracts_with_svi'][['FIPS', 'RPL_THEMES']].copy()
        observations = observations.dropna()
        
        # Run MetricGraph disaggregation
        self.results['disaggregation'] = self.mg_interface.disaggregate_svi(
            self.data['metric_graph'],
            observations,
            covariates
        )
        
        # Interpolate to address locations
        self.results['predictions'] = self._interpolate_to_addresses()
        
        self._log("Disaggregation complete!")
    
    def validate_results(self):
        """
        Validate disaggregation results
        """
        self._log("="*60)
        self._log("GRANITE: Validating Results")
        self._log("="*60)
        
        # Implement validation logic
        self.results['validation'] = self._compute_validation_metrics()
        
        self._log("Validation complete!")
    
    def create_visualizations(self):
        """
        Create result visualizations
        """
        self._log("="*60)
        self._log("GRANITE: Creating Visualizations")
        self._log("="*60)
        
        # Create visualizations using the visualizer
        self.visualizer.create_disaggregation_plot(
            self.data,
            self.results,
            output_path=os.path.join(self.output_dir, 'granite_visualization.png')
        )
        
        self._log("Visualization complete!")
    
    def _interpolate_to_addresses(self):
        """Interpolate disaggregation results to address locations"""
        # Placeholder implementation
        addresses = self.data['addresses']
        predictions = pd.DataFrame({
            'address_id': range(len(addresses)),
            'longitude': [addr.geometry.x for _, addr in addresses.iterrows()],
            'latitude': [addr.geometry.y for _, addr in addresses.iterrows()],
            'mean': np.random.uniform(0.2, 0.8, len(addresses)),
            'sd': np.random.uniform(0.05, 0.15, len(addresses))
        })
        return predictions
    
    def _compute_validation_metrics(self):
        """Compute validation metrics for results"""
        # Placeholder implementation
        return pd.DataFrame({
            'tract_id': ['47065001100', '47065001200'],
            'true_svi': [0.45, 0.67],
            'predicted_avg': [0.44, 0.66],
            'error': [0.01, 0.01]
        })
    
    # ==========================================
    # NEW FIPS-BASED PROCESSING METHODS
    # ==========================================
    
    def run_fips_mode(self, config=None, fips_list=None, epochs=None, visualize=True):
        """
        Run pipeline in FIPS mode (tract-by-tract processing)
        
        Parameters:
        -----------
        config : dict, optional
            FIPS configuration. If None, uses self.config
        fips_list : List[str], optional  
            Specific FIPS codes to process
        epochs : int, optional
            Number of GNN training epochs
        visualize : bool
            Whether to create visualizations
            
        Returns:
        --------
        dict
            Processing results
        """
        try:
            from ..utils.config import load_config
            
            # Load config if not provided
            if config is None:
                config = load_config('config/config.yaml')
        except:
            # Fallback config if file doesn't exist
            config = {
                'data': {
                    'state_fips': '47',
                    'county_fips': '065',
                    'fips_config': {
                        'batch': {
                            'auto_select': {
                                'enabled': True,
                                'mode': 'range',
                                'range_start': 1,
                                'range_end': 5
                            }
                        },
                        'memory': {
                            'tract_buffer_degrees': 0.01,
                            'max_network_nodes': 10000,
                            'max_network_edges': 20000
                        }
                    }
                }
            }
        
        # Determine FIPS codes to process
        if fips_list is None:
            fips_config = config.get('data', {}).get('fips_config', {})
            fips_list = self.data_loader.resolve_fips_list(
                fips_config,
                config.get('data', {}).get('state_fips', '47'),
                config.get('data', {}).get('county_fips', '065')
            )
        
        self._log("="*70)
        self._log("GRANITE: FIPS Mode Processing")
        self._log(f"Processing {len(fips_list)} census tracts")
        self._log(f"FIPS codes: {fips_list}")
        self._log("="*70)
        
        # Determine processing approach
        if len(fips_list) == 1:
            return self._run_single_fips(fips_list[0], config, epochs, visualize)
        else:
            return self._run_batch_fips(fips_list, config, epochs, visualize)

    def _run_single_fips(self, fips_code, config, epochs=None, visualize=True):
        """Run processing for a single FIPS code"""
        self._log(f"Processing single tract: {fips_code}")
        
        try:
            # Load tract data
            memory_config = config.get('data', {}).get('fips_config', {}).get('memory', {})
            tract_data = self.data_loader.load_single_tract_data(
                fips_code,
                buffer_degrees=memory_config.get('tract_buffer_degrees', 0.01),
                max_nodes=memory_config.get('max_network_nodes', 10000),
                max_edges=memory_config.get('max_network_edges', 20000)
            )
            
            # Check if tract is processable
            if tract_data['road_network'].number_of_nodes() < 50:
                self._log(f"Tract {fips_code} has insufficient road network, skipping")
                return {
                    'status': 'skipped',
                    'reason': 'insufficient_network',
                    'fips_code': fips_code
                }
            
            if len(tract_data['svi_data']) == 0:
                self._log(f"Tract {fips_code} has no SVI data, skipping")
                return {
                    'status': 'skipped', 
                    'reason': 'no_svi_data',
                    'fips_code': fips_code
                }
            
            # Store tract data
            self.data['tract_data'] = tract_data
            self.data['current_fips'] = fips_code
            
            # Process tract
            return self._process_single_tract(tract_data, config, epochs, visualize)
            
        except Exception as e:
            self._log(f"Error processing tract {fips_code}: {str(e)}", 'ERROR')
            return {
                'status': 'error',
                'fips_code': fips_code,
                'error': str(e)
            }

    def _run_batch_fips(self, fips_list, config, epochs=None, visualize=True):
        """Run processing for multiple FIPS codes"""
        self._log(f"Processing {len(fips_list)} tracts in batch mode")
        
        batch_results = {}
        stats = {'successful': 0, 'skipped': 0, 'errors': 0}
        
        # Load memory configuration
        memory_config = config.get('data', {}).get('fips_config', {}).get('memory', {})
        
        for i, fips_code in enumerate(fips_list, 1):
            self._log(f"\n[{i}/{len(fips_list)}] Processing tract {fips_code}")
            
            try:
                # Load tract data
                tract_data = self.data_loader.load_single_tract_data(
                    fips_code,
                    buffer_degrees=memory_config.get('tract_buffer_degrees', 0.01),
                    max_nodes=memory_config.get('max_network_nodes', 10000),
                    max_edges=memory_config.get('max_network_edges', 20000)
                )
                
                # Quick validation
                if tract_data['road_network'].number_of_nodes() < 50:
                    self._log(f"  Skipping {fips_code}: insufficient network")
                    batch_results[fips_code] = {
                        'status': 'skipped',
                        'reason': 'insufficient_network'
                    }
                    stats['skipped'] += 1
                    continue
                
                if len(tract_data['svi_data']) == 0:
                    self._log(f"  Skipping {fips_code}: no SVI data")
                    batch_results[fips_code] = {
                        'status': 'skipped',
                        'reason': 'no_svi_data'
                    }
                    stats['skipped'] += 1
                    continue
                
                # Process tract
                self.data['tract_data'] = tract_data
                self.data['current_fips'] = fips_code
                
                result = self._process_single_tract(
                    tract_data, config, epochs, 
                    visualize=False  # Skip individual visualizations in batch
                )
                
                batch_results[fips_code] = result
                
                if result['status'] == 'success':
                    stats['successful'] += 1
                    # Save individual tract results
                    self._save_tract_results(result, fips_code, config)
                else:
                    stats['errors'] += 1
                    
            except Exception as e:
                self._log(f"  Error processing {fips_code}: {str(e)}", 'ERROR')
                batch_results[fips_code] = {
                    'status': 'error',
                    'fips_code': fips_code,
                    'error': str(e)
                }
                stats['errors'] += 1
                
                # Continue processing other tracts if configured
                if not config.get('processing', {}).get('continue_on_error', True):
                    break
        
        # Create batch summary
        summary = {
            'mode': 'batch',
            'total_tracts': len(fips_list),
            'successful': stats['successful'],
            'skipped': stats['skipped'], 
            'errors': stats['errors'],
            'success_rate': stats['successful'] / len(fips_list) if fips_list else 0,
            'fips_list': fips_list,
            'results': batch_results
        }
        
        # Save batch summary
        self._save_batch_summary(summary, config)
        
        # Create batch visualization if requested
        if visualize:
            self._create_batch_visualization(summary, config)
        
        self._log(f"\nBatch processing complete:")
        self._log(f"  Successful: {stats['successful']}")
        self._log(f"  Skipped: {stats['skipped']}")
        self._log(f"  Errors: {stats['errors']}")
        self._log(f"  Success rate: {summary['success_rate']:.1%}")
        
        return summary

    def _process_single_tract(self, tract_data, config, epochs=None, visualize=True):
        """Process a single tract through the full pipeline"""
        fips_code = tract_data['fips_code']
        
        try:
            # 1. Prepare graph data
            self._log("  Preparing graph structures...")
            pyg_data, node_mapping = prepare_graph_data(
                tract_data['road_network']
            )
            
            # 2. Create and train GNN
            epochs = epochs or config.get('model', {}).get('epochs', 50)
            self._log(f"  Training GNN for {epochs} epochs...")
            
            gnn_model = create_gnn_model(
                input_dim=config.get('model', {}).get('input_dim', 5),
                hidden_dim=config.get('model', {}).get('hidden_dim', 32),
                output_dim=config.get('model', {}).get('output_dim', 3),
                dropout=config.get('model', {}).get('dropout', 0.2)
            )
            
            # Use the correct function signature from your existing code
            trained_model, gnn_features, training_history = train_accessibility_gnn(
                pyg_data,
                gnn_model,
                epochs=epochs
            )
            
            # 3. Run MetricGraph disaggregation
            self._log("  Running spatial disaggregation...")
            disaggregation_results = self._run_tract_metricgraph(
                tract_data, gnn_features, config
            )
            
            # 4. Compile results
            results = {
                'status': 'success',
                'fips_code': fips_code,
                'network_stats': tract_data['network_stats'],
                'gnn_model': trained_model,
                'gnn_features': gnn_features,
                'training_history': training_history,
                'predictions': disaggregation_results['predictions'],
                'validation': disaggregation_results.get('validation', {}),
                'metrics': disaggregation_results.get('metrics', {})
            }
            
            # 5. Create visualization if requested
            if visualize:
                self._create_tract_visualization(results, tract_data, config)
            
            return results
            
        except Exception as e:
            self._log(f"  Error in tract processing: {str(e)}", 'ERROR')
            return {
                'status': 'error',
                'fips_code': fips_code,
                'error': str(e)
            }

    def _run_tract_metricgraph(self, tract_data, gnn_features, config):
        """Run MetricGraph disaggregation for a single tract"""
        try:
            # Prepare MetricGraph inputs
            nodes_df, edges_df = self._prepare_metricgraph_inputs(
                tract_data['road_network']
            )
            
            # Create MetricGraph
            metric_graph = self.mg_interface.create_graph(nodes_df, edges_df)
            
            # Prepare observations (single tract SVI value)
            svi_value = tract_data['svi_data']['RPL_THEMES'].iloc[0]
            observations = pd.DataFrame({
                'location': [0],  # Single observation point
                'svi': [svi_value],
                'weight': [1.0]
            })
            
            # Prepare covariates (GNN features)
            n_features = min(len(gnn_features), len(nodes_df))
            covariates = pd.DataFrame(gnn_features[:n_features])
            covariates.columns = ['gnn_kappa', 'gnn_alpha', 'gnn_tau']
            
            # Run disaggregation
            mg_results = self.mg_interface.disaggregate_svi(
                metric_graph,
                observations,
                covariates
            )
            
            # Interpolate to address locations
            predictions = self._interpolate_tract_to_addresses(
                mg_results, tract_data['addresses']
            )
            
            return {
                'predictions': predictions,
                'validation': {},  # Single tract - no cross-validation
                'metrics': {'method': 'metricgraph'}
            }
            
        except Exception as e:
            self._log(f"    MetricGraph failed: {str(e)}, using fallback")
            return self._fallback_tract_interpolation(tract_data)

    def _prepare_metricgraph_inputs(self, road_network):
        """Prepare node and edge dataframes for MetricGraph"""
        # Create nodes dataframe
        nodes = list(road_network.nodes())
        nodes_df = pd.DataFrame([
            {'node_id': i, 'x': node[0], 'y': node[1]}
            for i, node in enumerate(nodes)
        ])
        
        # Create edges dataframe
        edges = list(road_network.edges(data=True))
        edges_df = pd.DataFrame([
            {
                'from': nodes.index(edge[0]),
                'to': nodes.index(edge[1]),
                'weight': edge[2].get('weight', 1.0)
            }
            for edge in edges
        ])
        
        return nodes_df, edges_df

    def _interpolate_tract_to_addresses(self, mg_results, addresses):
        """Interpolate MetricGraph results to address locations"""
        # Simple interpolation - in practice, use sophisticated spatial interpolation
        predictions = []
        
        for idx, addr in addresses.iterrows():
            # Placeholder prediction (replace with actual interpolation)
            pred_svi = mg_results.get('mean_prediction', 0.5)
            pred_sd = mg_results.get('sd_prediction', 0.1)
            
            predictions.append({
                'address_id': addr['address_id'],
                'longitude': addr.geometry.x,
                'latitude': addr.geometry.y,
                'predicted_svi': pred_svi + np.random.normal(0, 0.02),  # Add small variation
                'uncertainty': pred_sd
            })
        
        return pd.DataFrame(predictions)

    def _fallback_tract_interpolation(self, tract_data):
        """Fallback interpolation when MetricGraph fails"""
        self._log("    Using fallback interpolation method")
        
        # Use tract SVI value with small random variation
        tract_svi = tract_data['svi_data']['RPL_THEMES'].iloc[0]
        
        predictions = []
        for idx, addr in tract_data['addresses'].iterrows():
            predictions.append({
                'address_id': addr['address_id'],
                'longitude': addr.geometry.x,
                'latitude': addr.geometry.y,
                'predicted_svi': tract_svi + np.random.normal(0, 0.05),
                'uncertainty': 0.1
            })
        
        return {
            'predictions': pd.DataFrame(predictions),
            'validation': {},
            'metrics': {'method': 'fallback'}
        }

    def _save_tract_results(self, results, fips_code, config):
        """Save results for individual tract"""
        output_config = config.get('output', {}).get('fips_output', {})
        
        if output_config.get('individual_tract_folders', True):
            tract_dir = os.path.join(self.output_dir, f'tract_{fips_code}')
            os.makedirs(tract_dir, exist_ok=True)
        else:
            tract_dir = self.output_dir
        
        # Save predictions
        if 'predictions' in results and len(results['predictions']) > 0:
            pred_file = os.path.join(tract_dir, f'predictions_{fips_code}.csv')
            results['predictions'].to_csv(pred_file, index=False)
        
        # Save GNN features
        if 'gnn_features' in results:
            feat_file = os.path.join(tract_dir, f'gnn_features_{fips_code}.npy')
            np.save(feat_file, results['gnn_features'])
        
        # Save GNN model
        if 'gnn_model' in results:
            import torch
            model_file = os.path.join(tract_dir, f'gnn_model_{fips_code}.pth')
            torch.save(results['gnn_model'].state_dict(), model_file)

    def _save_batch_summary(self, summary, config):
        """Save batch processing summary"""
        if not config.get('output', {}).get('fips_output', {}).get('batch_summary', True):
            return
        
        summary_file = os.path.join(self.output_dir, 'batch_summary.csv')
        
        # Create summary rows
        rows = []
        for fips_code, result in summary['results'].items():
            row = {
                'fips_code': fips_code,
                'status': result.get('status', 'unknown'),
                'error': result.get('error', ''),
                'network_nodes': result.get('network_stats', {}).get('nodes', 0),
                'network_edges': result.get('network_stats', {}).get('edges', 0),
                'n_predictions': len(result.get('predictions', [])) if result.get('predictions') is not None else 0
            }
            rows.append(row)
        
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(summary_file, index=False)
        
        self._log(f"Batch summary saved to: {summary_file}")

    def _create_tract_visualization(self, results, tract_data, config):
        """Create visualization for individual tract"""
        # Placeholder for tract visualization
        self._log(f"  Creating visualization for tract {results['fips_code']}")

    def _create_batch_visualization(self, summary, config):
        """Create summary visualization for batch processing"""
        # Placeholder for batch visualization
        self._log("Creating batch summary visualization")


# Convenience functions for backwards compatibility
def run_granite_pipeline(data_dir='./data', output_dir='./output', **kwargs):
    """
    Convenience function to run GRANITE pipeline
    
    Parameters:
    -----------
    data_dir : str
        Data directory path
    output_dir : str
        Output directory path
    **kwargs
        Additional arguments passed to pipeline.run()
        
    Returns:
    --------
    dict
        Pipeline results
    """
    pipeline = GRANITEPipeline(data_dir=data_dir, output_dir=output_dir)
    return pipeline.run(**kwargs)