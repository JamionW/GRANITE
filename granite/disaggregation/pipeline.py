"""
Main disaggregation pipeline for GRANITE framework

This module implements the core pipeline for SVI disaggregation using
GNN-MetricGraph integration with Whittle-Matérn spatial processes.
"""
import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
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
    """
    
    def __init__(self, config: Dict = None, data_dir: str = './data', 
                 output_dir: str = './output', verbose: bool = True):
        """
        Initialize GRANITE pipeline
        
        Parameters:
        -----------
        config : Dict
            Configuration dictionary
        data_dir : str
            Data directory path
        output_dir : str
            Output directory path
        verbose : bool
            Enable verbose logging
        """
        self.config = config or {}
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.verbose = verbose
        
        # Initialize components
        self.data_loader = DataLoader(data_dir, verbose)
        self.mg_interface = MetricGraphInterface(config, verbose)
        self.visualizer = DisaggregationVisualizer()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Storage
        self.data = {}
        self.results = {}
        self.timing = {}
    
    def _log(self, message: str):
        """Log message with timestamp"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def _estimate_time_remaining(self, current: int, total: int, elapsed: float) -> str:
        """Estimate time remaining based on progress"""
        if current == 0:
            return "calculating..."
        
        rate = current / elapsed
        remaining = (total - current) / rate
        
        if remaining < 60:
            return f"{remaining:.0f}s"
        elif remaining < 3600:
            return f"{remaining/60:.1f}m"
        else:
            return f"{remaining/3600:.1f}h"
    
    def run(self) -> Dict:
        """
        Run the GRANITE pipeline based on configuration
        
        Returns:
        --------
        Dict
            Processing results
        """
        mode = self.config.get('data', {}).get('processing_mode', 'fips')
        
        if mode == 'fips':
            return self._run_fips_mode()
        else:
            return self._run_county_mode()
    
    def _run_fips_mode(self) -> Dict:
        """Run FIPS-based processing"""
        start_time = time.time()
        
        self._log("="*60)
        self._log("GRANITE: FIPS-Based Processing Mode")
        self._log("="*60)
        
        # Resolve FIPS codes to process
        fips_config = self.config.get('data', {}).get('fips_config', {})
        fips_list = self.data_loader.resolve_fips_list(
            fips_config,
            self.config['data']['state_fips'],
            self.config['data']['county_fips']
        )
        
        self._log(f"Processing {len(fips_list)} census tracts")
        
        # Process each tract
        results = {
            'mode': 'fips',
            'tracts': {},
            'summary': {
                'total': len(fips_list),
                'successful': 0,
                'failed': 0,
                'skipped': 0
            }
        }
        
        for i, fips_code in enumerate(fips_list, 1):
            tract_start = time.time()
            
            # Progress and time estimation
            elapsed = time.time() - start_time
            eta = self._estimate_time_remaining(i-1, len(fips_list), elapsed)
            self._log(f"\n[{i}/{len(fips_list)}] Processing tract {fips_code} (ETA: {eta})")
            
            try:
                # Process single tract
                tract_result = self._process_single_fips(fips_code)
                
                if tract_result['status'] == 'success':
                    results['summary']['successful'] += 1
                    results['tracts'][fips_code] = tract_result
                    
                    # Save individual tract results
                    self._save_tract_results(fips_code, tract_result)
                    
                elif tract_result['status'] == 'skipped':
                    results['summary']['skipped'] += 1
                    self._log(f"  Skipped: {tract_result.get('reason', 'unknown')}")
                    
                else:
                    results['summary']['failed'] += 1
                    self._log(f"  Failed: {tract_result.get('error', 'unknown error')}")
                
            except Exception as e:
                results['summary']['failed'] += 1
                self._log(f"  Error processing tract: {str(e)}")
                
                if not self.config.get('processing', {}).get('continue_on_error', True):
                    break
            
            # Log tract timing
            tract_time = time.time() - tract_start
            self._log(f"  Tract processing time: {tract_time:.2f}s")
        
        # Calculate total time
        total_time = time.time() - start_time
        results['summary']['total_time'] = total_time
        results['summary']['success_rate'] = (
            results['summary']['successful'] / len(fips_list) 
            if fips_list else 0
        )
        
        # Save summary
        self._save_batch_summary(results)
        
        # Create visualization if enabled
        if self.config.get('output', {}).get('save_plots', True):
            self._create_batch_visualization(results)
        
        self._log("\n" + "="*60)
        self._log("FIPS Processing Summary:")
        self._log(f"  Total tracts: {results['summary']['total']}")
        self._log(f"  Successful: {results['summary']['successful']}")
        self._log(f"  Failed: {results['summary']['failed']}")
        self._log(f"  Skipped: {results['summary']['skipped']}")
        self._log(f"  Success rate: {results['summary']['success_rate']:.1%}")
        self._log(f"  Total time: {total_time:.2f}s")
        self._log("="*60)
        
        return results
    
    def _process_single_fips(self, fips_code: str) -> Dict:
        """Process a single census tract"""
        # Load tract data
        memory_config = self.config['data']['fips_config']['memory']
        tract_data = self.data_loader.load_single_tract_data(
            fips_code,
            buffer_degrees=memory_config['tract_buffer_degrees'],
            max_nodes=memory_config['max_network_nodes'],
            max_edges=memory_config['max_network_edges']
        )
        
        # Validate tract data
        if tract_data['road_network'].number_of_nodes() < 10:
            return {
                'status': 'skipped',
                'reason': 'insufficient_network',
                'fips_code': fips_code
            }
        
        if len(tract_data['svi_data']) == 0:
            return {
                'status': 'skipped',
                'reason': 'no_svi_data',
                'fips_code': fips_code
            }
        
        # Process tract through pipeline
        try:
            # Step 1: Train GNN for accessibility features
            self._log("  Training GNN...")
            gnn_start = time.time()
            
            graph_data, node_mapping = prepare_graph_data(tract_data['road_network'])
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
                spatial_weight=self.config['model'].get('spatial_weight', 1.0),
                reg_weight=self.config['model'].get('regularization_weight', 0.01)
            )
            
            gnn_time = time.time() - gnn_start
            self._log(f"    GNN training completed in {gnn_time:.2f}s")
            
            # Step 2: Create MetricGraph
            self._log("  Creating MetricGraph...")
            mg_start = time.time()
            
            nodes_df, edges_df = self._prepare_metricgraph_data(tract_data['road_network'])
            
            # Check if network needs sampling
            enable_sampling = (
                self.config['metricgraph']['enable_sampling'] or
                len(edges_df) > self.config['metricgraph']['max_edges']
            )
            
            metric_graph = self.mg_interface.create_graph(
                nodes_df, 
                edges_df,
                enable_sampling=enable_sampling
            )
            
            mg_create_time = time.time() - mg_start
            self._log(f"    MetricGraph created in {mg_create_time:.2f}s")
            
            # Step 3: Perform disaggregation
            self._log("  Performing Whittle-Matérn disaggregation...")
            disagg_start = time.time()
            
            # Prepare data for disaggregation
            svi_value = tract_data['svi_data']['RPL_THEMES'].iloc[0]
            
            # Use road network centroid as observation location
            node_coords = np.array([[n[0], n[1]] for n in tract_data['road_network'].nodes()])
            centroid = node_coords.mean(axis=0)
            
            observations = pd.DataFrame({
                'coord_x': [centroid[0]],  
                'coord_y': [centroid[1]],
                'svi_value': [svi_value] 
            })
            
            # Prepare prediction locations (addresses)
            prediction_locations = pd.DataFrame({
                'x': [addr.geometry.x for _, addr in tract_data['addresses'].iterrows()],
                'y': [addr.geometry.y for _, addr in tract_data['addresses'].iterrows()]
            })
            
            # Prepare GNN features
            gnn_features_df = pd.DataFrame(
                gnn_features[:len(nodes_df)],  # Ensure size match
                columns=['gnn_kappa', 'gnn_alpha', 'gnn_tau']
            )
            
            # Run disaggregation
            predictions = self.mg_interface.disaggregate_svi(
                metric_graph,
                observations,
                prediction_locations,
                gnn_features_df
            )
            
            disagg_time = time.time() - disagg_start
            self._log(f"    Disaggregation completed in {disagg_time:.2f}s")
            
            # Format results
            predictions['address_id'] = range(len(predictions))
            predictions['longitude'] = prediction_locations['x']
            predictions['latitude'] = prediction_locations['y']
            predictions['predicted_svi'] = predictions['mean']
            predictions['uncertainty'] = predictions['sd']
            
            return {
                'status': 'success',
                'fips_code': fips_code,
                'predictions': predictions,
                'network_stats': {
                    'nodes': tract_data['road_network'].number_of_nodes(),
                    'edges': tract_data['road_network'].number_of_edges()
                },
                'timing': {
                    'gnn_training': gnn_time,
                    'metricgraph_creation': mg_create_time,
                    'disaggregation': disagg_time,
                    'total': gnn_time + mg_create_time + disagg_time
                },
                'model_artifacts': {
                    'gnn_model': trained_model,
                    'gnn_features': gnn_features,
                    'training_metrics': training_metrics
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'fips_code': fips_code,
                'error': str(e)
            }
    
    def _prepare_metricgraph_data(self, road_network) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare node and edge dataframes for MetricGraph"""
        nodes = list(road_network.nodes())
        nodes_df = pd.DataFrame([
            {'node_id': i, 'x': node[0], 'y': node[1]}
            for i, node in enumerate(nodes)
        ])
        
        edges = list(road_network.edges())
        edges_df = pd.DataFrame([
            {
                'from': nodes.index(edge[0]),
                'to': nodes.index(edge[1])
            }
            for edge in edges
        ])
        
        return nodes_df, edges_df
    
    def _save_tract_results(self, fips_code: str, results: Dict):
        """Save results for individual tract"""
        if not self.config['output']['save_predictions']:
            return
        
        # Create tract directory if configured
        if self.config['output'].get('create_tract_folders', True):
            tract_dir = os.path.join(self.output_dir, f'tract_{fips_code}')
            os.makedirs(tract_dir, exist_ok=True)
        else:
            tract_dir = self.output_dir
        
        # Save predictions
        if 'predictions' in results:
            pred_file = os.path.join(tract_dir, f'predictions_{fips_code}.csv')
            results['predictions'].to_csv(pred_file, index=False)
        
        # Save GNN features if configured
        if self.config['output']['save_features'] and 'model_artifacts' in results:
            feat_file = os.path.join(tract_dir, f'gnn_features_{fips_code}.npy')
            np.save(feat_file, results['model_artifacts']['gnn_features'])
        
        # Save model if configured
        if self.config['output']['save_model'] and 'model_artifacts' in results:
            model_file = os.path.join(tract_dir, f'gnn_model_{fips_code}.pth')
            torch.save(
                results['model_artifacts']['gnn_model'].state_dict(), 
                model_file
            )
    
    def _save_batch_summary(self, results: Dict):
        """Save batch processing summary"""
        summary_file = os.path.join(self.output_dir, 'batch_summary.json')
        
        # Create summary data
        summary = {
            'processing_mode': results['mode'],
            'timestamp': datetime.now().isoformat(),
            'configuration': self.config,
            'summary_statistics': results['summary'],
            'tract_results': {}
        }
        
        # Add tract-level summaries
        for fips_code, tract_result in results['tracts'].items():
            summary['tract_results'][fips_code] = {
                'status': tract_result['status'],
                'n_predictions': len(tract_result.get('predictions', [])),
                'network_nodes': tract_result.get('network_stats', {}).get('nodes', 0),
                'network_edges': tract_result.get('network_stats', {}).get('edges', 0),
                'processing_time': tract_result.get('timing', {}).get('total', 0)
            }
        
        # Save as JSON
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also save as CSV for easy analysis
        if results['tracts']:
            summary_df = pd.DataFrame.from_dict(summary['tract_results'], orient='index')
            summary_df.index.name = 'fips_code'
            summary_df.to_csv(os.path.join(self.output_dir, 'batch_summary.csv'))
    
    def _create_batch_visualization(self, results: Dict):
        """Create summary visualization for batch processing"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('GRANITE Batch Processing Summary', fontsize=14, fontweight='bold')
        
        # Plot 1: Success rates
        ax1 = axes[0, 0]
        summary = results['summary']
        categories = ['Successful', 'Failed', 'Skipped']
        counts = [summary['successful'], summary['failed'], summary['skipped']]
        colors = ['#2ecc71', '#e74c3c', '#f39c12']
        
        ax1.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Processing Outcomes')
        
        # Plot 2: Processing times
        ax2 = axes[0, 1]
        tract_times = []
        tract_labels = []
        
        for fips_code, tract_result in results['tracts'].items():
            if 'timing' in tract_result:
                tract_times.append(tract_result['timing']['total'])
                tract_labels.append(fips_code[-4:])  # Last 4 digits
        
        if tract_times:
            ax2.bar(range(len(tract_times)), tract_times, color='#3498db')
            ax2.set_xlabel('Tract (last 4 digits)')
            ax2.set_ylabel('Processing Time (s)')
            ax2.set_title('Processing Times by Tract')
            ax2.set_xticks(range(len(tract_labels)))
            ax2.set_xticklabels(tract_labels, rotation=45)
        
        # Plot 3: Network sizes
        ax3 = axes[1, 0]
        node_counts = []
        edge_counts = []
        
        for tract_result in results['tracts'].values():
            if 'network_stats' in tract_result:
                node_counts.append(tract_result['network_stats']['nodes'])
                edge_counts.append(tract_result['network_stats']['edges'])
        
        if node_counts:
            ax3.scatter(node_counts, edge_counts, alpha=0.6, color='#9b59b6')
            ax3.set_xlabel('Number of Nodes')
            ax3.set_ylabel('Number of Edges')
            ax3.set_title('Network Complexity')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        stats_text = [
            f"Total tracts: {summary['total']}",
            f"Successful: {summary['successful']}",
            f"Failed: {summary['failed']}",
            f"Skipped: {summary['skipped']}",
            f"Success rate: {summary['success_rate']:.1%}",
            f"Total time: {summary['total_time']:.1f}s",
            f"Avg time/tract: {summary['total_time']/summary['total']:.1f}s"
        ]
        
        ax4.text(0.1, 0.9, '\n'.join(stats_text), transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace')
        ax4.axis('off')
        ax4.set_title('Summary Statistics')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'batch_summary.png'), dpi=300)
        plt.close()
    
    def _run_county_mode(self) -> Dict:
        """Run county-wide processing mode"""
        self._log("="*60)
        self._log("GRANITE: County-wide Processing Mode")
        self._log("="*60)
        
        # This mode is computationally intensive and not recommended
        # Included for completeness but simplified
        
        self._log("Warning: County-wide mode is computationally intensive.")
        self._log("Consider using FIPS mode for better performance.")
        
        # Implementation would follow similar pattern to FIPS mode
        # but process entire county at once
        
        raise NotImplementedError(
            "County-wide mode is not recommended. "
            "Please use FIPS mode (--mode fips) for efficient processing."
        )
    
    def validate_results(self, predictions: pd.DataFrame, 
                        ground_truth: Optional[pd.DataFrame] = None) -> Dict:
        """
        Validate disaggregation results
        
        Parameters:
        -----------
        predictions : pd.DataFrame
            Model predictions
        ground_truth : pd.DataFrame, optional
            Ground truth values if available
            
        Returns:
        --------
        Dict
            Validation metrics
        """
        metrics = {}
        
        # Basic statistics
        metrics['n_predictions'] = len(predictions)
        metrics['mean_svi'] = predictions['predicted_svi'].mean()
        metrics['std_svi'] = predictions['predicted_svi'].std()
        metrics['min_svi'] = predictions['predicted_svi'].min()
        metrics['max_svi'] = predictions['predicted_svi'].max()
        
        # Uncertainty statistics
        if 'uncertainty' in predictions.columns:
            metrics['mean_uncertainty'] = predictions['uncertainty'].mean()
            metrics['uncertainty_range'] = (
                predictions['uncertainty'].max() - 
                predictions['uncertainty'].min()
            )
        
        # Validation against ground truth if available
        if ground_truth is not None:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            # Merge predictions with ground truth
            merged = predictions.merge(ground_truth, on='address_id', how='inner')
            
            if len(merged) > 0:
                metrics['mae'] = mean_absolute_error(
                    merged['true_svi'], 
                    merged['predicted_svi']
                )
                metrics['rmse'] = np.sqrt(mean_squared_error(
                    merged['true_svi'], 
                    merged['predicted_svi']
                ))
                metrics['r2'] = r2_score(
                    merged['true_svi'], 
                    merged['predicted_svi']
                )
                metrics['n_validated'] = len(merged)
        
        return metrics