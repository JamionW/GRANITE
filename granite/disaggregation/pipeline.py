"""
Main disaggregation pipeline for GRANITE framework
"""
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
    
    def load_data(self, roads_file=None):
        """
        Load all required data
        
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
        """Prepare graph structures for both GNN and MetricGraph"""
        self._log("="*60)
        self._log("GRANITE: Preparing Graph Structures")
        self._log("="*60)
        
        # Prepare PyTorch Geometric data
        self._log("Preparing data for GNN...")
        self.data['pyg_data'], self.data['node_mapping'] = prepare_graph_data(
            self.data['road_network']
        )
        
        # Prepare MetricGraph structure
        self._log("Preparing data for MetricGraph...")
        
        # Extract nodes and edges for MetricGraph
        nodes_list = []
        for node, idx in self.data['node_mapping'].items():
            nodes_list.append({
                'node_id': idx,
                'x': node[0],
                'y': node[1]
            })
        
        nodes_df = pd.DataFrame(nodes_list)
        
        # Extract edges (convert to 1-indexed for R)
        edges_list = []
        for u, v in self.data['road_network'].edges():
            edges_list.append({
                'from': self.data['node_mapping'][u] + 1,  # R uses 1-indexing
                'to': self.data['node_mapping'][v] + 1
            })
        
        edges_df = pd.DataFrame(edges_list)
        
        # Create MetricGraph
        self.data['metric_graph'] = self.mg_interface.create_graph(
            nodes_df, edges_df
        )
        
        self._log("Graph structures prepared!")
        
    def learn_accessibility_features(self, epochs=100, **training_kwargs):
        """
        Train GNN to learn accessibility features
        
        Parameters:
        -----------
        epochs : int
            Number of training epochs
        **training_kwargs : dict
            Additional training parameters
        """
        self._log("="*60)
        self._log("GRANITE: Learning Accessibility Features with GNN")
        self._log("="*60)
        
        # Create GNN model
        input_dim = self.data['pyg_data'].x.shape[1]
        self.results['gnn_model'] = create_gnn_model(input_dim)
        
        # Train model
        self.results['gnn_model'], self.results['gnn_features'], self.results['training_history'] = \
            train_accessibility_gnn(
                self.data['pyg_data'],
                self.results['gnn_model'],
                epochs=epochs,
                **training_kwargs
            )
        
        self._log("GNN training complete!")
        
    def disaggregate_svi(self):
        """Perform SVI disaggregation using MetricGraph with GNN features"""
        self._log("="*60)
        self._log("GRANITE: SVI Disaggregation")
        self._log("="*60)
        
        # Prepare observations (census tract level SVI)
        tract_observations = []
        
        for idx, tract in self.data['tracts_with_svi'].iterrows():
            if pd.notna(tract['RPL_THEMES']):
                centroid = tract.geometry.centroid
                tract_observations.append({
                    'x': centroid.x,
                    'y': centroid.y,
                    'value': tract['RPL_THEMES']
                })
        
        obs_df = pd.DataFrame(tract_observations)
        self._log(f"Prepared {len(obs_df)} tract-level observations")
        
        # Prepare prediction locations (addresses)
        pred_locations = self.data['addresses'][['longitude', 'latitude']].copy()
        pred_locations.columns = ['x', 'y']
        
        # Perform disaggregation
        self.results['predictions'] = self.mg_interface.fit_with_gnn_features(
            self.data['metric_graph'],
            obs_df,
            self.results['gnn_features'],
            pred_locations,
            alpha=1.5
        )
        
        # Add predictions to addresses
        self.data['addresses']['svi_predicted'] = self.results['predictions']['mean']
        self.data['addresses']['svi_sd'] = self.results['predictions']['sd']
        self.data['addresses']['svi_lower_95'] = self.results['predictions']['lower_95']
        self.data['addresses']['svi_upper_95'] = self.results['predictions']['upper_95']
        
        self._log("SVI disaggregation complete!")
        self._log(f"  - Predicted SVI for {len(self.data['addresses'])} addresses")
        self._log(f"  - Mean SVI: {self.data['addresses']['svi_predicted'].mean():.3f}")
        self._log(f"  - Average uncertainty (SD): {self.data['addresses']['svi_sd'].mean():.3f}")
        
    def validate_results(self):
        """Validate disaggregation results"""
        self._log("="*60)
        self._log("GRANITE: Validating Results")
        self._log("="*60)
        
        # Check mass preservation
        # The average of disaggregated values should approximate tract averages
        
        # For each tract, find addresses within it
        validation_results = []
        
        for idx, tract in self.data['tracts_with_svi'].iterrows():
            if pd.notna(tract['RPL_THEMES']):
                # Find addresses in this tract
                addresses_in_tract = self.data['addresses'][
                    self.data['addresses'].geometry.within(tract.geometry)
                ]
                
                if len(addresses_in_tract) > 0:
                    tract_avg_predicted = addresses_in_tract['svi_predicted'].mean()
                    tract_true = tract['RPL_THEMES']
                    
                    validation_results.append({
                        'tract_id': tract['FIPS'],
                        'true_svi': tract_true,
                        'predicted_avg': tract_avg_predicted,
                        'n_addresses': len(addresses_in_tract),
                        'error': abs(tract_true - tract_avg_predicted)
                    })
        
        validation_df = pd.DataFrame(validation_results)
        
        if len(validation_df) > 0:
            self._log(f"Validation results for {len(validation_df)} tracts:")
            self._log(f"  - Mean absolute error: {validation_df['error'].mean():.4f}")
            self._log(f"  - Max absolute error: {validation_df['error'].max():.4f}")
            self._log(f"  - Correlation: {validation_df['true_svi'].corr(validation_df['predicted_avg']):.3f}")
            
            self.results['validation'] = validation_df
        
    def save_results(self):
        """Save all results"""
        self._log("="*60)
        self._log("GRANITE: Saving Results")
        self._log("="*60)
        
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save predictions
        output_file = os.path.join(self.output_dir, 'granite_predictions.csv')
        self.data['addresses'].to_csv(output_file, index=False)
        self._log(f"  ✓ Saved predictions to {output_file}")
        
        # Save GNN features
        features_file = os.path.join(self.output_dir, 'gnn_features.npy')
        np.save(features_file, self.results['gnn_features'])
        self._log(f"  ✓ Saved GNN features to {features_file}")
        
        # Save validation results
        if 'validation' in self.results:
            validation_file = os.path.join(self.output_dir, 'validation_results.csv')
            self.results['validation'].to_csv(validation_file, index=False)
            self._log(f"  ✓ Saved validation to {validation_file}")
        
    def run(self, roads_file=None, epochs=100, visualize=True):
        """
        Run complete GRANITE pipeline
        
        Parameters:
        -----------
        roads_file : str, optional
            Path to roads shapefile
        epochs : int
            Number of GNN training epochs
        visualize : bool
            Whether to create visualizations
        """
        self._log("="*70)
        self._log("GRANITE: Graph-Refined Accessibility Network for Integrated Transit Equity")
        self._log("Starting SVI Disaggregation Pipeline")
        self._log("="*70 + "\n")
        
        # Step 1: Load data
        self.load_data(roads_file)
        
        # Step 2: Prepare graph structures
        self.prepare_graph_structures()
        
        # Step 3: Learn accessibility features
        self.learn_accessibility_features(epochs=epochs)
        
        # Step 4: Disaggregate SVI
        self.disaggregate_svi()
        
        # Step 5: Validate results
        self.validate_results()
        
        # Step 6: Save results
        self.save_results()
        
        # Step 7: Visualize (if requested)
        if visualize:
            self._log("\nCreating visualizations...")
            fig = self.visualizer.create_summary_plot(
                self.data['tracts_with_svi'],
                self.data['addresses'],
                self.results['gnn_features'],
                self.results.get('validation')
            )
            
            import os
            viz_file = os.path.join(self.output_dir, 'granite_visualization.png')
            fig.savefig(viz_file, dpi=300, bbox_inches='tight')
            self._log(f"  ✓ Saved visualization to {viz_file}")
        
        self._log("\n" + "="*70)
        self._log("GRANITE Pipeline Complete!")
        self._log("="*70)
        
        return self.results


# Convenience function
def run_granite_pipeline(roads_file=None, **kwargs):
    """
    Run GRANITE pipeline with default settings
    
    Parameters:
    -----------
    roads_file : str, optional
        Path to roads shapefile
    **kwargs : dict
        Additional pipeline parameters
        
    Returns:
    --------
    dict
        Pipeline results
    """
    pipeline = GRANITEPipeline()
    results = pipeline.run(roads_file=roads_file, **kwargs)
    return results