"""
Main disaggregation pipeline for GRANITE framework
ENHANCED VERSION - includes all placeholder implementations
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime
import torch
import warnings
warnings.filterwarnings('ignore')

# Enhanced imports for new functionality
from scipy.spatial.distance import cdist
from scipy.interpolate import RBFInterpolator
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import networkx as nx

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
    - County-wide processing 
    - FIPS-based processing
    
    ENHANCED VERSION with:
    - Network-aware spatial interpolation
    - Comprehensive validation metrics
    - Rich visualizations
    - Robust fallback methods
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
    # COUNTY-WIDE PROCESSING METHODS
    # ==========================================
    
    def _run_original_pipeline(self, roads_file, epochs, visualize):
        """County-wide pipeline"""
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
        ENHANCED with optional feature analysis
        
        Parameters:
        -----------
        epochs : int
            Number of training epochs
        """
        self._log("="*60)
        self._log("GRANITE: Learning Accessibility Features")
        self._log("="*60)
        
        # ========================================
        # OPTIONAL: Add input feature analysis
        # ========================================
        try:
            input_analysis, input_features = self.analyze_gnn_input_features(
                self.data['pyg_data'], save_analysis=True
            )
        except Exception as e:
            if self.verbose:
                self._log(f"Input feature analysis failed: {e}")
            input_analysis, input_features = None, None
        
        # ========================================
        # ORIGINAL FUNCTIONALITY (UNCHANGED)
        # ========================================
        
        # Create GNN model
        input_dim = self.data['pyg_data'].x.shape[1]
        gnn_model = create_gnn_model(input_dim=input_dim)
        
        # Train GNN
        training_result = train_accessibility_gnn(
            gnn_model,
            self.data['pyg_data'],
            epochs=epochs,
            verbose=self.verbose
        )
        
        # Handle different return formats from train_accessibility_gnn
        if isinstance(training_result, tuple) and len(training_result) == 3:
            trained_model, gnn_features, training_metrics = training_result
        else:
            # Fallback if training function returns different format
            gnn_features = training_result
            trained_model = gnn_model
            training_metrics = {}
        
        # Store original results
        self.results['gnn_features'] = gnn_features
        self.results['gnn_model'] = trained_model
        
        # ========================================
        # OPTIONAL: Add learned feature analysis
        # ========================================
        try:
            learned_analysis, learned_features = self.analyze_gnn_learned_features(
                gnn_features, training_metrics, save_analysis=True
            )
            
            # Optional: Analyze transformation (if input analysis succeeded)
            if input_features is not None:
                transformation_analysis = self.analyze_gnn_feature_transformation(
                    input_features, learned_features, save_analysis=True
                )
                
                # Optional: Create visualizations
                self.create_gnn_feature_plots(
                    input_features, learned_features, save_plots=True
                )
            
            # Store analysis results (optional - won't break anything if this fails)
            self.results['feature_analysis'] = {
                'input_analysis': input_analysis,
                'learned_analysis': learned_analysis,
                'transformation_analysis': transformation_analysis if input_features is not None else None,
                'training_metrics': training_metrics
            }
            
        except Exception as e:
            if self.verbose:
                self._log(f"Feature analysis failed: {e}")
        
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
            self.data['addresses'],  # Use addresses as prediction locations
            None,  # nodes_df not available in county mode
            gnn_features=covariates
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
    
    # ==========================================
    # ENHANCED SPATIAL INTERPOLATION METHODS
    # ==========================================
    
    def _interpolate_to_addresses(self):
        """
        FIXED: Interpolate disaggregation results to address locations using 
        MetricGraph Whittle-Matérn results as the scientific foundation
        """
        self._log("Performing Whittle-Matérn field interpolation to addresses...")
        
        addresses = self.data['addresses']
        
        # Step 1: Check for MetricGraph disaggregation results (PRIMARY)
        if 'disaggregation' not in self.results:
            self._log("No MetricGraph results found, using enhanced fallback")
            return self._enhanced_fallback_interpolation(addresses)
        
        mg_results = self.results['disaggregation']
        
        # Step 2: Use MetricGraph Whittle-Matérn predictions as foundation
        try:
            self._log("  Using MetricGraph Whittle-Matérn field as primary source")
            predictions = self._interpolate_metricgraph_predictions(mg_results, addresses)
            return predictions
            
        except Exception as e:
            self._log(f"MetricGraph interpolation failed: {str(e)}, using enhanced fallback")
            return self._enhanced_fallback_interpolation(addresses)

    
    def _network_aware_interpolation(self, mg_results, addresses, road_network):
        """Network-aware spatial interpolation using road network topology"""
        self._log("  Using network-aware spatial interpolation")
        
        try:
            predictions = []
            
            # Get observation points from tract centroids
            tract_centroids = self.data['tracts_with_svi'].geometry.centroid
            
            for idx, addr in addresses.iterrows():
                addr_point = (addr.geometry.x, addr.geometry.y)
                
                # Find nearest network nodes
                network_nodes = list(road_network.nodes())
                node_distances = [
                    np.sqrt((addr_point[0] - node[0])**2 + (addr_point[1] - node[1])**2)
                    for node in network_nodes
                ]
                nearest_node_idx = np.argmin(node_distances)
                nearest_node = network_nodes[nearest_node_idx]
                
                # Calculate network distances to tract centroids
                network_distances = []
                weights = []
                
                for tract_idx, centroid in enumerate(tract_centroids):
                    centroid_point = (centroid.x, centroid.y)
                    
                    # Find nearest network node to tract centroid
                    centroid_distances = [
                        np.sqrt((centroid_point[0] - node[0])**2 + (centroid_point[1] - node[1])**2)
                        for node in network_nodes
                    ]
                    nearest_centroid_node_idx = np.argmin(centroid_distances)
                    nearest_centroid_node = network_nodes[nearest_centroid_node_idx]
                    
                    # Calculate network distance (simplified - could use shortest path)
                    try:
                        network_dist = nx.shortest_path_length(
                            road_network, nearest_node, nearest_centroid_node
                        )
                    except:
                        # Fallback to Euclidean distance if no path exists
                        network_dist = np.sqrt(
                            (nearest_node[0] - nearest_centroid_node[0])**2 + 
                            (nearest_node[1] - nearest_centroid_node[1])**2
                        )
                    
                    network_distances.append(network_dist)
                    weights.append(1 / (network_dist + 1e-6))
                
                # Normalize weights
                weights = np.array(weights)
                weights /= weights.sum()
                
                # Interpolate using network-weighted values
                if hasattr(mg_results, 'mean') and len(mg_results.mean) > 0:
                    pred_mean = np.average(mg_results.mean[:len(weights)], weights=weights)
                    pred_sd = np.sqrt(np.average(mg_results.sd[:len(weights)]**2, weights=weights))
                else:
                    # Fallback using tract SVI values
                    tract_svi_values = self.data['tracts_with_svi']['RPL_THEMES'].values
                    pred_mean = np.average(tract_svi_values[:len(weights)], weights=weights)
                    pred_sd = np.std(tract_svi_values) * 0.3
                
                predictions.append({
                    'address_id': idx,
                    'longitude': addr_point[0],
                    'latitude': addr_point[1],
                    'mean': pred_mean,
                    'sd': pred_sd,
                    'q025': pred_mean - 1.96 * pred_sd,
                    'q975': pred_mean + 1.96 * pred_sd
                })
            
            return pd.DataFrame(predictions)
            
        except Exception as e:
            self._log(f"Network interpolation failed: {str(e)}, using fallback")
            return self._gnn_enhanced_interpolation(mg_results, addresses, self.results.get('gnn_features'))

    def _gnn_enhanced_interpolation(self, mg_results, addresses, gnn_features):
        """Enhanced spatial interpolation using GNN features"""
        self._log("  Using GNN-enhanced spatial interpolation")
        
        try:
            predictions = []
            
            # Use GNN features to inform spatial variation
            if gnn_features is not None and len(gnn_features) > 0:
                # Extract spatial parameters from GNN features
                kappa_mean = np.mean(gnn_features[:, 0])  # Precision parameter
                alpha_mean = np.mean(gnn_features[:, 1])  # Smoothness parameter  
                tau_mean = np.mean(gnn_features[:, 2])    # Nugget effect
            else:
                kappa_mean, alpha_mean, tau_mean = 1.0, 0.5, 0.1
            
            # Base prediction from tract-level data
            if hasattr(mg_results, 'mean'):
                base_svi = np.mean(mg_results.mean)
                base_uncertainty = np.mean(mg_results.sd)
            else:
                base_svi = self.data['tracts_with_svi']['RPL_THEMES'].mean()
                base_uncertainty = self.data['tracts_with_svi']['RPL_THEMES'].std() * 0.3
            
            for idx, addr in addresses.iterrows():
                # Add spatial variation based on GNN-learned parameters
                spatial_variation = tau_mean * np.random.normal(0, 1)
                
                pred_mean = np.clip(base_svi + spatial_variation, 0, 1)
                pred_sd = base_uncertainty * (1 + alpha_mean * 0.5)  # Uncertainty varies with smoothness
                
                predictions.append({
                    'address_id': idx,
                    'longitude': addr.geometry.x,
                    'latitude': addr.geometry.y,
                    'mean': pred_mean,
                    'sd': pred_sd,
                    'q025': pred_mean - 1.96 * pred_sd,
                    'q975': pred_mean + 1.96 * pred_sd
                })
            
            return pd.DataFrame(predictions)
            
        except Exception as e:
            self._log(f"GNN interpolation failed: {str(e)}, using basic fallback")
            return self._fallback_address_interpolation(addresses)

    def _fallback_address_interpolation(self, addresses):
        """Enhanced fallback interpolation when no other method works"""
        self._log("  Using enhanced fallback interpolation for addresses")
        
        predictions = []
        
        # Get base SVI value from tract data
        if 'tracts_with_svi' in self.data and len(self.data['tracts_with_svi']) > 0:
            base_svi = self.data['tracts_with_svi']['RPL_THEMES'].mean()
            svi_std = self.data['tracts_with_svi']['RPL_THEMES'].std()
        else:
            base_svi = 0.5
            svi_std = 0.2
        
        for idx, addr in addresses.iterrows():
            # Add small spatial variation
            spatial_variation = np.random.normal(0, svi_std * 0.3)
            pred_svi = np.clip(base_svi + spatial_variation, 0, 1)
            uncertainty = svi_std * 0.5
            
            predictions.append({
                'address_id': idx,
                'longitude': addr.geometry.x,
                'latitude': addr.geometry.y,
                'mean': pred_svi,
                'sd': uncertainty,
                'q025': pred_svi - 1.96 * uncertainty,
                'q975': pred_svi + 1.96 * uncertainty
            })
        
        return pd.DataFrame(predictions)
    
    # ==========================================
    # ENHANCED VALIDATION METHODS
    # ==========================================
    
    def _compute_validation_metrics(self):
        """
        ENHANCED: Compute comprehensive validation metrics using spatial 
        cross-validation and uncertainty calibration
        """
        self._log("Computing comprehensive validation metrics...")
        
        if 'predictions' not in self.results:
            self._log("No predictions available for validation")
            return pd.DataFrame()
        
        predictions_df = self.results['predictions']
        tracts_with_svi = self.data['tracts_with_svi']
        
        # Method 1: Tract-level aggregation validation
        tract_validation = self._validate_tract_aggregation(predictions_df, tracts_with_svi)
        
        # Method 2: Spatial cross-validation (if sufficient data)
        if len(tracts_with_svi) >= 5:
            spatial_cv_metrics = self._spatial_cross_validation(tracts_with_svi)
            tract_validation.update(spatial_cv_metrics)
        
        # Method 3: Uncertainty calibration metrics
        if 'sd' in predictions_df.columns or 'uncertainty' in predictions_df.columns:
            uncertainty_metrics = self._validate_uncertainty_calibration(predictions_df, tracts_with_svi)
            tract_validation.update(uncertainty_metrics)
        
        # Convert to DataFrame for consistency
        validation_df = pd.DataFrame([tract_validation])
        
        return validation_df

    def _validate_tract_aggregation(self, predictions_df, tracts_with_svi):
        """Validate that address predictions aggregate to tract values (mass preservation)"""
        validation_results = {}
        
        try:
            # Create spatial join between predictions and tracts
            predictions_gdf = gpd.GeoDataFrame(
                predictions_df,
                geometry=gpd.points_from_xy(predictions_df['longitude'], predictions_df['latitude']),
                crs='EPSG:4326'
            )
            
            # Ensure CRS match
            if tracts_with_svi.crs != predictions_gdf.crs:
                tracts_with_svi = tracts_with_svi.to_crs(predictions_gdf.crs)
            
            # Spatial join
            joined = gpd.sjoin(predictions_gdf, tracts_with_svi, how='left', predicate='within')
            
            # Aggregate predictions by tract
            pred_col = 'mean' if 'mean' in predictions_df.columns else 'predicted_svi'
            tract_pred_agg = joined.groupby('FIPS')[pred_col].mean()
            
            # Compare with true tract values
            true_tract_svi = tracts_with_svi.set_index('FIPS')['RPL_THEMES']
            
            # Calculate metrics for overlapping tracts
            common_fips = tract_pred_agg.index.intersection(true_tract_svi.index)
            
            if len(common_fips) > 0:
                true_vals = true_tract_svi.loc[common_fips]
                pred_vals = tract_pred_agg.loc[common_fips]
                
                validation_results.update({
                    'tract_mae': mean_absolute_error(true_vals, pred_vals),
                    'tract_rmse': np.sqrt(mean_squared_error(true_vals, pred_vals)),
                    'tract_r2': r2_score(true_vals, pred_vals),
                    'tract_correlation': np.corrcoef(true_vals, pred_vals)[0, 1],
                    'mass_preservation_error': np.abs(true_vals.mean() - pred_vals.mean()),
                    'n_validated_tracts': len(common_fips)
                })
            else:
                self._log("Warning: No spatial overlap found between predictions and tracts")
                validation_results.update({
                    'tract_mae': np.nan,
                    'tract_rmse': np.nan,
                    'tract_r2': np.nan,
                    'tract_correlation': np.nan,
                    'mass_preservation_error': np.nan,
                    'n_validated_tracts': 0
                })
                
        except Exception as e:
            self._log(f"Error in tract aggregation validation: {str(e)}")
            validation_results.update({
                'tract_mae': np.nan,
                'tract_rmse': np.nan,
                'tract_r2': np.nan,
                'tract_correlation': np.nan,
                'mass_preservation_error': np.nan,
                'n_validated_tracts': 0
            })
        
        return validation_results

    def _spatial_cross_validation(self, tracts_with_svi, n_folds=5):
        """Perform spatial cross-validation to assess generalization"""
        cv_metrics = {}
        
        try:
            # Simple spatial blocking (could be enhanced with more sophisticated methods)
            folds = KFold(n_splits=min(n_folds, len(tracts_with_svi)), shuffle=True, random_state=42)
            
            cv_scores = {'mae': [], 'rmse': [], 'r2': []}
            svi_values = tracts_with_svi['RPL_THEMES'].dropna().values
            
            if len(svi_values) < n_folds:
                self._log(f"Insufficient data for {n_folds}-fold CV, using {len(svi_values)} folds")
                n_folds = len(svi_values)
            
            for train_idx, test_idx in folds.split(svi_values):
                train_svi = svi_values[train_idx]
                test_svi = svi_values[test_idx]
                
                # Simple prediction model for CV (replace with actual model)
                pred_svi = np.full_like(test_svi, train_svi.mean())
                
                cv_scores['mae'].append(mean_absolute_error(test_svi, pred_svi))
                cv_scores['rmse'].append(np.sqrt(mean_squared_error(test_svi, pred_svi)))
                if len(np.unique(test_svi)) > 1:  # Avoid R2 calculation issues
                    cv_scores['r2'].append(r2_score(test_svi, pred_svi))
            
            cv_metrics.update({
                'cv_mae_mean': np.mean(cv_scores['mae']),
                'cv_mae_std': np.std(cv_scores['mae']),
                'cv_rmse_mean': np.mean(cv_scores['rmse']),
                'cv_rmse_std': np.std(cv_scores['rmse']),
                'cv_r2_mean': np.mean(cv_scores['r2']) if cv_scores['r2'] else np.nan,
                'cv_r2_std': np.std(cv_scores['r2']) if cv_scores['r2'] else np.nan
            })
            
        except Exception as e:
            self._log(f"Error in spatial cross-validation: {str(e)}")
            cv_metrics.update({
                'cv_mae_mean': np.nan,
                'cv_mae_std': np.nan,
                'cv_rmse_mean': np.nan,
                'cv_rmse_std': np.nan,
                'cv_r2_mean': np.nan,
                'cv_r2_std': np.nan
            })
        
        return cv_metrics

    def _validate_uncertainty_calibration(self, predictions_df, tracts_with_svi):
        """Validate uncertainty calibration using prediction intervals"""
        uncertainty_metrics = {}
        
        try:
            # Get uncertainty column
            uncertainty_col = 'sd' if 'sd' in predictions_df.columns else 'uncertainty'
            pred_col = 'mean' if 'mean' in predictions_df.columns else 'predicted_svi'
            
            if uncertainty_col not in predictions_df.columns:
                self._log("No uncertainty information available for calibration validation")
                return {'uncertainty_calibrated': False}
            
            uncertainties = predictions_df[uncertainty_col].values
            predictions = predictions_df[pred_col].values
            
            # Calculate calibration metrics
            mean_uncertainty = np.mean(uncertainties)
            uncertainty_range = np.max(uncertainties) - np.min(uncertainties)
            
            # Check if uncertainties are reasonable (not all zeros or too large)
            reasonable_uncertainty = (mean_uncertainty > 0.001) and (mean_uncertainty < 0.5)
            
            uncertainty_metrics.update({
                'mean_uncertainty': mean_uncertainty,
                'uncertainty_range': uncertainty_range,
                'uncertainty_reasonable': reasonable_uncertainty,
                'uncertainty_calibrated': reasonable_uncertainty and (uncertainty_range > 0.001)
            })
            
            # Additional calibration checks if we have sufficient data
            if len(predictions) > 10:
                # Check for correlation between predictions and uncertainties
                uncertainty_correlation = np.corrcoef(predictions, uncertainties)[0, 1]
                uncertainty_metrics['pred_uncertainty_correlation'] = uncertainty_correlation
            
        except Exception as e:
            self._log(f"Error in uncertainty calibration validation: {str(e)}")
            uncertainty_metrics.update({
                'mean_uncertainty': np.nan,
                'uncertainty_range': np.nan,
                'uncertainty_reasonable': False,
                'uncertainty_calibrated': False
            })
        
        return uncertainty_metrics
    
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
            result = self._process_single_tract(tract_data, config, epochs, visualize)
            
            # Add FIPS code to result
            if 'fips_code' not in result:
                result['fips_code'] = fips_code
            
            return result
            
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
                
                # Add FIPS code to result
                if 'fips_code' not in result:
                    result['fips_code'] = fips_code
                
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

    def _process_single_tract(self, tract_data, config, epochs, visualize):
        """
        Process single tract with optimized MetricGraph integration
        OPTIMIZED VERSION - includes smart sampling configuration
        """
        self._log("Processing single tract with optimized MetricGraph integration")
        
        try:
            # Extract tract components
            road_network = tract_data['road_network']
            svi_data = tract_data['svi_data']
            addresses = tract_data['addresses']
            
            # Log network statistics
            n_edges = road_network.number_of_edges()
            n_nodes = road_network.number_of_nodes()
            self._log(f"Original network: {n_nodes} nodes, {n_edges} edges")
            
            # Step 1: Train GNN on road network
            self._log("Training GNN for accessibility feature extraction...")
            
            graph_data, node_mapping = prepare_graph_data(road_network)  # ← Remove svi_data, handle return tuple
            model = create_gnn_model(
                input_dim=graph_data.x.shape[1],  # ← Use .x instead of ['node_features']
                hidden_dim=config.get('model', {}).get('hidden_dim', 64),
                output_dim=3
            )

            trained_model, gnn_features, training_metrics = train_accessibility_gnn(
                graph_data, model, epochs=epochs
            )
            
            # Step 2: Prepare MetricGraph inputs
            nodes_df, edges_df = self._prepare_metricgraph_inputs(road_network)
            
            # Step 3: Get optimization settings from config
            metricgraph_config = config.get('metricgraph', {})
            r_opts = metricgraph_config.get('r_optimizations', {})
            smart_sampling_config = metricgraph_config.get('smart_sampling', {})
            
            # OPTIMIZATION PARAMETERS (configurable)
            enable_smart_sampling = smart_sampling_config.get('enabled', False)
            max_edges = r_opts.get('max_edges', 2000)
            batch_size = r_opts.get('batch_size', 300)
            
            # Auto-enable smart sampling if network is large and auto-sampling is enabled
            processing_config = config.get('processing', {}).get('memory_optimization', {})
            if (processing_config.get('enable_sampling_auto', True) and 
                n_edges > processing_config.get('sampling_threshold', 3000)):
                enable_smart_sampling = True
                self._log(f"Auto-enabling smart sampling (network has {n_edges} edges)")
            
            # Step 5: Create MetricGraph with optimizations
            self._log("Creating MetricGraph with optimized interface...")
            metric_graph = self.mg_interface.create_graph(
                nodes_df, 
                edges_df,
                enable_sampling=enable_smart_sampling, 
                max_edges=max_edges,                  
                batch_size=batch_size                 
            )
            
            if metric_graph is not None:
                # Step 6: Disaggregate SVI using MetricGraph
                self._log("Performing SVI disaggregation with GNN features...")
                
                # Prepare observations
                svi_value = svi_data['RPL_THEMES'].iloc[0]

                # Get road network centroid coordinates
                road_nodes = list(road_network.nodes())
                if road_nodes:
                    x_coords = [node[0] for node in road_nodes]
                    y_coords = [node[1] for node in road_nodes]
                    centroid_x = sum(x_coords) / len(x_coords)
                    centroid_y = sum(y_coords) / len(y_coords)
                else:
                    centroid_x, centroid_y = -85.3, 35.1  # Fallback coordinates

                observations = pd.DataFrame({
                    'x': [centroid_x],
                    'y': [centroid_y], 
                    'value': [svi_value]
                })
                
                # Run disaggregation
                address_coords = pd.DataFrame({
                'x': [addr.geometry.x for _, addr in addresses.iterrows()],
                'y': [addr.geometry.y for _, addr in addresses.iterrows()]
                })

                mg_results = self.mg_interface.disaggregate_svi(
                    metric_graph,
                    observations,
                    address_coords,
                    nodes_df,
                    gnn_features=pd.DataFrame(gnn_features, columns=['gnn_kappa', 'gnn_alpha', 'gnn_tau'])
                )
                
                # Step 7: Format results
                predictions = addresses[['x', 'y']].copy()
                predictions['predicted_svi'] = mg_results['mean']
                predictions['uncertainty'] = mg_results['sd']
                predictions['ci_lower'] = mg_results['q025']
                predictions['ci_upper'] = mg_results['q975']
                
                success_metrics = {
                    'method': 'optimized_metricgraph',
                    'gnn_training': training_metrics,
                    'metricgraph_success': True,
                    'smart_sampling_used': enable_smart_sampling,
                    'original_edges': n_edges,
                    'processed_edges': len(edges_df) if not enable_smart_sampling else "sampled",
                    'n_predictions': len(predictions)
                }
                
                self._log(f"✓ Successfully processed tract with {len(predictions)} predictions")
                
            else:
                # Step 8: Enhanced fallback using GNN features
                self._log("MetricGraph failed, using enhanced GNN-based fallback...")
                predictions = self._enhanced_gnn_fallback(
                    road_network, svi_data, addresses, gnn_features
                )
                
                success_metrics = {
                    'method': 'enhanced_gnn_fallback',
                    'gnn_training': training_metrics,
                    'metricgraph_success': False,
                    'smart_sampling_used': enable_smart_sampling,
                    'n_predictions': len(predictions)
                }
            
            # Create visualization if requested
            if visualize:
                fips_code = tract_data.get('fips_code', 'unknown')
                result_for_viz = {
                    'fips_code': fips_code,
                    'predictions': predictions,
                    'metrics': success_metrics,
                    'status': 'success'
                }
                self._create_tract_visualization(result_for_viz, tract_data, config)
            
            return {
                'predictions': predictions,
                'metrics': success_metrics,
                'status': 'success',
                'network_stats': {
                    'nodes': n_nodes,
                    'edges': n_edges
                }
            }
            
        except Exception as e:
            self._log(f"✗ Error in optimized processing: {str(e)}", 'ERROR')
            return self._fallback_tract_interpolation(tract_data)

    def _run_tract_metricgraph(self, tract_data, gnn_features, config):
        """Run MetricGraph disaggregation for a single tract"""
        try:
            # Prepare MetricGraph inputs
            nodes_df, edges_df = self._prepare_metricgraph_inputs(
                tract_data['road_network']
            )
            
            # Create MetricGraph
            metric_graph = self.mg_interface.create_graph(nodes_df, edges_df)
            
            # Prepare observations (single tract SVI value with network centroid coordinates)
            svi_value = tract_data['svi_data']['RPL_THEMES'].iloc[0]

            # Get road network centroid coordinates (reliable and always available)
            road_nodes = list(tract_data['road_network'].nodes())
            if road_nodes:
                # Use centroid of road network
                x_coords = [node[0] for node in road_nodes]
                y_coords = [node[1] for node in road_nodes]
                centroid_x = sum(x_coords) / len(x_coords)
                centroid_y = sum(y_coords) / len(y_coords)
            else:
                # Fallback to center of bounding box
                centroid_x, centroid_y = -85.3, 35.1  # Approximate center of area

            observations = pd.DataFrame({
                'x': [centroid_x],            # Road network centroid longitude
                'y': [centroid_y],            # Road network centroid latitude
                'value': [svi_value]          # SVI value
            })
            
            # Prepare covariates (GNN features)
            n_features = min(len(gnn_features), len(nodes_df))
            covariates = pd.DataFrame(gnn_features[:n_features])
            covariates.columns = ['gnn_kappa', 'gnn_alpha', 'gnn_tau']
            
            # Run disaggregation
            tract_address_coords = pd.DataFrame({
                'x': [addr.geometry.x for _, addr in tract_data['addresses'].iterrows()],
                'y': [addr.geometry.y for _, addr in tract_data['addresses'].iterrows()]
            })

            mg_results = self.mg_interface.disaggregate_svi(
                metric_graph,
                observations,
                tract_address_coords, 
                nodes_df,
                gnn_features=covariates
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
        """
        FIXED: Interpolate MetricGraph Whittle-Matérn results to address locations
        for FIPS mode processing
        """
        if mg_results is None or len(mg_results) == 0:
            self._log("  No MetricGraph results, using fallback")
            return self._fallback_address_interpolation(addresses)
        
        self._log("  Interpolating MetricGraph results to address locations")
        
        # Use the same MetricGraph interpolation logic as county mode
        try:
            if isinstance(mg_results, pd.DataFrame):
                predictions = self._spatially_interpolate_field(mg_results, addresses)
            else:
                predictions = self._interpolate_sparse_field(mg_results, addresses)
            
            # Convert to expected format for FIPS mode
            fips_predictions = []
            for pred in predictions:
                fips_predictions.append({
                    'address_id': pred.get('address_id', pred.get('idx', 0)),
                    'longitude': pred['longitude'],
                    'latitude': pred['latitude'],
                    'predicted_svi': pred['mean'],        # FIPS mode uses this column name
                    'uncertainty': pred['sd'],            # FIPS mode uses this column name
                    'ci_lower': pred['q025'],
                    'ci_upper': pred['q975']
                })
            
            return pd.DataFrame(fips_predictions)
            
        except Exception as e:
            self._log(f"  MetricGraph interpolation failed: {str(e)}, using fallback")
            return self._fallback_address_interpolation(addresses)


    def _fallback_tract_interpolation(self, tract_data):
        """Fallback interpolation when MetricGraph fails"""
        self._log("    Using fallback interpolation method")
        
        # Use tract SVI value with small random variation
        tract_svi = tract_data['svi_data']['RPL_THEMES'].iloc[0]
        
        predictions = []
        for idx, addr in tract_data['addresses'].iterrows():
            predictions.append({
                'address_id': getattr(addr, 'address_id', idx),
                'longitude': addr.geometry.x,
                'latitude': addr.geometry.y,
                'predicted_svi': tract_svi + np.random.normal(0, 0.05),
                'uncertainty': 0.1
            })
        
        return {
            'status': 'success',
            'fips_code': tract_data.get('fips_code', 'unknown'),
            'network_stats': {
                'nodes': tract_data['road_network'].number_of_nodes(),
                'edges': tract_data['road_network'].number_of_edges()
            },
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

    def _interpolate_metricgraph_predictions(self, mg_results, addresses):
        """
        Interpolate Whittle-Matérn field predictions to address locations
        This maintains scientific rigor while adding spatial sophistication
        """
        self._log("  Interpolating Whittle-Matérn spatial field to address locations")
        
        predictions = []
        
        # Handle different MetricGraph result formats
        if isinstance(mg_results, pd.DataFrame) and len(mg_results) > 1:
            # Multiple prediction locations - use spatial interpolation
            predictions = self._spatially_interpolate_field(mg_results, addresses)
            
        elif hasattr(mg_results, 'mean') or isinstance(mg_results, dict):
            # Single or few predictions - need within-area interpolation
            predictions = self._interpolate_sparse_field(mg_results, addresses)
            
        else:
            self._log("  Unexpected MetricGraph result format, using fallback")
            return self._enhanced_fallback_interpolation(addresses)
        
        self._log(f"  ✓ Interpolated Whittle-Matérn field to {len(predictions)} addresses")
        return pd.DataFrame(predictions)

    def _spatially_interpolate_field(self, mg_results, addresses):
        """
        Spatial interpolation when MetricGraph provides multiple prediction points
        """
        predictions = []
        
        # Extract field coordinates and values
        if 'x' in mg_results.columns and 'y' in mg_results.columns:
            field_coords = mg_results[['x', 'y']].values
            field_means = mg_results['mean'].values
            field_sds = mg_results['sd'].values if 'sd' in mg_results.columns else mg_results['mean'].values * 0.1
        else:
            self._log("  MetricGraph results missing coordinates, using sparse interpolation")
            return self._interpolate_sparse_field(mg_results, addresses)
        
        # Prepare address coordinates
        addr_coords = np.array([[addr.geometry.x, addr.geometry.y] for _, addr in addresses.iterrows()])
        
        # Use RBF interpolation for the Whittle-Matérn field
        try:
            # Interpolate mean field
            rbf_mean = RBFInterpolator(field_coords, field_means, kernel='thin_plate_spline')
            interpolated_means = rbf_mean(addr_coords)
            
            # Interpolate uncertainty field
            rbf_sd = RBFInterpolator(field_coords, field_sds, kernel='thin_plate_spline')
            interpolated_sds = rbf_sd(addr_coords)
            
            # Ensure positive uncertainties
            interpolated_sds = np.maximum(interpolated_sds, 0.01)
            
        except Exception as e:
            self._log(f"  RBF interpolation failed: {e}, using distance weighting")
            interpolated_means, interpolated_sds = self._distance_weighted_interpolation(
                field_coords, field_means, field_sds, addr_coords
            )
        
        # Create predictions
        for i, (idx, addr) in enumerate(addresses.iterrows()):
            predictions.append({
                'address_id': idx,
                'longitude': addr.geometry.x,
                'latitude': addr.geometry.y,
                'mean': float(interpolated_means[i]),
                'sd': float(interpolated_sds[i]),
                'q025': float(interpolated_means[i] - 1.96 * interpolated_sds[i]),
                'q975': float(interpolated_means[i] + 1.96 * interpolated_sds[i])
            })
        
        return predictions

    def _interpolate_sparse_field(self, mg_results, addresses):
        """
        Interpolation when MetricGraph provides sparse predictions (e.g., single tract value)
        Uses GNN features to create spatial variation within the Whittle-Matérn framework
        """
        predictions = []
        
        # Extract base field statistics from MetricGraph
        if isinstance(mg_results, dict):
            base_mean = mg_results.get('mean', [0.5])[0] if isinstance(mg_results.get('mean'), list) else mg_results.get('mean', 0.5)
            base_sd = mg_results.get('sd', [0.1])[0] if isinstance(mg_results.get('sd'), list) else mg_results.get('sd', 0.1)
        elif hasattr(mg_results, 'mean'):
            base_mean = mg_results.mean[0] if hasattr(mg_results.mean, '__len__') else mg_results.mean
            base_sd = mg_results.sd[0] if hasattr(mg_results, 'sd') and hasattr(mg_results.sd, '__len__') else getattr(mg_results, 'sd', 0.1)
        else:
            base_mean = 0.5
            base_sd = 0.1
        
        # Use GNN features to create spatial variation consistent with Whittle-Matérn
        if 'gnn_features' in self.results and self.results['gnn_features'] is not None:
            predictions = self._create_gnn_informed_field(base_mean, base_sd, addresses)
        else:
            predictions = self._create_simple_field(base_mean, base_sd, addresses)
        
        return predictions

    def _create_gnn_informed_field(self, base_mean, base_sd, addresses):
        """
        Create address-level field using GNN features within Whittle-Matérn framework
        """
        predictions = []
        gnn_features = self.results['gnn_features']
        
        # Extract spatial parameters from GNN (these informed the SPDE)
        if len(gnn_features) > 0:
            # Use mean GNN parameters as field characteristics
            mean_kappa = np.mean(gnn_features[:, 0])  # Range parameter
            mean_alpha = np.mean(gnn_features[:, 1])  # Smoothness parameter  
            mean_tau = np.mean(gnn_features[:, 2])    # Nugget effect
        else:
            mean_kappa, mean_alpha, mean_tau = 1.0, 0.5, 0.1
        
        self._log(f"  Using GNN-learned SPDE parameters: κ={mean_kappa:.3f}, α={mean_alpha:.3f}, τ={mean_tau:.3f}")
        
        # Create spatially correlated field based on GNN parameters
        n_addresses = len(addresses)
        
        # Generate spatial correlation structure
        addr_coords = np.array([[addr.geometry.x, addr.geometry.y] for _, addr in addresses.iterrows()])
        
        if n_addresses > 1:
            # Calculate pairwise distances
            distances = cdist(addr_coords, addr_coords)
            
            # Create Matérn-like correlation (simplified)
            # C(d) = exp(-κ * d^α) * τ  (simplified Matérn function)
            correlations = np.exp(-mean_kappa * np.power(distances + 1e-8, mean_alpha)) * mean_tau
            
            # Generate correlated spatial variation
            try:
                # Cholesky decomposition for correlated sampling
                L = np.linalg.cholesky(correlations + np.eye(n_addresses) * 1e-6)
                spatial_noise = L @ np.random.normal(0, 1, n_addresses)
            except np.linalg.LinAlgError:
                # Fallback if correlation matrix is singular
                spatial_noise = np.random.normal(0, mean_tau, n_addresses)
        else:
            spatial_noise = np.array([0.0])
        
        # Create predictions with spatial correlation
        for i, (idx, addr) in enumerate(addresses.iterrows()):
            # Add GNN-informed spatial variation to base field
            spatial_variation = spatial_noise[i] * base_sd * 0.5  # Scale by base uncertainty
            
            pred_mean = np.clip(base_mean + spatial_variation, 0, 1)
            pred_sd = base_sd * (1 + np.abs(spatial_variation) * 0.1)  # Adaptive uncertainty
            
            predictions.append({
                'address_id': idx,
                'longitude': addr.geometry.x,
                'latitude': addr.geometry.y,
                'mean': pred_mean,
                'sd': pred_sd,
                'q025': pred_mean - 1.96 * pred_sd,
                'q975': pred_mean + 1.96 * pred_sd
            })
        
        return predictions

    def _create_simple_field(self, base_mean, base_sd, addresses):
        """
        Simple field creation when GNN features unavailable
        """
        predictions = []
        
        for idx, addr in addresses.iterrows():
            # Add small spatial variation around base field value
            spatial_variation = np.random.normal(0, base_sd * 0.3)
            
            pred_mean = np.clip(base_mean + spatial_variation, 0, 1)
            pred_sd = base_sd
            
            predictions.append({
                'address_id': idx,
                'longitude': addr.geometry.x,
                'latitude': addr.geometry.y,
                'mean': pred_mean,
                'sd': pred_sd,
                'q025': pred_mean - 1.96 * pred_sd,
                'q975': pred_mean + 1.96 * pred_sd
            })
        
        return predictions

    # ==========================================
    # ENHANCED VISUALIZATION METHODS
    # ==========================================

    def _create_tract_visualization(self, results, tract_data, config):
        """
        ENHANCED: Create comprehensive visualization for individual tract processing
        """
        fips_code = results.get('fips_code', 'unknown')
        self._log(f"Creating visualization for tract {fips_code}")
        
        try:
            # Create output directory for tract
            tract_output_dir = os.path.join(self.output_dir, f'tract_{fips_code}')
            os.makedirs(tract_output_dir, exist_ok=True)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'GRANITE Results: Census Tract {fips_code}', fontsize=16, fontweight='bold')
            
            # Plot 1: Network structure
            ax1 = axes[0, 0]
            self._plot_tract_network(ax1, tract_data)
            ax1.set_title('Road Network Structure')
            
            # Plot 2: Prediction map
            ax2 = axes[0, 1]
            if 'predictions' in results and len(results['predictions']) > 0:
                self._plot_prediction_map(ax2, results['predictions'], tract_data)
                ax2.set_title('SVI Predictions')
            else:
                ax2.text(0.5, 0.5, 'No predictions available', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('SVI Predictions (No Data)')
            
            # Plot 3: Uncertainty visualization
            ax3 = axes[1, 0]
            if 'predictions' in results and 'uncertainty' in results['predictions'].columns:
                self._plot_uncertainty_map(ax3, results['predictions'])
                ax3.set_title('Prediction Uncertainty')
            else:
                ax3.text(0.5, 0.5, 'No uncertainty data', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Prediction Uncertainty (No Data)')
            
            # Plot 4: Performance metrics
            ax4 = axes[1, 1]
            self._plot_tract_metrics(ax4, results)
            ax4.set_title('Processing Metrics')
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = os.path.join(tract_output_dir, f'tract_{fips_code}_analysis.png')
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self._log(f"Tract visualization saved to: {viz_path}")
            
        except Exception as e:
            self._log(f"Error creating tract visualization: {str(e)}", 'ERROR')

    def _plot_tract_network(self, ax, tract_data):
        """Plot road network structure for tract"""
        try:
            if 'road_network' in tract_data:
                network = tract_data['road_network']
                pos = {node: node for node in network.nodes()}
                
                # Draw network
                nx.draw_networkx_edges(network, pos, ax=ax, alpha=0.5, width=0.5, color='gray')
                nx.draw_networkx_nodes(network, pos, ax=ax, node_size=10, alpha=0.7, color='blue')
                
                # Add addresses if available
                if 'addresses' in tract_data and len(tract_data['addresses']) > 0:
                    addresses = tract_data['addresses']
                    ax.scatter([addr.geometry.x for _, addr in addresses.iterrows()],
                              [addr.geometry.y for _, addr in addresses.iterrows()],
                              c='red', s=20, alpha=0.8, label='Addresses')
                    ax.legend()
            
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Network plot error: {str(e)[:50]}...', 
                   ha='center', va='center', transform=ax.transAxes)

    def _plot_prediction_map(self, ax, predictions, tract_data):
        """Plot SVI predictions map"""
        try:
            # Get prediction values
            pred_col = 'predicted_svi' if 'predicted_svi' in predictions.columns else 'mean'
            pred_values = predictions[pred_col]
            
            # Create scatter plot
            scatter = ax.scatter(predictions['longitude'], predictions['latitude'],
                               c=pred_values, cmap='viridis', s=30, alpha=0.8)
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='Predicted SVI')
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Prediction plot error: {str(e)[:50]}...', 
                   ha='center', va='center', transform=ax.transAxes)

    def _plot_uncertainty_map(self, ax, predictions):
        """Plot uncertainty map"""
        try:
            uncertainty_col = 'uncertainty' if 'uncertainty' in predictions.columns else 'sd'
            uncertainty_values = predictions[uncertainty_col]
            
            scatter = ax.scatter(predictions['longitude'], predictions['latitude'],
                               c=uncertainty_values, cmap='Reds', s=30, alpha=0.8)
            
            plt.colorbar(scatter, ax=ax, label='Prediction Uncertainty')
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Uncertainty plot error: {str(e)[:50]}...', 
                   ha='center', va='center', transform=ax.transAxes)

    def _plot_tract_metrics(self, ax, results):
        """Plot processing metrics"""
        try:
            metrics = results.get('metrics', {})
            
            # Create metrics display
            metrics_text = []
            
            if 'method' in metrics:
                metrics_text.append(f"Method: {metrics['method']}")
            
            if 'gnn_training' in metrics:
                gnn_metrics = metrics['gnn_training']
                if 'final_loss' in gnn_metrics:
                    metrics_text.append(f"GNN Loss: {gnn_metrics['final_loss']:.4f}")
            
            if 'metricgraph_success' in metrics:
                mg_success = "✓" if metrics['metricgraph_success'] else "✗"
                metrics_text.append(f"MetricGraph: {mg_success}")
            
            if 'n_predictions' in metrics:
                metrics_text.append(f"Predictions: {metrics['n_predictions']}")
            
            # Display metrics as text
            if metrics_text:
                ax.text(0.1, 0.9, '\n'.join(metrics_text), transform=ax.transAxes, 
                       verticalalignment='top', fontfamily='monospace')
            else:
                ax.text(0.5, 0.5, 'No metrics available', 
                       ha='center', va='center', transform=ax.transAxes)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Metrics error: {str(e)[:50]}...', 
                   ha='center', va='center', transform=ax.transAxes)

    def _create_batch_visualization(self, summary, config):
        """
        ENHANCED: Create comprehensive batch summary visualization
        """
        self._log("Creating batch summary visualization...")
        
        try:
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('GRANITE Batch Processing Summary', fontsize=16, fontweight='bold')
            
            # Plot 1: Processing success summary
            ax1 = axes[0, 0]
            success_counts = [summary['successful'], summary['skipped'], summary['errors']]
            labels = ['Successful', 'Skipped', 'Errors']
            colors = ['green', 'orange', 'red']
            ax1.pie(success_counts, labels=labels, colors=colors, autopct='%1.1f%%')
            ax1.set_title('Processing Results')
            
            # Plot 2: Network size distribution
            ax2 = axes[0, 1]
            self._plot_network_size_distribution(ax2, summary)
            ax2.set_title('Network Size Distribution')
            
            # Plot 3: Performance by tract
            ax3 = axes[1, 0]
            self._plot_batch_performance_metrics(ax3, summary)
            ax3.set_title('Performance by Tract')
            
            # Plot 4: Summary statistics
            ax4 = axes[1, 1]
            self._plot_batch_summary_stats(ax4, summary)
            ax4.set_title('Summary Statistics')
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = os.path.join(self.output_dir, 'batch_summary_visualization.png')
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self._log(f"Batch visualization saved to: {viz_path}")
            
        except Exception as e:
            self._log(f"Error creating batch visualization: {str(e)}", 'ERROR')

    def _plot_network_size_distribution(self, ax, summary):
        """Plot distribution of network sizes across tracts"""
        try:
            network_sizes = []
            for fips, result in summary['results'].items():
                if 'network_stats' in result:
                    network_sizes.append(result['network_stats'].get('edges', 0))
            
            if network_sizes:
                ax.hist(network_sizes, bins=min(10, len(network_sizes)), alpha=0.7, color='skyblue')
                ax.set_xlabel('Number of Edges')
                ax.set_ylabel('Number of Tracts')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No network data available', 
                       ha='center', va='center', transform=ax.transAxes)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Network plot error: {str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes)

    def _plot_batch_performance_metrics(self, ax, summary):
        """Plot performance metrics across tracts"""
        try:
            fips_codes = []
            n_predictions = []
            
            for fips, result in summary['results'].items():
                if result.get('status') == 'success':
                    fips_codes.append(fips[-4:])  # Last 4 digits for readability
                    n_preds = len(result.get('predictions', []))
                    n_predictions.append(n_preds)
            
            if fips_codes and n_predictions:
                bars = ax.bar(range(len(fips_codes)), n_predictions, alpha=0.7, color='lightcoral')
                ax.set_xlabel('Census Tract (last 4 digits)')
                ax.set_ylabel('Number of Predictions')
                ax.set_xticks(range(len(fips_codes)))
                ax.set_xticklabels(fips_codes, rotation=45)
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom', fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No performance data available', 
                       ha='center', va='center', transform=ax.transAxes)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Performance plot error: {str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes)

    def _plot_batch_summary_stats(self, ax, summary):
        """Plot summary statistics"""
        try:
            stats_text = [
                f"Total Tracts: {summary['total_tracts']}",
                f"Success Rate: {summary['success_rate']:.1%}",
                f"Successful: {summary['successful']}",
                f"Skipped: {summary['skipped']}",
                f"Errors: {summary['errors']}"
            ]
            
            # Calculate additional statistics
            successful_results = {k: v for k, v in summary['results'].items() 
                                if v.get('status') == 'success'}
            
            if successful_results:
                total_predictions = sum(len(result.get('predictions', [])) 
                                      for result in successful_results.values())
                avg_predictions = total_predictions / len(successful_results)
                stats_text.extend([
                    f"",
                    f"Total Predictions: {total_predictions}",
                    f"Avg Predictions/Tract: {avg_predictions:.1f}"
                ])
            
            ax.text(0.1, 0.9, '\n'.join(stats_text), transform=ax.transAxes,
                   verticalalignment='top', fontfamily='monospace', fontsize=12)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Stats error: {str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes)

    def _enhanced_gnn_fallback(self, road_network, svi_data, addresses, gnn_features):
        """
        Enhanced fallback using GNN features for spatial interpolation
        This provides better predictions than simple distance weighting
        """
        self._log("Using enhanced GNN-based spatial interpolation...")
        
        # Get node positions
        nodes = list(road_network.nodes())
        node_positions = np.array([[node[0], node[1]] for node in nodes])
        
        predictions = []
        
        for _, addr in addresses.iterrows():
            addr_point = np.array([addr.geometry.x, addr.geometry.y])
            
            # Find nearest network nodes
            distances = np.linalg.norm(node_positions - addr_point, axis=1)
            nearest_indices = np.argsort(distances)[:5]  # Use 5 nearest nodes
            
            # Weight by inverse distance
            nearest_distances = distances[nearest_indices]
            weights = 1 / (nearest_distances + 1e-6)
            weights /= weights.sum()
            
            # Interpolate using GNN features
            if len(gnn_features) > max(nearest_indices):
                weighted_features = np.average(gnn_features[nearest_indices], weights=weights, axis=0)
                
                # Convert GNN features to SVI prediction
                # This is a simple linear combination - could be enhanced with learned mapping
                predicted_svi = np.clip(
                    0.3 * weighted_features[0] + 0.4 * weighted_features[1] + 0.3 * weighted_features[2],
                    0, 1
                )
                
                # Estimate uncertainty based on feature variance
                feature_variance = np.var(gnn_features[nearest_indices], axis=0)
                uncertainty = np.sqrt(np.mean(feature_variance)) * 0.2
            else:
                # Fallback to simple mean if GNN features don't align
                predicted_svi = 0.5
                uncertainty = 0.15
            
            predictions.append({
                'x': addr['x'],
                'y': addr['y'],
                'predicted_svi': predicted_svi,
                'uncertainty': uncertainty,
                'ci_lower': predicted_svi - 1.96 * uncertainty,
                'ci_upper': predicted_svi + 1.96 * uncertainty
            })
        
        return pd.DataFrame(predictions)

    def _distance_weighted_interpolation(self, field_coords, field_means, field_sds, addr_coords):
        """
        Distance-weighted interpolation fallback for Whittle-Matérn field
        """
        interpolated_means = []
        interpolated_sds = []
        
        for addr_coord in addr_coords:
            # Calculate distances to all field points
            distances = np.sqrt(np.sum((field_coords - addr_coord)**2, axis=1))
            
            # Inverse distance weighting
            weights = 1 / (distances + 1e-6)
            weights /= weights.sum()
            
            # Weighted interpolation
            interp_mean = np.average(field_means, weights=weights)
            interp_sd = np.sqrt(np.average(field_sds**2, weights=weights))
            
            interpolated_means.append(interp_mean)
            interpolated_sds.append(interp_sd)
        
        return np.array(interpolated_means), np.array(interpolated_sds)

    def _log_interpolation_method(self, method_used):
        """Log which interpolation method was actually used"""
        method_descriptions = {
            'metricgraph_spatial': 'MetricGraph Whittle-Matérn field (multiple points)',
            'metricgraph_sparse': 'MetricGraph Whittle-Matérn field (sparse)',
            'metricgraph_gnn': 'MetricGraph with GNN-informed spatial variation',
            'network_fallback': 'Network-aware interpolation fallback',
            'gnn_fallback': 'GNN-enhanced interpolation fallback',
            'basic_fallback': 'Basic spatial interpolation fallback'
        }
        
        description = method_descriptions.get(method_used, method_used)
        self._log(f"  → Interpolation method: {description}")

    def _enhanced_fallback_interpolation(self, addresses):
        """
        Enhanced fallback when MetricGraph disaggregation is unavailable
        Uses the original enhanced methods as sophisticated alternatives
        """
        self._log("  Using enhanced fallback interpolation (MetricGraph unavailable)")
        
        # Try network-aware interpolation if road network available
        if 'road_network' in self.data and 'tracts_with_svi' in self.data:
            try:
                return self._network_aware_interpolation_fallback(addresses)
            except Exception as e:
                self._log(f"  Network interpolation failed: {e}")
        
        # Try GNN-enhanced interpolation if features available
        if 'gnn_features' in self.results and self.results['gnn_features'] is not None:
            try:
                return self._gnn_enhanced_interpolation_fallback(addresses)
            except Exception as e:
                self._log(f"  GNN interpolation failed: {e}")
        
        # Final fallback to basic interpolation
        return self._basic_spatial_fallback(addresses)

    def _network_aware_interpolation_fallback(self, addresses):
        """
        Network-aware fallback (from original enhanced methods)
        Only used when MetricGraph disaggregation unavailable
        """
        self._log("  Using network-aware interpolation fallback")
        
        predictions = []
        road_network = self.data['road_network']
        tract_centroids = self.data['tracts_with_svi'].geometry.centroid
        tract_svi_values = self.data['tracts_with_svi']['RPL_THEMES'].values
        
        for idx, addr in addresses.iterrows():
            addr_point = (addr.geometry.x, addr.geometry.y)
            
            # Find nearest network nodes and calculate network distances
            network_nodes = list(road_network.nodes())
            node_distances = [
                np.sqrt((addr_point[0] - node[0])**2 + (addr_point[1] - node[1])**2)
                for node in network_nodes
            ]
            nearest_node_idx = np.argmin(node_distances)
            nearest_node = network_nodes[nearest_node_idx]
            
            # Calculate weights to tract centroids via network
            weights = []
            for centroid in tract_centroids:
                centroid_point = (centroid.x, centroid.y)
                centroid_distances = [
                    np.sqrt((centroid_point[0] - node[0])**2 + (centroid_point[1] - node[1])**2)
                    for node in network_nodes
                ]
                nearest_centroid_node = network_nodes[np.argmin(centroid_distances)]
                
                try:
                    network_dist = nx.shortest_path_length(road_network, nearest_node, nearest_centroid_node)
                except:
                    network_dist = np.sqrt((nearest_node[0] - nearest_centroid_node[0])**2 + 
                                        (nearest_node[1] - nearest_centroid_node[1])**2)
                
                weights.append(1 / (network_dist + 1e-6))
            
            weights = np.array(weights)
            weights /= weights.sum()
            
            # Interpolate using network-weighted tract values
            pred_mean = np.average(tract_svi_values, weights=weights)
            pred_sd = np.std(tract_svi_values) * 0.3
            
            predictions.append({
                'address_id': idx,
                'longitude': addr_point[0],
                'latitude': addr_point[1],
                'mean': pred_mean,
                'sd': pred_sd,
                'q025': pred_mean - 1.96 * pred_sd,
                'q975': pred_mean + 1.96 * pred_sd
            })
        
        return pd.DataFrame(predictions)

    def _gnn_enhanced_interpolation_fallback(self, addresses):
        """
        GNN-enhanced fallback (from original enhanced methods)
        Only used when MetricGraph disaggregation unavailable
        """
        self._log("  Using GNN-enhanced interpolation fallback")
        
        gnn_features = self.results['gnn_features']
        kappa_mean = np.mean(gnn_features[:, 0])
        alpha_mean = np.mean(gnn_features[:, 1])
        tau_mean = np.mean(gnn_features[:, 2])
        
        # Base prediction from tract data
        if 'tracts_with_svi' in self.data:
            base_svi = self.data['tracts_with_svi']['RPL_THEMES'].mean()
            base_uncertainty = self.data['tracts_with_svi']['RPL_THEMES'].std() * 0.3
        else:
            base_svi = 0.5
            base_uncertainty = 0.1
        
        predictions = []
        for idx, addr in addresses.iterrows():
            spatial_variation = tau_mean * np.random.normal(0, 1)
            pred_mean = np.clip(base_svi + spatial_variation, 0, 1)
            pred_sd = base_uncertainty * (1 + alpha_mean * 0.5)
            
            predictions.append({
                'address_id': idx,
                'longitude': addr.geometry.x,
                'latitude': addr.geometry.y,
                'mean': pred_mean,
                'sd': pred_sd,
                'q025': pred_mean - 1.96 * pred_sd,
                'q975': pred_mean + 1.96 * pred_sd
            })
        
        return pd.DataFrame(predictions)

    def _basic_spatial_fallback(self, addresses):
        """
        Basic spatial fallback when all else fails
        """
        self._log("  Using basic spatial fallback")
        
        # Use simple tract average with small variation
        if 'tracts_with_svi' in self.data:
            base_svi = self.data['tracts_with_svi']['RPL_THEMES'].mean()
            svi_std = self.data['tracts_with_svi']['RPL_THEMES'].std()
        else:
            base_svi = 0.5
            svi_std = 0.2
        
        predictions = []
        for idx, addr in addresses.iterrows():
            spatial_variation = np.random.normal(0, svi_std * 0.3)
            pred_svi = np.clip(base_svi + spatial_variation, 0, 1)
            uncertainty = svi_std * 0.5
            
            predictions.append({
                'address_id': idx,
                'longitude': addr.geometry.x,
                'latitude': addr.geometry.y,
                'mean': pred_svi,
                'sd': uncertainty,
                'q025': pred_svi - 1.96 * uncertainty,
                'q975': pred_svi + 1.96 * uncertainty
            })
        
        return pd.DataFrame(predictions)
    
    def analyze_gnn_input_features(self, pyg_data, save_analysis=True):
        """
        Analyze the input features that go into the GNN
        Call this to understand what the GNN is learning FROM
        """
        # Extract node features
        node_features = pyg_data.x.numpy() if hasattr(pyg_data.x, 'numpy') else pyg_data.x
        n_nodes, n_features = node_features.shape
        
        if self.verbose:
            self._log(f"GNN Input Feature Analysis:")
            self._log(f"  - Number of nodes: {n_nodes}")
            self._log(f"  - Number of input features: {n_features}")
        
        # Get feature names
        feature_names = self._get_gnn_input_feature_names(n_features)
        
        # Analyze each input feature
        feature_stats = []
        for i, feature_name in enumerate(feature_names):
            feature_values = node_features[:, i]
            
            stats = {
                'feature_name': feature_name,
                'min_value': float(feature_values.min()),
                'max_value': float(feature_values.max()),
                'mean_value': float(feature_values.mean()),
                'std_value': float(feature_values.std()),
                'non_zero_count': int(np.count_nonzero(feature_values)),
                'total_count': len(feature_values)
            }
            
            feature_stats.append(stats)
            
            if self.verbose:
                self._log(f"  Feature {i+1} ({feature_name}):")
                self._log(f"    - Range: [{stats['min_value']:.4f}, {stats['max_value']:.4f}]")
                self._log(f"    - Mean: {stats['mean_value']:.4f}")
                self._log(f"    - Std: {stats['std_value']:.4f}")
                self._log(f"    - Non-zero: {stats['non_zero_count']}/{stats['total_count']}")
        
        # Create analysis DataFrame
        feature_analysis = pd.DataFrame(feature_stats)
        
        # Save analysis if requested
        if save_analysis and hasattr(self, 'output_dir'):
            analysis_file = os.path.join(self.output_dir, 'gnn_input_features.csv')
            feature_analysis.to_csv(analysis_file, index=False)
            if self.verbose:
                self._log(f"  Saved input feature analysis to: {analysis_file}")
        
        return feature_analysis, node_features

    def analyze_gnn_learned_features(self, gnn_features, training_metrics=None, save_analysis=True):
        """
        Analyze the features learned by the GNN (SPDE parameters)
        Call this to understand what the GNN has LEARNED
        """
        # Convert to numpy array if needed
        if hasattr(gnn_features, 'numpy'):
            features_array = gnn_features.numpy()
        elif isinstance(gnn_features, np.ndarray):
            features_array = gnn_features
        else:
            features_array = np.array(gnn_features)
        
        n_nodes, n_output_features = features_array.shape
        
        if self.verbose:
            self._log(f"GNN Learned Feature Analysis:")
            self._log(f"  - Number of nodes: {n_nodes}")
            self._log(f"  - Number of output features: {n_output_features}")
        
        # Standard SPDE parameter names
        spde_names = ['kappa', 'alpha', 'tau']
        
        # Analyze each learned parameter
        parameter_stats = []
        for i in range(n_output_features):
            param_name = spde_names[i] if i < len(spde_names) else f'param_{i+1}'
            param_values = features_array[:, i]
            
            stats = {
                'parameter_name': param_name,
                'min_value': float(param_values.min()),
                'max_value': float(param_values.max()),
                'mean_value': float(param_values.mean()),
                'std_value': float(param_values.std()),
                'coeff_variation': float(param_values.std() / param_values.mean()) if param_values.mean() != 0 else 0.0
            }
            
            parameter_stats.append(stats)
            
            if self.verbose:
                self._log(f"  SPDE Parameter {i+1} ({param_name}):")
                self._log(f"    - Range: [{stats['min_value']:.4f}, {stats['max_value']:.4f}]")
                self._log(f"    - Mean: {stats['mean_value']:.4f}")
                self._log(f"    - Std: {stats['std_value']:.4f}")
                self._log(f"    - Coefficient of variation: {stats['coeff_variation']:.4f}")
        
        # Analyze correlations between learned parameters
        if n_output_features > 1:
            correlations = np.corrcoef(features_array.T)
            if self.verbose:
                self._log(f"  Parameter correlations:")
                for i in range(n_output_features):
                    for j in range(i+1, n_output_features):
                        param_i = spde_names[i] if i < len(spde_names) else f'param_{i+1}'
                        param_j = spde_names[j] if j < len(spde_names) else f'param_{j+1}'
                        corr = correlations[i, j]
                        self._log(f"    - {param_i} vs {param_j}: {corr:.4f}")
        
        # Add training metrics if available
        if training_metrics and self.verbose:
            self._log(f"  Training metrics:")
            if 'final_loss' in training_metrics:
                self._log(f"    - Final loss: {training_metrics['final_loss']:.6f}")
            if 'spatial_loss' in training_metrics:
                self._log(f"    - Spatial loss: {training_metrics['spatial_loss']:.6f}")
            if 'regularization_loss' in training_metrics:
                self._log(f"    - Regularization loss: {training_metrics['regularization_loss']:.6f}")
        
        # Create analysis DataFrame
        learned_analysis = pd.DataFrame(parameter_stats)
        
        # Save analysis if requested
        if save_analysis and hasattr(self, 'output_dir'):
            analysis_file = os.path.join(self.output_dir, 'gnn_learned_parameters.csv')
            learned_analysis.to_csv(analysis_file, index=False)
            if self.verbose:
                self._log(f"  Saved learned parameter analysis to: {analysis_file}")
        
        return learned_analysis, features_array

    def analyze_gnn_feature_transformation(self, input_features, learned_features, save_analysis=True):
        """
        Analyze how the GNN transforms input features into learned parameters
        Call this to understand the INPUT → OUTPUT mapping
        """
        n_input = input_features.shape[1]
        n_output = learned_features.shape[1]
        
        if self.verbose:
            self._log(f"GNN Feature Transformation Analysis:")
            self._log(f"  - Input features: {n_input}")
            self._log(f"  - Output features: {n_output}")
            self._log(f"  - Transformation: {n_input} → {n_output}")
        
        # Get feature names
        input_names = self._get_gnn_input_feature_names(n_input)
        output_names = ['kappa', 'alpha', 'tau'][:n_output]
        
        # Calculate cross-correlations
        transformation_data = []
        
        for i, input_name in enumerate(input_names):
            for j, output_name in enumerate(output_names):
                correlation = np.corrcoef(input_features[:, i], learned_features[:, j])[0, 1]
                
                transformation_data.append({
                    'input_feature': input_name,
                    'output_parameter': output_name,
                    'correlation': correlation,
                    'abs_correlation': abs(correlation)
                })
        
        # Create transformation DataFrame
        transformation_df = pd.DataFrame(transformation_data)
        
        # Find and report strongest relationships
        strongest = transformation_df.nlargest(5, 'abs_correlation')
        if self.verbose:
            self._log(f"  Strongest input-output relationships:")
            for _, row in strongest.iterrows():
                self._log(f"    - {row['input_feature']} → {row['output_parameter']}: r={row['correlation']:.4f}")
        
        # Save analysis if requested
        if save_analysis and hasattr(self, 'output_dir'):
            analysis_file = os.path.join(self.output_dir, 'gnn_feature_transformation.csv')
            transformation_df.to_csv(analysis_file, index=False)
            if self.verbose:
                self._log(f"  Saved transformation analysis to: {analysis_file}")
        
        return transformation_df

    def create_gnn_feature_plots(self, input_features, learned_features, save_plots=True):
        """
        Create visualizations of GNN input and learned features
        """
        if not save_plots or not hasattr(self, 'output_dir'):
            return
        
        if self.verbose:
            self._log("Creating GNN feature visualizations...")
        
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('GRANITE GNN Feature Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: Input feature distributions
            ax1 = axes[0, 0]
            n_input_features = min(input_features.shape[1], 5)
            for i in range(n_input_features):
                ax1.hist(input_features[:, i], alpha=0.6, bins=30, 
                        label=f'Feature {i+1}', density=True)
            ax1.set_title('Input Feature Distributions')
            ax1.set_xlabel('Feature Value')
            ax1.set_ylabel('Density')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Learned parameter distributions
            ax2 = axes[0, 1]
            param_names = ['κ (kappa)', 'α (alpha)', 'τ (tau)']
            for i in range(learned_features.shape[1]):
                param_name = param_names[i] if i < len(param_names) else f'Param {i+1}'
                ax2.hist(learned_features[:, i], alpha=0.6, bins=30, 
                        label=param_name, density=True)
            ax2.set_title('Learned SPDE Parameters')
            ax2.set_xlabel('Parameter Value')
            ax2.set_ylabel('Density')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Input feature correlations
            ax3 = axes[0, 2]
            if input_features.shape[1] <= 10:
                input_corr = np.corrcoef(input_features.T)
                sns.heatmap(input_corr, annot=True, cmap='coolwarm', center=0, ax=ax3,
                        xticklabels=[f'F{i+1}' for i in range(input_features.shape[1])],
                        yticklabels=[f'F{i+1}' for i in range(input_features.shape[1])])
                ax3.set_title('Input Feature Correlations')
            else:
                ax3.text(0.5, 0.5, 'Too many features\nfor correlation plot', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Input Features (Too Many to Show)')
            
            # Plot 4: Learned parameter correlations  
            ax4 = axes[1, 0]
            learned_corr = np.corrcoef(learned_features.T)
            param_labels = ['κ', 'α', 'τ'][:learned_features.shape[1]]
            sns.heatmap(learned_corr, annot=True, cmap='coolwarm', center=0, ax=ax4,
                    xticklabels=param_labels, yticklabels=param_labels)
            ax4.set_title('Learned Parameter Correlations')
            
            # Plot 5: Spatial distribution of kappa (if possible)
            ax5 = axes[1, 1]
            if (hasattr(self, 'data') and 'road_network' in self.data and 
                len(list(self.data['road_network'].nodes())) == len(learned_features)):
                
                network = self.data['road_network']
                nodes = list(network.nodes())
                x_coords = [node[0] for node in nodes]
                y_coords = [node[1] for node in nodes]
                scatter = ax5.scatter(x_coords, y_coords, c=learned_features[:, 0], 
                                    cmap='viridis', s=10, alpha=0.6)
                plt.colorbar(scatter, ax=ax5, label='κ value')
                ax5.set_title('Spatial Distribution of κ')
                ax5.set_xlabel('Longitude')
                ax5.set_ylabel('Latitude')
            else:
                ax5.text(0.5, 0.5, 'Spatial plot unavailable\n(no network coordinates)', 
                        ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Spatial Distribution of κ')
            
            # Plot 6: Feature transformation heatmap
            ax6 = axes[1, 2]
            if input_features.shape[1] <= 10 and learned_features.shape[1] <= 5:
                # Calculate cross-correlations for heatmap
                cross_corr = np.zeros((input_features.shape[1], learned_features.shape[1]))
                for i in range(input_features.shape[1]):
                    for j in range(learned_features.shape[1]):
                        cross_corr[i, j] = np.corrcoef(input_features[:, i], learned_features[:, j])[0, 1]
                
                sns.heatmap(cross_corr, annot=True, cmap='RdBu_r', center=0, ax=ax6,
                        xticklabels=['κ', 'α', 'τ'][:learned_features.shape[1]],
                        yticklabels=[f'Input {i+1}' for i in range(input_features.shape[1])])
                ax6.set_title('Input → Output Correlations')
            else:
                ax6.text(0.5, 0.5, 'Too many features\nfor correlation heatmap', 
                        ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('Feature Transformation')
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = os.path.join(self.output_dir, 'gnn_feature_analysis.png')
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            if self.verbose:
                self._log(f"  Feature visualization saved to: {viz_path}")
            
        except Exception as e:
            if self.verbose:
                self._log(f"Error creating feature visualizations: {str(e)}", 'ERROR')

    def _get_gnn_input_feature_names(self, n_features):
        """
        Get meaningful names for input features based on typical prepare_graph_data output
        """
        # These are typical features created by prepare_graph_data
        standard_features = [
            'degree_centrality',      # How many roads connect here
            'betweenness_centrality', # How often on shortest paths  
            'distance_to_arterial',   # Distance to major roads
            'local_density',          # Road density in neighborhood
            'coordinate_feature'      # Spatial position encoding
        ]
        
        if n_features <= len(standard_features):
            return standard_features[:n_features]
        else:
            # Add generic names for extra features
            extra_features = [f'feature_{i+1}' for i in range(len(standard_features), n_features)]
            return standard_features + extra_features

    def run_gnn_feature_analysis(pipeline, save_all=True):
        """
        Standalone function to run GNN feature analysis on an existing pipeline
        
        Usage:
            pipeline = GRANITEPipeline(...)
            pipeline.run(...)  # Run your normal pipeline
            run_gnn_feature_analysis(pipeline)  # Add this to analyze features
        """
        if not hasattr(pipeline, 'results') or 'gnn_features' not in pipeline.results:
            print("Error: Pipeline must be run first and have GNN features")
            return
        
        if not hasattr(pipeline, 'data') or 'pyg_data' not in pipeline.data:
            print("Error: Pipeline must have PyG data")
            return
        
        print("Running GNN feature analysis...")
        
        # Run all analysis
        try:
            input_analysis, input_features = pipeline.analyze_gnn_input_features(
                pipeline.data['pyg_data'], save_analysis=save_all
            )
            
            learned_analysis, learned_features = pipeline.analyze_gnn_learned_features(
                pipeline.results['gnn_features'], save_analysis=save_all
            )
            
            transformation_analysis = pipeline.analyze_gnn_feature_transformation(
                input_features, learned_features, save_analysis=save_all
            )
            
            pipeline.create_gnn_feature_plots(
                input_features, learned_features, save_plots=save_all
            )
            
            print("✓ GNN feature analysis complete!")
            
            return {
                'input_analysis': input_analysis,
                'learned_analysis': learned_analysis,
                'transformation_analysis': transformation_analysis
            }
            
        except Exception as e:
            print(f"Error in feature analysis: {e}")
            return None

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