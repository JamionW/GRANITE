"""
MetricGraph R interface for GRANITE framework

This module provides the interface to the R MetricGraph package for
Whittle-Matérn spatial modeling on metric graphs.
"""
import numpy as np
import pandas as pd
import warnings
import time
from datetime import datetime
from typing import Optional, Tuple, Dict

# Suppress R environment warnings
warnings.filterwarnings("ignore", message="Environment variable.*redefined by R")
warnings.filterwarnings("ignore", category=UserWarning, module="rpy2")

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter


class MetricGraphInterface:
    """Interface to R MetricGraph package for Whittle-Matérn spatial modeling"""
    
    def __init__(self, config: Dict = None, verbose: bool = True):
        """
        Initialize MetricGraph interface
        
        Parameters:
        -----------
        config : Dict
            Configuration dictionary
        verbose : bool
            Enable verbose logging
        """
        self.verbose = verbose
        self.config = config or {}
        self.mg_config = self.config.get('metricgraph', {})
        
        # Initialize timing
        self.timing_stats = {}
        
        self._log("Initializing MetricGraph R interface...")
        
        try:
            # Initialize R interface
            self.converter = ro.default_converter + pandas2ri.converter
            self.base = importr('base')
            
            # Load MetricGraph package
            try:
                self.mg = importr('MetricGraph')
                self._log("MetricGraph package loaded successfully")
                
                # Define R functions
                self._define_r_functions()
                
            except Exception as e:
                self._log(f"Warning: MetricGraph package not available: {e}")
                self.mg = None
                
        except Exception as e:
            self._log(f"Error initializing R interface: {str(e)}")
            self.mg = None
    
    def _log(self, message: str):
        """Log message with timestamp"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] MetricGraph: {message}")
    
    def _log_timing(self, operation: str, start_time: float):
        """Log operation timing"""
        elapsed = time.time() - start_time
        self.timing_stats[operation] = elapsed
        if self.verbose:
            self._log(f"{operation} completed in {elapsed:.2f}s")
    
    def _define_r_functions(self):
        """Define R functions for MetricGraph operations"""
        if self.mg is None:
            return
        
        # Get configuration parameters
        max_edges = self.mg_config.get('max_edges', 2000)
        batch_size = self.mg_config.get('batch_size', 300)
        
        ro.r(f'''
        create_metric_graph <- function(nodes, edges, max_edges = {max_edges}, batch_size = {batch_size}) {{
            # Validate inputs
            if(nrow(nodes) == 0 || nrow(edges) == 0) {{
                return(list(success = FALSE, error = "Empty input data"))
            }}
            
            # Prepare data
            V <- as.matrix(nodes[, c("x", "y")])
            E <- as.matrix(edges[, c("from", "to")])
            
            cat("DEBUG: V dimensions:", dim(V), "\\n")
            cat("DEBUG: E dimensions:", dim(E), "\\n") 
            cat("DEBUG: E range: from", min(E[,1]), "to", max(E[,1]), ", to", min(E[,2]), "to", max(E[,2]), "\\n")
            
            # Check size constraints
            if(nrow(E) > max_edges) {{
                cat("Warning: Network has", nrow(E), "edges, exceeding limit of", max_edges, "\\n")
            }}
            
            # Create MetricGraph using V/E matrices (avoid edge_list approach)
            tryCatch({{
                cat("DEBUG: Trying V/E matrix approach...\\n")
                
                graph <- metric_graph$new(
                    V = V,                    # Vertex coordinate matrix
                    E = E,                    # Edge connectivity matrix  
                    longlat = FALSE,
                    perform_merges = TRUE,
                    tolerance = 0.001,
                    verbose = 1
                )
                
                cat("DEBUG: V/E matrix approach succeeded!\\n")
                return(list(success = TRUE, graph = graph))
                
            }}, error = function(e) {{
                cat("DEBUG: V/E approach failed:", conditionMessage(e), "\\n")
                
                # Fallback: Try without merges
                tryCatch({{
                    cat("DEBUG: Trying V/E without merges...\\n")
                    
                    graph <- metric_graph$new(
                        V = V,
                        E = E,
                        longlat = FALSE,
                        perform_merges = FALSE,  # Disable merges
                        verbose = 1
                    )
                    
                    cat("DEBUG: V/E without merges succeeded!\\n")
                    return(list(success = TRUE, graph = graph))
                    
                }}, error = function(e2) {{
                    cat("DEBUG: All approaches failed. Final error:", conditionMessage(e2), "\\n")
                    return(list(success = FALSE, error = conditionMessage(e2)))
                }})
            }})
        }}
        
        fit_whittle_matern <- function(graph, observations, gnn_features = NULL, 
                                     alpha = 1.5, mesh_resolution = 0.05) {{
            tryCatch({{
                # Build mesh
                graph$build_mesh(h = mesh_resolution)
                
                # Add observations
                graph$add_observations(
                    data = observations,
                    data_coords = "spatial",  
                    coord_x = "coord_x",     
                    coord_y = "coord_y",     
                    tolerance = 1e-6
                )
                
                # Prepare formula
                if(!is.null(gnn_features) && ncol(gnn_features) >= 3) {{
                    # Add GNN features as covariates
                    graph_data <- graph$get_data()
                    graph_data$gnn_kappa <- gnn_features[, 1]
                    graph_data$gnn_alpha <- gnn_features[, 2] 
                    graph_data$gnn_tau <- gnn_features[, 3]
                    
                    graph$clear_observations()
                    graph$add_observations(
                        data = graph_data,
                        data_coords = "spatial",
                        coord_x = "coord_x", 
                        coord_y = "coord_y",
                        tolerance = 1e-6
                    )
                    
                    formula_str <- "y ~ gnn_kappa + gnn_alpha + gnn_tau"
                }} else {{
                    formula_str <- "y ~ 1"
                }}
                
                # Fit Whittle-Matérn model
                model <- graph_spde(
                    graph = graph,
                    alpha = alpha,
                    model = list(formula = as.formula(formula_str))
                )
                
                return(list(success = TRUE, model = model))
                
            }}, error = function(e) {{
                return(list(success = FALSE, error = conditionMessage(e)))
            }})
        }}
        
        predict_whittle_matern <- function(model, graph, prediction_locations) {{
            tryCatch({{
                # Project prediction locations
                pred_projected <- graph$project_observations(
                    prediction_locations,
                    tolerance = 1e-6
                )
                
                # Make predictions
                preds <- predict(model, 
                               newdata = pred_projected,
                               normalized = TRUE)
                
                # Format results
                result <- data.frame(
                    mean = preds$mean,
                    sd = sqrt(pmax(preds$variance, 1e-8)),
                    q025 = preds$mean - 1.96 * sqrt(pmax(preds$variance, 1e-8)),
                    q975 = preds$mean + 1.96 * sqrt(pmax(preds$variance, 1e-8))
                )
                
                return(result)
                
            }}, error = function(e) {{
                cat("Prediction error:", conditionMessage(e), "\\n")
                return(NULL)
            }})
        }}
        ''')
    
    def create_graph(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, 
                    enable_sampling: bool = None) -> Optional[object]:
        """
        Create MetricGraph object from network data
        
        Parameters:
        -----------
        nodes_df : pd.DataFrame
            Nodes with x, y coordinates
        edges_df : pd.DataFrame
            Edges with from, to indices (0-based)
        enable_sampling : bool, optional
            Override config setting for smart sampling
            
        Returns:
        --------
        R MetricGraph object or None if failed
        """
        if self.mg is None:
            self._log("MetricGraph not available")
            return None
        
        start_time = time.time()
        self._log(f"Creating MetricGraph from {len(nodes_df)} nodes, {len(edges_df)} edges")
        
        # Check if sampling should be enabled
        if enable_sampling is None:
            enable_sampling = self.mg_config.get('enable_sampling', False)
        
        # Apply smart sampling if enabled
        if enable_sampling and len(edges_df) > self.mg_config.get('max_edges', 2000):
            nodes_df, edges_df = self._apply_smart_sampling(nodes_df, edges_df)
        
        # Prepare data
        nodes_clean = nodes_df[['x', 'y']].copy()
        edges_clean = edges_df[['from', 'to']].copy()
        
        # Adjust indices for R (1-based)
        edges_clean['from'] += 1
        edges_clean['to'] += 1
        
        # Call R function
        try:
            with localconverter(self.converter):
                r_nodes = ro.conversion.py2rpy(nodes_clean)
                r_edges = ro.conversion.py2rpy(edges_clean)
                
                create_func = ro.r['create_metric_graph']
                result = create_func(
                    r_nodes, 
                    r_edges,
                    self.mg_config.get('max_edges', 2000),
                    self.mg_config.get('batch_size', 300)
                )
                
                success = result[0][0]
                
                if success:
                    graph = result[1]
                    self._log_timing("Graph creation", start_time)
                    return graph
                else:
                    error_msg = str(result[1])
                    self._log(f"Graph creation failed: {error_msg}")
                    return None
                    
        except Exception as e:
            self._log(f"Error creating graph: {str(e)}")
            return None
    
    def disaggregate_svi(self, metric_graph: object, observations: pd.DataFrame,
                        prediction_locations: pd.DataFrame, 
                        gnn_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Perform SVI disaggregation using Whittle-Matérn model
        
        Parameters:
        -----------
        metric_graph : R MetricGraph object
            The metric graph
        observations : pd.DataFrame
            Tract-level SVI observations with x, y, value columns
        prediction_locations : pd.DataFrame
            Address locations for prediction with x, y columns
        gnn_features : pd.DataFrame, optional
            GNN-learned SPDE parameters
            
        Returns:
        --------
        pd.DataFrame
            Predictions with mean, sd, q025, q975 columns
        """
        if metric_graph is None:
            self._log("No MetricGraph provided, using fallback")
            return self._fallback_prediction(observations, prediction_locations)
        
        start_time = time.time()
        self._log("Starting Whittle-Matérn disaggregation")
        
        try:
            # Prepare data
            obs_for_r = pd.DataFrame({
                'coord_x': observations['coord_x'],
                'coord_y': observations['coord_y'], 
                'svi_value': observations['svi_value']  # Keep separate from coordinates
            })
            
            pred_locs = prediction_locations[['x', 'y']].copy()
            
            with localconverter(self.converter):
                # Convert to R
                r_obs = ro.conversion.py2rpy(obs_for_r)
                r_pred_locs = ro.conversion.py2rpy(pred_locs)
                
                r_gnn_features = None
                if gnn_features is not None:
                    r_gnn_features = ro.conversion.py2rpy(gnn_features)
                
                # Fit model
                self._log("Fitting Whittle-Matérn model...")
                fit_func = ro.r['fit_whittle_matern']
                fit_result = fit_func(
                    metric_graph,
                    r_obs,
                    r_gnn_features,
                    self.mg_config.get('alpha', 1.5),
                    self.mg_config.get('mesh_resolution', 0.05)
                )
                
                if not fit_result[0][0]:
                    self._log(f"Model fitting failed: {fit_result[1]}")
                    return self._fallback_prediction(observations, prediction_locations)
                
                model = fit_result[1]
                
                # Make predictions
                self._log(f"Predicting at {len(prediction_locations)} locations...")
                predict_func = ro.r['predict_whittle_matern']
                r_result = predict_func(model, metric_graph, r_pred_locs)
                
                if r_result is None:
                    self._log("Prediction failed, using fallback")
                    return self._fallback_prediction(observations, prediction_locations)
                
                # Convert back to pandas
                result_df = ro.conversion.rpy2py(r_result)
                
                self._log_timing("Disaggregation", start_time)
                self._log(f"Successfully predicted {len(result_df)} values")
                
                return result_df
                
        except Exception as e:
            self._log(f"Error in disaggregation: {str(e)}")
            return self._fallback_prediction(observations, prediction_locations)
    
    def _apply_smart_sampling(self, nodes_df: pd.DataFrame, 
                            edges_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply smart network sampling to reduce complexity
        
        Uses centrality-based or spatial sampling based on configuration
        """
        strategy = self.mg_config.get('sampling_strategy', 'centrality')
        max_edges = self.mg_config.get('max_edges', 2000)
        
        self._log(f"Applying {strategy} sampling: {len(edges_df)} -> {max_edges} edges")
        
        if strategy == 'centrality':
            return self._centrality_sampling(nodes_df, edges_df, max_edges)
        else:
            return self._spatial_sampling(nodes_df, edges_df, max_edges)
    
    def _centrality_sampling(self, nodes_df: pd.DataFrame, 
                           edges_df: pd.DataFrame, max_edges: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Sample edges based on network centrality"""
        import networkx as nx
        
        # Create networkx graph
        G = nx.Graph()
        for _, edge in edges_df.iterrows():
            G.add_edge(edge['from'], edge['to'])
        
        # Calculate edge betweenness centrality
        try:
            centrality = nx.edge_betweenness_centrality(G, k=min(100, G.number_of_nodes()))
            
            # Sort edges by centrality
            edge_scores = []
            for _, edge in edges_df.iterrows():
                edge_tuple = (edge['from'], edge['to'])
                score = centrality.get(edge_tuple, centrality.get((edge['to'], edge['from']), 0))
                edge_scores.append(score)
            
            # Select top edges
            edges_df['centrality'] = edge_scores
            sampled_edges = edges_df.nlargest(max_edges, 'centrality').drop('centrality', axis=1)
            
        except:
            # Fallback to random sampling
            sampled_edges = edges_df.sample(n=min(max_edges, len(edges_df)))
        
        return nodes_df, sampled_edges
    
    def _spatial_sampling(self, nodes_df: pd.DataFrame, 
                         edges_df: pd.DataFrame, max_edges: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Sample edges based on spatial distribution"""
        # Calculate edge midpoints
        edge_midpoints = []
        for _, edge in edges_df.iterrows():
            from_node = nodes_df.iloc[edge['from']]
            to_node = nodes_df.iloc[edge['to']]
            midpoint = [(from_node['x'] + to_node['x'])/2, (from_node['y'] + to_node['y'])/2]
            edge_midpoints.append(midpoint)
        
        # Use k-means to find representative edges
        from sklearn.cluster import KMeans
        n_clusters = min(max_edges, len(edges_df))
        
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(edge_midpoints)
            
            # Select one edge per cluster (closest to center)
            sampled_indices = []
            for i in range(n_clusters):
                cluster_edges = np.where(clusters == i)[0]
                if len(cluster_edges) > 0:
                    # Find edge closest to cluster center
                    center = kmeans.cluster_centers_[i]
                    distances = [np.linalg.norm(np.array(edge_midpoints[j]) - center) 
                               for j in cluster_edges]
                    best_idx = cluster_edges[np.argmin(distances)]
                    sampled_indices.append(best_idx)
            
            sampled_edges = edges_df.iloc[sampled_indices]
            
        except:
            # Fallback to random sampling
            sampled_edges = edges_df.sample(n=min(max_edges, len(edges_df)))
        
        return nodes_df, sampled_edges
    
    def _fallback_prediction(self, observations: pd.DataFrame, 
                           prediction_locations: pd.DataFrame) -> pd.DataFrame:
        """
        Fallback prediction using inverse distance weighting
        
        This maintains the spatial nature of the problem when MetricGraph fails
        """
        self._log("Using inverse distance weighting fallback")
        
        n_pred = len(prediction_locations)
        obs_values = observations['svi_value'].values
        
        predictions = []
        
        for _, pred_loc in prediction_locations.iterrows():
            # Calculate distances to observations
            distances = np.sqrt(
                (observations['coord_x'] - pred_loc['x'])**2 + 
                (observations['coord_y'] - pred_loc['y'])**2
            )
            
            # Inverse distance weighting
            weights = 1 / (distances + 1e-8)
            weights /= weights.sum()
            
            pred_value = np.sum(weights * obs_values)
            pred_uncertainty = np.std(obs_values) * 0.3
            
            predictions.append({
                'mean': pred_value,
                'sd': pred_uncertainty,
                'q025': pred_value - 1.96 * pred_uncertainty,
                'q975': pred_value + 1.96 * pred_uncertainty
            })
        
        return pd.DataFrame(predictions)
    
    def get_timing_report(self) -> str:
        """Get timing statistics report"""
        if not self.timing_stats:
            return "No timing data available"
        
        report = "\nTiming Report:\n"
        report += "-" * 30 + "\n"
        total_time = sum(self.timing_stats.values())
        
        for operation, time_taken in self.timing_stats.items():
            percentage = (time_taken / total_time) * 100
            report += f"{operation}: {time_taken:.2f}s ({percentage:.1f}%)\n"
        
        report += "-" * 30 + "\n"
        report += f"Total: {total_time:.2f}s\n"
        
        return report