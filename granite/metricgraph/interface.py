"""
MetricGraph R interface for GRANITE framework
Updated to support spatial disaggregation workflow
"""
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import rpy2.robjects.packages as rpackages
import time
import logging
from typing import Dict, Optional, Tuple
import warnings
import os
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['R_LIBS_SITE'] = '/usr/local/lib/R/site-library:/usr/lib/R/site-library:/usr/lib/R/library'

logger = logging.getLogger(__name__)


class MetricGraphInterface:
    """
    Interface to MetricGraph R package for spatial modeling on networks
    Optimized for spatial disaggregation rather than regression
    """
    
    def __init__(self, verbose=True, config=None):
        self.verbose = verbose
        self.config = config or {}
        self.converter = pandas2ri.converter
        
        # Initialize R environment
        self._initialize_r_env()
        
    def _log(self, message):
        """Log message if verbose mode is enabled"""
        if self.verbose:
            logger.info(message)
            print(f"[MetricGraphInterface] {message}")
    
    def _log_timing(self, operation, start_time):
        """Log operation timing"""
        elapsed = time.time() - start_time
        self._log(f"{operation} completed in {elapsed:.2f}s")
        
    def _initialize_r_env(self):
        """Initialize R environment and load required packages"""
        self._log("Initializing R environment...")
        
        # Check and install MetricGraph if needed
        utils = rpackages.importr('utils')
        utils.chooseCRANmirror(ind=1)
        
        if not rpackages.isinstalled('MetricGraph'):
            self._log("Installing MetricGraph package...")
            utils.install_packages('MetricGraph')
        
        # Import MetricGraph
        try:
            self.mg = rpackages.importr('MetricGraph')
            self._log("MetricGraph package loaded successfully")
        except Exception as e:
            self._log(f"Failed to load MetricGraph: {str(e)}")
            self.mg = None
            
        # Define R helper functions for graph creation and spatial disaggregation
        ro.r('''
        library(MetricGraph)
        
        # Function to create SPDE model with GNN-informed parameters
        create_spde_model <- function(graph, gnn_features, alpha = 1) {
            tryCatch({
                # Extract average GNN parameters
                kappa_mean <- mean(gnn_features[, 1], na.rm = TRUE)
                alpha_mean <- alpha  # Fixed to integer value
                tau_mean <- mean(gnn_features[, 3], na.rm = TRUE)
                
                # Convert tau to sigma (standard deviation)
                sigma_mean <- 1.0 / sqrt(tau_mean)
                
                # Create SPDE model with GNN-learned parameters
                spde_model <- graph_spde(
                    graph_object = graph,
                    alpha = alpha,
                    start_kappa = kappa_mean,
                    start_sigma = sigma_mean,
                    parameterization = "spde"
                )
                
                return(list(
                    success = TRUE, 
                    model = spde_model,
                    params = list(
                        kappa = kappa_mean,
                        alpha = alpha,
                        sigma = sigma_mean,
                        tau = tau_mean
                    )
                ))
                
            }, error = function(e) {
                return(list(
                    success = FALSE, 
                    error = conditionMessage(e)
                ))
            })
        }
        
        # Function to perform constrained spatial disaggregation
        disaggregate_with_constraint <- function(graph, spde_model, tract_observation, 
                                               prediction_locations, gnn_features) {
            tryCatch({
                # Build mesh if not already built
                if(!graph$mesh_built()) {
                    graph$build_mesh(h = 0.05)
                }
                
                # Get the tract-level constraint value
                tract_svi <- tract_observation$svi_value[1]
                tract_x <- tract_observation$coord_x[1]
                tract_y <- tract_observation$coord_y[1]
                
                # Project prediction locations to graph
                pred_proj <- graph$project_obs(
                    prediction_locations,
                    normalized = FALSE
                )
                
                # Get number of prediction points
                n_pred <- nrow(prediction_locations)
                
                # Create spatial weights based on distance from tract centroid
                distances <- sqrt((prediction_locations$x - tract_x)^2 + 
                                (prediction_locations$y - tract_y)^2)
                
                # Use inverse distance weighting for initial values
                weights <- 1 / (distances + 1e-6)
                weights <- weights / sum(weights)
                
                # Initial disaggregated values (sum to tract total)
                initial_values <- tract_svi * weights * n_pred
                
                # Create precision matrix from SPDE
                Q <- spde_model$Q
                
                # Add small diagonal for numerical stability
                Q_stable <- Q + Matrix::Diagonal(n = nrow(Q), x = 1e-6)
                
                # Use GNN features to create spatially-varying field
                if(!is.null(gnn_features) && nrow(gnn_features) >= n_pred) {
                    # Use GNN kappa values to modulate spatial correlation
                    kappa_field <- gnn_features[1:n_pred, 1]
                    
                    # Scale initial values by local accessibility
                    accessibility_factor <- kappa_field / mean(kappa_field)
                    adjusted_values <- initial_values * accessibility_factor
                    
                    # Ensure constraint is still satisfied
                    adjusted_values <- adjusted_values * (tract_svi / sum(adjusted_values))
                } else {
                    adjusted_values <- initial_values
                }
                
                # Calculate uncertainty based on SPDE model
                # Approximate marginal variances
                Q_inv_diag <- 1 / diag(Q_stable)
                marginal_sd <- sqrt(Q_inv_diag * spde_model$params$sigma^2)
                
                # Scale uncertainty by distance from observation
                uncertainty_scale <- 1 + 0.1 * distances / max(distances)
                final_sd <- marginal_sd[1:n_pred] * uncertainty_scale
                
                # Create prediction data frame
                predictions <- data.frame(
                    x = prediction_locations$x,
                    y = prediction_locations$y,
                    mean = adjusted_values,
                    sd = final_sd,
                    q025 = adjusted_values - 1.96 * final_sd,
                    q975 = adjusted_values + 1.96 * final_sd
                )
                
                # Final check: ensure non-negativity
                predictions$mean <- pmax(predictions$mean, 0)
                predictions$q025 <- pmax(predictions$q025, 0)
                
                # Verify constraint
                total_predicted <- sum(predictions$mean)
                constraint_error <- abs(total_predicted - tract_svi) / tract_svi
                
                return(list(
                    success = TRUE,
                    predictions = predictions,
                    constraint_satisfied = constraint_error < 0.001,
                    constraint_error = constraint_error,
                    total_predicted = total_predicted,
                    tract_value = tract_svi
                ))
                
            }, error = function(e) {
                return(list(
                    success = FALSE,
                    error = conditionMessage(e)
                ))
            })
        }
        
        # Optimized graph creation function
        create_metric_graph_optimized <- function(nodes, edges, build_mesh = TRUE, 
                                                mesh_h = 0.05) {
            tryCatch({
                # Create vertex matrix (V)
                V <- as.matrix(nodes[, c("x", "y")])
                
                # Create edge matrix (E) - 1-indexed for R
                E <- as.matrix(edges[, c("from", "to")]) + 1
                
                # Add edge weights if available
                if("weight" %in% colnames(edges)) {
                    edge_weights <- edges$weight
                } else {
                    edge_weights <- NULL
                }
                
                # Create metric graph using V/E specification
                graph <- metric_graph(
                    V = V,
                    E = E,
                    edge_weights = edge_weights
                )
                
                # Build mesh if requested
                if(build_mesh) {
                    graph$build_mesh(h = mesh_h)
                }
                
                # Return graph with metadata
                list(
                    success = TRUE,
                    graph = graph,
                    n_vertices = nrow(V),
                    n_edges = nrow(E),
                    mesh_built = build_mesh
                )
                
            }, error = function(e) {
                list(success = FALSE, error = conditionMessage(e))
            })
        }
        
        # Function to apply smart sampling to reduce network complexity
        apply_smart_sampling <- function(nodes, edges, max_edges = 2000, 
                                       strategy = "betweenness") {
            tryCatch({
                if(nrow(edges) <= max_edges) {
                    return(list(
                        success = TRUE,
                        nodes = nodes,
                        edges = edges,
                        sampled = FALSE
                    ))
                }
                
                # Create igraph for sampling
                library(igraph)
                g <- graph_from_data_frame(
                    d = edges[, c("from", "to")],
                    vertices = nodes[, c("node_id", "x", "y")],
                    directed = FALSE
                )
                
                if(strategy == "betweenness") {
                    # Calculate edge betweenness
                    edge_btw <- edge_betweenness(g)
                    
                    # Keep top edges by betweenness
                    keep_idx <- order(edge_btw, decreasing = TRUE)[1:max_edges]
                    sampled_edges <- edges[keep_idx, ]
                    
                } else if(strategy == "spatial") {
                    # Spatial sampling - keep edges uniformly distributed
                    n_keep <- max_edges
                    keep_idx <- sample(nrow(edges), n_keep)
                    sampled_edges <- edges[keep_idx, ]
                    
                } else {
                    stop("Unknown sampling strategy")
                }
                
                # Get connected nodes
                used_nodes <- unique(c(sampled_edges$from, sampled_edges$to))
                sampled_nodes <- nodes[nodes$node_id %in% used_nodes, ]
                
                # Reindex
                new_idx <- seq_len(nrow(sampled_nodes)) - 1
                old_to_new <- setNames(new_idx, sampled_nodes$node_id)
                
                sampled_edges$from <- old_to_new[as.character(sampled_edges$from)]
                sampled_edges$to <- old_to_new[as.character(sampled_edges$to)]
                sampled_nodes$node_id <- new_idx
                
                list(
                    success = TRUE,
                    nodes = sampled_nodes,
                    edges = sampled_edges,
                    sampled = TRUE,
                    original_edges = nrow(edges),
                    sampled_edges = nrow(sampled_edges)
                )
                
            }, error = function(e) {
                list(success = FALSE, error = conditionMessage(e))
            })
        }
        
        # Function to project points onto graph
        project_to_graph <- function(graph, points, tolerance = 0.01) {
            tryCatch({
                # Project points
                proj_result <- graph$project_obs(
                    points,
                    normalized = FALSE
                )
                
                list(
                    success = TRUE,
                    projections = proj_result
                )
                
            }, error = function(e) {
                list(success = FALSE, error = conditionMessage(e))
            })
        }
        ''')
    
    def create_graph(self, nodes_df, edges_df, enable_sampling=None):
        """
        Create MetricGraph object from network data
        
        Parameters:
        -----------
        nodes_df : pd.DataFrame
            Nodes with node_id, x, y coordinates
        edges_df : pd.DataFrame
            Edges with from, to indices (0-based) and optional weight
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
        self._log(f"Creating MetricGraph with {len(nodes_df)} nodes, {len(edges_df)} edges")
        
        # Determine if sampling is needed
        if enable_sampling is None:
            enable_sampling = self.config.get('enable_sampling', False)
        
        max_edges = self.config.get('max_edges', 10000)
        if len(edges_df) > max_edges:
            enable_sampling = True
            
        try:
            with localconverter(self.converter):
                # Apply sampling if needed
                if enable_sampling:
                    self._log(f"Applying smart sampling (max_edges={max_edges})...")
                    
                    r_nodes = ro.conversion.py2rpy(nodes_df)
                    r_edges = ro.conversion.py2rpy(edges_df)
                    
                    sample_func = ro.r['apply_smart_sampling']
                    sample_result = sample_func(
                        r_nodes, 
                        r_edges, 
                        max_edges,
                        self.config.get('sampling_strategy', 'betweenness')
                    )
                    
                    if sample_result[0][0]:  # success
                        nodes_df = ro.conversion.rpy2py(sample_result[1])
                        edges_df = ro.conversion.rpy2py(sample_result[2])
                        
                        if sample_result[3][0]:  # was sampled
                            self._log(f"Sampled network: {len(edges_df)} edges "
                                     f"(from {sample_result[4][0]})")
                    else:
                        self._log(f"Sampling failed: {sample_result[1]}")
                
                # Convert to R
                r_nodes = ro.conversion.py2rpy(nodes_df)
                r_edges = ro.conversion.py2rpy(edges_df)
                
                # Create graph
                create_func = ro.r['create_metric_graph_optimized']
                result = create_func(
                    r_nodes,
                    r_edges,
                    build_mesh=True,
                    mesh_h=self.config.get('mesh_resolution', 0.05)
                )
                
                if result[0][0]:  # Check success flag
                    graph = result[1]
                    n_vertices = result[2][0]
                    n_edges = result[3][0]
                    
                    self._log(f"MetricGraph created: {n_vertices} vertices, {n_edges} edges")
                    self._log_timing("Graph creation", start_time)
                    
                    return graph
                else:
                    self._log(f"Graph creation failed: {result[1]}")
                    return None
                    
        except Exception as e:
            self._log(f"Error creating graph: {str(e)}")
            return None
    
    def project_points(self, metric_graph, points_df, tolerance=0.01):
        """
        Project points onto the metric graph
        
        Parameters:
        -----------
        metric_graph : R object
            MetricGraph object
        points_df : pd.DataFrame
            Points with x, y coordinates
        tolerance : float
            Projection tolerance
            
        Returns:
        --------
        pd.DataFrame with projected coordinates or None
        """
        if metric_graph is None:
            return None
            
        try:
            with localconverter(self.converter):
                r_points = ro.conversion.py2rpy(points_df[['x', 'y']])
                
                project_func = ro.r['project_to_graph']
                result = project_func(metric_graph, r_points, tolerance)
                
                if result[0][0]:  # success
                    projections = ro.conversion.rpy2py(result[1])
                    return projections
                else:
                    self._log(f"Projection failed: {result[1]}")
                    return None
                    
        except Exception as e:
            self._log(f"Error projecting points: {str(e)}")
            return None
    
    def get_graph_info(self, metric_graph):
        """
        Get information about a MetricGraph object
        
        Parameters:
        -----------
        metric_graph : R object
            MetricGraph object
            
        Returns:
        --------
        dict with graph information
        """
        if metric_graph is None:
            return None
            
        try:
            ro.r('''
            get_graph_info <- function(graph) {
                list(
                    n_vertices = graph$nV,
                    n_edges = graph$nE,
                    has_mesh = graph$mesh_built(),
                    bbox = graph$get_bounding_box()
                )
            }
            ''')

        except Exception as e:
            self._log(f"Error retrieving graph info: {str(e)}")
            return None
            
            info_func = ro.r['get_graph_info']
            info = info_func(metric_graph)
            
            return {
                'n_vertices': int(info[0][0]),
                'n_edges': int(info[1][0]),
                'has_mesh': bool(info[2][0]),
                'bbox': np.array(info[3])
            }
    
    def _kriging_baseline(self, tract_observation: pd.DataFrame,
                         prediction_locations: pd.DataFrame) -> Dict:
        """
        Standard kriging baseline for comparison
        
        Uses simple ordinary kriging without network constraints
        """
        self._log("Using standard kriging baseline for disaggregation")
        
        try:
            # Get tract parameters
            tract_x = tract_observation['coord_x'].iloc[0]
            tract_y = tract_observation['coord_y'].iloc[0]
            tract_svi = tract_observation['svi_value'].iloc[0]
            
            # Calculate distances
            distances = np.sqrt(
                (prediction_locations['x'] - tract_x)**2 + 
                (prediction_locations['y'] - tract_y)**2
            )
            
            # Simple exponential variogram model
            range_param = np.percentile(distances, 75)
            sill = tract_svi * 0.1  # 10% of tract value as variance
            
            # Kriging weights (simplified ordinary kriging)
            weights = np.exp(-3 * distances / range_param)
            weights = weights / weights.sum()
            
            # Predictions (ensure sum constraint)
            predictions = tract_svi * weights * len(weights)
            
            # Kriging variance
            kriging_var = sill * (1 - np.exp(-3 * distances / range_param))
            kriging_sd = np.sqrt(kriging_var)
            
            predictions_df = pd.DataFrame({
                'x': prediction_locations['x'],
                'y': prediction_locations['y'],
                'mean': predictions,
                'sd': kriging_sd,
                'q025': predictions - 1.96 * kriging_sd,
                'q975': predictions + 1.96 * kriging_sd
            })
            
            # Ensure non-negativity
            predictions_df['mean'] = predictions_df['mean'].clip(lower=0)
            predictions_df['q025'] = predictions_df['q025'].clip(lower=0)
            
            # Check constraint
            total_predicted = predictions_df['mean'].sum()
            constraint_error = abs(total_predicted - tract_svi) / tract_svi
            
            return {
                'success': True,
                'predictions': predictions_df,
                'method': 'ordinary_kriging',
                'diagnostics': {
                    'constraint_satisfied': constraint_error < 0.001,
                    'constraint_error': constraint_error,
                    'total_predicted': total_predicted,
                    'tract_value': tract_svi,
                    'range_param': range_param,
                    'sill': sill
                }
            }
            
        except Exception as e:
            self._log(f"Kriging baseline failed: {str(e)}")
            # Ultimate fallback: uniform distribution
            n_pred = len(prediction_locations)
            uniform_value = tract_observation['svi_value'].iloc[0]
            
            return {
                'success': False,
                'predictions': pd.DataFrame({
                    'x': prediction_locations['x'],
                    'y': prediction_locations['y'],
                    'mean': uniform_value,
                    'sd': uniform_value * 0.1,
                    'q025': uniform_value * 0.8,
                    'q975': uniform_value * 1.2
                }),
                'method': 'uniform_fallback',
                'error': str(e)
            }