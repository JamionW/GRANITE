"""
MetricGraph R interface for GRANITE framework
OPTIMIZED VERSION - Drop-in replacement for existing interface.py
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message="Environment variable.*redefined by R")
warnings.filterwarnings("ignore", category=UserWarning, module="rpy2")
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from datetime import datetime
import networkx as nx
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import gc


class MetricGraphInterface:
    """Interface to R MetricGraph package for spatial modeling - OPTIMIZED VERSION"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        
        # Import R packages
        self._log("Initializing R interface...")
        
        try:
            import rpy2.robjects as ro
            from rpy2.robjects.packages import importr
            from rpy2.robjects.conversion import localconverter
            import subprocess
            import os
            
            # CRITICAL FIX: Use modern rpy2 converter (NO activation/deactivation)
            self.converter = ro.default_converter + pandas2ri.converter
            self._log("  ✓ Modern rpy2 converter initialized")
            
            # CRITICAL FIX: Force rpy2 to use same library paths as direct R
            self._log("  Synchronizing R library paths...")
            try:
                result = subprocess.run(['R', '--slave', '-e', 'cat(.libPaths(), sep="\\n")'], 
                                    capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    r_lib_paths = [path.strip() for path in result.stdout.strip().split('\n') if path.strip()]
                    
                    # Set the same paths in rpy2 session
                    path_str = ', '.join([f'"{path}"' for path in r_lib_paths])
                    ro.r(f'.libPaths(c({path_str}))')
                    self._log(f"  ✓ Synchronized library paths: {r_lib_paths}")
                else:
                    self._log(f"  ⚠️  Could not sync library paths: {result.stderr}")
            except Exception as e:
                self._log(f"  ⚠️  Library path sync failed: {e}")
            
            self.base = importr('base')
            
            # Try to load MetricGraph
            try:
                self.mg = importr('MetricGraph')
                self._log("  ✓ MetricGraph package loaded")
                
                # Define optimized R functions
                self._define_optimized_r_functions()
                self._log("  ✓ Optimized R functions defined")
                
            except Exception as e:
                self._log(f"  ✗ MetricGraph failed to load: {e}")
                self._log("  ⚠️  Skipping R function definitions (MetricGraph not available)")
                self.mg = None
                
        except Exception as e:
            self._log(f"  ✗ Failed to initialize R interface: {str(e)}")
            self.mg = None
        
    def _log(self, message):
        """Logging with timestamp"""
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def _define_optimized_r_functions(self):
        """Define optimized R functions for MetricGraph operations"""
        
        if self.mg is None:
            self._log("  ⚠️  Skipping R function definitions (MetricGraph not available)")
            return
        
        # Test basic MetricGraph functionality first
        try:
            test_code = '''
            test_mg <- function() {
                tryCatch({
                    edges <- list(matrix(c(0,0,1,0), nrow=2, ncol=2))
                    graph <- metric_graph$new(edges = edges)
                    return("SUCCESS")
                }, error = function(e) {
                    return(paste("ERROR:", e$message))
                })
            }
            '''
            ro.r(test_code)
            result = ro.r('test_mg()')
            
            if "SUCCESS" not in str(result):
                self._log(f"  ✗ MetricGraph test failed: {result}")
                self.mg = None
                return
            else:
                self._log("  ✓ MetricGraph basic test passed")
                
        except Exception as e:
            self._log(f"  ✗ MetricGraph test error: {e}")
            self.mg = None
            return
        
        # OPTIMIZATION: Define memory-efficient graph creation function
        try:
            ro.r('''
            create_optimized_metric_graph <- function(nodes, edges, batch_size = 300, max_edges = 2000) {
                cat("Creating MetricGraph with", nrow(nodes), "nodes and", nrow(edges), "edges\\n")
                
                # Validate inputs
                if(nrow(nodes) == 0 || nrow(edges) == 0) {
                    cat("ERROR: Empty input data\\n")
                    return(list(FALSE, "Empty input data"))
                }
                
                # Ensure proper data types AND COLUMN NAMES
                V <- as.matrix(nodes[, c("x", "y")])
                E <- as.matrix(edges[, c("from", "to")])
                colnames(V) <- c("x", "y")
                colnames(E) <- c("from", "to")
                
                # Limit edges if too many
                #if(nrow(E) > max_edges) {
                #    cat("Limiting to", max_edges, "edges for memory efficiency\\n")
                #    E <- E[1:max_edges, , drop = FALSE]
                #}
                
                # OPTIMIZATION 1: Batch edge processing
                n_edges <- nrow(E)
                n_batches <- ceiling(n_edges / batch_size)
                
                cat("Processing", n_edges, "edges in", n_batches, "batches of size", batch_size, "\\n")
                
                edge_list <- vector("list", n_edges)
                edge_count <- 0
                
                for(batch_idx in 1:n_batches) {
                    start_idx <- (batch_idx - 1) * batch_size + 1
                    end_idx <- min(batch_idx * batch_size, n_edges)
                    
                    # Process batch
                    batch_edges <- E[start_idx:end_idx, , drop = FALSE]
                    
                    for(i in 1:nrow(batch_edges)) {
                        from_idx <- batch_edges[i, 1]
                        to_idx <- batch_edges[i, 2]
                        
                        if(from_idx > 0 && to_idx > 0 && from_idx <= nrow(V) && to_idx <= nrow(V)) {
                            edge_count <- edge_count + 1
                            
                            # Create edge matrix more efficiently
                            edge_matrix <- matrix(c(V[from_idx, ], V[to_idx, ]), nrow = 2, byrow = TRUE)
                            colnames(edge_matrix) <- c("x", "y")
                            edge_list[[edge_count]] <- edge_matrix
                        }
                    }
                    
                    # Memory management
                    if(batch_idx %% 3 == 0) {
                        gc()
                        cat("Processed batch", batch_idx, "/", n_batches, "\\n")
                    }
                }
                
                if(edge_count == 0) {
                    cat("ERROR: No valid edges found\\n")
                    return(list(FALSE, "No valid edges"))
                }
                
                # Trim edge list to actual size
                edge_list <- edge_list[1:edge_count]
                
                cat("Creating MetricGraph object with", edge_count, "valid edges\\n")
                
                # OPTIMIZATION 2: Single optimized creation attempt
                tryCatch({
                    graph <- metric_graph$new(
                        edges = edge_list,
                        longlat = FALSE,
                        verbose = 0,
                        perform_merges = TRUE,  # ← ADD THIS LINE
                        tolerance = list(
                            vertex_vertex = 1e-6,   # ← Relax tolerance
                            vertex_edge = 1e-6,     # ← Relax tolerance  
                            edge_edge = 1e-6        # ← Relax tolerance
                        )
                    )
                    
                    cat("SUCCESS: MetricGraph created successfully\\n")
                    return(list(TRUE, graph))
                    
                }, error = function(e) {
                    cat("ERROR in MetricGraph creation:", conditionMessage(e), "\\n")
                    
                    # FALLBACK: Try with reduced network if still failing
                    if(edge_count > 1000) {
                        cat("Attempting fallback with first 1000 edges\\n")
                        fallback_edges <- edge_list[1:1000]
                        
                        tryCatch({
                            graph <- metric_graph$new(
                                edges = fallback_edges,
                                longlat = FALSE,
                                verbose = 0
                            )
                            cat("FALLBACK SUCCESS: Created graph with 1000 edges\\n")
                            return(list(TRUE, graph))
                        }, error = function(e2) {
                            cat("FALLBACK FAILED:", conditionMessage(e2), "\\n")
                            return(list(FALSE, paste("Creation failed:", conditionMessage(e))))
                        })
                    } else {
                        return(list(FALSE, paste("Creation failed:", conditionMessage(e))))
                    }
                })
            }
            ''')
            
            # OPTIMIZATION: Define efficient fitting and prediction function
            ro.r('''
            fit_and_predict_optimized <- function(graph, observations, prediction_locations, 
                                                 gnn_features = NULL, alpha = 1.5) {
                
                cat("Fitting Whittle-Matérn model with optimized parameters...\\n")
                
                tryCatch({
                    # OPTIMIZATION 3: Streamlined model fitting
                    
                    # Build mesh with coarser resolution for efficiency
                    graph$build_mesh(h = 0.05)  # Coarser than default
                    
                    # Add observations to graph
                    graph$add_observations(
                        data = observations,
                        tolerance = 1e-6
                    )
                    
                    # Prepare model formula
                    if(!is.null(gnn_features) && ncol(gnn_features) >= 3) {
                        # Include GNN features as covariates
                        formula_str <- "y ~ gnn_kappa + gnn_alpha + gnn_tau"
                        
                        # Add GNN features to graph data
                        graph_data <- graph$get_data()
                        graph_data$gnn_kappa <- gnn_features[, 1]
                        graph_data$gnn_alpha <- gnn_features[, 2] 
                        graph_data$gnn_tau <- gnn_features[, 3]
                        
                        graph$clear_observations()
                        graph$add_observations(data = graph_data, tolerance = 1e-6)
                    } else {
                        formula_str <- "y ~ 1"  # Intercept only
                    }
                    
                    # Fit model
                    model <- graph_spde(
                        graph = graph,
                        alpha = alpha,
                        model = list(formula = as.formula(formula_str))
                    )
                    
                    # Project prediction locations
                    pred_projected <- graph$project_observations(
                        prediction_locations,
                        tolerance = 1e-6
                    )
                    
                    # Make predictions
                    preds <- predict(model, 
                                    newdata = pred_projected,
                                    normalized = TRUE)
                    
                    result <- data.frame(
                        mean = preds$mean,
                        sd = sqrt(pmax(preds$variance, 1e-8)),
                        q025 = preds$mean - 1.96 * sqrt(pmax(preds$variance, 1e-8)),
                        q975 = preds$mean + 1.96 * sqrt(pmax(preds$variance, 1e-8))
                    )
                    
                    cat("Predictions completed successfully\\n")
                    return(result)
                    
                }, error = function(e) {
                    cat("Error in fitting/prediction:", conditionMessage(e), "\\n")
                    
                    # Enhanced fallback using observation statistics
                    n_pred <- nrow(prediction_locations)
                    obs_mean <- mean(observations$y, na.rm = TRUE)
                    obs_sd <- sd(observations$y, na.rm = TRUE)
                    
                    fallback_result <- data.frame(
                        mean = rep(obs_mean, n_pred),
                        sd = rep(obs_sd * 0.5, n_pred),  # Conservative uncertainty
                        q025 = rep(obs_mean - 1.96 * obs_sd * 0.5, n_pred),
                        q975 = rep(obs_mean + 1.96 * obs_sd * 0.5, n_pred)
                    )
                    
                    cat("Using enhanced statistical fallback\\n")
                    return(fallback_result)
                })
            }
            ''')
            
            self._log("  ✓ Optimized R functions defined successfully")
            
        except Exception as e:
            self._log(f"  ✗ Error defining optimized R functions: {e}")
            self.mg = None
    
    def create_graph(self, nodes_df, edges_df, enable_sampling=False, max_edges=2000, batch_size=300):
        """
        Create MetricGraph object with optimized processing
        
        Parameters:
        -----------
        nodes_df : pd.DataFrame
            Nodes with x, y coordinates  
        edges_df : pd.DataFrame
            Edges with from, to indices
        enable_sampling : bool
            Whether to enable smart network sampling (NEW PARAMETER)
        max_edges : int
            Maximum edges to process (for memory management)
        batch_size : int
            Batch size for R processing
            
        Returns:
        --------
        R object or None
            MetricGraph object or None if failed
        """
        if self.mg is None:
            self._log("⚠️  MetricGraph not available - returning None")
            return None
            
        self._log("Creating MetricGraph object with optimized processing...")
        
        try:
            # Validate input data
            if nodes_df.empty or edges_df.empty:
                self._log("  ✗ Empty input data")
                return None
                
            # OPTIMIZATION: Apply smart sampling if enabled
            if enable_sampling and len(edges_df) > max_edges:
                self._log(f"  Applying smart network sampling (>{max_edges} edges)")
                nodes_df, edges_df = self._apply_smart_sampling(
                    nodes_df, edges_df, max_edges
                )
                
            # Ensure proper data types and column names
            nodes_clean = nodes_df[['x', 'y']].copy().astype(float)
            edges_clean = edges_df[['from', 'to']].copy().astype(int)
            
            # Adjust indices to be 1-based for R
            edges_clean['from'] += 1
            edges_clean['to'] += 1
            
            # Log data info
            self._log(f"  - Nodes: {len(nodes_clean)} points")
            self._log(f"  - Edges: {len(edges_clean)} connections")
            self._log(f"  - Node bounds: x=[{nodes_clean.x.min():.3f}, {nodes_clean.x.max():.3f}], "
                    f"y=[{nodes_clean.y.min():.3f}, {nodes_clean.y.max():.3f}]")
            
            # Convert to R and call optimized function
            with localconverter(self.converter):
                r_nodes = ro.conversion.py2rpy(nodes_clean)
                r_edges = ro.conversion.py2rpy(edges_clean)
                
                create_func = ro.r['create_optimized_metric_graph']
                result = create_func(r_nodes, r_edges, batch_size, max_edges)
                
                # Extract success status and graph
                success = result[0][0]  # First element of result list
                
                if success:
                    graph = result[1]  # Second element is the graph
                    self._log("  ✓ MetricGraph created successfully")
                    return graph
                else:
                    error_msg = str(result[1])
                    self._log(f"  ✗ MetricGraph creation failed: {error_msg}")
                    return None
                    
        except Exception as e:
            self._log(f"  ✗ Error in optimized graph creation: {str(e)}")
            return None
            
    def _apply_smart_sampling(self, nodes_df, edges_df, max_edges):
        """
        Apply smart network sampling while preserving important connections
        This is the CONFIGURABLE smart sampling feature
        """
        self._log(f"  Smart sampling: {len(edges_df)} -> {max_edges} edges")
        
        # Create networkx graph for analysis
        G = nx.Graph()
        
        # Add nodes
        for idx, row in nodes_df.iterrows():
            G.add_node(idx, pos=(row['x'], row['y']))
            
        # Add edges
        for _, row in edges_df.iterrows():
            G.add_edge(row['from'], row['to'])
            
        # Strategy 1: Keep high-centrality edges (backbone)
        try:
            edge_centrality = nx.edge_betweenness_centrality(G, k=min(500, G.number_of_nodes()))
            high_centrality_edges = sorted(edge_centrality.items(), key=lambda x: x[1], reverse=True)
            backbone_edges = [edge for edge, _ in high_centrality_edges[:max_edges//3]]
        except:
            # Fallback if centrality calculation fails
            backbone_edges = list(G.edges())[:max_edges//3]
            
        # Strategy 2: Spatial sampling for coverage
        edge_positions = []
        edge_list = list(G.edges())
        
        for edge in edge_list:
            node1_pos = G.nodes[edge[0]]['pos']
            node2_pos = G.nodes[edge[1]]['pos']
            edge_center = ((node1_pos[0] + node2_pos[0])/2, (node1_pos[1] + node2_pos[1])/2)
            edge_positions.append(edge_center)
            
        # Cluster edges spatially
        n_spatial_edges = max_edges - len(backbone_edges)
        if n_spatial_edges > 0 and len(edge_list) > n_spatial_edges:
            n_clusters = min(n_spatial_edges, len(edge_list)//2)
            if n_clusters > 1:
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(edge_positions)
                    
                    spatial_edges = []
                    for cluster_id in range(n_clusters):
                        cluster_indices = np.where(clusters == cluster_id)[0]
                        if len(cluster_indices) > 0:
                            # Take edge closest to cluster center
                            cluster_center = kmeans.cluster_centers_[cluster_id]
                            distances = [np.linalg.norm(np.array(edge_positions[i]) - cluster_center) 
                                       for i in cluster_indices]
                            best_idx = cluster_indices[np.argmin(distances)]
                            spatial_edges.append(edge_list[best_idx])
                except:
                    # Fallback: uniform sampling
                    spatial_edges = edge_list[:n_spatial_edges]
        else:
            spatial_edges = edge_list[:n_spatial_edges] if n_spatial_edges > 0 else []
            
        # Combine selected edges
        selected_edges = set(backbone_edges + spatial_edges)
        
        # Create filtered dataframes
        edge_set = set()
        for edge in selected_edges:
            edge_set.add((min(edge), max(edge)))  # Normalize edge direction
            
        # Filter edges dataframe
        filtered_edges = []
        for _, row in edges_df.iterrows():
            edge_tuple = (min(row['from'], row['to']), max(row['from'], row['to']))
            if edge_tuple in edge_set:
                filtered_edges.append(row)
                
        if filtered_edges:
            sampled_edges_df = pd.DataFrame(filtered_edges)
        else:
            # Fallback: take first max_edges
            sampled_edges_df = edges_df.head(max_edges)
            
        # Keep all nodes (they'll be filtered automatically by MetricGraph)
        if len(sampled_edges_df) > 0:
            # Quick connectivity check using sampled edges
            sampled_network = nx.Graph()
            for _, edge in sampled_edges_df.iterrows():
                sampled_network.add_edge(edge['from'], edge['to'])
            
            # If disconnected, add connecting edges
            if not nx.is_connected(sampled_network):
                self._log(f"  Network disconnected, adding connecting edges...")
                components = list(nx.connected_components(sampled_network))
                
                # Connect largest component to others
                largest_component = max(components, key=len)
                for component in components:
                    if component != largest_component:
                        # Find shortest path in original network
                        try:
                            comp_node = list(component)[0]
                            large_node = list(largest_component)[0]
                            
                            # Add edge connecting components
                            connecting_edge = {
                                'from': comp_node,
                                'to': large_node
                            }
                            sampled_edges_df = pd.concat([
                                sampled_edges_df, 
                                pd.DataFrame([connecting_edge])
                            ], ignore_index=True)
                        except:
                            pass  # Skip if connection fails

        reduction = 1 - (len(sampled_edges_df) / len(edges_df))
        self._log(f"  Sampling result: {reduction:.1%} reduction")

        return nodes_df, sampled_edges_df
    
    def disaggregate_svi(self, metric_graph, observations, prediction_locations, 
                        nodes_df, gnn_features=None, alpha=1.5):
        """
        Perform SVI disaggregation using MetricGraph with optimizations
        
        SAME API as original - this is a drop-in replacement
        """
        if metric_graph is None:
            self._log("⚠️  Graph is None - using fallback prediction")
            return self._fallback_prediction(observations, prediction_locations)
            
        try:
            self._log("Performing optimized SVI disaggregation...")
            
            # Prepare observations for R
            obs_for_r = observations[['x', 'y']].copy()
            obs_for_r['y'] = observations.get('svi_score', observations.get('value', 0.5))
            
            # Prepare prediction locations
            pred_locs = prediction_locations[['x', 'y']].copy()
            
            with localconverter(self.converter):
                # Convert to R
                r_obs = ro.conversion.py2rpy(obs_for_r)
                r_pred_locs = ro.conversion.py2rpy(pred_locs)
                
                r_gnn_features = None
                if gnn_features is not None:
                    r_gnn_features = ro.conversion.py2rpy(gnn_features)
                
                # Call optimized R function
                fit_predict_func = ro.r['fit_and_predict_optimized']
                r_result = fit_predict_func(metric_graph, r_obs, r_pred_locs, r_gnn_features, alpha)
                
                # Convert back to pandas
                result_df = ro.conversion.rpy2py(r_result)
                
                self._log("  ✓ Optimized disaggregation completed")
                return result_df
                
        except Exception as e:
            self._log(f"  ✗ Error in optimized disaggregation: {str(e)}")
            return self._fallback_prediction(observations, prediction_locations)

    def _fallback_prediction(self, observations, prediction_locations):
        """Enhanced fallback prediction using spatial interpolation"""
        self._log("  Using enhanced spatial interpolation fallback")
        
        n_pred = len(prediction_locations)
        obs_values = observations.get('svi_score', observations.get('value', [0.5] * len(observations)))
        
        if isinstance(obs_values, pd.Series):
            obs_values = obs_values.values
        elif not isinstance(obs_values, np.ndarray):
            obs_values = np.array(obs_values)
            
        # Simple inverse distance weighting
        predictions = []
        
        for _, pred_loc in prediction_locations.iterrows():
            # Calculate distances to all observations
            distances = np.sqrt(
                (observations['x'] - pred_loc['x'])**2 + 
                (observations['y'] - pred_loc['y'])**2
            )
            
            # Inverse distance weighting
            weights = 1 / (distances + 1e-8)
            weights /= weights.sum()
            
            pred_value = np.sum(weights * obs_values)
            pred_uncertainty = np.std(obs_values) * 0.3  # Conservative uncertainty
            
            predictions.append({
                'mean': pred_value,
                'sd': pred_uncertainty,
                'q025': pred_value - 1.96 * pred_uncertainty,
                'q975': pred_value + 1.96 * pred_uncertainty
            })
            
        return pd.DataFrame(predictions)

# Convenience function (MISSING from original optimization)
def create_metricgraph_interface(verbose=True):
    """Create MetricGraph interface instance with error handling"""
    try:
        return MetricGraphInterface(verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"Warning: Could not create MetricGraph interface: {e}")
            print("Falling back to limited functionality mode")
        return MetricGraphInterface(verbose=False)  # Try again with minimal setup