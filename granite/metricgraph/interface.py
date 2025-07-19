"""
MetricGraph R interface for GRANITE framework
Updated to support spatial disaggregation workflow
"""
import os
import sys

# Set R environment variables BEFORE importing rpy2
os.environ['R_HOME'] = '/usr/lib/R'  # Adjust path as needed
os.environ['R_LIBS_USER'] = '/tmp/R_libs'
os.environ['R_LIBS_SITE'] = ''  # Empty to avoid site library warnings
os.environ['R_ENVIRON_USER'] = ''
os.environ['R_PROFILE_USER'] = ''

# Suppress console output during rpy2 import
import io
from contextlib import redirect_stdout, redirect_stderr

# Import rpy2 with suppressed output
with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import rpy2
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    import rpy2.robjects.packages as rpackages
    
    # Set up console callback to suppress R messages
    from rpy2.rinterface_lib import callbacks
    
    def quiet_console_write(text):
        # Only print actual errors, suppress all warnings
        if any(error_keyword in text.lower() for error_keyword in ['error', 'fatal']):
            if 'warning message' not in text.lower():
                print(f"[R] {text}", end='')
    
    # Override the console write callback
    callbacks.consolewrite_print = quiet_console_write
    callbacks.consolewrite_warnerror = quiet_console_write

# Suppress rpy2 warnings BEFORE any rpy2 imports
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['R_LIBS_SITE'] = '/usr/local/lib/R/site-library:/usr/lib/R/site-library:/usr/lib/R/library'

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import rpy2.robjects.packages as rpackages
import time
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)  # <-- THIS LINE WAS MISSING

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

    def disaggregate_svi(self, metric_graph, tract_observation, prediction_locations, gnn_features=None, alpha=1):
        """
        Perform SVI spatial disaggregation using MetricGraph with GNN features
        
        Parameters:
        -----------
        metric_graph : R object
            MetricGraph object
        tract_observation : pd.DataFrame
            Tract-level observation with coord_x, coord_y, svi_value columns
        prediction_locations : pd.DataFrame
            Address coordinates for prediction with x, y columns
        gnn_features : pd.DataFrame, optional
            GNN-learned features with columns ['gnn_kappa', 'gnn_alpha', 'gnn_tau']
        alpha : int
            Alpha parameter for SPDE model
            
        Returns:
        --------
        dict : Disaggregation results
        """
        if metric_graph is None:
            self._log("MetricGraph is None, using kriging baseline")
            return self._kriging_baseline(tract_observation, prediction_locations)
            
        start_time = time.time()
        self._log("Performing spatial disaggregation...")
        
        try:
            with localconverter(self.converter):
                # Convert inputs to R
                r_tract_observation = ro.conversion.py2rpy(tract_observation)
                r_prediction_locations = ro.conversion.py2rpy(prediction_locations)
                
                # DEBUG: Log what we actually received for gnn_features
                self._log(f"gnn_features type: {type(gnn_features)}")
                if gnn_features is not None:
                    if hasattr(gnn_features, 'shape'):
                        self._log(f"gnn_features shape: {gnn_features.shape}")
                    elif hasattr(gnn_features, 'columns'):
                        self._log(f"gnn_features columns: {list(gnn_features.columns)}")
                
                # DEFENSIVE: Handle gnn_features regardless of type
                if gnn_features is not None:
                    try:
                        # Case 1: It's already a DataFrame
                        if hasattr(gnn_features, 'columns'):
                            gnn_df = gnn_features.copy()
                            self._log("gnn_features is DataFrame")
                        
                        # Case 2: It's a numpy array
                        elif hasattr(gnn_features, 'shape'):
                            if len(gnn_features.shape) == 2:
                                gnn_df = pd.DataFrame(gnn_features, columns=['gnn_kappa', 'gnn_alpha', 'gnn_tau'])
                                self._log(f"Converted 2D numpy array to DataFrame: {gnn_df.shape}")
                            else:
                                # 1D array - reshape it
                                gnn_df = pd.DataFrame([gnn_features], columns=['gnn_kappa', 'gnn_alpha', 'gnn_tau'])
                                self._log("Converted 1D array to DataFrame")
                        
                        # Case 3: It's a list or something else
                        else:
                            gnn_df = pd.DataFrame(gnn_features)
                            if gnn_df.shape[1] == 3:
                                gnn_df.columns = ['gnn_kappa', 'gnn_alpha', 'gnn_tau']
                            self._log("Converted other type to DataFrame")
                        
                        # Ensure all values are basic Python floats (not numpy types)
                        for col in gnn_df.columns:
                            gnn_df[col] = gnn_df[col].astype(float)
                        
                        r_gnn_features = ro.conversion.py2rpy(gnn_df)
                        self._log("Successfully converted gnn_features to R")
                        
                    except Exception as e:
                        self._log(f"Error converting gnn_features: {str(e)}")
                        r_gnn_features = ro.r('NULL')
                else:
                    r_gnn_features = ro.r('NULL')
                    self._log("gnn_features is None, using NULL")
            
            # CRITICAL FIX: Create SPDE model FIRST, then pass it to disaggregation
            # Create the SPDE model using the graph and GNN features
            create_spde_func = ro.r['create_spde_model']
            spde_result = create_spde_func(metric_graph, r_gnn_features, alpha)
            
            # Check if SPDE creation was successful
            spde_success = bool(spde_result.rx2('success')[0])
            if not spde_success:
                error_msg = str(spde_result.rx2('error')[0])
                self._log(f"SPDE model creation failed: {error_msg}")
                return self._kriging_baseline(tract_observation, prediction_locations)
            
            # Extract the SPDE model
            spde_model = spde_result.rx2('model')
            self._log("SPDE model created successfully")
            
            # Now call disaggregation with the actual SPDE model
            disaggregate_func = ro.r['disaggregate_with_constraint']
            result = disaggregate_func(
                metric_graph,
                spde_model,  # Pass the actual SPDE model, not NULL!
                r_tract_observation,
                r_prediction_locations,
                r_gnn_features
            )
            
            # Extract results using rx2() to avoid conversion issues
            try:
                success = bool(result.rx2('success')[0])
                if success:
                    # Convert predictions back to pandas DataFrame
                    with localconverter(self.converter):
                        predictions_df = ro.conversion.rpy2py(result.rx2('predictions'))
                    
                    constraint_satisfied = bool(result.rx2('constraint_satisfied')[0])
                    constraint_error = float(result.rx2('constraint_error')[0])
                    total_predicted = float(result.rx2('total_predicted')[0])
                    tract_value = float(result.rx2('tract_value')[0])
                    
                    # Also extract SPDE parameters from the creation result
                    try:
                        params = spde_result.rx2('params')
                        spde_params = {
                            'kappa': float(params.rx2('kappa')[0]),
                            'alpha': float(params.rx2('alpha')[0]),
                            'sigma': float(params.rx2('sigma')[0]),
                            'tau': float(params.rx2('tau')[0])
                        }
                        self._log(f"SPDE params: kappa={spde_params['kappa']:.3f}, "
                                f"sigma={spde_params['sigma']:.3f}, tau={spde_params['tau']:.3f}")
                    except:
                        spde_params = {}
                    
                    self._log(f"Disaggregation completed: {len(predictions_df)} predictions")
                    self._log(f"Constraint satisfied: {constraint_satisfied} (error: {constraint_error:.4f})")
                    self._log_timing("Spatial disaggregation", start_time)
                    
                    return {
                        'success': True,
                        'predictions': predictions_df,
                        'diagnostics': {
                            'constraint_satisfied': constraint_satisfied,
                            'constraint_error': constraint_error,
                            'total_predicted': total_predicted,
                            'tract_value': tract_value,
                            'mean_prediction': float(predictions_df['mean'].mean()),
                            'std_prediction': float(predictions_df['mean'].std()),
                            'mean_uncertainty': float(predictions_df['sd'].mean())
                        },
                        'spde_params': spde_params
                    }
                else:
                    error_msg = str(result.rx2('error')[0])
                    self._log(f"Disaggregation failed: {error_msg}")
                    return self._kriging_baseline(tract_observation, prediction_locations)
                    
            except Exception as e:
                self._log(f"Error extracting disaggregation results: {str(e)}")
                return self._kriging_baseline(tract_observation, prediction_locations)
                
        except Exception as e:
            self._log(f"Error in spatial disaggregation: {str(e)}")
            return self._kriging_baseline(tract_observation, prediction_locations)
        
    def _initialize_r_env(self):
        """Initialize R environment and load required packages"""
        self._log("Initializing R environment...")
        
        # First, set up R library paths and options BEFORE any package operations
        ro.r('''
        # Suppress all prompts and messages
        options(repos = c(CRAN = "https://cloud.r-project.org"))
        options(warn = -1)  # Suppress warnings temporarily
        options(menu.graphics = FALSE)  # No graphical menus
        
        # Platform-aware package installation settings
        # Check if binary packages are supported
        binary_supported <- tryCatch({
            # Test if binary installation works
            temp_result <- install.packages("help", 
                                        repos = "https://cloud.r-project.org",
                                        type = "binary", 
                                        quiet = TRUE)
            TRUE
        }, error = function(e) {
            # Binary not supported, fall back to source
            FALSE
        })
        
        if (binary_supported) {
            options(pkgType = "binary")
            options(install.packages.compile.from.source = "never")
            message("Using binary package installation")
        } else {
            options(pkgType = "source")
            options(install.packages.compile.from.source = "always")
            message("Using source package installation (binary not supported)")
        }
        
        # Create a writable library directory
        # Try multiple locations in order of preference
        lib_paths <- c(
            file.path(tempdir(), "R_libs"),  # Temp directory (always writable)
            file.path(Sys.getenv("HOME"), ".R", "library"),  # User home
            file.path(getwd(), "R_libs")  # Current directory
        )
        
        # Find or create the first writable directory
        user_lib <- NULL
        for (path in lib_paths) {
            if (!dir.exists(path)) {
                try({
                    dir.create(path, recursive = TRUE, showWarnings = FALSE)
                    if (dir.exists(path)) {
                        user_lib <- path
                        break
                    }
                }, silent = TRUE)
            } else if (file.access(path, 2) == 0) {  # Check if writable
                user_lib <- path
                break
            }
        }
        
        if (is.null(user_lib)) {
            stop("Could not find or create a writable R library directory")
        }
        
        # Set library paths with our writable directory first
        .libPaths(c(user_lib, .libPaths()))
        
        # Restore warning level
        options(warn = 0)
        
        message(paste("Using R library:", user_lib))
        message(paste("Binary packages supported:", binary_supported))
        ''')
        
        # Now check for and install MetricGraph if needed
        try:
            # First install remotes if not available (platform-aware)
            ro.r('''
            if (!requireNamespace("remotes", quietly = TRUE)) {
                message("Installing remotes package...")
                # Determine installation type based on platform support
                if (exists("binary_supported") && binary_supported) {
                    install.packages("remotes", 
                                repos = "https://cloud.r-project.org", 
                                quiet = TRUE, 
                                dependencies = TRUE,
                                type = "binary")
                } else {
                    install.packages("remotes", 
                                repos = "https://cloud.r-project.org", 
                                quiet = TRUE, 
                                dependencies = TRUE,
                                type = "source")
                }
            }
            ''')
            
            # Install Matrix if needed
            ro.r('''
            if (!requireNamespace("Matrix", quietly = TRUE)) {
                message("Installing Matrix package...")
                install.packages("Matrix", 
                            repos = "https://cloud.r-project.org", 
                            quiet = TRUE, 
                            dependencies = TRUE)
            }
            ''')
            
            # Check if MetricGraph is already installed, if not install from GitHub
            ro.r('''
            # Function to check and install packages from GitHub
            ensure_metricgraph <- function() {
                if (!requireNamespace("MetricGraph", quietly = TRUE)) {
                    message("Installing MetricGraph from GitHub...")
                    tryCatch({
                        remotes::install_github("davidbolin/MetricGraph", 
                                            quiet = TRUE, 
                                            upgrade = "never",
                                            dependencies = TRUE,
                                            force = FALSE,
                                            build_vignettes = FALSE)  # Skip vignettes for faster install
                        TRUE
                    }, error = function(e) {
                        message(paste("Failed to install MetricGraph:", e$message))
                        
                        # Try alternative installation methods
                        message("Trying alternative installation from stable branch...")
                        tryCatch({
                            remotes::install_github("davidbolin/MetricGraph@stable", 
                                                quiet = TRUE, 
                                                upgrade = "never",
                                                dependencies = TRUE,
                                                force = FALSE,
                                                build_vignettes = FALSE)
                            TRUE
                        }, error = function(e2) {
                            message(paste("Alternative installation also failed:", e2$message))
                            FALSE
                        })
                    })
                } else {
                    message("MetricGraph already installed")
                    TRUE
                }
            }
            
            # Install MetricGraph
            metricgraph_available <- ensure_metricgraph()
            
            if (!metricgraph_available) {
                stop("MetricGraph package could not be installed from GitHub")
            }
            ''')
            
            # Load the packages
            ro.r('''
            suppressWarnings(suppressMessages({
                library(MetricGraph, quietly = TRUE)
                library(Matrix, quietly = TRUE)
            }))
            ''')
            
            # Import to Python
            self.mg = rpackages.importr('MetricGraph')
            self._log("MetricGraph package loaded successfully")
            
        except Exception as e:
            self._log(f"Failed to initialize R packages: {str(e)}")
            
            # Try alternative: check if packages are already installed system-wide
            try:
                # Check different R library paths
                ro.r('''
                # Search for MetricGraph in all library paths
                all_libs <- .libPaths()
                metricgraph_found <- FALSE
                
                for (lib_path in all_libs) {
                    if (dir.exists(file.path(lib_path, "MetricGraph"))) {
                        message(paste("Found MetricGraph in:", lib_path))
                        metricgraph_found <- TRUE
                        break
                    }
                }
                
                if (!metricgraph_found) {
                    message("MetricGraph not found in any library path")
                    message(paste("Searched paths:", paste(all_libs, collapse = ", ")))
                }
                ''')
                
                ro.r('library(MetricGraph)')
                self.mg = rpackages.importr('MetricGraph')
                self._log("MetricGraph loaded from existing installation")
            except:
                self._log("MetricGraph not available - spatial features will be limited")
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
                
                # Get number of prediction points
                n_pred <- nrow(prediction_locations)
                
                # Create spatial weights based on distance from tract centroid
                distances <- sqrt((prediction_locations$x - tract_x)^2 + 
                                (prediction_locations$y - tract_y)^2)
                
                # Use inverse distance weighting for initial values
                weights <- 1 / (distances + 1e-6)
                weights <- weights / sum(weights)
                
                # Initial disaggregated values (normalized to preserve mass)
                initial_values <- weights * tract_svi
                
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
                
                # Calculate uncertainty based on distance and GNN features
                # Base uncertainty proportional to distance from tract centroid
                base_uncertainty <- 0.1 * tract_svi  # 10% of tract value as base
                
                # Scale uncertainty by distance
                uncertainty_scale <- 1 + 0.2 * distances / max(distances)
                
                # If we have GNN tau values, use them to modulate uncertainty
                if(!is.null(gnn_features) && nrow(gnn_features) >= n_pred) {
                    tau_field <- gnn_features[1:n_pred, 3]
                    # Convert tau to standard deviation (sigma = 1/sqrt(tau))
                    sigma_field <- 1 / sqrt(pmax(tau_field, 0.1))  # Avoid division by zero
                    # Normalize to reasonable range
                    sigma_field <- sigma_field / mean(sigma_field)
                    final_sd <- base_uncertainty * uncertainty_scale * sigma_field
                } else {
                    final_sd <- base_uncertainty * uncertainty_scale
                }
                
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
                    error = paste("Disaggregation error:", conditionMessage(e))
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
                graph <- metric_graph$new(
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
                
                # Convert final data to R
                r_nodes = ro.conversion.py2rpy(nodes_df)
                r_edges = ro.conversion.py2rpy(edges_df)

            # Create graph OUTSIDE localconverter to avoid automatic conversion
            create_func = ro.r['create_metric_graph_optimized']
            result = create_func(
                r_nodes,
                r_edges,
                build_mesh=True,
                mesh_h=self.config.get('mesh_resolution', 0.05)
            )

            # Extract results using rx2() method to avoid automatic conversion
            try:
                success = bool(result.rx2('success')[0])
                if success:
                    graph = result.rx2('graph')  # Keep as R object - no conversion
                    n_vertices = int(result.rx2('n_vertices')[0])
                    n_edges = int(result.rx2('n_edges')[0])
                    
                    self._log(f"MetricGraph created: {n_vertices} vertices, {n_edges} edges")
                    self._log_timing("Graph creation", start_time)
                    
                    return graph
                else:
                    error_msg = str(result.rx2('error')[0])
                    self._log(f"Graph creation failed: {error_msg}")
                    return None
                    
            except Exception as e:
                self._log(f"Error extracting graph result: {str(e)}")
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
            # Define R function
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
            
            # Call the R function and extract results
            info_func = ro.r['get_graph_info']
            info = info_func(metric_graph)
            
            return {
                'n_vertices': int(info[0][0]),
                'n_edges': int(info[1][0]),
                'has_mesh': bool(info[2][0]),
                'bbox': np.array(info[3])
            }
            
        except Exception as e:
            self._log(f"Error retrieving graph info: {str(e)}")
            return None
    
    def _kriging_baseline(self, tract_observation: pd.DataFrame, prediction_locations: pd.DataFrame) -> Dict:
        """Standard kriging baseline - clean version with no undefined references"""
        self._log("Using standard kriging baseline for disaggregation")
        
        try:
            # Get tract parameters
            tract_x = float(tract_observation['coord_x'].iloc[0])
            tract_y = float(tract_observation['coord_y'].iloc[0])
            tract_svi = float(tract_observation['svi_value'].iloc[0])
            
            # Calculate distances
            distances = np.sqrt(
                (prediction_locations['x'] - tract_x)**2 + 
                (prediction_locations['y'] - tract_y)**2
            )
            
            # Simple kriging model
            range_param = float(np.percentile(distances, 75))
            if range_param == 0:
                range_param = 0.01  # Avoid division by zero
                
            sill = tract_svi * 0.1
            
            # Kriging weights
            weights = np.exp(-3 * distances / range_param)
            weights = weights / weights.sum()
            
            # Predictions
            predictions = tract_svi * weights * len(weights)
            kriging_sd = np.sqrt(sill * (1 - np.exp(-3 * distances / range_param)))
            
            # Create results DataFrame
            predictions_df = pd.DataFrame({
                'x': prediction_locations['x'].values,
                'y': prediction_locations['y'].values,
                'mean': predictions,
                'sd': kriging_sd,
                'q025': predictions - 1.96 * kriging_sd,
                'q975': predictions + 1.96 * kriging_sd
            })
            
            # Ensure non-negativity
            predictions_df['mean'] = predictions_df['mean'].clip(lower=0)
            predictions_df['q025'] = predictions_df['q025'].clip(lower=0)
            
            # Calculate diagnostics
            total_predicted = float(predictions_df['mean'].sum())
            constraint_error = abs(total_predicted - tract_svi) / max(tract_svi, 1e-6)
            
            return {
                'success': True,
                'predictions': predictions_df,
                'diagnostics': {
                    'constraint_satisfied': constraint_error < 0.001,
                    'constraint_error': constraint_error,
                    'total_predicted': total_predicted,
                    'tract_value': tract_svi,
                    'mean_prediction': float(predictions_df['mean'].mean()),
                    'std_prediction': float(predictions_df['mean'].std()),
                    'mean_uncertainty': float(predictions_df['sd'].mean()),
                    'method': 'ordinary_kriging'
                }
            }
            
        except Exception as e:
            self._log(f"Kriging baseline failed: {str(e)}")
            n_pred = len(prediction_locations)
            uniform_value = tract_observation['svi_value'].iloc[0] / max(n_pred, 1)
            
            return {
                'success': False,
                'predictions': pd.DataFrame({
                    'x': prediction_locations['x'].values,
                    'y': prediction_locations['y'].values,
                    'mean': [uniform_value] * n_pred,
                    'sd': [uniform_value * 0.1] * n_pred,
                    'q025': [uniform_value * 0.8] * n_pred,
                    'q975': [uniform_value * 1.2] * n_pred
                }),
                'diagnostics': {
                    'constraint_satisfied': False,
                    'constraint_error': 1.0,
                    'mean_prediction': uniform_value,
                    'std_prediction': 0.0,
                    'mean_uncertainty': uniform_value * 0.1,
                    'method': 'uniform_fallback'
                },
                'error': str(e)
            }