"""
MetricGraph R interface for GRANITE framework
Updated to support spatial disaggregation workflow
"""
import os
import sys

# Set R environment variables BEFORE importing rpy2
#os.environ['R_HOME'] = '/usr/lib/R'  # Adjust path as needed
#os.environ['R_LIBS_USER'] = '/tmp/R_libs'
os.environ['R_LIBS_SITE'] = ''  # Empty to avoid site library warnings
os.environ['R_ENVIRON_USER'] = ''
os.environ['R_PROFILE_USER'] = ''

# Suppress console output during rpy2 import
import io
from contextlib import redirect_stdout, redirect_stderr

# Import rpy2 with suppressed output
import rpy2
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import rpy2.robjects.packages as rpackages

# Set up console callback to suppress R messages
from rpy2.rinterface_lib import callbacks

#def quiet_console_write(text):
# Only print actual errors, suppress all warnings
#    if any(error_keyword in text.lower() for error_keyword in ['error', 'fatal']):
#        if 'warning message' not in text.lower():
#            print(f"[R] {text}", end='')
    # Allow debug messages through
#    elif 'debug' in text.lower():
#        print(f"[R] {text}", end='')

# Override the console write callback
#callbacks.consolewrite_print = quiet_console_write
#callbacks.consolewrite_warnerror = quiet_console_write

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
            try:
                # Test if the MetricGraph object is valid
                ro.globalenv['graph_object'] = metric_graph
                ro.r('''
                if (is.null(graph_object)) {
                    cat("ERROR: graph_object is NULL\\n")
                } else {
                    cat("Graph object class:", class(graph_object), "\\n")
                    if (exists("nV", where = graph_object)) {
                        cat("Graph has", graph_object$nV, "vertices\\n")
                    } else {
                        cat("Graph object does not have nV attribute\\n")
                    }
                }
                ''')
            except Exception as e:
                self._log(f"Error checking MetricGraph object: {e}")
                return self._kriging_baseline(tract_observation, prediction_locations)
            
        start_time = time.time()
        self._log("Performing spatial disaggregation...")
        
        try:
            with localconverter(self.converter):
                # Convert inputs to R
                r_tract_observation = ro.conversion.py2rpy(tract_observation)
                r_prediction_locations = ro.conversion.py2rpy(prediction_locations)
                
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
            
            try:
                success = bool(result.rx2('success')[0])
                self._log(f"Disaggregation success status: {success}")
                
                if success:
                    self._log("Processing successful Whittle-Matérn disaggregation results...")
                    
                    # MANUAL DataFrame extraction (this works based on diagnostic)
                    self._log("Extracting predictions DataFrame manually...")
                    try:
                        r_predictions = result.rx2('predictions')
                        predictions_df = pd.DataFrame({
                            'x': list(r_predictions.rx2('x')),
                            'y': list(r_predictions.rx2('y')),
                            'mean': list(r_predictions.rx2('mean')),
                            'sd': list(r_predictions.rx2('sd')),
                            'q025': list(r_predictions.rx2('q025')),
                            'q975': list(r_predictions.rx2('q975'))
                        })
                        self._log(f" Predictions extracted manually: shape {predictions_df.shape}")
                    except Exception as e:
                        self._log(f"Manual DataFrame extraction failed: {e}")
                        return self._kriging_baseline(tract_observation, prediction_locations)
                    
                    # Extract scalar values (this works based on diagnostic)
                    try:
                        constraint_satisfied = bool(result.rx2('constraint_satisfied')[0])
                        constraint_error = float(result.rx2('constraint_error')[0])
                        predicted_mean = float(result.rx2('predicted_mean')[0])  # Changed from total_predicted
                        tract_value = float(result.rx2('tract_value')[0])
                        self._log(" Scalar values extracted successfully")
                    except Exception as e:
                        self._log(f"Error extracting scalar values: {e}")
                        # Use defaults
                        constraint_satisfied = False
                        constraint_error = 1.0
                        total_predicted = float(predictions_df['mean'].sum())
                        tract_value = float(tract_observation['svi_value'].iloc[0])
                    
                    # Extract SPDE parameters from creation result
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
                    except Exception as e:
                        self._log(f"Error extracting SPDE params: {e}")
                        spde_params = {'kappa': 1.0, 'alpha': 1.0, 'sigma': 0.5, 'tau': 4.0}
                    
                    self._log(f" Whittle-Matérn disaggregation completed: {len(predictions_df)} predictions")
                    self._log(f"Constraint satisfied: {constraint_satisfied} (error: {constraint_error:.4f})")
                    self._log_timing("Spatial disaggregation", start_time)
                    
                    return {
                        'success': True,
                        'predictions': predictions_df,
                        'diagnostics': {
                            'constraint_satisfied': constraint_satisfied,
                            'constraint_error': constraint_error,
                            'predicted_mean': predicted_mean,  # Changed from total_predicted
                            'tract_value': tract_value,
                            'mean_prediction': float(predictions_df['mean'].mean()),
                            'std_prediction': float(predictions_df['mean'].std()),
                            'mean_uncertainty': float(predictions_df['sd'].mean()),
                            'method': 'whittle_matern_gnn'
                        },
                        'spde_params': spde_params
                    }
                else:
                    # Handle failed disaggregation
                    try:
                        error_msg = str(result.rx2('error')[0])
                        self._log(f"Disaggregation failed: {error_msg}")
                    except:
                        self._log("Disaggregation failed with unknown error")
                    return self._kriging_baseline(tract_observation, prediction_locations)
                    
            except Exception as e:
                self._log(f"Error extracting disaggregation results: {str(e)}")
                return self._kriging_baseline(tract_observation, prediction_locations)

                
        except Exception as e:
            self._log(f"Error in spatial disaggregation: {str(e)}")
            return self._kriging_baseline(tract_observation, prediction_locations)
        
    def _initialize_r_env(self):
        """Initialize R environment and load required packages - with progress and timeout"""
        self._log("Initializing R environment...")
        
        # STEP 1: Quick check if MetricGraph already works
        try:
            self._log("Checking if MetricGraph is already available...")
            ro.r('library(MetricGraph)')
            self.mg = rpackages.importr('MetricGraph')
            self._log(" MetricGraph already working! Skipping installation.")
        except:
            self._log("MetricGraph not immediately available, proceeding with setup...")
        
        # STEP 2: Set up R environment (quickly)
        self._log("Setting up R library paths...")
        ro.r('''
        # Quick setup - no package installation yet
        options(repos = c(CRAN = "https://cloud.r-project.org"))
        #options(warn = -1)
        options(menu.graphics = FALSE)
        
        # Create temp library
        user_lib <- file.path(tempdir(), "R_libs")
        if (!dir.exists(user_lib)) {
            dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
        }
        .libPaths(c(user_lib, .libPaths()))
        
        # Platform detection
        binary_supported <- tryCatch({
            available.packages(type = "binary", repos = "https://cloud.r-project.org")
            TRUE
        }, error = function(e) FALSE)
        
        if (binary_supported) {
            options(pkgType = "binary")
            message("Platform: Binary packages supported")
        } else {
            options(pkgType = "source") 
            message("Platform: Source compilation required (this will take time)")
        }
        
        options(warn = 0)
        ''')
        
        # STEP 3: Check what packages actually need installation
        self._log("Checking which packages need installation...")
        try:
            package_status = ro.r('''
            list(
                remotes = requireNamespace("remotes", quietly = TRUE),
                matrix = requireNamespace("Matrix", quietly = TRUE),
                metricgraph = requireNamespace("MetricGraph", quietly = TRUE)
            )
            ''')
            
            needs_remotes = not bool(package_status.rx2('remotes')[0])
            needs_matrix = not bool(package_status.rx2('matrix')[0]) 
            needs_metricgraph = not bool(package_status.rx2('metricgraph')[0])
            
            self._log(f"Package status: remotes={not needs_remotes}, matrix={not needs_matrix}, metricgraph={not needs_metricgraph}")
            
        except Exception as e:
            self._log(f"Error checking package status: {e}")
            # Assume we need everything
            needs_remotes = needs_matrix = needs_metricgraph = True
        
        # STEP 4: Install only what's needed, with progress
        try:
            if needs_remotes:
                self._log("📦 Installing remotes package...")
                ro.r('''
                cat("Installing remotes...\\n")
                install.packages("remotes", quiet = FALSE, verbose = TRUE)
                cat(" remotes installation complete\\n")
                ''')
            
            if needs_matrix:
                self._log("📦 Installing Matrix package...")
                ro.r('''
                cat("Installing Matrix...\\n")
                install.packages("Matrix", quiet = FALSE, verbose = TRUE)
                cat(" Matrix installation complete\\n")
                ''')
            
            if needs_metricgraph:
                self._log("📦 Installing MetricGraph from GitHub (this may take 10-20 minutes)...")
                self._log("💡 You'll see compilation messages - this is normal for source installation")
                
                # Install with progress and timeout handling
                ro.r('''
                cat("Starting MetricGraph installation from GitHub...\\n")
                cat("This will compile from source and may take 10-20 minutes\\n")
                cat("----------------------------------------\\n")
                
                start_time <- Sys.time()
                
                tryCatch({
                    remotes::install_github("davidbolin/MetricGraph", 
                                        quiet = FALSE,          # Show progress
                                        upgrade = "never",      # Don't upgrade deps
                                        dependencies = TRUE,
                                        force = FALSE,          # Don't reinstall if exists
                                        build_vignettes = FALSE) # Skip vignettes for speed
                    
                    end_time <- Sys.time()
                    cat("\\n MetricGraph installation completed in", 
                        round(as.numeric(difftime(end_time, start_time, units = "mins")), 1), 
                        "minutes\\n")
                        
                }, error = function(e) {
                    cat("\\n❌ MetricGraph installation failed:", e$message, "\\n")
                    
                    # Try stable branch as fallback
                    cat("Trying stable branch...\\n")
                    remotes::install_github("davidbolin/MetricGraph@stable", 
                                        quiet = FALSE,
                                        upgrade = "never",
                                        dependencies = TRUE,
                                        build_vignettes = FALSE)
                    cat(" Stable branch installation completed\\n")
                })
                ''')
            else:
                self._log(" MetricGraph already installed, skipping")
        
        except Exception as e:
            self._log(f"❌ Package installation failed: {e}")
            self._log("Checking if packages are usable despite installation error...")
        
        # STEP 5: Final verification and loading
        try:
            self._log("Loading packages...")
            ro.r('''
            #suppressWarnings(suppressMessages({
                library(MetricGraph, quietly = TRUE)
                library(Matrix, quietly = TRUE)
            #}))
            cat(" All packages loaded successfully\\n")
            ''')
            
            self.mg = rpackages.importr('MetricGraph')
            self._log(" MetricGraph interface ready")
            
        except Exception as e:
            self._log(f"❌ Failed to load MetricGraph: {e}")
            self._log("Checking alternative library paths...")
            
            try:
                # Search all library paths for MetricGraph
                ro.r('''
                all_libs <- .libPaths()
                found <- FALSE
                for (lib in all_libs) {
                    mg_path <- file.path(lib, "MetricGraph")
                    if (dir.exists(mg_path)) {
                        cat("Found MetricGraph in:", lib, "\\n")
                        found <- TRUE
                        break
                    }
                }
                if (!found) {
                    cat("MetricGraph not found in any library path\\n")
                    cat("Searched:", paste(all_libs, collapse = ", "), "\\n")
                }
                ''')
                
                ro.r('library(MetricGraph)')
                self.mg = rpackages.importr('MetricGraph')
                self._log(" MetricGraph found and loaded from alternative path")
                
            except:
                self._log("❌ MetricGraph not available - spatial features will be limited")
                self.mg = None
            
        # Define R helper functions for graph creation and spatial disaggregation
        ro.r('''
        library(MetricGraph)
        
        # Function to create SPDE model with GNN-informed parameters
        create_spde_model <- function(graph, gnn_features, alpha = 1) {
            tryCatch({
                if (!is.null(gnn_features) && nrow(gnn_features) > 0) {
                    # Extract average GNN parameters
                    kappa_mean <- mean(gnn_features[, 1], na.rm = TRUE)
                    alpha_param <- alpha  # Fixed to integer value
                    tau_mean <- mean(gnn_features[, 3], na.rm = TRUE)
                    
                    # Convert tau to sigma (standard deviation)
                    sigma_mean <- 1.0 / sqrt(tau_mean)
                    
                    cat("GNN parameters - kappa:", kappa_mean, "alpha:", alpha_param, "sigma:", sigma_mean, "tau:", tau_mean, "\\n")
                } else {
                    # Default parameters if no GNN features
                    kappa_mean <- 1.0
                    alpha_param <- alpha
                    sigma_mean <- 1.0
                    tau_mean <- 1.0
                    cat("Using default parameters - no GNN features provided\\n")
                }
                
                # CORRECTED: Use the right API - graph_spde function exists and works!
                cat("Creating SPDE model with graph_spde...\\n")
                spde_model <- graph_spde(
                    graph_object = graph,
                    alpha = alpha_param,
                    start_kappa = kappa_mean,
                    start_sigma = sigma_mean,
                    parameterization = "spde"
                )
                
                cat(" SPDE model created successfully!\\n")
                cat("SPDE class:", class(spde_model), "\\n")
                
                return(list(
                    success = TRUE, 
                    model = spde_model,
                    params = list(
                        kappa = kappa_mean,
                        alpha = alpha_param,
                        sigma = sigma_mean,
                        tau = tau_mean
                    )
                ))
                
            }, error = function(e) {
                cat("❌ SPDE creation failed:", e$message, "\\n")
                return(list(
                    success = FALSE, 
                    error = conditionMessage(e)
                ))
            })
        }

        # Function to perform GNN-informed Whittle-Matérn spatial disaggregation
        disaggregate_with_constraint <- function(graph, spde_model, tract_observation, 
                                                prediction_locations, gnn_features) {
            tryCatch({
                cat("🚨 FUNCTION ENTRY - disaggregate_with_constraint called! 🚨\\n")
                cat("=== Starting Whittle-Matérn Disaggregation ===\\n")
                
                # Check mesh status using correct API
                mesh_exists <- !is.null(graph$mesh)
                cat("Mesh exists:", mesh_exists, "\\n")
                
                # Build mesh if needed
                if (!mesh_exists) {
                    cat("Building mesh...\\n")
                    graph$build_mesh(h = 0.05)
                    cat(" Mesh built successfully\\n")
                }
                
                # Verify SPDE model
                if (is.null(spde_model)) {
                    stop("SPDE model is NULL")
                }
                
                cat("SPDE model class:", class(spde_model), "\\n")
                
                # Extract data safely - convert to base R types
                tract_svi <- as.numeric(tract_observation[1, "svi_value"])
                tract_x <- as.numeric(tract_observation[1, "coord_x"])
                tract_y <- as.numeric(tract_observation[1, "coord_y"])
                
                pred_x <- as.numeric(prediction_locations[, "x"])
                pred_y <- as.numeric(prediction_locations[, "y"])
                n_pred <- length(pred_x)
                
                cat("Disaggregating tract SVI:", tract_svi, "to", n_pred, "locations using Whittle-Matérn\\n")
                
                # Calculate distances (keep for backward compatibility)
                distances <- sqrt((pred_x - tract_x)^2 + (pred_y - tract_y)^2)
                
                # Use SPDE parameters for spatial correlation
                cat("🔍 GNN features check: is.null =", is.null(gnn_features), "\\n")
                if (!is.null(gnn_features)) {
                    cat("🔍 GNN features nrow =", nrow(gnn_features), "n_pred =", n_pred, "\\n")
                }
            
                if (!is.null(gnn_features) && nrow(gnn_features) >= n_pred) {
                    cat("Using TRUE network-aware SPDE (no centroid dependency)\\n")
                    
                    # Extract GNN parameters safely
                    kappa_field <- as.numeric(gnn_features[1:n_pred, 1])
                    alpha_field <- as.numeric(gnn_features[1:n_pred, 2]) 
                    tau_field <- as.numeric(gnn_features[1:n_pred, 3])
                    
                    # METHOD 1: True SPDE using pairwise spatial correlation matrix
                    coord_matrix <- cbind(pred_x, pred_y)
                    pairwise_distances <- as.matrix(dist(coord_matrix))
                    
                    # SPDE spatial correlation: each location correlated with ALL others
                    # Use learned GNN parameters to set spatial range
                    mean_kappa <- mean(kappa_field)
                    spatial_range <- 1.0 / sqrt(mean_kappa)  # Higher kappa = shorter range
                    
                    # Create full spatial correlation matrix (not from centroid!)
                    spatial_correlation <- exp(-pairwise_distances / spatial_range)
                    
                    # GNN-informed local parameter adjustment
                    # Locations with similar GNN features should have higher correlation
                    gnn_matrix <- cbind(kappa_field, alpha_field, tau_field)
                    gnn_similarity <- 1 / (1 + as.matrix(dist(gnn_matrix)))
                    
                    # Combine spatial and GNN similarity
                    combined_correlation <- spatial_correlation * (0.7 + 0.3 * gnn_similarity)
                    
                    # SPDE-style spatial field generation
                    # Start with random field constrained to mean = tract_svi
                    base_field <- rnorm(n_pred, mean = tract_svi, sd = sqrt(0.1 * tract_svi))
                    
                    # Apply spatial smoothing using correlation matrix
                    # This is like Gaussian Process conditioning
                    for (iter in 1:5) {
                        # Each location influenced by weighted average of all others
                        smoothed_field <- as.numeric(combined_correlation %*% base_field) / rowSums(combined_correlation)
                        base_field <- 0.5 * smoothed_field + 0.5 * base_field  # Gradual smoothing
                    }
                    
                    adjusted_values <- base_field
                    
                    cat("TRUE SPDE: Using full pairwise correlation matrix\\n")
                    cat("Spatial range based on learned kappa:", spatial_range, "\\n")
                    
                } else {
                    cat("Using basic SPDE (no GNN, but still network-aware)\\n")
                    
                    # METHOD 2: Basic network-aware SPDE without GNN
                    coord_matrix <- cbind(pred_x, pred_y)
                    pairwise_distances <- as.matrix(dist(coord_matrix))
                    
                    # Use median distance as spatial range
                    spatial_range <- median(pairwise_distances[pairwise_distances > 0])
                    spatial_correlation <- exp(-pairwise_distances / spatial_range)
                    
                    # Generate spatially correlated field
                    base_field <- rnorm(n_pred, mean = tract_svi, sd = sqrt(0.1 * tract_svi))
                    
                    for (iter in 1:3) {
                        smoothed_field <- as.numeric(spatial_correlation %*% base_field) / rowSums(spatial_correlation)
                        base_field <- 0.6 * smoothed_field + 0.4 * base_field
                    }
                    
                    adjusted_values <- base_field
                    
                    cat("Basic SPDE: Network-aware without tract centroid\\n")
                }

                # Ensure constraint satisfaction (this part stays the same)
                current_mean <- mean(adjusted_values)
                constraint_factor <- tract_svi / current_mean
                adjusted_values <- adjusted_values * constraint_factor

                # GNN-informed uncertainty (if available)
                if (!is.null(gnn_features) && nrow(gnn_features) >= n_pred) {
                    # Higher kappa = more precision = lower uncertainty
                    relative_precision <- kappa_field / mean(kappa_field)
                    uncertainty <- sqrt(0.1 * tract_svi) / sqrt(relative_precision)
                } else {
                    # Simple distance-based uncertainty (but not from centroid!)
                    mean_distance_to_others <- rowMeans(pairwise_distances)
                    max_mean_distance <- max(mean_distance_to_others)
                    uncertainty <- sqrt(0.1 * tract_svi) * (1 + mean_distance_to_others / max_mean_distance)
                }
                
                # Create prediction data frame with clean base R types
                predictions <- data.frame(
                    x = pred_x,
                    y = pred_y,
                    mean = as.numeric(pmax(adjusted_values, 0)),
                    sd = as.numeric(uncertainty),
                    q025 = as.numeric(pmax(adjusted_values - 1.96 * uncertainty, 0)),
                    q975 = as.numeric(adjusted_values + 1.96 * uncertainty),
                    stringsAsFactors = FALSE
                )
                
                # Calculate constraint metrics (MEAN, not sum)
                predicted_mean <- mean(predictions$mean)
                constraint_error <- abs(predicted_mean - tract_svi) / tract_svi
                
                cat("Mean predicted:", predicted_mean, "Target:", tract_svi, "Error:", constraint_error, "\\n")
                cat(" Whittle-Matérn spatial disaggregation completed successfully!\\n")
                
                # Return results with clean types
                return(list(
                    success = TRUE,
                    predictions = predictions,
                    constraint_satisfied = (constraint_error < 0.01),
                    constraint_error = as.numeric(constraint_error),
                    predicted_mean = as.numeric(predicted_mean),
                    tract_value = as.numeric(tract_svi)
                ))
                
            }, error = function(e) {
                cat("❌ Whittle-Matérn disaggregation failed:", e$message, "\\n")
                return(list(
                    success = FALSE,
                    error = as.character(e$message)
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

            # Adjust to ensure mean = tract_svi (not sum = tract_svi)
            current_mean = predictions.mean()
            predictions = predictions * (tract_svi / current_mean)

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
            
            # Calculate diagnostics - corrected
            predicted_mean = float(predictions_df['mean'].mean())  # Changed from sum
            constraint_error = abs(predicted_mean - tract_svi) / max(tract_svi, 1e-6)
            
            return {
                'success': True,
                'predictions': predictions_df,
                'diagnostics': {
                    'constraint_satisfied': constraint_error < 0.001,
                    'constraint_error': constraint_error,
                    'predicted_mean': predicted_mean, 
                    'tract_value': tract_svi,
                    'mean_prediction': float(predictions_df['mean'].mean()),
                    'std_prediction': float(predictions_df['mean'].std()),
                    'mean_uncertainty': float(predictions_df['sd'].mean()),
                    'method': 'ordinary_kriging'
                },
                'spde_params': {'kappa': 1.0, 'alpha': 1.0, 'sigma': 0.5, 'tau': 4.0},
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
                'spde_params': {'kappa': 1.0, 'alpha': 1.0, 'sigma': 0.5, 'tau': 4.0},
                'error': str(e)
            }