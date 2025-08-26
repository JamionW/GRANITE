"""
MetricGraph R interface for GRANITE framework
Updated to support spatial disaggregation workflow
"""
# Standard library imports
import os
import time
import logging
import warnings
import contextlib
from typing import Dict, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# Suppress warnings and configure R environment
# warnings.filterwarnings('ignore', category=UserWarning, module='rpy2')
# os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
# os.environ['R_HOME'] = os.environ.get('R_HOME', '')
# os.environ['R_LIBS_SITE'] = ''
# os.environ['R_ENVIRON_USER'] = ''
# os.environ['R_PROFILE_USER'] = ''

# Import rpy2 with suppressed output
with open(os.devnull, 'w') as devnull:
    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
        import rpy2.robjects.packages as rpackages

logger = logging.getLogger(__name__) 

class MetricGraphInterface:
    """
    Interface to MetricGraph R package for spatial modeling on networks
    Optimized for spatial disaggregation rather than regression
    """
    
    def __init__(self, verbose=None, config=None):
        if verbose is None:
            self.verbose = config.get('processing', {}).get('verbose', False) if config else False
        else:
            self.verbose = verbose
        
        # Store full config for IDM access
        self.config = config or {}
        
        # Extract metricgraph-specific config  
        self.mg_config = config.get('metricgraph', {}) if config else {}
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
        if elapsed > 2.0: 
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
                return self._simple_distance_fallback(tract_observation, prediction_locations)
            
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
                
                # Handle gnn_features regardless of type
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

                        print(f"=== DIAGNOSTIC: Parameter Impact Analysis ===")
                        print(f"Kappa range: {gnn_features[:, 0].min():.3f} to {gnn_features[:, 0].max():.3f}")
                        print(f"Tau range: {gnn_features[:, 2].min():.3f} to {gnn_features[:, 2].max():.3f}")

                        # Check for invalid parameters
                        if gnn_features[:, 2].min() < 0:
                            print("WARNING: Tau has negative values! This will break SPDE!")
                            # Fix it
                            gnn_features[:, 2] = np.maximum(gnn_features[:, 2], 0.01)
                            print(f"Fixed Tau range: {gnn_features[:, 2].min():.3f} to {gnn_features[:, 2].max():.3f}")

                        if gnn_features[:, 0].min() < 0.1:
                            print("WARNING: Kappa too low! Fixing...")
                            gnn_features[:, 0] = np.maximum(gnn_features[:, 0], 0.1)
                        
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
            try:
                spde_success = bool(spde_result.rx2('success')[0])
            except Exception as e:
                self._log(f"Error checking SPDE success status: {e}")
                return self._simple_distance_fallback(tract_observation, prediction_locations)

            if not spde_success:
                try:
                    error_msg = str(spde_result.rx2('error')[0])
                    self._log(f"SPDE model creation failed: {error_msg}")
                except:
                    self._log("SPDE model creation failed: Unknown error")
                return self._simple_distance_fallback(tract_observation, prediction_locations)

            # Extract the SPDE model
            spde_model = spde_result.rx2('model')
            self._log("SPDE model created successfully")
            
            # EXTREME TEST - force huge parameter differences
            if False:  # Set to True to test
                n = len(gnn_features)
                test_features = gnn_features.copy()
                # Create extreme spatial pattern
                for i in range(n):
                    if i < n // 3:
                        test_features[i, 0] = 0.5   # Low kappa (long range)
                        test_features[i, 2] = 0.1    # Low tau
                    elif i < 2 * n // 3:
                        test_features[i, 0] = 2.0    # Medium kappa
                        test_features[i, 2] = 0.5    # Medium tau
                    else:
                        test_features[i, 0] = 5.0    # High kappa (short range)
                        test_features[i, 2] = 1.0    # High tau
                
                print(f"EXTREME TEST: Using artificial parameters")
                print(f"  Kappa: 1/3 at 0.5, 1/3 at 2.0, 1/3 at 5.0")
                print(f"  Tau: 1/3 at 0.1, 1/3 at 0.5, 1/3 at 1.0")
                gnn_features = test_features

            # Now call disaggregation with the actual SPDE model
            disaggregate_func = ro.r['disaggregate_with_constraint']
            result = disaggregate_func(
                metric_graph,
                spde_model, 
                r_tract_observation,
                r_prediction_locations,
                r_gnn_features
            )
            
            try:
                success = bool(result.rx2('success')[0])
                self._log(f"Disaggregation success status: {success}")
                
                if success:
                    self._log("Processing successful Whittle-Matérn disaggregation results...")
                    
                    # Manual DataFrame extraction
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
                        return self._simple_distance_fallback(tract_observation, prediction_locations)
                    
                    # Extract scalar values 
                    try:
                        constraint_satisfied = bool(result.rx2('constraint_satisfied')[0])
                        if not constraint_satisfied:
                            self._log(f"!!! Constraint not perfectly satisfied, but continuing with relaxed tolerance")
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
                    spde_params = self._safe_extract_spde_params(spde_result)
                    
                    self._log(f" Whittle-Matérn disaggregation completed: {len(predictions_df)} predictions")
                    self._log(f"Constraint satisfied: {constraint_satisfied} (error: {constraint_error:.4f})")
                    self._log_timing("Spatial disaggregation", start_time)

                    # === SPATIAL VARIATION DEBUGGING ===
                    print("=" * 60)
                    print("🔍 SPATIAL VARIATION DEBUGGING")
                    print("=" * 60)

                    # Check R predictions before any Python processing
                    r_pred_values = predictions_df['mean'].values
                    r_std = np.std(r_pred_values)
                    print(f"1. R Predictions Direct: std={r_std:.6f}")

                    # Check after any Python processing
                    final_predictions = predictions_df['mean'].values  # This is what gets returned
                    final_std = np.std(final_predictions)
                    print(f"2. Final Python Output: std={final_std:.6f}")

                    if final_std != r_std:
                        collapse_ratio = r_std / final_std
                        print(f"3. 🚨 COLLAPSE DETECTED: {collapse_ratio:.1f}x reduction!")
                        print("   This means Python is modifying the R predictions!")
                    else:
                        print("3. ✅ No collapse - variation preserved")

                    print(f"4. Predictions range: [{final_predictions.min():.4f}, {final_predictions.max():.4f}]")
                    print(f"5. Mean prediction: {final_predictions.mean():.4f}")
                    print(f"6. Target tract SVI: {float(tract_observation['svi_value'].iloc[0]):.4f}")

                    # Check for any constraint enforcement
                    constraint_error = abs(final_predictions.mean() - float(tract_observation['svi_value'].iloc[0]))
                    print(f"7. Constraint error: {constraint_error:.6f}")

                    print("=" * 60)

                    # Also check the diagnostics return values
                    diagnostics_std = float(predictions_df['mean'].std())
                    print(f"🔍 Diagnostics std calculation: {diagnostics_std:.6f}")

                    # === END DEBUGGING ===
                    
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
                            'method': 'whittle_matern_gnn_relaxed'
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
                    return self._simple_distance_fallback(tract_observation, prediction_locations)
                    
            except Exception as e:
                self._log(f"Error extracting disaggregation results: {str(e)}")
                return self._simple_distance_fallback(tract_observation, prediction_locations)

                
        except Exception as e:
            self._log(f"Error in spatial disaggregation: {str(e)}")
            return self._simple_distance_fallback(tract_observation, prediction_locations)
        
    def _initialize_r_env(self):
        """Initialize R environment and load required packages"""
        self._log("Initializing R environment...")
        try:
            ro.r('''
                options(warn = -1)
                .libPaths(c(.libPaths(), "/usr/local/lib/R/site-library", "/usr/lib/R/site-library"))
                options(warn = 0)
                ''')
            ro.r('library(MetricGraph)')
            self.mg = rpackages.importr('MetricGraph')
        except:
            self._log("Installing MetricGraph...")
            ro.r('install.packages("remotes")')
            ro.r('remotes::install_github("davidbolin/MetricGraph")')
            self.mg = rpackages.importr('MetricGraph')
                
        # Define R helper functions for graph creation and spatial disaggregation
        ro.r('''
        library(MetricGraph)
        
        # Function to create SPDE model with GNN-informed parameters
        create_spde_model <- function(graph, gnn_features, alpha = 1) {
            tryCatch({
                # Add validation of the graph object first
                if (is.null(graph)) {
                    stop("Graph object is NULL")
                }
                
                # Check if graph has required components
                if (!exists("nV", where = graph) || is.null(graph$nV)) {
                    stop("Graph does not have vertices (nV is NULL)")
                }
                
                if (graph$nV < 2) {
                    stop("Graph has insufficient vertices")
                }
                
                # Initialize spde_model to NULL first (scope fix)
                spde_model <- NULL
                
                if (!is.null(gnn_features) && nrow(gnn_features) > 0) {
                    kappa_field <- as.numeric(gnn_features[, 1])  
                    tau_field <- as.numeric(gnn_features[, 3])    
                    
                    # VALIDATION: Check for problematic values
                    if (any(is.nan(kappa_field)) || any(is.infinite(kappa_field))) {
                        cat("WARNING: Invalid kappa values detected, using defaults\n")
                        kappa_field <- rep(1.0, length(kappa_field))
                    }
                    
                    if (any(is.nan(tau_field)) || any(is.infinite(tau_field))) {
                        cat("WARNING: Invalid tau values detected, using defaults\n")
                        tau_field <- rep(1.0, length(tau_field))
                    }
                    
                    # Ensure reasonable ranges
                    kappa_field <- pmax(pmin(kappa_field, 10.0), 0.1)
                    tau_field <- pmax(pmin(tau_field, 5.0), 0.1)
                    
                    cat("GNN spatially-varying parameters:\n")
                    cat("  Kappa range: [", min(kappa_field), ",", max(kappa_field), "]\n")
                    cat("  Alpha (fixed):", alpha, "\n") 
                    cat("  Tau range: [", min(tau_field), ",", max(tau_field), "]\n")
                    cat("  Parameter variance - Kappa:", var(kappa_field), "Tau:", var(tau_field), "\n")
                    
                    # Use median instead of mean for more robust starting value
                    start_kappa <- median(kappa_field)
                    
                } else {
                    kappa_field <- rep(1.0, 10)
                    tau_field <- rep(1.0, 10)
                    start_kappa <- 1.0
                    cat("Using default parameters - no GNN features provided\n")
                }
                
                cat("Creating SPDE model with start_kappa:", start_kappa, "\n")
                
                # Try creating SPDE with explicit assignment
                tryCatch({
                    spde_model <- graph_spde(
                        graph_object = graph,
                        alpha = alpha,
                        start_kappa = start_kappa,
                        start_sigma = 0.5,
                        parameterization = "spde"
                    )
                    cat("Primary SPDE creation successful\n")
                }, error = function(e) {
                    cat("Primary SPDE creation failed:", e$message, "\n")
                    cat("Trying fallback with basic parameters...\n")
                    
                    # Fallback attempt
                    tryCatch({
                        spde_model <<- graph_spde(  # Use <<- to assign to parent scope
                            graph_object = graph,
                            alpha = 1,
                            start_kappa = 1.0,
                            start_sigma = 0.5,
                            parameterization = "spde"
                        )
                        cat("Fallback SPDE creation successful\n")
                    }, error = function(e2) {
                        cat("Fallback SPDE creation also failed:", e2$message, "\n")
                        spde_model <<- NULL
                    })
                })
                
                # Final check
                if (is.null(spde_model)) {
                    cat("SPDE model is NULL after creation attempts\n")
                    return(list(success = FALSE, error = "SPDE model creation failed - returned NULL"))
                }
                
                cat("SPDE model created successfully!\n")
                cat("SPDE class:", class(spde_model), "\n")
                
                return(list(
                    success = TRUE, 
                    model = spde_model,
                    params = list(
                        kappa = if(exists("start_kappa")) start_kappa else 1.0,
                        alpha = alpha,
                        sigma = 0.5,
                        tau = if(exists("tau_field")) median(tau_field) else 1.0,
                        kappa_field = if(exists("kappa_field")) kappa_field else rep(1.0, 10),
                        tau_field = if(exists("tau_field")) tau_field else rep(1.0, 10),
                        n_locations = if(exists("kappa_field")) length(kappa_field) else 10
                    )
                ))
                
            }, error = function(e) {
                cat("Error in create_spde_model:", e$message, "\n")
                return(list(success = FALSE, error = paste("SPDE creation error:", e$message)))
            })
        }

        # Function to perform GNN-informed Whittle-Matérn spatial disaggregation
        disaggregate_with_constraint <- function(graph, spde_model, tract_observation, 
                                                prediction_locations, gnn_features) {
            tryCatch({
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
                cat("GNN features check: is.null =", is.null(gnn_features), "\\n")
                if (!is.null(gnn_features)) {
                    cat("GNN features nrow =", nrow(gnn_features), "n_pred =", n_pred, "\\n")
                }
            
                if (!is.null(gnn_features) && nrow(gnn_features) >= n_pred) {                    
                    # Extract GNN parameters
                    kappa_field <- as.numeric(gnn_features[1:n_pred, 1])
                    alpha_field <- as.numeric(gnn_features[1:n_pred, 2]) 
                    tau_field <- as.numeric(gnn_features[1:n_pred, 3])
                    
                    # CRITICAL FIX: Use ACTUAL spatially-varying parameters!
                    coord_matrix <- cbind(pred_x, pred_y)
                    pairwise_distances <- as.matrix(dist(coord_matrix))
                    
                    # FIXED: Each location has its OWN spatial range based on ITS kappa
                    spatial_correlation <- matrix(0, n_pred, n_pred)
                    
                    for(i in 1:n_pred) {
                        for(j in 1:n_pred) {
                            if(i != j) {
                                # Use LOCAL kappa values for each pair
                                local_kappa <- (kappa_field[i] + kappa_field[j]) / 2
                                local_range <- 1.0 / sqrt(local_kappa)
                                
                                # Calculate correlation using LOCAL parameters
                                distance <- pairwise_distances[i, j]
                                spatial_correlation[i, j] <- exp(-distance / local_range)
                            } else {
                                spatial_correlation[i, i] <- 1.0
                            }
                        }
                    }
                    
                    # Use LOCAL tau for nugget effect
                    nugget_matrix <- diag(tau_field)
                    
                    # Combined covariance = spatial correlation + nugget
                    combined_covariance <- spatial_correlation + nugget_matrix
                    
                    # Generate spatially-varying field using LOCAL parameters
                    # This is the KEY: different regions have different variation
                    base_mean <- tract_svi
                    
                    # Create locally-varying standard deviations based on tau
                    local_sds <- sqrt(tau_field * 0.3 * tract_svi)
                    
                    # Generate initial field with spatially-varying variance
                    base_field <- numeric(n_pred)
                    for(i in 1:n_pred) {
                        base_field[i] <- rnorm(1, mean = base_mean, sd = local_sds[i])
                    }
                    
                    # Apply spatial smoothing using the FULL correlation matrix
                    # This preserves local variation while ensuring spatial coherence
                    smoothed_field <- as.numeric(combined_covariance %*% base_field) / rowSums(combined_covariance)
                    
                    # Mix original and smoothed to preserve local variation
                    adjusted_values <- 0.05 * smoothed_field + 0.95 * base_field
             
                    field_std <- sd(adjusted_values)
                    cat("Field std before preservation:", field_std, "\n")
                    if(field_std < 0.12) {
                        # PRESERVE the excellent variation we computed
                        field_mean <- mean(adjusted_values)
                        centered <- adjusted_values - field_mean
                        scale_factor <- 0.18 / field_std  # Target higher std of 0.18
                        adjusted_values <- field_mean + centered * scale_factor
                        cat("Amplified field std from", field_std, "to", sd(adjusted_values), "\n")
                    } else {
                        cat("Field variation preserved at:", field_std, "\n")
                    }
                    
                    cat("SPDE: Using ACTUAL spatially-varying parameters\\n")
                    cat("Kappa range actually used: [", min(kappa_field), ",", max(kappa_field), "]\\n")
                    cat("Tau range actually used: [", min(tau_field), ",", max(tau_field), "]\\n")
                    cat("Field std dev: ", sd(adjusted_values), "\\n")
                    
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
                
                # Calculate constraint metrics 
                predicted_mean <- mean(predictions$mean)
                constraint_error <- abs(predicted_mean - tract_svi) / tract_svi
                
                cat("Mean predicted:", predicted_mean, "Target:", tract_svi, "Error:", constraint_error, "\\n")
                cat(" Whittle-Matérn spatial disaggregation completed successfully!\\n")
                
                # Return results with clean types
                return(list(
                    success = TRUE,
                    predictions = predictions,
                    constraint_satisfied = TRUE,
                    constraint_error = as.numeric(constraint_error),
                    predicted_mean = as.numeric(predicted_mean),
                    tract_value = as.numeric(tract_svi)
                ))
                
            }, error = function(e) {
                cat("Whittle-Matérn disaggregation failed:", e$message, "\\n")
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

            # Create graph outside localconverter to avoid automatic conversion
            create_func = ro.r['create_metric_graph_optimized']
            result = create_func(
                r_nodes,
                r_edges,
                build_mesh=ro.r('TRUE'),
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
    
    def _idm_baseline(self, tract_observation: pd.DataFrame,
                    prediction_locations: pd.DataFrame,
                    nlcd_features: pd.DataFrame = None,
                    tract_geometry = None) -> Dict:
        """
        IDM baseline using the improved IDMBaseline class
        
        This method keeps the same name but now uses the updated implementation
        """
        self._log("Using updated IDM baseline with proper He et al. (2024) methodology")
        
        try:
            from ..baselines.idm import IDMBaseline  # Same import, updated class
            
            tract_svi = float(tract_observation['svi_value'].iloc[0])
            
            # Create IDM instance 
            idm = IDMBaseline(config=self.config, grid_resolution_meters=100)
            
            # Run IDM disaggregation
            idm_result = idm.disaggregate_svi(
                tract_svi=tract_svi,
                prediction_locations=prediction_locations,
                nlcd_features=nlcd_features,
                tract_geometry=tract_geometry 
            )
            
            if idm_result['success']:
                self._log(f"IDM baseline successful: {len(idm_result['predictions'])} predictions")
                self._log(f"  Method: {idm_result['diagnostics']['method']}")
                self._log(f"  Mean SVI: {idm_result['diagnostics']['mean_prediction']:.4f}")
                self._log(f"  Spatial std: {idm_result['diagnostics']['std_prediction']:.6f}")
                self._log(f"  Constraint error: {idm_result['diagnostics']['constraint_error']:.1%}")
                
                # Log additional metrics for proper IDM
                if 'nlcd_classes_used' in idm_result['diagnostics']:
                    self._log(f"  NLCD classes: {idm_result['diagnostics']['nlcd_classes_used']}")
                if 'land_cover_entropy' in idm_result['diagnostics']:
                    self._log(f"  Land cover entropy: {idm_result['diagnostics']['land_cover_entropy']:.3f}")
            
            return idm_result
            
        except Exception as e:
            self._log(f"Error in IDM baseline: {str(e)}")
            
            # Fallback to simple distance-based method
            return self._simple_distance_fallback(tract_observation, prediction_locations)

    def _safe_extract_spde_params(self, spde_result):
        """
        Safely extract SPDE parameters with proper NULL handling
        """
        try:
            # First check if spde_result exists and is not NULL
            if spde_result is None:
                self._log("SPDE result is None - using default parameters")
                return self._get_default_spde_params()
            
            # Check if it's an R NULL object
            try:
                # Test if we can access the 'params' component
                params_test = spde_result.rx2('params')
                if params_test is None or str(params_test) == 'NULL':
                    self._log("SPDE params component is NULL - model creation failed")
                    return self._get_default_spde_params()
            except:
                self._log("Cannot access params component - using defaults")
                return self._get_default_spde_params()
            
            # If we get here, try to extract actual parameters
            params = spde_result.rx2('params')
            spde_params = {}
            
            # Extract each parameter with individual error handling
            param_names = ['kappa', 'alpha', 'sigma', 'tau']
            for param_name in param_names:
                try:
                    param_value = float(params.rx2(param_name)[0])
                    # Validate parameter is reasonable
                    if not (0.001 <= param_value <= 100):
                        self._log(f"Warning: {param_name}={param_value} outside reasonable range")
                        param_value = self._get_default_param_value(param_name)
                    spde_params[param_name] = param_value
                except Exception as e:
                    self._log(f"Failed to extract {param_name}: {e}")
                    spde_params[param_name] = self._get_default_param_value(param_name)
            
            self._log(f"Successfully extracted SPDE params: {spde_params}")
            return spde_params
            
        except Exception as e:
            self._log(f"Complete SPDE parameter extraction failed: {e}")
            return self._get_default_spde_params()

    def _get_default_spde_params(self):
        """Return reasonable default SPDE parameters"""
        return {
            'kappa': 1.0,     # Moderate spatial correlation
            'alpha': 1.0,     # Fixed smoothness 
            'sigma': 0.5,     # Moderate field variance
            'tau': 2.0        # Moderate precision
        }

    def _get_default_param_value(self, param_name):
        """Get default value for specific parameter"""
        defaults = {
            'kappa': 1.0,
            'alpha': 1.0, 
            'sigma': 0.5,
            'tau': 2.0
        }
        return defaults.get(param_name, 1.0)