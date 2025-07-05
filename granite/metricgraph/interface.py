"""
MetricGraph R interface for GRANITE framework
"""
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from datetime import datetime

# Enable automatic pandas conversion
pandas2ri.activate()


class MetricGraphInterface:
    """Interface to R MetricGraph package for spatial modeling"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        
        # Import R packages
        self._log("Initializing R interface...")
        
        try:
            self.base = importr('base')
            self.mg = importr('MetricGraph')
            self._log("  ✓ MetricGraph package loaded")
        except:
            self._log("  ✗ MetricGraph not found. Installing...")
            ro.r('install.packages("MetricGraph", repos="http://cran.r-project.org")')
            self.mg = importr('MetricGraph')
            
        # Define R functions
        self._define_r_functions()
        
    def _log(self, message):
        """Logging with timestamp"""
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def _define_r_functions(self):
        """Define R functions for MetricGraph operations"""
        
        # Create metric graph
        ro.r('''
        create_metric_graph <- function(nodes, edges) {
            # nodes: data frame with x, y columns
            # edges: data frame with from, to columns (1-indexed)
            
            V <- as.matrix(nodes[, c("x", "y")])
            E <- as.matrix(edges[, c("from", "to")])
            
            # Create metric graph
            graph <- metric_graph$new(V = V, E = E)
            
            # Build mesh for continuous representation
            graph$build_mesh(h = 0.01)
            
            # Compute Laplacian
            graph$compute_laplacian()
            
            return(graph)
        }
        ''')
        
        # Add observations to graph
        ro.r('''
        add_observations_to_graph <- function(graph, obs_data) {
            # obs_data: data frame with x, y, value columns
            
            # Map observations to graph edges
            obs_on_graph <- graph$get_data(obs_data[, c("x", "y")])
            
            # Add response variable
            obs_on_graph$y <- obs_data$value
            
            # Add to graph
            graph$add_observations(
                data = obs_on_graph,
                normalized = TRUE
            )
            
            return(graph)
        }
        ''')
        
        # Fit Whittle-Matérn model
        ro.r('''
        fit_whittle_matern <- function(graph, formula_str, covariates=NULL, alpha=1.5) {
            # Build model data
            if (!is.null(covariates)) {
                model_data <- cbind(graph$get_data(), covariates)
            } else {
                model_data <- graph$get_data()
            }
            
            # Create formula
            formula <- as.formula(formula_str)
            
            # Fit model
            model <- graph_lme(
                formula = formula,
                data = model_data,
                graph = graph,
                model = list(
                    type = "WhittleMatern",
                    alpha = alpha
                )
            )
            
            return(model)
        }
        ''')
        
        # Predict at new locations
        ro.r('''
        predict_at_locations <- function(model, graph, new_locations, covariates=NULL) {
            # Map new locations to graph
            new_on_graph <- graph$get_data(new_locations[, c("x", "y")])
            
            # Add covariates if provided
            if (!is.null(covariates)) {
                pred_data <- cbind(new_on_graph, covariates)
            } else {
                pred_data <- new_on_graph
            }
            
            # Get predictions
            preds <- predict(
                model,
                newdata = pred_data,
                compute_variances = TRUE,
                posterior_samples = FALSE
            )
            
            # Create results data frame
            results <- data.frame(
                x = new_locations$x,
                y = new_locations$y,
                mean = preds$mean,
                variance = preds$variance,
                sd = sqrt(preds$variance),
                lower_95 = preds$mean - 1.96 * sqrt(preds$variance),
                upper_95 = preds$mean + 1.96 * sqrt(preds$variance)
            )
            
            return(results)
        }
        ''')
        
        self._log("  ✓ R functions defined")
    
    def create_metric_graph(self, nodes_df, edges_df):
        """
        Create MetricGraph object from network data
        
        Parameters:
        -----------
        nodes_df : pd.DataFrame
            Nodes with columns: node_id, x, y
        edges_df : pd.DataFrame
            Edges with columns: from, to (1-indexed for R)
            
        Returns:
        --------
        R object
            MetricGraph object
        """
        self._log("Creating MetricGraph object...")
        
        # Ensure correct column names
        nodes_r = nodes_df[['x', 'y']].copy()
        edges_r = edges_df[['from', 'to']].copy()
        
        # Convert to R
        r_nodes = pandas2ri.py2rpy(nodes_r)
        r_edges = pandas2ri.py2rpy(edges_r)
        
        # Create graph
        graph = ro.r['create_metric_graph'](r_nodes, r_edges)
        
        self._log(f"  ✓ Created MetricGraph with {len(nodes_df)} nodes and {len(edges_df)} edges")
        
        return graph
    
    def add_observations(self, graph, observations_df):
        """
        Add observations to the graph
        
        Parameters:
        -----------
        graph : R object
            MetricGraph object
        observations_df : pd.DataFrame
            Observations with columns: x, y, value
            
        Returns:
        --------
        R object
            Updated graph
        """
        self._log(f"Adding {len(observations_df)} observations to graph...")
        
        # Convert to R
        r_obs = pandas2ri.py2rpy(observations_df)
        
        # Add observations
        graph = ro.r['add_observations_to_graph'](graph, r_obs)
        
        self._log("  ✓ Observations added")
        
        return graph
    
    def fit_model(self, graph, formula="y ~ 1", covariates_df=None, alpha=1.5):
        """
        Fit Whittle-Matérn model
        
        Parameters:
        -----------
        graph : R object
            MetricGraph with observations
        formula : str
            R formula string
        covariates_df : pd.DataFrame, optional
            Covariate data
        alpha : float
            Smoothness parameter
            
        Returns:
        --------
        R object
            Fitted model
        """
        self._log(f"Fitting Whittle-Matérn model (alpha={alpha})...")
        self._log(f"  Formula: {formula}")
        
        # Convert covariates if provided
        if covariates_df is not None:
            r_cov = pandas2ri.py2rpy(covariates_df)
            self._log(f"  Covariates: {list(covariates_df.columns)}")
        else:
            r_cov = ro.NULL
        
        # Fit model
        model = ro.r['fit_whittle_matern'](graph, formula, r_cov, alpha)
        
        # Extract some model info
        try:
            # Get model summary
            summary = ro.r('summary')(model)
            self._log("  ✓ Model fitted successfully")
        except:
            self._log("  ✓ Model fitted (summary not available)")
        
        return model
    
    def predict(self, model, graph, locations_df, covariates_df=None):
        """
        Predict at new locations
        
        Parameters:
        -----------
        model : R object
            Fitted model
        graph : R object
            MetricGraph object
        locations_df : pd.DataFrame
            Locations with columns: x, y
        covariates_df : pd.DataFrame, optional
            Covariates at prediction locations
            
        Returns:
        --------
        pd.DataFrame
            Predictions with uncertainty
        """
        self._log(f"Predicting at {len(locations_df)} locations...")
        
        # Convert to R
        r_locs = pandas2ri.py2rpy(locations_df)
        
        if covariates_df is not None:
            r_cov = pandas2ri.py2rpy(covariates_df)
        else:
            r_cov = ro.NULL
        
        # Get predictions
        r_preds = ro.r['predict_at_locations'](model, graph, r_locs, r_cov)
        
        # Convert back to pandas
        predictions_df = pandas2ri.rpy2py(r_preds)
        
        self._log("  ✓ Predictions complete")
        self._log(f"    - Mean prediction range: [{predictions_df['mean'].min():.3f}, {predictions_df['mean'].max():.3f}]")
        self._log(f"    - Average uncertainty (SD): {predictions_df['sd'].mean():.3f}")
        
        return predictions_df
    
    def fit_with_gnn_features(self, graph, observations_df, gnn_features, 
                             locations_df, alpha=1.5):
        """
        Fit model using GNN features as covariates
        
        Parameters:
        -----------
        graph : R object
            MetricGraph object
        observations_df : pd.DataFrame
            Observations with x, y, value
        gnn_features : np.ndarray
            GNN-learned features [num_nodes, 3]
        locations_df : pd.DataFrame
            Locations for predictions
        alpha : float
            Smoothness parameter
            
        Returns:
        --------
        pd.DataFrame
            Predictions
        """
        self._log("Fitting model with GNN features...")
        
        # Add observations to graph
        graph = self.add_observations(graph, observations_df)
        
        # Create covariate data frame from GNN features
        # This is simplified - in practice, need to map features to observation locations
        n_obs = len(observations_df)
        obs_features = pd.DataFrame(
            gnn_features[:n_obs, :],
            columns=['gnn_kappa', 'gnn_alpha', 'gnn_tau']
        )
        
        # Fit model with GNN features as covariates
        formula = "y ~ gnn_kappa + gnn_alpha + gnn_tau"
        model = self.fit_model(graph, formula, obs_features, alpha)
        
        # Prepare prediction features
        n_pred = len(locations_df)
        pred_features = pd.DataFrame(
            gnn_features[:n_pred, :],
            columns=['gnn_kappa', 'gnn_alpha', 'gnn_tau']
        )
        
        # Get predictions
        predictions = self.predict(model, graph, locations_df, pred_features)
        
        return predictions


# Convenience function
def create_metricgraph_interface():
    """Create MetricGraph interface instance"""
    return MetricGraphInterface()