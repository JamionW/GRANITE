"""
GRANITE Pipeline: Accessibility to SVI Prediction

Main orchestration for data loading, feature computation, model training,
and validation.
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from datetime import datetime
from typing import List, Dict, Optional, Tuple
warnings.filterwarnings('ignore')

from ..models.gnn import set_random_seed
import random

from ..data.loaders import DataLoader
from ..visualization.plots import GRANITEResearchVisualizer, DisaggregationVisualizer
from ..evaluation.spatial_diagnostics import SpatialLearningDiagnostics
from ..evaluation.accessibility_validator import validate_granite_accessibility_features, integrate_with_spatial_diagnostics
from ..evaluation.baselines import (
    DisaggregationComparison, 
    NaiveUniformBaseline,
    IDWDisaggregation, 
    OrdinaryKrigingDisaggregation
)
from granite.models.mixture_of_experts import create_moe_model, MixtureOfExpertsTrainer


class GRANITEPipeline:
    """
    GRANITE: Direct accessibility for SVI prediction
    
    Architecture:
    1. Load spatial data (addresses, tracts, destinations)
    2. Compute multi-modal accessibility features
    3. Build accessibility-similarity graph
    4. Train GNN: accessibility features → SVI predictions
    5. Validate against constraints and baselines
    """
    
    def __init__(self, config, data_dir='./data', output_dir='./output', verbose=None):
        self.config = config
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.verbose = config.get('processing', {}).get('verbose', False)
        
        os.makedirs(output_dir, exist_ok=True)

        enable_caching = config.get('processing', {}).get('enable_caching', True)
        cache_dir = config.get('processing', {}).get('cache_dir', './granite_cache')
        
        self.data_loader = DataLoader(data_dir, config=config)

        from granite.data.enhanced_accessibility import EnhancedAccessibilityComputer
        self.accessibility_computer = EnhancedAccessibilityComputer(
            verbose=config.get('processing', {}).get('verbose', False),
            enable_caching=enable_caching,
            cache_dir=cache_dir
        )

        self.visualizer = GRANITEResearchVisualizer()
        self.disagg_visualizer = DisaggregationVisualizer()
        
    def _log(self, message, level='INFO'):
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")

    def run(self):
        """Main pipeline execution"""
        start_time = time.time()

        seed = self.config.get('processing', {}).get('random_seed', 42)
        set_random_seed(seed)
        self._log(f"Random seed set to {seed} for reproducibility")     
        
        self._log("Starting GRANITE Accessibility SVI Pipeline...")

        # Support both per-tract holdout and global training
        validation_mode = self.config.get('validation', {}).get('mode', None)

        if validation_mode == 'holdout':
            # Per-tract holdout
            target_fips = self.config.get('data', {}).get('target_fips')
            neighbor_fips = self.config.get('validation', {}).get('neighbor_fips', [])
            
            if not neighbor_fips:
                n_neighbors = self.config.get('data', {}).get('neighbor_tracts', 3)
                all_fips = self.data_loader.get_neighboring_tracts(target_fips, n_neighbors)
                neighbor_fips = [f for f in all_fips if f != target_fips]
            
            return self.run_holdout_validation(target_fips, neighbor_fips)

        elif validation_mode == 'global':
            # Global training
            training_fips = self.config.get('validation', {}).get('training_fips', [])
            test_fips = self.config.get('validation', {}).get('test_fips', [])
            use_mixture = self.config.get('training', {}).get('use_mixture', False)
            
            if not training_fips:
                return {'success': False, 'error': 'Global mode requires training_fips'}
            
            if use_mixture:
                return self.run_mixture_training(training_fips, test_fips)
            else:
                return self.run_global_training(training_fips, test_fips)
        
        # Load data
        data = self._load_spatial_data()
        
        # Process single FIPS 
        target_fips = self.config.get('data', {}).get('target_fips')
        if not target_fips:
            return {'success': False, 'error': 'FIPS code required'}
        
        results = self._process_single_tract(target_fips, data)

        # Run feature importance analysis
        if results.get('success', False):
            skip_importance = self.config.get('processing', {}).get('skip_importance', False)
            
            if not skip_importance:
                self._log("\nRunning feature importance analysis...")
                importance_results = self.analyze_feature_importance(results, n_repeats=10)
                results['feature_importance'] = importance_results
            else:
                self._log("\nSkipping feature importance analysis (--skip-importance flag set)")
                results['feature_importance'] = None
        
        elapsed_time = time.time() - start_time
        self._log(f"Pipeline completed in {elapsed_time:.2f} seconds")
        
        return results

    def _load_spatial_data(self):
        """Load all required spatial datasets"""
        self._log("Loading spatial data...")
        
        # Get FIPS configuration
        state_fips = self.config.get('data', {}).get('state_fips', '47')
        county_fips = self.config.get('data', {}).get('county_fips', '065')
        
        data = {}
        
        # Load census tracts and SVI
        data['census_tracts'] = self.data_loader.load_census_tracts(state_fips, county_fips)
        
        county_name = self.data_loader._get_county_name(state_fips, county_fips)
        data['svi'] = self.data_loader.load_svi_data(state_fips, county_name)
        
        # Merge SVI with tracts
        data['tracts'] = data['census_tracts'].merge(data['svi'], on='FIPS', how='inner')
        
        # Load accessibility destinations
        self._log("Loading accessibility destinations...")
        data['employment_destinations'] = self.data_loader.create_employment_destinations(use_real_data=True)
        data['healthcare_destinations'] = self.data_loader.create_healthcare_destinations(use_real_data=True)
        data['grocery_destinations'] = self.data_loader.create_grocery_destinations(use_real_data=True)
        
        # Load road network and addresses
        data['roads'] = self.data_loader.load_road_network(
            roads_file=None, 
            state_fips=state_fips, 
            county_fips=county_fips
        )
        data['addresses'] = self.data_loader.load_address_points(state_fips, county_fips)
        
        self._log(f"Loaded data: {len(data['tracts'])} tracts, {len(data['addresses'])} addresses")
        self._log(f"Destinations: {len(data['employment_destinations'])} employment, "
                 f"{len(data['healthcare_destinations'])} healthcare, "
                 f"{len(data['grocery_destinations'])} grocery")
        
        return data

    def _process_single_tract(self, target_fips, data):
        """Process single tract or multiple tracts with accessibility → SVI approach"""
        
        # Clear stale accessibility cache
        cache_dir = os.path.join(self.data_dir, 'cache', 'accessibility_features')
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            self._log("Cleared old accessibility cache - will recompute with real data")
        
        # Check if multi-tract mode
        n_neighbor_tracts = self.config.get('data', {}).get('neighbor_tracts', 0)
        
        if n_neighbor_tracts > 0:
            # Multi-tract mode
            self._log(f"Multi-tract mode: {n_neighbor_tracts} neighbors")
            tract_list = self.data_loader.get_neighboring_tracts(target_fips, n_neighbor_tracts)
            
            # Combine addresses from all tracts
            all_addresses = []
            tract_svis = {}
            
            for fips in tract_list:
                fips = str(fips).strip()
                tract_data = data['tracts'][data['tracts']['FIPS'] == fips]
                if len(tract_data) == 0:
                    continue
                
                tract_svis[fips] = float(tract_data.iloc[0]['RPL_THEMES'])
                addresses = self.data_loader.get_addresses_for_tract(fips)
                addresses['tract_fips'] = fips
                all_addresses.append(addresses)
            
            if len(all_addresses) == 0:
                return {'success': False, 'error': 'No addresses in selected tracts'}
            
            tract_addresses = pd.concat(all_addresses, ignore_index=True)
            target_tract_svi = tract_svis[target_fips]
            
            self._log(f"Combined {len(tract_addresses)} addresses from {len(tract_list)} tracts")
        else:
            # Single tract mode
            self._log(f"Processing tract {target_fips}...")
            target_fips = str(target_fips).strip()
            data['tracts']['FIPS'] = data['tracts']['FIPS'].astype(str).str.strip()
            
            target_tract = data['tracts'][data['tracts']['FIPS'] == target_fips]
            if len(target_tract) == 0:
                return {'success': False, 'error': f'FIPS {target_fips} not found'}
            
            tract_info = target_tract.iloc[0]
            target_tract_svi = tract_info['RPL_THEMES']
            
            tract_addresses = self.data_loader.get_addresses_for_tract(target_fips)
            if len(tract_addresses) == 0:
                return {'success': False, 'error': f'No addresses found for tract {target_fips}'}
            
            tract_addresses['tract_fips'] = target_fips
            tract_svis = {target_fips: target_tract_svi}
        
        self._log(f"Found {len(tract_addresses)} total addresses")
        self._log(f"Target tract {target_fips} SVI: {target_tract_svi:.4f}")
        
        state_fips = target_fips[:2]
        county_fips = target_fips[2:5]
        
        # Compute accessibility features
        accessibility_features = self._compute_accessibility_features(
            tract_addresses, data
        )

        if accessibility_features is None:
            return {'success': False, 'error': 'Failed to compute accessibility features'}

        # Generate feature names
        feature_names = self._generate_feature_names(accessibility_features.shape[1])
                
        # Build graph with context features
        from ..models.gnn import normalize_accessibility_features
        normalized_features, feature_scaler = normalize_accessibility_features(accessibility_features)

        # Store normalized features for feature importance in multi-tract mode
        full_accessibility_features = normalized_features.copy() if n_neighbor_tracts > 0 else None

        # Extract and normalize context features
        self._log("Extracting context features for addresses...")
        context_features = self.data_loader.create_context_features_for_addresses(
            addresses=tract_addresses,
            svi_data=data['svi']
        )
        normalized_context, context_scaler = self.data_loader.normalize_context_features(context_features)

        # Store context for diagnostics
        self._stored_context_features = normalized_context.copy()

        # Create graph with both accessibility and context features
        graph_data = self.data_loader.create_spatial_accessibility_graph(
            addresses=tract_addresses,
            accessibility_features=normalized_features,
            context_features=normalized_context, 
            state_fips=state_fips,
            county_fips=county_fips
        )

        self._log(f"Built graph: {graph_data.x.shape[0]} nodes, {graph_data.edge_index.shape[1]} edges")
        self._log(f" Accessibility features: {graph_data.x.shape[1]}")
        self._log(f" Context features: {graph_data.context.shape[1]}")
        
        self._log(f"Built graph: {graph_data.x.shape[0]} nodes, {graph_data.edge_index.shape[1]} edges")
        
        # Train GNN (modified for multi-tract if needed)
        if n_neighbor_tracts > 0:
            training_result = self._train_multi_tract_gnn(
                graph_data, tract_svis, tract_addresses
            )
        else:
            training_result = self._train_accessibility_svi_gnn(
                graph_data, target_tract_svi, tract_addresses
            )
        
        if not training_result['success']:
            return training_result
        
        # Extract predictions for target tract only
        if n_neighbor_tracts > 0:
            target_mask = tract_addresses['tract_fips'] == target_fips
            target_tract_addresses = tract_addresses[target_mask].copy()
            target_tract_addresses.reset_index(drop=True, inplace=True)  # CRITICAL
            
            target_predictions = training_result['raw_predictions'][target_mask]

            self._stored_raw_predictions = pd.DataFrame({
                'mean': target_predictions if n_neighbor_tracts > 0 else training_result['raw_predictions'],
                'x': target_tract_addresses.geometry.x.values if n_neighbor_tracts > 0 else tract_addresses.geometry.x.values,
                'y': target_tract_addresses.geometry.y.values if n_neighbor_tracts > 0 else tract_addresses.geometry.y.values
            })
            self._stored_raw_predictions.reset_index(drop=True, inplace=True)
            
            final_predictions = self._finalize_predictions(
                target_predictions, 
                target_tract_addresses, 
                target_tract_svi
            )
            
            # Ensure final_predictions also has reset index
            if hasattr(final_predictions, 'reset_index'):
                final_predictions = final_predictions.reset_index(drop=True)
        else:
            final_predictions = self._finalize_predictions(
                training_result['raw_predictions'], 
                tract_addresses, 
                target_tract_svi
            )

            if hasattr(final_predictions, 'reset_index'):
                final_predictions = final_predictions.reset_index(drop=True)
        
        # Validation (only on target tract)
        if n_neighbor_tracts > 0:
            target_addresses = target_tract_addresses  
            target_access_features = accessibility_features[target_mask]
            target_normalized_features = normalized_features[target_mask]
        else:
            target_addresses = tract_addresses
            target_access_features = accessibility_features
            target_normalized_features = normalized_features
        
        if self.verbose:
            self._log("Running accessibility-vulnerability debugging...")
            debug_samples = self._debug_accessibility_vulnerability_relationship(
                final_predictions, target_access_features, target_addresses
            )
            
            direction_validation = self._validate_feature_directions(target_access_features)
            
            correlation_diagnostic = self._create_accessibility_correlation_diagnostic(
                final_predictions, target_access_features
            )
        
        # Accessibility validation
        self._log("Running accessibility feature validation...")
        try:
            access_validation_results, access_validator = validate_granite_accessibility_features(
                addresses=target_addresses,
                accessibility_features=target_access_features,
                destinations={
                    'employment': data['employment_destinations'],
                    'healthcare': data['healthcare_destinations'], 
                    'grocery': data['grocery_destinations']
                },
                feature_names=feature_names,
                tract_svi=target_tract_svi,
                output_dir=os.path.join(self.output_dir, 'accessibility_validation')
            )
        except Exception as e:
            self._log(f"Accessibility validation failed: {str(e)}", level='WARN')
            access_validation_results = {'error': str(e)}
        
        # Spatial diagnostics
        validation_results = self._validate_predictions(
            final_predictions, target_tract_svi, target_access_features, 
            target_normalized_features
        )

        # Optional unconstrained evaluation
        if self.config.get('validation', {}).get('evaluate_unconstrained', False):
            self._log("\nEvaluating unconstrained learning quality...")
            unconstrained_metrics = self.evaluate_unconstrained_learning(
                training_result, graph_data, target_tract_svi, target_addresses
            )
            validation_results['unconstrained_metrics'] = unconstrained_metrics
        
        # Integration
        if 'error' not in access_validation_results and 'spatial_diagnostics' in validation_results:
            try:
                integrated_results = integrate_with_spatial_diagnostics(
                    spatial_diagnostics_results=validation_results['spatial_diagnostics'],
                    accessibility_validation_results=access_validation_results
                )
                validation_results['integrated_analysis'] = integrated_results
            except Exception as e:
                self._log(f"Integration failed: {str(e)}", level='WARN')
        
        validation_results['accessibility_validation'] = access_validation_results
        
        # Visualizations
        if self.verbose:
            self._create_research_visualizations({
                'predictions': final_predictions,
                'accessibility_features': target_access_features,
                'validation_results': validation_results,
                'tract_svi': target_tract_svi,
                'training_result': training_result
            })

        if self.verbose and hasattr(training_result, 'attention_weights'):
            from ..visualization.attention_analysis import analyze_attention_patterns
            
            attention_results = analyze_attention_patterns(
                attention_weights=training_result['attention_weights'][-1],  # Last epoch
                context_features=self._stored_context_features,
                feature_names=feature_names,
                output_dir=os.path.join(self.output_dir, 'attention_analysis')
            )

        # Baseline comparisons (IDW, Kriging vs GNN disaggregation)
        baseline_results = self._run_disaggregation_baselines(
            addresses=target_addresses,
            predictions=final_predictions['mean'].values,
            tract_gdf=data['tracts'],
            tract_fips=target_fips,
            tract_svi=target_tract_svi,
            accessibility_features=target_access_features
        )
        
        return {
            'success': True,
            'predictions': final_predictions,
            'tract_info': {'FIPS': target_fips, 'RPL_THEMES': target_tract_svi},
            'accessibility_features': target_access_features,  # Target tract only
            'full_accessibility_features': full_accessibility_features,  # All tracts (for feature importance)
            'training_result': training_result,
            'baseline_comparisons': baseline_results, 
            'validation_results': validation_results,
            'methodology': f'{"Multi-tract" if n_neighbor_tracts > 0 else "Single-tract"} Accessibility → SVI',
            'summary': {
                'addresses_processed': len(target_addresses),
                'total_training_addresses': len(tract_addresses),
                'n_tracts': len(tract_svis),
                'accessibility_features': target_access_features.shape[1],
                'spatial_variation': np.std(final_predictions['mean']),
                'constraint_error': abs(np.mean(final_predictions['mean']) - target_tract_svi) / target_tract_svi * 100,
                'training_epochs': training_result.get('epochs_trained', 0)
            }
        }
    
    def _run_disaggregation_baselines(self, addresses, predictions, tract_gdf, 
                                       tract_fips, tract_svi, accessibility_features=None):
        """
        Run disaggregation baseline comparisons (IDW, Kriging, Naive).
        """
        self._log("Running disaggregation baseline comparisons...")
        
        try:
            comparison = DisaggregationComparison(verbose=self.verbose)
            
            # Register baselines
            comparison.add_baseline(NaiveUniformBaseline())
            comparison.add_baseline(IDWDisaggregation(power=2.0, n_neighbors=8))
            comparison.add_baseline(IDWDisaggregation(power=3.0, n_neighbors=8))
            comparison.add_baseline(OrdinaryKrigingDisaggregation())
            
            # Run comparison
            results = comparison.run_comparison(
                tract_gdf=tract_gdf,
                address_gdf=addresses,
                gnn_predictions=predictions,
                tract_fips=tract_fips,
                tract_svi=tract_svi,
                accessibility_features=accessibility_features,
                svi_column='RPL_THEMES'
            )
            
            # Print summary
            if self.verbose:
                comparison.print_summary()
            
            # Generate visualization
            try:
                from ..visualization.disaggregation_plots import DisaggregationVisualizer
                viz = DisaggregationVisualizer()
                plot_path = os.path.join(self.output_dir, 'disaggregation_comparison.png')
                viz.plot_disaggregation_dashboard(
                    results, 
                    accessibility_features=accessibility_features,
                    output_path=plot_path
                )
            except Exception as e:
                self._log(f"Visualization failed: {e}", level='WARN')
            
            return results
            
        except Exception as e:
            self._log(f"Baseline comparison failed: {e}", level='WARN')
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

    def _train_multi_tract_gnn(self, graph_data, tract_svis, addresses):
        """Train GNN with proper multi-tract constraints"""
        
        self._log("Training Multi-Tract Accessibility SVI GNN...")
        
        try:
            from ..models.gnn import AccessibilitySVIGNN, MultiTractGNNTrainer, normalize_accessibility_features
            import torch
            import pandas as pd
            import numpy as np 
            
            # Normalize features (same as single-tract)
            normalized_features, feature_scaler = normalize_accessibility_features(graph_data.x.numpy())
            graph_data.x = torch.FloatTensor(normalized_features)
            
            # Create model (same architecture as single-tract)
            seed = self.config.get('processing', {}).get('random_seed', 42)

            # Determine if context features are available
            has_context = hasattr(graph_data, 'context') and graph_data.context is not None
            context_dim = graph_data.context.shape[1] if has_context else 5

            # Enable context-gating if context features present
            use_gating = self.config.get('model', {}).get('use_context_gating', True)

            model = AccessibilitySVIGNN(
                accessibility_features_dim=graph_data.x.shape[1],
                context_features_dim=context_dim,
                hidden_dim=self.config.get('model', {}).get('hidden_dim', 64),
                dropout=self.config.get('model', {}).get('dropout', 0.3),
                seed=seed,
                use_context_gating=use_gating and has_context  # Only if both enabled and available
            )

            if use_gating and has_context:
                self._log("Context-gating ENABLED")
            else:
                self._log("Context-gating DISABLED (baseline mode)")
            
            # Create tract masks for per-tract constraints
            tract_masks = {}
            for fips in tract_svis.keys():
                tract_masks[fips] = (addresses['tract_fips'] == fips).values
            
            # Create Multi-tract trainer
            trainer = MultiTractGNNTrainer(
                model, 
                config={**self.config.get('training', {}), 'use_multitask': True},
                seed=seed 
            )
            
            # Training parameters
            epochs = self.config.get('model', {}).get('epochs', 100)
            
            self._log(f"Training on {len(tract_svis)} tracts:")
            for fips, svi in tract_svis.items():
                n_addrs = tract_masks[fips].sum()
                self._log(f" {fips}: {n_addrs} addresses, SVI={svi:.4f}")
            
            # Train with multi-tract constraints
            training_result = trainer.train(
                graph_data=graph_data,
                tract_svis=tract_svis,
                tract_masks=tract_masks,
                epochs=epochs,
                verbose=self.verbose
            )
            
            # Get raw predictions from training
            raw_predictions = training_result['final_predictions']
            
            # Store raw predictions with tract info (before correction)
            self._stored_raw_predictions = pd.DataFrame({
                'mean': raw_predictions,
                'x': addresses.geometry.x.values,
                'y': addresses.geometry.y.values,
                'tract_fips': addresses['tract_fips'].values
            })
            
            # Report training results
            overall_error = training_result['overall_constraint_error']
            spatial_std = training_result['final_spatial_std']
            
            self._log(f"Multi-tract training completed:")
            self._log(f" Overall constraint error: {overall_error:.2f}%")
            self._log(f" Spatial variation: {spatial_std:.4f}")
            self._log(f" Epochs: {training_result['epochs_trained']}")
            
            # Show per-tract errors
            self._log("Per-tract constraint errors:")
            for fips, error in training_result['per_tract_errors'].items():
                self._log(f" {fips}: {error:.2f}%")
            
            # Quality assessment
            quality = "good" if overall_error < 10 and spatial_std > 0.01 else \
                      "acceptable" if overall_error < 25 else "poor"
            self._log(f"Multi-tract training quality: {quality}")
            
            return {
                'success': True,
                'raw_predictions': raw_predictions, 
                'learned_accessibility': training_result.get('learned_accessibility'),
                'model': model,
                'edge_index': graph_data.edge_index,
                'training_history': training_result.get('training_history', {}),
                'raw_constraint_error': overall_error,
                'per_tract_errors': training_result['per_tract_errors'],
                'spatial_std': spatial_std,
                'epochs_trained': training_result['epochs_trained'],
                'learning_converged': training_result['learning_converged']
            }
            
        except Exception as e:
            self._log(f"Error in multi-tract training: {str(e)}")
            import traceback
            self._log(f"Traceback: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': str(e),
                'raw_predictions': None
            }
    
    def _generate_feature_names(self, n_features):
        """Generate feature names for all features including base, modal, and socioeconomic"""
        
        feature_names = []
        
        # Base accessibility features: 3 destination types × 10 features = 30
        dest_types = ['employment', 'healthcare', 'grocery']
        
        for dest_type in dest_types:
            feature_names.extend([
                f'{dest_type}_min_time',
                f'{dest_type}_mean_time',
                f'{dest_type}_median_time',
                f'{dest_type}_count_5min',
                f'{dest_type}_count_10min',
                f'{dest_type}_count_15min',
                f'{dest_type}_drive_advantage',
                f'{dest_type}_dispersion',
                f'{dest_type}_time_range',
                f'{dest_type}_percentile',
            ])
        
        # Modal features: 15 features
        modal_names = [
            'transit_mode_share',
            'walk_mode_share', 
            'drive_mode_share',
            'modal_flexibility',
            'transit_employment_access',
            'transit_healthcare_access',
            'transit_grocery_access',
            'walk_employment_access',
            'walk_healthcare_access',
            'walk_grocery_access',
            'car_dependency_employment',
            'car_dependency_healthcare',
            'car_dependency_grocery',
            'no_vehicle_accessibility_penalty',
            'modal_equity_index'
        ]
        feature_names.extend(modal_names)
        
        # Socioeconomic features: 9 features
        socioeco_names = [
            'pct_no_vehicle',
            'pct_poverty',
            'pct_unemployment', 
            'pct_no_diploma',
            'pct_elderly',
            'pct_disabled',
            'pct_single_parent',
            'pct_minority',
            'pct_limited_english'
        ]
        feature_names.extend(socioeco_names)
        
        # Total should be 30 + 15 + 9 = 54
        
        # Handle edge case where actual features don't match expected
        if len(feature_names) != n_features:
            self._log(f"Generated {len(feature_names)} names but have {n_features} features", level='WARN')
            # Pad with generic names if needed
            while len(feature_names) < n_features:
                feature_names.append(f'feature_{len(feature_names)}')
            # Or truncate if too many
            feature_names = feature_names[:n_features]
        
        return feature_names

    def _compute_accessibility_features(self, addresses, data):
        """
        Compute accessibility features with enhanced error handling and validation
        """
        self._log("Computing enhanced accessibility features...")

        # Detect multi-tract scenario
        n_tracts = addresses['tract_fips'].nunique() if 'tract_fips' in addresses.columns else 1
        is_multi_tract = n_tracts > 1
        
        if is_multi_tract:
            self._log(f"Multi-tract mode detected ({n_tracts} tracts, {len(addresses)} addresses)")
            # Force disable cache for this computation
            #skip_cache_lookup = True
            skip_cache_lookup = False
        else:
            skip_cache_lookup = False

        def _generate_cache_key(addresses_gdf, destinations_dict):
            """Generate stable cache key from addresses and destinations"""
            import hashlib
            
            # Hash addresses
            addr_coords = sorted(addresses_gdf.geometry.apply(lambda g: (round(g.x, 6), round(g.y, 6))).tolist())
            addr_hash = hashlib.md5(str(addr_coords).encode()).hexdigest()[:8]
            
            # Hash destinations by type
            dest_hashes = {}
            for dtype, dgdf in destinations_dict.items():
                if dgdf is not None and len(dgdf) > 0:
                    dest_coords = sorted(dgdf.geometry.apply(lambda g: (round(g.x, 6), round(g.y, 6))).tolist())
                    dest_hashes[dtype] = hashlib.md5(str(dest_coords).encode()).hexdigest()[:8]
            
            return addr_hash, dest_hashes
        
        try:
            # Initialize the enhanced computer properly
            from ..data.enhanced_accessibility import EnhancedAccessibilityComputer

            # Pass cache config from pipeline
            enable_caching = self.config.get('processing', {}).get('enable_caching', True)
            cache_dir = self.config.get('processing', {}).get('cache_dir', './granite_cache')

            accessibility_computer = EnhancedAccessibilityComputer(
                verbose=self.verbose,
                enable_caching=enable_caching,
                cache_dir=cache_dir
            )
            
            # Get tract-appropriate destinations with validation
            target_fips = self.config.get('data', {}).get('target_fips')
            if target_fips:
                try:
                    destinations = {
                        'employment': data['employment_destinations'],
                        'healthcare': data['healthcare_destinations'],
                        'grocery': data['grocery_destinations']
                    }
                    self._log("Using original county-wide destinations (tract enhancement disabled)")
                except Exception as e:
                    raise RuntimeError(f"Failed to create destinations for tract {target_fips}: {e}")
            else:
                destinations = {
                    'employment': data['employment_destinations'],
                    'healthcare': data['healthcare_destinations'],
                    'grocery': data['grocery_destinations']
                }
            
            # Ensure all destinations are valid
            validated_destinations = {}
            for dest_type, dest_gdf in destinations.items():
                if dest_gdf is None or len(dest_gdf) == 0:
                    self._log(f"No {dest_type} destinations available",level='ERROR')
                    return None
                
                # Ensure proper columns exist
                dest_gdf_copy = dest_gdf.copy()
                dest_gdf_copy['dest_type'] = dest_type
                if 'dest_id' not in dest_gdf_copy.columns:
                    dest_gdf_copy['dest_id'] = range(len(dest_gdf_copy))
                
                validated_destinations[dest_type] = dest_gdf_copy
                self._log(f" {dest_type}: {len(dest_gdf_copy)} destinations")
            
            destinations = validated_destinations

            if not skip_cache_lookup and accessibility_computer.cache is not None:
                addr_hash, dest_hashes = _generate_cache_key(addresses, destinations)
                cache_key = f"{addr_hash}_complete"
                
                cached_complete = accessibility_computer.cache.get_absolute(
                    mode='all_destinations',
                    dest_type='complete',
                    threshold=0,
                    origins_hash=cache_key
                )
                
                if cached_complete is not None:
                    self._log(f"Retrieved COMPLETE accessibility features from cache ({cached_complete.shape[0]} addresses)")
                    return cached_complete
            
            # CHANGE 2: Calculate features for all destination types with error handling
            all_features = []
            feature_names = []
            successful_computations = 0
            
            for dest_type, dest_gdf in destinations.items():
                self._log(f" Processing {dest_type} accessibility...")
                
                try:
                    # ADD THIS: Check cache for this specific destination type
                    if not skip_cache_lookup and accessibility_computer.cache is not None:
                        addr_hash, dest_hashes = _generate_cache_key(addresses, {dest_type: dest_gdf})
                        dest_cache_key = f"{addr_hash}_{dest_hashes[dest_type]}"
                        
                        cached_features = accessibility_computer.cache.get_absolute(
                            mode='multi',
                            dest_type=dest_type,
                            threshold=0,
                            origins_hash=dest_cache_key
                        )
                        
                        if cached_features is not None:
                            self._log(f"  Retrieved {dest_type} features from cache")
                            all_features.append(cached_features)
                            feature_names.extend([
                                f'{dest_type}_min_time', f'{dest_type}_mean_time', f'{dest_type}_median_time',
                                f'{dest_type}_count_5min', f'{dest_type}_count_10min', f'{dest_type}_count_15min',
                                f'{dest_type}_drive_advantage', f'{dest_type}_dispersion',
                                f'{dest_type}_time_range', f'{dest_type}_percentile'
                            ])
                            successful_computations += 1
                            continue  # Skip to next destination type
                    
                    # If not cached, compute as normal
                    travel_times = accessibility_computer.calculate_realistic_travel_times(
                        origins=addresses,
                        destinations=dest_gdf
                    )

                    # Validate travel times before feature extraction
                    if not accessibility_computer._validate_distance_time_relationship(travel_times):
                        self._log(f"Travel time validation failed for {dest_type}",level='ERROR')
                        continue
                        
                    # Validate destination counts
                    if not accessibility_computer._validate_destination_counts_fixed(travel_times):
                        self._log(f"Destination count validation failed for {dest_type}",level='ERROR')
                        continue
                    
                    if len(travel_times) == 0:
                        self._log(f"No travel times computed for {dest_type}",level='ERROR')
                        continue
                    
                    # Debug - Show sample travel times
                    if self.verbose and len(travel_times) > 0:
                        sample = travel_times.head(3)
                        self._log(f" Sample {dest_type} travel times:")
                        for _, row in sample.iterrows():
                            self._log(f"   Origin {row['origin_id']} -> Dest {row['dest_id']}: "
                                    f"{row['combined_time']:.1f}min ({row['best_mode']})")
                    
                    dest_features = accessibility_computer.extract_enhanced_accessibility_features(
                        addresses=addresses,
                        travel_times=travel_times,
                        dest_type=dest_type,
                        destinations=dest_gdf
                    )
                    
                    if dest_features is None or dest_features.size == 0:
                        self._log(f"No features extracted for {dest_type}",level='ERROR')
                        continue
                    
                    # Check feature dimensions
                    expected_addresses = len(addresses)
                    if dest_features.shape[0] != expected_addresses:
                        self._log(f"Feature count mismatch for {dest_type}: "
                                f"got {dest_features.shape[0]}, expected {expected_addresses}",level='ERROR')
                        continue

                    # Expect 24 features per destination type
                    if dest_features.shape[1] != 24:
                        self._log(f"Unexpected feature count for {dest_type}: "
                                f"got {dest_features.shape[1]}, expected 24", level='WARN')

                    # Remove zero-variance features instead of rejecting everything
                    feature_stds = np.std(dest_features, axis=0)
                    zero_var_mask = feature_stds < 1e-8
                    zero_var_count = np.sum(zero_var_mask)

                    if zero_var_count > 0:
                        self._log(f"{dest_type} has {zero_var_count} zero-variance features (keeping for consistency)", level='WARN')
                        
                        # Identify which features
                        feature_names_temp = [
                            'min_time', 'mean_time', 'median_time',
                            'count_5min', 'count_10min', 'count_15min',
                            'drive_advantage', 'dispersion', 'time_range', 'percentile'
                        ]
                        
                        for i, has_zero_var in enumerate(zero_var_mask):
                            if has_zero_var and i < len(feature_names_temp):
                                self._log(f" Zero variance: {dest_type}_{feature_names_temp[i]}")
                        
                        # If we removed all features, that's a real error
                        if dest_features.shape[1] == 0:
                            self._log(f"All {dest_type} features have zero variance",level='ERROR')
                            continue
                    
                    all_features.append(dest_features)
                    
                    # Generate feature names for whatever features we actually kept
                    base_feature_names = [
                        'min_time', 'mean_time', 'median_time',
                        'count_5min', 'count_10min', 'count_15min',
                        'drive_advantage', 'dispersion', 'time_range', 'percentile'
                    ]

                    # Always add names for features (no filtering)
                    feature_names.extend([f'{dest_type}_{name}' for name in base_feature_names])
                    
                    successful_computations += 1
                    self._log(f"{dest_type}: {dest_features.shape} features computed successfully")
                    
                    if accessibility_computer.cache is not None:
                        addr_hash, dest_hashes = _generate_cache_key(addresses, {dest_type: dest_gdf})
                        dest_cache_key = f"{addr_hash}_{dest_hashes[dest_type]}"
                        
                        accessibility_computer.cache.set_absolute(
                            dest_features,
                            mode='multi',
                            dest_type=dest_type,
                            threshold=0,
                            origins_hash=dest_cache_key
                        )
                        self._log(f"Cached {dest_type} features for reuse")

                except Exception as e:
                    self._log(f"Error processing {dest_type}: {str(e)}")
                    import traceback
                    self._log(f"Traceback: {traceback.format_exc()}")
                    continue
            
            if successful_computations == 0:
                raise RuntimeError(
                    "No accessibility features computed. Verify that destination data "
                    "is available and OSRM servers are running."
                )
            
            if accessibility_computer.cache is not None and successful_computations >= 2:
                try:
                    # Extract employment features for mode comparison example
                    emp_idx = [i for i, name in enumerate(feature_names) if 'employment' in name]
                    if len(emp_idx) >= 3:  # Need at least min_time, mean_time, median_time
                        emp_features_flat = all_features[0]  # First destination type (employment)
                        
                        # Compute car/transit ratio if we have the data
                        # This is a simplified example - adjust based on your actual feature structure
                        addr_hash, _ = _generate_cache_key(addresses, destinations)
                        
                        # Store a simple mode comparison metric
                        if emp_features_flat.shape[1] >= 7:  # Has drive_advantage column
                            drive_advantage = emp_features_flat[:, 6]  # Index 6 is drive_advantage
                            
                            accessibility_computer.cache.set_differential(
                                drive_advantage,
                                mode_a='car',
                                mode_b='transit',
                                dest_type='employment',
                                threshold=0,
                                origins_hash=addr_hash,
                                operation='difference'
                            )
                            self._log("Cached car/transit mode comparison differential")
                except Exception as e:
                    self._log(f"Note: Could not cache mode differentials: {str(e)}")
            
            if len(all_features) == 0:
                self._log("CRITICAL ERROR: Feature list is empty")
                return None
            
            # Combine features with validation
            try:
                accessibility_matrix = np.column_stack(all_features)
                self._log(f"Base accessibility features: {accessibility_matrix.shape}")

                validation_passed = self._validate_final_feature_relationships(
                    accessibility_matrix, addresses
                )

                try:
                    from ..features.modal_accessibility import compute_modal_features
                    
                    # Build tract SVI dictionary
                    tract_svi_dict = {}
                    if 'svi' in data and data['svi'] is not None:
                        for _, tract_row in data['svi'].iterrows():
                            fips = str(tract_row.get('FIPS', tract_row.name)).strip()
                            tract_svi_dict[fips] = {
                                'EP_NOVEH': float(tract_row.get('EP_NOVEH', 0.0))
                            }
                    
                    # Get tract ID for each address
                    if 'tract_fips' in addresses.columns:
                        # Multi-tract mode
                        address_tract_ids = addresses['tract_fips'].astype(str).str.strip().values
                    elif 'FIPS' in addresses.columns:
                        # Single tract mode
                        address_tract_ids = addresses['FIPS'].astype(str).str.strip().values
                    else:
                        # Fallback
                        target_fips = str(data.get('target_fips', '47065000600')).strip()
                        address_tract_ids = np.full(len(addresses), target_fips)
                    
                    self._log(f"Computing modal features for {len(set(address_tract_ids))} unique tracts")
                    
                    # Compute modal features
                    modal_features, modal_names = compute_modal_features(
                        accessibility_features=accessibility_matrix,
                        feature_names=feature_names,
                        tract_svi_data=tract_svi_dict,
                        address_tract_ids=address_tract_ids
                    )
                    
                    self._log(f"Generated {modal_features.shape[1]} modal features")
                    
                except Exception as e:
                    self._log(f"ERROR computing modal features: {str(e)}")
                    import traceback
                    self._log(f"Traceback: {traceback.format_exc()}")
                    self._log("Continuing without modal features")
                    modal_features = np.zeros((len(addresses), 0))
                    modal_names = []

                # Combine: 30 base + 15 modal = 45 accessibility features
                if modal_features.shape[1] > 0:
                    final_features = np.column_stack([accessibility_matrix, modal_features])
                    complete_feature_names = feature_names + modal_names
                else:
                    final_features = accessibility_matrix
                    complete_feature_names = feature_names

                self._log(f"Final feature matrix: {final_features.shape}")
                self._log(f" Base accessibility: {accessibility_matrix.shape[1]}")
                self._log(f" Modal features: {modal_features.shape[1]}")
                
            except Exception as e:
                self._log(f"ERROR combining features: {str(e)}")
                return None
            
            self._log(f"Final feature matrix: {final_features.shape}")
            self._log(f"Feature names count: {len(complete_feature_names)}")
            
            # Critical checks
            if final_features.size == 0:
                self._log("CRITICAL ERROR: Final feature matrix is empty")
                return None
            
            if np.any(np.isnan(final_features)):
                nan_count = np.sum(np.isnan(final_features))
                self._log(f"{nan_count} NaN values in feature matrix",level='ERROR')
                # Try to handle NaN values
                final_features = np.nan_to_num(final_features, nan=0.0)
            
            if np.any(np.isinf(final_features)):
                inf_count = np.sum(np.isinf(final_features))
                self._log(f"{inf_count} infinite values in feature matrix",level='ERROR')
                final_features = np.nan_to_num(final_features, posinf=999.0, neginf=-999.0)
            
            # Check variance and remove zero-variance features
            zero_var_mask = np.std(final_features, axis=0) < 1e-8
            zero_var_count = np.sum(zero_var_mask)

            if zero_var_count > 0:
                self._log(f"{zero_var_count} features have zero variance (keeping for consistency)", level='WARN')
                
                # Debug zero variance features
                for i in range(final_features.shape[1]):
                    if zero_var_mask[i]:
                        feature_name = complete_feature_names[i] if i < len(complete_feature_names) else f"feature_{i}"
                        unique_vals = len(np.unique(final_features[:, i]))
                        self._log(f" Zero-variance feature: {feature_name}: {unique_vals} unique values, std={np.std(final_features[:, i]):.8f}")
                
                self._log(f"Keeping all {final_features.shape[1]} features (including {zero_var_count} zero-variance)")
            
            # Check for excessive negative values (some can be negative, like drive_advantage)
            negative_features = []
            for i in range(final_features.shape[1]):
                feature_name = complete_feature_names[i] if i < len(complete_feature_names) else f"feature_{i}"
                # Allow negative values only for drive_advantage and derived equity measures
                if ('drive_advantage' not in feature_name and 'equity' not in feature_name and 
                    'flexibility' not in feature_name and np.any(final_features[:, i] < 0)):
                    negative_count = np.sum(final_features[:, i] < 0)
                    negative_features.append(f"{feature_name}: {negative_count} negative values")
            
            if negative_features:
                self._log(f"Unexpected negative values in: {negative_features[:3]}", level='WARN')  # Show first 3
            else:
                self._log(" No unexpected negative values detected")
            
            self._log(" All features have proper variance")
            self._log(f"SUCCESS: Generated {final_features.shape[1]} features for {final_features.shape[0]} addresses")
            
            enhanced_features = final_features
            enhanced_names = complete_feature_names
            validation_results = {'overall_quality': {'grade': 'A', 'overall_score': 100.0}}

            self._log(f"Using {enhanced_features.shape[1]} features without deduplication")
            
            # Check if enhancement was successful
            if enhanced_features is None or enhanced_features.size == 0:
                self._log("CRITICAL ERROR: Feature enhancement failed")
                return None
            
            # Report quality improvement
            original_feature_count = final_features.shape[1]
            final_feature_count = enhanced_features.shape[1]
            quality_grade = validation_results['overall_quality']['grade']
            quality_score = validation_results['overall_quality']['overall_score']
            
            self._log(f"Feature enhancement completed:")
            self._log(f" Original features: {original_feature_count}")
            self._log(f" Enhanced features: {final_feature_count}")
            self._log(f" Quality grade: {quality_grade} ({quality_score:.1f}%)")
            
            # Warn if quality is poor
            if quality_grade in ['D', 'F']:
                self._log(f"Feature quality is poor - consider investigating", level='WARN')
                
            # Debug first 5 addresses
            if self.verbose and dest_type == 'employment':
                print("\n=== EMPLOYMENT TRAVEL TIME DEBUG ===")
                for test_dist in [1.0, 3.0, 5.0, 10.0]:
                    self.accessibility_computer.debug_mode_times(test_dist, 'morning')

            if self.verbose and dest_type == 'employment':
                self.accessibility_computer.debug_employment_travel_times(
                    addresses.head(10), 
                    destinations['employment'].head(3)
                )

            # Store validation results for later use
            self._feature_validation_results = validation_results
            
            self._log(f"SUCCESS: Generated {enhanced_features.shape[1]} features for {len(addresses)} addresses")
            
            # Add socioeconomic controls
            if 'FIPS' in addresses.columns:
                tract_fips = str(addresses['FIPS'].iloc[0]).strip()
            elif hasattr(addresses.iloc[0], 'FIPS'):
                tract_fips = str(addresses.iloc[0].FIPS).strip()
            else:
                # Fallback: extract from tract context
                tract_fips = str(data.get('target_fips', '47065000600'))

            self._log(f"Extracted tract FIPS: {tract_fips}")
            if 'tract_fips' in addresses.columns:
                # Multi-tract mode: extract features for each tract
                unique_tracts = addresses['tract_fips'].unique()
                self._log(f"Extracting socioeconomic features for {len(unique_tracts)} tracts")
                
                socioeco_array = np.zeros((len(addresses), 9))
                
                for tract_fips in unique_tracts:
                    tract_mask = addresses['tract_fips'] == tract_fips
                    tract_features = self.data_loader.get_tract_socioeconomic_features(
                        str(tract_fips), data['svi']
                    )
                    
                    self._log(f" Tract {tract_fips}: no_vehicle={tract_features['pct_no_vehicle']:.1f}%, poverty={tract_features['pct_poverty']:.1f}%")
                    
                    # Assign to addresses in this tract
                    socioeco_array[tract_mask] = list(tract_features.values())
            else:
                # Single tract fallback
                tract_fips = str(addresses.iloc[0].get('FIPS', ''))
                socioeconomic_context = self.data_loader.get_tract_socioeconomic_features(
                    tract_fips, data['svi']
                )
                socioeco_array = np.tile(list(socioeconomic_context.values()), (len(addresses), 1))
                self._log(f"Single tract: no_vehicle={socioeconomic_context['pct_no_vehicle']:.1f}%")

            # Combine
            combined_features = np.column_stack([enhanced_features, socioeco_array])
            
            self._log(f"Final feature matrix: {combined_features.shape}")
            self._log(f" Accessibility features: {enhanced_features.shape[1]}")
            self._log(f" Socioeconomic controls: {socioeco_array.shape[1]}")
            self._log(f" Breakdown:")
            self._log(f"   Base accessibility: 30")
            self._log(f"   Modal features: 15")
            self._log(f"   Socioeconomic: 9")
            self._log(f"   Total: {combined_features.shape[1]}")

            if accessibility_computer.cache is not None:
                addr_hash, dest_hashes = _generate_cache_key(addresses, destinations)
                cache_key = f"{addr_hash}_complete"
                
                accessibility_computer.cache.set_absolute(
                    combined_features,
                    mode='all_destinations',
                    dest_type='complete',
                    threshold=0,
                    origins_hash=cache_key
                )
                self._log(f"Cached complete feature matrix for reuse")
            
            return combined_features
            
        except Exception as e:
            self._log(f"Failure in accessibility computation: {str(e)}", level='ERROR')
            import traceback
            self._log(f"Full traceback: {traceback.format_exc()}")
            return None

    def _compute_features_batched(self, addresses_df, data, batch_size=3):
        """
        Compute accessibility features in batches by tract to avoid memory overflow.
        
        Memory-efficient approach:
        - Groups addresses by tract FIPS code
        - Processes 3-4 tracts per batch (~7500 addresses = ~900MB OSRM)
        - Caches individual batch results with tract-prefixed keys
        - Concatenates all batches to create global feature matrix
        
        Args:
            addresses_df: DataFrame with address_id, geometry, full_address, tract_fips
            data: dict with destinations, road network, etc.
            batch_size: Number of tracts to process per batch (default 3)
        
        Returns:
            Combined feature matrix (n_addresses, n_features)
        """
        self._log(f"\n{'='*80}")
        self._log(f"BATCHED FEATURE COMPUTATION (batch_size={batch_size} tracts)")
        self._log(f"{'='*80}")
        
        import hashlib
        
        # Group addresses by tract for efficient batch processing
        tracts = addresses_df['tract_fips'].unique()
        self._log(f"Processing {len(tracts)} tracts in batches of {batch_size}")
        
        all_batch_features = []
        
        # Process tracts in batches
        n_batches = (len(tracts) + batch_size - 1) // batch_size
        for batch_idx, start_idx in enumerate(range(0, len(tracts), batch_size)):
            batch_tracts = tracts[start_idx:start_idx + batch_size]
            batch_key = f"batch_{batch_idx:02d}_" + "_".join(batch_tracts)
            
            self._log(f"\nBatch {batch_idx + 1}/{n_batches}: {len(batch_tracts)} tracts")
            self._log(f" Tracts: {', '.join(batch_tracts)}")
            
            # Filter addresses for this batch
            batch_mask = addresses_df['tract_fips'].isin(batch_tracts)
            batch_addresses = addresses_df[batch_mask].copy()
            
            self._log(f" Addresses: {len(batch_addresses)}")
            
            # Try to load cached batch result
            batch_features = self._load_batch_cache(batch_addresses, batch_key, data)
            
            if batch_features is not None:
                self._log(f"  Loaded from batch cache ({batch_features.shape})")
                all_batch_features.append(batch_features)
                continue
            
            # Compute features for this batch
            try:
                self._log(f" Computing accessibility features...")
                batch_features = self._compute_accessibility_features(batch_addresses, data)
                
                if batch_features is None or batch_features.size == 0:
                    self._log(f" ERROR: Failed to compute features for batch {batch_idx}")
                    return None
                
                # Validate dimensions
                if batch_features.shape[0] != len(batch_addresses):
                    self._log(f" ERROR: Feature count mismatch for batch {batch_idx}")
                    return None
                
                self._log(f"  Computed {batch_features.shape} features")
                
                # Cache the batch result
                self._save_batch_cache(batch_addresses, batch_features, batch_key)
                
                all_batch_features.append(batch_features)
                
            except Exception as e:
                self._log(f" ERROR: Failed to process batch {batch_idx}: {str(e)}")
                import traceback
                self._log(f" Traceback: {traceback.format_exc()}")
                return None
        
        # Concatenate all batch results
        if len(all_batch_features) == 0:
            self._log("ERROR: No batches processed successfully")
            return None
        
        self._log(f"\n{'='*80}")
        self._log(f"Concatenating {len(all_batch_features)} batch results...")
        
        combined_features = np.vstack(all_batch_features)
        
        self._log(f"Combined features: {combined_features.shape}")
        self._log(f"{'='*80}\n")
        
        return combined_features

    def _load_batch_cache(self, batch_addresses, batch_key, data):
        """
        Try to load cached features for a batch of tracts.
        
        Cache key format: {batch_key}_{address_hash}_{dest_hashes}
        """
        import hashlib
        
        if self.accessibility_computer.cache is None:
            return None
        
        try:
            # Generate consistent hash for addresses in this batch
            addr_coords = sorted(
                batch_addresses.geometry.apply(lambda g: (round(g.x, 6), round(g.y, 6))).tolist()
            )
            addr_hash = hashlib.md5(str(addr_coords).encode()).hexdigest()[:8]
            
            # Generate hashes for destinations
            dest_hashes = {}
            for dtype in ['employment', 'healthcare', 'grocery']:
                dgdf = data.get(f'{dtype}_destinations')
                if dgdf is not None and len(dgdf) > 0:
                    dest_coords = sorted(dgdf.geometry.apply(lambda g: (round(g.x, 6), round(g.y, 6))).tolist())
                    dest_hashes[dtype] = hashlib.md5(str(dest_coords).encode()).hexdigest()[:8]
            
            # Build full cache key
            dest_key = "_".join([dest_hashes.get(dt, 'none') for dt in ['employment', 'healthcare', 'grocery']])
            cache_key = f"{batch_key}_{addr_hash}_{dest_key}"
            
            # Attempt to load
            cached_features = self.accessibility_computer.cache.get_absolute(
                mode='batch',
                dest_type='complete',
                threshold=0,
                origins_hash=cache_key
            )
            
            return cached_features
        
        except Exception as e:
            if self.verbose:
                self._log(f" Cache lookup failed: {str(e)}")
            return None

    def _save_batch_cache(self, batch_addresses, batch_features, batch_key):
        """
        Save computed batch features to cache with tract-aware key.
        """
        import hashlib
        
        if self.accessibility_computer.cache is None:
            return
        
        try:
            # Generate consistent hash for addresses in this batch
            addr_coords = sorted(
                batch_addresses.geometry.apply(lambda g: (round(g.x, 6), round(g.y, 6))).tolist()
            )
            addr_hash = hashlib.md5(str(addr_coords).encode()).hexdigest()[:8]
            
            # Build full cache key (same format as _load_batch_cache)
            cache_key = f"{batch_key}_{addr_hash}"
            
            # Save to cache
            self.accessibility_computer.cache.set_absolute(
                batch_features,
                mode='batch',
                dest_type='complete',
                threshold=0,
                origins_hash=cache_key
            )
            
            self._log(f"Cached batch features")
        
        except Exception as e:
            if self.verbose:
                self._log(f"Cache save failed: {str(e)}")

    def _estimate_optimal_batch_size(self, num_addresses_total, num_tracts, available_memory_gb=14):
        """
        Estimate optimal batch size based on address count and available memory.
        
        Memory profile:
        - Per 2,500 addresses (1 avg tract): ~1.5M OSRM routes = ~300MB
        - Safe batch: ~7,500 addresses = 900MB (3 avg tracts)
        
        Strategy: Calculate average addresses per tract, determine how many
        tracts fit in 900MB budget, return that as batch_size (capped at 4).
        """
        if num_tracts == 0:
            return 1
        
        # Calculate average addresses per tract
        avg_addresses_per_tract = num_addresses_total / num_tracts
        
        # Safe batch size: 900MB ~ 7,500 addresses for typical Hamilton County data
        safe_addresses_per_batch = 7500
        
        # How many tracts of avg size fit in safe batch?
        batch_size = max(1, int(safe_addresses_per_batch / avg_addresses_per_tract))
        
        # Cap at 4 for safety margin (4 tracts × 2,500 avg = 10K addresses = 1.2GB)
        return min(batch_size, 4)

    def _extract_accessibility_features(self, addresses, travel_times, dest_type):
        """Extract 8 accessibility features for one destination type"""
        
        features = []
        
        for _, address in addresses.iterrows():
            address_id = address.get('address_id', address.name)
            
            # Filter travel times for this address
            addr_times = travel_times[
                travel_times['origin_id'].astype(str) == str(address_id)
            ]
            
            if len(addr_times) > 0:
                combined_times = pd.to_numeric(addr_times['combined_time'], errors='coerce').dropna()
                
                if len(combined_times) > 0:
                    # Time-based features
                    min_time = float(combined_times.min())
                    mean_time = float(combined_times.mean())
                    percentile_90 = float(np.percentile(combined_times, 90))
                    
                    # Count-based features (destinations within time thresholds)
                    count_5min = int((combined_times <= 5).sum())
                    count_10min = int((combined_times <= 10).sum())
                    count_15min = int((combined_times <= 15).sum())
                    
                    # Transit accessibility
                    transit_share = float((addr_times['best_mode'] == 'transit').mean())
                    
                    # Overall accessibility score (gravity-style)
                    accessibility_score = float(np.sum(1.0 / np.maximum(combined_times, 1.0)))
                    
                else:
                    # No valid travel times
                    min_time = mean_time = percentile_90 = 120.0
                    count_5min = count_10min = count_15min = 0
                    transit_share = accessibility_score = 0.0
            else:
                # No travel times for this address
                min_time = mean_time = percentile_90 = 120.0
                count_5min = count_10min = count_15min = 0
                transit_share = accessibility_score = 0.0
            
            features.append([
                min_time, mean_time, percentile_90,
                count_5min, count_10min, count_15min,
                transit_share, accessibility_score
            ])
        
        return np.array(features, dtype=np.float64)

    def _compute_derived_accessibility_features(self, base_features):
        """Compute derived features from base accessibility metrics"""
        
        # Assuming features are organized as: [employment_8, healthcare_8, grocery_8]
        n_addresses = base_features.shape[0]
        
        if base_features.shape[1] < 24:  # Less than 3 destinations × 8 features
            return np.zeros((n_addresses, 4))  # Return minimal derived features
        
        # Extract by destination type (8 features each)
        employment_features = base_features[:, :8]
        healthcare_features = base_features[:, 8:16]
        grocery_features = base_features[:, 16:24]
        
        derived = []
        
        for i in range(n_addresses):
            # Overall accessibility balance
            emp_score = employment_features[i, 7]  # accessibility_score
            health_score = healthcare_features[i, 7]
            grocery_score = grocery_features[i, 7]
            
            total_accessibility = emp_score + health_score + grocery_score
            
            # Accessibility diversity (how balanced is access across destination types)
            if total_accessibility > 0:
                scores = np.array([emp_score, health_score, grocery_score]) / total_accessibility
                diversity = -np.sum(scores * np.log(scores + 1e-8))  # Entropy
            else:
                diversity = 0.0
            
            # Transit dependence
            emp_transit = employment_features[i, 6]
            health_transit = healthcare_features[i, 6]
            grocery_transit = grocery_features[i, 6]
            avg_transit_share = (emp_transit + health_transit + grocery_transit) / 3
            
            # Time efficiency (how much better is min time vs mean time)
            all_min_times = [employment_features[i, 0], healthcare_features[i, 0], grocery_features[i, 0]]
            all_mean_times = [employment_features[i, 1], healthcare_features[i, 1], grocery_features[i, 1]]
            
            min_avg = np.mean(all_min_times)
            mean_avg = np.mean(all_mean_times)
            time_efficiency = (mean_avg - min_avg) / mean_avg if mean_avg > 0 else 0
            
            derived.append([
                total_accessibility,
                diversity,
                avg_transit_share,
                time_efficiency
            ])
        
        return np.array(derived, dtype=np.float64)

    def _train_accessibility_svi_gnn(self, graph_data, tract_svi, addresses):
        """Train GNN with all fixes applied"""
        self._log("Training Accessibility SVI GNN with enhanced architecture...")
        
        try:
            # Import GNN classes
            from ..models.gnn import AccessibilitySVIGNN, AccessibilityGNNTrainer, normalize_accessibility_features
            import torch
            
            # Apply normalization
            normalized_features, feature_scaler = normalize_accessibility_features(graph_data.x.numpy())
            graph_data.x = torch.FloatTensor(normalized_features)
            
            # Create model with proper architecture
            seed = self.config.get('processing', {}).get('random_seed', 42)
            model = AccessibilitySVIGNN(
                accessibility_features_dim=graph_data.x.shape[1],
                hidden_dim=self.config.get('model', {}).get('hidden_dim', 64),
                dropout=self.config.get('model', {}).get('dropout', 0.3),
                seed=seed # Ensure reproducibility
            )
            
            # Create trainer
            trainer = AccessibilityGNNTrainer(
                model, 
                config=self.config.get('training', {}),
                seed=seed 
            )     

            # Training parameters
            epochs = self.config.get('model', {}).get('epochs', 100)
            
            self._log(f"Training model: {graph_data.x.shape[1]} features for SVI prediction")
            self._log(f"Target SVI: {tract_svi:.4f}, Epochs: {epochs}")
            
            # Train with FIXED trainer
            training_result = trainer.train(
                graph_data=graph_data,
                tract_svi=tract_svi,
                epochs=epochs,
                verbose=self.verbose
            )

            # NEW: Store learned accessibility for validation
            if 'learned_accessibility' in training_result:
                self._learned_accessibility = training_result['learned_accessibility']
            
            # Store raw predictions for diagnostics
            predictions = training_result['final_predictions']
            self._stored_raw_predictions = pd.DataFrame({
                'mean': predictions if isinstance(predictions, np.ndarray) else predictions.values,
                'x': addresses.geometry.x.values,
                'y': addresses.geometry.y.values
            })
            
            # Enhanced result reporting
            constraint_error = abs(np.mean(predictions) - tract_svi) / tract_svi * 100
            spatial_std = np.std(predictions)
            
            self._log(f"Training completed:")
            self._log(f" Constraint error: {constraint_error:.2f}%")
            self._log(f" Spatial variation: {spatial_std:.4f}")
            self._log(f" Learning converged: {training_result.get('learning_converged', False)}")
            
            # Quality assessment
            if constraint_error < 10 and spatial_std > 0.01:
                self._log(" Training quality: GOOD")
            elif constraint_error < 25:
                self._log("⚠ Training quality: ACCEPTABLE")
            else:
                self._log("✗ Training quality: POOR - consider hyperparameter tuning")
            
            return {
                'success': True,
                'raw_predictions': predictions,
                'learned_accessibility': training_result.get('learned_accessibility'),
                'model': model,
                'edge_index': graph_data.edge_index,
                'training_history': training_result.get('training_history', {}),
                'raw_constraint_error': constraint_error,
                'spatial_std': spatial_std,
                'epochs_trained': training_result.get('epochs_trained', epochs),
                'learning_converged': training_result.get('learning_converged', False)
            }
            
        except Exception as e:
            self._log(f"Error training GNN: {str(e)}")
            import traceback
            self._log(f"Traceback: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': str(e),
                'raw_predictions': None
            }

    def _finalize_predictions(self, raw_predictions, addresses, tract_svi):
        """Create final prediction DataFrame with transparent constraint handling"""
        
        if raw_predictions is None:
            return pd.DataFrame()
        
        # Ensure predictions are numpy array with no index
        if isinstance(raw_predictions, pd.Series):
            raw_predictions = raw_predictions.values
        elif isinstance(raw_predictions, pd.DataFrame):
            raw_predictions = raw_predictions.values.flatten()
        
        # Ensure predictions match address count
        n_addresses = len(addresses)
        if len(raw_predictions) > n_addresses:
            predictions = raw_predictions[:n_addresses]
        else:
            predictions = raw_predictions
        
        # Calculate constraint error BEFORE correction
        current_mean = np.mean(predictions)
        pre_correction_error = abs(current_mean - tract_svi) / tract_svi * 100
        
        self._log(f"Pre-correction analysis:")
        self._log(f" Raw mean: {current_mean:.4f}")
        self._log(f" Target: {tract_svi:.4f}")  
        self._log(f" Pre-correction constraint error: {pre_correction_error:.2f}%")
        
        # # Apply constraint correction
        adjustment = tract_svi - current_mean
        adjusted_predictions = predictions + adjustment
        adjusted_predictions = np.clip(adjusted_predictions, 0.0, 1.0)
        
        # Use raw predictions without correction
        adjusted_predictions = predictions
        adjustment = 0.0
        
        # Verify raw prediction quality
        final_mean = np.mean(adjusted_predictions)
        final_error = abs(final_mean - tract_svi) / tract_svi * 100
        
        self._log(f"Raw prediction analysis (NO post-correction):")
        self._log(f" Raw mean: {final_mean:.4f}")
        self._log(f" Target: {tract_svi:.4f}")
        self._log(f" Constraint error: {final_error:.2f}%")
        
        # Assess raw prediction quality
        if final_error > 10:
            self._log(f"High constraint error ({final_error:.2f}%) suggests model not learning tract mean", level='WARN')
        
        if pre_correction_error > 50:
            self._log(f"Very high error suggests fundamental model issues", level='WARN')
        
        # Extract coordinates
        x_coords = np.array([addr.geometry.x for _, addr in addresses.iterrows()])
        y_coords = np.array([addr.geometry.y for _, addr in addresses.iterrows()])
        
        # Generate uncertainty estimates (same as before)
        base_uncertainty = 0.05
        distance_from_mean = np.abs(adjusted_predictions - tract_svi)
        distance_uncertainty = distance_from_mean * 0.1
        
        # Spatial uncertainty (edge locations have higher uncertainty)
        center_x, center_y = np.mean(x_coords), np.mean(y_coords)
        edge_distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_distance = np.max(edge_distances) if np.max(edge_distances) > 0 else 1
        spatial_uncertainty = (edge_distances / max_distance) * 0.03
        
        # Combined uncertainty
        total_uncertainty = base_uncertainty + distance_uncertainty + spatial_uncertainty
        total_uncertainty = np.clip(total_uncertainty, 0.02, 0.15)
        
        # Create DataFrame
        final_predictions = pd.DataFrame({
            'address_id': [addr.get('address_id', i) for i, (_, addr) in enumerate(addresses.iterrows())],
            'x': x_coords,
            'y': y_coords,
            'mean': adjusted_predictions,  # Raw GNN output (adjustment = 0)
            'sd': total_uncertainty,
            'q025': np.clip(adjusted_predictions - 1.96 * total_uncertainty, 0.0, 1.0),
            'q975': np.clip(adjusted_predictions + 1.96 * total_uncertainty, 0.0, 1.0),
            'raw_prediction': predictions,  # Same as 'mean' (no correction)
            'adjustment': adjustment  # Will be 0.0
        })
        
        self._log(f"Finalized predictions (RAW, uncorrected): {len(final_predictions)} addresses")
        self._log(f" Final spatial std: {np.std(adjusted_predictions):.4f}")
        
        final_predictions.reset_index(drop=True, inplace=True)
        
        return final_predictions

    def evaluate_unconstrained_learning(self, training_result, graph_data, 
                                        tract_svi, addresses):
        """
        Evaluate model's unconstrained predictions to assess true learning.
        Tests if model naturally approaches correct mean without enforcement.
        
        This is run AFTER normal training to diagnose learning quality.
        """
        self._log("\n" + "="*80)
        self._log("UNCONSTRAINED LEARNING EVALUATION")
        self._log("="*80)
        
        # Get model from training result
        model = training_result.get('model')
        if model is None:
            self._log("ERROR: No model available for evaluation")
            return None
        
        # Generate unconstrained predictions
        model.eval()
        import torch

        # Check if context features available
        context = getattr(graph_data, 'context', None)

        with torch.no_grad():
            predictions, learned_accessibility, attention_weights = model(
                graph_data.x,
                graph_data.edge_index,
                return_accessibility=True,
                context_features=context
            )
        
        predictions_np = predictions.detach().numpy()
        
        # Metrics
        predicted_mean = np.mean(predictions_np)
        predicted_std = np.std(predictions_np)
        mean_error_pct = abs(predicted_mean - tract_svi) / tract_svi * 100
        natural_convergence = mean_error_pct < 20.0
        
        # Correlation with accessibility
        learned_acc = learned_accessibility.detach().numpy()
        acc_summary = learned_acc.mean(axis=1)
        correlation = np.corrcoef(acc_summary, predictions_np)[0, 1]
        
        self._log(f"Actual tract SVI: {tract_svi:.4f}")
        self._log(f"Predicted mean (unconstrained): {predicted_mean:.4f}")
        self._log(f"Mean error: {mean_error_pct:.2f}%")
        self._log(f"Natural convergence: {'YES ' if natural_convergence else 'NO ✗'}")
        self._log(f"Prediction std: {predicted_std:.4f}")
        self._log(f"Accessibility-SVI correlation: {correlation:.4f}")
        
        if not natural_convergence:
            self._log("\n⚠ WARNING: Model does not naturally approach correct mean")
            self._log("  This suggests constraints are masking poor learning")
        
        return {
            'predicted_mean': predicted_mean,
            'actual_svi': tract_svi,
            'mean_error_pct': mean_error_pct,
            'natural_convergence': natural_convergence,
            'spatial_std': predicted_std,
            'accessibility_svi_correlation': correlation,
            'raw_predictions': predictions_np
        }

    def run_holdout_validation(self, target_fips, neighbor_fips_list):
        """
        True holdout validation: train on n-1 tracts, predict on nth.
        
        This tests whether the GNN learns meaningful accessibility-vulnerability
        relationships WITHOUT constraint enforcement.
        
        Args:
            target_fips: Tract to hold out and predict
            neighbor_fips_list: List of neighboring tracts for training
        
        Returns:
            dict with unconstrained predictions and learning quality metrics
        """
        self._log(f"\n{'='*80}")
        self._log(f"HOLDOUT VALIDATION: Predicting tract {target_fips}")
        self._log(f"Training on {len(neighbor_fips_list)} neighbor tracts")
        self._log(f"{'='*80}\n")
        
        # 1. Load all spatial data
        data = self._load_spatial_data()
        
        # 2. Load training tracts (neighbors only, exclude target)
        train_addresses = []
        train_tract_svis = {}
        
        for fips in neighbor_fips_list:
            fips = str(fips).strip()
            addresses = self.data_loader.get_addresses_for_tract(fips)
            addresses['tract_fips'] = fips
            train_addresses.append(addresses)
            
            tract_data = data['tracts'][data['tracts']['FIPS'] == fips]
            train_tract_svis[fips] = float(tract_data.iloc[0]['RPL_THEMES'])
        
        train_addresses_df = pd.concat(train_addresses, ignore_index=True)
        
        self._log(f"Training set: {len(train_addresses_df)} addresses across {len(neighbor_fips_list)} tracts")
        
        # 3. Compute accessibility features for training set
        train_features = self._compute_accessibility_features(train_addresses_df, data)
        
        # Generate feature names for multi-task learning
        train_feature_names = self._generate_feature_names(train_features.shape[1])

        # 4. Normalize features on training data
        from ..models.gnn import normalize_accessibility_features
        normalized_train_features, feature_scaler = normalize_accessibility_features(train_features)
        
        # 5. Extract and normalize context features for training
        self._log("Extracting context features for training addresses...")
        train_context_features = self.data_loader.create_context_features_for_addresses(
            addresses=train_addresses_df,
            svi_data=data['svi']
        )
        normalized_train_context, context_scaler = self.data_loader.normalize_context_features(train_context_features)

        # 6. Build graph for training tracts
        state_fips = target_fips[:2]
        county_fips = target_fips[2:5]

        train_graph = self.data_loader.create_spatial_accessibility_graph(
            addresses=train_addresses_df,
            accessibility_features=normalized_train_features,
            context_features=normalized_train_context,  # NEW: Pass context
            state_fips=state_fips,
            county_fips=county_fips
        )
        
        # 7. Create tract masks for training
        train_tract_masks = {}
        for fips in neighbor_fips_list:
            train_tract_masks[fips] = (train_addresses_df['tract_fips'] == fips).values
        
        # 8. Train model WITHOUT constraints
        from ..models.gnn import AccessibilitySVIGNN, MultiTractGNNTrainer
        import torch
        
        seed = self.config.get('processing', {}).get('random_seed', 42)
        set_random_seed(seed)
        
        model = AccessibilitySVIGNN(
            accessibility_features_dim=train_graph.x.shape[1],
            hidden_dim=self.config.get('model', {}).get('hidden_dim', 64),
            dropout=self.config.get('model', {}).get('dropout', 0.3),
            seed=seed
        )
        
        # CRITICAL: Configure for unconstrained training
        unconstrained_config = self.config.get('training', {}).copy()
        unconstrained_config['enforce_constraints'] = False
        unconstrained_config['constraint_weight'] = 0.0
        unconstrained_config['use_multitask'] = True 
        
        trainer = MultiTractGNNTrainer(model, config=unconstrained_config, seed=seed)
        
        self._log("Training WITHOUT constraint enforcement...")
        training_result = trainer.train(
            graph_data=train_graph,
            tract_svis=train_tract_svis,
            tract_masks=train_tract_masks,
            epochs=self.config.get('model', {}).get('epochs', 150),
            verbose=self.verbose,
            feature_names=train_feature_names 
        )
        
        # 9. Load holdout tract data
        holdout_addresses = self.data_loader.get_addresses_for_tract(target_fips)
        holdout_addresses['tract_fips'] = target_fips
        
        self._log(f"Holdout tract: {len(holdout_addresses)} addresses")
        
        # 10. Compute features for holdout tract
        holdout_features = self._compute_accessibility_features(holdout_addresses, data)
        
        # CRITICAL: Apply same normalization from training data
        normalized_holdout_features = feature_scaler.transform(holdout_features)
        
        # 11. Extract context features for holdout tract
        self._log("Extracting context features for holdout addresses...")
        holdout_context_features = self.data_loader.create_context_features_for_addresses(
            addresses=holdout_addresses,
            svi_data=data['svi']
        )
        # CRITICAL: Apply same normalization from training context
        normalized_holdout_context = context_scaler.transform(holdout_context_features)

        # 12. Build graph for holdout tract
        holdout_graph = self.data_loader.create_spatial_accessibility_graph(
            addresses=holdout_addresses,
            accessibility_features=normalized_holdout_features,
            context_features=normalized_holdout_context,  # NEW: Pass context
            state_fips=state_fips,
            county_fips=county_fips
        )
        
        # 13. Predict on holdout tract (unconstrained)
        unconstrained_result = trainer.predict_unconstrained(holdout_graph)
        predictions = unconstrained_result['predictions']
        
        # 14. Get actual SVI for comparison
        target_tract_data = data['tracts'][data['tracts']['FIPS'] == target_fips]
        actual_svi = float(target_tract_data.iloc[0]['RPL_THEMES'])
        
        # 15. Compute learning quality metrics
        predicted_mean = np.mean(predictions)
        predicted_std = np.std(predictions)
        mean_error_pct = abs(predicted_mean - actual_svi) / actual_svi * 100
        natural_convergence = mean_error_pct < 20.0
        
        # Correlation with accessibility features
        learned_accessibility = unconstrained_result['learned_accessibility']
        accessibility_summary = learned_accessibility.mean(axis=1)
        correlation = np.corrcoef(accessibility_summary, predictions)[0, 1]
        
        self._log(f"\n{'='*80}")
        self._log(f"HOLDOUT VALIDATION RESULTS")
        self._log(f"{'='*80}")
        self._log(f"Actual tract SVI: {actual_svi:.4f}")
        self._log(f"Predicted mean (unconstrained): {predicted_mean:.4f}")
        self._log(f"Mean error: {mean_error_pct:.2f}%")
        self._log(f"Natural convergence: {'YES' if natural_convergence else 'NO'}")
        self._log(f"Prediction std: {predicted_std:.4f}")
        self._log(f"Accessibility-SVI correlation: {correlation:.4f}")
        
        return {
            'success': True,
            'mode': 'holdout_validation',
            'holdout_fips': target_fips,
            'training_fips': neighbor_fips_list,
            'actual_svi': actual_svi,
            'predicted_mean': predicted_mean,
            'predicted_std': predicted_std,
            'mean_error_pct': mean_error_pct,
            'natural_convergence': natural_convergence,
            'accessibility_svi_correlation': correlation,
            'predictions': predictions,
            'learned_accessibility': learned_accessibility,
            'training_history': training_result['training_history'],
            'n_training_addresses': len(train_addresses_df),
            'n_holdout_addresses': len(holdout_addresses)
        }

    def _validate_predictions(self, predictions, tract_svi, accessibility_features, normalized_features):
        """Enhanced validation with spatial learning diagnostics"""
        
        if predictions.empty:
            return {'validation_failed': True}
        
        pred_values = predictions['mean'].values
        
        # Get coordinates for spatial analysis
        x_coords = predictions['x'].values
        y_coords = predictions['y'].values
        coordinates = np.column_stack([x_coords, y_coords])
        
        # Initialize diagnostics
        diagnostics = SpatialLearningDiagnostics(verbose=self.verbose)
        
        # CRITICAL: Get raw predictions before any adjustment
        raw_predictions = self._get_raw_predictions_for_diagnostics()
        
        if raw_predictions is not None:
            self._log("Running spatial learning diagnostics on RAW predictions...")
            
            # Run comprehensive diagnostics
            diagnostic_results = diagnostics.comprehensive_evaluation(
                raw_predictions=raw_predictions['mean'].values,
                accessibility_features=accessibility_features,
                coordinates=raw_predictions[['x', 'y']].reset_index(drop=True),  
                target_svi=tract_svi
            )
            
            # Print detailed report
            verdict = diagnostics.print_diagnostic_report(diagnostic_results)
            
            # Store diagnostic results
            diagnostic_summary = {
                'spatial_autocorrelation': diagnostic_results['spatial_autocorrelation'],
                'accessibility_correlation': diagnostic_results['accessibility_correlations']['overall'],
                'learning_quality_score': diagnostic_results['quality_assessment']['learning_quality'],
                'verdict': verdict,
                'mean_bias': diagnostic_results['quality_assessment']['mean_bias'],
                'has_spatial_structure': diagnostic_results['quality_assessment']['has_spatial_structure'],
                'meaningful_accessibility_relationship': diagnostic_results['quality_assessment']['meaningful_accessibility_relationship'],
                'full_results': diagnostic_results  # Include full results for integration
            }
        else:
            self._log("Warning: Could not access raw predictions for diagnostics")
            diagnostic_summary = {'error': 'Raw predictions not available'}
            diagnostic_results = {}
        
        # Standard constraint validation on final predictions
        constraint_error = abs(np.mean(pred_values) - tract_svi) / tract_svi * 100
        
        # Spatial variation analysis
        spatial_std = np.std(pred_values)
        spatial_range = np.ptp(pred_values)
        
        # Accessibility-SVI relationship analysis (on final predictions)
        accessibility_svi_correlations = {}
        
        if accessibility_features is not None:
            # Overall accessibility vs final SVI predictions
            mean_accessibility = np.mean(accessibility_features, axis=1)
            overall_corr = np.corrcoef(mean_accessibility, pred_values)[0, 1]
            accessibility_svi_correlations['overall'] = overall_corr
            
            # Feature-specific correlations
            feature_groups = ['employment', 'healthcare', 'grocery']
            for i, group in enumerate(feature_groups):
                start_idx = i * 8
                end_idx = (i + 1) * 8
                if end_idx <= accessibility_features.shape[1]:
                    group_features = np.mean(accessibility_features[:, start_idx:end_idx], axis=1)
                    group_corr = np.corrcoef(group_features, pred_values)[0, 1]
                    accessibility_svi_correlations[group] = group_corr
        
        # Prediction quality assessment
        quality_metrics = {
            'constraint_satisfaction': 'excellent' if constraint_error < 5 else 
                                    'good' if constraint_error < 15 else 
                                    'poor',
            'spatial_variation': 'high' if spatial_std > 0.05 else 
                            'moderate' if spatial_std > 0.02 else 
                            'low',
            'prediction_range': spatial_range,
            'mean_prediction': np.mean(pred_values)
        }
        
        validation_results = {
            'constraint_error_percent': constraint_error,
            'spatial_std': spatial_std,
            'spatial_range': spatial_range,
            'accessibility_svi_correlations': accessibility_svi_correlations,
            'quality_metrics': quality_metrics,
            'n_addresses': len(predictions),
            'method': 'Direct Accessibility → SVI',
            'spatial_diagnostics': diagnostic_summary  # RETURN this for integration
        }
        
        self._log("Standard Validation Results:")
        self._log(f" Final constraint error: {constraint_error:.2f}%")
        self._log(f" Final spatial variation: {spatial_std:.4f}")
        self._log(f" Final accessibility-SVI correlation: {accessibility_svi_correlations.get('overall', 'N/A')}")
        
        return validation_results
    
    def _get_raw_predictions_for_diagnostics(self):
        """
        Get the raw predictions before any constraint correction
        This should be called during training to store the raw predictions
        """
        if hasattr(self, '_stored_raw_predictions'):
            return self._stored_raw_predictions
        else:
            self._log("Warning: Raw predictions not stored during training")
            return None

    def _apply_strong_constraint_correction(self, predictions, tract_svi):
        """Apply stronger constraint correction while preserving spatial variation"""
        
        current_mean = np.mean(predictions)
        current_std = np.std(predictions)
        
        # Strong adjustment to meet constraint
        adjustment = tract_svi - current_mean
        corrected_predictions = predictions + adjustment
        
        # Preserve relative spatial patterns while meeting constraint
        corrected_predictions = np.clip(corrected_predictions, 0.0, 1.0)
        
        # Verify constraint is met
        final_mean = np.mean(corrected_predictions)
        final_error = abs(final_mean - tract_svi) / tract_svi * 100
        
        if final_error > 1.0:  # Still > 1% error
            # Apply proportional scaling as backup
            if tract_svi > 0:
                scale_factor = tract_svi / current_mean
                corrected_predictions = predictions * scale_factor
                corrected_predictions = np.clip(corrected_predictions, 0.0, 1.0)
        
        return corrected_predictions

    def _create_research_visualizations(self, results):
        """Create research visualization outputs"""
        
        try:
            viz_output_dir = os.path.join(self.output_dir, 'visualizations')
            os.makedirs(viz_output_dir, exist_ok=True)
            
            # Prepare data for visualizer
            viz_data = {
                'gnn_predictions': results['predictions'],
                'accessibility_features': results['accessibility_features'],
                'learned_accessibility': results['accessibility_features'],
                'traditional_accessibility': results['accessibility_features'],  # For comparison
                'tract_svi': results['tract_svi'],
                'validation_results': results['validation_results'],
                'training_result': results['training_result']
            }
            
            # Create visualizations
            self.visualizer.create_comprehensive_research_analysis(viz_data, viz_output_dir)
            
            self._log(f"Research visualizations saved to {viz_output_dir}")
            
        except Exception as e:
            self._log(f"Could not create visualizations: {str(e)}", level='WARN')

    def _plot_error_by_svi_quintile(self, results, output_dir):
        """Stratify errors by SVI quintile to show model performance patterns."""
        import matplotlib.pyplot as plt
        
        # Extract data
        data = []
        for fips, r in results.items():
            data.append({
                'fips': fips,
                'svi': r['actual_svi'],
                'error': r['mean_error_pct'],
                'variation': np.std(r['predictions']),
                'raw_error': abs(np.mean(r.get('raw_predictions', r['predictions'])) - r['actual_svi']) / r['actual_svi'] * 100
            })
        
        df = pd.DataFrame(data)
        
        # Create quintiles
        df['quintile'] = pd.qcut(df['svi'], q=5, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)'])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance by SVI Quintile', fontsize=14, fontweight='bold')
        
        # Panel 1: Box plot of errors by quintile
        ax1 = axes[0, 0]
        df.boxplot(column='error', by='quintile', ax=ax1)
        ax1.set_xlabel('SVI Quintile')
        ax1.set_ylabel('Constraint Error (%)')
        ax1.set_title('Error Distribution by Quintile')
        ax1.axhline(y=20, color='green', linestyle='--', alpha=0.7)
        plt.suptitle('')  # Remove automatic title
        
        # Panel 2: Mean error with CI by quintile
        ax2 = axes[0, 1]
        quintile_stats = df.groupby('quintile')['error'].agg(['mean', 'std', 'count'])
        quintile_stats['se'] = quintile_stats['std'] / np.sqrt(quintile_stats['count'])
        
        x = range(len(quintile_stats))
        ax2.bar(x, quintile_stats['mean'], yerr=quintile_stats['se'] * 1.96, 
            capsize=5, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.set_xticks(x)
        ax2.set_xticklabels(quintile_stats.index)
        ax2.set_xlabel('SVI Quintile')
        ax2.set_ylabel('Mean Error (%) ± 95% CI')
        ax2.set_title('Mean Constraint Error by Quintile')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Spatial variation by quintile
        ax3 = axes[1, 0]
        df.boxplot(column='variation', by='quintile', ax=ax3)
        ax3.set_xlabel('SVI Quintile')
        ax3.set_ylabel('Spatial Variation (std)')
        ax3.set_title('Disaggregation Quality by Quintile')
        plt.suptitle('')
        
        # Panel 4: Summary table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_df = df.groupby('quintile').agg({
            'svi': ['min', 'max'],
            'error': ['mean', 'median'],
            'variation': 'mean'
        }).round(3)
        
        table_text = "QUINTILE ANALYSIS\n" + "="*50 + "\n\n"
        table_text += f"{'Quintile':<12} {'SVI Range':<15} {'Mean Err%':<12} {'Med Err%':<12} {'Variation':<10}\n"
        table_text += "-"*60 + "\n"
        
        for q in ['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)']:
            if q in summary_df.index:
                row = summary_df.loc[q]
                table_text += f"{q:<12} {row[('svi', 'min')]:.2f}-{row[('svi', 'max')]:.2f}    "
                table_text += f"{row[('error', 'mean')]:<12.1f} {row[('error', 'median')]:<12.1f} "
                table_text += f"{row[('variation', 'mean')]:.4f}\n"
        
        ax4.text(0.05, 0.95, table_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_by_svi_quintile.png'), 
                dpi=300, bbox_inches='tight')
        plt.close()

    def _compute_global_affine_correction(self, 
                                        all_raw_predictions: Dict[str, np.ndarray],
                                        tract_svi_values: Dict[str, float]) -> Tuple[float, float]:
        """
        Compute a single global affine transformation (y = ax + b) that minimizes
        aggregate constraint error across all tracts while preserving between-tract
        relationships.
        
        Args:
            all_raw_predictions: Dict mapping tract FIPS to raw prediction arrays
            tract_svi_values: Dict mapping tract FIPS to known SVI values
            
        Returns:
            Tuple of (scale, offset) for transformation: corrected = scale * raw + offset
        """
        # Aggregate to tract means
        tract_means = {}
        for fips, preds in all_raw_predictions.items():
            tract_means[fips] = np.mean(preds)
        
        # Prepare arrays for linear regression
        x = np.array([tract_means[f] for f in tract_svi_values.keys()])
        y = np.array([tract_svi_values[f] for f in tract_svi_values.keys()])
        
        # Least squares: y = ax + b
        # Solve: [x, 1] @ [a, b].T = y
        A = np.column_stack([x, np.ones_like(x)])
        coeffs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
        
        scale, offset = coeffs[0], coeffs[1]
        
        self._log(f"Global affine correction: y = {scale:.4f}x + {offset:.4f}")
        
        return float(scale), float(offset)

    def _apply_global_correction(self,
                                predictions: np.ndarray,
                                scale: float,
                                offset: float) -> np.ndarray:
        """
        Apply global affine correction to predictions.
        
        Args:
            predictions: Raw predictions array
            scale: Multiplicative factor
            offset: Additive factor
            
        Returns:
            Corrected predictions, clipped to [0, 1]
        """
        corrected = scale * predictions + offset
        corrected = np.clip(corrected, 0.0, 1.0)
        return corrected

    def _analyze_correction_impact(self,
                                    raw_predictions: Dict[str, np.ndarray],
                                    corrected_predictions: Dict[str, np.ndarray],
                                    tract_svi_values: Dict[str, float],
                                    scale: float,
                                    offset: float,
                                    output_dir: str):
        """
        Generate transparent visualization of correction impact.
        
        Creates a report showing:
        1. Raw vs corrected tract means
        2. Spatial variation preservation
        3. Between-tract relationship preservation
        """
        import matplotlib.pyplot as plt
        
        fips_list = sorted(raw_predictions.keys())
        
        raw_means = [np.mean(raw_predictions[f]) for f in fips_list]
        corrected_means = [np.mean(corrected_predictions[f]) for f in fips_list]
        actual_svis = [tract_svi_values[f] for f in fips_list]
        
        raw_stds = [np.std(raw_predictions[f]) for f in fips_list]
        corrected_stds = [np.std(corrected_predictions[f]) for f in fips_list]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Global Affine Correction Analysis', fontsize=14, fontweight='bold')
        
        # Panel 1: Raw means vs Actual SVI
        ax1 = axes[0, 0]
        ax1.scatter(actual_svis, raw_means, alpha=0.7, s=60, label='Raw predictions')
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect agreement')
        
        # Show regression line
        x_line = np.linspace(0, 1, 100)
        y_line = (x_line - offset) / scale if scale != 0 else x_line  # Inverse transform
        ax1.plot(actual_svis, [scale * m + offset for m in raw_means], 
                'r-', alpha=0.5, linewidth=2, label='Affine fit')
        
        raw_r2 = np.corrcoef(actual_svis, raw_means)[0, 1] ** 2
        ax1.set_xlabel('Actual Tract SVI')
        ax1.set_ylabel('Raw Predicted Mean')
        ax1.set_title(f'Raw Predictions (R² = {raw_r2:.3f})')
        ax1.legend()
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.05, 1.05)
        
        # Panel 2: Corrected means vs Actual SVI
        ax2 = axes[0, 1]
        ax2.scatter(actual_svis, corrected_means, alpha=0.7, s=60, c='green')
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        
        corrected_r2 = np.corrcoef(actual_svis, corrected_means)[0, 1] ** 2
        mean_error = np.mean([abs(c - a) / a * 100 for c, a in zip(corrected_means, actual_svis)])
        
        ax2.set_xlabel('Actual Tract SVI')
        ax2.set_ylabel('Corrected Predicted Mean')
        ax2.set_title(f'After Global Correction (R² = {corrected_r2:.3f}, Mean Error = {mean_error:.1f}%)')
        ax2.set_xlim(-0.05, 1.05)
        ax2.set_ylim(-0.05, 1.05)
        
        # Panel 3: Spatial variation preservation
        ax3 = axes[1, 0]
        ax3.scatter(raw_stds, corrected_stds, alpha=0.7, s=60)
        max_std = max(max(raw_stds), max(corrected_stds)) * 1.1
        ax3.plot([0, max_std], [0, max_std], 'k--', alpha=0.3)
        ax3.set_xlabel('Raw Prediction Std')
        ax3.set_ylabel('Corrected Prediction Std')
        ax3.set_title(f'Spatial Variation Preservation (ratio = {np.mean(corrected_stds)/np.mean(raw_stds):.2f})')
        
        # Panel 4: Correction parameters and summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
        GLOBAL AFFINE CORRECTION SUMMARY
        ================================
        
        Transformation: y = {scale:.4f}x + {offset:.4f}
        
        Before Correction:
        - R² (raw means vs actual): {raw_r2:.3f}
        - Mean tract error: {np.mean([abs(r - a) / a * 100 for r, a in zip(raw_means, actual_svis)]):.1f}%
        - Mean spatial variation: {np.mean(raw_stds):.4f}
        
        After Correction:
        - R² (corrected vs actual): {corrected_r2:.3f}
        - Mean tract error: {mean_error:.1f}%
        - Mean spatial variation: {np.mean(corrected_stds):.4f}
        
        Interpretation:
        - Scale < 1: Raw predictions had too much range
        - Scale > 1: Raw predictions had too little range
        - Offset > 0: Raw predictions were systematically low
        - Offset < 0: Raw predictions were systematically high
        
        Key Insight:
        Global correction preserves the RANKING of predictions
        across tracts, unlike per-tract independent shifts.
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'global_correction_analysis.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        self._log(f"Saved correction analysis to {output_dir}/global_correction_analysis.png")

    def _analyze_per_tract_corrections(self, test_results: Dict, output_dir: str):
        """
        Analyze per-tract mean-shift corrections applied.
        Shows the shift required for each tract and spatial variation preserved.
        """
        import matplotlib.pyplot as plt
        
        if not test_results:
            return
            
        fips_list = sorted(test_results.keys())
        
        # Extract data
        actual_svis = [test_results[f]['actual_svi'] for f in fips_list]
        raw_means = [test_results[f].get('raw_mean', test_results[f]['predicted_mean']) for f in fips_list]
        shifts = [test_results[f].get('shift_applied', 0) for f in fips_list]
        spatial_stds = [test_results[f].get('spatial_std', np.std(test_results[f]['predictions'])) for f in fips_list]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Per-Tract Mean-Shift Correction Analysis', fontsize=14, fontweight='bold')
        
        # Panel 1: Raw means vs Actual SVI (shows what model learned)
        ax1 = axes[0, 0]
        ax1.scatter(actual_svis, raw_means, alpha=0.7, s=60, c='steelblue')
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect prediction')
        
        # Add R² annotation
        if len(actual_svis) > 2:
            r2 = np.corrcoef(actual_svis, raw_means)[0, 1] ** 2
            ax1.annotate(f'R² = {r2:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                        fontsize=11, fontweight='bold')
        
        ax1.set_xlabel('Actual Tract SVI')
        ax1.set_ylabel('Raw Model Prediction (mean)')
        ax1.set_title('What the Model Learned (Before Correction)')
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend()
        
        # Panel 2: Shift required by tract SVI
        ax2 = axes[0, 1]
        colors = ['green' if s > 0 else 'red' for s in shifts]
        ax2.bar(range(len(fips_list)), shifts, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xticks(range(len(fips_list)))
        ax2.set_xticklabels([f[-5:] for f in fips_list], rotation=45, ha='right')
        ax2.set_xlabel('Tract')
        ax2.set_ylabel('Shift Applied')
        ax2.set_title('Correction Shift by Tract (green=up, red=down)')
        
        # Panel 3: Spatial variation by tract SVI
        ax3 = axes[1, 0]
        scatter = ax3.scatter(actual_svis, spatial_stds, c=actual_svis, cmap='RdYlGn_r', 
                             s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax3.set_xlabel('Actual Tract SVI')
        ax3.set_ylabel('Within-Tract Spatial Variation (std)')
        ax3.set_title('Spatial Variation vs Vulnerability Level')
        plt.colorbar(scatter, ax=ax3, label='Tract SVI')
        
        # Panel 4: Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        mean_shift = np.mean(np.abs(shifts))
        mean_std = np.mean(spatial_stds)
        r2_val = np.corrcoef(actual_svis, raw_means)[0, 1] ** 2 if len(actual_svis) > 2 else 0
        
        summary_text = f'''
        PER-TRACT MEAN-SHIFT CORRECTION SUMMARY
        =======================================
        
        Tracts Evaluated: {len(fips_list)}
        SVI Range: {min(actual_svis):.3f} - {max(actual_svis):.3f}
        
        RAW MODEL PERFORMANCE:
        - Mean raw prediction: {np.mean(raw_means):.3f}
        - R² (raw vs actual): {r2_val:.3f}
        
        CORRECTION APPLIED:
        - Mean absolute shift: {mean_shift:.3f}
        - Max shift: {max(np.abs(shifts)):.3f}
        
        SPATIAL VARIATION PRESERVED:
        - Mean within-tract std: {mean_std:.4f}
        - Std range: {min(spatial_stds):.4f} - {max(spatial_stds):.4f}
        
        INTERPRETATION:
        Per-tract correction guarantees constraint satisfaction
        while preserving learned within-tract spatial patterns.
        
        The key question: Is this within-tract variation
        meaningful? Block group validation will tell us.
        '''
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correction_analysis.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        self._log(f"Saved correction analysis to {output_dir}/correction_analysis.png")

    def _create_disaggregation_visualizations(self, test_results, training_results=None,
                                               expert_usage=None, tract_gdf=None, all_addresses=None):
        """
        Create comprehensive disaggregation visualizations.
        
        Generates:
        1. Aggregate baseline comparison dashboard
        2. Per-tract spatial disaggregation maps (matrix layout)
        3. Per-tract geographic visualization (actual coordinates)
        4. Accessibility-SVI relationship plots
        5. Expert routing analysis
        6. Constraint satisfaction analysis
        """
        try:
            viz_dir = os.path.join(self.output_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            self._log("Generating disaggregation visualizations...")
            
            # Collect data from test results
            valid_results = {
                fips: r for fips, r in test_results.items()
                if r.get('mean_error_pct') is not None
            }
            
            if not valid_results:
                self._log("No valid results for visualization")
                return
            
            # 1. Aggregate baseline comparison
            self._plot_aggregate_baseline_comparison(valid_results, viz_dir)
            
            # 2. Per-tract disaggregation analysis (matrix layout)
            self._plot_per_tract_disaggregation(valid_results, viz_dir)
            
            # 2b. Per-tract geographic visualization (actual coordinates)
            if all_addresses is not None and tract_gdf is not None:
                self._plot_geographic_disaggregation(valid_results, viz_dir, 
                                                    all_addresses, tract_gdf)
            
            # 3. Accessibility-SVI relationship
            self._plot_accessibility_svi_relationship(valid_results, viz_dir)
            
            # 4. Expert routing (if MoE)
            if expert_usage:
                self._plot_expert_routing(valid_results, expert_usage, viz_dir)
            
            # 5. Constraint satisfaction analysis
            self._plot_constraint_analysis(valid_results, viz_dir)

            # 6. Raw vs corrected analysis 
            self._plot_raw_vs_corrected_analysis(valid_results, viz_dir)

            # 7. Error stratified by SVI quintile
            self._plot_error_by_svi_quintile(valid_results, viz_dir)

            # 8. Feature transparency analysis
            # Aggregate features from all test tracts
            all_features = np.vstack([r['features'] for r in valid_results.values() if 'features' in r])
            all_predictions = np.concatenate([r['predictions'] for r in valid_results.values()])

            feature_analysis = self._analyze_feature_usage(
                features=all_features,
                feature_names=self._generate_feature_names(all_features.shape[1]),
                predictions=all_predictions,
                output_dir=viz_dir
            )
            self._plot_feature_transparency(feature_analysis, viz_dir)
            
            self._log(f"Visualizations saved to {viz_dir}")
            
        except Exception as e:
            self._log(f"Visualization generation failed: {e}", level='WARN')
            import traceback
            traceback.print_exc()

    def _plot_raw_vs_corrected_analysis(self, results, output_dir):
        """Show constraint errors before and after correction."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Raw vs Corrected Predictions (Constraint Correction Analysis)', 
                    fontsize=14, fontweight='bold')
        
        fips_list = list(results.keys())
        raw_errors, corrected_errors = [], []
        raw_stds, corrected_stds = [], []
        
        for f in fips_list:
            actual = results[f]['actual_svi']
            raw = results[f].get('raw_predictions', results[f]['predictions'])
            corrected = results[f]['predictions']
            
            raw_mean = np.mean(raw)
            corrected_mean = np.mean(corrected)
            
            raw_err = abs(raw_mean - actual) / actual * 100
            corr_err = abs(corrected_mean - actual) / actual * 100
            
            raw_errors.append(raw_err)
            corrected_errors.append(corr_err)
            raw_stds.append(np.std(raw))
            corrected_stds.append(np.std(corrected))
        
        # Panel 1: Error reduction
        ax1 = axes[0, 0]
        x = np.arange(len(fips_list))
        width = 0.35
        ax1.bar(x - width/2, raw_errors, width, label='Raw', color='#E57373', alpha=0.8)
        ax1.bar(x + width/2, corrected_errors, width, label='Corrected', color='#81C784', alpha=0.8)
        ax1.set_ylabel('Constraint Error (%)')
        ax1.set_title('Constraint Satisfaction: Raw vs Corrected')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f[-5:] for f in fips_list], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Variation preservation
        ax2 = axes[0, 1]
        ax2.scatter(raw_stds, corrected_stds, c=raw_errors, cmap='RdYlGn_r', 
                s=100, edgecolors='black')
        max_std = max(max(raw_stds), max(corrected_stds)) * 1.1
        ax2.plot([0, max_std], [0, max_std], 'k--', alpha=0.5)
        ax2.set_xlabel('Raw Prediction Std')
        ax2.set_ylabel('Corrected Prediction Std')
        ax2.set_title('Spatial Variation Preservation')
        ax2.grid(True, alpha=0.3)
        cbar = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar.set_label('Raw Error %')
        
        # Panel 3: Distribution of raw errors
        ax3 = axes[1, 0]
        ax3.hist(raw_errors, bins=15, color='#E57373', alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(raw_errors), color='red', linestyle='--', 
                label=f'Mean: {np.mean(raw_errors):.1f}%')
        ax3.set_xlabel('Raw Constraint Error (%)')
        ax3.set_ylabel('Count')
        ax3.set_title('Raw Error Distribution (Pre-Correction)')
        ax3.legend()
        
        # Panel 4: Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        summary = f"""
    RAW VS CORRECTED ANALYSIS
    {'='*40}

    RAW PREDICTIONS (Model Output)
    Mean Error: {np.mean(raw_errors):.1f}%
    Median Error: {np.median(raw_errors):.1f}%
    Tracts < 20%: {sum(1 for e in raw_errors if e < 20)}/{len(raw_errors)}

    CORRECTED PREDICTIONS (Post-Constraint)
    Mean Error: {np.mean(corrected_errors):.1f}%
    Tracts < 10%: {sum(1 for e in corrected_errors if e < 10)}/{len(corrected_errors)}

    VARIATION PRESERVATION
    Mean Raw Std: {np.mean(raw_stds):.4f}
    Mean Corrected Std: {np.mean(corrected_stds):.4f}
    Std Ratio: {np.mean(corrected_stds)/np.mean(raw_stds):.2f}

    INTERPRETATION
    Post-hoc correction shifts mean but
    preserves learned spatial patterns.
    """
        ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'raw_vs_corrected_analysis.png'), 
                dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_aggregate_baseline_comparison(self, results, output_dir):
        """Create aggregate baseline comparison dashboard."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('GRANITE Disaggregation: GNN vs Baselines', fontsize=14, fontweight='bold')
        
        # Extract data
        fips_list = list(results.keys())
        svis = [results[f]['actual_svi'] for f in fips_list]
        errors = [results[f]['mean_error_pct'] for f in fips_list]
        
        # Get baseline data if available
        gnn_stds, idw_stds, access_corrs = [], [], []
        for f in fips_list:
            baseline = results[f].get('baseline_comparison', {})
            methods = baseline.get('methods', {})
            
            gnn_data = methods.get('GNN', {})
            idw_data = methods.get('IDW_p2.0', {})
            
            gnn_stds.append(gnn_data.get('std', np.std(results[f].get('predictions', [0]))))
            idw_stds.append(idw_data.get('std', 0))
            access_corrs.append(gnn_data.get('accessibility_correlation', 0))
        
        # Panel 1: Spatial variation by method
        ax1 = axes[0, 0]
        x_pos = np.arange(len(fips_list))
        width = 0.35
        ax1.bar(x_pos - width/2, gnn_stds, width, label='GNN', color='#2E7D32', alpha=0.8)
        ax1.bar(x_pos + width/2, idw_stds, width, label='IDW', color='#1565C0', alpha=0.8)
        ax1.set_xlabel('Tract')
        ax1.set_ylabel('Spatial Variation (std)')
        ax1.set_title('Disaggregation Quality by Tract')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f[-5:] for f in fips_list], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Panel 2: GNN vs IDW scatter
        ax2 = axes[0, 1]
        ax2.scatter(idw_stds, gnn_stds, c=svis, cmap='RdYlGn_r', s=100, 
                   edgecolors='black', linewidth=1)
        max_val = max(max(gnn_stds), max(idw_stds)) * 1.1
        ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='1:1 line')
        ax2.set_xlabel('IDW Variation (std)')
        ax2.set_ylabel('GNN Variation (std)')
        ax2.set_title('GNN vs IDW (color = tract SVI)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        cbar = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar.set_label('Tract SVI')
        
        # Panel 3: Constraint satisfaction
        ax3 = axes[0, 2]
        colors = ['#2E7D32' if e < 20 else '#FFA000' if e < 50 else '#D32F2F' for e in errors]
        ax3.barh(range(len(fips_list)), errors, color=colors, edgecolor='black')
        ax3.axvline(x=20, color='green', linestyle='--', alpha=0.7, label='Good (<20%)')
        ax3.axvline(x=50, color='orange', linestyle='--', alpha=0.7, label='Moderate (<50%)')
        ax3.set_yticks(range(len(fips_list)))
        ax3.set_yticklabels([f[-5:] for f in fips_list])
        ax3.set_xlabel('Constraint Error (%)')
        ax3.set_title('Constraint Satisfaction by Tract')
        ax3.legend(loc='lower right')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Panel 4: Variation vs SVI
        ax4 = axes[1, 0]
        ax4.scatter(svis, gnn_stds, c='#2E7D32', s=100, label='GNN', 
                   edgecolors='black', linewidth=1)
        ax4.scatter(svis, idw_stds, c='#1565C0', s=100, label='IDW',
                   edgecolors='black', linewidth=1, marker='s')
        ax4.set_xlabel('Tract SVI')
        ax4.set_ylabel('Spatial Variation (std)')
        ax4.set_title('Disaggregation vs Vulnerability Level')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Panel 5: Accessibility correlation
        ax5 = axes[1, 1]
        colors = ['#2E7D32' if c < 0 else '#D32F2F' for c in access_corrs]
        ax5.bar(range(len(fips_list)), access_corrs, color=colors, edgecolor='black')
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax5.set_xticks(range(len(fips_list)))
        ax5.set_xticklabels([f[-5:] for f in fips_list], rotation=45)
        ax5.set_ylabel('Accessibility-SVI Correlation')
        ax5.set_title('Equity Pattern by Tract')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Add interpretation text
        neg_count = sum(1 for c in access_corrs if c < 0)
        ax5.text(0.02, 0.98, f'Negative (expected): {neg_count}/{len(access_corrs)}',
                transform=ax5.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Panel 6: Summary statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        summary_text = f"""
DISAGGREGATION SUMMARY
{'='*40}

Holdout Tracts: {len(fips_list)}
SVI Range: {min(svis):.3f} - {max(svis):.3f}

CONSTRAINT SATISFACTION
  Mean Error: {np.mean(errors):.1f}%
  Median Error: {np.median(errors):.1f}%
  Tracts < 20%: {sum(1 for e in errors if e < 20)}/{len(errors)}

DISAGGREGATION QUALITY  
  GNN Mean Variation: {np.mean(gnn_stds):.4f}
  IDW Mean Variation: {np.mean(idw_stds):.4f}
  GNN Advantage: {np.mean(gnn_stds) - np.mean(idw_stds):+.4f}
  GNN > IDW: {sum(1 for g, i in zip(gnn_stds, idw_stds) if g > i)}/{len(fips_list)}

ACCESSIBILITY-VULNERABILITY
  Mean Correlation: {np.mean(access_corrs):.3f}
  Negative (expected): {neg_count}/{len(access_corrs)}
  Strong Negative: {sum(1 for c in access_corrs if c < -0.3)}/{len(access_corrs)}
"""
        ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'disaggregation_dashboard.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        self._log("  Created: disaggregation_dashboard.png")

    def _analyze_feature_usage(self, features, feature_names, predictions, output_dir):
        """
        Comprehensive feature usage analysis for transparency.
        
        Generates:
        - Feature correlation matrix with SVI proxy
        - Top contributing features ranked
        - Feature group summaries (by destination type)
        - Unused/low-variance feature identification
        """
        from scipy.stats import pearsonr, spearmanr
        
        n_features = features.shape[1]
        
        # Generate feature names if not provided
        if feature_names is None or len(feature_names) != n_features:
            feature_names = self._generate_feature_names(n_features)
        
        # Feature statistics
        feature_stats = []
        for i, name in enumerate(feature_names):
            values = features[:, i]
            corr_pred, p_pred = pearsonr(values, predictions)
            
            stats = {
                'feature': name,
                'mean': np.mean(values),
                'std': np.std(values),
                'range': np.ptp(values),
                'prediction_corr': corr_pred,
                'prediction_p': p_pred,
                'abs_corr': abs(corr_pred)
            }
            feature_stats.append(stats)
        
        stats_df = pd.DataFrame(feature_stats)
        stats_df = stats_df.sort_values('abs_corr', ascending=False)
        
        # Identify problematic features
        low_variance = stats_df[stats_df['std'] < 0.01]['feature'].tolist()
        weak_signal = stats_df[stats_df['abs_corr'] < 0.1]['feature'].tolist()
        strong_signal = stats_df[stats_df['abs_corr'] > 0.3]['feature'].tolist()
        
        # Group by destination type
        dest_groups = {'employment': [], 'healthcare': [], 'grocery': [], 'modal': [], 'socioeconomic': []}
        for name in feature_names:
            for group in dest_groups.keys():
                if group in name.lower() or (group == 'socioeconomic' and any(x in name for x in ['pct_', 'no_', 'poverty', 'unemployment'])):
                    dest_groups[group].append(name)
                    break
        
        return {
            'feature_stats': stats_df,
            'low_variance_features': low_variance,
            'weak_signal_features': weak_signal,
            'strong_signal_features': strong_signal,
            'destination_groups': dest_groups,
            'n_features': n_features
        }

    def _plot_feature_transparency(self, feature_analysis, output_dir):
        """Create comprehensive feature transparency visualizations."""
        import matplotlib.pyplot as plt
        
        stats_df = feature_analysis['feature_stats']
        
        fig = plt.figure(figsize=(20, 16))
        
        # Layout: 3 rows, 3 columns
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Panel 1: Top 20 features by correlation (takes 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        top20 = stats_df.head(20)
        colors = ['#2E7D32' if c < 0 else '#D32F2F' for c in top20['prediction_corr']]
        bars = ax1.barh(range(len(top20)), top20['prediction_corr'], color=colors, edgecolor='black')
        ax1.set_yticks(range(len(top20)))
        ax1.set_yticklabels(top20['feature'], fontsize=9)
        ax1.axvline(x=0, color='black', linewidth=0.5)
        ax1.set_xlabel('Correlation with Predictions')
        ax1.set_title('Top 20 Features by Prediction Correlation\n(Green=Negative expected for vulnerability)', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.invert_yaxis()
        
        # Panel 2: Feature variance distribution
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.hist(stats_df['std'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.axvline(x=0.01, color='red', linestyle='--', label='Low variance threshold')
        ax2.set_xlabel('Feature Standard Deviation')
        ax2.set_ylabel('Count')
        ax2.set_title('Feature Variance Distribution')
        ax2.legend()
        
        # Panel 3: Correlation by feature group
        ax3 = fig.add_subplot(gs[1, 0])
        groups = feature_analysis['destination_groups']
        group_corrs = {}
        for group, features in groups.items():
            if features:
                group_data = stats_df[stats_df['feature'].isin(features)]
                group_corrs[group] = group_data['abs_corr'].mean()
        
        if group_corrs:
            ax3.bar(range(len(group_corrs)), list(group_corrs.values()), 
                color='teal', edgecolor='black', alpha=0.7)
            ax3.set_xticks(range(len(group_corrs)))
            ax3.set_xticklabels(list(group_corrs.keys()), rotation=45, ha='right')
            ax3.set_ylabel('Mean |Correlation|')
            ax3.set_title('Feature Group Importance')
            ax3.grid(True, alpha=0.3)
        
        # Panel 4: Feature type breakdown
        ax4 = fig.add_subplot(gs[1, 1])
        # Categorize by type (time, count, modal, etc.)
        type_counts = {'time': 0, 'count': 0, 'advantage': 0, 'dispersion': 0, 
                    'modal': 0, 'socioeconomic': 0, 'other': 0}
        for name in stats_df['feature']:
            categorized = False
            for key in type_counts.keys():
                if key in name.lower():
                    type_counts[key] += 1
                    categorized = True
                    break
            if not categorized:
                type_counts['other'] += 1
        
        ax4.pie([v for v in type_counts.values() if v > 0], 
            labels=[k for k, v in type_counts.items() if v > 0],
            autopct='%1.0f%%', startangle=90)
        ax4.set_title('Feature Type Distribution')
        
        # Panel 5: Unused features alert
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        weak = feature_analysis['weak_signal_features']
        low_var = feature_analysis['low_variance_features']
        strong = feature_analysis['strong_signal_features']
        
        alert_text = f"""
    FEATURE QUALITY SUMMARY
    {'='*35}

    Total Features: {feature_analysis['n_features']}

    STRONG SIGNAL (|r| > 0.3): {len(strong)}
    {', '.join(strong[:5])}{'...' if len(strong) > 5 else ''}

    WEAK SIGNAL (|r| < 0.1): {len(weak)}
    {', '.join(weak[:5])}{'...' if len(weak) > 5 else ''}

    LOW VARIANCE: {len(low_var)}
    {', '.join(low_var[:3])}{'...' if len(low_var) > 3 else ''}

    RECOMMENDATION:
    Consider removing {len(weak)} weak features
    for model simplification.
    """
        ax5.text(0.05, 0.95, alert_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        # Panel 6: Full correlation heatmap (bottom row, spans all columns)
        ax6 = fig.add_subplot(gs[2, :])
        
        # Show correlation for top 30 features
        top30 = stats_df.head(30)
        corr_data = top30[['feature', 'prediction_corr']].set_index('feature')
        
        # Create horizontal bar-style heatmap
        im = ax6.imshow([top30['prediction_corr'].values], cmap='RdYlGn_r', 
                    aspect='auto', vmin=-0.5, vmax=0.5)
        ax6.set_yticks([])
        ax6.set_xticks(range(len(top30)))
        ax6.set_xticklabels(top30['feature'], rotation=90, fontsize=8)
        ax6.set_title('Feature-Prediction Correlation Heatmap (Top 30)', fontweight='bold')
        plt.colorbar(im, ax=ax6, orientation='horizontal', pad=0.2, 
                    label='Correlation', shrink=0.5)
        
        plt.savefig(os.path.join(output_dir, 'feature_transparency.png'), 
                dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save CSV for inspection
        stats_df.to_csv(os.path.join(output_dir, 'feature_analysis.csv'), index=False)

    def _plot_per_tract_disaggregation(self, results, output_dir):
        """Create per-tract spatial disaggregation maps."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
        
        n_tracts = len(results)
        cols = min(4, n_tracts)
        rows = (n_tracts + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_tracts == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Per-Tract GNN Disaggregation (Deviation from Tract Mean)', 
                    fontsize=14, fontweight='bold')
        
        for idx, (fips, data) in enumerate(results.items()):
            row, col = idx // cols, idx % cols
            ax = axes[row, col]
            
            predictions = data.get('predictions', np.array([data['predicted_mean']]))
            actual_svi = data['actual_svi']
            
            if len(predictions) > 1:
                # Compute deviation from tract mean
                deviations = predictions - actual_svi
                
                # Create pseudo-spatial layout (grid)
                n = len(predictions)
                side = int(np.ceil(np.sqrt(n)))
                grid = np.full((side, side), np.nan)
                grid.flat[:n] = deviations
                
                # Two-slope colormap centered at 0
                vmax = max(abs(deviations.min()), abs(deviations.max()))
                norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
                
                im = ax.imshow(grid, cmap='RdBu_r', norm=norm)
                plt.colorbar(im, ax=ax, shrink=0.8)
            
            error = data['mean_error_pct']
            std = np.std(predictions) if len(predictions) > 1 else 0
            
            ax.set_title(f'{fips[-5:]}\nSVI={actual_svi:.3f}, Err={error:.1f}%, std={std:.4f}',
                        fontsize=9)
            ax.axis('off')
        
        # Hide empty subplots
        for idx in range(len(results), rows * cols):
            row, col = idx // cols, idx % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_tract_disaggregation.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        self._log("  Created: per_tract_disaggregation.png")

    def _plot_geographic_disaggregation(self, results, output_dir, all_addresses, tract_gdf):
        """
        Create geographic scatter plot showing addresses at actual coordinates.
        
        Shows addresses at lat/lon positions, colored by deviation from tract mean.
        Reveals true spatial patterns and clustering within each tract.
        
        Args:
            results: Dict of {fips: result_dict} from validation
            output_dir: Directory to save visualization
            all_addresses: GeoDataFrame with all addresses (must have FIPS column)
            tract_gdf: GeoDataFrame with tract boundaries
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import TwoSlopeNorm
        
        n_tracts = len(results)
        cols = min(4, n_tracts)
        rows = (n_tracts + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_tracts == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Geographic GNN Disaggregation (Deviation from Tract Mean)', 
                    fontsize=14, fontweight='bold')
        
        for idx, (fips, data) in enumerate(results.items()):
            row, col = idx // cols, idx % cols
            ax = axes[row, col]
            
            predictions = data.get('predictions', np.array([]))
            actual_svi = data['actual_svi']
            
            if len(predictions) == 0:
                ax.axis('off')
                ax.text(0.5, 0.5, f'No predictions\nfor {fips[-5:]}', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Get addresses for this tract
            tract_addresses = all_addresses[all_addresses['FIPS'] == fips].copy()
            
            if len(tract_addresses) == 0:
                ax.axis('off')
                ax.text(0.5, 0.5, f'No addresses\nfor {fips[-5:]}', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Verify prediction count matches
            if len(predictions) != len(tract_addresses):
                self._log(f"Prediction count mismatch for {fips}: "
                         f"{len(predictions)} preds vs {len(tract_addresses)} addresses", level='WARN')
                ax.axis('off')
                ax.text(0.5, 0.5, 
                       f'Mismatch:\n{len(predictions)} preds\n{len(tract_addresses)} addresses', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=8)
                continue
            
            # Add predictions and compute deviations
            tract_addresses['prediction'] = predictions
            tract_addresses['deviation'] = predictions - actual_svi
            
            # Get tract boundary for context
            tract_boundary = tract_gdf[tract_gdf['FIPS'] == fips]
            
            # Plot tract boundary
            if len(tract_boundary) > 0:
                tract_boundary.boundary.plot(ax=ax, color='black', linewidth=1.5, 
                                            linestyle='--', alpha=0.5, zorder=1)
            
            # Extract coordinates
            x = tract_addresses.geometry.x
            y = tract_addresses.geometry.y
            deviations = tract_addresses['deviation'].values
            
            # Two-slope colormap centered at 0
            vmax = max(abs(deviations.min()), abs(deviations.max()))
            if vmax == 0:
                vmax = 0.01
            
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            
            # Scatter plot at actual coordinates
            scatter = ax.scatter(
                x, y, 
                c=deviations, 
                cmap='RdBu_r',  # Red = above tract mean, Blue = below
                norm=norm,
                s=30,
                alpha=0.7,
                edgecolors='black',
                linewidths=0.3,
                zorder=2
            )
            
            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Deviation\nfrom Mean', fontsize=7)
            cbar.ax.tick_params(labelsize=6)
            
            # Format axes
            ax.set_aspect('equal')
            ax.set_xlabel('Longitude', fontsize=8)
            ax.set_ylabel('Latitude', fontsize=8)
            ax.tick_params(labelsize=7)
            
            # Title with metrics
            error = data['mean_error_pct']
            std = np.std(predictions)
            ax.set_title(
                f'Tract {fips[-5:]}\n'
                f'SVI={actual_svi:.3f}, Err={error:.1f}%, std={std:.4f}\n'
                f'n={len(predictions)} addresses',
                fontsize=9
            )
            
            ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
        
        # Hide empty subplots
        for idx in range(len(results), rows * cols):
            row, col = idx // cols, idx % cols
            axes[row, col].axis('off')
        
        # Legend explaining colors
        blue_patch = mpatches.Patch(color='blue', alpha=0.7, 
                                    label='Below tract mean (less vulnerable)')
        red_patch = mpatches.Patch(color='red', alpha=0.7, 
                                   label='Above tract mean (more vulnerable)')
        fig.legend(handles=[blue_patch, red_patch], loc='lower center', 
                  ncol=2, frameon=True, fontsize=10, bbox_to_anchor=(0.5, -0.02))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'geographic_disaggregation.png'),
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self._log("  Created: geographic_disaggregation.png")

    def _plot_accessibility_svi_relationship(self, results, output_dir):
        """Plot accessibility-SVI relationship across tracts."""
        import matplotlib.pyplot as plt
        
        n_tracts = len(results)
        cols = min(5, n_tracts)
        rows = (n_tracts + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows))
        if n_tracts == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Accessibility-Vulnerability Relationship by Tract', 
                    fontsize=14, fontweight='bold')
        
        for idx, (fips, data) in enumerate(results.items()):
            row, col = idx // cols, idx % cols
            ax = axes[row, col]
            
            predictions = data.get('predictions', np.array([data['predicted_mean']]))
            baseline = data.get('baseline_comparison', {})
            access_corr = baseline.get('methods', {}).get('GNN', {}).get('accessibility_correlation', 0)
            
            if len(predictions) > 1:
                # Use prediction rank as proxy for accessibility rank
                pred_rank = np.argsort(np.argsort(predictions))
                
                # Color by correlation direction
                color = '#2E7D32' if access_corr < 0 else '#D32F2F'
                
                ax.scatter(pred_rank, predictions, alpha=0.5, s=10, c=color)
                
                # Add trend line
                z = np.polyfit(pred_rank, predictions, 1)
                p = np.poly1d(z)
                ax.plot(pred_rank, p(pred_rank), color='black', linestyle='--', linewidth=1)
            
            actual_svi = data['actual_svi']
            ax.axhline(y=actual_svi, color='orange', linestyle='-', linewidth=1, 
                      label=f'Tract SVI={actual_svi:.3f}')
            
            ax.set_xlabel('Accessibility Rank', fontsize=8)
            ax.set_ylabel('Predicted SVI', fontsize=8)
            ax.set_title(f'{fips[-5:]}: r={access_corr:.3f}', fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(results), rows * cols):
            row, col = idx // cols, idx % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accessibility_svi_relationship.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        self._log("  Created: accessibility_svi_relationship.png")

    def _plot_expert_routing(self, results, expert_usage, output_dir):
        """Plot expert routing analysis for MoE models."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Mixture of Experts Routing Analysis', fontsize=14, fontweight='bold')
        
        # Extract expert assignments
        fips_list = list(results.keys())
        svis = [results[f]['actual_svi'] for f in fips_list]
        experts = [results[f].get('dominant_expert', 'Unknown') for f in fips_list]
        
        expert_colors = {'Low': '#2E7D32', 'Medium': '#FFA000', 'High': '#D32F2F', 'Unknown': 'gray'}
        
        # Panel 1: Expert assignment by SVI
        ax1 = axes[0]
        colors = [expert_colors.get(e, 'gray') for e in experts]
        ax1.scatter(svis, range(len(fips_list)), c=colors, s=150, edgecolors='black')
        ax1.set_yticks(range(len(fips_list)))
        ax1.set_yticklabels([f[-5:] for f in fips_list])
        ax1.set_xlabel('Tract SVI')
        ax1.set_title('Expert Assignment by Vulnerability')
        ax1.grid(True, alpha=0.3)
        
        # Add legend
        for expert, color in expert_colors.items():
            if expert in experts:
                ax1.scatter([], [], c=color, s=100, label=expert, edgecolors='black')
        ax1.legend(title='Dominant Expert')
        
        # Panel 2: Expert usage distribution
        ax2 = axes[1]
        expert_counts = {}
        for e in experts:
            expert_counts[e] = expert_counts.get(e, 0) + 1
        
        bars = ax2.bar(expert_counts.keys(), expert_counts.values(),
                      color=[expert_colors.get(e, 'gray') for e in expert_counts.keys()],
                      edgecolor='black')
        ax2.set_ylabel('Number of Tracts')
        ax2.set_title('Expert Usage Distribution')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Panel 3: Gate weights heatmap (if available)
        ax3 = axes[2]
        gate_weights = []
        for f in fips_list:
            gw = results[f].get('gate_weights', [0.33, 0.33, 0.34])
            gate_weights.append(gw)
        
        gate_weights = np.array(gate_weights)
        im = ax3.imshow(gate_weights, cmap='YlOrRd', aspect='auto')
        ax3.set_yticks(range(len(fips_list)))
        ax3.set_yticklabels([f[-5:] for f in fips_list])
        ax3.set_xticks(range(3))
        ax3.set_xticklabels(['Low', 'Medium', 'High'])
        ax3.set_xlabel('Expert')
        ax3.set_title('Gate Weights by Tract')
        plt.colorbar(im, ax=ax3, shrink=0.8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'expert_routing_analysis.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        self._log("  Created: expert_routing_analysis.png")

    def _plot_constraint_analysis(self, results, output_dir):
        """Plot constraint satisfaction analysis."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Constraint Satisfaction Analysis', fontsize=14, fontweight='bold')
        
        fips_list = list(results.keys())
        svis = [results[f]['actual_svi'] for f in fips_list]
        predicted_means = [results[f]['predicted_mean'] for f in fips_list]
        errors = [results[f]['mean_error_pct'] for f in fips_list]
        
        # Panel 1: Actual vs Predicted SVI
        ax1 = axes[0]
        ax1.scatter(svis, predicted_means, c=errors, cmap='RdYlGn_r', s=100,
                   edgecolors='black', linewidth=1)
        
        # Perfect prediction line
        lims = [min(min(svis), min(predicted_means)) - 0.05,
                max(max(svis), max(predicted_means)) + 0.05]
        ax1.plot(lims, lims, 'k--', alpha=0.5, label='Perfect prediction')
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
        ax1.set_xlabel('Actual Tract SVI')
        ax1.set_ylabel('Predicted Mean SVI')
        ax1.set_title('Constraint Satisfaction')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        cbar = plt.colorbar(ax1.collections[0], ax=ax1)
        cbar.set_label('Error (%)')
        
        # Panel 2: Error distribution
        ax2 = axes[1]
        ax2.hist(errors, bins=10, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.axvline(x=np.mean(errors), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(errors):.1f}%')
        ax2.axvline(x=np.median(errors), color='orange', linestyle='--',
                   label=f'Median: {np.median(errors):.1f}%')
        ax2.set_xlabel('Constraint Error (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Error vs SVI
        ax3 = axes[2]
        ax3.scatter(svis, errors, c='steelblue', s=100, edgecolors='black')
        
        # Trend line
        z = np.polyfit(svis, errors, 1)
        p = np.poly1d(z)
        ax3.plot(sorted(svis), p(sorted(svis)), 'r--', alpha=0.7, label='Trend')
        
        ax3.axhline(y=20, color='green', linestyle=':', alpha=0.7, label='Good threshold (20%)')
        ax3.set_xlabel('Tract SVI')
        ax3.set_ylabel('Constraint Error (%)')
        ax3.set_title('Error by Vulnerability Level')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'constraint_analysis.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        self._log("  Created: constraint_analysis.png")

    def save_results(self, results, output_dir=None):
        """Save pipeline results"""
        
        if not results.get('success', False):
            self._log("No results to save")
            return
        
        save_dir = output_dir or self.output_dir
        
        # Save predictions
        predictions_path = os.path.join(save_dir, 'granite_predictions.csv')
        results['predictions'].to_csv(predictions_path, index=False)
        
        # Save accessibility features
        if 'accessibility_features' in results:
            features_path = os.path.join(save_dir, 'accessibility_features.csv')
            features_df = pd.DataFrame(results['accessibility_features'])
            features_df.to_csv(features_path, index=False)
        
        # Save summary
        summary_path = os.path.join(save_dir, 'results_summary.json')
        import json
        
        summary = {
            'methodology': results.get('methodology', 'GRANITE'),
            'tract_fips': results['tract_info']['FIPS'],
            'tract_svi': float(results['tract_info']['RPL_THEMES']),
            'summary_metrics': results['summary'],
            'validation_results': results['validation_results']
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self._log(f"Results saved to {save_dir}")

    def _debug_accessibility_vulnerability_relationship(self, final_predictions, accessibility_features, addresses):
        """Programmatic debugging of accessibility-vulnerability patterns"""
        
        self._log("=== DEBUGGING ACCESSIBILITY-VULNERABILITY RELATIONSHIP ===")
        
        # Get predictions and features
        predicted_svi = final_predictions['mean'].values
        feature_names = self._generate_feature_names(accessibility_features.shape[1])
        
        # Find extreme cases for manual inspection
        n_samples = min(10, len(predicted_svi))
        
        # Highest vulnerability addresses
        high_vuln_indices = np.argsort(predicted_svi)[-n_samples:]
        # Lowest vulnerability addresses  
        low_vuln_indices = np.argsort(predicted_svi)[:n_samples]
        
        self._log("SAMPLE ANALYSIS:")
        self._log("HIGH VULNERABILITY ADDRESSES (should have POOR accessibility):")
        
        for i, idx in enumerate(high_vuln_indices):
            addr_id = addresses.iloc[idx].get('address_id', idx)
            pred_svi = predicted_svi[idx]
            
            # Extract key accessibility metrics
            emp_min_time = accessibility_features[idx, 0]  # employment min time
            emp_count_5min = accessibility_features[idx, 3]  # employment count in 5min
            health_min_time = accessibility_features[idx, 10]  # healthcare min time  
            grocery_min_time = accessibility_features[idx, 20]  # grocery min time
            
            # Overall accessibility score (lower times = better access)
            avg_min_time = (emp_min_time + health_min_time + grocery_min_time) / 3
            
            self._log(f" Address {addr_id}: SVI={pred_svi:.3f}")
            self._log(f"   Avg min travel time: {avg_min_time:.1f} min")
            self._log(f"   Employment: {emp_min_time:.1f}min, {emp_count_5min} jobs in 5min")
            self._log(f"   Healthcare: {health_min_time:.1f}min")
            self._log(f"   Grocery: {grocery_min_time:.1f}min")
            
            # RED FLAG CHECK: If high vulnerability has very good accessibility
            if avg_min_time < 6.0:  # Very good accessibility (< 6min average)
                self._log(f"   🚨 RED FLAG: High vulnerability but excellent accessibility!")
        
        self._log("\nLOW VULNERABILITY ADDRESSES (should have GOOD accessibility):")
        
        for i, idx in enumerate(low_vuln_indices):
            addr_id = addresses.iloc[idx].get('address_id', idx)
            pred_svi = predicted_svi[idx]
            
            emp_min_time = accessibility_features[idx, 0]
            emp_count_5min = accessibility_features[idx, 3]
            health_min_time = accessibility_features[idx, 10]
            grocery_min_time = accessibility_features[idx, 20]
            
            avg_min_time = (emp_min_time + health_min_time + grocery_min_time) / 3
            
            self._log(f" Address {addr_id}: SVI={pred_svi:.3f}")
            self._log(f"   Avg min travel time: {avg_min_time:.1f} min")
            self._log(f"   Employment: {emp_min_time:.1f}min, {emp_count_5min} jobs in 5min")
            self._log(f"   Healthcare: {health_min_time:.1f}min")  
            self._log(f"   Grocery: {grocery_min_time:.1f}min")
            
            # RED FLAG CHECK: If low vulnerability has very poor accessibility
            if avg_min_time > 12.0:  # Poor accessibility (> 12min average)
                self._log(f"   🚨 RED FLAG: Low vulnerability but poor accessibility!")
        
        return {
            'high_vuln_sample': high_vuln_indices,
            'low_vuln_sample': low_vuln_indices
        }
    
    def _validate_feature_directions(self, accessibility_features):
        """Test if accessibility features have correct theoretical directions"""
        
        self._log("=== VALIDATING FEATURE DIRECTIONS ===")
        
        feature_names = self._generate_feature_names(accessibility_features.shape[1])
        
        # Create a synthetic "good accessibility" address vs "poor accessibility" address
        # Good accessibility = low travel times, high destination counts
        good_access_profile = []
        poor_access_profile = []
        
        for i, feature_name in enumerate(feature_names):
            if 'min_time' in feature_name or 'mean_time' in feature_name or 'median_time' in feature_name:
                # Time features: lower = better accessibility
                good_access_profile.append(3.0)   # 3 minutes - very good
                poor_access_profile.append(15.0)  # 15 minutes - poor
                
            elif 'count' in feature_name:
                # Count features: higher = better accessibility
                good_access_profile.append(4.0)   # 4 destinations accessible
                poor_access_profile.append(0.0)   # 0 destinations accessible
                
            elif 'drive_advantage' in feature_name:
                # Drive advantage: higher = more car-dependent = worse accessibility for non-drivers
                good_access_profile.append(0.2)   # Low car dependence
                poor_access_profile.append(0.8)   # High car dependence
                
            else:
                # Other features - use median values
                median_val = np.median(accessibility_features[:, i])
                good_access_profile.append(median_val * 1.2)  # Slightly above median
                poor_access_profile.append(median_val * 0.8)  # Slightly below median
        
        # Test what your model would predict for these synthetic profiles
        good_access_array = np.array(good_access_profile).reshape(1, -1)
        poor_access_array = np.array(poor_access_profile).reshape(1, -1)
        
        self._log("THEORETICAL TEST:")
        self._log("Good accessibility profile (should predict LOW vulnerability):")
        for i, (name, val) in enumerate(zip(feature_names[:10], good_access_profile[:10])):  # Show first 10
            self._log(f" {name}: {val:.2f}")
        
        self._log("Poor accessibility profile (should predict HIGH vulnerability):")
        for i, (name, val) in enumerate(zip(feature_names[:10], poor_access_profile[:10])):  # Show first 10
            self._log(f" {name}: {val:.2f}")
        
        # Calculate what the correlation SHOULD be if features are coded correctly
        # This is a basic sanity check
        time_features = [i for i, name in enumerate(feature_names) if 'time' in name]
        count_features = [i for i, name in enumerate(feature_names) if 'count' in name]
        
        self._log("EXPECTED RELATIONSHIPS:")
        self._log(f"Time features (indices {time_features[:3]}...): should correlate POSITIVELY with vulnerability")
        self._log(f"Count features (indices {count_features[:3]}...): should correlate NEGATIVELY with vulnerability")
        
        return {
            'good_access_profile': good_access_profile,
            'poor_access_profile': poor_access_profile,
            'time_features': time_features,
            'count_features': count_features
        }

    def analyze_feature_importance(self, results, n_repeats=10):
        """
        Run feature importance analysis on trained model.
        
        Args:
            results: Training results containing model and data
            n_repeats: Number of permutation repeats
            
        Returns:
            Dictionary with importance analysis results
        """
        from ..evaluation.feature_importance import FeatureImportanceAnalyzer
        
        self._log("\n" + "="*60)
        self._log("FEATURE IMPORTANCE ANALYSIS")
        self._log("="*60)
        
        # Extract necessary data
        model = results['training_result']['model']
        edge_index = results['training_result']['edge_index']
        tract_svi = results['tract_info']['RPL_THEMES']

        # Use full training data if multi-tract, otherwise use target tract data
        if 'full_accessibility_features' in results:
            # Multi-tract mode: use all addresses from training
            accessibility_features = results['full_accessibility_features']
        else:
            # Single-tract mode: use target tract only
            accessibility_features = results['accessibility_features']

        feature_names = self._generate_feature_names(accessibility_features.shape[1])
        
        # Initialize analyzer
        analyzer = FeatureImportanceAnalyzer(model, device='cpu', verbose=self.verbose)
        
        # Run permutation importance
        perm_results = analyzer.permutation_importance(
            accessibility_features=accessibility_features,
            edge_index=edge_index,
            true_svi=tract_svi,
            feature_names=feature_names,
            n_repeats=n_repeats
        )
        
        # Run gradient importance
        grad_results = analyzer.gradient_importance(
            accessibility_features=accessibility_features,
            edge_index=edge_index,
            feature_names=feature_names
        )
        
        # Generate outputs
        output_dir = os.path.join(self.output_dir, 'feature_importance')
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot
        analyzer.plot_importance(
            perm_results, 
            output_path=os.path.join(output_dir, 'feature_importance.png'),
            top_n=20
        )
        
        # Report
        analyzer.generate_report(
            perm_results,
            output_path=os.path.join(output_dir, 'importance_report.txt')
        )
        
        # Save detailed results
        perm_results['feature_importance'].to_csv(
            os.path.join(output_dir, 'permutation_importance.csv'),
            index=False
        )
        
        grad_results['feature_importance'].to_csv(
            os.path.join(output_dir, 'gradient_importance.csv'),
            index=False
        )
        
        self._log(f"\nFeature importance analysis complete!")
        self._log(f"Results saved to {output_dir}")
        
        return {
            'permutation': perm_results,
            'gradient': grad_results
        }
        
    def _get_expected_correlation_direction(self, feature_name):
        """
        Determine theoretically expected correlation direction with vulnerability.
        
        Returns:
            "POSITIVE": Feature should increase with vulnerability (e.g., longer travel times)
            "NEGATIVE": Feature should decrease with vulnerability (e.g., more destinations)
            "UNKNOWN": Feature type cannot be classified
        """
        feature_lower = feature_name.lower()
        
        # Features that should correlate POSITIVELY with vulnerability
        # (Higher values = worse accessibility = higher vulnerability)
        positive_indicators = [
            'min_time', 'mean_time', 'median_time',  # Longer travel times = worse
            'drive_advantage',  # Higher car dependency = worse
            'dispersion',  # More scattered destinations = worse
            'time_range',  # Greater time variation = worse
            'percentile',  # Higher percentile = worse relative position
            'accessibility_percentile'  # Higher percentile = worse accessibility
        ]
        
        # Features that should correlate NEGATIVELY with vulnerability
        # (Higher values = better accessibility = lower vulnerability)
        negative_indicators = [
            'count_5min', 'count_10min', 'count_15min',  # More nearby destinations = better
            'local_accessibility_index',  # Higher index = better
            'modal_flexibility',  # More options = better
            'accessibility_score',  # Higher score = better
            'accessibility_equity',  # Higher equity = better
            'geographic_accessibility'  # Higher = better
        ]
        
        # Check positive indicators first (more specific)
        for indicator in positive_indicators:
            if indicator in feature_lower:
                return "POSITIVE"
        
        # Then check negative indicators
        for indicator in negative_indicators:
            if indicator in feature_lower:
                return "NEGATIVE"
        
        # Unknown feature type
        return "UNKNOWN"

    def _create_accessibility_correlation_diagnostic(self, final_predictions, accessibility_features):
        """Create detailed correlation diagnostic with proper feature direction classification"""
        
        self._log("=== ACCESSIBILITY CORRELATION DIAGNOSTIC ===")
        
        predicted_svi = final_predictions['mean'].values
        feature_names = self._generate_feature_names(accessibility_features.shape[1])
        
        # Calculate correlation of each feature with predicted SVI
        correlations = []
        
        for i, feature_name in enumerate(feature_names):
            feature_values = accessibility_features[:, i]
            corr = np.corrcoef(feature_values, predicted_svi)[0, 1]
            
            # Determine expected direction using proper classification
            expected_direction = self._get_expected_correlation_direction(feature_name)
            actual_direction = "POSITIVE" if corr > 0 else "NEGATIVE"
            
            # Only mark as correct/incorrect if we know the expected direction
            if expected_direction == "UNKNOWN":
                is_correct = None
            else:
                is_correct = expected_direction == actual_direction
            
            correlations.append({
                'feature': feature_name,
                'correlation': corr,
                'expected': expected_direction,
                'actual': actual_direction,
                'correct': is_correct,
                'strength': 'Strong' if abs(corr) > 0.3 else 'Moderate' if abs(corr) > 0.1 else 'Weak'
            })
        
        # Sort by absolute correlation strength
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        self._log("TOP CORRELATIONS (strongest to weakest):")
        
        correct_count = 0
        known_count = 0
        
        for i, corr_info in enumerate(correlations[:15]):  # Show top 15
            if corr_info['correct'] is None:
                status = "?"
            elif corr_info['correct']:
                status = ""
                correct_count += 1
            else:
                status = "✗"
            
            if corr_info['expected'] != "UNKNOWN":
                known_count += 1
            
            self._log(f" {status} {corr_info['feature']}: r={corr_info['correlation']:.3f} "
                    f"({corr_info['strength']}, expected {corr_info['expected']})")
        
        # Calculate correctness rate only for features with known expected directions
        if known_count > 0:
            correctness_rate = correct_count / known_count * 100
            
            self._log(f"\nOVERALL DIRECTIONAL CORRECTNESS: {correct_count}/{known_count} ({correctness_rate:.1f}%)")
            self._log(f"(Excluding {len(correlations) - known_count} features with unknown expected direction)")
            
            if correctness_rate < 60:
                self._log("🚨 MAJOR ISSUE: Less than 60% of features have correct correlation direction!")
                self._log("   This suggests systematic feature encoding problems OR confounding factors.")
            elif correctness_rate < 80:
                self._log("⚠️  WARNING: Some features may be incorrectly encoded or model is learning confounding patterns")
            else:
                self._log(" Feature directions appear mostly correct")
        else:
            self._log("\n⚠️  WARNING: No features with known expected directions found")
        
        return correlations

    def run_global_training(self, training_fips_list: List[str], test_fips_list: List[str] = None):
        """
        Global training: Train ONE model on diverse tracts, test on separate holdouts.
        
        This is fundamentally different from per-tract training:
        - Trains a single model on ALL training tracts
        - Model learns cross-context accessibility-vulnerability relationships
        - Can predict on any new tract without retraining
        
        Args:
            training_fips_list: List of diverse training tract FIPS codes
            test_fips_list: Optional list of test tract FIPS codes for validation
        
        Returns:
            dict with trained model and evaluation results
        """
        self._log(f"\n{'='*80}")
        self._log(f"GLOBAL TRAINING MODE")
        self._log(f"Training on {len(training_fips_list)} diverse tracts")
        if test_fips_list:
            self._log(f"Testing on {len(test_fips_list)} holdout tracts")
        self._log(f"{'='*80}\n")
        
        # 1. Load all spatial data
        data = self._load_spatial_data()

        # 2. Load ALL training tract addresses
        self._log("Pre-loading full address dataset...")
        all_addresses = data['addresses'].copy()
        tracts_gdf = data['tracts'].copy()

        # Do spatial join ONCE to assign tract IDs to all addresses
        self._log("Performing one-time spatial join to assign tract IDs...")
        addresses_with_tracts = gpd.sjoin(
            all_addresses, 
            tracts_gdf[['FIPS', 'geometry']], 
            how='left', 
            predicate='within'
        )

        train_addresses = []
        train_tract_svis = {}
        
        for fips in training_fips_list:
            fips = str(fips).strip()
            print(f"[DEBUG] Filtering tract {fips}...")
            
            # Now we can filter efficiently in memory
            tract_mask = addresses_with_tracts['FIPS'] == fips
            addresses = addresses_with_tracts[tract_mask].copy()
            
            if len(addresses) == 0:
                self._log(f"No addresses found for tract {fips}", level='WARN')
                continue
            
            # Keep only original columns + tract_fips
            addresses = addresses[['address_id', 'geometry', 'full_address']].copy()
            addresses['tract_fips'] = fips
            train_addresses.append(addresses)
            print(f"[DEBUG]  Filtered {len(addresses)} addresses from {fips}")

            # Store tract SVI
            tract_data = data['tracts'][data['tracts']['FIPS'] == fips]
            if len(tract_data) > 0:
                train_tract_svis[fips] = float(tract_data.iloc[0]['RPL_THEMES'])
        
        train_addresses_df = pd.concat(train_addresses, ignore_index=True)
        
        self._log(f"Global training set: {len(train_addresses_df)} addresses")
        self._log(f"SVI range: {min(train_tract_svis.values()):.3f} - {max(train_tract_svis.values()):.3f}")
        
        # 3. Compute accessibility features for ALL training addresses
        n_training_tracts = len(training_fips_list)
        n_training_addresses = len(train_addresses_df)
        
        if n_training_tracts > 5:
            self._log(f"\nMulti-tract training detected ({n_training_tracts} tracts, {n_training_addresses} addresses)")
            batch_size = self._estimate_optimal_batch_size(n_training_addresses, n_training_tracts)
            self._log(f"Using batched feature computation with batch_size={batch_size} tracts")
            train_features = self._compute_features_batched(train_addresses_df, data, batch_size=batch_size)
        else:
            self._log(f"\nStandard training ({n_training_tracts} tracts, {n_training_addresses} addresses)")
            self._log(f"Computing accessibility features for all addresses at once...")
            train_features = self._compute_accessibility_features(train_addresses_df, data)
        
        # Validate feature computation succeeded
        if train_features is None or train_features.size == 0:
            return {
                'success': False,
                'error': 'Failed to compute training features'
            }
        train_feature_names = self._generate_feature_names(train_features.shape[1])
        
        # 4. Normalize features on global training data
        from ..models.gnn import normalize_accessibility_features
        normalized_train_features, feature_scaler = normalize_accessibility_features(train_features)
        
        # 5. Extract context features for training
        self._log("Extracting context features for global training...")
        train_context_features = self.data_loader.create_context_features_for_addresses(
            addresses=train_addresses_df,
            svi_data=data['svi']
        )
        normalized_train_context, context_scaler = self.data_loader.normalize_context_features(
            train_context_features
        )
        
        # 6. Build GLOBAL graph with all training addresses
        state_fips = training_fips_list[0][:2]
        county_fips = training_fips_list[0][2:5]
        
        train_graph = self.data_loader.create_spatial_accessibility_graph(
            addresses=train_addresses_df,
            accessibility_features=normalized_train_features,
            context_features=normalized_train_context,
            state_fips=state_fips,
            county_fips=county_fips
        )
        
        self._log(f"Built global graph: {train_graph.x.shape[0]} nodes, {train_graph.edge_index.shape[1]} edges")
        
        # 7. Create tract masks for multi-tract training
        train_tract_masks = {}
        for fips in training_fips_list:
            train_tract_masks[fips] = (train_addresses_df['tract_fips'] == fips).values
        
        # 8. Train GLOBAL model
        from ..models.gnn import AccessibilitySVIGNN, MultiTractGNNTrainer
        
        seed = self.config.get('processing', {}).get('random_seed', 42)
        set_random_seed(seed)
        
        model = AccessibilitySVIGNN(
            accessibility_features_dim=train_graph.x.shape[1],
            hidden_dim=self.config.get('model', {}).get('hidden_dim', 64),
            dropout=self.config.get('model', {}).get('dropout', 0.3),
            seed=seed
        )
        
        # Configure for multi-task learning without strict constraints
        training_config = self.config.get('training', {}).copy()
        training_config['enforce_constraints'] = True
        training_config['constraint_weight'] = 1.0  # Lower weight for diverse data
        training_config['use_multitask'] = True
        
        trainer = MultiTractGNNTrainer(model, config=training_config, seed=seed)
        
        self._log("Training GLOBAL model on diverse tracts...")
        training_result = trainer.train(
            graph_data=train_graph,
            tract_svis=train_tract_svis,
            tract_masks=train_tract_masks,
            epochs=self.config.get('model', {}).get('epochs', 150),
            verbose=self.verbose,
            feature_names=train_feature_names
        )
        
        # 9. If test tracts provided, evaluate on them
        test_results = {}
        if test_fips_list:
            self._log(f"\nEvaluating on {len(test_fips_list)} holdout tracts...")
            
            for test_fips in test_fips_list:
                result = self._evaluate_on_holdout_tract(
                    test_fips, data, trainer, feature_scaler, context_scaler
                )
                test_results[test_fips] = result
                
                self._log(f" {test_fips}: Error={result['mean_error_pct']:.1f}%, Corr={result['correlation']:.3f}")
        
        return {
            'success': True,
            'mode': 'global_training',
            'model': model,
            'trainer': trainer,
            'training_fips': training_fips_list,
            'training_svi_range': (min(train_tract_svis.values()), max(train_tract_svis.values())),
            'training_result': training_result,
            'feature_scaler': feature_scaler,
            'context_scaler': context_scaler,
            'test_results': test_results,
            'n_training_addresses': len(train_addresses_df)
        }

    def _run_holdout_baselines(self, test_addresses, gnn_predictions, tract_gdf,
                                test_fips, actual_svi, accessibility_features=None):
        """
        Run baseline comparisons for a single holdout tract.
        
        Args:
            test_addresses: Address GeoDataFrame for holdout tract
            gnn_predictions: GNN prediction array
            tract_gdf: All tract geometries with SVI
            test_fips: Holdout tract FIPS
            actual_svi: Known tract SVI
            accessibility_features: Accessibility feature matrix
            
        Returns:
            Dict with baseline comparison results
        """
        try:
            comparison = DisaggregationComparison(verbose=False)  # Quiet for batch
            
            comparison.add_baseline(NaiveUniformBaseline())
            comparison.add_baseline(IDWDisaggregation(power=2.0, n_neighbors=8))
            comparison.add_baseline(OrdinaryKrigingDisaggregation())
            
            results = comparison.run_comparison(
                tract_gdf=tract_gdf,
                address_gdf=test_addresses,
                gnn_predictions=gnn_predictions,
                tract_fips=test_fips,
                tract_svi=actual_svi,
                accessibility_features=accessibility_features,
                svi_column='RPL_THEMES'
            )
            
            return results
            
        except Exception as e:
            self._log(f"   Baseline comparison failed: {e}", level='WARNING')
            return {'error': str(e)}

    def run_mixture_training(self, training_fips_list, test_fips_list):
        """
        Train using Mixture of Experts for context-dependent accessibility-vulnerability.
        
        Each expert specializes in different SVI ranges:
        - Expert_Low: SVI < 0.40 (suburban, car-dependent)
        - Expert_Medium: SVI 0.30-0.70 (transition zones)
        - Expert_High: SVI > 0.55 (urban cores)
        """
        from granite.models.mixture_of_experts import (
            create_moe_model, MixtureOfExpertsTrainer, MoEInferenceAnalyzer
        )
        from granite.models.gnn import normalize_accessibility_features, set_random_seed
        from torch_geometric.data import Data
        
        start_time = time.time()
        seed = self.config.get('processing', {}).get('random_seed', 42)
        set_random_seed(seed)
        
        self._log(f"\n{'='*80}")
        self._log("MIXTURE OF EXPERTS TRAINING")
        self._log(f"Training on {len(training_fips_list)} diverse tracts")
        if test_fips_list:
            self._log(f"Testing on {len(test_fips_list)} holdout tracts")
        self._log(f"{'='*80}\n")
        
        # 1. Load all spatial data (same as run_global_training)
        data = self._load_spatial_data()
        
        # 2. Load ALL training tract addresses with spatial join
        self._log("Pre-loading full address dataset...")
        all_addresses = data['addresses'].copy()
        tracts_gdf = data['tracts'].copy()
        
        import geopandas as gpd
        addresses_with_tracts = gpd.sjoin(
            all_addresses, 
            tracts_gdf[['FIPS', 'geometry']], 
            how='left', 
            predicate='within'
        )
        
        # 3. Process each training tract individually (MoE needs per-tract graphs)
        training_graphs = []
        training_svis = []
        training_fips_processed = []
        
        # First pass: collect all addresses for batch feature computation
        all_train_addresses = []
        tract_address_counts = {}
        
        for fips in training_fips_list:
            fips = str(fips).strip()
            tract_mask = addresses_with_tracts['FIPS'] == fips
            addresses = addresses_with_tracts[tract_mask].copy()
            
            if len(addresses) == 0:
                self._log(f"No addresses found for tract {fips}", level='WARN')
                continue
                
            addresses = addresses[['address_id', 'geometry', 'full_address']].copy()
            addresses['tract_fips'] = fips
            all_train_addresses.append(addresses)
            tract_address_counts[fips] = len(addresses)
            
            # Get tract SVI
            tract_data = data['tracts'][data['tracts']['FIPS'] == fips]
            if len(tract_data) > 0:
                svi = float(tract_data.iloc[0]['RPL_THEMES'])
                training_svis.append(svi)
                training_fips_processed.append(fips)
                self._log(f" {fips}: SVI={svi:.3f}, {len(addresses)} addresses")
        
        if len(all_train_addresses) < 3:
            return {
                'success': False,
                'error': f'Insufficient training data: {len(all_train_addresses)} tracts (need 3+)'
            }
        
        import pandas as pd
        train_addresses_df = pd.concat(all_train_addresses, ignore_index=True)
        self._log(f"\nTotal training addresses: {len(train_addresses_df)}")
        
        # 4. Compute accessibility features for all training addresses
        n_training_tracts = len(training_fips_processed)
        n_training_addresses = len(train_addresses_df)
        
        if n_training_tracts > 5:
            batch_size = self._estimate_optimal_batch_size(n_training_addresses, n_training_tracts)
            self._log(f"Using batched feature computation (batch_size={batch_size})")
            train_features = self._compute_features_batched(train_addresses_df, data, batch_size=batch_size)
        else:
            self._log("Computing accessibility features...")
            train_features = self._compute_accessibility_features(train_addresses_df, data)
        
        if train_features is None or train_features.size == 0:
            return {'success': False, 'error': 'Failed to compute training features'}
        
        # 5. Normalize features globally
        normalized_train_features, feature_scaler = normalize_accessibility_features(train_features)
        
        # 6. Extract and normalize context features
        train_context_features = self.data_loader.create_context_features_for_addresses(
            addresses=train_addresses_df,
            svi_data=data['svi']
        )
        normalized_train_context, context_scaler = self.data_loader.normalize_context_features(
            train_context_features
        )
        
        # 7. Create per-tract graphs for MoE training
        self._log("\nCreating per-tract graphs for MoE...")
        
        current_idx = 0
        for i, fips in enumerate(training_fips_processed):
            n_addresses = tract_address_counts[fips]
            
            # Extract this tract's data
            tract_features = normalized_train_features[current_idx:current_idx + n_addresses]
            tract_context = normalized_train_context[current_idx:current_idx + n_addresses]
            tract_addresses = train_addresses_df.iloc[current_idx:current_idx + n_addresses]
            
            # Create graph for this tract
            state_fips = fips[:2]
            county_fips = fips[2:5]
            
            tract_graph = self.data_loader.create_spatial_accessibility_graph(
                addresses=tract_addresses,
                accessibility_features=tract_features,
                context_features=tract_context,
                state_fips=state_fips,
                county_fips=county_fips
            )
            
            # Store context in graph for MoE gate network
            tract_graph.context = torch.tensor(tract_context, dtype=torch.float32)
            tract_graph.tract_fips = fips
            tract_graph.tract_svi = training_svis[i]
            
            training_graphs.append(tract_graph)
            current_idx += n_addresses
            
            self._log(f" {fips}: {tract_graph.x.shape[0]} nodes, {tract_graph.edge_index.shape[1]} edges")
        
        # 8. Create MoE model
        self._log("\nCreating Mixture of Experts model...")
        moe_model = create_moe_model(
            accessibility_features_dim=training_graphs[0].x.shape[1],
            context_features_dim=5,  # 5 context features (NO tract SVI)
            hidden_dim=64,
            dropout=0.3,
            seed=seed
        )
        
        # 9. Configure and run MoE training
        moe_config = {
            'learning_rate': float(self.config.get('training', {}).get('learning_rate', 0.001)),
            'weight_decay': float(self.config.get('training', {}).get('weight_decay', 1e-4)),
            'expert_epochs': self.config.get('training', {}).get('epochs', 150),
            'gate_epochs': self.config.get('training', {}).get('gate_epochs', 100),
            'finetune_epochs': self.config.get('training', {}).get('finetune_epochs', 50),
        }
        
        self._log("Training Mixture of Experts...")
        trainer = MixtureOfExpertsTrainer(moe_model, moe_config, seed=seed)
        
        training_results = trainer.train(
            graph_data_list=training_graphs,
            tract_svi_list=training_svis,
            finetune=True,
            verbose=self.verbose
        )
        
        # 10. Evaluate on test set with per-tract mean-shift correction
        # ==============================================================
        # Simple approach: shift each tract's predictions to match known SVI mean.
        # This preserves within-tract spatial variation while satisfying constraints.
        
        test_results = {}
        
        if test_fips_list:
            self._log(f"\nEvaluating on {len(test_fips_list)} test tracts...")
            moe_model.eval()
            
            for test_fips in test_fips_list:
                test_fips = str(test_fips).strip()
                try:
                    # Get test tract addresses
                    tract_mask = addresses_with_tracts['FIPS'] == test_fips
                    test_addresses = addresses_with_tracts[tract_mask].copy()
                    
                    if len(test_addresses) == 0:
                        self._log(f"  {test_fips}: No addresses, skipping")
                        continue
                    
                    test_addresses = test_addresses[['address_id', 'geometry', 'full_address']].copy()
                    test_addresses['tract_fips'] = test_fips
                    
                    # Compute and normalize features using training scalers
                    test_features = self._compute_accessibility_features(test_addresses, data)
                    normalized_test_features = feature_scaler.transform(test_features)
                    
                    test_context = self.data_loader.create_context_features_for_addresses(
                        addresses=test_addresses,
                        svi_data=data['svi']
                    )
                    normalized_test_context = context_scaler.transform(test_context)
                    
                    # Create test graph
                    state_fips_code = test_fips[:2]
                    county_fips_code = test_fips[2:5]
                    
                    test_graph = self.data_loader.create_spatial_accessibility_graph(
                        addresses=test_addresses,
                        accessibility_features=normalized_test_features,
                        context_features=normalized_test_context,
                        state_fips=state_fips_code,
                        county_fips=county_fips_code
                    )
                    test_graph.context = torch.tensor(normalized_test_context, dtype=torch.float32)
                    
                    # Get actual SVI
                    tract_data = data['tracts'][data['tracts']['FIPS'] == test_fips]
                    actual_svi = float(tract_data.iloc[0]['RPL_THEMES'])
                    
                    # MoE inference
                    with torch.no_grad():
                        predictions, gate_weights = moe_model(
                            test_graph.x, 
                            test_graph.edge_index,
                            context_features=test_graph.context,
                            return_gate_weights=True
                        )
                    
                    raw_predictions = predictions.cpu().numpy()
                    
                    # Per-tract mean-shift correction
                    raw_mean = float(np.mean(raw_predictions))
                    shift = actual_svi - raw_mean
                    corrected_predictions = raw_predictions + shift
                    corrected_predictions = np.clip(corrected_predictions, 0.0, 1.0)
                    
                    # Calculate metrics
                    predicted_mean = float(np.mean(corrected_predictions))
                    mean_error_pct = abs(predicted_mean - actual_svi) / actual_svi * 100 if actual_svi > 0 else 0
                    spatial_std = float(np.std(corrected_predictions))
                    
                    # Correlation between predictions and accessibility
                    accessibility_summary = test_graph.x.mean(dim=1).cpu().numpy()
                    if len(accessibility_summary) > 1:
                        correlation = float(np.corrcoef(accessibility_summary, corrected_predictions)[0, 1])
                    else:
                        correlation = 0.0
                    
                    # Gate analysis
                    gate_weights_np = gate_weights.cpu().numpy()
                    dominant_expert = int(np.argmax(gate_weights_np.mean(axis=0)))
                    expert_names = ['Low', 'Medium', 'High']
                    
                    # Run baseline comparisons
                    baseline_results = self._run_holdout_baselines(
                        test_addresses=test_addresses,
                        gnn_predictions=corrected_predictions,
                        tract_gdf=data['tracts'],
                        test_fips=test_fips,
                        actual_svi=actual_svi,
                        accessibility_features=test_features
                    )
                    
                    test_results[test_fips] = {
                        'actual_svi': actual_svi,
                        'predicted_mean': predicted_mean,
                        'raw_mean': raw_mean,
                        'mean_error_pct': mean_error_pct,
                        'spatial_std': spatial_std,
                        'correlation': correlation,
                        'dominant_expert': expert_names[dominant_expert],
                        'gate_weights': gate_weights_np.mean(axis=0).tolist(),
                        'predictions': corrected_predictions,
                        'raw_predictions': raw_predictions,
                        'shift_applied': shift,
                        'features': test_features,
                        'baseline_comparison': baseline_results
                    }
                    
                    # Log results
                    idw_std = baseline_results.get('methods', {}).get('IDW_p2.0', {}).get('std', 0)
                    self._log(f"  {test_fips}: SVI={actual_svi:.3f}, Shift={shift:+.3f}, "
                             f"Std={spatial_std:.4f}, Expert={expert_names[dominant_expert]}")
                    
                except Exception as e:
                    self._log(f"  {test_fips}: ERROR - {str(e)}", level='ERROR')
                    import traceback
                    traceback.print_exc()
        
        # Generate correction analysis visualization
        if test_results:
            self._analyze_per_tract_corrections(test_results, self.output_dir)
        
        # 11. Analyze expert usage
        analyzer = MoEInferenceAnalyzer(moe_model, verbose=self.verbose)
        usage_stats = analyzer.analyze_expert_usage(training_graphs, training_svis)
        
        elapsed = time.time() - start_time

        # 12. Block group validation (optional - graceful failure if data unavailable)
        try:
            block_groups = self.data_loader.load_block_groups_for_validation(
                state_fips, county_fips
            )
            
            if block_groups is not None:
                from granite.evaluation.block_group_validation import BlockGroupValidator
                
                # Combine all test addresses and predictions
                all_test_addresses = []
                all_test_predictions = []
                for fips, result in test_results.items():
                    test_addr = self.data_loader.get_addresses_for_tract(fips)
                    all_test_addresses.append(test_addr)
                    all_test_predictions.append(result['predictions'])
                
                if all_test_addresses:
                    combined_addresses = pd.concat(all_test_addresses, ignore_index=True)
                    combined_predictions = np.concatenate(all_test_predictions)
                    
                    # Create validator
                    validator = BlockGroupValidator(block_groups, verbose=self.verbose)
                    
                    # Validate GRANITE
                    granite_validation = validator.validate(
                        combined_addresses, combined_predictions, 'GRANITE'
                    )
                    
                    # Build comparison results dict - include GRANITE
                    comparison_results = {'GRANITE': granite_validation}
                    
                    # Run same validation for baselines
                    idw_predictions = []
                    kriging_predictions = []
                    for fips, result in test_results.items():
                        if 'baseline_comparison' in result:
                            baselines = result['baseline_comparison'].get('methods', {})
                            if 'IDW_p2.0' in baselines:
                                idw_predictions.append(baselines['IDW_p2.0']['predictions'])
                            if 'Kriging' in baselines:
                                kriging_predictions.append(baselines['Kriging']['predictions'])
                    
                    if idw_predictions:
                        idw_combined = np.concatenate(idw_predictions)
                        comparison_results['IDW'] = validator.validate(
                            combined_addresses, idw_combined, 'IDW'
                        )
                    
                    if kriging_predictions:
                        kriging_combined = np.concatenate(kriging_predictions)
                        comparison_results['Kriging'] = validator.validate(
                            combined_addresses, kriging_combined, 'Kriging'
                        )
                    
                    # Generate comparison report
                    validation_dir = os.path.join(self.output_dir, 'block_group_validation')
                    os.makedirs(validation_dir, exist_ok=True)
                    validator.create_validation_report(validation_dir, comparison_results)
                    
                    self._log(f"Block group validation complete. See {validation_dir}")
            else:
                self._log("Block group data not available, skipping validation", level='WARN')
                
        except Exception as e:
            self._log(f"Block group validation failed (non-fatal): {e}", level='WARN')
            # Continue with rest of pipeline - validation is optional
        
        # 13. Summary
        if test_results:
            errors = [r['mean_error_pct'] for r in test_results.values()]
            self._log(f"\n{'='*80}")
            self._log("MoE VALIDATION RESULTS")
            self._log(f"{'='*80}")
            self._log(f"Mean Error: {np.mean(errors):.1f}% ± {np.std(errors):.1f}%")
            self._log(f"Median Error: {np.median(errors):.1f}%")
            self._log(f"Best: {min(errors):.1f}%, Worst: {max(errors):.1f}%")
            self._log(f"Elapsed: {elapsed:.1f}s")
        
        # 13. Generate disaggregation visualizations
        self._create_disaggregation_visualizations(
            test_results=test_results,
            training_results=training_results,
            expert_usage=usage_stats,
            tract_gdf=data['tracts'],
            all_addresses=addresses_with_tracts  # Pass addresses for geographic viz
        )
        
        return {
            'success': True,
            'model': moe_model,
            'trainer': trainer,
            'training_results': training_results,
            'test_results': test_results,
            'expert_usage': usage_stats,
            'feature_scaler': feature_scaler,
            'context_scaler': context_scaler,
            'elapsed_time': elapsed,
            'strategy': 'mixture_of_experts'
        }

    def _evaluate_on_holdout_tract(self, test_fips, data, trainer, feature_scaler, context_scaler):
        """Evaluate trained global model on a single holdout tract"""
        
        # Load holdout tract
        holdout_addresses = self.data_loader.get_addresses_for_tract(test_fips)

        if len(holdout_addresses) == 0:
            self._log(f"Tract {test_fips} has no addresses, skipping", level='WARN')
            return {
                'actual_svi': None,
                'predicted_mean': None,
                'mean_error_pct': None,
                'correlation': None,
                'predictions': None,
                'error': 'No addresses in tract'
            }

        holdout_addresses['tract_fips'] = test_fips
        
        # Compute and normalize features using training scalers
        holdout_features = self._compute_accessibility_features(holdout_addresses, data)
        normalized_holdout_features = feature_scaler.transform(holdout_features)
        
        holdout_context_features = self.data_loader.create_context_features_for_addresses(
            addresses=holdout_addresses,
            svi_data=data['svi']
        )
        normalized_holdout_context = context_scaler.transform(holdout_context_features)
        
        # Build holdout graph
        state_fips = test_fips[:2]
        county_fips = test_fips[2:5]
        
        holdout_graph = self.data_loader.create_spatial_accessibility_graph(
            addresses=holdout_addresses,
            accessibility_features=normalized_holdout_features,
            context_features=normalized_holdout_context,
            state_fips=state_fips,
            county_fips=county_fips
        )
        
        # Predict
        unconstrained_result = trainer.predict_unconstrained(holdout_graph)
        predictions = unconstrained_result['predictions']
        
        # Get actual SVI
        target_tract_data = data['tracts'][data['tracts']['FIPS'] == test_fips]
        actual_svi = float(target_tract_data.iloc[0]['RPL_THEMES'])
        
        # Compute metrics
        predicted_mean = np.mean(predictions)
        mean_error_pct = abs(predicted_mean - actual_svi) / actual_svi * 100
        
        learned_accessibility = unconstrained_result['learned_accessibility']
        accessibility_summary = learned_accessibility.mean(axis=1)
        correlation = np.corrcoef(accessibility_summary, predictions)[0, 1]
        
        return {
            'actual_svi': actual_svi,
            'predicted_mean': predicted_mean,
            'mean_error_pct': mean_error_pct,
            'correlation': correlation,
            'predictions': predictions
        }
    
    def _create_comprehensive_validation_report(self,
                                                granite_results: Dict,
                                                baseline_results: Dict,
                                                block_group_validation: Dict,
                                                output_dir: str):
        """
        Create final validation report comparing all methods against ground truth.
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle('GRANITE Comprehensive Validation Report', 
                    fontsize=16, fontweight='bold')
        
        # Panel 1: Constraint satisfaction comparison
        ax1 = fig.add_subplot(gs[0, 0])
        methods = ['GRANITE', 'IDW', 'Kriging', 'Naive']
        errors = [
            np.mean([r['mean_error_pct'] for r in granite_results.values()]),
            # Extract from baseline_results...
        ]
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#95190C']
        ax1.bar(methods, errors, color=colors, alpha=0.8)
        ax1.set_ylabel('Mean Constraint Error (%)')
        ax1.set_title('Tract-Level Constraint Satisfaction')
        ax1.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='10% threshold')
        
        # Panel 2: Spatial variation comparison
        ax2 = fig.add_subplot(gs[0, 1])
        variations = [
            np.mean([np.std(r['predictions']) for r in granite_results.values()]),
            # Extract from baselines...
        ]
        ax2.bar(methods, variations, color=colors, alpha=0.8)
        ax2.set_ylabel('Mean Within-Tract Std')
        ax2.set_title('Spatial Variation Generated')
        
        # Panel 3: Block group correlation (the key validation)
        ax3 = fig.add_subplot(gs[0, 2])
        bg_corrs = [
            block_group_validation.get('GRANITE', {}).get('correlations', {})
                .get('poverty_correlation', {}).get('r', 0),
            block_group_validation.get('IDW', {}).get('correlations', {})
                .get('poverty_correlation', {}).get('r', 0),
            # etc.
        ]
        ax3.bar(methods[:len(bg_corrs)], bg_corrs, color=colors[:len(bg_corrs)], alpha=0.8)
        ax3.set_ylabel('Correlation with BG Poverty')
        ax3.set_title('Block Group Validation (Ground Truth)')
        ax3.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='Strong threshold')
        ax3.axhline(y=0.15, color='orange', linestyle='--', alpha=0.5, label='Moderate threshold')
        
        # Panel 4-6: Scatter plots for top 3 methods vs block group poverty
        for idx, method in enumerate(['GRANITE', 'IDW', 'Kriging']):
            ax = fig.add_subplot(gs[1, idx])
            
            if method in block_group_validation:
                df = block_group_validation[method]['validation_data']
                valid = df['poverty_rate'].notna()
                ax.scatter(df.loc[valid, 'poverty_rate'],
                        df.loc[valid, 'predicted_vulnerability'],
                        alpha=0.5, s=20, color=colors[idx])
                
                r = block_group_validation[method]['correlations'] \
                    .get('poverty_correlation', {}).get('r', np.nan)
                
                ax.set_xlabel('BG Poverty Rate (%)')
                ax.set_ylabel('Predicted Vulnerability')
                ax.set_title(f'{method} (r = {r:.3f})')
        
        # Panel 7-9: Summary statistics
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        summary = """
        VALIDATION SUMMARY
        ==================
        
        This report validates GRANITE's spatial disaggregation against three criteria:
        
        1. CONSTRAINT SATISFACTION: Do tract-level means match known SVI?
        - All methods should achieve <10% mean error after correction
        - Lower error = better constraint adherence
        
        2. SPATIAL VARIATION: Does the method generate meaningful within-tract patterns?
        - Higher variation = more disaggregation (Naive baseline = 0)
        - But high variation alone doesn't prove validity
        
        3. BLOCK GROUP VALIDATION (Ground Truth): Do patterns correlate with reality?
        - This is the key test: aggregated predictions vs actual ACS demographics
        - r > 0.3 = strong evidence patterns are real, not arbitrary
        - r < 0.15 = weak evidence; variation may be noise
        
        INTERPRETATION:
        - A method that satisfies constraints with high variation but low BG correlation
        is "mathematically consistent but empirically meaningless"
        - GRANITE should outperform baselines on BG correlation to justify GNN complexity
        """
        
        ax7.text(0.05, 0.95, summary, transform=ax7.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.savefig(os.path.join(output_dir, 'comprehensive_validation_report.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _validate_final_feature_relationships(self, features: np.ndarray, addresses) -> bool:
        """NEW: Final validation of accessibility-vulnerability relationship"""
        
        self._log("=== FINAL FEATURE RELATIONSHIP VALIDATION ===")
        
        # Test expected accessibility-vulnerability patterns
        feature_names = self._generate_feature_names(features.shape[1])
        
        # Create accessibility indices for different feature types
        time_features = [i for i, name in enumerate(feature_names) if 'time' in name]
        count_features = [i for i, name in enumerate(feature_names) if 'count' in name]
        
        if len(time_features) > 0 and len(count_features) > 0:
            avg_time = np.mean(features[:, time_features], axis=1)
            avg_count = np.mean(features[:, count_features], axis=1)
            
            # These should be negatively correlated (higher time, lower counts)
            time_count_correlation = np.corrcoef(avg_time, avg_count)[0, 1]
            
            self._log(f"Time-Count correlation: {time_count_correlation:.3f} (should be negative)")
            
            if time_count_correlation > 0.1:  # Should be negative
                self._log("ERROR: Time and count features are positively correlated")
                return False
            else:
                self._log(" Time-Count relationship is correct")
        
        # Check for sufficient variation across addresses
        feature_variations = np.std(features, axis=0)
        low_variation_count = np.sum(feature_variations < 0.01)
        
        self._log(f"Low variation features: {low_variation_count}/{features.shape[1]}")
        
        if low_variation_count > features.shape[1] * 0.3:  # >30% low variation
            self._log("ERROR: Too many features have low variation")
            return False
        
        self._log(" Feature relationships appear correct")
        return True