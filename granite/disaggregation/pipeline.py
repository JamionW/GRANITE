"""
Simplified GRANITE Pipeline: Accessibility → SVI Direct Prediction
Eliminates two-stage complexity, focuses on robust accessibility feature extraction
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime
warnings.filterwarnings('ignore')

from ..data.loaders import DataLoader
from ..visualization.plots import GRANITEResearchVisualizer
from ..evaluation.spatial_diagnostics import SpatialLearningDiagnostics
from ..evaluation.accessibility_validator import validate_granite_accessibility_features, integrate_with_spatial_diagnostics

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
        
        self.data_loader = DataLoader(data_dir, config=config)
        self.visualizer = GRANITEResearchVisualizer()
        
    def _log(self, message, level='INFO'):
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")

    def run(self):
        """Main pipeline execution"""
        start_time = time.time()
        
        self._log("Starting GRANITE Accessibility SVI Pipeline...")
        
        # Load data
        data = self._load_spatial_data()
        
        # Process single FIPS 
        target_fips = self.config.get('data', {}).get('target_fips')
        if not target_fips:
            return {'success': False, 'error': 'FIPS code required'}
        
        results = self._process_single_tract(target_fips, data)
        
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
        data['employment_destinations'] = self.data_loader.create_employment_destinations()
        data['healthcare_destinations'] = self.data_loader.create_healthcare_destinations()
        data['grocery_destinations'] = self.data_loader.create_grocery_destinations()
        
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
        """Process single tract with simplified accessibility → SVI approach"""
        self._log(f"Processing tract {target_fips}...")
        
        # Get tract data
        target_fips = str(target_fips).strip()
        data['tracts']['FIPS'] = data['tracts']['FIPS'].astype(str).str.strip()
        
        target_tract = data['tracts'][data['tracts']['FIPS'] == target_fips]
        if len(target_tract) == 0:
            return {'success': False, 'error': f'FIPS {target_fips} not found'}
        
        state_fips = target_fips[:2]
        county_fips = target_fips[2:5]
        
        tract_info = target_tract.iloc[0]
        tract_svi = tract_info['RPL_THEMES']
        
        # Get addresses for this tract
        tract_addresses = self.data_loader.get_addresses_for_tract(target_fips)
        
        if len(tract_addresses) == 0:
            return {'success': False, 'error': f'No addresses found for tract {target_fips}'}
        
        self._log(f"Found {len(tract_addresses)} addresses in tract {target_fips}")
        self._log(f"Target SVI: {tract_svi:.4f}")
        
        # Step 1: Compute accessibility features
        accessibility_features = self._compute_accessibility_features(
            tract_addresses, data
        )
        
        if accessibility_features is None:
            return {'success': False, 'error': 'Failed to compute accessibility features'}
        
        # Generate feature names (you need this defined)
        feature_names = self._generate_feature_names(accessibility_features.shape[1])
        
        # Step 2: Build accessibility-based graph
        from ..models.gnn import normalize_accessibility_features

        normalized_features, feature_scaler = normalize_accessibility_features(accessibility_features)

        # Create spatial graph based on road network and geographic proximity
        graph_data = self.data_loader.create_spatial_accessibility_graph(
            addresses=tract_addresses,
            accessibility_features=normalized_features,
            state_fips=state_fips,
            county_fips=county_fips
        )
        
        self._log(f"Built graph: {graph_data.x.shape[0]} nodes, {graph_data.edge_index.shape[1]} edges")
        
        # Step 3: Train GNN for accessibility and SVI prediction
        training_result = self._train_accessibility_svi_gnn(
            graph_data, tract_svi, tract_addresses
        )
        
        if not training_result['success']:
            return training_result
        
        # Step 4: Create final predictions with constraint satisfaction
        final_predictions = self._finalize_predictions(
            training_result['raw_predictions'], 
            tract_addresses, 
            tract_svi
        )
        
        # Step 5a: Run accessibility feature validation FIRST
        self._log("Running accessibility feature validation...")
        try:
            access_validation_results, access_validator = validate_granite_accessibility_features(
                addresses=tract_addresses,
                accessibility_features=accessibility_features,
                destinations={
                    'employment': data['employment_destinations'],
                    'healthcare': data['healthcare_destinations'], 
                    'grocery': data['grocery_destinations']
                },
                feature_names=feature_names,
                tract_svi=tract_svi,
                output_dir=os.path.join(self.output_dir, 'accessibility_validation')
            )
        except Exception as e:
            self._log(f"Warning: Accessibility validation failed: {str(e)}")
            access_validation_results = {'error': str(e)}
        
        # Step 5b: Run spatial diagnostics (modified to return results)
        validation_results = self._validate_predictions(
            final_predictions, tract_svi, accessibility_features, normalized_features
        )
        
        # Step 5c: Integrate results if both successful
        if 'error' not in access_validation_results and 'spatial_diagnostics' in validation_results:
            try:
                integrated_results = integrate_with_spatial_diagnostics(
                    spatial_diagnostics_results=validation_results['spatial_diagnostics'],
                    accessibility_validation_results=access_validation_results
                )
                validation_results['integrated_analysis'] = integrated_results
            except Exception as e:
                self._log(f"Warning: Integration failed: {str(e)}")
        
        # Store accessibility validation results
        validation_results['accessibility_validation'] = access_validation_results
        
        # Save results and create visualizations
        if self.verbose:
            self._create_research_visualizations({
                'predictions': final_predictions,
                'accessibility_features': accessibility_features,
                'validation_results': validation_results,
                'tract_svi': tract_svi,
                'training_result': training_result
            })
        
        return {
            'success': True,
            'predictions': final_predictions,
            'tract_info': tract_info,
            'accessibility_features': accessibility_features,
            'training_result': training_result,
            'validation_results': validation_results,
            'methodology': 'Direct Accessibility → SVI Prediction',
            'summary': {
                'addresses_processed': len(tract_addresses),
                'accessibility_features': accessibility_features.shape[1],
                'spatial_variation': np.std(final_predictions['mean']),
                'constraint_error': abs(np.mean(final_predictions['mean']) - tract_svi) / tract_svi * 100,
                'training_epochs': training_result.get('epochs_trained', 0)
            }
        }
    
    def _generate_feature_names(self, n_features):
        """Generate feature names based on GRANITE structure"""
        base_features = []
        
        # Employment, healthcare, grocery features (8 each)
        for dest_type in ['employment', 'healthcare', 'grocery']:
            base_features.extend([
                f'{dest_type}_min_time',
                f'{dest_type}_mean_time', 
                f'{dest_type}_90th_time',
                f'{dest_type}_count_30min',
                f'{dest_type}_count_60min',
                f'{dest_type}_count_90min',
                f'{dest_type}_transit_share',
                f'{dest_type}_accessibility_score'
            ])
        
        # Derived features (4)
        derived_features = [
            'total_accessibility',
            'accessibility_diversity',
            'avg_transit_dependence', 
            'time_efficiency'
        ]
        
        # Return appropriate subset
        all_features = base_features + derived_features
        return all_features[:n_features]

    def _compute_accessibility_features(self, addresses, data):
        """
        Compute comprehensive accessibility features for addresses
        
        Features:
        - Travel times to employment, healthcare, grocery (min, mean, 90th percentile)
        - Destination counts within time thresholds (30, 60, 90 minutes)
        - Transit accessibility scores
        - Multi-modal accessibility balance
        """
        self._log("Computing accessibility features...")
        
        try:
            # Prepare destinations dictionary
            destinations = {
                'employment': data['employment_destinations'],
                'healthcare': data['healthcare_destinations'],
                'grocery': data['grocery_destinations']
            }
            
            # Calculate travel times for all destination types
            all_features = []
            feature_names = []
            
            for dest_type, dest_gdf in destinations.items():
                if dest_gdf is None or len(dest_gdf) == 0:
                    self._log(f"Warning: No {dest_type} destinations available")
                    continue
                
                self._log(f"  Calculating {dest_type} accessibility...")
                
                # Add destination metadata
                dest_gdf = dest_gdf.copy()
                dest_gdf['dest_type'] = dest_type
                if 'dest_id' not in dest_gdf.columns:
                    dest_gdf['dest_id'] = range(len(dest_gdf))
                
                # Calculate travel times
                travel_times = self.data_loader.calculate_multimodal_travel_times_batch(
                    addresses, dest_gdf, time_periods=['morning']
                )
                
                # Extract features for this destination type
                dest_features = self._extract_accessibility_features(
                    addresses, travel_times, dest_type
                )
                
                all_features.append(dest_features)
                feature_names.extend([
                    f'{dest_type}_min_time', f'{dest_type}_mean_time', f'{dest_type}_90th_time',
                    f'{dest_type}_count_30min', f'{dest_type}_count_60min', f'{dest_type}_count_90min',
                    f'{dest_type}_transit_share', f'{dest_type}_accessibility_score'
                ])
            
            if not all_features:
                self._log("Error: No accessibility features could be computed")
                return None
            
            # Combine all features
            accessibility_matrix = np.column_stack(all_features)
            
            self._log(f"Computed accessibility features: {accessibility_matrix.shape}")
            self._log(f"Feature names: {feature_names}")
            
            # Add derived features
            derived_features = self._compute_derived_accessibility_features(accessibility_matrix)
            final_features = np.column_stack([accessibility_matrix, derived_features])
            
            self._log(f"Final feature matrix: {final_features.shape}")
            
            return final_features
            
        except Exception as e:
            self._log(f"Error computing accessibility features: {str(e)}")
            import traceback
            self._log(f"Traceback: {traceback.format_exc()}")

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
                    count_30min = int((combined_times <= 30).sum())
                    count_60min = int((combined_times <= 60).sum())
                    count_90min = int((combined_times <= 90).sum())
                    
                    # Transit accessibility
                    transit_share = float((addr_times['best_mode'] == 'transit').mean())
                    
                    # Overall accessibility score (gravity-style)
                    accessibility_score = float(np.sum(1.0 / np.maximum(combined_times, 1.0)))
                    
                else:
                    # No valid travel times
                    min_time = mean_time = percentile_90 = 120.0
                    count_30min = count_60min = count_90min = 0
                    transit_share = accessibility_score = 0.0
            else:
                # No travel times for this address
                min_time = mean_time = percentile_90 = 120.0
                count_30min = count_60min = count_90min = 0
                transit_share = accessibility_score = 0.0
            
            features.append([
                min_time, mean_time, percentile_90,
                count_30min, count_60min, count_90min,
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
        """Train GNN for direct accessibility and SVI predictions"""
        
        self._log("Training Accessibility SVI GNN...")
        
        try:
            from ..models.gnn import AccessibilitySVIGNN, AccessibilityGNNTrainer
            import torch
            
            # Create model
            model = AccessibilitySVIGNN(
                accessibility_features_dim=graph_data.x.shape[1],
                hidden_dim=self.config.get('model', {}).get('hidden_dim', 64),
                dropout=self.config.get('model', {}).get('dropout', 0.3)
            )
            
            # Create trainer
            trainer = AccessibilityGNNTrainer(model, config=self.config.get('training', {}))
            
            # Training parameters
            epochs = self.config.get('model', {}).get('epochs', 100)
            
            self._log(f"Training model: {graph_data.x.shape[1]} features for SVI prediction")
            self._log(f"Target SVI: {tract_svi:.4f}, Epochs: {epochs}")
            
            # Train
            training_result = trainer.train(
                graph_data=graph_data,
                tract_svi=tract_svi,
                epochs=epochs,
                verbose=self.verbose
            )
            
            # STORE RAW PREDICTIONS for diagnostics
            predictions = training_result['final_predictions']
            self._stored_raw_predictions = predictions.copy()  # Store a copy for diagnostics
            
            # Calculate constraint error on RAW predictions
            raw_constraint_error = abs(np.mean(predictions) - tract_svi) / tract_svi
            spatial_std = np.std(predictions)
            
            self._log(f"Training completed:")
            self._log(f"  RAW constraint error: {raw_constraint_error:.6f}")
            self._log(f"  RAW spatial variation: {spatial_std:.6f}")
            self._log(f"  RAW prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
            self._log(f"  RAW prediction mean: {np.mean(predictions):.4f} (target: {tract_svi:.4f})")
            
            # Quality checks on RAW predictions
            if raw_constraint_error > 0.5:  # 50% tolerance for raw predictions
                self._log("Warning: Very high RAW constraint error - model struggling to learn target scale")
            
            if spatial_std < 0.005:
                self._log("Warning: Very low RAW spatial variation - model may be predicting constant values")
            
            return {
                'success': True,
                'raw_predictions': predictions,
                'model': model,
                'training_history': training_result.get('training_history', []),
                'raw_constraint_error': raw_constraint_error,
                'spatial_std': spatial_std,
                'epochs_trained': training_result.get('epochs_trained', epochs)
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
        self._log(f"  Raw mean: {current_mean:.4f}")
        self._log(f"  Target: {tract_svi:.4f}")  
        self._log(f"  Pre-correction constraint error: {pre_correction_error:.2f}%")
        
        # Apply constraint correction
        adjustment = tract_svi - current_mean
        adjusted_predictions = predictions + adjustment
        adjusted_predictions = np.clip(adjusted_predictions, 0.0, 1.0)
        
        # Verify constraint is satisfied
        final_mean = np.mean(adjusted_predictions)
        final_error = abs(final_mean - tract_svi) / tract_svi * 100
        
        self._log(f"Post-correction analysis:")
        self._log(f"  Adjustment applied: {adjustment:.4f}")
        self._log(f"  Final mean: {final_mean:.4f}")
        self._log(f"  Post-correction constraint error: {final_error:.2f}%")
        
        # Assess if correction is reasonable
        if abs(adjustment) > 0.1:  # More than 10% of SVI scale
            self._log(f"WARNING: Large adjustment ({adjustment:.4f}) suggests model is not learning appropriate scale")
        
        if pre_correction_error > 50:  # More than 50% error
            self._log(f"WARNING: Very high pre-correction error suggests fundamental model issues")
        
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
            'mean': adjusted_predictions,
            'sd': total_uncertainty,
            'q025': np.clip(adjusted_predictions - 1.96 * total_uncertainty, 0.0, 1.0),
            'q975': np.clip(adjusted_predictions + 1.96 * total_uncertainty, 0.0, 1.0),
            'raw_prediction': predictions,  # NEW: Include raw predictions for reference
            'adjustment': adjustment  # NEW: Show the adjustment applied
        })
        
        self._log(f"Finalized predictions: {len(final_predictions)} addresses")
        self._log(f"  Final spatial std: {np.std(adjusted_predictions):.4f}")
        
        return final_predictions

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
                raw_predictions=raw_predictions,
                accessibility_features=accessibility_features,
                coordinates=coordinates, 
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
        self._log(f"  Final constraint error: {constraint_error:.2f}%")
        self._log(f"  Final spatial variation: {spatial_std:.4f}")
        self._log(f"  Final accessibility-SVI correlation: {accessibility_svi_correlations.get('overall', 'N/A')}")
        
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
                'tract_svi': results['tract_svi'],
                'validation_results': results['validation_results'],
                'training_result': results['training_result']
            }
            
            # Create visualizations
            self.visualizer.create_comprehensive_research_analysis(viz_data, viz_output_dir)
            
            self._log(f"Research visualizations saved to {viz_output_dir}")
            
        except Exception as e:
            self._log(f"Warning: Could not create visualizations: {str(e)}")

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