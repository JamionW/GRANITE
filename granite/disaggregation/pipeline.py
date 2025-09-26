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

        if self.verbose:
            self._log("Running accessibility-vulnerability debugging...")
            debug_samples = self._debug_accessibility_vulnerability_relationship(
                final_predictions, accessibility_features, tract_addresses
            )
            
            direction_validation = self._validate_feature_directions(accessibility_features)
            
            correlation_diagnostic = self._create_accessibility_correlation_diagnostic(
                final_predictions, accessibility_features
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
        """Updated feature names with corrected time thresholds"""
        base_features = []
        for dest_type in ['employment', 'healthcare', 'grocery']:
            base_features.extend([
                f'{dest_type}_min_time',
                f'{dest_type}_mean_time', 
                f'{dest_type}_median_time',
                f'{dest_type}_count_15min',  # CHANGED
                f'{dest_type}_count_5min',  # CHANGED
                f'{dest_type}_count_45min',  # CHANGED
                f'{dest_type}_drive_advantage',
                f'{dest_type}_concentration',
                f'{dest_type}_time_range',
                f'{dest_type}_percentile'
            ])
        
        derived_features = [
            'local_accessibility_index',
            'modal_flexibility',
            'accessibility_equity', 
            'geographic_advantage'
        ]
        
        all_features = base_features + derived_features
        return all_features[:n_features]

    def _compute_accessibility_features(self, addresses, data):
        """
        FIXED: Compute accessibility features with enhanced error handling and validation
        """
        self._log("Computing enhanced accessibility features...")
        
        try:
            # FIXED: Initialize the enhanced computer properly
            from ..data.enhanced_accessibility import EnhancedAccessibilityComputer
            accessibility_computer = EnhancedAccessibilityComputer(verbose=self.verbose)
            
            # CHANGE 1: Get tract-appropriate destinations with validation
            target_fips = self.config.get('data', {}).get('target_fips')
            if target_fips:
                try:
                    destinations = self.data_loader.create_tract_appropriate_destinations(target_fips)
                    self._log("Using tract-appropriate destinations for better intra-tract variation")
                except Exception as e:
                    self._log(f"Warning: Could not create tract-appropriate destinations: {str(e)}")
                    # Fallback to original destinations
                    destinations = {
                        'employment': data['employment_destinations'],
                        'healthcare': data['healthcare_destinations'],
                        'grocery': data['grocery_destinations']
                    }
            else:
                destinations = {
                    'employment': data['employment_destinations'],
                    'healthcare': data['healthcare_destinations'],
                    'grocery': data['grocery_destinations']
                }
            
            # VALIDATION: Ensure all destinations are valid
            validated_destinations = {}
            for dest_type, dest_gdf in destinations.items():
                if dest_gdf is None or len(dest_gdf) == 0:
                    self._log(f"ERROR: No {dest_type} destinations available")
                    return None
                
                # Ensure proper columns exist
                dest_gdf_copy = dest_gdf.copy()
                dest_gdf_copy['dest_type'] = dest_type
                if 'dest_id' not in dest_gdf_copy.columns:
                    dest_gdf_copy['dest_id'] = range(len(dest_gdf_copy))
                
                validated_destinations[dest_type] = dest_gdf_copy
                self._log(f"  {dest_type}: {len(dest_gdf_copy)} destinations")
            
            destinations = validated_destinations
            
            # CHANGE 2: Calculate features for all destination types with error handling
            all_features = []
            feature_names = []
            successful_computations = 0
            
            for dest_type, dest_gdf in destinations.items():
                self._log(f"  Processing {dest_type} accessibility...")
                
                try:
                    # CHANGE 3: Use the FIXED travel time calculation
                    travel_times = accessibility_computer.calculate_realistic_travel_times(
                        origins=addresses,
                        destinations=dest_gdf,
                        time_period='morning'
                    )

                    # NEW: Validate travel times before feature extraction
                    if not accessibility_computer._validate_distance_time_relationship(travel_times):
                        self._log(f"ERROR: Travel time validation failed for {dest_type}")
                        continue
                        
                    # NEW: Validate destination counts
                    if not accessibility_computer._validate_destination_counts_fixed(travel_times):
                        self._log(f"ERROR: Destination count validation failed for {dest_type}")
                        continue
                    
                    if len(travel_times) == 0:
                        self._log(f"ERROR: No travel times computed for {dest_type}")
                        continue
                    
                    # Debug: Show sample travel times
                    if self.verbose and len(travel_times) > 0:
                        sample = travel_times.head(3)
                        self._log(f"  Sample {dest_type} travel times:")
                        for _, row in sample.iterrows():
                            self._log(f"    Origin {row['origin_id']} -> Dest {row['destination_id']}: "
                                    f"{row['combined_time']:.1f}min ({row['best_mode']})")
                    
                    # CHANGE 4: Use the FIXED feature extraction
                    dest_features = accessibility_computer.extract_enhanced_accessibility_features(
                        addresses=addresses,
                        travel_times=travel_times,
                        dest_type=dest_type
                    )
                    
                    if dest_features is None or dest_features.size == 0:
                        self._log(f"ERROR: No features extracted for {dest_type}")
                        continue
                    
                    # VALIDATION: Check feature dimensions
                    expected_addresses = len(addresses)
                    if dest_features.shape[0] != expected_addresses:
                        self._log(f"ERROR: Feature count mismatch for {dest_type}: "
                                f"got {dest_features.shape[0]}, expected {expected_addresses}")
                        continue
                    
                    if dest_features.shape[1] != 10:  # Expected 10 features per destination type
                        self._log(f"WARNING: Unexpected feature count for {dest_type}: "
                                f"got {dest_features.shape[1]}, expected 10")
                    
                    # VALIDATION: Check for systematic issues
                    zero_var_features = np.sum(np.std(dest_features, axis=0) < 1e-8)
                    if zero_var_features > 0:
                        self._log(f"ERROR: {dest_type} has {zero_var_features} zero-variance features")
                        continue
                    
                    all_features.append(dest_features)
                    feature_names.extend([
                        f'{dest_type}_min_time', f'{dest_type}_mean_time', f'{dest_type}_median_time',
                        f'{dest_type}_count_5min', f'{dest_type}_count_10min', f'{dest_type}_count_15min',
                        f'{dest_type}_drive_advantage', f'{dest_type}_concentration',
                        f'{dest_type}_time_range', f'{dest_type}_percentile'
                    ])
                    
                    successful_computations += 1
                    self._log(f"  ✓ {dest_type}: {dest_features.shape} features computed successfully")
                    
                except Exception as e:
                    self._log(f"ERROR processing {dest_type}: {str(e)}")
                    import traceback
                    self._log(f"Traceback: {traceback.format_exc()}")
                    continue
            
            # CRITICAL CHECK: Ensure we have features
            if successful_computations == 0:
                self._log("CRITICAL ERROR: No accessibility features could be computed for any destination type")
                return None
            
            if len(all_features) == 0:
                self._log("CRITICAL ERROR: Feature list is empty")
                return None
            
            # CHANGE 5: Combine features with validation
            try:
                accessibility_matrix = np.column_stack(all_features)
                self._log(f"Base accessibility features: {accessibility_matrix.shape}")

                validation_passed = self._validate_final_feature_relationships(
                    accessibility_matrix, addresses
                )
                
            except Exception as e:
                self._log(f"ERROR combining features: {str(e)}")
                return None
            
            # CHANGE 6: Compute derived features with validation and graceful handling
            # Initialize final_features with base features first
            final_features = accessibility_matrix.copy()
            complete_feature_names = self._generate_feature_names(final_features.shape[1])

            try:
                derived_features = accessibility_computer.compute_enhanced_derived_features(accessibility_matrix)
                
                if derived_features is None or derived_features.size == 0:
                    self._log("WARNING: Could not compute derived features, continuing with base features only")
                else:
                    # Check if derived features have variance
                    derived_var_mask = np.std(derived_features, axis=0) > 1e-8
                    valid_derived_count = np.sum(derived_var_mask)
                    
                    if valid_derived_count > 0:
                        # Only add derived features that have variance
                        valid_derived_features = derived_features[:, derived_var_mask]
                        final_features = np.column_stack([final_features, valid_derived_features])
                        
                        # Add only valid derived feature names
                        derived_names = ['local_accessibility_index', 'modal_flexibility', 
                                    'accessibility_equity', 'geographic_advantage']
                        valid_derived_names = [name for i, name in enumerate(derived_names) if i < len(derived_var_mask) and derived_var_mask[i]]
                        
                        # Update complete feature names to include valid derived features
                        base_feature_names = self._generate_feature_names(accessibility_matrix.shape[1])
                        complete_feature_names = base_feature_names + valid_derived_names
                        
                        self._log(f"Added {valid_derived_count} valid derived features")
                    else:
                        self._log("WARNING: All derived features have zero variance, skipping them")
                
            except Exception as e:
                self._log(f"ERROR computing derived features: {str(e)}")
                self._log("Continuing with base features only")
            
            # FINAL VALIDATION
            complete_feature_names = feature_names + [
                'local_accessibility_index', 'modal_flexibility', 
                'accessibility_equity', 'geographic_advantage'
            ]
            
            self._log(f"Final feature matrix: {final_features.shape}")
            self._log(f"Feature names count: {len(complete_feature_names)}")
            
            # Critical checks
            if final_features.size == 0:
                self._log("CRITICAL ERROR: Final feature matrix is empty")
                return None
            
            if np.any(np.isnan(final_features)):
                nan_count = np.sum(np.isnan(final_features))
                self._log(f"ERROR: {nan_count} NaN values in feature matrix")
                # Try to handle NaN values
                final_features = np.nan_to_num(final_features, nan=0.0)
            
            if np.any(np.isinf(final_features)):
                inf_count = np.sum(np.isinf(final_features))
                self._log(f"ERROR: {inf_count} infinite values in feature matrix")
                final_features = np.nan_to_num(final_features, posinf=999.0, neginf=-999.0)
            
            # Check variance and remove zero-variance features
            zero_var_mask = np.std(final_features, axis=0) < 1e-8
            zero_var_count = np.sum(zero_var_mask)

            if zero_var_count > 0:
                self._log(f"WARNING: {zero_var_count} features have zero variance, removing them")
                
                # Debug zero variance features
                for i in range(final_features.shape[1]):
                    if zero_var_mask[i]:
                        feature_name = complete_feature_names[i] if i < len(complete_feature_names) else f"feature_{i}"
                        unique_vals = len(np.unique(final_features[:, i]))
                        self._log(f"  Removing {feature_name}: {unique_vals} unique values, std={np.std(final_features[:, i]):.8f}")
                
                # Remove zero-variance features
                valid_feature_mask = ~zero_var_mask
                final_features = final_features[:, valid_feature_mask]
                complete_feature_names = [name for i, name in enumerate(complete_feature_names) if valid_feature_mask[i]]
                
                # Check if we have enough features remaining
                if final_features.shape[1] < 10:
                    self._log(f"CRITICAL ERROR: Only {final_features.shape[1]} features remaining after removing zero-variance")
                    return None
                
                self._log(f"Continuing with {final_features.shape[1]} valid features")
            
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
                self._log(f"WARNING: Unexpected negative values in: {negative_features[:3]}")  # Show first 3
            else:
                self._log("✓ No unexpected negative values detected")
            
            self._log("✓ All features have proper variance")
            self._log(f"SUCCESS: Generated {final_features.shape[1]} features for {final_features.shape[0]} addresses")
            
            from ..evaluation.feature_deduplication import enhance_accessibility_features_with_validation
        
            self._log("Applying feature enhancement and validation...")
            
            enhanced_features, enhanced_names, validation_results = enhance_accessibility_features_with_validation(
                features=final_features,
                feature_names=complete_feature_names, 
                addresses_count=len(addresses),
                verbose=self.verbose
            )
            
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
            self._log(f"  Original features: {original_feature_count}")
            self._log(f"  Enhanced features: {final_feature_count}")
            self._log(f"  Quality grade: {quality_grade} ({quality_score:.1f}%)")
            
            # Warn if quality is poor
            if quality_grade in ['D', 'F']:
                self._log(f"WARNING: Feature quality is poor - consider investigating")
                
            # Store validation results for later use
            self._feature_validation_results = validation_results
            
            return enhanced_features
            
        except Exception as e:
            self._log(f"CRITICAL ERROR in accessibility computation: {str(e)}")
            import traceback
            self._log(f"Full traceback: {traceback.format_exc()}")
            return None

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
                    count_5min = int((combined_times <= 30).sum())
                    count_10min = int((combined_times <= 60).sum())
                    count_15min = int((combined_times <= 90).sum())
                    
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
        """INTEGRATED: Train GNN with all fixes applied"""
        self._log("Training Accessibility SVI GNN with enhanced architecture...")
        
        try:
            # Import FIXED GNN classes
            from ..models.gnn import AccessibilitySVIGNN, AccessibilityGNNTrainer, normalize_accessibility_features
            import torch
            
            # Apply FIXED normalization
            normalized_features, feature_scaler = normalize_accessibility_features(graph_data.x.numpy())
            graph_data.x = torch.FloatTensor(normalized_features)
            
            # Create FIXED model with proper architecture
            model = AccessibilitySVIGNN(
                accessibility_features_dim=graph_data.x.shape[1],
                hidden_dim=self.config.get('model', {}).get('hidden_dim', 64),
                dropout=self.config.get('model', {}).get('dropout', 0.3)
            )
            
            # Create FIXED trainer
            trainer = AccessibilityGNNTrainer(model, config=self.config.get('training', {}))
            
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
            self._stored_raw_predictions = predictions.copy()
            
            # Enhanced result reporting
            constraint_error = abs(np.mean(predictions) - tract_svi) / tract_svi * 100
            spatial_std = np.std(predictions)
            
            self._log(f"Training completed:")
            self._log(f"  Constraint error: {constraint_error:.2f}%")
            self._log(f"  Spatial variation: {spatial_std:.4f}")
            self._log(f"  Learning converged: {training_result.get('learning_converged', False)}")
            
            # Quality assessment
            if constraint_error < 10 and spatial_std > 0.01:
                self._log("✓ Training quality: GOOD")
            elif constraint_error < 25:
                self._log("⚠ Training quality: ACCEPTABLE")
            else:
                self._log("✗ Training quality: POOR - consider hyperparameter tuning")
            
            return {
                'success': True,
                'raw_predictions': predictions,
                'learned_accessibility': training_result.get('learned_accessibility'),
                'model': model,
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
                'learned_accessibility': results['training_result'].get('learned_accessibility'),  # NEW
                'traditional_accessibility': results['accessibility_features'],  # For comparison
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
            
            self._log(f"  Address {addr_id}: SVI={pred_svi:.3f}")
            self._log(f"    Avg min travel time: {avg_min_time:.1f} min")
            self._log(f"    Employment: {emp_min_time:.1f}min, {emp_count_5min} jobs in 5min")
            self._log(f"    Healthcare: {health_min_time:.1f}min")
            self._log(f"    Grocery: {grocery_min_time:.1f}min")
            
            # RED FLAG CHECK: If high vulnerability has very good accessibility
            if avg_min_time < 6.0:  # Very good accessibility (< 6min average)
                self._log(f"    🚨 RED FLAG: High vulnerability but excellent accessibility!")
        
        self._log("\nLOW VULNERABILITY ADDRESSES (should have GOOD accessibility):")
        
        for i, idx in enumerate(low_vuln_indices):
            addr_id = addresses.iloc[idx].get('address_id', idx)
            pred_svi = predicted_svi[idx]
            
            emp_min_time = accessibility_features[idx, 0]
            emp_count_5min = accessibility_features[idx, 3]
            health_min_time = accessibility_features[idx, 10]
            grocery_min_time = accessibility_features[idx, 20]
            
            avg_min_time = (emp_min_time + health_min_time + grocery_min_time) / 3
            
            self._log(f"  Address {addr_id}: SVI={pred_svi:.3f}")
            self._log(f"    Avg min travel time: {avg_min_time:.1f} min")
            self._log(f"    Employment: {emp_min_time:.1f}min, {emp_count_5min} jobs in 5min")
            self._log(f"    Healthcare: {health_min_time:.1f}min")  
            self._log(f"    Grocery: {grocery_min_time:.1f}min")
            
            # RED FLAG CHECK: If low vulnerability has very poor accessibility
            if avg_min_time > 12.0:  # Poor accessibility (> 12min average)
                self._log(f"    🚨 RED FLAG: Low vulnerability but poor accessibility!")
        
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
            self._log(f"  {name}: {val:.2f}")
        
        self._log("Poor accessibility profile (should predict HIGH vulnerability):")
        for i, (name, val) in enumerate(zip(feature_names[:10], poor_access_profile[:10])):  # Show first 10
            self._log(f"  {name}: {val:.2f}")
        
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
    
    def _create_accessibility_correlation_diagnostic(self, final_predictions, accessibility_features):
        """Create detailed correlation diagnostic"""
        
        self._log("=== ACCESSIBILITY CORRELATION DIAGNOSTIC ===")
        
        predicted_svi = final_predictions['mean'].values
        feature_names = self._generate_feature_names(accessibility_features.shape[1])
        
        # Calculate correlation of each feature with predicted SVI
        correlations = []
        
        for i, feature_name in enumerate(feature_names):
            feature_values = accessibility_features[:, i]
            corr = np.corrcoef(feature_values, predicted_svi)[0, 1]
            
            # Determine if correlation direction is theoretically correct
            expected_direction = "POSITIVE" if any(word in feature_name for word in ['time', 'drive_advantage']) else "NEGATIVE"
            actual_direction = "POSITIVE" if corr > 0 else "NEGATIVE"
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
        for i, corr_info in enumerate(correlations[:15]):  # Show top 15
            status = "✓" if corr_info['correct'] else "✗"
            self._log(f"  {status} {corr_info['feature']}: r={corr_info['correlation']:.3f} "
                    f"({corr_info['strength']}, expected {corr_info['expected']})")
            
            if corr_info['correct']:
                correct_count += 1
        
        total_features = len(correlations)
        correctness_rate = correct_count / total_features * 100
        
        self._log(f"\nOVERALL DIRECTIONAL CORRECTNESS: {correct_count}/{total_features} ({correctness_rate:.1f}%)")
        
        if correctness_rate < 60:
            self._log("🚨 MAJOR ISSUE: Less than 60% of features have correct correlation direction!")
            self._log("   This suggests systematic feature encoding problems.")
        elif correctness_rate < 80:
            self._log("⚠️  WARNING: Some features may be incorrectly encoded or model is learning confounding patterns")
        else:
            self._log("✓ Feature directions appear mostly correct")
        
        return correlations
    
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
                self._log("✓ Time-Count relationship is correct")
        
        # Check for sufficient variation across addresses
        feature_variations = np.std(features, axis=0)
        low_variation_count = np.sum(feature_variations < 0.01)
        
        self._log(f"Low variation features: {low_variation_count}/{features.shape[1]}")
        
        if low_variation_count > features.shape[1] * 0.3:  # >30% low variation
            self._log("ERROR: Too many features have low variation")
            return False
        
        self._log("✓ Feature relationships appear correct")
        return True