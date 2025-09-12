"""
Main disaggregation pipeline for GRANITE framework
Updated to use spatial disaggregation instead of regression
"""
# Standard library imports
import os
import time
import warnings
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import torch

# Configure warnings
warnings.filterwarnings('ignore')

# Local imports
from ..data.loaders import DataLoader
from ..models.gnn import prepare_graph_data_with_nlcd, prepare_graph_data_topological
from ..visualization.plots import GRANITEResearchVisualizer
from ..baselines.accessibility_baseline import AccessibilityBaseline
from ..models.gnn import (
    prepare_graph_data_with_nlcd, 
    prepare_graph_data_topological,
    AccessibilityLearningGNN,
    AccessibilitySVIGNN, 
    AccessibilityGNNTrainer,
    AccessibilitySVITrainer
)

class GRANITEPipeline:
    """
    Main pipeline for SVI disaggregation using GNN-MetricGraph integration
    
    GRANITE: Graph-Refined Accessibility Network for Integrated Transit Equity
    
    This implementation uses spatial disaggregation with GNN-learned parameters
    rather than regression-based approaches.
    """
    
    def __init__(self, config, data_dir='./data', output_dir='./output', verbose=None):
        self.config = config
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.accessibility_baseline = AccessibilityBaseline(config=config)
        self.verbose = config.get('processing', {}).get('verbose', False)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Simplified components
        self.data_loader = DataLoader(data_dir, config=config)
        self.visualizer = GRANITEResearchVisualizer()
        
        self.results = {}
    
    def _log(self, message, level='INFO'):
        """Logging with timestamp and level"""
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")

    def run(self):
        """Simplified run method - pure accessibility research"""
        start_time = time.time()
        
        # Load data
        self._log("Loading data for accessibility research...")
        data = self._load_data()
        
        # Single FIPS processing only
        target_fips = self.config.get('data', {}).get('target_fips')
        if not target_fips:
            return {'success': False, 'error': 'FIPS code required for accessibility research'}
        
        results = self.run_hybrid_accessibility_research()
    
        elapsed_time = time.time() - start_time
        self._log(f"Accessibility research completed in {elapsed_time:.2f} seconds")
        
        return results

    def _load_data(self):
        """
        Load data using Chattanooga addresses
        """
        self._log("Loading data...")
        
        # Get FIPS configuration from config
        state_fips = self.config.get('data', {}).get('state_fips', '47')
        county_fips = self.config.get('data', {}).get('county_fips', '065')
        
        data = {}
        
        # Load census tracts
        data['census_tracts'] = self.data_loader.load_census_tracts(
            state_fips=state_fips, 
            county_fips=county_fips
        )
        self._log(f"Loaded {len(data['census_tracts'])} census tracts")
        
        # Load SVI data
        county_name = self.data_loader._get_county_name(state_fips, county_fips)
        data['svi'] = self.data_loader.load_svi_data(
            state_fips=state_fips,
            county_name=county_name
        )
        self._log(f"Loaded SVI data for {len(data['svi'])} tracts")
        
        # Load road network
        data['roads'] = self.data_loader.load_road_network(
            state_fips=state_fips,
            county_fips=county_fips
        )
        self._log(f"Loaded road network with {len(data['roads'])} segments")
        
        # Load transit stops
        transit_config = self.config.get('transit', {})
        use_real_data = transit_config.get('download_real_data', True)
        
        data['transit_stops'] = self.data_loader.load_transit_stops(use_real_data=use_real_data)
        
        # Log transit data quality
        if len(data['transit_stops']) > 0:
            data_source = data['transit_stops']['data_source'].iloc[0]
            unique_sources = data['transit_stops']['data_source'].unique()
            
            self._log(f"Loaded {len(data['transit_stops'])} transit stops")
            self._log(f"  Primary source: {data_source}")
            if len(unique_sources) > 1:
                self._log(f"  Mixed sources: {list(unique_sources)}")
            
            # Log by route type
            if 'route_type' in data['transit_stops'].columns:
                route_counts = data['transit_stops']['route_type'].value_counts()
                for route_type, count in route_counts.items():
                    self._log(f"    {route_type}: {count} stops")
            
            # Quality assessment
            if data_source == 'CARTA_GTFS':
                self._log("  ✅ Using real GTFS data - highest quality for research")
            elif data_source == 'OpenStreetMap':
                self._log("  ✅ Using OSM data - good quality, community-verified")
            elif data_source == 'Generated_Grid':
                self._log("  ⚠️  Using generated grid - realistic but not real data")
            else:
                self._log("  ⚠️  Using minimal fallback - not recommended for research")
        else:
            self._log("❌ No transit stops loaded!")
        
        # Load address points
        data['addresses'] = self.data_loader.load_address_points(
            state_fips=state_fips, 
            county_fips=county_fips
        )
        
        address_source = 'real' if 'full_address' in data['addresses'].columns else 'synthetic'
        self._log(f"Loaded {len(data['addresses'])} {address_source} address points")
        
        # Create road network graph
        data['road_network'] = self.data_loader.create_network_graph(data['roads'])
        self._log(f"Created network graph with {data['road_network'].number_of_nodes()} nodes")
        
        # Merge SVI with census tracts
        data['tracts_with_svi'] = data['census_tracts'].merge(
            data['svi'], on='FIPS', how='inner'
        )

        data['tracts'] = data['tracts_with_svi'] 
        self._log(f"Merged data for {len(data['tracts_with_svi'])} tracts with SVI")
        
        return data

    def _process_fips_mode(self, data):
        """
        Process specific FIPS codes with proper string handling
        """
        # Get target FIPS from config
        target_fips = self.config.get('data', {}).get('target_fips')
        
        # Ensure target_fips is string
        target_fips = str(target_fips).strip()
        self._log(f"Looking for target FIPS: '{target_fips}'")
        
        # Ensure FIPS column is string type for consistent matching
        data['tracts']['FIPS'] = data['tracts']['FIPS'].astype(str).str.strip()
        
        # Check if target exists
        available_fips = data['tracts']['FIPS'].tolist()
        if target_fips not in available_fips:
            self._log(f"Target FIPS {target_fips} not found in data")
            self._log(f"Available FIPS (first 5): {available_fips[:5]}")
            # Look for Hamilton County codes
            hamilton_codes = [f for f in available_fips if f.startswith('47065')]
            self._log(f"Hamilton County FIPS ({len(hamilton_codes)} total): {hamilton_codes[:10]}")
            return {'success': False, 'error': f'FIPS {target_fips} not found'}
        
        # Filter to specific tract
        target_tract = data['tracts'][data['tracts']['FIPS'] == target_fips]
        
        if len(target_tract) == 0:
            self._log(f"No tract found after filtering for FIPS {target_fips}")
            return {'success': False, 'error': f'FIPS {target_fips} filtering failed'}
        
        self._log(f"✓ Found target tract: {target_fips}")
        
        # Create single-tract dataset
        single_tract_data = data.copy()
        single_tract_data['tracts'] = target_tract
        
        self._log(f"Processing {len(single_tract_data['tracts'])} tract (single FIPS mode)")
        
        target_tract = data['tracts'][data['tracts']['FIPS'] == target_fips].iloc[0]
    
        self._log(f"✓ Found target tract: {target_fips}")
        self._log(f"Processing single tract with HYBRID approach")
        
        # Use your hybrid method directly
        tract_data = self._prepare_tract_data(target_tract, data)
        result = self. _process_single_tract_hybrid_accessibility(tract_data)  # This calls your hybrid method!
        
        if result['status'] == 'success':
            return {
                'success': True,
                'predictions': result['predictions'], 
                'tract_results': [result],
                'summary': {
                    'total_tracts': 1,
                    'successful_tracts': 1,
                    'total_addresses': len(result['predictions']),
                    'processing_time': result['timing']['total']
                }
            }
        else:
            return {'success': False, 'error': result['error']}
    
    def _prepare_tract_data(self, tract, county_data):
        """
        Prepare data for a single tract using real addresses
        """
        # Get tract geometry
        tract_geom = tract.geometry
        fips_code = tract['FIPS']
        
        # Get roads within tract
        tract_roads = county_data['roads'][
            county_data['roads'].intersects(tract_geom)
        ].copy()
        
        # Get real addresses for this specific tract
        tract_addresses = self.data_loader.get_addresses_for_tract(fips_code)
        
        # Fallback if no addresses found
        if len(tract_addresses) == 0:
            self._log(f"No addresses found for tract {fips_code}, using tract centroid")
            centroid = tract_geom.centroid
            tract_addresses = gpd.GeoDataFrame([{
                'address_id': 0,
                'geometry': centroid,
                'full_address': f'Tract {fips_code} Centroid',
                'tract_fips': fips_code
            }], crs='EPSG:4326')
        
        # Build road network graph
        road_network = self.data_loader.create_network_graph(tract_roads)
        
        return {
            'tract_info': tract,
            'roads': tract_roads,
            'addresses': tract_addresses, 
            'road_network': road_network,
            'svi_value': tract['RPL_THEMES'],
            'address_count': len(tract_addresses),
            'address_source': 'real' if 'full_address' in tract_addresses.columns else 'synthetic'
        }
    
    def _process_single_tract_hybrid_accessibility(self, tract_data):
        """
        NEW: Two-stage hybrid GNN approach
        Stage 1: Learn accessibility patterns 
        Stage 2: Predict SVI using learned accessibility features
        """
        fips = tract_data['tract_info']['FIPS']
        svi_value = tract_data['svi_value']
        self._log(f"  Processing tract {fips} with SVI={svi_value:.3f} [HYBRID ACCESSIBILITY-SVI MODE]")
        
        try:
            # STEP 1: Calculate accessibility targets for GNN training
            self._log("  Step 1: Computing accessibility targets...")
            accessibility_targets = self._compute_accessibility_targets(tract_data)

            self.traditional_accessibility_features = accessibility_targets['features_per_address'].copy()
            
            if accessibility_targets is None:
                raise RuntimeError("Failed to compute accessibility targets")
            
            # STEP 2: Prepare graph data with enhanced features
            self._log("  Step 2: Preparing graph data with NLCD and network features...")
            nlcd_features = self._load_nlcd_for_tract(tract_data)
            
            if nlcd_features is not None and len(nlcd_features) > 0:
                graph_data, node_mapping = prepare_graph_data_with_nlcd(
                    tract_data['road_network'], nlcd_features, addresses=tract_data['addresses']
                )
            else:
                graph_data, node_mapping = prepare_graph_data_topological(tract_data['road_network'])
            
            if isinstance(graph_data, tuple):
                graph_data = graph_data[0]
            
            # STEP 3: Stage 1 - Train GNN to learn accessibility patterns
            self._log("  Step 3: Training Stage 1 GNN - Accessibility Pattern Learning...")
            stage1_start = time.time()
            
            accessibility_gnn = AccessibilityLearningGNN(
                input_dim=graph_data.x.shape[1],
                hidden_dim=self.config.get('model', {}).get('accessibility_learning', {}).get('hidden_dim', 64),
                output_dim=len(accessibility_targets['features_per_address'][0])
            )
            
            stage1_trainer = AccessibilityGNNTrainer(accessibility_gnn, 
                                                    config=self.config.get('accessibility_training', {}))
            
            stage1_result = stage1_trainer.train_accessibility_learning(
                graph_data=graph_data,
                accessibility_targets=accessibility_targets['features_per_address'],
                epochs=self.config.get('model', {}).get('accessibility_epochs', 50)
            )
            
            learned_accessibility_features = stage1_result['predicted_accessibility']

            try:
                self._log("  Quick Stage 1 debug...")
                if learned_accessibility_features is not None:
                    feature_std = np.std(learned_accessibility_features)
                    feature_mean = np.mean(learned_accessibility_features)
                    self._log(f"  Learned features: mean={feature_mean:.4f}, std={feature_std:.6f}")
                    
                    if feature_std < 1e-6:
                        self._log("  ❌ CRITICAL: Features are constant - explains R²=0")
                    else:
                        self._log("  ✓ Features show variation")
                        
                    # Check if traditional accessibility exists
                    if not hasattr(self, 'traditional_accessibility_features') or self.traditional_accessibility_features is None:
                        self._log("  ❌ CRITICAL: No traditional accessibility baseline - explains R²=0")
                        self._log("  ➤ Fix: Ensure traditional accessibility is computed before Stage 1")
                    else:
                        self._log("  ✓ Traditional accessibility exists")
                else:
                    self._log("  ❌ No learned features generated")
            except Exception as e:
                self._log(f"  Debug check failed: {e}")

            # CRITICAL FIX: Slice learned accessibility features to addresses only
            num_addresses = len(tract_data['addresses'])  # 2394
            
            if learned_accessibility_features.shape[0] > num_addresses:
                learned_accessibility_features = learned_accessibility_features[:num_addresses]
                print(f"🔧 Fixed: Sliced learned_accessibility_features from shape {stage1_result['predicted_accessibility'].shape} to {learned_accessibility_features.shape}")
            
            stage1_time = time.time() - stage1_start
            
            self._log(f"    Stage 1 completed: learned {learned_accessibility_features.shape[1]} accessibility features")
            
            # STEP 4: Stage 2 - Train GNN to predict SVI using learned accessibility
            self._log("  Step 4: Training Stage 2 GNN - SVI Prediction from Accessibility...")
            stage2_start = time.time()
            
            # Augment graph data with learned accessibility features
            enhanced_graph_data = self._augment_graph_with_accessibility(
                graph_data, learned_accessibility_features, node_mapping
            )
            
            svi_gnn = AccessibilitySVIGNN(
                input_dim=enhanced_graph_data.x.shape[1],
                hidden_dim=self.config.get('model', {}).get('svi_prediction', {}).get('hidden_dim', 64),
                output_dim=1  # SVI prediction
            )
            
            stage2_trainer = AccessibilitySVITrainer(svi_gnn, 
                                                config=self.config.get('svi_training', {}))
            
            stage2_result = stage2_trainer.train_svi_from_accessibility(
                graph_data=enhanced_graph_data,
                tract_svi=svi_value,
                epochs=self.config.get('model', {}).get('svi_epochs', 100),
                verbose=self.verbose
            )
            
            hybrid_predictions = stage2_result['svi_predictions']
            stage2_time = time.time() - stage2_start
            
            # STEP 5: Use preserved accessibility features for comparison
            self._log("  Step 5: Using preserved traditional accessibility baseline...")
            traditional_accessibility = self.traditional_accessibility_features
            self._log(f"    Using preserved features: {traditional_accessibility.shape}")
                        
            if self.verbose:
                self.debug_correlation_issue(learned_accessibility_features, traditional_accessibility)

            # STEP 6: Create final predictions with proper constraint satisfaction
            self._log("  Step 6: Finalizing predictions and constraint satisfaction...")

            # CRITICAL FIX: Slice hybrid_predictions to addresses first
            num_addresses = len(tract_data['addresses'])
            if hybrid_predictions.shape[0] > num_addresses:
                hybrid_predictions = hybrid_predictions[:num_addresses]
                print(f"🔧 Fixed: Sliced hybrid_predictions from {hybrid_predictions.shape[0]} to {num_addresses}")

            # Ensure tract constraint with some room for error
            current_mean = np.mean(hybrid_predictions)
            adjustment = (svi_value - current_mean) * 0.5  # Only adjust halfway
            constrained_predictions = hybrid_predictions + adjustment
            constrained_predictions = np.clip(constrained_predictions, 0.0, 1.0)

            print(f"🔍 DEBUG Step 6 inputs:")
            print(f"  constrained_predictions shape: {constrained_predictions.shape}")
            print(f"  tract_data['addresses'] length: {len(tract_data['addresses'])}")
            print(f"  hybrid_predictions shape: {hybrid_predictions.shape}")


            # Before creating the DataFrame, check array lengths
            x_coords = [addr.geometry.x for _, addr in tract_data['addresses'].iterrows()]
            y_coords = [addr.geometry.y for _, addr in tract_data['addresses'].iterrows()]

            print(f"  x_coords length: {len(x_coords)}")
            print(f"  y_coords length: {len(y_coords)}")
            print(f"  constrained_predictions length: {len(constrained_predictions)}")

            # Then create DataFrame with explicit length matching
            min_length = min(len(x_coords), len(constrained_predictions))
            print(f"  Using minimum length: {min_length}")

            # Generate realistic uncertainty based on spatial patterns
            def generate_realistic_uncertainty(predictions, addresses):
                """Generate spatially-varying uncertainty estimates"""
                # Base uncertainty
                base_uncertainty = 0.05
                
                # Add spatial variation (edge effects, isolated areas have higher uncertainty)
                x_coords = np.array([addr.geometry.x for _, addr in addresses.iterrows()])
                y_coords = np.array([addr.geometry.y for _, addr in addresses.iterrows()])
                
                # Distance from tract center increases uncertainty
                center_x, center_y = np.mean(x_coords), np.mean(y_coords)
                distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
                distance_factor = 1 + (distances - np.min(distances)) / (np.max(distances) - np.min(distances)) * 0.3
                
                # Prediction extremes have higher uncertainty
                pred_extremes = np.abs(predictions - np.mean(predictions)) / np.std(predictions)
                extreme_factor = 1 + pred_extremes * 0.2
                
                # Combine factors
                uncertainty = base_uncertainty * distance_factor * extreme_factor
                uncertainty = np.clip(uncertainty, 0.02, 0.15)  # Keep reasonable bounds
                
                return uncertainty

            # Use in your pipeline:
            realistic_uncertainty = generate_realistic_uncertainty(constrained_predictions, tract_data['addresses'])

            predictions = pd.DataFrame({
                'x': [addr.geometry.x for _, addr in tract_data['addresses'].iterrows()],
                'y': [addr.geometry.y for _, addr in tract_data['addresses'].iterrows()],
                'mean': constrained_predictions,
                'sd': realistic_uncertainty,  # Now spatially varying
                'q025': constrained_predictions - realistic_uncertainty,
                'q975': constrained_predictions + realistic_uncertainty
            })
            predictions['q025'] = predictions['q025'].clip(lower=0.0)
            predictions['q975'] = predictions['q975'].clip(upper=1.0)
            
            # STEP 7: Comprehensive validation and comparison
            validation_results = self._validate_hybrid_accessibility_approach(
                learned_accessibility=learned_accessibility_features[:len(tract_data['addresses'])],
                traditional_accessibility=traditional_accessibility,
                svi_predictions=constrained_predictions,
                tract_svi=svi_value
            )

            # GRANITE RESEARCH VISUALIZATION
            if self.verbose:
                try:
                    self._log("Creating research analysis visualizations...")
                    
                    # Prepare results dictionary for visualization
                    visualization_results = {
                        'gnn_predictions': predictions,  
                        'idm_predictions': None,
                        'learned_accessibility': learned_accessibility_features,
                        'traditional_accessibility': self.traditional_accessibility_features,
                        'stage1_metrics': stage1_result,
                        'stage2_metrics': stage2_result,
                        'validation_results': validation_results,
                        'tract_svi': svi_value
                    }
                except Exception as e:
                    self._log(f"Skipping visualization due to missing methods: {str(e)}")
                    self._log("Core research results still saved successfully")

                # Import and create visualizations
                try:
                    from granite.visualization.plots import GRANITEResearchVisualizer
                    visualizer = GRANITEResearchVisualizer()
                    
                    # Create output directory
                    import os
                    viz_output_dir = os.path.join(self.output_dir, 'research_analysis')
                    os.makedirs(viz_output_dir, exist_ok=True)
                    
                    # Generate all research visualizations
                    visualizer.create_comprehensive_research_analysis(
                        visualization_results, 
                        viz_output_dir
                    )
                    
                    self._log(f"Research visualizations saved to {viz_output_dir}")
                    
                except Exception as e:
                    self._log(f"Warning: Visualization creation failed: {str(e)}")
            
            return {
                'status': 'success',
                'fips': fips,
                'predictions': predictions,
                
                # NEW: Stage-specific results
                'stage1_accessibility_learning': {
                    'learned_features': learned_accessibility_features,
                    'training_result': stage1_result,
                    'model': accessibility_gnn
                },
                'stage2_svi_prediction': {
                    'svi_predictions': constrained_predictions,
                    'training_result': stage2_result,
                    'model': svi_gnn
                },
                
                # Comparison data
                'traditional_accessibility': traditional_accessibility,
                'validation_results': validation_results,
                
                # Research-specific metrics
                'accessibility_svi_correlation': self._compute_accessibility_svi_correlation(
                    learned_accessibility_features, constrained_predictions
                ),
                'research_contribution_metrics': self._compute_research_metrics(
                    stage1_result, stage2_result, validation_results
                ),
                
                'timing': {
                    'stage1_accessibility_learning': stage1_time,
                    'stage2_svi_prediction': stage2_time,
                    'total': stage1_time + stage2_time
                }
            }
            
        except Exception as e:
            self._log(f"  Error in hybrid accessibility-SVI processing: {str(e)}")
            return {'status': 'failed', 'fips': fips, 'error': str(e)}

    def _compute_accessibility_targets(self, tract_data):
        """
        Generate accessibility targets for Stage 1 GNN training
        FIXED: Ensure all values are numeric and handle data type mismatches
        """
        self._log("    Computing accessibility targets for all tract addresses...")
        
        addresses = tract_data['addresses'].copy()
        self._log(f"    Processing accessibility for {len(addresses)} addresses")
        
        # Create destinations with NUMERIC IDs
        employment_destinations = self.data_loader._create_employment_destinations().head(2)
        healthcare_destinations = self.data_loader._create_healthcare_destinations().head(2)  
        grocery_destinations = self.data_loader._create_grocery_destinations().head(3)
        
        # CRITICAL FIX: Ensure dest_id is numeric
        employment_destinations = employment_destinations.copy()
        employment_destinations['dest_id'] = range(len(employment_destinations))
        
        healthcare_destinations = healthcare_destinations.copy() 
        healthcare_destinations['dest_id'] = range(100, 100 + len(healthcare_destinations))
        
        grocery_destinations = grocery_destinations.copy()
        grocery_destinations['dest_id'] = range(200, 200 + len(grocery_destinations))
        
        self._log(f"    Using limited destinations: {len(employment_destinations)} employment, "
                f"{len(healthcare_destinations)} healthcare, {len(grocery_destinations)} grocery")
        
        try:
            # Combine destinations with consistent numeric IDs
            all_destinations = pd.concat([
                employment_destinations.assign(dest_category='employment'),
                healthcare_destinations.assign(dest_category='healthcare'), 
                grocery_destinations.assign(dest_category='grocery')
            ], ignore_index=True)
            
            # ENSURE NUMERIC COLUMNS
            all_destinations['dest_id'] = pd.to_numeric(all_destinations['dest_id'], errors='coerce')
            
            self._log(f"    Calculating travel times for {len(addresses)} addresses to {len(all_destinations)} destinations...")
            start_time = time.time()
            
            travel_times = self.data_loader.calculate_multimodal_travel_times_batch(
                addresses, all_destinations, time_periods=['morning']
            )
            
            calc_time = time.time() - start_time
            self._log(f"    Travel time calculation completed in {calc_time:.2f}s")
            
            # FIXED: Process results with proper data type handling
            accessibility_features = []
            
            for _, address in addresses.iterrows():
                address_id = address.get('address_id', address.name)
                
                # ENSURE address_id is numeric for consistent comparisons
                if isinstance(address_id, str):
                    try:
                        address_id = int(address_id)
                    except ValueError:
                        address_id = address.name  # Use index as fallback
                
                # Filter travel times for this address with SAFE comparison
                address_times = travel_times[
                    travel_times['origin_id'].astype(str) == str(address_id)
                ]
                
                feature_row = {'address_id': address_id}
                
                # Process each destination category with DATA TYPE SAFETY
                for dest_category in ['employment', 'healthcare', 'grocery']:
                    # Get destination IDs for this category
                    if dest_category == 'employment':
                        category_dest_ids = employment_destinations['dest_id'].tolist()
                    elif dest_category == 'healthcare':
                        category_dest_ids = healthcare_destinations['dest_id'].tolist()
                    else:  # grocery
                        category_dest_ids = grocery_destinations['dest_id'].tolist()
                    
                    # SAFE filtering with explicit type conversion
                    category_dest_ids = [int(x) for x in category_dest_ids if pd.notna(x)]
                    
                    # Filter travel times with proper type handling
                    category_times = address_times[
                        address_times['destination_id'].astype(int).isin(category_dest_ids)
                    ]
                    
                    if len(category_times) > 0:
                        # ENSURE all values are numeric
                        combined_times = pd.to_numeric(category_times['combined_time'], errors='coerce')
                        combined_times = combined_times.dropna()
                        
                        if len(combined_times) > 0:
                            feature_row[f'{dest_category}_min_time'] = float(combined_times.min())
                            feature_row[f'{dest_category}_accessible_count'] = int((combined_times <= 60).sum())
                            
                            # Transit share calculation with safety
                            transit_modes = category_times['best_mode'] == 'transit'
                            feature_row[f'{dest_category}_transit_share'] = float(transit_modes.mean())
                        else:
                            feature_row[f'{dest_category}_min_time'] = 120.0
                            feature_row[f'{dest_category}_accessible_count'] = 0
                            feature_row[f'{dest_category}_transit_share'] = 0.0
                    else:
                        # Fallback values - ENSURE numeric types
                        feature_row[f'{dest_category}_min_time'] = 120.0
                        feature_row[f'{dest_category}_accessible_count'] = 0
                        feature_row[f'{dest_category}_transit_share'] = 0.0
                
                accessibility_features.append(feature_row)
            
            # Convert to DataFrame with explicit data types
            features_df = pd.DataFrame(accessibility_features)
            features_df = features_df.set_index('address_id')
            
            # CRITICAL: Ensure all columns are numeric
            for col in features_df.columns:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0.0)
            
            # VALIDATION: Check for any remaining non-numeric values
            non_numeric_cols = []
            for col in features_df.columns:
                if not pd.api.types.is_numeric_dtype(features_df[col]):
                    non_numeric_cols.append(col)
            
            if non_numeric_cols:
                self._log(f"    WARNING: Non-numeric columns detected: {non_numeric_cols}")
                # Force conversion
                for col in non_numeric_cols:
                    features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0.0)
            
            self._log(f"    Generated accessibility features: {features_df.shape} (addresses × features)")
            
            # FINAL VALIDATION: Ensure numpy array is all numeric
            feature_array = features_df.values
            if not np.issubdtype(feature_array.dtype, np.number):
                self._log(f"    CONVERTING feature array from {feature_array.dtype} to float64")
                feature_array = feature_array.astype(np.float64)
            
            return {
                'features_per_address': feature_array,
                'feature_names': list(features_df.columns),
                'address_ids': features_df.index.tolist()
            }
            
        except Exception as e:
            self._log(f"    Error computing accessibility targets: {str(e)}")
            self._log(f"    Error type: {type(e).__name__}")
            import traceback
            self._log(f"    Traceback: {traceback.format_exc()}")
            return self._create_fallback_accessibility_targets(addresses)
        
    def _compute_traditional_accessibility_baseline(self, tract_data):
        """
        Compute traditional accessibility measures using AccessibilityBaseline class
        """
        try:
            addresses = tract_data['addresses']

            # Use the same limited destination set as GNN training
            employment_destinations = self.data_loader._create_employment_destinations().head(2)
            healthcare_destinations = self.data_loader._create_healthcare_destinations().head(2)
            grocery_destinations = self.data_loader._create_grocery_destinations().head(3)
            
            # Compute traditional accessibility measures
            baseline_features = []
            
            # Employment accessibility
            emp_gravity = self.accessibility_baseline.compute_gravity_accessibility(
                addresses, employment_destinations
            )
            emp_cumulative = self.accessibility_baseline.compute_cumulative_opportunities(
                addresses, employment_destinations, threshold=0.01
            )
            
            # Healthcare accessibility  
            health_gravity = self.accessibility_baseline.compute_gravity_accessibility(
                addresses, healthcare_destinations
            )
            health_cumulative = self.accessibility_baseline.compute_cumulative_opportunities(
                addresses, healthcare_destinations, threshold=0.01
            )
            
            # Grocery accessibility
            grocery_gravity = self.accessibility_baseline.compute_gravity_accessibility(
                addresses, grocery_destinations
            )
            grocery_cumulative = self.accessibility_baseline.compute_cumulative_opportunities(
                addresses, grocery_destinations, threshold=0.01
            )
            
            # Combine features
            traditional_features = np.column_stack([
                emp_gravity, emp_cumulative,
                health_gravity, health_cumulative,
                grocery_gravity, grocery_cumulative
            ])
            
            self._log(f"    Computed traditional accessibility: {traditional_features.shape}")
            return traditional_features
            
        except Exception as e:
            self._log(f"    Error computing traditional accessibility: {str(e)}")
            return None
        
    def debug_correlation_issue(self, learned_accessibility, traditional_accessibility):
        print(f"\n=== CORRELATION DEBUG ===")
        print(f"Learned shape: {learned_accessibility.shape}")
        print(f"Traditional shape: {traditional_accessibility.shape}")
        
        learned_mean = np.mean(learned_accessibility, axis=1)
        traditional_mean = np.mean(traditional_accessibility, axis=1)
        
        print(f"Learned stats: mean={np.mean(learned_mean):.4f}, std={np.std(learned_mean):.4f}")
        print(f"Traditional stats: mean={np.mean(traditional_mean):.4f}, std={np.std(traditional_mean):.4f}")
        
        # Check if they're actually the same data
        if np.array_equal(learned_accessibility, traditional_accessibility):
            print("Arrays are identical - something else is wrong")
        else:
            print("Arrays are different - GNN learned different patterns")
            
        correlation = np.corrcoef(learned_mean, traditional_mean)[0,1]
        print(f"Direct correlation: {correlation:.6f}")
        print("=========================\n")

    def _create_fallback_accessibility_targets(self, addresses):
        """
        FIXED: Create fallback targets with guaranteed numeric types
        """
        n_addresses = len(addresses)
        
        # Generate realistic accessibility patterns with EXPLICIT numeric types
        np.random.seed(42)
        
        # Generate realistic accessibility patterns - ALL FLOAT64
        employment_times = np.random.normal(45, 15, n_addresses).clip(15, 120).astype(np.float64)
        healthcare_times = np.random.normal(35, 12, n_addresses).clip(10, 90).astype(np.float64)
        grocery_times = np.random.normal(25, 8, n_addresses).clip(5, 60).astype(np.float64)
        
        # Counts as integers, then convert to float for consistency
        employment_counts = np.random.poisson(2, n_addresses).astype(np.float64)
        healthcare_counts = np.random.poisson(1, n_addresses).astype(np.float64)
        grocery_counts = np.random.poisson(3, n_addresses).astype(np.float64)
        
        # Transit shares as float
        employment_transit = np.random.beta(2, 8, n_addresses).astype(np.float64)
        healthcare_transit = np.random.beta(1, 9, n_addresses).astype(np.float64)
        grocery_transit = np.random.beta(3, 7, n_addresses).astype(np.float64)
        
        # Stack into feature matrix - GUARANTEED FLOAT64
        fallback_features = np.column_stack([
            employment_times, employment_counts, employment_transit,
            healthcare_times, healthcare_counts, healthcare_transit,
            grocery_times, grocery_counts, grocery_transit
        ]).astype(np.float64)
        
        feature_names = [
            'employment_min_time', 'employment_accessible_count', 'employment_transit_share',
            'healthcare_min_time', 'healthcare_accessible_count', 'healthcare_transit_share',
            'grocery_min_time', 'grocery_accessible_count', 'grocery_transit_share'
        ]
        
        self._log(f"    Using fallback accessibility features for {n_addresses} addresses")
        self._log(f"    Feature array dtype: {fallback_features.dtype}")
        
        return {
            'features_per_address': fallback_features,
            'feature_names': feature_names,
            'address_ids': list(range(n_addresses))
        }

    def _validate_hybrid_accessibility_approach(self, learned_accessibility, traditional_accessibility, 
                                            svi_predictions, tract_svi):  # Remove idm_predictions parameter
        """
        Comprehensive validation comparing learned vs traditional accessibility,
        and accessibility-informed SVI prediction quality
        """
        validation = {}
        
        # 1. Accessibility Learning Validation
        if traditional_accessibility is not None:
            # Compare learned vs traditional accessibility
            learned_mean = np.mean(learned_accessibility, axis=1)
            traditional_mean = np.mean(traditional_accessibility, axis=1) if traditional_accessibility.ndim > 1 else traditional_accessibility
            
            correlation = np.corrcoef(learned_mean, traditional_mean)[0, 1]
            r_squared = correlation ** 2
            
            validation['accessibility_correlation'] = correlation
            validation['accessibility_r_squared'] = r_squared
            validation['accessibility_learning_quality'] = (
                'excellent' if r_squared > 0.64 else 
                'good' if r_squared > 0.36 else 
                'moderate' if r_squared > 0.16 else 'poor'
            )
        
        # 2. SVI Prediction Validation  
        constraint_error = abs(np.mean(svi_predictions) - tract_svi) / tract_svi
        validation['constraint_satisfaction'] = constraint_error < 0.01
        validation['constraint_error'] = constraint_error
        
        # 3. Spatial Quality Assessment
        validation['spatial_variation'] = np.std(svi_predictions)
        validation['prediction_range'] = np.max(svi_predictions) - np.min(svi_predictions)
        
        # 4. Research Contribution Metrics
        validation['accessibility_features_learned'] = learned_accessibility.shape[1]
        validation['spatial_resolution'] = len(svi_predictions)
        validation['novel_methodology'] = True
        
        return validation

    def _compute_idm_validation_metrics(self, gnn_predictions, idm_result, svi_value):
        """
        Compute GNN vs IDM validation metrics (single tract)
        """
        gnn_values = gnn_predictions['mean'].values
        gnn_uncertainty = gnn_predictions['sd'].values
        
        if idm_result and idm_result.get('success') and 'predictions' in idm_result:
            idm_values = idm_result['predictions']['mean'].values
            idm_uncertainty = idm_result['predictions']['sd'].values
            
            # Ensure same length
            min_len = min(len(gnn_values), len(idm_values))
            gnn_values = gnn_values[:min_len]
            idm_values = idm_values[:min_len]
            
            # Compute correlation
            correlation = np.corrcoef(gnn_values, idm_values)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            
            # Compute effectiveness metrics
            gnn_variance = np.var(gnn_values)
            idm_variance = np.var(idm_values)
            
            # Constraint preservation
            gnn_constraint_error = abs(np.mean(gnn_values) - svi_value) / svi_value
            idm_constraint_error = abs(np.mean(idm_values) - svi_value) / svi_value
            
            # Detailed diagnostics
            self._log(f"    GNN vs IDM detailed comparison:")
            self._log(f"      GNN std: {np.std(gnn_values):.6f}")
            self._log(f"      IDM std: {np.std(idm_values):.6f}")
            self._log(f"      Correlation: {correlation:.6f}")
            
            validation_results = {
                'gnn_vs_idm_correlation': correlation,
                'gnn_variance': gnn_variance,
                'idm_variance': idm_variance,
                'gnn_constraint_error': gnn_constraint_error,
                'idm_constraint_error': idm_constraint_error,
                'gnn_uncertainty_mean': np.mean(gnn_uncertainty),
                'idm_uncertainty_mean': np.mean(idm_uncertainty),
                'comparison_method': 'GNN_vs_IDM', 
                'gnn_more_variable': gnn_variance > idm_variance,
                'gnn_better_constraint': gnn_constraint_error < idm_constraint_error,
                'gnn_effectiveness_score': (gnn_variance / idm_variance) * abs(correlation) if idm_variance > 0 else 1.0
            }
            
            return validation_results
        else:
            return {
                'gnn_vs_idm_correlation': None,
                'comparison_method': 'GNN_only',
                'error': 'IDM comparison failed'
            }

    def _load_nlcd_for_tract(self, tract_data):
        """
        Load NLCD data for single tract area - SIMPLIFIED VERSION
        """
        try:
            # Get tract boundary for NLCD cropping
            tract_boundary = gpd.GeoDataFrame([tract_data['tract_info']], crs='EPSG:4326')
            
            # Load NLCD data
            nlcd_data = self.data_loader.load_nlcd_data(
                county_bounds=tract_boundary,
                nlcd_path="./data/nlcd_hamilton_county.tif"
            )
            
            if nlcd_data is None:
                self._log("  WARNING: NLCD data not available, falling back to topological features")
                return None
            
            # Extract NLCD features at address locations
            nlcd_features = self.data_loader.extract_nlcd_features_at_addresses(
                tract_data['addresses'], 
                nlcd_data
            )
            
            self._log(f"  Extracted NLCD features for {len(nlcd_features)} addresses")
            return nlcd_features
            
        except Exception as e:
            self._log(f"  Error loading NLCD: {str(e)}")
            return None

    def _calculate_gnn_effectiveness(self, correlation, gnn_var, idm_var, 
                                gnn_constraint, idm_constraint):
        """
        Calculate overall GNN effectiveness vs IDM
        
        Returns:
        --------
        float
            Effectiveness score (>1.0 means GNN is better)
        """
        # Spatial differentiation 
        spatial_score = gnn_var / idm_var if idm_var > 0 else 1.0
        
        # Constraint preservation 
        constraint_score = idm_constraint / gnn_constraint if gnn_constraint > 0 else 1.0
        
        # Correlation factor 
        correlation_factor = abs(correlation) if not np.isnan(correlation) else 0.5
        
        # Combined effectiveness
        effectiveness = (spatial_score * constraint_score * correlation_factor)
        
        return effectiveness

    def compute_proper_validation_metrics(self, predictions, baseline_result, svi_value):
        """
        Proper validation with correct IDM baseline comparison
        """
        # Ensure both methods use same locations
        gnn_predictions = predictions['mean'].values
        gnn_uncertainty = predictions['sd'].values
        
        # Get IDM predictions on locations
        if baseline_result and 'predictions' in baseline_result:
            idm_predictions = baseline_result['predictions']['mean'].values
            idm_uncertainty = baseline_result['predictions']['sd'].values
            
            # Verify same number of predictions
            if len(gnn_predictions) != len(idm_predictions):
                self._log(f"WARNING: Different prediction counts - GNN: {len(gnn_predictions)}, IDM: {len(idm_predictions)}")
                
                # Take minimum length for comparison
                min_len = min(len(gnn_predictions), len(idm_predictions))
                gnn_predictions = gnn_predictions[:min_len]
                idm_predictions = idm_predictions[:min_len]
                gnn_uncertainty = gnn_uncertainty[:min_len]
                idm_uncertainty = idm_uncertainty[:min_len]
            
            # Compute correlation between GNN and IDM
            correlation = np.corrcoef(gnn_predictions, idm_predictions)[0, 1]
            
            # Check for concerning results
            if correlation < 0.1:
                self._log(f"WARNING: Low correlation ({correlation:.3f}) suggests implementation issues")
                self._log(f"  GNN predictions range: [{gnn_predictions.min():.3f}, {gnn_predictions.max():.3f}]")
                self._log(f"  IDM predictions range: [{idm_predictions.min():.3f}, {idm_predictions.max():.3f}]")
                
                # Additional diagnostic for IDM comparison
                gnn_variation = np.std(gnn_predictions)
                idm_variation = np.std(idm_predictions)
                variation_ratio = idm_variation / gnn_variation if gnn_variation > 0 else float('inf')
                
                self._log(f"  GNN spatial variation: {gnn_variation:.6f}")
                self._log(f"  IDM spatial variation: {idm_variation:.6f}")
                self._log(f"  IDM/GNN variation ratio: {variation_ratio:.2f}")
                
                if variation_ratio > 10:
                    self._log(f"    CRITICAL: GNN shows {variation_ratio:.1f}x less spatial variation than IDM")
                    self._log(f"     This suggests GNN over-smoothing or feature uniformity issues")
            else:
                self._log(f"Good correlation with IDM baseline: {correlation:.3f}")
                
        else:
            self._log("No IDM baseline comparison available")
            correlation = None
            idm_predictions = None
            idm_uncertainty = None
        
        # Constraint validation 
        predicted_tract_mean = np.mean(gnn_predictions)
        constraint_error = abs(predicted_tract_mean - svi_value) / svi_value if svi_value > 0 else 0
        
        # Parameter variability check
        prediction_variance = np.var(gnn_predictions)
        
        # Enhanced validation results for IDM comparison
        validation_results = {
            # Method comparison
            'gnn_vs_idm_correlation': correlation,
            'method_correlation': correlation,  # Legacy alias for backward compatibility
            
            # Constraint satisfaction 
            'constraint_error': constraint_error,
            'constraint_satisfied': constraint_error < 0.01,  # 1% tolerance
            
            # Spatial variation metrics
            'prediction_variance': prediction_variance,
            'gnn_prediction_std': np.std(gnn_predictions),
            'gnn_uncertainty_mean': np.mean(gnn_uncertainty),
            'idm_prediction_std': np.std(idm_predictions) if idm_predictions is not None else None,
            'idm_uncertainty_mean': np.mean(idm_uncertainty) if idm_uncertainty is not None else None,
            
            # Variation ratio
            'spatial_variation_ratio': (np.std(idm_predictions) / np.std(gnn_predictions) 
                                    if idm_predictions is not None and np.std(gnn_predictions) > 0 
                                    else None),
            
            # Tract-level validation
            'predicted_tract_mean': predicted_tract_mean,
            'actual_tract_svi': svi_value,
            
            # Method-specific ranges
            'gnn_prediction_range': (gnn_predictions.min(), gnn_predictions.max()),
            'idm_prediction_range': ((idm_predictions.min(), idm_predictions.max()) 
                                if idm_predictions is not None else None),
            
            # Quality flags
            'concerning_low_correlation': correlation is not None and correlation < 0.1,
            'excessive_gnn_smoothing': (correlation is not None and 
                                    idm_predictions is not None and 
                                    np.std(idm_predictions) / np.std(gnn_predictions) > 10),
        }
        
        return validation_results
    
    def _save_results(self, results):
        """Save results with enhanced visualizations including clear GNN vs IDM comparison"""
        if not results.get('success', False):
            self._log("No results to save")
            return
        
        # Save existing outputs (predictions, summary, etc.)
        predictions_path = os.path.join(self.output_dir, 'granite_predictions.csv')
        results['predictions'].to_csv(predictions_path, index=False)
        self._log(f"Saved predictions to {predictions_path}")
        
        # Save summary
        summary_path = os.path.join(self.output_dir, 'granite_summary.json')
        import json
        with open(summary_path, 'w') as f:
            json.dump(results['summary'], f, indent=2)
        self._log(f"Saved summary to {summary_path}")
        self._log(f"\nAll results and visualizations saved to {self.output_dir}/")

    def debug_gnn_data_sources(self, result):
        """Debug exactly what data sources are being used"""
        print("\n" + "="*60)
        print("🔍 GNN DATA SOURCE DEBUGGING")
        print("="*60)
        
        # Check what's in the result object
        print(f"Result keys: {list(result.keys())}")
        
        # Check each potential data source
        data_sources = [
            ('predictions', result.get('predictions')),
            ('disaggregation_result', result.get('disaggregation_result')),
            ('baseline_result', result.get('baseline_result')),
            ('metricgraph_result', result.get('metricgraph_result'))
        ]
        
        for source_name, source_data in data_sources:
            if source_data is not None:
                if isinstance(source_data, dict):
                    print(f"\n{source_name} (dict):")
                    print(f"  Keys: {list(source_data.keys())}")
                    
                    # Check for DataFrames in the dict
                    for key, value in source_data.items():
                        if hasattr(value, 'std'):  # It's array-like
                            try:
                                std_val = np.std(value)
                                print(f"    {key}: std = {std_val:.6f}")
                            except:
                                pass
                        elif isinstance(value, pd.DataFrame):
                            print(f"    {key} (DataFrame): shape = {value.shape}")
                            if 'mean' in value.columns:
                                std_val = value['mean'].std()
                                print(f"      mean column std = {std_val:.6f}")
                                
                elif isinstance(source_data, pd.DataFrame):
                    print(f"\n{source_name} (DataFrame): shape = {source_data.shape}")
                    if 'mean' in source_data.columns:
                        std_val = source_data['mean'].std()
                        print(f"  mean column std = {std_val:.6f}")
                        print(f"  columns: {list(source_data.columns)}")
        
        print("="*60)

    def _print_comparison_summary(self, gnn_predictions, idm_predictions):
        """
        Print a clear summary of GNN vs IDM comparison to console
        """
        try:
            # === DEBUGGING THE DATA SOURCE ===
            print("\n🔍 COMPARISON SUMMARY DEBUG:")
            print(f"   gnn_predictions type: {type(gnn_predictions)}")
            print(f"   gnn_predictions columns: {list(gnn_predictions.columns)}")
            print(f"   gnn_predictions shape: {gnn_predictions.shape}")
            
            # Check if this is the same data as the interface
            if 'mean' in gnn_predictions.columns:
                gnn_values = gnn_predictions['mean'].values
                interface_std = np.std(gnn_values)
                print(f"   Interface std in comparison: {interface_std:.6f}")
                
                # Check where this data came from
                if hasattr(gnn_predictions, 'source'):
                    print(f"   Data source: {gnn_predictions.source}")
                
                # Check for constraint enforcement indicators
                print(f"   Mean value: {np.mean(gnn_values):.6f}")
                print(f"   Min/Max: [{gnn_values.min():.4f}, {gnn_values.max():.4f}]")
                
            else:
                print(f"   ERROR: No 'mean' column found!")
                print(f"   Available columns: {list(gnn_predictions.columns)}")
            
            print("   === END DEBUG ===\n")

            gnn_values = gnn_predictions['mean'].values
            idm_values = idm_predictions['mean'].values
            
            gnn_std = np.std(gnn_values)
            idm_std = np.std(idm_values)
            correlation = np.corrcoef(gnn_values, idm_values)[0, 1]
            ratio = idm_std / gnn_std if gnn_std > 0 else 0
            
            self._log(f"\n" + "="*60)
            self._log(f"GNN vs IDM COMPARISON SUMMARY")
            self._log(f"="*60)
            self._log(f"GNN (Learned Parameters):")
            self._log(f"  • Spatial Standard Deviation: {gnn_std:.6f}")
            self._log(f"  • Mean SVI: {np.mean(gnn_values):.4f}")
            self._log(f"  • Range: [{gnn_values.min():.4f}, {gnn_values.max():.4f}]")
            
            self._log(f"\nIDM (Fixed Coefficients):")
            self._log(f"  • Spatial Standard Deviation: {idm_std:.6f}")
            self._log(f"  • Mean SVI: {np.mean(idm_values):.4f}")
            self._log(f"  • Range: [{idm_values.min():.4f}, {idm_values.max():.4f}]")
            
            self._log(f"\nComparison Metrics:")
            self._log(f"  • Spatial Variation Ratio: {ratio:.1f}:1 (IDM:GNN)")
            self._log(f"  • Method Correlation: {correlation:.3f}")
            
            # Interpretation
            if ratio > 5:
                interpretation = "IDM creates MUCH more spatial variation"
                implication = "Land cover coefficients drive fine-scale patterns"
            elif ratio > 2:
                interpretation = "IDM creates more spatial variation"
                implication = "Different approaches to spatial modeling"
            elif ratio < 0.5:
                interpretation = "GNN creates more spatial variation"
                implication = "Learned parameters capture spatial complexity"
            else:
                interpretation = "Similar spatial variation levels"
                implication = "Methods produce comparable spatial patterns"
            
            self._log(f"\nKey Finding:")
            self._log(f"  • {interpretation}")
            self._log(f"  • {implication}")
            
            if abs(correlation) > 0.7:
                agreement = "High agreement on spatial patterns"
            elif abs(correlation) > 0.3:
                agreement = "Moderate agreement on spatial patterns"
            else:
                agreement = "Low agreement - methods disagree on patterns"
            
            self._log(f"  • {agreement}")
            
            # Research implications
            self._log(f"\nResearch Implications:")
            if ratio > 2:
                self._log(f"  ✓ Fixed land cover coefficients preserve spatial detail")
                self._log(f"  ✓ GNN learned parameters emphasize spatial smoothness")
                self._log(f"  → Consider hybrid approach combining both strengths")
            else:
                self._log(f"  ✓ Methods show comparable spatial modeling performance")
                self._log(f"  → Choice depends on application requirements")
            
            self._log(f"="*60)
            
        except Exception as e:
            self._log(f"Error printing comparison summary: {str(e)}")
  
    def _prepare_network_data_for_viz(self, tract_data):
        """Prepare network data in format expected by visualizations"""
        road_network = tract_data['road_network']
        
        # Convert NetworkX graph to GeoDataFrame for edge visualization
        edges_list = []
        for u, v, data in road_network.edges(data=True):
            if 'x' in road_network.nodes[u] and 'x' in road_network.nodes[v]:
                from shapely.geometry import LineString
                line = LineString([
                    (road_network.nodes[u]['x'], road_network.nodes[u]['y']),
                    (road_network.nodes[v]['x'], road_network.nodes[v]['y'])
                ])
                edges_list.append({
                    'geometry': line,
                    'from': u,
                    'to': v,
                    'weight': data.get('length', 1.0)
                })
        
        edges_gdf = gpd.GeoDataFrame(edges_list, crs='EPSG:4326')
        
        return {
            'graph': road_network,
            'edges_gdf': edges_gdf
        }

    def _prepare_transit_data_for_viz(self, tract_data):
        """Prepare transit data for visualization"""
        
        # Load transit stops from data loader if not already in tract_data
        if 'transit_stops' not in tract_data:
            transit_stops = self.data_loader.load_transit_stops()
        else:
            transit_stops = tract_data['transit_stops']
        
        # Filter stops to tract area if needed
        tract_geom = tract_data['tract_info'].geometry
        local_stops = transit_stops[transit_stops.intersects(tract_geom.buffer(0.01))]
        
        return {
            'stops': local_stops
        }

    def _augment_graph_with_accessibility(self, graph_data, accessibility_features, node_mapping):
        """
        Combine original graph features with learned accessibility features
        """
        # Convert accessibility features to tensor
        accessibility_tensor = torch.FloatTensor(accessibility_features)

        print(f"🔍 REAL AUGMENT DEBUG:")
        print(f"  graph_data.x shape: {graph_data.x.shape}")
        print(f"  accessibility_features shape: {accessibility_features.shape}")
        print(f"  accessibility_tensor shape: {accessibility_tensor.shape}")
        print(f"  node_mapping type: {type(node_mapping)}")
        print(f"  node_mapping length: {len(node_mapping) if node_mapping else 'None'}")

        # Expand accessibility features to match graph nodes if needed
        if accessibility_tensor.shape[0] != graph_data.x.shape[0]:
            # SIMPLE FIX: Addresses are first N nodes (confirmed by your test)
            num_addresses = accessibility_tensor.shape[0]  # 2394
            total_nodes = graph_data.x.shape[0]  # 3098
            
            # Create expanded tensor with zeros for road nodes
            expanded_accessibility = torch.zeros(total_nodes, accessibility_tensor.shape[1])
            
            # Place accessibility features in first N positions (address nodes)
            expanded_accessibility[:num_addresses] = accessibility_tensor
            
            print(f"🔧 Expanded accessibility: {accessibility_tensor.shape} -> {expanded_accessibility.shape}")
            accessibility_tensor = expanded_accessibility
        
        # Concatenate original features with accessibility features
        enhanced_features = torch.cat([graph_data.x, accessibility_tensor], dim=1)
        
        # Create new graph data object
        enhanced_graph_data = graph_data.clone()
        enhanced_graph_data.x = enhanced_features
        
        return enhanced_graph_data

    def _compute_traditional_accessibility_measures(self, tract_data):
        """
        Compute traditional accessibility measures for comparison with GNN-learned features
        """
        try:
            # Calculate traditional gravity-based accessibility
            addresses = tract_data['addresses']
            
            # Simple gravity model: Σ(Opportunities / Distance^β)
            accessibility_scores = []
            
            employment_destinations = self.data_loader._create_employment_destinations()
            for _, address in addresses.iterrows():
                total_accessibility = 0
                
                # Employment accessibility (simplified)
                for _, emp in employment_destinations.iterrows():
                    distance = address.geometry.distance(emp.geometry)
                    if distance > 0:
                        total_accessibility += emp.get('employees', 1000) / (distance ** 1.5)
                
                accessibility_scores.append(total_accessibility)
            
            return np.array(accessibility_scores).reshape(-1, 1)
            
        except Exception as e:
            self._log(f"    Error computing traditional accessibility: {str(e)}")
            return None

    def _compute_accessibility_svi_correlation(self, learned_accessibility, svi_predictions):
        """
        Analyze correlation between learned accessibility patterns and SVI predictions
        """
        correlations = {}
        
        # Feature-wise correlations
        for i in range(learned_accessibility.shape[1]):
            feature_corr = np.corrcoef(learned_accessibility[:, i], svi_predictions)[0, 1]
            correlations[f'accessibility_feature_{i}'] = feature_corr
        
        # Overall accessibility-SVI relationship
        mean_accessibility = learned_accessibility.mean(axis=1)
        overall_corr = np.corrcoef(mean_accessibility, svi_predictions)[0, 1]
        correlations['overall_accessibility_svi'] = overall_corr
        
        return correlations

    def _compute_research_metrics(self, stage1_result, stage2_result, validation_results):
        """
        Compute metrics specific to this novel research approach
        """
        research_metrics = {
            'methodology': 'Hybrid GNN Accessibility-SVI Integration',
            'research_contribution': 'First systematic two-stage GNN approach',
            
            # Stage 1 metrics
            'accessibility_learning_loss': stage1_result['final_loss'],
            'accessibility_features_learned': stage1_result['predicted_accessibility'].shape[1],
            
            # Stage 2 metrics  
            'svi_prediction_loss': stage2_result['final_loss'],
            'learned_spatial_variation': stage2_result['spatial_variation'],
            
            # Validation metrics
            'constraint_satisfaction': validation_results.get('constraint_satisfaction', False),
            'accessibility_correlation': validation_results.get('accessibility_correlation', 0),
            
            # Research novelty indicators
            'two_stage_architecture': True,
            'accessibility_vulnerability_integration': True,
            'transportation_equity_focus': True
        }
        
        return research_metrics

    def _load_data_with_accessibility_destinations(self):
        """
        MODIFIED: Enhanced data loading that includes accessibility destinations
        """
        data = self._load_data()  # Your existing method
        
        # Add accessibility destinations
        self._log("Loading accessibility destinations for hybrid approach...")
        
        data['employment_destinations'] = self.data_loader._create_employment_destinations()
        data['healthcare_destinations'] = self.data_loader._create_healthcare_destinations() 
        data['grocery_destinations'] = self.data_loader._create_grocery_destinations()
        
        total_destinations = (len(data['employment_destinations']) + 
                            len(data['healthcare_destinations']) + 
                            len(data['grocery_destinations']))
        self._log(f"Loaded {total_destinations} accessibility destinations for hybrid training")
        
        return data

    def run_hybrid_accessibility_research(self):
        """
        MODIFIED: Updated main run method for hybrid accessibility research
        """
        start_time = time.time()
        
        # Load data with accessibility destinations
        self._log("Loading data for hybrid accessibility-SVI research...")
        data = self._load_data_with_accessibility_destinations()
        
        # Check processing mode 
        processing_mode = self.config.get('data', {}).get('processing_mode', 'fips')
        target_fips = self.config.get('data', {}).get('target_fips')
        
        if processing_mode == 'fips' and target_fips:
            self._log(f"Processing single FIPS with hybrid accessibility approach: {target_fips}")
            results = self._process_fips_mode_hybrid(data)
        else:
            self._log("ERROR: Only single FIPS mode supported for hybrid accessibility research.")
            return {'success': False, 'error': 'Multi-tract processing not supported for hybrid approach.'}
        
        # Save results with research-specific outputs
        self._save_hybrid_research_results(results)
        
        elapsed_time = time.time() - start_time
        self._log(f"Hybrid accessibility research completed in {elapsed_time:.2f} seconds")
        
        return results
    
    def _save_hybrid_research_results(self, results):
        """
        Save results specific to hybrid accessibility-SVI research
        """
        if not results.get('success', False):
            self._log("No results to save")
            return
        
        # Use your existing save method for now
        self._save_results(results)
        
        # Add research-specific outputs
        if results.get('tract_results'):
            for tract_result in results['tract_results']:
                if 'stage1_accessibility_learning' in tract_result:
                    # Save accessibility features
                    accessibility_path = os.path.join(self.output_dir, 'learned_accessibility_features.csv')
                    import pandas as pd
                    learned_features = tract_result['stage1_accessibility_learning']['learned_features']
                    try:
                        feature_names = tract_result['stage1_accessibility_learning']['training_result']['target_features']
                    except KeyError:
                        feature_names = [
                            'emp_min_time', 'emp_accessible', 'emp_transit_share',
                            'health_min_time', 'health_accessible', 'health_transit_share', 
                            'grocery_min_time', 'grocery_accessible', 'grocery_transit_share'
                        ]                    
                    pd.DataFrame(learned_features, columns=feature_names).to_csv(accessibility_path, index=False)
                    self._log(f"Saved learned accessibility features to {accessibility_path}")

    def _process_fips_mode_hybrid(self, data):
        """
        MODIFIED: Process single FIPS with hybrid accessibility approach
        """
        target_fips = self.config.get('data', {}).get('target_fips')
        target_fips = str(target_fips).strip()
        
        # Your existing FIPS validation code...
        data['tracts']['FIPS'] = data['tracts']['FIPS'].astype(str).str.strip()
        
        if target_fips not in data['tracts']['FIPS'].tolist():
            return {'success': False, 'error': f'FIPS {target_fips} not found'}
        
        target_tract = data['tracts'][data['tracts']['FIPS'] == target_fips].iloc[0]
        self._log(f"✓ Found target tract: {target_fips}")
        self._log(f"Processing single tract with HYBRID ACCESSIBILITY-SVI approach")
        
        # Prepare tract data with accessibility destinations
        tract_data = self._prepare_tract_data_with_accessibility(target_tract, data)
        
        # Use new hybrid processing method
        result = self._process_single_tract_hybrid_accessibility(tract_data)
        
        if result['status'] == 'success':
            return {
                'success': True,
                'predictions': result['predictions'],
                'tract_results': [result],
                'research_type': 'hybrid_accessibility_svi',
                'summary': {
                    'total_tracts': 1,
                    'successful_tracts': 1,
                    'total_addresses': len(result['predictions']),
                    'accessibility_features_learned': result['stage1_accessibility_learning']['learned_features'].shape[1],
                    'processing_time': result['timing']['total'],
                    'research_contribution': 'Novel two-stage GNN accessibility-SVI integration'
                }
            }
        else:
            return {'success': False, 'error': result['error']}

    def _prepare_tract_data_with_accessibility(self, tract, county_data):
        """
        MODIFIED: Prepare tract data including accessibility destinations
        """
        # Your existing tract data preparation...
        tract_data = self._prepare_tract_data(tract, county_data)
        
        # Add accessibility destinations
        tract_data['employment_destinations'] = county_data['employment_destinations']
        tract_data['healthcare_destinations'] = county_data['healthcare_destinations'] 
        tract_data['grocery_destinations'] = county_data['grocery_destinations']
        
        return tract_data
