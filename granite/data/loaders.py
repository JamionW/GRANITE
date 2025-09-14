"""
Streamlined Data Loaders for Simplified GRANITE
Focus on robust accessibility feature computation
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import networkx as nx
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """
    Streamlined data loader focused on accessibility feature computation
    Simplifies the original DataLoader for the direct accessibility → SVI approach
    """
    
    def __init__(self, data_dir: str = './data', config: dict = None):
        self.data_dir = data_dir
        self.config = config or {}
        self.verbose = config.get('processing', {}).get('verbose', False) if config else False
        
        # Simplified caching
        self._address_cache = None
        self._svi_cache = None
        self._transit_cache = None
        
        os.makedirs(data_dir, exist_ok=True)
    
    def _log(self, message: str):
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] AccessibilityLoader: {message}")

    # Core data loading methods (simplified from original)
    def load_census_tracts(self, state_fips: str = '47', county_fips: str = '065') -> gpd.GeoDataFrame:
        """Load census tract geometries"""
        self._log(f"Loading census tracts for {state_fips}-{county_fips}...")
        
        local_file = os.path.join(self.data_dir, 'raw', f'tl_2020_{state_fips}_tract.shp')
        
        try:
            if os.path.exists(local_file):
                tracts = gpd.read_file(local_file)
                self._log(f"Loaded {len(tracts)} tracts from local file")
            else:
                raise FileNotFoundError(f"Census tracts not found at {local_file}")
            
            # Filter to county
            county_tracts = tracts[tracts['COUNTYFP'] == county_fips].copy()
            county_tracts['FIPS'] = county_tracts['GEOID'].astype(str).str.strip()
            
            if county_tracts.crs is None:
                county_tracts.set_crs(epsg=4326, inplace=True)
                
            self._log(f"Filtered to {len(county_tracts)} tracts")
            return county_tracts
            
        except Exception as e:
            self._log(f"Error loading census tracts: {str(e)}")
            raise

    def load_svi_data(self, state_fips: str = '47', county_name: str = 'Hamilton') -> pd.DataFrame:
        """Load Social Vulnerability Index data"""
        self._log(f"Loading SVI data for {county_name} County, {state_fips}...")
        
        svi_file = os.path.join(self.data_dir, 'raw', f'SVI_2020_US.csv')
        
        try:
            if os.path.exists(svi_file):
                svi_data = pd.read_csv(svi_file, dtype={'FIPS': str, 'ST': str})
                self._log(f"Loaded local SVI data ({len(svi_data)} records)")
            else:
                raise FileNotFoundError(f"SVI data not found at {svi_file}")
            
            # Filter to county
            county_svi = svi_data[
                (svi_data['ST'] == state_fips) & 
                (svi_data['COUNTY'] == county_name)
            ].copy()
            
            county_svi['FIPS'] = county_svi['FIPS'].astype(str)
            county_svi['RPL_THEMES'] = county_svi['RPL_THEMES'].replace(-999, np.nan)
            
            columns = ['FIPS', 'LOCATION', 'RPL_THEMES', 'E_TOTPOP']
            county_svi = county_svi[columns]
            
            valid_count = county_svi['RPL_THEMES'].notna().sum()
            self._log(f"Loaded {len(county_svi)} tracts ({valid_count} with valid SVI)")
            
            return county_svi
            
        except Exception as e:
            self._log(f"Error loading SVI data: {str(e)}")
            raise

    def calculate_multimodal_travel_times_batch(self, origins: gpd.GeoDataFrame, 
                                            destinations: gpd.GeoDataFrame,
                                            time_periods: list = ['morning']) -> pd.DataFrame:
        """
        Network-based travel time calculation for accessibility analysis
        """
        from geopy.distance import geodesic
        
        results = []
        
        for orig_idx, origin in origins.iterrows():
            orig_id = origin.get('address_id', orig_idx)
            orig_coord = (origin.geometry.y, origin.geometry.x)
            
            for dest_idx, destination in destinations.iterrows():
                dest_id = destination.get('dest_id', dest_idx)
                dest_coord = (destination.geometry.y, destination.geometry.x)
                
                # Use geodesic distance with network factor
                straight_distance = geodesic(orig_coord, dest_coord).kilometers
                network_distance = straight_distance * 1.3  # Network routing factor
                
                # Calculate multi-modal times
                walk_time = network_distance / 5.0 * 60  # 5 km/h walking
                drive_time = network_distance / 25.0 * 60  # 25 km/h urban driving
                
                # Transit time estimation
                if 1 <= network_distance <= 15:
                    transit_base = network_distance / 12.0 * 60  # 12 km/h transit
                    transit_wait = 8  # Average wait time
                    transit_time = transit_base + transit_wait
                else:
                    transit_time = walk_time * 1.2
                
                # Best mode
                times = {'walk': walk_time, 'drive': drive_time, 'transit': transit_time}
                best_mode = min(times.keys(), key=lambda k: times[k])
                combined_time = times[best_mode]
                
                results.append({
                    'origin_id': orig_id,
                    'destination_id': dest_id,
                    'destination_type': destination.get('dest_type', 'unknown'),
                    'walk_time': walk_time,
                    'drive_time': drive_time,
                    'transit_time': transit_time,
                    'combined_time': combined_time,
                    'best_mode': best_mode
                })
        
        return pd.DataFrame(results)

    def load_road_network(self, roads_file: Optional[str] = None, 
                        state_fips: str = '47', county_fips: str = '065') -> gpd.GeoDataFrame:
        """
        Load road network data
        
        Parameters:
        -----------
        roads_file : str, optional
            Path to roads shapefile
        state_fips : str
            State FIPS code
        county_fips : str
            County FIPS code
            
        Returns:
        --------
        gpd.GeoDataFrame
            Road network geometries
        """
        self._log("Loading road network...")
        
        try:
            if roads_file and os.path.exists(roads_file):
                roads = gpd.read_file(roads_file)
                self._log(f"Loaded {len(roads)} road segments from {roads_file}")
            else:
                # Try default location
                default_file = os.path.join(
                    self.data_dir, 'raw', 
                    f'tl_2023_{state_fips}{county_fips}_roads.shp'
                )
                
                if os.path.exists(default_file):
                    roads = gpd.read_file(default_file)
                    self._log(f"Loaded {len(roads)} road segments")
                else:
                    url = f"https://www2.census.gov/geo/tiger/TIGER2023/ROADS/"
                    self._log(f"Road file not found. Please download from: {url}")
                    self._log("Returning empty road network - accessibility will be limited")
                    return gpd.GeoDataFrame(geometry=[])
            
            # Ensure CRS
            if roads.crs is None:
                roads.set_crs(epsg=4326, inplace=True)
            
            return roads
            
        except Exception as e:
            self._log(f"Error loading roads: {str(e)}")
            # Return empty GeoDataFrame
            return gpd.GeoDataFrame(geometry=[])

    def load_address_points(self, state_fips: str = '47', county_fips: str = '065') -> gpd.GeoDataFrame:
        """Load real Chattanooga address points (simplified)"""
        
        if self._address_cache is not None:
            self._log(f"Using cached address data ({len(self._address_cache)} addresses)")
            return self._address_cache
        
        self._log("Loading Chattanooga address data...")
        
        # Try multiple locations for address file
        address_files = [
            os.path.join(self.data_dir, 'chattanooga.geojson'),
            os.path.join(self.data_dir, 'raw', 'chattanooga.geojson'),
            './chattanooga.geojson'
        ]
        
        addresses_gdf = None
        for address_file in address_files:
            if os.path.exists(address_file):
                try:
                    addresses_gdf = gpd.read_file(address_file)
                    
                    if len(addresses_gdf) > 0:
                        # Standardize columns
                        addresses_gdf = addresses_gdf.copy()
                        
                        if 'address_id' not in addresses_gdf.columns:
                            addresses_gdf['address_id'] = range(len(addresses_gdf))
                        
                        # Ensure proper CRS
                        if addresses_gdf.crs is None:
                            addresses_gdf.set_crs(epsg=4326, inplace=True)
                        elif addresses_gdf.crs != 'EPSG:4326':
                            addresses_gdf = addresses_gdf.to_crs('EPSG:4326')
                        
                        # Keep essential columns
                        keep_cols = ['address_id', 'geometry']
                        if 'full_address' in addresses_gdf.columns:
                            keep_cols.append('full_address')
                        elif 'street' in addresses_gdf.columns:
                            addresses_gdf['full_address'] = addresses_gdf['street'].fillna('Unknown Address')
                            keep_cols.append('full_address')
                        else:
                            addresses_gdf['full_address'] = 'Address ' + addresses_gdf['address_id'].astype(str)
                            keep_cols.append('full_address')
                        
                        addresses_gdf = addresses_gdf[keep_cols]
                        
                        self._log(f"Loaded {len(addresses_gdf)} real addresses from {address_file}")
                        break
                        
                except Exception as e:
                    self._log(f"Error loading {address_file}: {str(e)}")
                    continue
        
        if addresses_gdf is None or len(addresses_gdf) == 0:
            self._log("WARNING: No real address data found, creating synthetic addresses")
            addresses_gdf = self._create_synthetic_addresses(state_fips, county_fips)
        
        # Cache the result
        self._address_cache = addresses_gdf
        
        self._log(f"Final address count: {len(addresses_gdf)}")
        return addresses_gdf

    def get_addresses_for_tract(self, fips_code: str) -> gpd.GeoDataFrame:
        """Get addresses within a specific census tract"""
        
        try:
            # Load all addresses
            all_addresses = self.load_address_points()
            
            if len(all_addresses) == 0:
                return gpd.GeoDataFrame(columns=['address_id', 'geometry'])
            
            # Load tract geometry
            state_fips = fips_code[:2]
            county_fips = fips_code[2:5]
            tracts = self.load_census_tracts(state_fips, county_fips)
            
            tract = tracts[tracts['FIPS'] == fips_code]
            if len(tract) == 0:
                self._log(f"Tract {fips_code} not found")
                return gpd.GeoDataFrame(columns=['address_id', 'geometry'])
            
            tract_geom = tract.iloc[0].geometry
            
            # Spatial filter
            tract_addresses = all_addresses[all_addresses.geometry.within(tract_geom)].copy()
            tract_addresses['tract_fips'] = fips_code
            
            self._log(f"Found {len(tract_addresses)} addresses in tract {fips_code}")
            return tract_addresses
            
        except Exception as e:
            self._log(f"Error getting addresses for tract {fips_code}: {str(e)}")
            return gpd.GeoDataFrame(columns=['address_id', 'geometry'])

    def _create_synthetic_addresses(self, state_fips: str, county_fips: str, density: int = 200) -> gpd.GeoDataFrame:
        """Create synthetic addresses as fallback"""
        
        try:
            # Load county boundary from tracts
            tracts = self.load_census_tracts(state_fips, county_fips)
            county_boundary = tracts.geometry.unary_union
            
            bounds = county_boundary.bounds
            addresses = []
            address_id = 0
            
            # Generate points within county bounds
            attempts = 0
            max_attempts = density * 10
            
            while len(addresses) < density and attempts < max_attempts:
                x = np.random.uniform(bounds[0], bounds[2])
                y = np.random.uniform(bounds[1], bounds[3])
                point = Point(x, y)
                
                if county_boundary.contains(point):
                    addresses.append({
                        'address_id': address_id,
                        'geometry': point,
                        'full_address': f'Synthetic Address {address_id}'
                    })
                    address_id += 1
                
                attempts += 1
            
            addresses_gdf = gpd.GeoDataFrame(addresses, crs='EPSG:4326')
            self._log(f"Created {len(addresses_gdf)} synthetic addresses")
            
            return addresses_gdf
            
        except Exception as e:
            self._log(f"Error creating synthetic addresses: {str(e)}")
            return gpd.GeoDataFrame(columns=['address_id', 'geometry', 'full_address'])

    def _get_county_name(self, state_fips: str, county_fips: str) -> str:
        """Map FIPS codes to county name"""
        county_map = {
            ('47', '065'): 'Hamilton'
        }
        return county_map.get((state_fips, county_fips), 'Unknown')

    # Destination creation methods (core accessibility destinations)
    def create_employment_destinations(self) -> gpd.GeoDataFrame:
        """Create employment destinations for accessibility analysis"""
        
        employers = [
            {'name': 'Downtown Chattanooga', 'lat': 35.0456, 'lon': -85.3097, 'employees': 5000, 'type': 'mixed'},
            {'name': 'Volkswagen Chattanooga', 'lat': 35.0614, 'lon': -85.1580, 'employees': 4000, 'type': 'manufacturing'},
            {'name': 'BlueCross BlueShield TN', 'lat': 35.0456, 'lon': -85.3097, 'employees': 3500, 'type': 'insurance'},
            {'name': 'Erlanger Health System', 'lat': 35.0539, 'lon': -85.3083, 'employees': 8000, 'type': 'healthcare'},
            {'name': 'University of Tennessee Chattanooga', 'lat': 35.0456, 'lon': -85.3011, 'employees': 2500, 'type': 'education'},
            {'name': 'Tennessee Valley Authority', 'lat': 35.0398, 'lon': -85.3062, 'employees': 1500, 'type': 'utilities'},
            {'name': 'Hamilton County Government', 'lat': 35.0456, 'lon': -85.3097, 'employees': 2000, 'type': 'government'},
            {'name': 'Hamilton Place Mall Area', 'lat': 35.0407, 'lon': -85.2111, 'employees': 3000, 'type': 'retail'},
            {'name': 'East Brainerd Business District', 'lat': 35.0156, 'lon': -85.2180, 'employees': 1500, 'type': 'mixed'},
            {'name': 'Northshore Business District', 'lat': 35.0722, 'lon': -85.2967, 'employees': 1200, 'type': 'mixed'}
        ]
        
        geometries = [Point(emp['lon'], emp['lat']) for emp in employers]
        employment_gdf = gpd.GeoDataFrame(employers, geometry=geometries, crs='EPSG:4326')
        employment_gdf['dest_id'] = range(len(employment_gdf))
        employment_gdf['dest_type'] = 'employment'
        
        self._log(f"Created {len(employment_gdf)} employment destinations")
        return employment_gdf

    def create_healthcare_destinations(self) -> gpd.GeoDataFrame:
        """Create healthcare destinations for accessibility analysis"""
        
        hospitals = [
            {'name': 'Erlanger Baroness Hospital', 'lat': 35.0539, 'lon': -85.3083, 'beds': 400, 'type': 'General'},
            {'name': 'CHI Memorial Hospital', 'lat': 35.0627, 'lon': -85.2985, 'beds': 300, 'type': 'General'},
            {'name': 'Parkridge Medical Center', 'lat': 35.0456, 'lon': -85.2597, 'beds': 368, 'type': 'General'},
            {'name': 'TriStar StoneCrest Medical Center', 'lat': 35.1156, 'lon': -85.2441, 'beds': 101, 'type': 'General'},
            {'name': 'Erlanger East Hospital', 'lat': 35.0407, 'lon': -85.2111, 'beds': 140, 'type': 'General'},
            {'name': 'Siskin Hospital', 'lat': 35.0456, 'lon': -85.3097, 'beds': 79, 'type': 'Rehabilitation'},
            {'name': 'Moccasin Bend Mental Health', 'lat': 35.0722, 'lon': -85.3365, 'beds': 150, 'type': 'Psychiatric'},
            {'name': 'Parkridge Valley Hospital', 'lat': 35.0175, 'lon': -85.3365, 'beds': 60, 'type': 'General'}
        ]
        
        geometries = [Point(hosp['lon'], hosp['lat']) for hosp in hospitals]
        healthcare_gdf = gpd.GeoDataFrame(hospitals, geometry=geometries, crs='EPSG:4326')
        healthcare_gdf['dest_id'] = range(len(healthcare_gdf))
        healthcare_gdf['dest_type'] = 'healthcare'
        
        self._log(f"Created {len(healthcare_gdf)} healthcare destinations")
        return healthcare_gdf

    def create_grocery_destinations(self) -> gpd.GeoDataFrame:
        """Create grocery destinations for accessibility analysis"""
        
        stores = [
            {'name': 'Walmart Supercenter - Hamilton Place', 'lat': 35.0407, 'lon': -85.2111, 'type': 'supermarket'},
            {'name': 'Kroger - East Brainerd', 'lat': 35.0156, 'lon': -85.2180, 'type': 'supermarket'},
            {'name': 'Publix - Signal Mountain', 'lat': 35.1456, 'lon': -85.3456, 'type': 'supermarket'},
            {'name': 'Food City - Northgate', 'lat': 35.0722, 'lon': -85.2967, 'type': 'supermarket'},
            {'name': 'IGA - Downtown', 'lat': 35.0456, 'lon': -85.3097, 'type': 'grocery'},
            {'name': 'Walmart Neighborhood Market - Hixson', 'lat': 35.1256, 'lon': -85.2441, 'type': 'grocery'},
            {'name': 'Fresh Market - Northshore', 'lat': 35.0627, 'lon': -85.2985, 'type': 'grocery'},
            {'name': 'Bi-Lo - East Ridge', 'lat': 35.0495, 'lon': -85.1938, 'type': 'supermarket'},
            {'name': 'Save-A-Lot - South Chattanooga', 'lat': 35.0175, 'lon': -85.3365, 'type': 'discount'},
            {'name': 'Food Lion - East Chattanooga', 'lat': 35.0456, 'lon': -85.2580, 'type': 'supermarket'}
        ]
        
        geometries = [Point(store['lon'], store['lat']) for store in stores]
        grocery_gdf = gpd.GeoDataFrame(stores, geometry=geometries, crs='EPSG:4326')
        grocery_gdf['dest_id'] = range(len(grocery_gdf))
        grocery_gdf['dest_type'] = 'grocery'
        
        self._log(f"Created {len(grocery_gdf)} grocery destinations")
        return grocery_gdf

    # Simplified travel time computation
    def compute_accessibility_features(self, addresses: gpd.GeoDataFrame) -> np.ndarray:
        """
        Compute comprehensive accessibility features for all addresses
        
        This is the main method called by the pipeline
        Returns matrix: [n_addresses, n_features]
        """
        self._log(f"Computing accessibility features for {len(addresses)} addresses...")
        
        # Create destinations
        destinations = {
            'employment': self.create_employment_destinations(),
            'healthcare': self.create_healthcare_destinations(),
            'grocery': self.create_grocery_destinations()
        }
        
        # Compute features for each destination type
        all_features = []
        
        for dest_type, dest_gdf in destinations.items():
            self._log(f"  Computing {dest_type} accessibility...")
            
            # Calculate travel times
            travel_times = self._calculate_simple_travel_times(addresses, dest_gdf)
            
            # Extract accessibility features
            features = self._extract_accessibility_features_from_times(
                addresses, dest_gdf, travel_times, dest_type
            )
            
            all_features.append(features)
        
        # Combine all destination features
        accessibility_matrix = np.column_stack(all_features)
        
        # Add derived/summary features
        derived_features = self._compute_derived_features(accessibility_matrix)
        
        final_features = np.column_stack([accessibility_matrix, derived_features])
        
        self._log(f"Final accessibility features: {final_features.shape}")
        return final_features

    def _calculate_simple_travel_times(self, addresses: gpd.GeoDataFrame, 
                                     destinations: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Simplified travel time calculation using distance approximation
        More reliable than complex multi-modal routing for this use case
        """
        results = []
        
        for addr_idx, address in addresses.iterrows():
            addr_id = address.get('address_id', addr_idx)
            addr_point = address.geometry
            
            for dest_idx, destination in destinations.iterrows():
                dest_id = destination.get('dest_id', dest_idx)
                dest_point = destination.geometry
                
                # Calculate great circle distance
                distance_deg = addr_point.distance(dest_point)
                distance_km = distance_deg * 111  # Rough conversion to km
                
                # Estimate travel times for different modes
                walk_time = distance_km / 5.0 * 60  # 5 km/h walking speed → minutes
                drive_time = distance_km / 30.0 * 60  # 30 km/h average urban speed → minutes
                
                # Simple transit time estimation
                # Transit is competitive for distances 2-15km, poor outside this range
                if 2 <= distance_km <= 15:
                    transit_base_time = distance_km / 15.0 * 60  # 15 km/h average transit speed
                    transit_wait_time = 10  # Average wait time
                    transit_time = transit_base_time + transit_wait_time
                else:
                    transit_time = walk_time * 1.5  # Transit not competitive
                
                # Best mode and combined time
                times = {'walk': walk_time, 'drive': drive_time, 'transit': transit_time}
                best_mode = min(times.keys(), key=lambda k: times[k])
                combined_time = times[best_mode]
                
                results.append({
                    'origin_id': addr_id,
                    'destination_id': dest_id,
                    'destination_type': destination.get('dest_type', 'unknown'),
                    'walk_time': walk_time,
                    'drive_time': drive_time,
                    'transit_time': transit_time,
                    'combined_time': combined_time,
                    'best_mode': best_mode
                })
        
        return pd.DataFrame(results)

    def _extract_accessibility_features_from_times(self, addresses: gpd.GeoDataFrame,
                                                 destinations: gpd.GeoDataFrame,
                                                 travel_times: pd.DataFrame,
                                                 dest_type: str) -> np.ndarray:
        """Extract 8 accessibility features for one destination type"""
        
        features = []
        
        for addr_idx, address in addresses.iterrows():
            addr_id = address.get('address_id', addr_idx)
            
            # Get travel times for this address to all destinations of this type
            addr_times = travel_times[travel_times['origin_id'] == addr_id]
            
            if len(addr_times) > 0:
                combined_times = addr_times['combined_time'].values
                
                # Time-based features
                min_time = float(np.min(combined_times))
                mean_time = float(np.mean(combined_times))
                percentile_90 = float(np.percentile(combined_times, 90))
                
                # Count-based features (destinations within time thresholds)
                count_30min = int(np.sum(combined_times <= 30))
                count_60min = int(np.sum(combined_times <= 60))
                count_90min = int(np.sum(combined_times <= 90))
                
                # Transit accessibility
                transit_trips = addr_times['best_mode'] == 'transit'
                transit_share = float(transit_trips.mean())
                
                # Overall accessibility score (gravity-style)
                accessibility_score = float(np.sum(1.0 / np.maximum(combined_times, 1.0)))
                
            else:
                # No destinations accessible
                min_time = mean_time = percentile_90 = 120.0
                count_30min = count_60min = count_90min = 0
                transit_share = accessibility_score = 0.0
            
            features.append([
                min_time, mean_time, percentile_90,
                count_30min, count_60min, count_90min,
                transit_share, accessibility_score
            ])
        
        return np.array(features, dtype=np.float64)

    def _compute_derived_features(self, base_features: np.ndarray) -> np.ndarray:
        """Compute derived accessibility features from base metrics"""
        
        n_addresses = base_features.shape[0]
        
        # Assuming base_features has shape [n_addresses, 24] (3 dest types × 8 features each)
        if base_features.shape[1] < 24:
            # Return minimal derived features
            return np.zeros((n_addresses, 4), dtype=np.float64)
        
        derived = []
        
        for i in range(n_addresses):
            # Extract accessibility scores for each destination type (8th feature)
            emp_score = base_features[i, 7]    # Employment accessibility score
            health_score = base_features[i, 15]  # Healthcare accessibility score
            grocery_score = base_features[i, 23] # Grocery accessibility score
            
            # Overall accessibility
            total_accessibility = emp_score + health_score + grocery_score
            
            # Accessibility balance (entropy measure)
            if total_accessibility > 0:
                scores = np.array([emp_score, health_score, grocery_score]) / total_accessibility
                scores = np.maximum(scores, 1e-8)  # Avoid log(0)
                balance = -np.sum(scores * np.log(scores))  # Shannon entropy
            else:
                balance = 0.0
            
            # Transit dependence
            emp_transit = base_features[i, 6]
            health_transit = base_features[i, 14]
            grocery_transit = base_features[i, 22]
            avg_transit_dependence = (emp_transit + health_transit + grocery_transit) / 3
            
            # Time efficiency (min vs mean time performance)
            emp_min, emp_mean = base_features[i, 0], base_features[i, 1]
            health_min, health_mean = base_features[i, 8], base_features[i, 9]
            grocery_min, grocery_mean = base_features[i, 16], base_features[i, 17]
            
            all_mins = [emp_min, health_min, grocery_min]
            all_means = [emp_mean, health_mean, grocery_mean]
            
            min_avg = np.mean(all_mins)
            mean_avg = np.mean(all_means)
            
            time_efficiency = (mean_avg - min_avg) / mean_avg if mean_avg > 0 else 0
            
            derived.append([
                total_accessibility,
                balance,
                avg_transit_dependence,
                time_efficiency
            ])
        
        return np.array(derived, dtype=np.float64)

    def create_accessibility_baseline_comparison(self, addresses: gpd.GeoDataFrame) -> Dict:
        """
        Create baseline accessibility measures for comparison with GNN predictions
        
        Returns traditional accessibility metrics computed using same destinations
        """
        self._log("Computing baseline accessibility metrics for comparison...")
        
        # Use same destinations as main analysis
        destinations = {
            'employment': self.create_employment_destinations(),
            'healthcare': self.create_healthcare_destinations(),
            'grocery': self.create_grocery_destinations()
        }
        
        baseline_results = {}
        
        for dest_type, dest_gdf in destinations.items():
            # Simple gravity model
            gravity_scores = self._compute_gravity_accessibility(addresses, dest_gdf)
            
            # Cumulative opportunities within thresholds
            opportunities_30min = self._compute_cumulative_opportunities(addresses, dest_gdf, threshold_minutes=30)
            opportunities_60min = self._compute_cumulative_opportunities(addresses, dest_gdf, threshold_minutes=60)
            
            baseline_results[dest_type] = {
                'gravity_scores': gravity_scores,
                'opportunities_30min': opportunities_30min,
                'opportunities_60min': opportunities_60min
            }
        
        return baseline_results

    def _compute_gravity_accessibility(self, addresses: gpd.GeoDataFrame, 
                                     destinations: gpd.GeoDataFrame) -> np.ndarray:
        """Traditional gravity model accessibility"""
        
        gravity_scores = []
        
        for _, address in addresses.iterrows():
            addr_point = address.geometry
            gravity_score = 0
            
            for _, destination in destinations.iterrows():
                dest_point = destination.geometry
                distance_deg = addr_point.distance(dest_point)
                distance_km = distance_deg * 111
                
                if distance_km > 0:
                    # Simple gravity: attraction / distance^2
                    attraction = destination.get('employees', destination.get('beds', 100))
                    gravity_score += attraction / (distance_km ** 1.5)
            
            gravity_scores.append(gravity_score)
        
        return np.array(gravity_scores)

    def _compute_cumulative_opportunities(self, addresses: gpd.GeoDataFrame,
                                        destinations: gpd.GeoDataFrame,
                                        threshold_minutes: float = 30) -> np.ndarray:
        """Count destinations within time threshold"""
        
        opportunity_counts = []
        
        # Convert time threshold to approximate distance threshold
        threshold_km = threshold_minutes / 60 * 30  # Assume 30 km/h average speed
        threshold_deg = threshold_km / 111  # Convert to degrees
        
        for _, address in addresses.iterrows():
            addr_point = address.geometry
            count = 0
            
            for _, destination in destinations.iterrows():
                dest_point = destination.geometry
                distance_deg = addr_point.distance(dest_point)
                
                if distance_deg <= threshold_deg:
                    count += 1
            
            opportunity_counts.append(count)
        
        return np.array(opportunity_counts)


# Convenience functions for backward compatibility
def load_tract_accessibility_data(fips_code: str, data_dir: str = './data', config: dict = None) -> Dict:
    """
    Load all accessibility data for a single tract
    
    Main function called by the simplified pipeline
    """
    loader = DataLoader(data_dir=data_dir, config=config)
    
    # Load tract addresses
    addresses = loader.get_addresses_for_tract(fips_code)
    
    if len(addresses) == 0:
        raise ValueError(f"No addresses found for tract {fips_code}")
    
    # Compute accessibility features
    accessibility_features = loader.compute_accessibility_features(addresses)
    
    # Load tract info and SVI
    state_fips = fips_code[:2]
    county_fips = fips_code[2:5]
    
    tracts = loader.load_census_tracts(state_fips, county_fips)
    county_name = loader._get_county_name(state_fips, county_fips)
    svi_data = loader.load_svi_data(state_fips, county_name)
    
    tract_info = tracts[tracts['FIPS'] == fips_code].iloc[0]
    tract_svi = svi_data[svi_data['FIPS'] == fips_code]['RPL_THEMES'].iloc[0]
    
    return {
        'addresses': addresses,
        'accessibility_features': accessibility_features,
        'tract_info': tract_info,
        'tract_svi': tract_svi,
        'feature_shape': accessibility_features.shape
    }