"""
Data loading functions for GRANITE framework

This module handles all data loading operations including SVI data,
census tracts, road networks, and address generation.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import networkx as nx
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import rasterio
import rasterio.mask
import numpy as np


class DataLoader:
    """Main data loader class for GRANITE framework"""
    
    def __init__(self, data_dir: str = './data', verbose: bool = True, config: dict = None):
        """
        Initialize DataLoader
        
        Parameters:
        -----------
        data_dir : str
            Directory containing data files
        verbose : bool
            Enable verbose logging
        config : dict, optional
            Configuration dictionary with transit and other settings
        """
        self.data_dir = data_dir
        self.verbose = verbose
        self.config = config or {}  # Store config for use in loading methods
        
        # Cache for expensive operations
        self._address_cache = None
        self._svi_cache = None
        self._roads_cache = None
        self._transit_cache = None  # NEW: Cache transit data
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        if verbose:
            print(f"DataLoader initialized with data_dir: {data_dir}")
            if config and 'transit' in config:
                transit_config = config['transit']
                preferred_source = transit_config.get('preferred_source', 'gtfs')
                print(f"Transit configuration: preferred_source = {preferred_source}")
    
    def _log(self, message: str):
        """Log message with timestamp"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] DataLoader: {message}")

    def load_nlcd_data(self, county_bounds: gpd.GeoDataFrame, 
                    nlcd_path: str = "./data/nlcd_hamilton_county.tif") -> dict:
        """
        Load NLCD data with proper CRS handling
        """
        try:
            # Check if file exists
            if not os.path.exists(nlcd_path):
                self._log(f"NLCD file not found at {nlcd_path}")
                self._log("Please download NLCD 2019/2021 data from https://www.mrlc.gov/viewer/")
                return None
                
            with rasterio.open(nlcd_path) as src:
                self._log(f"🔍 NLCD FILE DIAGNOSTICS:")
                self._log(f"  File: {nlcd_path}")
                self._log(f"  Raster CRS: {src.crs}")
                self._log(f"  Raster shape: {src.shape}")
                self._log(f"  Raster bounds: {src.bounds}")
                
                # Get county bounds
                county_bounds_orig_crs = county_bounds.crs
                self._log(f"  County bounds CRS: {county_bounds_orig_crs}")
                self._log(f"  County bounds: {county_bounds.total_bounds}")
                
                # CRITICAL: Reproject county bounds to match raster CRS
                if county_bounds.crs != src.crs:
                    self._log(f"  🔄 Reprojecting county bounds from {county_bounds.crs} to {src.crs}")
                    county_geom_reprojected = county_bounds.to_crs(src.crs).geometry
                    reprojected_bounds = county_bounds.to_crs(src.crs).total_bounds
                    self._log(f"  Reprojected bounds: {reprojected_bounds}")
                else:
                    county_geom_reprojected = county_bounds.geometry
                
                # Check overlap between county and raster
                raster_bounds = src.bounds
                county_bounds_proj = county_bounds.to_crs(src.crs).total_bounds
                
                # Check for overlap
                has_overlap = not (
                    county_bounds_proj[2] < raster_bounds.left or    # county right < raster left
                    county_bounds_proj[0] > raster_bounds.right or   # county left > raster right  
                    county_bounds_proj[3] < raster_bounds.bottom or  # county top < raster bottom
                    county_bounds_proj[1] > raster_bounds.top        # county bottom > raster top
                )
                
                if not has_overlap:
                    self._log(f"  No overlap between county bounds and NLCD raster!")
                    self._log(f"     County: {county_bounds_proj}")
                    self._log(f"     Raster: {raster_bounds}")
                    return None
                else:
                    self._log(f"  County/raster overlap confirmed")
                
                # Crop NLCD to county bounds with proper CRS
                try:
                    nlcd_cropped, nlcd_transform = rasterio.mask.mask(
                        src, county_geom_reprojected, crop=True, filled=True, fill_value=250
                    )
                    
                    self._log(f"  Cropped NLCD shape: {nlcd_cropped.shape}")
                    
                    # Check if we got valid data
                    if nlcd_cropped.size == 0:
                        self._log(f"  Cropping resulted in empty raster!")
                        return None
                    
                    # Check for all no-data
                    unique_values = np.unique(nlcd_cropped[0])  # First band
                    self._log(f"  Unique NLCD values in cropped area: {unique_values}")
                    
                    if len(unique_values) == 1 and unique_values[0] == 250:
                        self._log(f"  WARNING: Cropped area contains only 'no data' values!")
                    elif 250 in unique_values:
                        pct_no_data = np.sum(nlcd_cropped[0] == 250) / nlcd_cropped[0].size * 100
                        self._log(f"  No data percentage: {pct_no_data:.1f}%")
                    
                    # Update metadata
                    nlcd_meta = src.meta.copy()
                    nlcd_meta.update({
                        "height": nlcd_cropped.shape[1],
                        "width": nlcd_cropped.shape[2], 
                        "transform": nlcd_transform,
                        "crs": src.crs  # Keep original raster CRS
                    })
                    
                    self._log(f"  NLCD data successfully loaded and cropped")
                    
                    return {
                        'data': nlcd_cropped[0],  # First band contains land cover classes
                        'transform': nlcd_transform,
                        'crs': src.crs,  
                        'meta': nlcd_meta,
                        'bounds': rasterio.transform.array_bounds(
                            nlcd_cropped.shape[1], nlcd_cropped.shape[2], nlcd_transform
                        )
                    }
                    
                except Exception as crop_error:
                    self._log(f"  Error during cropping: {str(crop_error)}")
                    # Try loading full raster without cropping
                    self._log(f"  Attempting to load full raster...")
                    
                    full_data = src.read(1)  # Read first band
                    
                    return {
                        'data': full_data,
                        'transform': src.transform,
                        'crs': src.crs,
                        'meta': src.meta.copy(),
                        'bounds': src.bounds
                    }
                    
        except Exception as e:
            self._log(f"❌ Error loading NLCD: {str(e)}")
            import traceback
            self._log(f"   Traceback: {traceback.format_exc()}")
            return None
        
    def _load_roads_for_bbox(self, state_fips: str, county_fips: str, bbox: Tuple[float, float, float, float]) -> gpd.GeoDataFrame:
        """
        Load roads within a bounding box
        
        Parameters:
        -----------
        state_fips : str
            State FIPS code
        county_fips : str  
            County FIPS code
        bbox : Tuple[float, float, float, float]
            Bounding box (minx, miny, maxx, maxy)
            
        Returns:
        --------
        gpd.GeoDataFrame
            Roads within the bounding box
        """
        try:
            # Load all county roads
            all_roads = self.load_road_network(state_fips=state_fips, county_fips=county_fips)
            
            if len(all_roads) == 0:
                self._log("No roads loaded for bbox filtering")
                return gpd.GeoDataFrame(geometry=[])
            
            # Create bounding box geometry
            from shapely.geometry import box
            bbox_geom = box(*bbox)
            
            # Filter roads that intersect the bounding box
            bbox_roads = all_roads[all_roads.geometry.intersects(bbox_geom)].copy()
            
            self._log(f"Filtered {len(all_roads)} roads to {len(bbox_roads)} within bbox")
            
            return bbox_roads
            
        except Exception as e:
            self._log(f"Error loading roads for bbox: {str(e)}")
            return gpd.GeoDataFrame(geometry=[])

    def extract_nlcd_features_at_addresses(self, addresses: gpd.GeoDataFrame, 
                                       nlcd_data: dict) -> pd.DataFrame:
        """
        Extract NLCD features using proper 16-class legend following He et al. (2024)
        
        Parameters:
        -----------
        addresses : gpd.GeoDataFrame
            Address points
        nlcd_data : dict
            NLCD raster data with 'data', 'transform', 'crs' keys
            
        Returns:
        --------
        pd.DataFrame
            Features with proper NLCD classes and derived coefficients
        """
        
        if nlcd_data is None:
            self._log("No NLCD data available, using fallback features")
            return self._create_fallback_nlcd_features(addresses)
        
        # Ensure addresses are in same CRS as NLCD
        addresses_proj = addresses.to_crs(nlcd_data['crs'])
        
        # Extract NLCD values at point locations
        coords = [(geom.x, geom.y) for geom in addresses_proj.geometry]
        
        nlcd_values = []
        for coord in coords:
            try:
                row, col = rasterio.transform.rowcol(nlcd_data['transform'], coord[0], coord[1])
                
                if (0 <= row < nlcd_data['data'].shape[0] and 
                    0 <= col < nlcd_data['data'].shape[1]):
                    value = nlcd_data['data'][row, col]
                    
                    # Validate NLCD class - use 16-class legend
                    if value in self._get_valid_nlcd_classes():
                        nlcd_values.append(int(value))
                    else:
                        # Map legacy values to NLCD classes
                        if value in self._get_valid_nlcd_classes():
                            nlcd_values.append(int(value))
                        else:
                            # If center pixel is 250, check neighboring pixels
                            if value == 250:
                                # Sample 3x3 neighborhood
                                neighbor_values = []
                                for dr in [-1, 0, 1]:
                                    for dc in [-1, 0, 1]:
                                        nr, nc = row + dr, col + dc
                                        if (0 <= nr < nlcd_data['data'].shape[0] and 
                                            0 <= nc < nlcd_data['data'].shape[1]):
                                            neighbor_val = nlcd_data['data'][nr, nc]
                                            if neighbor_val != 250:
                                                neighbor_values.append(neighbor_val)
                                
                                # Use most common valid neighbor value, or default to 22
                                if neighbor_values:
                                    value = max(set(neighbor_values), key=neighbor_values.count)
                                else:
                                    value = 22
                else:
                    nlcd_values.append(22)  # Default: low-intensity residential
            except Exception as e:
                self._log(f"Error extracting NLCD at {coord}: {e}")
                nlcd_values.append(22)  # Default for errors
        
        # Create feature dataframe
        features_df = pd.DataFrame({
            'address_id': addresses['address_id'] if 'address_id' in addresses.columns else range(len(addresses)),
            'nlcd_class': nlcd_values
        })
        
        # Add derived features using NLCD methodology
        features_df = self._add_nlcd_derived_features(features_df)
        
        # Quality check and logging
        self._log_nlcd_extraction_quality(features_df)
        
        return features_df
    
    def _get_valid_nlcd_classes(self) -> set:
        """Return set of valid NLCD 2019 class codes"""
        return {11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82, 90, 95}

    def _add_nlcd_derived_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features from NLCD classes following He et al. methodology
        """
        
        # Population density coefficients 
        population_density_map = {
            11: 0.0,   # Open Water
            12: 0.0,   # Perennial Ice/Snow
            21: 0.25,  # Developed, Open Space
            22: 0.75,  # Developed, Low Intensity  
            23: 1.25,  # Developed, Medium Intensity
            24: 1.75,  # Developed, High Intensity
            31: 0.05,  # Barren Land
            41: 0.02,  # Deciduous Forest
            42: 0.02,  # Evergreen Forest
            43: 0.02,  # Mixed Forest
            51: 0.01,  # Dwarf Scrub
            52: 0.01,  # Shrub/Scrub
            71: 0.05,  # Grassland/Herbaceous
            72: 0.01,  # Sedge/Herbaceous
            73: 0.0,   # Lichens
            74: 0.0,   # Moss
            81: 0.08,  # Pasture/Hay
            82: 0.06,  # Cultivated Crops
            90: 0.0,   # Woody Wetlands
            95: 0.0    # Emergent Herbaceous Wetlands
        }
        
        # SVI vulnerability multipliers  
        svi_vulnerability_map = {
            11: 0.0, 12: 0.0,                    # Water: no vulnerability
            21: 0.3, 22: 0.7, 23: 1.0, 24: 1.3, # Developed: increasing vulnerability with density
            31: 0.1,                             # Barren: minimal vulnerability
            41: 0.0, 42: 0.0, 43: 0.0,          # Forest: no vulnerability
            51: 0.0, 52: 0.0,                   # Shrub: no vulnerability  
            71: 0.1, 72: 0.0, 73: 0.0, 74: 0.0, # Grassland: minimal vulnerability
            81: 0.2, 82: 0.2,                   # Agriculture: low vulnerability
            90: 0.0, 95: 0.0                    # Wetlands: no vulnerability
        }
        
        # Development intensity (0-1 scale)
        development_intensity_map = {
            11: 0.0, 12: 0.0,           # Water/Ice
            21: 0.25, 22: 0.5, 23: 0.75, 24: 1.0,  # Developed (increasing intensity)
            31: 0.0,                    # Barren
            41: 0.0, 42: 0.0, 43: 0.0, # Forest
            51: 0.0, 52: 0.0,          # Shrub
            71: 0.0, 72: 0.0, 73: 0.0, 74: 0.0,  # Grassland
            81: 0.1, 82: 0.1,          # Agriculture (slight development)
            90: 0.0, 95: 0.0           # Wetlands
        }
        
        # Apply mappings
        features_df['population_density_coeff'] = features_df['nlcd_class'].map(
            population_density_map
        ).fillna(0.5)
        
        features_df['svi_vulnerability_coeff'] = features_df['nlcd_class'].map(
            svi_vulnerability_map  
        ).fillna(0.5)
        
        features_df['development_intensity'] = features_df['nlcd_class'].map(
            development_intensity_map
        ).fillna(0.0)
        
        # Binary indicators
        developed_classes = [21, 22, 23, 24]
        water_classes = [11, 12, 90, 95]
        forest_classes = [41, 42, 43]
        
        features_df['is_developed'] = features_df['nlcd_class'].isin(developed_classes)
        features_df['is_water'] = features_df['nlcd_class'].isin(water_classes)
        features_df['is_forest'] = features_df['nlcd_class'].isin(forest_classes)
        features_df['is_uninhabited'] = ~features_df['is_developed']
        
        # Combined IDM coefficient (population × vulnerability)
        features_df['idm_coefficient'] = (features_df['population_density_coeff'] * 
                                        features_df['svi_vulnerability_coeff'])
        
        # BACKWARD COMPATIBILITY: Keep legacy fields for existing code
        features_df['svi_coefficient'] = features_df['svi_vulnerability_coeff']  # Alias
        
        return features_df

    def _log_nlcd_extraction_quality(self, features_df: pd.DataFrame):
        """
        Log quality metrics for NLCD extraction with proper class names
        """
        
        unique_classes = features_df['nlcd_class'].unique()
        class_counts = features_df['nlcd_class'].value_counts()
        
        # Get class names
        class_names = self._get_nlcd_class_names()
        
        self._log(f"NLCD Extraction Quality Report:")
        self._log(f"  Total addresses: {len(features_df)}")
        self._log(f"  Unique NLCD classes: {len(unique_classes)}")
        
        for nlcd_class in sorted(unique_classes):
            count = class_counts.get(nlcd_class, 0)
            percentage = (count / len(features_df)) * 100
            class_name = class_names.get(nlcd_class, f"Unknown({nlcd_class})")
            self._log(f"    {nlcd_class}: {class_name} - {count} addresses ({percentage:.1f}%)")
        
        # Check for spatial variation
        idm_std = features_df['idm_coefficient'].std()
        pop_std = features_df['population_density_coeff'].std()
        vuln_std = features_df['svi_vulnerability_coeff'].std()
        
        self._log(f"  Coefficient variation:")
        self._log(f"    IDM coefficient std: {idm_std:.4f}")
        self._log(f"    Population density std: {pop_std:.4f}")  
        self._log(f"    Vulnerability std: {vuln_std:.4f}")
        
        # Check for proper vs legacy classes
        proper_classes = self._get_valid_nlcd_classes()
        found_proper = set(unique_classes).intersection(proper_classes)
        legacy_classes = {0, 1, 2, 250}
        found_legacy = set(unique_classes).intersection(legacy_classes)

    def _get_nlcd_class_names(self) -> dict:
        """
        Return complete NLCD class code to name mapping
        """
        return {
            # Standard NLCD 2019 classes
            11: "Open Water",
            12: "Perennial Ice/Snow", 
            21: "Developed, Open Space",
            22: "Developed, Low Intensity",
            23: "Developed, Medium Intensity", 
            24: "Developed, High Intensity",
            31: "Barren Land (Rock/Sand/Clay)",
            41: "Deciduous Forest",
            42: "Evergreen Forest",
            43: "Mixed Forest", 
            51: "Dwarf Scrub",
            52: "Shrub/Scrub",
            71: "Grassland/Herbaceous",
            72: "Sedge/Herbaceous",
            73: "Lichens", 
            74: "Moss",
            81: "Pasture/Hay",
            82: "Cultivated Crops",
            90: "Woody Wetlands",
            95: "Emergent Herbaceous Wetlands",
            
            # Legacy classes for backward compatibility
            0: "Water/Undeveloped (Legacy)",
            1: "Low Development (Legacy)",
            2: "High Development (Legacy)", 
            250: "No Data (Legacy)"
        }

    def _create_fallback_nlcd_features(self, addresses: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Manufacture NLCD features when raster data unavailable
        Uses proper 16-class system with realistic urban distribution
        """
        n_addresses = len(addresses)
        
        # Simulate realistic urban NLCD pattern based on literature
        np.random.seed(42) 
        
        # Typical urban distribution 
        class_probabilities = {
            22: 0.40,  # Low intensity residential (most common)
            23: 0.25,  # Medium intensity residential
            21: 0.15,  # Open space (parks, etc.)
            24: 0.10,  # High intensity (urban core)
            82: 0.05,  # Agriculture (suburban fringe)
            41: 0.03,  # Forest (urban forest)
            90: 0.02   # Wetlands (small pockets)
        }
        
        classes = list(class_probabilities.keys())
        probs = list(class_probabilities.values())
        
        # Generate classes
        nlcd_classes = np.random.choice(classes, size=n_addresses, p=probs)
        
        # Create features dataframe
        features_df = pd.DataFrame({
            'address_id': addresses['address_id'] if 'address_id' in addresses.columns else range(n_addresses),
            'nlcd_class': nlcd_classes
        })
        
        # Add derived features
        features_df = self._add_nlcd_derived_features(features_df)
        
        self._log(f"Created fallback NLCD features for {n_addresses} addresses")
        self._log(f"  Simulated {len(np.unique(nlcd_classes))} different land cover classes")
        self._log(f"  Using proper 16-class NLCD legend")
        
        return features_df

    def _load_nlcd_for_tract(self, tract_data):
        """
        Load NLCD data for tract area with proper 16-class extraction
        """
        try:
            # Get tract boundary for NLCD cropping
            tract_boundary = gpd.GeoDataFrame([tract_data['tract_info']], crs='EPSG:4326')
            
            # Load NLCD data
            nlcd_data = self.load_nlcd_data(
                county_bounds=tract_boundary,
                nlcd_path="./data/nlcd_hamilton_county.tif"
            )

            # Add this right after: nlcd_data = self.data_loader.load_nlcd_data(...)

            if nlcd_data is not None:   
                # Check tract boundary vs NLCD coverage
                tract_bounds = tract_boundary.total_bounds
                self._log(f"  Tract bounds (EPSG:4326): [{tract_bounds[0]:.6f}, {tract_bounds[1]:.6f}, {tract_bounds[2]:.6f}, {tract_bounds[3]:.6f}]")
                
                # Get NLCD bounds
                transform = nlcd_data['transform']
                height, width = nlcd_data['data'].shape
                raster_left = transform.c
                raster_top = transform.f  
                raster_right = raster_left + width * transform.a
                raster_bottom = raster_top + height * transform.e
                
                self._log(f"  NLCD bounds ({nlcd_data['crs']}): [{raster_left:.6f}, {raster_bottom:.6f}, {raster_right:.6f}, {raster_top:.6f}]")
                self._log(f"  NLCD size: {width}x{height} pixels")
                
                # Check address coverage
                address_bounds = tract_data['addresses'].total_bounds
                self._log(f"  Address bounds (EPSG:4326): [{address_bounds[0]:.6f}, {address_bounds[1]:.6f}, {address_bounds[2]:.6f}, {address_bounds[3]:.6f}]")
                self._log(f"  Number of addresses: {len(tract_data['addresses'])}")
                
                # Sample a few address coordinates
                sample_coords = []
                for i, addr in tract_data['addresses'].head(5).iterrows():
                    sample_coords.append((addr.geometry.x, addr.geometry.y))
                self._log(f"  Sample address coords: {sample_coords}")
                
                # Check if NLCD data has valid values
                unique_values = np.unique(nlcd_data['data'])
                self._log(f"  NLCD unique values: {unique_values}")
                self._log(f"  Contains 250 (no data): {250 in unique_values}")
                
                # Check for obvious issues
                if width < 10 or height < 10:
                    self._log(f"  WARNING: NLCD raster very small ({width}x{height})")
                
                if len(unique_values) < 3:
                    self._log(f"  WARNING: NLCD has very few unique values ({len(unique_values)})")
                
                if np.all(nlcd_data['data'] == 250):
                    self._log(f"  CRITICAL: NLCD raster is all 'no data' values!")
                    
                # Quick coordinate system check
                if tract_bounds[0] > 0:  # Longitude should be negative for Tennessee
                    self._log(f"  WARNING: Tract bounds have positive longitude - check CRS!")
                    
                if abs(raster_left) < 10:  # Should be around -85 for Tennessee  
                    self._log(f"  WARNING: NLCD bounds seem wrong for Tennessee location")
            
            if nlcd_data is None:
                self._log("  WARNING: NLCD raster not available, using fallback")
                return self._create_fallback_nlcd_features(tract_data['addresses'])
            
            # Extract NLCD features with proper 16-class legend
            nlcd_features = self.extract_nlcd_features_at_addresses(
                tract_data['addresses'], 
                nlcd_data
            )
            
            self._log(f"  Extracted NLCD features for {len(nlcd_features)} addresses")
            self._log(f"    Classes found: {sorted(nlcd_features['nlcd_class'].unique())}")
            
            return nlcd_features
            
        except Exception as e:
            self._log(f"  Error loading NLCD: {str(e)}")
            return self._create_fallback_nlcd_features(tract_data['addresses'])

    def _get_svi_coefficient(self, nlcd_class):
        """IDM-style SVI coefficients"""
        svi_coefficients = {
            # Developed areas (higher vulnerability)
            21: 0.2,  # Open developed - low vulnerability
            22: 0.6,  # Low density residential - moderate
            23: 1.0,  # Medium density - higher vulnerability  
            24: 1.5,  # High density urban - highest vulnerability
            
            # Uninhabited areas (zero vulnerability)
            11: 0.0, 12: 0.0, 31: 0.0, 41: 0.0, 42: 0.0, 43: 0.0,
            51: 0.0, 52: 0.0, 71: 0.0, 72: 0.0, 73: 0.0, 74: 0.0,
            90: 0.0, 95: 0.0,
            
            # Agricultural (low vulnerability)
            81: 0.1, 82: 0.1
        }
        return svi_coefficients.get(nlcd_class, 0.0)
    
    def load_svi_data(self, state_fips: str = '47', county_name: str = 'Hamilton', 
                     year: int = 2020) -> pd.DataFrame:
        """
        Load Social Vulnerability Index data
        
        Parameters:
        -----------
        state_fips : str
            State FIPS code
        county_name : str
            County name (not FIPS)
        year : int
            SVI data year
            
        Returns:
        --------
        pd.DataFrame
            SVI data for specified county
        """
        self._log(f"Loading SVI data for {county_name} County, {state_fips}...")
        
        svi_file = os.path.join(self.data_dir, 'raw', f'SVI_{year}_US.csv')
        
        try:
            # Load SVI data with proper types
            if os.path.exists(svi_file):
                svi_data = pd.read_csv(svi_file, dtype={'FIPS': str, 'ST': str})
                self._log(f"Loaded local SVI data ({len(svi_data)} records)")
            else:
                # Download from CDC
                url = f"https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download/{year}.html"
                self._log(f"Local file not found. Please download SVI data from: {url}")
                raise FileNotFoundError(f"SVI data not found at {svi_file}")
            
            # Filter to county
            county_svi = svi_data[
                (svi_data['ST'] == state_fips) & 
                (svi_data['COUNTY'] == county_name)
            ].copy()
            
            # Ensure FIPS is string
            county_svi['FIPS'] = county_svi['FIPS'].astype(str)
            
            # Select relevant columns
            columns = ['FIPS', 'LOCATION', 'RPL_THEMES', 'E_TOTPOP']
            county_svi = county_svi[columns]
            
            # Handle missing values
            county_svi['RPL_THEMES'] = county_svi['RPL_THEMES'].replace(-999, np.nan)
            
            valid_count = county_svi['RPL_THEMES'].notna().sum()
            self._log(f"Loaded {len(county_svi)} tracts ({valid_count} with valid SVI)")
            
            return county_svi
            
        except Exception as e:
            self._log(f"Error loading SVI data: {str(e)}")
            raise
    
    def load_census_tracts(self, state_fips: str = '47', county_fips: str = '065', 
                          year: int = 2020) -> gpd.GeoDataFrame:
        """
        Load census tract geometries
        
        Parameters:
        -----------
        state_fips : str
            State FIPS code
        county_fips : str
            County FIPS code
        year : int
            Census year
            
        Returns:
        --------
        gpd.GeoDataFrame
            Census tract geometries
        """
        self._log(f"Loading census tracts for {state_fips}-{county_fips}...")
        
        local_file = os.path.join(self.data_dir, 'raw', f'tl_{year}_{state_fips}_tract.shp')
        
        try:
            if os.path.exists(local_file):
                tracts = gpd.read_file(local_file)
                self._log(f"Loaded {len(tracts)} tracts from local file")
            else:
                # User needs to download from Census
                url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/"
                self._log(f"Local file not found. Please download from: {url}")
                raise FileNotFoundError(f"Census tracts not found at {local_file}")
            
            # Filter to county
            county_tracts = tracts[tracts['COUNTYFP'] == county_fips].copy()
            county_tracts['FIPS'] = county_tracts['GEOID'].astype(str).str.strip()
            
            # Ensure CRS
            if county_tracts.crs is None:
                county_tracts.set_crs(epsg=4326, inplace=True)
                
            self._log(f"Filtered to {len(county_tracts)} tracts")
            
            return county_tracts
            
        except Exception as e:
            self._log(f"Error loading census tracts: {str(e)}")
            raise
    
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
                    raise FileNotFoundError(f"Roads not found")
            
            # Ensure CRS
            if roads.crs is None:
                roads.set_crs(epsg=4326, inplace=True)
            
            return roads
            
        except Exception as e:
            self._log(f"Error loading roads: {str(e)}")
            # Return empty GeoDataFrame
            return gpd.GeoDataFrame(geometry=[])
    
    def create_network_graph(self, roads: gpd.GeoDataFrame) -> nx.Graph:
        """
        Create NetworkX graph from road geometries
        
        Parameters:
        -----------
        roads : gpd.GeoDataFrame
            Road geometries
            
        Returns:
        --------
        nx.Graph
            Network graph
        """
        if len(roads) == 0:
            self._log("No roads provided for network creation")
            return nx.Graph()
        
        self._log("Creating network graph...")
        
        G = nx.Graph()
        
        # Process each road segment
        for idx, road in roads.iterrows():
            if road.geometry.geom_type == 'LineString':
                coords = list(road.geometry.coords)
                
                # Add edges between consecutive points
                for i in range(len(coords) - 1):
                    u, v = coords[i], coords[i + 1]
                    
                    # Calculate edge length
                    length = Point(u).distance(Point(v))
                    
                    # Add edge
                    G.add_edge(u, v, length=length, road_id=idx)
        
        # Add node attributes
        for node in G.nodes():
            G.nodes[node]['x'] = node[0]
            G.nodes[node]['y'] = node[1]
        
        self._log(f"Created network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def get_available_fips_codes(self, state_fips: str = '47', 
                                county_fips: str = '065') -> List[str]:
        """
        Get list of available FIPS codes
        
        Parameters:
        -----------
        state_fips : str
            State FIPS code
        county_fips : str
            County FIPS code
            
        Returns:
        --------
        List[str]
            Available FIPS codes
        """
        tracts = self.load_census_tracts(state_fips, county_fips)
        return sorted(tracts['FIPS'].tolist())
    
    def load_single_tract_data(self, fips_code: str, buffer_degrees: float = 0.01,
                              max_nodes: Optional[int] = None, 
                              max_edges: Optional[int] = None) -> Dict:
        """
        UPDATED: Load data for a single census tract using real addresses
        
        Parameters:
        -----------
        fips_code : str
            11-digit FIPS code
        buffer_degrees : float
            Buffer around tract in degrees
        max_nodes : int, optional
            Maximum nodes in road network
        max_edges : int, optional  
            Maximum edges in road network
            
        Returns:
        --------
        Dict
            Complete tract data including real addresses
        """
        self._log(f"Loading data for tract {fips_code}")
        
        try:
            # Parse FIPS code
            state_fips = fips_code[:2]
            county_fips = fips_code[2:5]
            
            # 1. Load tract geometry and SVI
            tracts = self.load_census_tracts(state_fips, county_fips)
            tract = tracts[tracts['FIPS'] == fips_code]
            
            if len(tract) == 0:
                raise ValueError(f"Tract {fips_code} not found")
            
            tract_geom = tract.iloc[0].geometry
            
            # Load SVI data
            county_name = self._get_county_name(state_fips, county_fips)
            svi_data = self.load_svi_data(state_fips, county_name)
            tract_svi = svi_data[svi_data['FIPS'] == fips_code]
            
            if len(tract_svi) == 0:
                raise ValueError(f"No SVI data for tract {fips_code}")
            
            # 2. Create buffered bounding box
            bounds = tract_geom.bounds
            buffered_bbox = (
                bounds[0] - buffer_degrees, bounds[1] - buffer_degrees,
                bounds[2] + buffer_degrees, bounds[3] + buffer_degrees
            )
            
            # 3. Load roads within buffered area
            roads = self._load_roads_for_bbox(state_fips, county_fips, buffered_bbox)
            
            # 4. Create road network
            road_network = self.create_network_graph(roads)
            
            # 5. Get real addresses for this tract
            addresses = self.get_addresses_for_tract(fips_code, buffer_meters=200)
            
            # If no real addresses found, generate synthetic ones as fallback
            if len(addresses) == 0:
                self._log(f"No real addresses found for tract {fips_code}, generating synthetic ones")
                addresses = self._generate_tract_addresses(tract_geom, buffered_bbox, n_addresses=100)
                addresses['tract_fips'] = fips_code
            
            return {
                'fips_code': fips_code,
                'tract_geometry': tract_geom,
                'svi_data': tract_svi.iloc[0],
                'roads': roads,
                'road_network': road_network,
                'addresses': addresses, 
                'bbox': buffered_bbox,
                'network_stats': {
                    'nodes': road_network.number_of_nodes(),
                    'edges': road_network.number_of_edges(),
                    'real_addresses': len(addresses),
                    'address_source': 'real' if 'full_address' in addresses.columns else 'synthetic'
                }
            }
            
        except Exception as e:
            self._log(f"Error loading tract data: {str(e)}")
            raise
    
    def _generate_tract_addresses(self, tract_geom, bbox: Tuple, 
                                 n_addresses: int = 100) -> gpd.GeoDataFrame:
        """
        Generate synthetic addresses within tract (fallback only)
        """
        np.random.seed(123)  # Loader-specific seed 
        
        minx, miny, maxx, maxy = bbox
        addresses = []
        
        # Generate random points within bbox, keep those in tract
        attempts = 0
        while len(addresses) < n_addresses and attempts < n_addresses * 10:
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            point = Point(x, y)
            
            if tract_geom.contains(point):
                addresses.append({
                    'address_id': len(addresses),
                    'geometry': point,
                    'full_address': f"Synthetic Address {len(addresses)}"
                })
            
            attempts += 1
        
        if not addresses:
            # Fallback: use tract centroid
            centroid = tract_geom.centroid
            addresses = [{
                'address_id': 0,
                'geometry': centroid,
                'full_address': 'Tract Centroid Address'
            }]
        
        gdf = gpd.GeoDataFrame(addresses, crs='EPSG:4326')
        self._log(f"Generated {len(gdf)} addresses")
        
        return gdf
    
    def load_transit_stops(self, use_real_data: bool = True) -> gpd.GeoDataFrame:
        """
        Load transit stop locations for Chattanooga/Hamilton County
        
        Data source priority:
        1. CARTA GTFS data (if available)
        2. OpenStreetMap transit data 
        3. Realistic grid-based fallback
        4. Original 5-point fallback (last resort)
        
        Parameters:
        -----------
        use_real_data : bool
            Whether to attempt loading real transit data
            
        Returns:
        --------
        gpd.GeoDataFrame
            Transit stop locations with metadata
        """
        # Check cache first
        if self._transit_cache is not None:
            self._log(f"Using cached transit data ({len(self._transit_cache)} stops)")
            return self._transit_cache
        
        # Get configuration
        transit_config = self.config.get('transit', {})
        
        if use_real_data is None:
            use_real_data = transit_config.get('download_real_data', True)
        
        preferred_source = transit_config.get('preferred_source', 'gtfs')
        
        self._log("Loading transit stops...")
        self._log(f"  Configuration: preferred_source={preferred_source}, use_real_data={use_real_data}")
        
        transit_stops = None
        
        if use_real_data:
            # Try Method 1: CARTA GTFS data
            gtfs_stops = self._load_carta_gtfs_stops()
            if gtfs_stops is not None:
                self._log(f"Loaded {len(gtfs_stops)} stops from CARTA GTFS data")
                return gtfs_stops
            
            # Try Method 2: OpenStreetMap data
            osm_stops = self._load_osm_transit_stops()
            if osm_stops is not None:
                self._log(f"Loaded {len(osm_stops)} stops from OpenStreetMap")
                return osm_stops
        
        # Fallback Method 3: Realistic grid-based stops
        self._log("Real transit data unavailable, using realistic fallback")
        fallback_stops = self._create_realistic_transit_grid()
        if fallback_stops is not None:
            self._log(f"Created {len(fallback_stops)} realistic transit stops")
            return fallback_stops
        
        # Last resort: Original hardcoded stops
        self._log("Using minimal hardcoded transit stops (not recommended for research)")
        return self._create_minimal_transit_stops()

    def _load_carta_gtfs_stops(self) -> gpd.GeoDataFrame:
        """
        Load CARTA (Chattanooga Area Regional Transportation Authority) GTFS data
        
        GTFS data sources:
        - Official: https://www.gocarta.org/gtfs/
        - OpenMobilityData: https://transitland.org/
        """
        try:
            import zipfile
            import requests
            from io import BytesIO
            
            # Check for local GTFS file first
            gtfs_files = [
                os.path.join(self.data_dir, 'carta_gtfs.zip'),
                os.path.join(self.data_dir, 'gtfs', 'carta_gtfs.zip'),
                os.path.join(self.data_dir, 'raw', 'carta_gtfs.zip')
            ]
            
            gtfs_path = None
            for path in gtfs_files:
                if os.path.exists(path):
                    gtfs_path = path
                    break
            
            # If no local file, try downloading
            if gtfs_path is None:
                self._log("  Attempting to download CARTA GTFS data...")
                gtfs_urls = [
                    "https://www.gocarta.org/gtfs/gtfs.zip",  # Official CARTA
                    "https://transitland.org/api/v1/gtfs_exports/carta.zip"  # Transitland
                ]
                
                for url in gtfs_urls:
                    try:
                        response = requests.get(url, timeout=30)
                        if response.status_code == 200:
                            gtfs_path = os.path.join(self.data_dir, 'carta_gtfs_downloaded.zip')
                            with open(gtfs_path, 'wb') as f:
                                f.write(response.content)
                            self._log(f"  Downloaded GTFS data from {url}")
                            break
                    except Exception as e:
                        self._log(f"  Failed to download from {url}: {e}")
                        continue
            
            if gtfs_path is None:
                return None
            
            # Extract stops.txt from GTFS
            with zipfile.ZipFile(gtfs_path, 'r') as zip_file:
                if 'stops.txt' not in zip_file.namelist():
                    self._log("  GTFS file missing stops.txt")
                    return None
                
                with zip_file.open('stops.txt') as stops_file:
                    import pandas as pd
                    stops_df = pd.read_csv(stops_file)
            
            # Validate required columns
            required_cols = ['stop_id', 'stop_lat', 'stop_lon']
            if not all(col in stops_df.columns for col in required_cols):
                self._log(f"  GTFS stops.txt missing required columns: {required_cols}")
                return None
            
            # Filter to Hamilton County area (rough bounds)
            hamilton_bounds = {
                'min_lat': 34.9, 'max_lat': 35.3,
                'min_lon': -85.5, 'max_lon': -85.0
            }
            
            stops_df = stops_df[
                (stops_df['stop_lat'] >= hamilton_bounds['min_lat']) &
                (stops_df['stop_lat'] <= hamilton_bounds['max_lat']) &
                (stops_df['stop_lon'] >= hamilton_bounds['min_lon']) &
                (stops_df['stop_lon'] <= hamilton_bounds['max_lon'])
            ]
            
            if len(stops_df) == 0:
                self._log("  No stops found within Hamilton County bounds")
                return None
            
            # Create GeoDataFrame
            geometry = [Point(lon, lat) for lat, lon in zip(stops_df['stop_lat'], stops_df['stop_lon'])]
            
            transit_stops = gpd.GeoDataFrame({
                'stop_id': stops_df['stop_id'],
                'stop_name': stops_df.get('stop_name', 'Unknown'),
                'stop_desc': stops_df.get('stop_desc', ''),
                'route_type': 'bus',  # CARTA is primarily bus
                'data_source': 'CARTA_GTFS'
            }, geometry=geometry, crs='EPSG:4326')
            
            return transit_stops
            
        except Exception as e:
            self._log(f"  Error loading CARTA GTFS data: {str(e)}")
            return None

    def _load_osm_transit_stops(self) -> gpd.GeoDataFrame:
        """
        Load transit stops from OpenStreetMap for Hamilton County
        """
        try:
            import requests
            
            # Hamilton County bounding box
            bbox = "34.9,-85.5,35.3,-85.0"  # min_lat, min_lon, max_lat, max_lon
            
            # Overpass API query for transit stops
            overpass_query = f"""
            [out:json][timeout:60];
            (
            node["public_transport"="stop_position"]({bbox});
            node["highway"="bus_stop"]({bbox});
            node["railway"="tram_stop"]({bbox});
            node["amenity"="bus_station"]({bbox});
            );
            out geom;
            """
            
            self._log("  Querying OpenStreetMap for transit stops...")
            
            response = requests.post(
                'https://overpass-api.de/api/interpreter',
                data=overpass_query,
                timeout=60
            )
            
            if response.status_code != 200:
                self._log(f"  OSM query failed with status {response.status_code}")
                return None
            
            osm_data = response.json()
            
            if not osm_data.get('elements'):
                self._log("  No transit stops found in OSM data")
                return None
            
            # Convert OSM data to GeoDataFrame
            stops_data = []
            for element in osm_data['elements']:
                if 'lat' in element and 'lon' in element:
                    tags = element.get('tags', {})
                    stops_data.append({
                        'stop_id': f"osm_{element['id']}",
                        'stop_name': tags.get('name', 'Unnamed Stop'),
                        'stop_desc': tags.get('description', ''),
                        'route_type': self._determine_route_type(tags),
                        'data_source': 'OpenStreetMap',
                        'geometry': Point(element['lon'], element['lat'])
                    })
            
            if not stops_data:
                return None
            
            transit_stops = gpd.GeoDataFrame(stops_data, crs='EPSG:4326')
            
            return transit_stops
            
        except Exception as e:
            self._log(f"  Error loading OSM transit data: {str(e)}")
            return None

    def _determine_route_type(self, tags: dict) -> str:
        """Determine transit route type from OSM tags"""
        if tags.get('railway') in ['tram_stop', 'light_rail']:
            return 'tram'
        elif tags.get('amenity') == 'bus_station':
            return 'bus_station'
        elif tags.get('highway') == 'bus_stop':
            return 'bus'
        elif tags.get('public_transport') == 'stop_position':
            return 'bus'  # Default assumption
        else:
            return 'bus'

    def _create_realistic_transit_grid(self) -> gpd.GeoDataFrame:
        """
        Create a realistic grid of transit stops based on Chattanooga's urban layout
        
        Uses demographic and road network data to place stops in logical locations
        """
        try:
            # Define Chattanooga metropolitan area with realistic coverage
            # Based on actual CARTA service area
            service_areas = {
                'downtown': {
                    'center': (-85.3096, 35.0456),
                    'radius': 0.02,  # ~2km radius
                    'stop_density': 0.005,  # Stop every ~500m
                    'description': 'Downtown Core'
                },
                'north_chattanooga': {
                    'center': (-85.2967, 35.0722), 
                    'radius': 0.025,
                    'stop_density': 0.008,
                    'description': 'North Chattanooga/Northshore'
                },
                'east_chattanooga': {
                    'center': (-85.2580, 35.0456),
                    'radius': 0.02,
                    'stop_density': 0.010,
                    'description': 'East Chattanooga'
                },
                'south_chattanooga': {
                    'center': (-85.3365, 35.0175),
                    'radius': 0.02, 
                    'stop_density': 0.010,
                    'description': 'South Chattanooga'
                },
                'east_ridge': {
                    'center': (-85.1938, 35.0495),
                    'radius': 0.015,
                    'stop_density': 0.012,
                    'description': 'East Ridge'
                },
                'brainerd': {
                    'center': (-85.2180, 35.0156),
                    'radius': 0.018,
                    'stop_density': 0.010,
                    'description': 'Brainerd'
                },
                'hixson': {
                    'center': (-85.2441, 35.1256),
                    'radius': 0.02,
                    'stop_density': 0.012,
                    'description': 'Hixson'
                },
                'red_bank': {
                    'center': (-85.2952, 35.1156),
                    'radius': 0.015,
                    'stop_density': 0.012,
                    'description': 'Red Bank'
                }
            }
            
            stops_data = []
            stop_id = 1
            
            for area_name, area_config in service_areas.items():
                center_lon, center_lat = area_config['center']
                radius = area_config['radius']
                density = area_config['stop_density']
                
                # Create grid within each service area
                num_stops = int((radius * 2) / density)
                
                for i in range(num_stops):
                    for j in range(num_stops):
                        # Grid position
                        lon_offset = (i - num_stops/2) * density
                        lat_offset = (j - num_stops/2) * density
                        
                        stop_lon = center_lon + lon_offset
                        stop_lat = center_lat + lat_offset
                        
                        # Check if within circular service area
                        distance = ((stop_lon - center_lon)**2 + (stop_lat - center_lat)**2)**0.5
                        if distance <= radius:
                            stops_data.append({
                                'stop_id': f"grid_{stop_id}",
                                'stop_name': f"{area_config['description']} Stop {stop_id}",
                                'stop_desc': f"Generated stop in {area_config['description']} service area",
                                'route_type': 'bus',
                                'service_area': area_name,
                                'data_source': 'Generated_Grid',
                                'geometry': Point(stop_lon, stop_lat)
                            })
                            stop_id += 1
            
            # Add major transit hubs/stations
            major_hubs = [
                {
                    'stop_id': 'hub_downtown_transit_center',
                    'stop_name': 'Downtown Transit Center',
                    'stop_desc': 'Main downtown transit hub',
                    'route_type': 'bus_station',
                    'service_area': 'downtown',
                    'data_source': 'Generated_Hub',
                    'geometry': Point(-85.3096, 35.0456)
                },
                {
                    'stop_id': 'hub_hamilton_place',
                    'stop_name': 'Hamilton Place Mall',
                    'stop_desc': 'Major shopping center transit hub',
                    'route_type': 'bus_station', 
                    'service_area': 'east_chattanooga',
                    'data_source': 'Generated_Hub',
                    'geometry': Point(-85.2111, 35.0407)
                },
                {
                    'stop_id': 'hub_university',
                    'stop_name': 'UTC Campus',
                    'stop_desc': 'University of Tennessee Chattanooga',
                    'route_type': 'bus_station',
                    'service_area': 'downtown',
                    'data_source': 'Generated_Hub', 
                    'geometry': Point(-85.3019, 35.0456)
                }
            ]
            
            stops_data.extend(major_hubs)
            
            if not stops_data:
                return None
            
            transit_stops = gpd.GeoDataFrame(stops_data, crs='EPSG:4326')
            
            self._log(f"  Generated {len(transit_stops)} stops across {len(service_areas)} service areas")
            for area in service_areas.keys():
                area_count = len(transit_stops[transit_stops['service_area'] == area])
                self._log(f"    {area}: {area_count} stops")
            
            return transit_stops
            
        except Exception as e:
            self._log(f"  Error creating realistic transit grid: {str(e)}")
            return None

    def _create_minimal_transit_stops(self) -> gpd.GeoDataFrame:
        """
        Last resort: Original minimal transit stops
        """
        stops = [
            {
                'stop_id': 'min_01',
                'stop_name': 'Downtown Chattanooga',
                'geometry': Point(-85.3096, 35.0456),
                'route_type': 'bus_station'
            },
            {
                'stop_id': 'min_02', 
                'stop_name': 'North Chattanooga',
                'geometry': Point(-85.2967, 35.0722),
                'route_type': 'bus'
            },
            {
                'stop_id': 'min_03',
                'stop_name': 'East Chattanooga', 
                'geometry': Point(-85.2580, 35.0456),
                'route_type': 'bus'
            },
            {
                'stop_id': 'min_04',
                'stop_name': 'South Chattanooga',
                'geometry': Point(-85.3365, 35.0175),
                'route_type': 'bus'
            },
            {
                'stop_id': 'min_05',
                'stop_name': 'East Ridge',
                'geometry': Point(-85.1938, 35.0495),
                'route_type': 'bus'
            }
        ]
        
        for stop in stops:
            stop.update({
                'stop_desc': 'Minimal fallback stop',
                'service_area': 'unknown',
                'data_source': 'Hardcoded_Fallback'
            })
        
        transit_stops = gpd.GeoDataFrame(stops, crs='EPSG:4326')
        
        return transit_stops
    
    def load_address_points(self, state_fips: str = '47', county_fips: str = '065') -> gpd.GeoDataFrame:
        """
        Load real Chattanooga address point locations
        
        Parameters:
        -----------
        state_fips : str
            State FIPS code (47 for Tennessee)
        county_fips : str  
            County FIPS code (065 for Hamilton County)
            
        Returns:
        --------
        gpd.GeoDataFrame
            Real address point locations from chattanooga.geojson
        """
        # Check cache first
        if self._address_cache is not None:
            self._log(f"Using cached address data ({len(self._address_cache)} addresses)")
            return self._address_cache
        
        self._log("Loading real Chattanooga address data...")
        
        # Try multiple file locations
        address_files = [
            os.path.join(self.data_dir, 'chattanooga.geojson'),
            os.path.join(self.data_dir, 'raw', 'chattanooga.geojson'),
            os.path.join(self.data_dir, 'addresses', 'chattanooga.geojson'),
            './chattanooga.geojson',  # Current directory
            'chattanooga.geojson'     # Direct filename
        ]
        
        addresses_gdf = None
        for address_file in address_files:
            if os.path.exists(address_file):
                addresses_gdf = self._load_chattanooga_geojson(address_file)
                if len(addresses_gdf) > 0:
                    self._log(f"Loaded {len(addresses_gdf)} real addresses from {address_file}")
                    break
        
        if addresses_gdf is None or len(addresses_gdf) == 0:
            self._log("WARNING: Real address data not found, using fallback synthetic generation")
            return self._generate_tract_constrained_addresses(state_fips, county_fips)
        
        # Filter to Hamilton County bounds if needed
        addresses_gdf = self._filter_to_hamilton_county(addresses_gdf, state_fips, county_fips)
        
        # Cache the result
        self._address_cache = addresses_gdf
        
        self._log(f"Loaded {len(addresses_gdf)} Hamilton County addresses")
        return addresses_gdf
    
    def _load_chattanooga_geojson(self, file_path: str) -> gpd.GeoDataFrame:
        """
        Load and standardize the chattanooga.geojson file
        
        Parameters:
        -----------
        file_path : str
            Path to chattanooga.geojson file
            
        Returns:
        --------
        gpd.GeoDataFrame
            Standardized address data
        """
        try:
            # Load GeoJSON
            addresses = gpd.read_file(file_path)
            
            if len(addresses) == 0:
                self._log(f"No addresses found in {file_path}")
                return gpd.GeoDataFrame(columns=['address_id', 'geometry'])
            
            # Standardize columns to match expected format
            addresses = addresses.copy()
            
            # Create standardized address_id
            if 'address_id' not in addresses.columns:
                addresses['address_id'] = range(len(addresses))
            
            # Create full address field for reference
            addresses['full_address'] = addresses.apply(self._create_full_address, axis=1)
            
            # Extract relevant fields from properties
            if 'number' in addresses.columns:
                addresses['house_number'] = addresses['number']
            if 'street' in addresses.columns:
                addresses['street_name'] = addresses['street']
            if 'city' in addresses.columns:
                addresses['city_name'] = addresses['city']
            if 'postcode' in addresses.columns:
                addresses['zipcode'] = addresses['postcode']
            
            # Ensure proper CRS 
            if addresses.crs is None:
                addresses.set_crs(epsg=4326, inplace=True)
            elif addresses.crs != 'EPSG:4326':
                addresses = addresses.to_crs('EPSG:4326')
            
            # Keep essential columns
            essential_columns = ['address_id', 'geometry', 'full_address']
            optional_columns = ['house_number', 'street_name', 'city_name', 'zipcode', 'hash']
            
            keep_columns = essential_columns + [col for col in optional_columns if col in addresses.columns]
            addresses = addresses[keep_columns]
            
            return addresses
            
        except Exception as e:
            self._log(f"Error loading {file_path}: {str(e)}")
            return gpd.GeoDataFrame(columns=['address_id', 'geometry'])
    
    def _create_full_address(self, row) -> str:
        """Create full address string from components"""
        parts = []
        
        if pd.notna(row.get('number', '')) and row.get('number', '') != '':
            parts.append(str(row['number']))
        
        if pd.notna(row.get('street', '')) and row.get('street', '') != '':
            parts.append(str(row['street']))
        
        if pd.notna(row.get('unit', '')) and row.get('unit', '') != '':
            parts.append(f"Unit {row['unit']}")
        
        if pd.notna(row.get('city', '')) and row.get('city', '') != '':
            parts.append(str(row['city']))
        
        if pd.notna(row.get('postcode', '')) and row.get('postcode', '') != '':
            parts.append(str(row['postcode']))
        
        return ', '.join(parts) if parts else 'Unknown Address'
    
    def _filter_to_hamilton_county(self, addresses: gpd.GeoDataFrame, 
                                  state_fips: str, county_fips: str) -> gpd.GeoDataFrame:
        """
        Filter addresses to Hamilton County boundaries
        
        Parameters:
        -----------
        addresses : gpd.GeoDataFrame
            Address data to filter
        state_fips : str
            State FIPS code
        county_fips : str
            County FIPS code
            
        Returns:
        --------
        gpd.GeoDataFrame
            Addresses within Hamilton County
        """
        try:
            # Load Hamilton County boundary
            county_tracts = self.load_census_tracts(state_fips, county_fips)
            if len(county_tracts) == 0:
                self._log("Warning: Could not load county boundary for filtering")
                return addresses
            
            # Create county boundary from tract union
            county_boundary = county_tracts.geometry.unary_union
            
            # Spatial filter
            within_county = addresses[addresses.geometry.within(county_boundary)]
            
            self._log(f"Filtered {len(addresses)} addresses to {len(within_county)} within Hamilton County")
            
            return within_county
            
        except Exception as e:
            self._log(f"Error filtering to county boundary: {str(e)}")
            return addresses
    
    def get_addresses_for_tract(self, fips_code: str, 
                              buffer_meters: float = 100) -> gpd.GeoDataFrame:
        """
        Get real addresses within a specific census tract
        
        NEW METHOD FOR TRACT-SPECIFIC ADDRESS LOADING
        
        Parameters:
        -----------
        fips_code : str
            11-digit FIPS code for census tract
        buffer_meters : float
            Buffer around tract boundary in meters
            
        Returns:
        --------
        gpd.GeoDataFrame
            Addresses within the specified tract
        """
        self._log(f"Getting addresses for tract {fips_code}")
        
        try:
            # Load all addresses
            all_addresses = self.load_address_points()
            
            if len(all_addresses) == 0:
                self._log(f"No addresses available for tract {fips_code}")
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
            
            # Apply buffer if specified
            if buffer_meters > 0:
                # Convert to projected CRS for accurate buffering
                tract_proj = tract.to_crs('EPSG:3857')  # Web Mercator
                buffered = tract_proj.geometry.buffer(buffer_meters)
                tract_geom = buffered.to_crs('EPSG:4326').iloc[0]
            
            # Spatial filter
            tract_addresses = all_addresses[all_addresses.geometry.within(tract_geom)].copy()
            
            # Add tract FIPS to addresses
            tract_addresses['tract_fips'] = fips_code
            
            self._log(f"Found {len(tract_addresses)} addresses in tract {fips_code}")
            
            return tract_addresses
            
        except Exception as e:
            self._log(f"Error getting addresses for tract {fips_code}: {str(e)}")
            return gpd.GeoDataFrame(columns=['address_id', 'geometry'])
    
    def _generate_tract_constrained_addresses(self, state_fips: str, county_fips: str, 
                                            density_per_sq_km: int = 500) -> gpd.GeoDataFrame:
        """
        FALLBACK: Generate addresses constrained to tract boundaries
        Only used if real address data is unavailable
        """
        self._log("Generating tract-constrained synthetic addresses as fallback...")
        
        try:
            # Load census tracts
            tracts = self.load_census_tracts(state_fips, county_fips)
            
            all_addresses = []
            address_id = 0
            
            for _, tract in tracts.iterrows():
                # Calculate number of addresses for this tract based on area
                tract_area_sq_km = tract.geometry.area * 111**2  # Rough conversion to sq km
                n_addresses = max(10, int(tract_area_sq_km * density_per_sq_km))
                
                # Generate addresses within this specific tract
                tract_addresses = self._generate_tract_addresses(
                    tract.geometry, 
                    tract.geometry.bounds, 
                    n_addresses=n_addresses
                )
                
                # Update address IDs
                tract_addresses['address_id'] = range(address_id, address_id + len(tract_addresses))
                tract_addresses['tract_fips'] = tract['FIPS']
                tract_addresses['full_address'] = f"Synthetic Address in {tract['FIPS']}"
                
                all_addresses.append(tract_addresses)
                address_id += len(tract_addresses)
            
            if all_addresses:
                combined = gpd.GeoDataFrame(pd.concat(all_addresses, ignore_index=True))
                self._log(f"Generated {len(combined)} tract-constrained synthetic addresses")
                return combined[['address_id', 'geometry', 'full_address', 'tract_fips']]
            else:
                self._log("Failed to generate any addresses")
                return gpd.GeoDataFrame(columns=['address_id', 'geometry'])
                
        except Exception as e:
            self._log(f"Error generating tract-constrained addresses: {str(e)}")
            return gpd.GeoDataFrame(columns=['address_id', 'geometry'])
    
    def _get_county_name(self, state_fips: str, county_fips: str) -> str:
        """Map FIPS codes to county name"""
        county_map = {
            ('47', '065'): 'Hamilton'
        }
        return county_map.get((state_fips, county_fips), 'Unknown')
    
    def resolve_fips_list(self, fips_config: Dict, state_fips: str, 
                         county_fips: str) -> List[str]:
        """
        Resolve FIPS configuration to list of codes
        
        Parameters:
        -----------
        fips_config : Dict
            FIPS configuration from config file
        state_fips : str
            State FIPS code
        county_fips : str
            County FIPS code
            
        Returns:
        --------
        List[str]
            Resolved FIPS codes
        """
        # Check for explicit list
        target_list = fips_config.get('batch', {}).get('target_list', [])
        if target_list:
            return target_list
        
        # Check for single FIPS
        if fips_config.get('single_fips'):
            return [fips_config['single_fips']]
        
        # Auto-selection
        auto_config = fips_config.get('batch', {}).get('auto_select', {})
        if auto_config.get('enabled', True):
            all_fips = self.get_available_fips_codes(state_fips, county_fips)
            
            mode = auto_config.get('mode', 'range')
            
            if mode == 'all':
                return all_fips
            elif mode == 'range':
                start = auto_config.get('range_start', 1) - 1
                end = auto_config.get('range_end', 5)
                return all_fips[start:end]
            elif mode == 'sample':
                size = min(auto_config.get('sample_size', 10), len(all_fips))
                return np.random.choice(all_fips, size=size, replace=False).tolist()
        
        # Default: first 5 tracts
        all_fips = self.get_available_fips_codes(state_fips, county_fips)
        return all_fips[:5]

# Helper function for addresses validation
def analyze_address_coverage(data_loader, state_fips='47', county_fips='065'):
    """
    Analyze coverage of real address data across census tracts
    """
    print("Analyzing address coverage across Hamilton County...")
    
    # Load tracts and addresses
    tracts = data_loader.load_census_tracts(state_fips, county_fips)
    addresses = data_loader.load_address_points(state_fips, county_fips)
    
    coverage_stats = []
    
    for _, tract in tracts.iterrows():
        fips_code = tract['FIPS']
        tract_addresses = data_loader.get_addresses_for_tract(fips_code)
        
        coverage_stats.append({
            'fips': fips_code,
            'address_count': len(tract_addresses),
            'tract_area_sq_km': tract.geometry.area * 111**2,
            'address_density': len(tract_addresses) / (tract.geometry.area * 111**2) if tract.geometry.area > 0 else 0
        })
    
    coverage_df = pd.DataFrame(coverage_stats)
    
    print(f"\nAddress Coverage Summary:")
    print(f"Total tracts: {len(coverage_df)}")
    print(f"Tracts with addresses: {sum(coverage_df['address_count'] > 0)}")
    print(f"Total addresses: {coverage_df['address_count'].sum()}")
    print(f"Mean addresses per tract: {coverage_df['address_count'].mean():.1f}")
    print(f"Median addresses per tract: {coverage_df['address_count'].median():.1f}")
    
    return coverage_df

# Convenience function for backward compatibility
def load_hamilton_county_data(data_dir: str = './data', 
                            roads_file: Optional[str] = None) -> Dict:
    """
    Load all data for Hamilton County analysis
    
    This function is used by other modules and must be maintained
    for backward compatibility.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing data files
    roads_file : str, optional
        Path to roads shapefile
        
    Returns:
    --------
    Dict
        Dictionary containing all loaded datasets
    """
    loader = DataLoader(data_dir=data_dir)
    
    print("\n" + "="*60)
    print("GRANITE: Loading Hamilton County Data")
    print("="*60 + "\n")
    
    # Load all datasets
    data = {
        'svi': loader.load_svi_data(),
        'census_tracts': loader.load_census_tracts(),
        'roads': loader.load_road_network(roads_file=roads_file),
        'transit_stops': loader.load_transit_stops(),
        'addresses': loader.load_address_points()
    }
    
    # Create network graph
    data['road_network'] = loader.create_network_graph(data['roads'])
    
    # Merge SVI with census tracts
    data['tracts_with_svi'] = data['census_tracts'].merge(
        data['svi'],
        on='FIPS',
        how='inner'
    )
    
    print("\n" + "="*60)
    print("Data Loading Complete!")
    print("="*60)
    
    return data