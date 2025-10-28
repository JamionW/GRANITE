"""
OSRM Routing Integration for GRANITE
Auto-detects and uses local OSRM servers (no config needed)
"""
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import subprocess
import time
import warnings
warnings.filterwarnings('ignore')

class OSRMRouter:
    """
    OSRM routing client - automatically uses local servers
    """
    
    def __init__(self, batch_size=100, verbose=True):
        """
        Args:
            batch_size: Maximum origins/destinations per batch (OSRM limit ~100)
            verbose: Enable logging
        """
        self.driving_url = 'http://localhost:5000'
        self.walking_url = 'http://localhost:5001'
        self.batch_size = batch_size
        self.verbose = verbose
        
        # Ensure OSRM servers are running
        self._ensure_osrm_running()
    
    def log(self, message):
        if self.verbose:
            print(f"[OSRMRouter] {message}")
    
    def _ensure_osrm_running(self):
        """
        Check if OSRM servers are running, start them if not
        """
        driving_ok = self._test_server(self.driving_url, 'driving')
        walking_ok = self._test_server(self.walking_url, 'foot')
        
        if not (driving_ok and walking_ok):
            self.log("OSRM servers not running, attempting to start...")
            try:
                subprocess.run(['bash', '/workspaces/GRANITE/scripts/start_osrm.sh'], 
                             check=True, capture_output=True, timeout=30)
                time.sleep(3)
                
                # Verify again
                driving_ok = self._test_server(self.driving_url, 'driving')
                walking_ok = self._test_server(self.walking_url, 'foot')
                
                if driving_ok and walking_ok:
                    self.log("✓ OSRM servers started successfully")
                else:
                    self.log("⚠ Could not start OSRM servers, will use fallback estimates")
            except Exception as e:
                self.log(f"⚠ Could not auto-start OSRM: {e}")
                self.log("Will use distance-based fallback estimates")
    
    def _test_server(self, base_url, profile):
        """Test if OSRM server is accessible"""
        try:
            test_coords = "-85.3097,35.0456;-85.2111,35.0407"
            url = f"{base_url}/route/v1/{profile}/{test_coords}"
            response = requests.get(url, timeout=3)
            
            if response.status_code == 200:
                self.log(f"✓ {profile.title()} server ready: {base_url}")
                return True
            else:
                return False
        except:
            return False
    
    def compute_travel_time_matrix(self, origins: gpd.GeoDataFrame, 
                                   destinations: gpd.GeoDataFrame,
                                   mode='driving') -> np.ndarray:
        """
        Compute all-to-all travel times using OSRM Table API
        
        Args:
            origins: GeoDataFrame with origin points
            destinations: GeoDataFrame with destination points
            mode: 'driving' or 'foot' (walking)
        
        Returns:
            Matrix [n_origins, n_destinations] with travel times in minutes
        """
        base_url = self.driving_url if mode == 'driving' else self.walking_url
        profile = 'driving' if mode == 'driving' else 'foot'
        
        self.log(f"Computing {mode} travel times: {len(origins)} origins × {len(destinations)} destinations")
        
        n_origins = len(origins)
        n_destinations = len(destinations)
        
        travel_time_matrix = np.zeros((n_origins, n_destinations))
        
        # Process in batches
        total_batches = 0
        for i in range(0, n_origins, self.batch_size):
            for j in range(0, n_destinations, self.batch_size):
                
                origin_batch = origins.iloc[i:i+self.batch_size]
                dest_batch = destinations.iloc[j:j+self.batch_size]
                
                # Compute batch
                batch_matrix = self._compute_batch(origin_batch, dest_batch, base_url, profile)
                
                # Fill into full matrix
                i_end = min(i + self.batch_size, n_origins)
                j_end = min(j + self.batch_size, n_destinations)
                travel_time_matrix[i:i_end, j:j_end] = batch_matrix
                
                total_batches += 1
                
                if total_batches % 10 == 0:
                    self.log(f"  Processed {total_batches} batches...")
                
                time.sleep(0.05)  # Brief pause
        
        self.log(f"  Completed: {total_batches} batches, {n_origins * n_destinations:,} route pairs")
        
        return travel_time_matrix
    
    def _compute_batch(self, origins: gpd.GeoDataFrame, 
                      destinations: gpd.GeoDataFrame,
                      base_url: str,
                      profile: str) -> np.ndarray:
        """
        Compute single batch using OSRM Table API
        """
        try:
            # Build coordinate string
            origin_coords = [(row.geometry.x, row.geometry.y) for _, row in origins.iterrows()]
            dest_coords = [(row.geometry.x, row.geometry.y) for _, row in destinations.iterrows()]
            all_coords = origin_coords + dest_coords
            
            coords_str = ";".join([f"{lon},{lat}" for lon, lat in all_coords])
            
            # Specify which are sources vs destinations
            sources_str = ",".join(str(k) for k in range(len(origin_coords)))
            dests_str = ",".join(str(k + len(origin_coords)) for k in range(len(dest_coords)))
            
            # Make API call
            url = f"{base_url}/table/v1/{profile}/{coords_str}"
            params = {
                'sources': sources_str,
                'destinations': dests_str
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'durations' in data:
                    # Convert seconds to minutes
                    batch_matrix = np.array(data['durations']) / 60.0
                    return batch_matrix
                else:
                    return self._fallback_batch(origins, destinations)
            else:
                return self._fallback_batch(origins, destinations)
                
        except Exception as e:
            if self.verbose:
                self.log(f"OSRM batch failed, using fallback: {e}")
            return self._fallback_batch(origins, destinations)
    
    def _fallback_batch(self, origins: gpd.GeoDataFrame, 
                       destinations: gpd.GeoDataFrame) -> np.ndarray:
        """
        Fallback to distance-based estimates if OSRM fails
        """
        from geopy.distance import geodesic
        
        batch_matrix = np.zeros((len(origins), len(destinations)))
        
        for i, (_, origin) in enumerate(origins.iterrows()):
            origin_coords = (origin.geometry.y, origin.geometry.x)
            
            for j, (_, dest) in enumerate(destinations.iterrows()):
                dest_coords = (dest.geometry.y, dest.geometry.x)
                
                distance_km = geodesic(origin_coords, dest_coords).km
                network_distance = distance_km * 1.3
                speed_kmh = 25.0
                travel_time_min = (network_distance / speed_kmh) * 60
                
                batch_matrix[i, j] = travel_time_min
        
        return batch_matrix
    
    def compute_multimodal_travel_times(self, origins: gpd.GeoDataFrame,
                                       destinations: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Compute travel times for multiple modes
        
        Returns:
            DataFrame with columns: origin_id, dest_id, walk_time, drive_time, combined_time, best_mode
        """
        self.log("Computing multimodal travel times...")
        
        # Compute driving times
        drive_matrix = self.compute_travel_time_matrix(origins, destinations, mode='driving')
        
        # Compute walking times
        walk_matrix = self.compute_travel_time_matrix(origins, destinations, mode='foot')
        
        # Build results dataframe
        results = []
        
        for i, (_, origin) in enumerate(origins.iterrows()):
            origin_id = origin.get('address_id', i)
            
            for j, (_, dest) in enumerate(destinations.iterrows()):
                dest_id = dest.get('dest_id', j)
                
                walk_time = walk_matrix[i, j]
                drive_time = drive_matrix[i, j]
                
                # Choose best mode
                if walk_time <= 15:  # Walkable
                    best_mode = 'walk'
                    combined_time = walk_time
                elif drive_time < walk_time * 0.5:  # Driving much faster
                    best_mode = 'drive'
                    combined_time = drive_time + 3  # Add parking/access time
                else:
                    best_mode = 'walk'
                    combined_time = walk_time
                
                results.append({
                    'origin_id': origin_id,
                    'dest_id': dest_id,
                    'walk_time': walk_time,
                    'drive_time': drive_time,
                    'combined_time': combined_time,
                    'best_mode': best_mode
                })
        
        return pd.DataFrame(results)