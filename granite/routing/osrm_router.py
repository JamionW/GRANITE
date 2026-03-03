"""
OSRM Router

Provides network-based routing using OSRM servers for driving and walking modes.
"""
import requests
import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, List, Tuple
import time

class OSRMRouter:
    """Routes using actual OSRM servers instead of fallback calculations"""
    
    def __init__(self, 
                 driving_url: str = "http://localhost:5000",
                 walking_url: str = "http://localhost:5001",
                 batch_size: int = 100,
                 verbose: bool = False):
        self.driving_url = driving_url
        self.walking_url = walking_url
        self.batch_size = batch_size
        self.verbose = verbose
        
        # Verify servers are running
        self._verify_servers()
    
    def log(self, message):
        if self.verbose:
            print(f"[OSRMRouter] {message}")
    
    def _verify_servers(self):
        """Check that OSRM servers are accessible"""
        for name, url, profile in [("Driving", self.driving_url, "driving"), 
                                     ("Foot", self.walking_url, "foot")]:
            try:
                # Test with a simple route query
                test_url = f"{url}/route/v1/{profile}/-85.3,35.0;-85.2,35.0?overview=false"
                response = requests.get(test_url, timeout=5)
                if response.status_code == 200:
                    self.log(f"{name} server ready: {url}")
                else:
                    raise ConnectionError(f"{name} server returned status {response.status_code}")
            except requests.exceptions.ConnectionError:
                raise ConnectionError(
                    f"Cannot connect to {name} server at {url}. "
                    f"Start OSRM with: docker run -t -i -p {url.split(':')[-1]}:5000 -v $(pwd):/data "
                    f"osrm/osrm-backend osrm-routed --algorithm mld /data/tennessee-latest.osrm"
                )
    
    def compute_multimodal_travel_times(self,
                                       origins: gpd.GeoDataFrame,
                                       destinations: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Compute travel times using actual OSRM routing
        
        Args:
            origins: GeoDataFrame with origin points
            destinations: GeoDataFrame with destination points
            
        Returns:
            DataFrame with columns: origin_id, dest_id, walk_time, drive_time, 
                                   combined_time, best_mode
        """
        self.log(f"Computing multimodal travel times...")
        self.log(f"Computing driving travel times: {len(origins)} origins × {len(destinations)} destinations")
        
        # Compute driving times
        drive_results = self._compute_mode_times(origins, destinations, mode='driving')
        
        self.log(f"Computing foot travel times: {len(origins)} origins × {len(destinations)} destinations")
        
        # Compute walking times
        walk_results = self._compute_mode_times(origins, destinations, mode='foot')
        
        # Merge results
        results = []
        for (orig_id, dest_id), drive_time in drive_results.items():
            walk_time = walk_results.get((orig_id, dest_id), np.nan)
            
            # Best mode is the faster one
            if pd.notna(walk_time) and pd.notna(drive_time):
                if walk_time < drive_time:
                    best_mode = 'walk'
                    combined_time = walk_time
                else:
                    best_mode = 'drive'
                    combined_time = drive_time
            elif pd.notna(walk_time):
                best_mode = 'walk'
                combined_time = walk_time
            elif pd.notna(drive_time):
                best_mode = 'drive'
                combined_time = drive_time
            else:
                best_mode = 'unknown'
                combined_time = np.nan
            
            results.append({
                'origin_id': orig_id,
                'dest_id': dest_id,
                'walk_time': walk_time,
                'drive_time': drive_time,
                'combined_time': combined_time,
                'best_mode': best_mode
            })
        
        return pd.DataFrame(results)
    
    def _compute_mode_times(self, 
                           origins: gpd.GeoDataFrame, 
                           destinations: gpd.GeoDataFrame,
                           mode: str) -> Dict[Tuple[int, int], float]:
        """
        Compute travel times for one mode using OSRM
        
        Returns:
            Dict mapping (origin_id, dest_id) -> time_in_minutes
        """
        url_base = self.driving_url if mode == 'driving' else self.walking_url
        profile = mode # 'driving' or 'foot'
        
        results = {}
        total_pairs = len(origins) * len(destinations)
        processed = 0
        batches = 0
        
        # Process in batches to avoid overwhelming the server
        for orig_idx, origin in origins.iterrows():
            orig_id = origin.get('address_id', orig_idx)
            orig_lon, orig_lat = origin.geometry.x, origin.geometry.y
            
            # Batch destinations
            dest_batch = []
            dest_ids = []
            
            for dest_idx, destination in destinations.iterrows():
                dest_id = destination.get('dest_id', dest_idx)
                dest_lon, dest_lat = destination.geometry.x, destination.geometry.y
                
                dest_batch.append((dest_lon, dest_lat))
                dest_ids.append(dest_id)
                
                # Process batch when full
                if len(dest_batch) >= self.batch_size:
                    batch_results = self._route_batch(
                        (orig_lon, orig_lat), dest_batch, url_base, profile
                    )
                    
                    for dest_id, travel_time in zip(dest_ids, batch_results):
                        results[(orig_id, dest_id)] = travel_time
                    
                    processed += len(dest_batch)
                    batches += 1
                    dest_batch = []
                    dest_ids = []
            
            # Process remaining destinations in batch
            if dest_batch:
                batch_results = self._route_batch(
                    (orig_lon, orig_lat), dest_batch, url_base, profile
                )
                
                for dest_id, travel_time in zip(dest_ids, batch_results):
                    results[(orig_id, dest_id)] = travel_time
                
                processed += len(dest_batch)
                batches += 1
        
        self.log(f" Completed: {batches} batches, {processed} route pairs")
        
        return results
    
    def _route_batch(self, 
                    origin: Tuple[float, float],
                    destinations: List[Tuple[float, float]],
                    url_base: str,
                    profile: str) -> List[float]:
        """
        Route from one origin to multiple destinations using OSRM table service
        
        Returns:
            List of travel times in minutes (nan if route not found)
        """
        # Build coordinates string: origin first, then all destinations
        coords = [origin] + destinations
        coords_str = ";".join([f"{lon},{lat}" for lon, lat in coords])
        
        # Use table service for one-to-many routing
        # sources=0 means only the first coordinate (origin)
        # destinations=1;2;3;... means all other coordinates (use semicolons!)
        dest_indices = ";".join([str(i) for i in range(1, len(coords))])
        
        url = f"{url_base}/table/v1/{profile}/{coords_str}?sources=0&destinations={dest_indices}"
        
        try:
            response = requests.get(url, timeout=30)
            
            # Check for specific error
            if response.status_code == 400:
                data = response.json()
                error_msg = data.get('message', 'Unknown error')
                
                # Common issue: coordinates not on network
                if 'Coordinate is invalid' in error_msg or 'NoSegment' in error_msg:
                    self.log(f"Warning: Coordinates not on network, trying /nearest service")
                    # Try using nearest service to snap coordinates to road network
                    return self._route_batch_with_snapping(origin, destinations, url_base, profile)
                else:
                    self.log(f"Warning: OSRM 400 error: {error_msg}")
                    return [np.nan] * len(destinations)
            
            response.raise_for_status()
            data = response.json()
            
            if data['code'] != 'Ok':
                self.log(f"Warning: OSRM returned code {data['code']}")
                return [np.nan] * len(destinations)
            
            # Extract durations (in seconds) and convert to minutes
            durations = data['durations'][0] # First row = origin to all destinations
            travel_times = [d / 60.0 if d is not None else np.nan for d in durations]
            
            return travel_times
            
        except requests.exceptions.Timeout:
            self.log(f"Warning: Request timeout for batch of {len(destinations)} destinations")
            return [np.nan] * len(destinations)
        except Exception as e:
            self.log(f"Warning: Error in routing batch: {e}")
            return [np.nan] * len(destinations)
    
    def _route_batch_with_snapping(self,
                                   origin: Tuple[float, float],
                                   destinations: List[Tuple[float, float]],
                                   url_base: str,
                                   profile: str) -> List[float]:
        """
        Route with coordinate snapping to nearest road
        """
        # Snap origin to nearest road
        snapped_origin = self._snap_to_road(origin, url_base, profile)
        if snapped_origin is None:
            return [np.nan] * len(destinations)
        
        # Snap destinations to nearest roads
        snapped_destinations = []
        for dest in destinations:
            snapped = self._snap_to_road(dest, url_base, profile)
            if snapped is None:
                snapped_destinations.append(dest) # Use original if snapping fails
            else:
                snapped_destinations.append(snapped)
        
        # Try routing again with snapped coordinates
        coords = [snapped_origin] + snapped_destinations
        coords_str = ";".join([f"{lon},{lat}" for lon, lat in coords])
        dest_indices = ";".join([str(i) for i in range(1, len(coords))]) # Use semicolons!
        
        url = f"{url_base}/table/v1/{profile}/{coords_str}?sources=0&destinations={dest_indices}"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data['code'] == 'Ok':
                durations = data['durations'][0]
                return [d / 60.0 if d is not None else np.nan for d in durations]
            else:
                return [np.nan] * len(destinations)
                
        except Exception as e:
            self.log(f"Warning: Snapped routing also failed: {e}")
            return [np.nan] * len(destinations)
    
    def _snap_to_road(self, 
                     coord: Tuple[float, float], 
                     url_base: str,
                     profile: str) -> Tuple[float, float]:
        """Snap coordinate to nearest road using OSRM nearest service"""
        lon, lat = coord
        url = f"{url_base}/nearest/v1/{profile}/{lon},{lat}?number=1"
        
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            if data['code'] == 'Ok' and len(data['waypoints']) > 0:
                waypoint = data['waypoints'][0]
                snapped_lon, snapped_lat = waypoint['location']
                return (snapped_lon, snapped_lat)
            else:
                return None
                
        except Exception:
            return None