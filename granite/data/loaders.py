"""
GRANITE Data Loaders (Spatial Version)

Provides data loading for census, SVI, addresses, and graph construction.
Removes all accessibility/routing dependencies.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
from torch_geometric.data import Data
import torch
from typing import List, Dict, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .block_group_loader import BlockGroupLoader


class DataLoader:
    """
    Streamlined data loader for spatial disaggregation.
    No routing or destination dependencies.
    """
    
    def __init__(self, data_dir: str = '/workspaces/GRANITE/data', config: dict = None):
        self.data_dir = data_dir
        self.config = config or {}
        self.verbose = config.get('processing', {}).get('verbose', False) if config else False
        
        # Caches
        self._address_cache = None
        self._tract_cache = None
        self._svi_cache = None
        
        # Block group loader for validation
        self.block_group_loader = BlockGroupLoader(data_dir, verbose=self.verbose)
        
        os.makedirs(data_dir, exist_ok=True)
    
    def _log(self, message: str):
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] DataLoader: {message}")
    
    # =========================================================================
    # CENSUS AND SVI DATA
    # =========================================================================
    
    def load_census_tracts(self, state_fips: str = '47', county_fips: str = '065') -> gpd.GeoDataFrame:
        """Load census tract geometries."""
        cache_key = f"{state_fips}_{county_fips}"
        if self._tract_cache is not None and hasattr(self, '_tract_cache_key') and self._tract_cache_key == cache_key:
            return self._tract_cache
        
        self._log(f"Loading census tracts for {state_fips}-{county_fips}")
        
        local_file = os.path.join(self.data_dir, 'raw', f'tl_2020_{state_fips}_tract.shp')
        
        if not os.path.exists(local_file):
            raise FileNotFoundError(f"Census tracts file not found: {local_file}")
        
        tracts = gpd.read_file(local_file)
        county_tracts = tracts[tracts['COUNTYFP'] == county_fips].copy()
        county_tracts['FIPS'] = county_tracts['GEOID'].astype(str).str.strip()
        
        if county_tracts.crs is None:
            county_tracts.set_crs(epsg=4326, inplace=True)
        
        self._log(f"Loaded {len(county_tracts)} tracts")
        self._tract_cache = county_tracts
        self._tract_cache_key = cache_key
        
        return county_tracts
    
    def load_svi_data(self, state_fips: str = '47', county_name: str = 'Hamilton') -> pd.DataFrame:
        """Load Social Vulnerability Index data."""
        if self._svi_cache is not None:
            return self._svi_cache
        
        self._log(f"Loading SVI data for {county_name} County")
        
        svi_file = os.path.join(self.data_dir, 'raw', 'SVI_2020_US.csv')
        
        if not os.path.exists(svi_file):
            raise FileNotFoundError(f"SVI file not found: {svi_file}")
        
        svi_data = pd.read_csv(svi_file, dtype={'FIPS': str, 'ST': str})
        
        county_svi = svi_data[
            (svi_data['ST'] == state_fips) & 
            (svi_data['COUNTY'] == county_name)
        ].copy()
        
        county_svi['FIPS'] = county_svi['FIPS'].astype(str)
        county_svi['RPL_THEMES'] = county_svi['RPL_THEMES'].replace(-999, np.nan)
        
        # Keep essential columns
        keep_cols = ['FIPS', 'LOCATION', 'RPL_THEMES', 'E_TOTPOP', 'E_HU',
                     'EP_NOVEH', 'EP_POV150', 'EP_UNEMP', 'EP_NOHSDP']
        available = [c for c in keep_cols if c in county_svi.columns]
        county_svi = county_svi[available].copy()
        
        self._log(f"Loaded SVI for {len(county_svi)} tracts")
        self._svi_cache = county_svi
        
        return county_svi
    
    def _get_county_name(self, state_fips: str, county_fips: str) -> str:
        """Get county name from FIPS codes."""
        county_map = {'47065': 'Hamilton'}
        return county_map.get(f"{state_fips}{county_fips}", 'Hamilton')
    
    # =========================================================================
    # ADDRESS LOADING
    # =========================================================================
    
    def load_address_points(self, state_fips: str = '47', county_fips: str = '065') -> gpd.GeoDataFrame:
        """Load address points from GeoJSON."""
        if self._address_cache is not None:
            return self._address_cache
        
        self._log("Loading address data...")
        
        address_files = [
            os.path.join(self.data_dir, 'raw', 'chattanooga.geojson'),
            os.path.join(self.data_dir, 'chattanooga.geojson'),
        ]
        
        for addr_file in address_files:
            if os.path.exists(addr_file):
                addresses = gpd.read_file(addr_file)
                
                if 'address_id' not in addresses.columns:
                    addresses['address_id'] = range(len(addresses))
                
                if addresses.crs is None:
                    addresses.set_crs(epsg=4326, inplace=True)
                elif addresses.crs != 'EPSG:4326':
                    addresses = addresses.to_crs('EPSG:4326')
                
                self._log(f"Loaded {len(addresses)} addresses")
                self._address_cache = addresses
                return addresses
        
        raise FileNotFoundError("No address file found")
    
    def get_addresses_for_tract(self, fips_code: str) -> gpd.GeoDataFrame:
        """Get addresses within a specific census tract."""
        all_addresses = self.load_address_points()
        
        state_fips = fips_code[:2]
        county_fips = fips_code[2:5]
        tracts = self.load_census_tracts(state_fips, county_fips)
        
        tract = tracts[tracts['FIPS'] == fips_code]
        if len(tract) == 0:
            self._log(f"Tract {fips_code} not found")
            return gpd.GeoDataFrame(columns=['address_id', 'geometry'])
        
        tract_geom = tract.iloc[0].geometry
        tract_addresses = all_addresses[all_addresses.geometry.within(tract_geom)].copy()
        tract_addresses['tract_fips'] = fips_code
        
        self._log(f"Found {len(tract_addresses)} addresses in tract {fips_code}")
        return tract_addresses
    
    def get_neighboring_tracts(self, target_fips: str, n_neighbors: int = 4) -> List[str]:
        """Get neighboring tracts by geographic proximity."""
        if n_neighbors == 0:
            return [target_fips]
        
        state_fips = target_fips[:2]
        county_fips = target_fips[2:5]
        all_tracts = self.load_census_tracts(state_fips, county_fips)
        
        target_tract = all_tracts[all_tracts['FIPS'] == target_fips]
        if len(target_tract) == 0:
            return [target_fips]
        
        target_geom = target_tract.geometry.iloc[0]
        all_tracts['distance'] = all_tracts.geometry.distance(target_geom)
        neighbors = all_tracts[all_tracts['FIPS'] != target_fips].nsmallest(n_neighbors, 'distance')
        
        return [target_fips] + neighbors['FIPS'].tolist()
    
    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================
    
    def create_spatial_graph(self,
                            addresses: gpd.GeoDataFrame,
                            features: np.ndarray,
                            k_neighbors: int = 8) -> Data:
        """
        Create PyG graph from addresses using k-NN connectivity.
        
        Args:
            addresses: GeoDataFrame with address points
            features: Spatial features [n_addresses, n_features]
            k_neighbors: Number of neighbors for graph edges
        
        Returns:
            PyG Data object with x (features) and edge_index
        """
        n_addresses = len(addresses)
        self._log(f"Building spatial graph: {n_addresses} nodes, k={k_neighbors}")
        
        # Extract coordinates
        coords = np.column_stack([
            addresses.geometry.x.values,
            addresses.geometry.y.values
        ])
        
        # Build k-NN graph
        tree = cKDTree(coords)
        
        # Query k+1 neighbors (includes self)
        _, indices = tree.query(coords, k=min(k_neighbors + 1, n_addresses))
        
        # Build edge list (exclude self-loops)
        edge_list = []
        for i in range(n_addresses):
            for j in indices[i, 1:]:  # Skip first (self)
                if j < n_addresses:  # Safety check
                    edge_list.append([i, j])
                    edge_list.append([j, i])  # Undirected
        
        # Remove duplicates
        edge_set = set(tuple(e) for e in edge_list)
        edge_list = list(edge_set)
        
        edge_index = torch.LongTensor(edge_list).t().contiguous()
        x = torch.FloatTensor(features)
        
        self._log(f"Graph: {n_addresses} nodes, {edge_index.shape[1]} edges")
        
        return Data(x=x, edge_index=edge_index)
    
    # =========================================================================
    # VALIDATION HELPERS
    # =========================================================================
    
    def get_block_groups_for_tract(self, tract_fips: str) -> gpd.GeoDataFrame:
        """Get block groups within a tract for validation."""
        state_fips = tract_fips[:2]
        county_fips = tract_fips[2:5]
        
        bg_data = self.block_group_loader.get_block_groups_with_demographics(
            state_fips, county_fips
        )
        
        # Filter to tract
        tract_bg = bg_data[bg_data['tract_fips'] == tract_fips].copy()
        return tract_bg