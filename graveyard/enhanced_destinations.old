"""
ADDITION: Local Destination Enhancement for Intra-Tract Analysis
Addresses scale mismatch by creating tract-appropriate destinations

Add this to your data/loaders.py after the existing destination creation methods
"""
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from typing import Dict, List, Tuple

def create_enhanced_destinations_for_tract(self, tract_addresses: gpd.GeoDataFrame, 
                                         existing_destinations: Dict[str, gpd.GeoDataFrame]) -> Dict[str, gpd.GeoDataFrame]:
    """
    ADDITION: Create enhanced destination set appropriate for intra-tract analysis
    
    Combines:
    1. Existing county-wide destinations (for regional context)
    2. Synthetic local destinations (for meaningful variation)
    3. Edge destinations (for accessibility gradients)
    
    Args:
        tract_addresses: Addresses within the target tract
        existing_destinations: County-wide destinations by type
    
    Returns:
        Enhanced destination dictionary with better spatial distribution
    """
    self._log("Creating enhanced destinations for intra-tract analysis...")
    
    enhanced_destinations = {}
    
    # Get tract boundary
    tract_bounds = tract_addresses.geometry.total_bounds  # [minx, miny, maxx, maxy]
    tract_center = Point(
        (tract_bounds[0] + tract_bounds[2]) / 2,
        (tract_bounds[1] + tract_bounds[3]) / 2
    )
    
    # Calculate tract dimensions
    tract_width = tract_bounds[2] - tract_bounds[0]  # degrees longitude
    tract_height = tract_bounds[3] - tract_bounds[1]  # degrees latitude
    
    for dest_type, existing_dests in existing_destinations.items():
        self._log(f"  Enhancing {dest_type} destinations...")
        
        # Strategy: Create multi-scale destination network
        enhanced_dests = []
        
        # 1. Keep relevant existing destinations (within reasonable distance)
        max_distance_deg = max(tract_width, tract_height) * 3  # 3x tract size
        
        for _, dest in existing_dests.iterrows():
            dist_to_center = tract_center.distance(dest.geometry)
            if dist_to_center <= max_distance_deg:
                enhanced_dests.append({
                    'name': dest.get('name', f'{dest_type}_existing'),
                    'geometry': dest.geometry,
                    'dest_id': len(enhanced_dests),
                    'dest_type': dest_type,
                    'scale': 'regional',
                    'importance': self._calculate_destination_importance(dest, dest_type)
                })
        
        # 2. Create local destinations for meaningful variation
        local_dests = self._create_local_destinations(
            tract_center, tract_width, tract_height, dest_type
        )
        enhanced_dests.extend(local_dests)
        
        # 3. Create edge destinations for accessibility gradients
        edge_dests = self._create_edge_destinations(
            tract_bounds, dest_type
        )
        enhanced_dests.extend(edge_dests)
        
        # Convert to GeoDataFrame
        if enhanced_dests:
            enhanced_gdf = gpd.GeoDataFrame(enhanced_dests, crs='EPSG:4326')
            enhanced_gdf['dest_id'] = range(len(enhanced_gdf))
        else:
            # Fallback to existing if no enhancements possible
            enhanced_gdf = existing_dests.copy()
            enhanced_gdf['scale'] = 'regional'
            enhanced_gdf['importance'] = 1.0
        
        enhanced_destinations[dest_type] = enhanced_gdf
        self._log(f"    {dest_type}: {len(enhanced_gdf)} destinations "
                 f"(was {len(existing_dests)})")
    
    return enhanced_destinations

def _create_local_destinations(self, tract_center: Point, tract_width: float, 
                             tract_height: float, dest_type: str) -> List[Dict]:
    """Create local destinations within and near the tract"""
    
    local_destinations = []
    
    # Define destination-specific parameters
    dest_params = {
        'employment': {
            'names': ['Local Business District', 'Commercial Center', 'Office Complex', 'Retail Cluster'],
            'count': 4,
            'spread_factor': 0.8  # Closer to center
        },
        'healthcare': {
            'names': ['Neighborhood Clinic', 'Urgent Care', 'Medical Center', 'Health Services'],
            'count': 3,
            'spread_factor': 1.0  # Normal spread
        },
        'grocery': {
            'names': ['Neighborhood Market', 'Corner Store', 'Shopping Center', 'Local Grocery'],
            'count': 4,
            'spread_factor': 0.6  # Very local
        }
    }
    
    params = dest_params.get(dest_type, {
        'names': [f'Local {dest_type.title()}'],
        'count': 2,
        'spread_factor': 0.8
    })
    
    # Create destinations in a pattern that ensures variation
    angles = np.linspace(0, 2*np.pi, params['count'], endpoint=False)
    
    for i, angle in enumerate(angles):
        # Vary distance from center
        base_distance = max(tract_width, tract_height) * params['spread_factor']
        distance_variation = np.random.uniform(0.3, 1.2)  # 30-120% of base
        distance = base_distance * distance_variation
        
        # Calculate position
        x_offset = distance * np.cos(angle)
        y_offset = distance * np.sin(angle)
        
        dest_point = Point(
            tract_center.x + x_offset,
            tract_center.y + y_offset
        )
        
        # Assign name
        name = params['names'][i % len(params['names'])]
        if len(params['names']) <= i:
            name += f" #{i+1}"
        
        local_destinations.append({
            'name': name,
            'geometry': dest_point,
            'dest_type': dest_type,
            'scale': 'local',
            'importance': np.random.uniform(0.6, 1.0),  # Vary importance
            'angle': angle,
            'distance_from_center': distance
        })
    
    return local_destinations

def _create_edge_destinations(self, tract_bounds: np.ndarray, dest_type: str) -> List[Dict]:
    """Create destinations at tract edges for accessibility gradients"""
    
    edge_destinations = []
    
    # Create destinations just outside tract boundaries
    minx, miny, maxx, maxy = tract_bounds
    
    # Extend bounds slightly
    width = maxx - minx
    height = maxy - miny
    
    edge_positions = [
        # North edge
        Point(minx + width * 0.5, maxy + height * 0.3),
        # South edge  
        Point(minx + width * 0.5, miny - height * 0.3),
        # East edge
        Point(maxx + width * 0.3, miny + height * 0.5),
        # West edge
        Point(minx - width * 0.3, miny + height * 0.5)
    ]
    
    edge_names = [
        f'North {dest_type.title()}',
        f'South {dest_type.title()}', 
        f'East {dest_type.title()}',
        f'West {dest_type.title()}'
    ]
    
    for i, (position, name) in enumerate(zip(edge_positions, edge_names)):
        edge_destinations.append({
            'name': name,
            'geometry': position,
            'dest_type': dest_type,
            'scale': 'edge',
            'importance': np.random.uniform(0.4, 0.8),  # Lower importance for edge
            'edge_direction': ['north', 'south', 'east', 'west'][i]
        })
    
    return edge_destinations

def _calculate_destination_importance(self, destination: gpd.GeoSeries, dest_type: str) -> float:
    """Calculate relative importance of existing destinations"""
    
    # Simple importance based on destination characteristics
    importance = 0.5  # Base importance
    
    if dest_type == 'employment':
        employees = destination.get('employees', 1000)
        # Normalize to 0.2-1.0 range
        importance = 0.2 + 0.8 * min(1.0, employees / 5000)
        
    elif dest_type == 'healthcare':
        beds = destination.get('beds', 100)
        # Normalize to 0.3-1.0 range
        importance = 0.3 + 0.7 * min(1.0, beds / 400)
        
    elif dest_type == 'grocery':
        store_type = destination.get('type', 'grocery')
        type_importance = {
            'supermarket': 1.0,
            'grocery': 0.7,
            'convenience': 0.4,
            'discount': 0.6
        }
        importance = type_importance.get(store_type, 0.5)
    
    return importance

# ADD this method to your DataLoader class in loaders.py
def create_tract_appropriate_destinations(self, tract_fips: str) -> Dict[str, gpd.GeoDataFrame]:
    """
    MAIN METHOD: Create tract-appropriate destinations
    
    Call this instead of the individual create_*_destinations methods
    when analyzing a single tract
    """
    
    # Get tract addresses
    tract_addresses = self.get_addresses_for_tract(tract_fips)
    
    if len(tract_addresses) == 0:
        self._log(f"No addresses found for tract {tract_fips}, using default destinations")
        return {
            'employment': self.create_employment_destinations(),
            'healthcare': self.create_healthcare_destinations(),
            'grocery': self.create_grocery_destinations()
        }
    
    # Create existing county-wide destinations
    existing_destinations = {
        'employment': self.create_employment_destinations(),
        'healthcare': self.create_healthcare_destinations(),
        'grocery': self.create_grocery_destinations()
    }
    
    # Enhance for tract analysis
    enhanced_destinations = self.create_enhanced_destinations_for_tract(
        tract_addresses, existing_destinations
    )
    
    return enhanced_destinations

# ADD these lines to your DataLoader.__init__ method to bind the new methods:
def bind_enhanced_destination_methods(self):
    """Bind the enhanced destination methods to DataLoader instance"""
    import types
    
    self.create_enhanced_destinations_for_tract = types.MethodType(
        create_enhanced_destinations_for_tract, self
    )
    self._create_local_destinations = types.MethodType(
        _create_local_destinations, self
    )
    self._create_edge_destinations = types.MethodType(
        _create_edge_destinations, self
    )
    self._calculate_destination_importance = types.MethodType(
        _calculate_destination_importance, self
    )
    self.create_tract_appropriate_destinations = types.MethodType(
        create_tract_appropriate_destinations, self
    )