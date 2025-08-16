"""
Graph Neural Network models for GRANITE framework

This module implements GNN architectures for learning SPDE parameters
from road network structure.
"""
# Standard library imports
from typing import Tuple, Dict

# Third-party imports
import numpy as np
import pandas as pd
import networkx as nx

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch Geometric imports
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class SPDEParameterGNN(nn.Module):
    """GNN with explicit variance preservation"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 3):
        super(SPDEParameterGNN, self).__init__()
        
        # Residual connections to preserve input variance
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Use BatchNorm to maintain variance
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Direct feature bypass to preserve input signal
        self.feature_bypass = nn.Linear(input_dim, output_dim)
        
        # Output heads
        self.param_head = nn.Linear(hidden_dim + output_dim, output_dim)
        
        # Minimal dropout
        self.dropout = nn.Dropout(0.05)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Save original features for bypass
        original_features = x
        
        # Project input
        x = self.input_projection(x)
        identity = x  # For residual
        
        # Conv block 1 with residual
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x + identity  # Residual connection
        
        # Conv block 2 with residual
        identity = x
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x + identity
        
        # Conv block 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Combine with bypassed features
        bypass_params = self.feature_bypass(original_features)
        combined = torch.cat([x, bypass_params], dim=1)
        
        # Final parameters with no compression
        params = self.param_head(combined)

        kappa = params[:, 0] * 5.0 + 2.5    # Kappa in [2.5-5, 2.5+5] = [-2.5, 7.5]
        alpha = params[:, 1] * 1.0           
        tau = params[:, 2] * 1.0 + 0.5       # Tau in [0.5-1, 0.5+1] = [-0.5, 1.5]

        # But clamp to valid ranges
        kappa = torch.clamp(kappa, min=0.2, max=10.0)  
        tau = torch.clamp(tau, min=0.05, max=2.0) 
        
        params = torch.stack([kappa, alpha, tau], dim=1)
        
        return params

def safe_feature_normalization_vectorized(node_features):
    """
    Fully vectorized normalization
    """
    # Compute min/max across all features at once
    col_mins = torch.min(node_features, dim=0, keepdim=True)[0]
    col_maxs = torch.max(node_features, dim=0, keepdim=True)[0]
    
    # Avoid division by zero
    col_ranges = col_maxs - col_mins
    col_ranges = torch.where(col_ranges > 0, col_ranges, torch.ones_like(col_ranges))
    
    # Vectorized normalization (creates new tensor)
    normalized_features = (node_features - col_mins) / col_ranges
    
    return normalized_features

def prepare_graph_data_with_nlcd(road_network: nx.Graph, 
                                nlcd_features: pd.DataFrame,
                                addresses: pd.DataFrame = None) -> Tuple[Data, Dict]:
    """
    Prepare graph data for GNN with enhanced NLCD-based features
    """
    import numpy as np
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import cdist
    
    nodes = list(road_network.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Build spatial index for NLCD feature lookup
    feature_tree = None
    feature_coords = None
    
    if len(nlcd_features) > 0 and addresses is not None and len(addresses) > 0:
        # Match nlcd_features to addresses by address_id
        if 'address_id' in nlcd_features.columns and 'address_id' in addresses.columns:
            features_with_coords = nlcd_features.merge(
                addresses[['address_id', 'geometry']], 
                on='address_id', 
                how='left'
            )
            
            valid_features = features_with_coords.dropna(subset=['geometry'])
            if len(valid_features) > 0:
                feature_coords = np.array([
                    [geom.x, geom.y] for geom in valid_features.geometry
                ])
                feature_tree = cKDTree(feature_coords)
                print(f"Built spatial index with {len(feature_coords)} address features")
    
    # Extract enhanced node features
    node_features = []
    successful_lookups = 0
    
    # Pre-compute spatial features for efficiency
    water_classes = [11, 12, 90, 95]
    forest_classes = [41, 42, 43]
    developed_classes = [21, 22, 23, 24]
    
    for node in nodes:
        node_x, node_y = node[0], node[1]
        
        # Default feature values
        development_intensity = 0.5
        svi_coefficient = 0.3
        land_cover_diversity = 0.0
        development_gradient = 0.0  
        distance_to_water = 1.0
        distance_to_forest = 1.0
        normalized_nlcd_class = 0.23
        
        # Try spatial lookup if available
        if feature_tree is not None and len(nlcd_features) > 0:
            try:
                # Find nearest address
                distance, nearest_idx = feature_tree.query([node_x, node_y])
                
                if distance < 1000:  # Within 1km
                    # Get corresponding feature row
                    feature_row = nlcd_features.iloc[nearest_idx]
                    
                    # Extract basic features
                    development_intensity = feature_row.get('development_intensity', 0.5)
                    svi_coefficient = feature_row.get('svi_coefficient', 
                                                   feature_row.get('svi_vulnerability_coeff', 0.3))
                    
                    # 1. Land Cover Diversity (Shannon entropy in neighborhood)
                    land_cover_diversity = calculate_land_cover_diversity(
                        node_x, node_y, feature_tree, feature_coords, 
                        nlcd_features, radius=500
                    )
                    
                    # 2. Development Gradient (rate of development change)
                    development_gradient = calculate_development_gradient(
                        node_x, node_y, feature_tree, feature_coords,
                        nlcd_features, radius=300
                    )
                    
                    # 3. Distance to Water Features  
                    distance_to_water = calculate_distance_to_feature_class(
                        node_x, node_y, feature_tree, feature_coords,
                        nlcd_features, target_classes=water_classes
                    )
                    
                    # 4. Distance to Forest Features
                    distance_to_forest = calculate_distance_to_feature_class(
                        node_x, node_y, feature_tree, feature_coords,
                        nlcd_features, target_classes=forest_classes
                    )
                    
                    # Normalize NLCD class
                    nlcd_class = feature_row.get('nlcd_class', 22)
                    normalized_nlcd_class = nlcd_class / 95.0
                    
                    successful_lookups += 1
                    
            except Exception as e:
                print(f"Error in spatial lookup for node {node}: {e}")
        
        # Construct enhanced feature vector
        features = [
            # Core NLCD-derived features (both GNN and IDM can use)
            development_intensity,     # 0.0-1.0 based on NLCD class
            svi_coefficient,          # 0.0-1.5 based on land cover vulnerability
            development_gradient,     # 0.0-1.0 rate of development change  
            
            # Network topology (GNN-specific)
            min(road_network.degree(node), 10) / 10.0,  # Normalized degree
        ]
        
        node_features.append(features)
    
    print(f"  Enhanced feature extraction complete:")
    print(f"   Total nodes: {len(nodes)}")
    print(f"   Successful NLCD lookups: {successful_lookups}")
    print(f"   Features per node: {len(features)}")
    
    # Convert to tensor and normalize
    node_features = torch.FloatTensor(node_features)
    node_features = safe_feature_normalization_vectorized(node_features)
    
    # Build edges (unchanged)
    edge_list = []
    edge_attrs = []
    
    for u, v, data in road_network.edges(data=True):
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        
        edge_list.append([u_idx, v_idx])
        edge_list.append([v_idx, u_idx])
        
        length = data.get('length', 1.0)
        edge_attrs.append([length])
        edge_attrs.append([length])
    
    # Create PyTorch Geometric data object
    x = node_features
    edge_index = torch.LongTensor(edge_list).t().contiguous()
    edge_attr = torch.FloatTensor(edge_attrs) if edge_attrs else None
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data, node_to_idx

def prepare_graph_data_topological(road_network: nx.Graph) -> Tuple[Data, Dict]:
    """
    Prepare graph data for GNN from road network
    
    Parameters:
    -----------
    road_network : nx.Graph
        Road network graph
        
    Returns:
    --------
    Tuple[Data, Dict]
        PyTorch Geometric Data object and node ID mapping
    """
    # Create node mapping
    nodes = list(road_network.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Extract node features
    node_features = []
    
    for node in nodes:
        # Compute node features
        features = [
            road_network.degree(node),  # Degree
            nx.closeness_centrality(road_network, node) if len(nodes) > 1 else 0.5,  # Closeness
            node[0],  # X coordinate (normalized later)
            node[1],  # Y coordinate (normalized later)
            len(road_network[node]),  # Local connectivity
            np.mean([1.0 for neighbor in road_network[node]]) if road_network[node] else 0.0,  # Simple neighbor count
            min(road_network.degree(node), 10) / 10.0,  # Normalized degree (0-1)
        ]

        node_features.append(features)
    
    # Convert to PyTorch tensor, then normalize
    node_features = torch.FloatTensor(node_features)
    node_features = safe_feature_normalization_vectorized(node_features)
    
    # Extract edges
    edge_list = []
    edge_attrs = []
    
    for u, v, data in road_network.edges(data=True):
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        
        # Add both directions for undirected graph
        edge_list.append([u_idx, v_idx])
        edge_list.append([v_idx, u_idx])
        
        # Edge attributes (e.g., length)
        length = data.get('length', 1.0)
        edge_attrs.append([length])
        edge_attrs.append([length])
    
    # Convert to tensors
    x = node_features
    edge_index = torch.LongTensor(edge_list).t().contiguous()
    edge_attr = torch.FloatTensor(edge_attrs) if edge_attrs else None
    
    # Create PyTorch Geometric data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data, node_to_idx

def create_gnn_model(input_dim: int = 5, hidden_dim: int = 128, 
                    output_dim: int = 3) -> nn.Module:
    """
    Create GNN model for SPDE parameter learning
    
    Parameters:
    -----------
    input_dim : int
        Number of input features
    hidden_dim : int
        Hidden layer dimension
    output_dim : int
        Output dimension (3 for SPDE parameters)
        
    Returns:
    --------
    nn.Module
        GNN model
    """
    return SPDEParameterGNN(input_dim, hidden_dim, output_dim)

def calculate_land_cover_diversity(node_x, node_y, feature_tree, feature_coords, 
                                  nlcd_features, radius=500):
    """Simplified land cover diversity calculation"""
    try:
        # Check if we have the required data
        if feature_tree is None or 'nlcd_class' not in nlcd_features.columns:
            return 0.0
            
        # Find neighbors within radius  
        neighbor_indices = feature_tree.query_ball_point([node_x, node_y], radius)
        
        if len(neighbor_indices) < 2:
            return 0.0
        
        # Get NLCD classes for neighbors
        neighbor_classes = nlcd_features.iloc[neighbor_indices]['nlcd_class'].values
        unique_classes = len(set(neighbor_classes))
        
        # Simple diversity measure: number of unique classes / max possible
        return min(unique_classes / 4.0, 1.0)  # Normalize by 4 expected classes
        
    except Exception as e:
        print(f"Land cover diversity error: {e}")
        return 0.0


def calculate_development_gradient(node_x, node_y, feature_tree, feature_coords,
                                 nlcd_features, radius=300):
    """Calculate rate of development intensity change in neighborhood"""
    try:
        # Find neighbors within radius
        neighbor_indices = feature_tree.query_ball_point([node_x, node_y], radius)
        
        if len(neighbor_indices) < 2:
            return 0.0
        
        # Get development intensities
        neighbor_features = nlcd_features.iloc[neighbor_indices]
        dev_intensities = neighbor_features['development_intensity'].values
        
        # Calculate distances from node
        neighbor_coords = feature_coords[neighbor_indices]
        distances = np.sqrt(np.sum((neighbor_coords - [node_x, node_y])**2, axis=1))
        
        # Calculate gradient (correlation between distance and development)
        if len(set(distances)) > 1 and len(set(dev_intensities)) > 1:
            correlation = np.corrcoef(distances, dev_intensities)[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.0
        
        return 0.0
        
    except:
        return 0.0


def calculate_distance_to_feature_class(node_x, node_y, feature_tree, feature_coords,
                                       nlcd_features, target_classes):
    """Simplified distance calculation"""
    try:
        if feature_tree is None or 'nlcd_class' not in nlcd_features.columns:
            return 1.0
            
        # Check if any target classes exist in the data
        has_targets = nlcd_features['nlcd_class'].isin(target_classes).any()
        if not has_targets:
            return 1.0  # No target classes found
            
        # Find all addresses within reasonable distance
        all_indices = feature_tree.query_ball_point([node_x, node_y], 2000)  # 2km radius
        
        if len(all_indices) == 0:
            return 1.0
            
        # Check which ones have target classes
        for idx in all_indices:
            if idx < len(nlcd_features):
                nlcd_class = nlcd_features.iloc[idx]['nlcd_class']
                if nlcd_class in target_classes:
                    # Calculate distance to this target
                    target_coord = feature_coords[idx]
                    distance = np.sqrt((target_coord[0] - node_x)**2 + (target_coord[1] - node_y)**2)
                    return min(distance / 2000.0, 1.0)  # Normalize by 2km
                    
        return 1.0  # No targets found
        
    except Exception as e:
        print(f"Distance calculation error: {e}")
        return 1.0