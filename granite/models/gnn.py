"""
Graph Neural Network models for GRANITE framework

This module implements GNN architectures for learning SPDE parameters
from road network structure.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
import networkx as nx
from typing import Tuple, Dict
import pandas as pd


class SPDEParameterGNN(nn.Module):
    """
    GNN for learning spatially-varying SPDE parameters
    
    This model learns to predict Whittle-Matérn SPDE parameters
    (kappa, alpha, tau) for each node in the road network.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 3):
        """
        Initialize GNN model
        
        Parameters:
        -----------
        input_dim : int
            Number of input node features
        hidden_dim : int
            Hidden layer dimension
        output_dim : int
            Output dimension (3 for kappa, alpha, tau)
        """
        super(SPDEParameterGNN, self).__init__()
        
        # Graph convolution layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim // 2) 
        
        # Output layer for SPDE parameters
        self.param_head = nn.Linear(hidden_dim // 2, output_dim) 
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Parameters:
        -----------
        x : torch.Tensor
            Node features [num_nodes, input_dim]
        edge_index : torch.Tensor
            Edge connectivity [2, num_edges]
            
        Returns:
        --------
        torch.Tensor
            SPDE parameters [num_nodes, 3]
        """
        # First convolution
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second convolution
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Third convolution
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Fourth convolution
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        
        # Output SPDE parameters
        params = self.param_head(x)
        
        # Apply constraints to ensure valid parameter ranges
        # Kappa (precision) should be positive
        # Alpha (smoothness) should be between 0 and 2
        # Tau (nugget) should be positive and small
        #params = torch.stack([
        #    0.2 + 2.0 * torch.sigmoid(params[:, 0]),  # Kappa: [0.2, 2.2]
        #    0.8 + 1.5 * torch.sigmoid(params[:, 1]),  # Alpha: [0.8, 2.3] 
        #    0.1 + 0.8 * torch.sigmoid(params[:, 2])   # Tau: [0.1, 0.9]
        #], dim=1)
        
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
    Prepare graph data for GNN with NLCD-based features
    
    Parameters:
    -----------
    road_network : nx.Graph
        Road network graph with nodes as (x, y) coordinates
    nlcd_features : pd.DataFrame
        NLCD features with derived coefficients
    addresses : pd.DataFrame, optional
        Address data with coordinates for spatial matching
        
    Returns:
    --------
    Tuple[Data, Dict]
        PyTorch Geometric Data object and node ID mapping
    """
    nodes = list(road_network.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Initialize spatial lookup components
    feature_tree = None
    feature_coords = None
    
    # Build spatial index for NLCD feature lookup
    if len(nlcd_features) > 0:
        # Try to get coordinates from addresses if provided
        if addresses is not None and len(addresses) > 0:
            # Match nlcd_features to addresses by address_id
            if 'address_id' in nlcd_features.columns and 'address_id' in addresses.columns:
                # Merge to get coordinates
                features_with_coords = nlcd_features.merge(
                    addresses[['address_id', 'geometry']], 
                    on='address_id', 
                    how='left'
                )
                
                # Extract coordinates
                valid_features = features_with_coords.dropna(subset=['geometry'])
                if len(valid_features) > 0:
                    feature_coords = np.array([
                        [geom.x, geom.y] for geom in valid_features.geometry
                    ])
                    
                    # Build spatial index
                    from scipy.spatial import cKDTree
                    feature_tree = cKDTree(feature_coords)
                    
                    print(f"Built spatial index with {len(feature_coords)} address features")
                else:
                    print("WARNING: No valid coordinates found in addresses")
            else:
                print("WARNING: Cannot match nlcd_features to addresses - missing address_id")
        else:
            print("WARNING: No addresses provided for spatial matching")
    
    # Extract node features
    node_features = []
    successful_lookups = 0
    
    for node in nodes:
        node_x, node_y = node[0], node[1]
        
        # Default feature values (fallback)
        development_intensity = 0.5
        svi_coefficient = 0.3
        is_developed = 1.0
        is_uninhabited = 0.0
        normalized_nlcd_class = 0.23
        
        # Try spatial lookup if available
        if feature_tree is not None and len(nlcd_features) > 0:
            try:
                # Find nearest address
                distance, nearest_idx = feature_tree.query([node_x, node_y])
                
                # Use a reasonable distance threshold (e.g., 1000 meters in projected coords)
                if distance < 1000: 
                    # Get corresponding feature row
                    if 'address_id' in nlcd_features.columns:
                        # Need to map back to original nlcd_features
                        feature_row = nlcd_features.iloc[nearest_idx]
                    else:
                        feature_row = nlcd_features.iloc[nearest_idx]
                    
                    # Extract actual feature values
                    development_intensity = feature_row.get('development_intensity', 0.5)
                    svi_coefficient = feature_row.get('svi_coefficient', 
                                                   feature_row.get('svi_vulnerability_coeff', 0.3))
                    is_developed = float(feature_row.get('is_developed', 1.0))
                    is_uninhabited = float(feature_row.get('is_uninhabited', 0.0))
                    
                    # Normalize NLCD class (0-1 scale)
                    nlcd_class = feature_row.get('nlcd_class', 22)
                    normalized_nlcd_class = nlcd_class / 95.0  # Max NLCD class is 95
                    
                    successful_lookups += 1
                    
            except Exception as e:
                print(f"Error in spatial lookup for node {node}: {e}")
                # Keep default values
        
        # Construct feature vector
        features = [
            # Geographic features
            node_x,                    # X coordinate
            node_y,                    # Y coordinate
            
            # NLCD-derived features 
            development_intensity,     # 0.0-1.0 based on NLCD class
            svi_coefficient,          # 0.0-1.5 based on land cover vulnerability
            is_developed,             # 0 or 1 binary indicator
            is_uninhabited,           # 0 or 1 binary indicator  
            normalized_nlcd_class,    # 0.0-1.0 normalized NLCD class
            
            # Topological features
            min(road_network.degree(node), 10) / 10.0,  # Normalized degree
        ]
        
        node_features.append(features)
    
    print(f"✅ Feature extraction complete:")
    print(f"   Total nodes: {len(nodes)}")
    print(f"   Successful NLCD lookups: {successful_lookups}")
    print(f"   Lookup success rate: {successful_lookups/len(nodes)*100:.1f}%")
    
    # Convert to tensor and normalize
    node_features = torch.FloatTensor(node_features)
    node_features = safe_feature_normalization_vectorized(node_features)
    
    # Build edges 
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