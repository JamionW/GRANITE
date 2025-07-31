"""
Graph Neural Network models for GRANITE framework

This module implements GNN architectures for learning SPDE parameters
from road network structure.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
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
        params = torch.stack([
            0.2 + 2.0 * torch.sigmoid(params[:, 0]),  # Kappa: [0.2, 2.2]
            0.8 + 1.5 * torch.sigmoid(params[:, 1]),  # Alpha: [0.8, 2.3] 
            0.1 + 0.8 * torch.sigmoid(params[:, 2])   # Tau: [0.1, 0.9]
        ], dim=1)
        
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
                                nlcd_features: pd.DataFrame) -> Tuple[Data, Dict]:
    nodes = list(road_network.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Create spatial index for faster NLCD feature lookup
    from scipy.spatial import cKDTree
    
    # Build feature lookup from address-based NLCD features
    if len(nlcd_features) > 0:
        # Assuming nlcd_features has coordinate info or can be mapped to addresses
        feature_lookup = nlcd_features.set_index('address_id').to_dict('index')
    else:
        # Fallback to default values
        feature_lookup = {}
    
    node_features = []
    for node in nodes:
        # Get nearest NLCD features
        # For now, using node coordinates to find nearest address features
        
        features = [
            # Geographic features 
            node[0],  # X coordinate 
            node[1],  # Y coordinate 
            
            0.5,      # development_intensity (placeholder - replace with lookup)
            0.3,      # svi_coefficient (placeholder - replace with lookup)  
            1.0,      # is_developed (placeholder - replace with lookup)
            0.0,      # is_uninhabited (placeholder - replace with lookup)
            0.23,     # normalized nlcd_class (placeholder - replace with lookup)
            
            min(road_network.degree(node), 10) / 10.0, 
        ]
        
        node_features.append(features)
    
    # Rest of function same as your current implementation
    node_features = torch.FloatTensor(node_features)
    node_features = safe_feature_normalization_vectorized(node_features)
    
    # Build edges (keep your existing edge construction code)
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

# Alias for backward compatibility
prepare_graph_data = prepare_graph_data_topological  # Default to old version for now

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