"""
Accessibility baseline methods for comparison with GNN approaches
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

class AccessibilityBaseline:
    """Traditional accessibility measures for validation"""
    
    def __init__(self, config=None):
        self.config = config or {}
    
    def compute_gravity_accessibility(self, addresses, destinations):
        """Standard gravity model accessibility"""
        gravity_scores = []
        
        for _, addr in addresses.iterrows():
            addr_coord = [addr.geometry.x, addr.geometry.y]
            gravity_score = 0
            
            for _, dest in destinations.iterrows():
                dest_coord = [dest.geometry.x, dest.geometry.y]
                distance = np.sqrt((addr_coord[0] - dest_coord[0])**2 + 
                                 (addr_coord[1] - dest_coord[1])**2)
                
                if distance > 0:
                    # Standard gravity model with distance decay
                    attraction = dest.get('capacity', dest.get('employees', 100))
                    gravity_score += attraction / (distance ** 2)
            
            gravity_scores.append(gravity_score)
        
        return np.array(gravity_scores)
    
    def compute_cumulative_opportunities(self, addresses, destinations, threshold=0.01):
        """Count destinations within threshold distance"""
        opportunity_counts = []
        
        for _, addr in addresses.iterrows():
            addr_coord = [addr.geometry.x, addr.geometry.y]
            count = 0
            
            for _, dest in destinations.iterrows():
                dest_coord = [dest.geometry.x, dest.geometry.y]
                distance = np.sqrt((addr_coord[0] - dest_coord[0])**2 + 
                                 (addr_coord[1] - dest_coord[1])**2)
                
                if distance <= threshold:
                    count += 1
            
            opportunity_counts.append(count)
        
        return np.array(opportunity_counts)