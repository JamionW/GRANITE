"""
Minimal interface for accessibility research - NO MetricGraph complexity
"""
import numpy as np
import pandas as pd

class MinimalInterface:
    """Simplified interface that doesn't use R/MetricGraph"""
    
    def __init__(self, config=None):
        self.config = config or {}
    
    def simple_accessibility_calculation(self, addresses, destinations):
        """Basic accessibility calculation without spatial complexity"""
        accessibility_scores = []
        
        for _, addr in addresses.iterrows():
            addr_geom = addr.geometry
            gravity_score = 0
            
            for _, dest in destinations.iterrows():
                distance = addr_geom.distance(dest.geometry)
                if distance > 0:
                    # Simple gravity model
                    gravity_score += 1.0 / (distance ** 1.5)
            
            accessibility_scores.append(gravity_score)
        
        return np.array(accessibility_scores)