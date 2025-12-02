"""
Attention Weight Analysis for Context-Gating
Visualizes which accessibility features matter in which contexts
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_attention_patterns(attention_weights, context_features, 
                               feature_names, address_svi=None, 
                               output_dir=None):
    """
    Analyze and visualize attention weight patterns.
    
    Args:
        attention_weights: [n_addresses, n_accessibility_features]
        context_features: [n_addresses, n_context_features]
        feature_names: List of accessibility feature names
        address_svi: Optional SVI values for each address
        output_dir: Where to save plots
    """
    
    # 1. Correlation between context and attention
    print("=== ATTENTION PATTERN ANALYSIS ===\n")
    
    # Vehicle ownership vs. transit feature attention
    vehicle_absence = context_features[:, 0]  # Assuming first context feature
    transit_features = [i for i, name in enumerate(feature_names) 
                       if 'walk' in name.lower() or 'transit' in name.lower()]
    
    if len(transit_features) > 0:
        avg_transit_attention = attention_weights[:, transit_features].mean(axis=1)
        
        corr = np.corrcoef(vehicle_absence, avg_transit_attention)[0, 1]
        print(f"Vehicle absence vs. Transit attention correlation: {corr:.3f}")
        
        if corr > 0.3:
            print("  ✓ Strong positive correlation - transit features upweighted in low-vehicle areas")
        elif corr > 0.1:
            print("  ~ Moderate correlation - some context-dependency learned")
        else:
            print("  ✗ Weak correlation - context-gating may not be working effectively")
    
    # 2. Feature importance by context
    print("\n=== FEATURE IMPORTANCE BY CONTEXT ===\n")
    
    # Split addresses by vehicle ownership (high vs. low)
    high_vehicle = vehicle_absence < np.median(vehicle_absence)
    low_vehicle = vehicle_absence >= np.median(vehicle_absence)
    
    high_vehicle_attn = attention_weights[high_vehicle].mean(axis=0)
    low_vehicle_attn = attention_weights[low_vehicle].mean(axis=0)
    
    # Show top features in each context
    print("High-vehicle areas (car-oriented):")
    top_high = np.argsort(high_vehicle_attn)[-5:][::-1]
    for idx in top_high:
        print(f"  {feature_names[idx]}: {high_vehicle_attn[idx]:.3f}")
    
    print("\nLow-vehicle areas (transit-dependent):")
    top_low = np.argsort(low_vehicle_attn)[-5:][::-1]
    for idx in top_low:
        print(f"  {feature_names[idx]}: {low_vehicle_attn[idx]:.3f}")
    
    # 3. Visualization if output_dir provided
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot 1: Attention heatmap by context
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # High-vehicle context
        sns.heatmap(high_vehicle_attn.reshape(-1, 1), 
                   yticklabels=feature_names,
                   xticklabels=['Attention'],
                   cmap='YlOrRd', ax=axes[0])
        axes[0].set_title('High-Vehicle Areas (Car-Oriented)')
        
        # Low-vehicle context
        sns.heatmap(low_vehicle_attn.reshape(-1, 1),
                   yticklabels=feature_names,
                   xticklabels=['Attention'],
                   cmap='YlOrRd', ax=axes[1])
        axes[1].set_title('Low-Vehicle Areas (Transit-Dependent)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'attention_by_context.png'), dpi=300)
        plt.close()
        
        print(f"\n✓ Saved attention heatmap to {output_dir}")
    
    return {
        'vehicle_transit_correlation': corr if len(transit_features) > 0 else None,
        'high_vehicle_attention': high_vehicle_attn,
        'low_vehicle_attention': low_vehicle_attn
    }