"""
Visualization functions for GRANITE framework
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class DisaggregationVisualizer:
    """Create visualizations for SVI disaggregation results"""
    
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        plt.style.use(style)
        self.colors = sns.color_palette("husl", 8)
        
    def create_summary_plot(self, tracts_gdf, addresses_gdf, gnn_features, 
                           validation_df=None, figsize=(16, 12)):
        """
        Create comprehensive summary visualization
        
        Parameters:
        -----------
        tracts_gdf : gpd.GeoDataFrame
            Census tracts with SVI
        addresses_gdf : gpd.GeoDataFrame
            Addresses with predictions
        gnn_features : np.ndarray
            GNN-learned features
        validation_df : pd.DataFrame, optional
            Validation results
            
        Returns:
        --------
        matplotlib.figure.Figure
            Summary figure
        """
        # Create figure with subplots
        if validation_df is not None:
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        else:
            fig = plt.figure(figsize=(figsize[0], figsize[1]*2/3))
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Tract-level SVI
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_tract_svi(ax1, tracts_gdf)
        
        # 2. Address-level predictions
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_address_predictions(ax2, addresses_gdf)
        
        # 3. Uncertainty map
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_uncertainty(ax3, addresses_gdf)
        
        # 4. GNN features
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_gnn_features(ax4, gnn_features)
        
        # 5. Prediction intervals
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_prediction_intervals(ax5, addresses_gdf)
        
        # 6. Summary statistics
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_summary_stats(ax6, tracts_gdf, addresses_gdf, gnn_features)
        
        # 7. Validation plot (if available)
        if validation_df is not None:
            ax7 = fig.add_subplot(gs[2, :2])
            self._plot_validation(ax7, validation_df)
            
            ax8 = fig.add_subplot(gs[2, 2])
            self._plot_error_distribution(ax8, validation_df)
        
        # Main title
        fig.suptitle('GRANITE: SVI Disaggregation Results', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        return fig
    
    def _plot_tract_svi(self, ax, tracts_gdf):
        """Plot tract-level SVI"""
        valid_tracts = tracts_gdf[tracts_gdf['RPL_THEMES'].notna()]
        
        valid_tracts.plot(
            column='RPL_THEMES',
            ax=ax,
            legend=True,
            cmap='RdYlBu_r',
            edgecolor='black',
            linewidth=0.5,
            legend_kwds={'label': 'SVI Score', 'shrink': 0.8}
        )
        
        ax.set_title('Original Tract-Level SVI', fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal')
        
    def _plot_address_predictions(self, ax, addresses_gdf):
        """Plot address-level predictions"""
        scatter = ax.scatter(
            addresses_gdf['longitude'],
            addresses_gdf['latitude'],
            c=addresses_gdf['svi_predicted'],
            cmap='RdYlBu_r',
            s=20,
            alpha=0.6,
            edgecolors='none'
        )
        
        cbar = plt.colorbar(scatter, ax=ax, label='Predicted SVI')
        ax.set_title('Disaggregated Address-Level SVI', fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal')
        
    def _plot_uncertainty(self, ax, addresses_gdf):
        """Plot prediction uncertainty"""
        scatter = ax.scatter(
            addresses_gdf['longitude'],
            addresses_gdf['latitude'],
            c=addresses_gdf['svi_sd'],
            cmap='viridis',
            s=20,
            alpha=0.6,
            edgecolors='none'
        )
        
        cbar = plt.colorbar(scatter, ax=ax, label='Std. Dev.')
        ax.set_title('Prediction Uncertainty', fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal')
        
    def _plot_gnn_features(self, ax, gnn_features):
        """Plot GNN-learned features using PCA"""
        # Apply PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(gnn_features)
        
        scatter = ax.scatter(
            features_2d[:, 0],
            features_2d[:, 1],
            c=gnn_features[:, 0],  # Color by kappa parameter
            cmap='plasma',
            s=30,
            alpha=0.7
        )
        
        cbar = plt.colorbar(scatter, ax=ax, label='κ parameter')
        ax.set_title('GNN Feature Space (PCA)', fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        
    def _plot_prediction_intervals(self, ax, addresses_gdf):
        """Plot prediction intervals"""
        # Sample addresses
        n_sample = min(100, len(addresses_gdf))
        sample_idx = np.random.choice(len(addresses_gdf), n_sample, replace=False)
        sample = addresses_gdf.iloc[sample_idx].sort_values('svi_predicted')
        
        x_pos = range(len(sample))
        
        # Plot intervals
        ax.fill_between(
            x_pos,
            sample['svi_lower_95'],
            sample['svi_upper_95'],
            alpha=0.3,
            color=self.colors[0],
            label='95% CI'
        )
        
        ax.plot(x_pos, sample['svi_predicted'], 'o-', 
               color=self.colors[1], markersize=4, label='Predicted')
        
        ax.set_xlabel('Address (sorted by prediction)')
        ax.set_ylabel('SVI Score')
        ax.set_title('Prediction Intervals (100 samples)', fontweight='bold')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_summary_stats(self, ax, tracts_gdf, addresses_gdf, gnn_features):
        """Plot summary statistics"""
        ax.axis('off')
        
        # Calculate statistics
        tract_svi = tracts_gdf['RPL_THEMES'].dropna()
        addr_svi = addresses_gdf['svi_predicted']
        
        summary_text = f"""GRANITE Summary Statistics
{'='*30}

Census Tracts:
  Total: {len(tracts_gdf)}
  With SVI: {len(tract_svi)}
  Mean SVI: {tract_svi.mean():.3f} ± {tract_svi.std():.3f}
  Range: [{tract_svi.min():.3f}, {tract_svi.max():.3f}]

Address Predictions:
  Total: {len(addresses_gdf)}
  Mean SVI: {addr_svi.mean():.3f} ± {addr_svi.std():.3f}
  Range: [{addr_svi.min():.3f}, {addr_svi.max():.3f}]
  Avg Uncertainty: {addresses_gdf['svi_sd'].mean():.3f}

GNN Features:
  Dimensions: {gnn_features.shape}
  κ range: [{gnn_features[:, 0].min():.2f}, {gnn_features[:, 0].max():.2f}]
  α range: [{gnn_features[:, 1].min():.2f}, {gnn_features[:, 1].max():.2f}]
  τ range: [{gnn_features[:, 2].min():.2f}, {gnn_features[:, 2].max():.2f}]"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
    def _plot_validation(self, ax, validation_df):
        """Plot validation results"""
        ax.scatter(validation_df['true_svi'], validation_df['predicted_avg'],
                  s=validation_df['n_addresses']*2, alpha=0.6,
                  c=validation_df['error'], cmap='Reds')
        
        # Perfect prediction line
        lims = [0, 1]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect prediction')
        
        # Regression line
        z = np.polyfit(validation_df['true_svi'], validation_df['predicted_avg'], 1)
        p = np.poly1d(z)
        ax.plot(lims, p(lims), 'r-', alpha=0.8, 
               label=f'Fit (R²={validation_df["true_svi"].corr(validation_df["predicted_avg"])**2:.3f})')
        
        ax.set_xlabel('True Tract SVI')
        ax.set_ylabel('Predicted Tract Average')
        ax.set_title('Validation: Tract-Level Comparison', fontweight='bold')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for error
        sm = plt.cm.ScalarMappable(cmap='Reds', 
                                   norm=plt.Normalize(vmin=0, vmax=validation_df['error'].max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Absolute Error')
        
    def _plot_error_distribution(self, ax, validation_df):
        """Plot error distribution"""
        ax.hist(validation_df['error'], bins=20, alpha=0.7, 
               color=self.colors[2], edgecolor='black')
        
        ax.axvline(validation_df['error'].mean(), color='red', 
                  linestyle='--', linewidth=2,
                  label=f'Mean: {validation_df["error"].mean():.3f}')
        
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Number of Tracts')
        ax.set_title('Error Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def plot_training_history(self, training_history, figsize=(12, 4)):
        """Plot GNN training history"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Total loss
        axes[0].plot(training_history['loss'], color=self.colors[0])
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Training Loss')
        axes[0].grid(True, alpha=0.3)
        
        # Spatial loss
        axes[1].plot(training_history['spatial_loss'], color=self.colors[1])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Spatial Loss')
        axes[1].set_title('Spatial Smoothness Loss')
        axes[1].grid(True, alpha=0.3)
        
        # Regularization loss
        axes[2].plot(training_history['reg_loss'], color=self.colors[2])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Regularization Loss')
        axes[2].set_title('Parameter Regularization')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig