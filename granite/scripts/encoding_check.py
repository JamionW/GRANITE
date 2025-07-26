#!/usr/bin/env python
"""
Investigate NLCD class encoding in your file
"""
import rasterio
import numpy as np
import matplotlib.pyplot as plt

def investigate_nlcd_encoding():
    """Figure out what your NLCD classes actually mean"""
    
    nlcd_path = "./data/nlcd_hamilton_county.tif"
    
    with rasterio.open(nlcd_path) as src:
        print(f"NLCD Investigation:")
        print(f"  Shape: {src.shape}")
        print(f"  Data type: {src.dtypes[0]}")
        print(f"  No data value: {src.nodata}")
        
        # Read full image
        full_data = src.read(1)
        
        # Get all unique values
        unique_values = np.unique(full_data)
        value_counts = {val: np.sum(full_data == val) for val in unique_values}
        
        print(f"\nAll unique values and their frequencies:")
        for val in sorted(unique_values):
            count = value_counts[val]
            percent = (count / full_data.size) * 100
            print(f"  {val}: {count:,} pixels ({percent:.1f}%)")
        
        # Check if this might be a mask or processed version
        print(f"\nData range: [{full_data.min()}, {full_data.max()}]")
        print(f"Non-zero values: {sorted([v for v in unique_values if v > 0])}")
        
        # Check different areas of the image
        print(f"\nSampling different regions:")
        h, w = full_data.shape
        
        regions = {
            "Top-left": full_data[:h//3, :w//3],
            "Center": full_data[h//3:2*h//3, w//3:2*w//3], 
            "Bottom-right": full_data[2*h//3:, 2*w//3:]
        }
        
        for region_name, region_data in regions.items():
            region_unique = np.unique(region_data)
            print(f"  {region_name}: {sorted(region_unique)}")
        
        # Create visualization
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Full image
        plt.subplot(1, 3, 1)
        plt.imshow(full_data, cmap='viridis')
        plt.title('Full NLCD Image')
        plt.colorbar()
        
        # Plot 2: Histogram
        plt.subplot(1, 3, 2)
        plt.hist(full_data.flatten(), bins=len(unique_values), alpha=0.7)
        plt.title('Value Distribution')
        plt.xlabel('NLCD Value')
        plt.ylabel('Frequency')
        
        # Plot 3: Center crop
        plt.subplot(1, 3, 3)
        center_crop = full_data[h//4:3*h//4, w//4:3*w//4]
        plt.imshow(center_crop, cmap='viridis')
        plt.title('Center Region')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig('./nlcd_investigation.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Suggest mapping
        print(f"\n🔧 SUGGESTED FIXES:")
        
        if 250 in unique_values:
            print("• Value 250 might be 'no data' - try masking it out")
        
        if max(unique_values) <= 10:
            print("• Values look like indices (0,1,2...) rather than NLCD classes")
            print("• This might be a classified/processed version")
            print("• Try mapping: 0→water, 1→developed, 2→forest, etc.")
        
        if len(unique_values) == 4:
            print("• Only 4 classes suggests simplified land cover")
            print("• Possible mapping:")
            print("  0 → No data/Water")
            print("  1 → Low development (class 21)")  
            print("  2 → High development (class 23)")
            print("  250 → No data")

if __name__ == "__main__":
    investigate_nlcd_encoding()