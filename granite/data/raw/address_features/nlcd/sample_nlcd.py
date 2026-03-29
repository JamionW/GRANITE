#!/usr/bin/env python3
"""
Sample NLCD rasters at GRANITE address locations.
Run after placing NLCD .img/.tif files in this directory.

Usage (from repo root):
    python data/raw/address_features/nlcd/sample_nlcd.py
"""
import os
import sys
import glob
import json
import numpy as np

try:
    import rasterio
    from rasterio.transform import rowcol
    import geopandas as gpd
except ImportError:
    print("Install: pip install rasterio geopandas --break-system-packages")
    sys.exit(1)

NLCD_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(NLCD_DIR, "..", "..", "..", "..", ".."))

# NLCD land cover class descriptions
NLCD_CLASSES = {
    11: "open_water", 12: "ice_snow",
    21: "developed_open", 22: "developed_low", 23: "developed_med", 24: "developed_high",
    31: "barren",
    41: "deciduous_forest", 42: "evergreen_forest", 43: "mixed_forest",
    51: "dwarf_scrub", 52: "shrub_scrub",
    71: "grassland", 72: "sedge", 73: "lichens", 74: "moss",
    81: "pasture_hay", 82: "cultivated_crops",
    90: "woody_wetlands", 95: "herbaceous_wetlands",
}

def sample_raster_at_points(raster_path, points_lon, points_lat):
    """Sample a raster at given lon/lat coordinates. Returns array of values."""
    with rasterio.open(raster_path) as src:
        values = []
        for lon, lat in zip(points_lon, points_lat):
            try:
                row, col = rowcol(src.transform, lon, lat)
                if 0 <= row < src.height and 0 <= col < src.width:
                    val = src.read(1, window=((row, row+1), (col, col+1)))[0, 0]
                    values.append(int(val))
                else:
                    values.append(-1)
            except Exception:
                values.append(-1)
    return np.array(values)


def main():
    # Load GRANITE addresses
    addr_path = os.path.join(REPO_ROOT, "data", "raw", "chattanooga.geojson")
    if not os.path.exists(addr_path):
        print(f"Address file not found: {addr_path}")
        print("Adjust path to your GRANITE address data.")
        sys.exit(1)

    addresses = gpd.read_file(addr_path)
    lons = addresses.geometry.x.values
    lats = addresses.geometry.y.values
    print(f"Loaded {len(addresses)} addresses")

    results = {"lon": lons.tolist(), "lat": lats.tolist()}

    # Find and sample each raster
    raster_files = glob.glob(os.path.join(NLCD_DIR, "*.img"))
    raster_files += glob.glob(os.path.join(NLCD_DIR, "*.tif"))

    for rpath in raster_files:
        fname = os.path.basename(rpath).lower()
        if "land_cover" in fname:
            key = "nlcd_land_cover"
        elif "impervious" in fname:
            key = "nlcd_impervious_pct"
        elif "treecanopy" in fname or "tree_canopy" in fname:
            key = "nlcd_tree_canopy_pct"
        else:
            key = os.path.splitext(os.path.basename(rpath))[0]

        print(f"Sampling {key} from {os.path.basename(rpath)}...")
        vals = sample_raster_at_points(rpath, lons, lats)
        results[key] = vals.tolist()
        valid = np.sum(vals >= 0)
        print(f"  {valid}/{len(vals)} valid samples")

    # Save
    out_path = os.path.join(NLCD_DIR, "hamilton_nlcd_samples.json")
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
