#!/usr/bin/env python3
"""
GRANITE Address-Level Feature Acquisition
==========================================
Downloads and processes address-level data from four public sources:

1. Microsoft Building Footprints (ODbL) - footprint area, shape complexity
2. FEMA National Flood Hazard Layer (public) - flood zone per address
3. USGS NLCD (public domain) - land cover class, impervious %, tree canopy %
4. OpenStreetMap building tags (ODbL) - building type, levels, material

Output: data/raw/address_features/ with one file per source,
        each keyed by lat/lon for spatial joining to GRANITE addresses.

Usage (from repo root):
    pip install geopandas shapely requests rasterio rasterstats fiona --break-system-packages
    python scripts/acquire_address_features.py

Notes:
    - NLCD requires a ~1 GB raster download (one-time); script will prompt.
    - FEMA download can be slow; the file is ~50-100 MB for Hamilton County.
    - MS Building Footprints TN file is ~700 MB unzipped.
    - All outputs use EPSG:4326.
"""

import os
import sys
import json
import time
import zipfile
import logging
import argparse
import tempfile
import requests
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("acquire")

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
# Hamilton County, TN bounding box (EPSG:4326)
HAMILTON_BBOX = {
    "min_lon": -85.55,
    "max_lon": -84.95,
    "min_lat": 34.98,
    "max_lat": 35.25,
}
HAMILTON_COUNTY_FIPS = "47065"
STATE_FIPS = "47"
COUNTY_FIPS = "065"

# Default output directory (relative to repo root)
DEFAULT_OUTPUT_DIR = "data/raw/address_features"

# Download URLs
MS_BUILDINGS_URL = (
    "https://minedbuildings.z5.web.core.windows.net/"
    "legacy/usbuildings-v2/Tennessee.geojson.zip"
)
FEMA_NFHL_URL = (
    "https://msc.fema.gov/portal/downloadProduct"
    f"?productID=NFHL_{HAMILTON_COUNTY_FIPS}C"
)
# NLCD 2021 land cover, impervious surface, and tree canopy
# These are CONUS-wide Cloud-Optimized GeoTIFFs from MRLC/ScienceBase.
# The script can read just the Hamilton County window from the COG
# if the file is hosted remotely, or from a local download.
NLCD_PRODUCTS = {
    "land_cover": {
        "description": "NLCD 2021 Land Cover (30m, 16-class)",
        "sciencebase_url": "https://www.sciencebase.gov/catalog/item/625e93b0d34e85fa62b7f3e9",
        "filename": "nlcd_2021_land_cover_l48_20230630.img",
        "manual_note": (
            "Download 'NLCD 2021 Land Cover' from https://www.mrlc.gov/data "
            "(filter: CONUS, Land Cover, 2021). Place the .img file in "
            "data/raw/address_features/nlcd/"
        ),
    },
    "impervious": {
        "description": "NLCD 2021 Impervious Surface (30m, 0-100%)",
        "filename": "nlcd_2021_impervious_l48_20230630.img",
        "manual_note": (
            "Download 'NLCD 2021 Impervious' from https://www.mrlc.gov/data. "
            "Place the .img file in data/raw/address_features/nlcd/"
        ),
    },
    "tree_canopy": {
        "description": "NLCD 2021 Tree Canopy Cover (30m, 0-100%)",
        "filename": "nlcd_2021_treecanopy_l48_20230630.img",
        "manual_note": (
            "Download 'NLCD 2021 USFS Tree Canopy Cover' from "
            "https://www.mrlc.gov/data. Place the .img file in "
            "data/raw/address_features/nlcd/"
        ),
    },
}

# Overpass API endpoint (same one GRANITE uses for grocery stores)
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def download_file(url, dest_path, description="file", timeout=300):
    """Download a file with progress logging. Returns True on success."""
    if os.path.exists(dest_path):
        log.info(f"Already exists: {dest_path}")
        return True

    log.info(f"Downloading {description}...")
    log.info(f"  URL: {url}")
    log.info(f"  Dest: {dest_path}")

    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        log.error(f"Download failed: {e}")
        return False

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 1024 * 1024  # 1 MB

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = (downloaded / total) * 100
                mb = downloaded / (1024 * 1024)
                if int(mb) % 50 == 0 and int(mb) > 0:
                    log.info(f"  {mb:.0f} MB ({pct:.0f}%)")

    log.info(f"  Done: {downloaded / (1024*1024):.1f} MB")
    return True


# ===================================================================
# Source 1: Microsoft Building Footprints
# ===================================================================
def acquire_ms_buildings(output_dir):
    """
    Download Microsoft Building Footprints for Tennessee,
    filter to Hamilton County bounding box, compute derived features,
    and save as GeoJSON.

    Per-building attributes extracted:
        - footprint_area_m2: polygon area in sq meters
        - centroid_lat, centroid_lon: building centroid
        - vertex_count: polygon complexity proxy
        - capture_date: imagery vintage (where available)
    """
    import geopandas as gpd
    from shapely.geometry import box

    buildings_dir = ensure_dir(os.path.join(output_dir, "ms_buildings"))
    output_file = os.path.join(buildings_dir, "hamilton_buildings.geojson")

    if os.path.exists(output_file):
        log.info(f"MS Buildings already processed: {output_file}")
        return output_file

    # Download TN state file
    zip_path = os.path.join(buildings_dir, "Tennessee.geojson.zip")
    if not download_file(MS_BUILDINGS_URL, zip_path, "MS Building Footprints (TN)"):
        return None

    # Extract
    geojson_path = os.path.join(buildings_dir, "Tennessee.geojson")
    if not os.path.exists(geojson_path):
        log.info("Extracting Tennessee.geojson...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(buildings_dir)
        log.info("Extraction complete.")

    # Filter to Hamilton County bbox
    log.info("Filtering to Hamilton County bounding box...")
    bbox = box(
        HAMILTON_BBOX["min_lon"], HAMILTON_BBOX["min_lat"],
        HAMILTON_BBOX["max_lon"], HAMILTON_BBOX["max_lat"],
    )

    # Read in chunks to manage memory (TN file is large)
    # geopandas bbox filter on read
    gdf = gpd.read_file(geojson_path, bbox=bbox)
    log.info(f"  {len(gdf)} buildings in Hamilton County bbox")

    if len(gdf) == 0:
        log.warning("No buildings found in bbox. Check coordinates.")
        return None

    # Compute derived features
    # Project to UTM 16N (EPSG:32616) for area calculation
    gdf_proj = gdf.to_crs(epsg=32616)
    gdf["footprint_area_m2"] = gdf_proj.geometry.area
    gdf["centroid_lon"] = gdf.geometry.centroid.x
    gdf["centroid_lat"] = gdf.geometry.centroid.y
    gdf["vertex_count"] = gdf.geometry.apply(
        lambda g: len(g.exterior.coords) if g.geom_type == "Polygon" else 0
    )

    # Save filtered result
    gdf.to_file(output_file, driver="GeoJSON")
    log.info(f"  Saved: {output_file}")

    # Clean up the large state file to save disk
    if os.path.exists(geojson_path):
        os.remove(geojson_path)
        log.info("  Removed full TN file to save space.")

    return output_file


# ===================================================================
# Source 2: FEMA National Flood Hazard Layer
# ===================================================================
def acquire_fema_nfhl(output_dir):
    """
    Download FEMA NFHL for Hamilton County, TN.
    Extract the S_FLD_HAZ_AR (flood hazard area) layer.

    Per-polygon attributes:
        - FLD_ZONE: flood zone code (A, AE, X, VE, etc.)
        - ZONE_SUBTY: zone subtype
        - SFHA_TF: whether in Special Flood Hazard Area (T/F)
        - STATIC_BFE: base flood elevation (where available)
    """
    import geopandas as gpd

    fema_dir = ensure_dir(os.path.join(output_dir, "fema_nfhl"))
    output_file = os.path.join(fema_dir, "hamilton_flood_zones.geojson")

    if os.path.exists(output_file):
        log.info(f"FEMA NFHL already processed: {output_file}")
        return output_file

    zip_path = os.path.join(fema_dir, f"NFHL_{HAMILTON_COUNTY_FIPS}C.zip")
    if not download_file(FEMA_NFHL_URL, zip_path, "FEMA NFHL (Hamilton County)", timeout=600):
        log.warning(
            "FEMA download may require browser access. "
            "Try manually: https://msc.fema.gov/portal/advanceSearch "
            "-> State: Tennessee, County: Hamilton -> Download NFHL Database"
        )
        return None

    # Extract
    extract_dir = os.path.join(fema_dir, "extracted")
    if not os.path.exists(extract_dir):
        log.info("Extracting NFHL archive...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    # Find the flood hazard area shapefile or geodatabase layer
    # NFHL comes as a file geodatabase (.gdb) or shapefiles
    flood_layer = None
    for root, dirs, files in os.walk(extract_dir):
        # Check for .gdb directory
        for d in dirs:
            if d.endswith(".gdb"):
                gdb_path = os.path.join(root, d)
                log.info(f"Found geodatabase: {gdb_path}")
                try:
                    import fiona
                    layers = fiona.listlayers(gdb_path)
                    log.info(f"  Layers: {layers}")
                    if "S_FLD_HAZ_AR" in layers:
                        flood_layer = gpd.read_file(gdb_path, layer="S_FLD_HAZ_AR")
                        log.info(f"  Loaded S_FLD_HAZ_AR: {len(flood_layer)} polygons")
                except Exception as e:
                    log.warning(f"  Could not read .gdb: {e}")

        # Fallback: look for shapefiles
        if flood_layer is None:
            for f in files:
                if "S_FLD_HAZ_AR" in f and f.endswith(".shp"):
                    shp_path = os.path.join(root, f)
                    flood_layer = gpd.read_file(shp_path)
                    log.info(f"  Loaded shapefile: {len(flood_layer)} polygons")
                    break

    if flood_layer is None:
        log.error("Could not find S_FLD_HAZ_AR layer in NFHL data.")
        log.info("Check extracted contents manually in: " + extract_dir)
        return None

    # Keep relevant columns
    keep_cols = ["FLD_ZONE", "ZONE_SUBTY", "SFHA_TF", "STATIC_BFE", "geometry"]
    available = [c for c in keep_cols if c in flood_layer.columns]
    flood_layer = flood_layer[available].to_crs(epsg=4326)

    flood_layer.to_file(output_file, driver="GeoJSON")
    log.info(f"  Saved: {output_file}")
    return output_file


# ===================================================================
# Source 3: USGS NLCD (requires manual raster download)
# ===================================================================
def acquire_nlcd(output_dir):
    """
    Sample NLCD rasters at Hamilton County address locations.
    Requires manual download of NLCD GeoTIFF/IMG files first.

    Per-address attributes (from point sampling):
        - nlcd_land_cover: 16-class land cover code
        - nlcd_impervious_pct: 0-100 impervious surface percentage
        - nlcd_tree_canopy_pct: 0-100 tree canopy percentage
    """
    nlcd_dir = ensure_dir(os.path.join(output_dir, "nlcd"))

    # Check for raster files
    found_any = False
    for product_key, info in NLCD_PRODUCTS.items():
        raster_path = os.path.join(nlcd_dir, info["filename"])
        # Also check for .tif variant
        tif_path = raster_path.replace(".img", ".tif")

        if os.path.exists(raster_path) or os.path.exists(tif_path):
            found_any = True
            log.info(f"NLCD {product_key}: found locally")
        else:
            log.warning(f"NLCD {product_key}: not found.")
            log.info(f"  {info['manual_note']}")

    if not found_any:
        log.info("")
        log.info("=" * 60)
        log.info("NLCD rasters require manual download (~1 GB each).")
        log.info("Visit https://www.mrlc.gov/data and download:")
        log.info("  1. Land Cover (CONUS, 2021)")
        log.info("  2. Impervious Surface (CONUS, 2021)")
        log.info("  3. Tree Canopy Cover (CONUS, 2021)")
        log.info(f"Place .img or .tif files in: {nlcd_dir}")
        log.info("")
        log.info("After downloading, rerun this script to process.")
        log.info("=" * 60)

    # Write a helper script for sampling (runs after rasters are present)
    sample_script = os.path.join(nlcd_dir, "sample_nlcd.py")
    if not os.path.exists(sample_script):
        _write_nlcd_sample_script(sample_script, nlcd_dir)
        log.info(f"  Wrote sampling helper: {sample_script}")

    return nlcd_dir


def _write_nlcd_sample_script(script_path, nlcd_dir):
    """Write a standalone script for sampling NLCD rasters at address points."""
    content = '''#!/usr/bin/env python3
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
REPO_ROOT = os.path.abspath(os.path.join(NLCD_DIR, "..", "..", "..", ".."))

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
'''
    with open(script_path, "w") as f:
        f.write(content)


# ===================================================================
# Source 4: OpenStreetMap Building Tags
# ===================================================================
def acquire_osm_buildings(output_dir):
    """
    Query Overpass API for building polygons with attribute tags
    in Hamilton County.

    Per-building attributes (where tagged):
        - building: type (house, apartments, commercial, industrial, etc.)
        - building:levels: number of floors
        - building:material: construction material
        - building:age / start_date: construction date
        - amenity: if building has an amenity tag
    """
    osm_dir = ensure_dir(os.path.join(output_dir, "osm_buildings"))
    output_file = os.path.join(osm_dir, "hamilton_osm_buildings.json")

    if os.path.exists(output_file):
        log.info(f"OSM buildings already downloaded: {output_file}")
        return output_file

    # Overpass query: buildings with any useful tags in Hamilton County bbox
    bbox_str = (
        f"{HAMILTON_BBOX['min_lat']},{HAMILTON_BBOX['min_lon']},"
        f"{HAMILTON_BBOX['max_lat']},{HAMILTON_BBOX['max_lon']}"
    )

    query = f"""
    [out:json][timeout:180];
    (
      way["building"]({bbox_str});
      relation["building"]({bbox_str});
    );
    out center tags;
    """

    log.info("Querying Overpass API for OSM building data...")
    log.info(f"  Bbox: {bbox_str}")

    try:
        resp = requests.post(
            OVERPASS_URL,
            data={"data": query},
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        log.error(f"Overpass query failed: {e}")
        log.info("The Overpass API may be rate-limited. Try again in a few minutes.")
        return None

    elements = data.get("elements", [])
    log.info(f"  Retrieved {len(elements)} building elements")

    # Extract relevant tags
    buildings = []
    for el in elements:
        tags = el.get("tags", {})
        center = el.get("center", {})
        if not center:
            continue

        record = {
            "osm_id": el.get("id"),
            "lat": center.get("lat"),
            "lon": center.get("lon"),
            "building_type": tags.get("building", "yes"),
            "levels": tags.get("building:levels"),
            "material": tags.get("building:material"),
            "start_date": tags.get("start_date"),
            "amenity": tags.get("amenity"),
            "shop": tags.get("shop"),
            "addr_street": tags.get("addr:street"),
            "addr_housenumber": tags.get("addr:housenumber"),
        }
        buildings.append(record)

    with open(output_file, "w") as f:
        json.dump(buildings, f, indent=2)

    # Stats
    typed = sum(1 for b in buildings if b["building_type"] != "yes")
    leveled = sum(1 for b in buildings if b["levels"] is not None)
    dated = sum(1 for b in buildings if b["start_date"] is not None)
    log.info(f"  With specific type: {typed}/{len(buildings)}")
    log.info(f"  With levels: {leveled}/{len(buildings)}")
    log.info(f"  With start_date: {dated}/{len(buildings)}")
    log.info(f"  Saved: {output_file}")

    return output_file


# ===================================================================
# Spatial join helper
# ===================================================================
def write_join_script(output_dir):
    """
    Write a helper script that joins all acquired features
    to GRANITE's address points.
    """
    script_path = os.path.join(output_dir, "join_to_addresses.py")
    if os.path.exists(script_path):
        return

    content = '''#!/usr/bin/env python3
"""
Join acquired address-level features to GRANITE address points.
Produces a single CSV with one row per address and all feature columns.

Usage (from repo root):
    python data/raw/address_features/join_to_addresses.py
"""
import os
import sys
import json
import numpy as np

try:
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point
except ImportError:
    print("Install: pip install geopandas pandas shapely --break-system-packages")
    sys.exit(1)

FEATURES_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(FEATURES_DIR, "..", "..", ".."))


def main():
    # Load GRANITE addresses
    addr_path = os.path.join(REPO_ROOT, "data", "raw", "chattanooga.geojson")
    if not os.path.exists(addr_path):
        print(f"Address file not found: {addr_path}")
        sys.exit(1)

    addresses = gpd.read_file(addr_path).to_crs(epsg=4326)
    print(f"Loaded {len(addresses)} addresses")

    # 1. Join MS Building Footprints (nearest building to each address)
    bldg_path = os.path.join(FEATURES_DIR, "ms_buildings", "hamilton_buildings.geojson")
    if os.path.exists(bldg_path):
        print("Joining MS Building Footprints...")
        buildings = gpd.read_file(bldg_path).to_crs(epsg=4326)
        # Spatial join: nearest building footprint to each address
        joined = gpd.sjoin_nearest(
            addresses[["geometry"]],
            buildings[["geometry", "footprint_area_m2", "vertex_count"]],
            how="left",
            max_distance=0.001,  # ~100m in degrees
        )
        addresses["bldg_footprint_m2"] = joined["footprint_area_m2"].values
        addresses["bldg_vertex_count"] = joined["vertex_count"].values
        print(f"  Matched: {addresses['bldg_footprint_m2'].notna().sum()}/{len(addresses)}")

    # 2. Join FEMA Flood Zones (point-in-polygon)
    flood_path = os.path.join(FEATURES_DIR, "fema_nfhl", "hamilton_flood_zones.geojson")
    if os.path.exists(flood_path):
        print("Joining FEMA Flood Zones...")
        floods = gpd.read_file(flood_path).to_crs(epsg=4326)
        joined = gpd.sjoin(addresses[["geometry"]], floods, how="left", predicate="within")
        # Take first match per address (in case of overlapping zones)
        joined = joined[~joined.index.duplicated(keep="first")]
        if "FLD_ZONE" in joined.columns:
            addresses["flood_zone"] = joined["FLD_ZONE"].values
        if "SFHA_TF" in joined.columns:
            addresses["in_sfha"] = (joined["SFHA_TF"] == "T").astype(int).values
        matched = addresses["flood_zone"].notna().sum() if "flood_zone" in addresses.columns else 0
        print(f"  Matched: {matched}/{len(addresses)}")

    # 3. Join OSM building tags (nearest OSM building to each address)
    osm_path = os.path.join(FEATURES_DIR, "osm_buildings", "hamilton_osm_buildings.json")
    if os.path.exists(osm_path):
        print("Joining OSM building tags...")
        with open(osm_path) as f:
            osm_data = json.load(f)
        osm_gdf = gpd.GeoDataFrame(
            osm_data,
            geometry=[Point(b["lon"], b["lat"]) for b in osm_data],
            crs="EPSG:4326",
        )
        joined = gpd.sjoin_nearest(
            addresses[["geometry"]],
            osm_gdf[["geometry", "building_type", "levels"]],
            how="left",
            max_distance=0.001,
        )
        addresses["osm_building_type"] = joined["building_type"].values
        addresses["osm_levels"] = pd.to_numeric(joined["levels"], errors="coerce").values
        matched = addresses["osm_building_type"].notna().sum()
        print(f"  Matched: {matched}/{len(addresses)}")

    # 4. NLCD samples (if available)
    nlcd_path = os.path.join(FEATURES_DIR, "nlcd", "hamilton_nlcd_samples.json")
    if os.path.exists(nlcd_path):
        print("Joining NLCD samples...")
        with open(nlcd_path) as f:
            nlcd = json.load(f)
        for key in ["nlcd_land_cover", "nlcd_impervious_pct", "nlcd_tree_canopy_pct"]:
            if key in nlcd:
                addresses[key] = nlcd[key]
        print(f"  Added NLCD columns")

    # Save combined features
    out_path = os.path.join(FEATURES_DIR, "combined_address_features.csv")
    # Drop geometry for CSV output, keep lat/lon
    addresses["lon"] = addresses.geometry.x
    addresses["lat"] = addresses.geometry.y
    addresses.drop(columns=["geometry"]).to_csv(out_path, index=False)
    print(f"\\nSaved combined features: {out_path}")
    print(f"Columns: {list(addresses.columns)}")
    print(f"Shape: {addresses.shape}")


if __name__ == "__main__":
    main()
'''
    with open(script_path, "w") as f:
        f.write(content)
    log.info(f"Wrote join helper: {script_path}")


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Acquire address-level features for GRANITE"
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--skip", nargs="*", default=[],
        choices=["buildings", "fema", "nlcd", "osm"],
        help="Skip specific data sources"
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    ensure_dir(output_dir)

    log.info("=" * 60)
    log.info("GRANITE Address-Level Feature Acquisition")
    log.info("=" * 60)
    log.info(f"Output: {output_dir}")
    log.info(f"Study area: Hamilton County, TN (FIPS {HAMILTON_COUNTY_FIPS})")
    log.info("")

    results = {}

    # 1. Microsoft Building Footprints
    if "buildings" not in args.skip:
        log.info("[1/4] Microsoft Building Footprints")
        results["ms_buildings"] = acquire_ms_buildings(output_dir)
    else:
        log.info("[1/4] Skipping MS Buildings")

    # 2. FEMA NFHL
    if "fema" not in args.skip:
        log.info("")
        log.info("[2/4] FEMA National Flood Hazard Layer")
        results["fema_nfhl"] = acquire_fema_nfhl(output_dir)
    else:
        log.info("[2/4] Skipping FEMA NFHL")

    # 3. NLCD
    if "nlcd" not in args.skip:
        log.info("")
        log.info("[3/4] USGS NLCD Land Cover / Impervious / Canopy")
        results["nlcd"] = acquire_nlcd(output_dir)
    else:
        log.info("[3/4] Skipping NLCD")

    # 4. OSM Building Tags
    if "osm" not in args.skip:
        log.info("")
        log.info("[4/4] OpenStreetMap Building Tags")
        results["osm_buildings"] = acquire_osm_buildings(output_dir)
    else:
        log.info("[4/4] Skipping OSM Buildings")

    # Write join helper
    write_join_script(output_dir)

    # Summary
    log.info("")
    log.info("=" * 60)
    log.info("Acquisition Summary")
    log.info("=" * 60)
    for source, path in results.items():
        status = "OK" if path else "NEEDS ATTENTION"
        log.info(f"  {source}: {status}")
        if path:
            log.info(f"    {path}")

    log.info("")
    log.info("Next steps:")
    log.info("  1. If NLCD rasters are missing, download from mrlc.gov (see above)")
    log.info("  2. Run NLCD sampling:  python data/raw/address_features/nlcd/sample_nlcd.py")
    log.info("  3. Run spatial join:   python data/raw/address_features/join_to_addresses.py")
    log.info("  4. Integrate combined_address_features.csv into GRANITE feature pipeline")


if __name__ == "__main__":
    main()