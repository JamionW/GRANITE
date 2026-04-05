#!/usr/bin/env python3
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
REPO_ROOT = os.path.abspath(os.path.join(FEATURES_DIR, "..", "..", "..", ".."))


def main():
    # Load GRANITE addresses
    addr_path = os.path.join(REPO_ROOT, "data", "raw", "chattanooga.geojson")
    if not os.path.exists(addr_path):
        print(f"Address file not found: {addr_path}")
        sys.exit(1)

    addresses = gpd.read_file(addr_path).to_crs(epsg=4326)
    if "hash" not in addresses.columns:
        print("WARNING: no 'hash' column in address data")
    print(f"Loaded {len(addresses)} addresses")

    # 1. Join MS Building Footprints (nearest building to each address)
    bldg_path = os.path.join(FEATURES_DIR, "ms_buildings", "hamilton_buildings.geojson")
    if os.path.exists(bldg_path):
        print("Joining MS Building Footprints...")
        buildings = gpd.read_file(bldg_path).to_crs(epsg=32616)
        addr_proj = addresses[["geometry"]].to_crs(epsg=32616)
        joined = gpd.sjoin_nearest(
            addr_proj,
            buildings[["geometry", "footprint_area_m2", "vertex_count"]],
            how="left",
            max_distance=100,  # 100 meters
        )
        joined = joined[~joined.index.duplicated(keep="first")]
        addresses["bldg_footprint_m2"] = joined["footprint_area_m2"].reindex(addresses.index).values
        addresses["bldg_vertex_count"] = joined["vertex_count"].reindex(addresses.index).values
        print(f"  Matched: {addresses['bldg_footprint_m2'].notna().sum()}/{len(addresses)}")

    # 2. Join FEMA Flood Zones (point-in-polygon)
    flood_path = os.path.join(FEATURES_DIR, "fema_nfhl", "hamilton_flood_zones.geojson")
    if os.path.exists(flood_path):
        print("Joining FEMA Flood Zones...")
        floods = gpd.read_file(flood_path).to_crs(epsg=4326)
        joined = gpd.sjoin(addresses[["geometry"]], floods, how="left", predicate="within")
        joined = joined[~joined.index.duplicated(keep="first")]
        if "FLD_ZONE" in joined.columns:
            addresses["flood_zone"] = joined["FLD_ZONE"].reindex(addresses.index).values
        if "SFHA_TF" in joined.columns:
            addresses["in_sfha"] = (joined["SFHA_TF"].reindex(addresses.index) == "T").astype(int).values
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
        addr_proj = addresses[["geometry"]].to_crs(epsg=32616)
        osm_proj = osm_gdf[["geometry", "building_type", "levels"]].to_crs(epsg=32616)
        joined = gpd.sjoin_nearest(
            addr_proj,
            osm_proj,
            how="left",
            max_distance=100,
        )
        joined = joined[~joined.index.duplicated(keep="first")]
        addresses["osm_building_type"] = joined["building_type"].reindex(addresses.index).values
        addresses["osm_levels"] = pd.to_numeric(joined["levels"].reindex(addresses.index), errors="coerce").values
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

    # 5. Join Hamilton County parcel data (spatial: address point in parcel polygon)
    parcel_path = os.path.join(REPO_ROOT, "data", "raw", "parcels", "parcels.shp")
    if os.path.exists(parcel_path):
        print("Joining Hamilton County parcel data...")
        parcels = gpd.read_file(parcel_path)
        parcels = parcels[parcels["APPVALUE"] > 0].copy()
        parcels = parcels.to_crs(epsg=4326)

        keep_parcel = [
            "geometry", "APPVALUE", "ASSVALUE", "LANDVALUE", "BUILDVALUE",
            "CALCACRES", "LUCODE", "PROPTYPE", "SALE1DATE", "SALE1CONSD",
        ]
        available_pcols = [c for c in keep_parcel if c in parcels.columns]
        parcels = parcels[available_pcols]

        joined = gpd.sjoin(
            addresses[["geometry"]],
            parcels,
            how="left",
            predicate="within",
        )
        joined = joined[~joined.index.duplicated(keep="first")]

        for col in available_pcols:
            if col == "geometry":
                continue
            addresses[col] = joined[col].reindex(addresses.index).values

        matched = addresses["APPVALUE"].notna().sum()
        print(f"  Matched: {matched}/{len(addresses)}")

        # derived features
        addresses["log_appvalue"] = np.log1p(
            pd.to_numeric(addresses["APPVALUE"], errors="coerce").fillna(0)
        )
        addresses["build_to_land_ratio"] = (
            pd.to_numeric(addresses["BUILDVALUE"], errors="coerce").fillna(0)
            / pd.to_numeric(addresses["LANDVALUE"], errors="coerce").replace(0, np.nan)
        )
        addresses["log_acres"] = np.log1p(
            pd.to_numeric(addresses["CALCACRES"], errors="coerce").fillna(0)
        )
    else:
        print(f"Parcel shapefile not found at {parcel_path}, skipping")

    # Save combined features
    out_path = os.path.join(FEATURES_DIR, "combined_address_features.csv")
    addresses["lon"] = addresses.geometry.x
    addresses["lat"] = addresses.geometry.y
    out_df = addresses.drop(columns=["geometry"])
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved combined features: {out_path}")
    print(f"Columns: {list(out_df.columns)}")
    print(f"Shape: {out_df.shape}")


if __name__ == "__main__":
    main()
