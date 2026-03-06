#!/bin/bash
# Download external data sources for GRANITE
# Run from repo root: bash granite/scripts/setup_data.sh
#
# Downloads:
#   1. LEHD employment data (Census LODES)
#   2. TIGER block geometries (for LEHD geocoding)
#   3. Healthcare facilities (CMS Hospital Compare)
#   4. Grocery stores (OpenStreetMap Overpass API)

set -e

DATA_DIR="./data/raw"

echo "=== GRANITE Data Setup ==="
echo ""

# -------------------------------------------------------
# 1. LEHD Workplace Area Characteristics
# -------------------------------------------------------
echo "--- LEHD Employment Data ---"
LEHD_DIR="$DATA_DIR/lehd"
mkdir -p "$LEHD_DIR"

LEHD_FILE="$LEHD_DIR/tn_wac_S000_JT00_2021.csv"
if [ -f "$LEHD_FILE" ]; then
    echo "  Already exists: $LEHD_FILE"
else
    echo "  Downloading LEHD WAC 2021 (~40MB compressed)..."
    wget -q --show-progress -O "${LEHD_FILE}.gz" \
        "https://lehd.ces.census.gov/data/lodes/LODES8/tn/wac/tn_wac_S000_JT00_2021.csv.gz"
    echo "  Decompressing..."
    gunzip "${LEHD_FILE}.gz"
    echo "  Done: $(wc -l < "$LEHD_FILE") records"
fi

# -------------------------------------------------------
# 2. TIGER Block Geometries (needed to geocode LEHD)
# -------------------------------------------------------
echo ""
echo "--- TIGER Block Geometries ---"
BLOCK_FILE="$LEHD_DIR/tl_2021_47_tabblock20.shp"
if [ -f "$BLOCK_FILE" ]; then
    echo "  Already exists: $BLOCK_FILE"
else
    echo "  Downloading Tennessee block geometries (~130MB)..."
    wget -q --show-progress -O "$LEHD_DIR/tl_2021_47_tabblock20.zip" \
        "https://www2.census.gov/geo/tiger/TIGER2021/TABBLOCK20/tl_2021_47_tabblock20.zip"
    echo "  Extracting..."
    cd "$LEHD_DIR"
    unzip -o tl_2021_47_tabblock20.zip
    rm -f tl_2021_47_tabblock20.zip
    cd - > /dev/null
    echo "  Done"
fi

# -------------------------------------------------------
# 3. Healthcare Facilities (CMS)
# -------------------------------------------------------
echo ""
echo "--- Healthcare Facilities ---"
HC_DIR="$DATA_DIR/healthcare"
mkdir -p "$HC_DIR"

HC_FILE="$HC_DIR/hamilton_county_healthcare.csv"
if [ -f "$HC_FILE" ]; then
    echo "  Already exists: $HC_FILE"
else
    echo "  Downloading from CMS Hospital Compare API..."
    python3 << 'PYEOF'
import requests
import csv
import os

# CMS Hospital Compare dataset (Socrata API, no auth required)
# Filter to Hamilton County TN hospitals
url = "https://data.cms.gov/provider-data/api/1/datastore/query/xubh-q36u/0"
params = {
    "conditions[0][property]": "state",
    "conditions[0][value]": "TN",
    "conditions[0][operator]": "=",
    "limit": 500,
    "offset": 0
}

print("  Querying CMS API...")
response = requests.get(url, params=params, timeout=60)

if response.status_code != 200:
    # Fallback: use well-known Hamilton County hospitals
    print("  CMS API unavailable, using known facility list...")
    facilities = [
        {"facility_name": "Erlanger Medical Center", "latitude": 35.0458, "longitude": -85.3094, "hospital_type": "Acute Care"},
        {"facility_name": "CHI Memorial Hospital", "latitude": 35.0225, "longitude": -85.2350, "hospital_type": "Acute Care"},
        {"facility_name": "Parkridge Medical Center", "latitude": 35.0314, "longitude": -85.2194, "hospital_type": "Acute Care"},
        {"facility_name": "Erlanger East Hospital", "latitude": 35.0867, "longitude": -85.1308, "hospital_type": "Acute Care"},
        {"facility_name": "Erlanger North Hospital", "latitude": 35.1142, "longitude": -85.2753, "hospital_type": "Acute Care"},
        {"facility_name": "Parkridge East Hospital", "latitude": 35.0831, "longitude": -85.1531, "hospital_type": "Acute Care"},
        {"facility_name": "Parkridge Valley Hospital", "latitude": 35.0267, "longitude": -85.1542, "hospital_type": "Psychiatric"},
        {"facility_name": "CHI Memorial Hixson", "latitude": 35.1117, "longitude": -85.2342, "hospital_type": "Acute Care"},
        {"facility_name": "Kindred Hospital Chattanooga", "latitude": 35.0397, "longitude": -85.2506, "hospital_type": "Long Term Care"},
        {"facility_name": "Siskin Hospital for Physical Rehab", "latitude": 35.0383, "longitude": -85.3006, "hospital_type": "Rehabilitation"},
        {"facility_name": "Moccasin Bend Mental Health Institute", "latitude": 35.0639, "longitude": -85.3461, "hospital_type": "Psychiatric"},
        {"facility_name": "T.C. Thompson Children's Hospital", "latitude": 35.0461, "longitude": -85.3089, "hospital_type": "Children's"},
    ]
else:
    data = response.json()
    results = data.get("results", [])

    # Filter to Hamilton County area (approximate bbox)
    facilities = []
    for r in results:
        try:
            lat = float(r.get("latitude", 0))
            lon = float(r.get("longitude", 0))
        except (ValueError, TypeError):
            continue
        # Hamilton County bounds
        if 34.95 < lat < 35.22 and -85.45 < lon < -85.05:
            facilities.append({
                "facility_name": r.get("hospital_name", r.get("facility_name", "Unknown")),
                "latitude": lat,
                "longitude": lon,
                "hospital_type": r.get("hospital_type", "Unknown"),
            })

    if len(facilities) == 0:
        print("  No facilities matched Hamilton County filter, using known list...")
        facilities = [
            {"facility_name": "Erlanger Medical Center", "latitude": 35.0458, "longitude": -85.3094, "hospital_type": "Acute Care"},
            {"facility_name": "CHI Memorial Hospital", "latitude": 35.0225, "longitude": -85.2350, "hospital_type": "Acute Care"},
            {"facility_name": "Parkridge Medical Center", "latitude": 35.0314, "longitude": -85.2194, "hospital_type": "Acute Care"},
            {"facility_name": "Erlanger East Hospital", "latitude": 35.0867, "longitude": -85.1308, "hospital_type": "Acute Care"},
            {"facility_name": "Erlanger North Hospital", "latitude": 35.1142, "longitude": -85.2753, "hospital_type": "Acute Care"},
            {"facility_name": "Parkridge East Hospital", "latitude": 35.0831, "longitude": -85.1531, "hospital_type": "Acute Care"},
            {"facility_name": "Parkridge Valley Hospital", "latitude": 35.0267, "longitude": -85.1542, "hospital_type": "Psychiatric"},
            {"facility_name": "CHI Memorial Hixson", "latitude": 35.1117, "longitude": -85.2342, "hospital_type": "Acute Care"},
            {"facility_name": "Kindred Hospital Chattanooga", "latitude": 35.0397, "longitude": -85.2506, "hospital_type": "Long Term Care"},
            {"facility_name": "Siskin Hospital for Physical Rehab", "latitude": 35.0383, "longitude": -85.3006, "hospital_type": "Rehabilitation"},
            {"facility_name": "Moccasin Bend Mental Health Institute", "latitude": 35.0639, "longitude": -85.3461, "hospital_type": "Psychiatric"},
            {"facility_name": "T.C. Thompson Children's Hospital", "latitude": 35.0461, "longitude": -85.3089, "hospital_type": "Children's"},
        ]

outfile = os.path.join("data", "raw", "healthcare", "hamilton_county_healthcare.csv")
with open(outfile, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["facility_name", "latitude", "longitude", "hospital_type"])
    writer.writeheader()
    writer.writerows(facilities)

print(f"  Saved {len(facilities)} facilities to {outfile}")
PYEOF
fi

# -------------------------------------------------------
# 4. Grocery Stores (OpenStreetMap Overpass)
# -------------------------------------------------------
echo ""
echo "--- Grocery Stores ---"
GROCERY_DIR="$DATA_DIR/osm_grocery"
mkdir -p "$GROCERY_DIR"

GROCERY_FILE="$GROCERY_DIR/hamilton_county_grocery_stores.csv"
if [ -f "$GROCERY_FILE" ]; then
    echo "  Already exists: $GROCERY_FILE"
else
    echo "  Querying Overpass API for Hamilton County grocery stores..."
    python3 << 'PYEOF'
import requests
import csv
import os

# Hamilton County bounding box
bbox = "34.9,-85.4,35.2,-85.1"

overpass_url = "http://overpass-api.de/api/interpreter"
overpass_query = f"""
[out:json][timeout:60];
(
  node["shop"="supermarket"]({bbox});
  way["shop"="supermarket"]({bbox});
  node["shop"="convenience"]({bbox});
  way["shop"="convenience"]({bbox});
  node["shop"="grocery"]({bbox});
  way["shop"="grocery"]({bbox});
);
out center;
"""

print("  Sending Overpass query...")
response = requests.get(overpass_url, params={"data": overpass_query}, timeout=120)
response.raise_for_status()

data = response.json()
stores = []
for element in data["elements"]:
    if element["type"] == "node":
        lat, lon = element["lat"], element["lon"]
    elif "center" in element:
        lat, lon = element["center"]["lat"], element["center"]["lon"]
    else:
        continue

    tags = element.get("tags", {})
    stores.append({
        "osm_id": element["id"],
        "name": tags.get("name", "Unnamed"),
        "type": tags.get("shop", "unknown"),
        "lat": lat,
        "lon": lon,
        "brand": tags.get("brand", ""),
        "operator": tags.get("operator", ""),
    })

outfile = os.path.join("data", "raw", "osm_grocery", "hamilton_county_grocery_stores.csv")
with open(outfile, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["osm_id", "name", "type", "lat", "lon", "brand", "operator"])
    writer.writeheader()
    writer.writerows(stores)

print(f"  Saved {len(stores)} grocery stores to {outfile}")
PYEOF
fi

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
echo ""
echo "=== Data Setup Complete ==="
echo ""
echo "Checking all required files:"
for f in \
    "$DATA_DIR/lehd/tn_wac_S000_JT00_2021.csv" \
    "$DATA_DIR/lehd/tl_2021_47_tabblock20.shp" \
    "$DATA_DIR/healthcare/hamilton_county_healthcare.csv" \
    "$DATA_DIR/osm_grocery/hamilton_county_grocery_stores.csv" \
    ; do
    if [ -f "$f" ]; then
        echo "  OK: $f"
    else
        echo "  MISSING: $f"
    fi
done
echo ""
echo "Next: granite --fips 47065000600 --verbose"