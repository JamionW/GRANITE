#!/usr/bin/env bash
# download required shapefiles and check prerequisites for GRANITE
set -euo pipefail

DATA_DIR="${1:-./data}"
RAW_DIR="$DATA_DIR/raw"
PROCESSED_DIR="$DATA_DIR/processed"

mkdir -p "$RAW_DIR" "$PROCESSED_DIR"

# block group shapefile (TIGER/Line 2020, Tennessee)
if [ ! -f "$RAW_DIR/tl_2020_47_bg.shp" ]; then
    echo "Downloading Tennessee block group shapefile..."
    curl -L -o /tmp/tl_2020_47_bg.zip \
        "https://www2.census.gov/geo/tiger/TIGER2020/BG/tl_2020_47_bg.zip"
    unzip -o /tmp/tl_2020_47_bg.zip -d "$RAW_DIR"
    rm /tmp/tl_2020_47_bg.zip
    echo "Block group shapefile downloaded."
else
    echo "Block group shapefile already exists."
fi

# tract shapefile (TIGER/Line 2020, Tennessee)
if [ ! -f "$RAW_DIR/tl_2020_47_tract.shp" ]; then
    echo "Downloading Tennessee tract shapefile..."
    curl -L -o /tmp/tl_2020_47_tract.zip \
        "https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_47_tract.zip"
    unzip -o /tmp/tl_2020_47_tract.zip -d "$RAW_DIR"
    rm /tmp/tl_2020_47_tract.zip
    echo "Tract shapefile downloaded."
else
    echo "Tract shapefile already exists."
fi

# check for cached block group SVI (requires Census API key to generate)
if [ ! -f "$PROCESSED_DIR/acs_block_groups_svi.csv" ]; then
    if [ -z "${CENSUS_API_KEY:-}" ]; then
        echo ""
        echo "ERROR: $PROCESSED_DIR/acs_block_groups_svi.csv does not exist."
        echo "To generate it, set the CENSUS_API_KEY environment variable:"
        echo "  export CENSUS_API_KEY=your_key_here"
        echo "Then run the pipeline, which will fetch ACS data and cache the file."
        echo "Get a free key at: https://api.census.gov/data/key_signup.html"
        exit 1
    else
        echo "ACS cache not found but CENSUS_API_KEY is set. Pipeline will fetch on first run."
    fi
else
    echo "ACS block group SVI cache already exists."
fi

echo "Data setup complete."
