#!/bin/bash
set -e

echo "=== OSRM Foot Profile Processing ==="
echo ""
echo "This script will create a proper tennessee-foot.osrm file"
echo "with all required auxiliary files for walking routes."
echo ""

OSRM_DATA_DIR="/workspaces/GRANITE/data/raw/osrm"
OSM_FILE="$OSRM_DATA_DIR/tennessee-latest.osm.pbf"

# Check if we have the source OSM file
if [ ! -f "$OSM_FILE" ]; then
    echo "✗ Source OSM file not found: $OSM_FILE"
    echo ""
    echo "You need the .osm.pbf file. Check if it's in:"
    ls -lh "$OSRM_DATA_DIR"/*.pbf 2>/dev/null || echo "  No .pbf files found"
    exit 1
fi

echo "✓ Found OSM source file: $(basename $OSM_FILE)"
echo ""

# Create temporary working directory
WORK_DIR="$OSRM_DATA_DIR/processing"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "Step 1: Extracting OSM data with foot profile..."
echo "  This may take 5-15 minutes depending on data size..."

docker run -t -v "$OSRM_DATA_DIR:/data" \
    osrm/osrm-backend \
    osrm-extract -p /opt/foot.lua /data/$(basename $OSM_FILE)

if [ ! -f "$OSRM_DATA_DIR/$(basename $OSM_FILE .osm.pbf).osrm" ]; then
    echo "✗ Extract failed"
    exit 1
fi

echo "✓ Extract complete"
echo ""

echo "Step 2: Partitioning graph..."
docker run -t -v "$OSRM_DATA_DIR:/data" \
    osrm/osrm-backend \
    osrm-partition /data/$(basename $OSM_FILE .osm.pbf).osrm

echo "✓ Partition complete"
echo ""

echo "Step 3: Customizing graph..."
docker run -t -v "$OSRM_DATA_DIR:/data" \
    osrm/osrm-backend \
    osrm-customize /data/$(basename $OSM_FILE .osm.pbf).osrm

echo "✓ Customize complete"
echo ""

# Rename to tennessee-foot.osrm
BASE_NAME=$(basename $OSM_FILE .osm.pbf)
if [ "$BASE_NAME" != "tennessee-foot" ]; then
    echo "Renaming processed files to tennessee-foot..."
    cd "$OSRM_DATA_DIR"
    for file in ${BASE_NAME}.osrm*; do
        new_name=$(echo "$file" | sed "s/${BASE_NAME}/tennessee-foot/")
        mv "$file" "$new_name"
        echo "  Renamed: $file → $new_name"
    done
fi

echo ""
echo "✓ All processing complete!"
echo ""
echo "Created files:"
ls -lh "$OSRM_DATA_DIR"/tennessee-foot.osrm*
echo ""
echo "You can now start OSRM with:"
echo "  bash granite/scripts/start_osrm.sh"