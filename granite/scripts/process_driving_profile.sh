#!/bin/bash
set -e

echo "=== OSRM Driving Profile Processing ==="
echo ""
echo "This script will create tennessee-car.osrm (driving profile)"
echo "with all required auxiliary files for car routes."
echo ""

OSRM_DATA_DIR="/workspaces/GRANITE/data/raw/osrm"
OSM_FILE="$OSRM_DATA_DIR/tennessee-latest.osm.pbf"

# Check if we have the source OSM file
if [ ! -f "$OSM_FILE" ]; then
    echo " Source OSM file not found: $OSM_FILE"
    exit 1
fi

echo " Found OSM source file: $(basename $OSM_FILE)"
echo ""

# Use the data directory directly (not a temp directory)
cd "$OSRM_DATA_DIR"

echo "Step 1: Extracting OSM data with car profile..."
echo "  This may take 5-15 minutes depending on data size..."
docker run -t -v "$OSRM_DATA_DIR:/data" \
    osrm/osrm-backend \
    osrm-extract -p /opt/car.lua /data/$(basename $OSM_FILE)

if [ ! -f "$OSRM_DATA_DIR/tennessee-latest.osrm" ]; then
    echo " Extract failed - tennessee-latest.osrm not created"
    echo ""
    echo "Files created:"
    ls -lh tennessee-latest.osrm* 2>/dev/null | head -10
    exit 1
fi

echo " Extract complete"
echo ""

echo "Step 2: Partitioning graph..."
docker run -t -v "$OSRM_DATA_DIR:/data" \
    osrm/osrm-backend \
    osrm-partition /data/tennessee-latest.osrm

echo " Partition complete"
echo ""

echo "Step 3: Customizing graph..."
docker run -t -v "$OSRM_DATA_DIR:/data" \
    osrm/osrm-backend \
    osrm-customize /data/tennessee-latest.osrm

echo " Customize complete"
echo ""

echo " Renaming to tennessee-car.osrm..."
cd "$OSRM_DATA_DIR"
for file in tennessee-latest.osrm*; do
    if [ -f "$file" ]; then
        new_name=$(echo "$file" | sed 's/tennessee-latest/tennessee-car/')
        mv "$file" "$new_name"
    fi
done

echo " All processing complete!"
echo ""
echo "Created files:"
ls -lh "$OSRM_DATA_DIR"/tennessee-car.osrm* | head -10
echo "..."
echo ""
echo "You can now start OSRM with:"
echo "  bash granite/scripts/start_osrm.sh"