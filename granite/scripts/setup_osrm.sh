#!/bin/bash
# One-time OSRM setup

OSRM_DIR="/workspaces/GRANITE/data/raw/osrm"
OSM_FILE="tennessee-latest.osm.pbf"

echo "=== OSRM One-Time Setup ==="

mkdir -p $OSRM_DIR
cd $OSRM_DIR

# Download OSM data if not present
if [ ! -f "$OSM_FILE" ]; then
    echo "Downloading Tennessee OSM data (~100MB)..."
    wget https://download.geofabrik.de/north-america/us/tennessee-latest.osm.pbf
fi

# Process for driving
echo "Processing for driving routes..."
docker run -t -v $(pwd):/data osrm/osrm-backend \
    osrm-extract -p /opt/car.lua /data/$OSM_FILE

docker run -t -v $(pwd):/data osrm/osrm-backend \
    osrm-partition /data/tennessee-latest.osrm

docker run -t -v $(pwd):/data osrm/osrm-backend \
    osrm-customize /data/tennessee-latest.osrm

echo " OSRM setup complete!"
echo ""
echo "To start servers, run: bash scripts/start_osrm.sh"