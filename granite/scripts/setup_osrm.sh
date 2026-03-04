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

# Rename to tennessee-car.osrm so foot profile can't overwrite
for file in tennessee-latest.osrm*; do
    if [ -f "$file" ]; then
        new_name=$(echo "$file" | sed 's/tennessee-latest/tennessee-car/')
        mv "$file" "$new_name"
    fi
done

echo " OSRM driving setup complete!"
echo ""
echo "Next: bash granite/scripts/process_foot_profile.sh"
echo "Then: bash granite/scripts/start_osrm.sh"