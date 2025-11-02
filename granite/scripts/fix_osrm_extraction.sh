#!/bin/bash

# Fix: Re-run just the extraction step to create the missing tennessee-latest.osrm file

set -e

WORK_DIR="$(pwd)"
DATA_DIR="data/raw/osrm"
PBF_FILE="tennessee-latest.osm.pbf"
OSRM_IMAGE="ghcr.io/project-osrm/osrm-backend"

echo "=== Re-running OSRM Extraction ==="
echo "This should create the missing tennessee-latest.osrm file"
echo ""

if [ ! -f "${DATA_DIR}/${PBF_FILE}" ]; then
    echo "Error: ${PBF_FILE} not found in ${DATA_DIR}"
    exit 1
fi

echo "Extracting with car profile..."
docker run -t -v "${WORK_DIR}/${DATA_DIR}:/data" ${OSRM_IMAGE} osrm-extract -p /opt/car.lua /data/${PBF_FILE}

echo ""
echo "Checking for base file..."
if [ -f "${DATA_DIR}/tennessee-latest.osrm" ]; then
    ls -lh ${DATA_DIR}/tennessee-latest.osrm
    echo "✓ Success: tennessee-latest.osrm created"
else
    echo "✗ Error: tennessee-latest.osrm still missing"
    echo ""
    echo "Files created:"
    ls -lh ${DATA_DIR}/tennessee-latest.osrm* 2>/dev/null | head -5
    exit 1
fi