#!/bin/bash
# Improved OSRM startup script with debugging

OSRM_DIR="/workspaces/GRANITE/data/raw/osrm"
OSM_FILE="tennessee-latest.osm.pbf"

echo "=== OSRM Auto-Startup Script (Enhanced) ==="

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -i :$port > /dev/null 2>&1 || netstat -tuln 2>/dev/null | grep -q ":$port "; then
        return 1  # Port in use
    else
        return 0  # Port available
    fi
}

# Function to wait for OSRM server to be ready
wait_for_osrm() {
    local port=$1
    local profile=$2
    local max_attempts=20
    local attempt=0
    
    echo "  Waiting for server to be ready..."
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:$port/route/v1/$profile/-85.3097,35.0456;-85.2111,35.0407" > /dev/null 2>&1; then
            return 0  # Success
        fi
        sleep 1
        attempt=$((attempt + 1))
        echo -n "."
    done
    
    echo ""
    return 1  # Failed
}

# Check if OSRM data is processed
if [ ! -f "$OSRM_DIR/tennessee-latest.osrm" ]; then
    echo "✗ OSRM data not processed. Run setup first:"
    echo "  bash scripts/setup_osrm.sh"
    exit 1
fi

echo "✓ OSRM data files found"

# Clean up any existing containers
echo "Cleaning up old containers..."
docker rm -f osrm-driving 2>/dev/null || true
docker rm -f osrm-walking 2>/dev/null || true

# Check port availability
echo "Checking port availability..."
if ! check_port 5000; then
    echo "✗ Port 5000 already in use"
    lsof -i :5000 || netstat -tulpn | grep 5000
    exit 1
fi
echo "✓ Port 5000 available"

if ! check_port 5001; then
    echo "✗ Port 5001 already in use"
    lsof -i :5001 || netstat -tulpn | grep 5001
    exit 1
fi
echo "✓ Port 5001 available"

# Start driving server
echo ""
echo "Starting OSRM driving server on port 5000..."
CONTAINER_ID=$(docker run -d --name osrm-driving \
    -p 5000:5000 \
    -v $OSRM_DIR:/data \
    osrm/osrm-backend osrm-routed --algorithm mld /data/tennessee-latest.osrm 2>&1)

if [ $? -ne 0 ]; then
    echo "✗ Failed to start container"
    exit 1
fi

echo "  Container ID: ${CONTAINER_ID:0:12}"

# Wait for server to be ready
if wait_for_osrm 5000 driving; then
    echo "✓ Driving server ready on port 5000"
else
    echo "✗ Driving server failed to start"
    echo ""
    echo "Container logs:"
    docker logs osrm-driving
    exit 1
fi

# Process walking data if needed
echo ""
echo "Checking walking route data..."
if [ ! -f "$OSRM_DIR/tennessee-foot.osrm" ]; then
    echo "Processing OSM data for walking routes..."
    
    # Extract with foot profile
    docker run -t -v $OSRM_DIR:/data osrm/osrm-backend \
        osrm-extract -p /opt/foot.lua /data/$OSM_FILE
    
    if [ ! -f "$OSRM_DIR/tennessee-latest.osrm" ]; then
        echo "✗ Extract failed"
        exit 1
    fi
    
    # Partition
    docker run -t -v $OSRM_DIR:/data osrm/osrm-backend \
        osrm-partition /data/tennessee-latest.osrm
    
    # Customize
    docker run -t -v $OSRM_DIR:/data osrm/osrm-backend \
        osrm-customize /data/tennessee-latest.osrm
    
    # Rename for clarity
    mv $OSRM_DIR/tennessee-latest.osrm $OSRM_DIR/tennessee-foot.osrm
    mv $OSRM_DIR/tennessee-latest.osrm.* $OSRM_DIR/tennessee-foot.osrm.* 2>/dev/null || true
    
    echo "✓ Walking data processed"
fi

# Start walking server
echo ""
echo "Starting OSRM walking server on port 5001..."
CONTAINER_ID=$(docker run -d --name osrm-walking \
    -p 5001:5000 \
    -v $OSRM_DIR:/data \
    osrm/osrm-backend osrm-routed --algorithm mld /data/tennessee-foot.osrm 2>&1)

if [ $? -ne 0 ]; then
    echo "✗ Failed to start walking container"
    exit 1
fi

echo "  Container ID: ${CONTAINER_ID:0:12}"

# Wait for server to be ready
if wait_for_osrm 5001 foot; then
    echo "✓ Walking server ready on port 5001"
else
    echo "✗ Walking server failed to start"
    echo ""
    echo "Container logs:"
    docker logs osrm-walking
    exit 1
fi

# Final status
echo ""
echo "=== OSRM Servers Ready ==="
echo "Driving: http://localhost:5000"
echo "Walking: http://localhost:5001"
echo ""
echo "To stop servers:"
echo "  docker stop osrm-driving osrm-walking"
echo ""
echo "To view logs:"
echo "  docker logs osrm-driving"
echo "  docker logs osrm-walking"