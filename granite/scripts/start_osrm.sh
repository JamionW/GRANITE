#!/bin/bash
set -e

echo "=== OSRM Auto-Startup Script ==="

# OSRM data location
OSRM_DATA_DIR="/workspaces/GRANITE/data/raw/osrm"

# Verify BOTH data files exist and are complete
check_osrm_file() {
    local base_file=$1
    local profile_name=$2
    
    if [ ! -f "$base_file" ]; then
        echo " Missing: $(basename $base_file)"
        return 1
    fi
    
    # Check for required auxiliary files
    local required_exts=".ramIndex .fileIndex .edges .geometry"
    local missing=0
    
    for ext in $required_exts; do
        if [ ! -f "${base_file}${ext}" ]; then
            ((missing++))
        fi
    done
    
    if [ $missing -gt 0 ]; then
        echo " $profile_name profile incomplete: $(basename $base_file) missing $missing auxiliary files"
        echo "  Run: bash granite/scripts/process_foot_profile.sh"
        return 1
    fi
    
    echo " $profile_name profile complete: $(basename $base_file)"
    return 0
}

echo ""
echo "Checking OSRM data files..."
check_osrm_file "$OSRM_DATA_DIR/tennessee-latest.osrm" "Driving" || exit 1
check_osrm_file "$OSRM_DATA_DIR/tennessee-foot.osrm" "Walking" || exit 1

echo ""

# Stop and remove existing containers
echo "Cleaning up old containers..."
docker stop osrm-driving osrm-walking 2>/dev/null || true
docker rm osrm-driving osrm-walking 2>/dev/null || true

# Check port availability
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo " Port $port is in use"
        return 1
    fi
    echo " Port $port available"
    return 0
}

echo "Checking port availability..."
check_port 5000 || exit 1
check_port 5001 || exit 1

# Wait for OSRM server to be ready
wait_for_osrm() {
    local name=$1
    local port=$2
    local mode=$3
    local max_wait=60
    local waited=0
    
    echo "  Waiting for $name server to be ready..."
    
    while [ $waited -lt $max_wait ]; do
        # Check if container is still running
        if ! docker ps --format '{{.Names}}' | grep -q "^$name$"; then
            echo "   Container $name stopped unexpectedly"
            echo "  Container logs:"
            docker logs $name 2>&1 | tail -20
            return 1
        fi
        
        # Try to connect to the server
        if curl -s "http://localhost:$port/route/v1/$mode/-85.3,35.0;-85.2,35.0?overview=false" >/dev/null 2>&1; then
            echo "   $name server ready (${waited}s)"
            return 0
        fi
        
        echo -n "."
        sleep 3
        waited=$((waited + 3))
    done
    
    echo ""
    echo "   $name server failed to become ready after ${max_wait}s"
    echo "  Container logs:"
    docker logs $name 2>&1 | tail -20
    return 1
}

# Start driving server with driving profile (tennessee-latest.osrm)
echo "Starting OSRM driving server on port 5000..."
DRIVING_ID=$(docker run -d \
    --name osrm-driving \
    -p 5000:5000 \
    -v "$OSRM_DATA_DIR:/data:ro" \
    osrm/osrm-backend \
    osrm-routed --algorithm mld /data/tennessee-latest.osrm)

echo "  Container ID: ${DRIVING_ID:0:12}"

if ! wait_for_osrm "osrm-driving" 5000 "driving"; then
    echo " Failed to start driving server"
    exit 1
fi

# Start walking server with foot profile (tennessee-foot.osrm)
echo "Starting OSRM walking server on port 5001..."
WALKING_ID=$(docker run -d \
    --name osrm-walking \
    -p 5001:5000 \
    -v "$OSRM_DATA_DIR:/data:ro" \
    osrm/osrm-backend \
    osrm-routed --algorithm mld /data/tennessee-foot.osrm)

echo "  Container ID: ${WALKING_ID:0:12}"

if ! wait_for_osrm "osrm-walking" 5001 "foot"; then
    echo " Failed to start walking server"
    exit 1
fi

echo ""
echo " Both OSRM servers started successfully!"
echo ""
echo "Servers:"
echo "  - Driving: http://localhost:5000 (using tennessee-latest.osrm)"
echo "  - Walking: http://localhost:5001 (using tennessee-foot.osrm)"
echo ""
echo "Test with:"
echo "  curl 'http://localhost:5000/route/v1/driving/-85.3,35.0;-85.2,35.0?overview=false'"
echo "  curl 'http://localhost:5001/route/v1/foot/-85.3,35.0;-85.2,35.0?overview=false'"