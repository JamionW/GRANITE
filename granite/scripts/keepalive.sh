#!/bin/bash
# GRANITE Keepalive Script
# Monitors for running GRANITE processes and keeps terminal alive

echo "=========================================="
echo "GRANITE Keepalive Monitor Started"
echo "=========================================="
echo "Watching for GRANITE processes..."
echo ""

while true; do
    # Check if any GRANITE-related process is running
    if ps aux | grep -v grep | grep -q "run_granite\|granite\|GRANITE|run_holdout\|test\|validation"; then
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$timestamp] GRANITE running - keeping session alive..."
        sleep 60 # Wait 1 minute
    else
        echo ""
        echo "=========================================="
        echo "No GRANITE processes detected - exiting"
        echo "=========================================="
        break
    fi
done