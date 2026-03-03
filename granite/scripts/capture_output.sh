#!/bin/bash
# Capture output from already-running GRANITE process

echo "=========================================="
echo "GRANITE Output Capture"
echo "=========================================="
echo ""

# Find GRANITE process
GRANITE_PID=$(ps aux | grep -v grep | grep "run_granite\|run_holdout" | awk '{print $2}' | head -1)

if [ -z "$GRANITE_PID" ]; then
    echo "ERROR: No GRANITE process found"
    exit 1
fi

echo "Found GRANITE process: PID $GRANITE_PID"
echo ""

# Create log file with timestamp
LOGFILE="/workspaces/GRANITE/output/granite_captured_$(date +%Y%m%d_%H%M%S).log"
echo "Capturing output to: $LOGFILE"
echo ""
echo "Note: This captures NEW output only (not what was already printed)"
echo "Press Ctrl+C to stop capturing"
echo "=========================================="
echo ""

# Follow stdout and stderr from the process
# This captures everything going forward
tail -f /proc/$GRANITE_PID/fd/1 /proc/$GRANITE_PID/fd/2 2>/dev/null | tee -a "$LOGFILE"

# When process ends or user stops
echo ""
echo "=========================================="
echo "Output saved to: $LOGFILE"
echo "=========================================="