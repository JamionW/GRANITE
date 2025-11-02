#!/bin/bash

echo "=== OSRM File Diagnostic ==="
echo ""

cd data/raw/osrm

echo "Checking for base .osrm files:"
ls -lh *.osrm 2>/dev/null || echo "No .osrm files found"

echo ""
echo "Checking for files without extension:"
ls -lh tennessee-latest 2>/dev/null || echo "No file named 'tennessee-latest' (without extension)"

echo ""
echo "All tennessee-latest files (first 10):"
ls -lh tennessee-latest* 2>/dev/null | head -10

echo ""
echo "All tennessee-foot files (first 10):"  
ls -lh tennessee-foot* 2>/dev/null | head -10

echo ""
echo "Checking file counts:"
echo "  tennessee-latest.* files: $(ls tennessee-latest.* 2>/dev/null | wc -l)"
echo "  tennessee-foot.* files: $(ls tennessee-foot.* 2>/dev/null | wc -l)"

echo ""
echo "Files in processing directory:"
ls -la processing/ 2>/dev/null || echo "No processing directory"