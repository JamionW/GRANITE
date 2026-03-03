#!/bin/bash
# GRANITE professionalization cleanup
# Review before running. Execute from repo root: bash cleanup.sh

set -e

echo "=== GRANITE Cleanup ==="

# 1. Remove dead debug/diagnostic files
echo ""
echo "Removing dead files..."
git rm -f test.py
git rm -f granite_inline_diagnostics.py
git rm -f granite/scripts/test_vars.py  # contains exposed Census API key

# 2. Remove duplicate NLCD tiff (identical md5 checksums)
echo ""
echo "Removing duplicate NLCD tiff..."
git rm -f data/NLCD_Hamilton_2016.tiff

# 3. Remove cached data that should not be in version control
echo ""
echo "Removing granite_cache from tracking..."
git rm -r --cached granite_cache/ 2>/dev/null || echo "  (granite_cache not tracked, skipping)"

# 4. Remove tract_inventory.txt (utility output, not source)
echo ""
echo "Removing tract_inventory.txt..."
git rm -f tract_inventory.txt

# 5. Verify new files are in place
echo ""
echo "Checking new files..."
for f in README.md docs/FEATURES.md; do
    [ -f "$f" ] && echo "  OK: $f" || echo "  MISSING: $f"
done

# 6. Check __init__.py files
echo ""
echo "Checking __init__.py files..."
for dir in granite/data granite/disaggregation granite/evaluation granite/features granite/models granite/routing granite/scripts granite/validation granite/visualization; do
    [ -f "$dir/__init__.py" ] && echo "  OK: $dir/__init__.py" || echo "  MISSING: $dir/__init__.py"
done

echo ""
echo "=== Cleanup complete ==="
echo ""
echo "Next steps:"
echo "  git add -A"
echo "  git status"
echo "  git commit -m 'Professionalize repo structure: add README, docs, fix packaging, remove dead code'"
echo "  git push origin main"