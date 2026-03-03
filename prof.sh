#!/bin/bash
# Remove AI-flavored artifacts from GRANITE codebase
# Run from repo root. Review diff before committing.

set -e

echo "=== Removing 'comprehensive' ==="
find . -name "*.py" -not -path "./.git/*" -not -path "./graveyard/*" \
  -exec sed -i 's/ [Cc]omprehensive//g' {} +
find . -name "*.md" -not -path "./.git/*" -not -path "./graveyard/*" \
  -exec sed -i 's/ [Cc]omprehensive//g' {} +

# second pass: COMPREHENSIVE in headers, and line-start "Comprehensive" in docstrings
find . -name "*.py" -not -path "./.git/*" -not -path "./graveyard/*" \
  -exec sed -i 's/COMPREHENSIVE //g; s/^    """Comprehensive /    """/g; s/ (comprehensive / (/g' {} +

echo "=== Removing check/x/warn icons ==="
find . \( -name "*.py" -o -name "*.sh" -o -name "*.md" \) \
  -not -path "./.git/*" -not -path "./graveyard/*" \
  -exec sed -i 's/[]//g; s/[]//g; s/\?//g; s///g' {} +

echo "=== Replacing arrows ==="
find . \( -name "*.py" -o -name "*.sh" -o -name "*.md" \) \
  -not -path "./.git/*" -not -path "./graveyard/*" \
  -exec sed -i 's/->/->/g' {} +
find . \( -name "*.py" -o -name "*.sh" -o -name "*.md" \) \
  -not -path "./.git/*" -not -path "./graveyard/*" \
  -exec sed -i 's/<-/<-/g' {} +

echo "=== Removing shouty comments ==="
find . -name "*.py" -not -path "./.git/*" -not -path "./graveyard/*" \
  -exec sed -i 's/ - NEW!//g; s/ NEW!//g' {} +

echo "=== Cleaning up extra spaces left by icon removal ==="
# only collapse double spaces that follow a non-space character
# this preserves all leading indentation
find . \( -name "*.py" -o -name "*.sh" -o -name "*.md" \) \
  -not -path "./.git/*" -not -path "./graveyard/*" \
  -exec sed -i 's/\([^ ]\) \([^ ]\)/\1 \2/g' {} +

echo ""
echo "Done. Review with: git diff --stat && git diff"
echo "Then: git add -A && git commit -m 'Clean up unicode symbols and AI prose artifacts'"