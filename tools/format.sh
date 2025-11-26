#!/bin/bash
# Auto-format all Python code in cyecca package
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Running black (code formatter)..."
poetry run black .

echo ""
echo "Running isort (import sorter)..."
poetry run isort .

echo ""
echo "✓ Formatting complete!"
