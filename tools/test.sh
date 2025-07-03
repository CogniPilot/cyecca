#!/bin/bash
set -x
set -e

# These lines below are here to make sure the script is run from the project root
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # Get the absolute path to the script
PROJECT_ROOT="$(dirname "$SCRIPT_PATH")" # Get the project root (parent directory of tools)
cd "$PROJECT_ROOT" # Change to the project root directory
echo "Running tests from project root: $PROJECT_ROOT" # Print the project root

poetry run black --check .
poetry run pytest --cov 
poetry run pytest --nbmake notebook
