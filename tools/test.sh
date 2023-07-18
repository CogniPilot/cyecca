#!/bin/bash
set -x
set -e
poetry run black --check .
poetry run pytest --cov
poetry run pytest --nbmake notebook
