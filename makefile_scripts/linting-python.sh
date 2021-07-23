#!/bin/bash

# linting python files:
#   - isort: import order
#   - black: linting

set -e

echo "Running 'isort'..."
isort .
echo "Running 'black'..."
black .
echo "All checks successfully passed!"
