#!/bin/bash

# linting python files:
#   - isort: import order
#   - black: linting

set -e

echo "Running 'isort'..."
isort .
echo "Running 'black'..."
black .
echo "Running 'jupyter-black'..."
jblack .
echo "All checks successfully passed!"
