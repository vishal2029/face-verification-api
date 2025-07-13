#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

# 1) Install dependencies from the new, clean requirements file
echo "--- Installing Python dependencies ---"
python3.10 -m pip install -r requirements.txt

# 2) Pre-download the SFace model and face detector at build time.
# This ensures they are baked into the container image.
echo "--- Caching models at build time ---"
python3.10 - << 'PYCODE'
from verification import download_and_load_models
print("⚡ Downloading models...")
download_and_load_models()
print("✅ Models cached successfully.")
PYCODE

echo "--- Build script finished successfully! ---"
