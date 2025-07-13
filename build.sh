#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

# 1) Install dependencies from the new, clean requirements file
echo "--- Installing Python dependencies ---"
# Use the generic python3 command provided by the build environment.
# This is safer than hardcoding a specific version like python3.10.
python3 -m pip install -r requirements.txt

# 2) Pre-download the SFace model and face detector at build time.
echo "--- Caching models at build time ---"
# Use the generic python3 command here as well.
python3 - << 'PYCODE'
from verification import download_and_load_models
print("⚡ Downloading models...")
download_and_load_models()
print("✅ Models cached successfully.")
PYCODE

echo "--- Build script finished successfully! ---"
