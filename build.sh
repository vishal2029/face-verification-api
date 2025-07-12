#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
# This is a good practice for build scripts.
set -e

# 1) Install dependencies from the new, clean requirements file
echo "--- Installing Python dependencies ---"
python3.10 -m pip install -r requirements.txt

# 2) Preload SFace weights so they’re cached in the container image at build time.
# This is crucial for avoiding runtime downloads and memory spikes on startup.
echo "--- Caching SFace model weights ---"
python3.10 - << 'PYCODE'
from deepface import DeepFace
print("⚡ Downloading SFace weights at build time…")
# This must match the model used in verification.py
DeepFace.build_model("SFace")
print("✅ SFace weights cached.")
PYCODE

echo "--- Build script finished successfully! ---"
