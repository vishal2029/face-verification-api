#!/usr/bin/env bash

# 1) Install dependencies
python3.10 -m pip install --upgrade pip setuptools wheel
python3.10 -m pip install -r requirements.txt

# 2) Preload ArcFace weights so they’re cached before runtime
python3.10 - << 'PYCODE'
from deepface import DeepFace
print("⚡ Downloading ArcFace weights at build time…")
DeepFace.build_model("ArcFace")
print("✅ ArcFace weights cached.")
PYCODE
