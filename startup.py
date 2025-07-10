from deepface.basemodels import ArcFace
import os

def preload_deepface_models():
    print("Preloading deepface models...")
    try:
        ArcFace.loadModel()
        print("✅ ArcFace model loaded successfully.")
    except Exception as e:
        print("❌ Error loading ArcFace model:", e)