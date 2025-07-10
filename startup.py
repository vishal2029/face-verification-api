from deepface.basemodels import ArcFace
import os

def preload_deepface_models():
    print("Preloading deepface models...")
    try:
        model = ArcFace.loadModel()
        print("ArcFace model loaded successfully.")
    except Exception as e:
        print("❌ Failed to preload ArcFace model:", e)