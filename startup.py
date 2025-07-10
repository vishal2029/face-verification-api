from deepface import DeepFace

def preload_deepface_models():
    print("⚡ Preloading DeepFace model weights...")
    try:
        # This will force downloading the default model (ArcFace) and cache it
        DeepFace.build_model("ArcFace")
        print("✅ ArcFace model loaded successfully.")
    except Exception as e:
        print("❌ Error loading DeepFace model:", e)