from deepface import DeepFace

def preload_deepface_models():
    print("⚡ Preloading DeepFace model weights...")
    try:
        DeepFace.build_model("ArcFace")
        print("✅ ArcFace model loaded successfully.")
    except Exception as e:
        print("❌ Error loading DeepFace model:", e)
