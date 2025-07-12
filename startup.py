from deepface import DeepFace

def preload_deepface_models():
    """
    Preloads the specified DeepFace model to avoid delays on the first request.
    This is updated to preload the lightweight 'SFace' model.
    """
    print("⚡ Preloading DeepFace model weights...")
    try:
        # Changed to SFace to match the new, memory-efficient configuration
        DeepFace.build_model("SFace")
        print("✅ SFace model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading DeepFace model: {e}")
