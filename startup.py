from deepface import DeepFace

arcface_model = None

def get_arcface_model():
    global arcface_model
    if arcface_model is None:
        print("⚡ Loading ArcFace model...")
        arcface_model = DeepFace.build_model("ArcFace")
        print("✅ ArcFace model loaded.")
    return arcface_model
