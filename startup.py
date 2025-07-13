# We now import the new function from our rewritten verification module
from verification import download_and_load_models

def preload_models():
    """
    This function is called by FastAPI on startup.
    It ensures our SFace model and face detector are downloaded and
    loaded into memory before the server starts accepting requests.
    """
    print("⚡ Preloading SFace model and face detector...")
    try:
        download_and_load_models()
        print("✅ Models loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")

