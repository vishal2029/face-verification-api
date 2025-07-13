import cv2
import numpy as np
import os
import requests
import onnxruntime
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import gdown
import traceback

# --- CONFIGURATION ---
CONFIG = {
    "NUM_FRAMES": 10,
    "SFACE_MODEL_PATH": "face_recognition_sface_2021dec.onnx",
    "FACE_DETECTOR_PATH": "haarcascade_frontalface_default.xml",
    "SIMILARITY_THRESHOLD": 0.96,
    "FLIP_FALLBACK_ENABLED": True,
    "FLIP_CHECK_THRESHOLD": 0.60
}

# --- MODEL AND DETECTOR LOADING ---
onnx_session = None
face_cascade = None

def download_and_load_models():
    """
    Downloads the SFace model and the face detector if they don't exist,
    then loads them into memory.
    """
    global onnx_session, face_cascade
    if onnx_session and face_cascade:
        return # Models already loaded

    sface_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    if not os.path.exists(CONFIG["SFACE_MODEL_PATH"]):
        print(f"Downloading SFace model...")
        gdown.download(sface_url, CONFIG["SFACE_MODEL_PATH"], quiet=False)
        print("SFace model downloaded.")

    detector_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    if not os.path.exists(CONFIG["FACE_DETECTOR_PATH"]):
        print(f"Downloading face detector...")
        gdown.download(detector_url, CONFIG["FACE_DETECTOR_PATH"], quiet=False)
        print("Face detector downloaded.")

    print("Loading models into memory...")
    onnx_session = onnxruntime.InferenceSession(CONFIG["SFACE_MODEL_PATH"])
    face_cascade = cv2.CascadeClassifier(CONFIG["FACE_DETECTOR_PATH"])
    print("Models loaded.")


def get_face_embedding(image: np.ndarray) -> np.ndarray | None:
    """Detects a face in an image, preprocesses it, and returns the SFace embedding."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            return None
        x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        face_roi = image[y:y+h, x:x+w]
        aligned_face = cv2.resize(face_roi, (112, 112))
        aligned_face = aligned_face.astype(np.float32) / 255.0
        aligned_face = (aligned_face - 0.5) / 0.5
        input_blob = np.expand_dims(aligned_face.transpose(2, 0, 1), axis=0)
        input_name = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name
        embedding = onnx_session.run([output_name], {input_name: input_blob})[0]
        return embedding
    except Exception:
        print("An unexpected error occurred in get_face_embedding:")
        traceback.print_exc()
        return None

def download_image_from_url(url: str):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        arr = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def download_video_from_url(url: str):
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None

def extract_frames_from_video(video_bytes: bytes) -> list:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        temp_video.write(video_bytes)
        temp_video_path = temp_video.name
    
    frames = []
    cap = None
    try:
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened(): return []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0: return []
        interval = max(total_frames // CONFIG["NUM_FRAMES"], 1)
        for i in range(CONFIG["NUM_FRAMES"]):
            frame_pos = i * interval
            if frame_pos >= total_frames: break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if ret: frames.append(frame)
    finally:
        if cap: cap.release()
        os.remove(temp_video_path)
    return frames

def run_face_verification(video_frames: list, profile_images: list) -> dict:
    """
    Iterates through frames and profile images, using pre-computed original embeddings
    and optimized flip logic to improve performance and prevent timeouts.
    """
    print("Pre-computing embeddings for original profile images...")
    
    # --- STEP 1: Pre-compute embeddings for all ORIGINAL profile images ---
    # This is much more efficient. We store the original image to flip it later if needed.
    original_profile_data = []
    for p_image in profile_images:
        embedding = get_face_embedding(p_image)
        if embedding is not None:
            original_profile_data.append({"image": p_image, "embedding": embedding})

    if not original_profile_data:
        return {"status": "Failed", "message": "Could not find a face in any profile image."}
    
    print(f"Finished pre-computing {len(original_profile_data)} original embeddings.")

    # --- STEP 2: Loop through video frames and apply verification logic ---
    for frame in video_frames:
        frame_embedding = get_face_embedding(frame)
        if frame_embedding is None:
            continue

        for profile_data in original_profile_data:
            p_emb = profile_data["embedding"]
            p_img = profile_data["image"]
            
            try:
                # --- Check original image ---
                original_similarity = cosine_similarity(frame_embedding, p_emb)[0][0]

                if original_similarity > CONFIG["SIMILARITY_THRESHOLD"]:
                    return {"status": "Successful", "message": f"Match found with similarity {original_similarity:.2f}"}

                # --- Decide if a flip check is worthwhile ---
                if original_similarity < CONFIG["FLIP_CHECK_THRESHOLD"]:
                    continue # Similarity is too low, don't bother flipping

                # --- If in the "maybe" zone, check the flipped image ---
                if CONFIG["FLIP_FALLBACK_ENABLED"]:
                    flipped_image = cv2.flip(p_img, 1)
                    flipped_embedding = get_face_embedding(flipped_image)
                    if flipped_embedding is None:
                        continue
                    
                    flipped_similarity = cosine_similarity(frame_embedding, flipped_embedding)[0][0]
                    if flipped_similarity > CONFIG["SIMILARITY_THRESHOLD"]:
                        return {"status": "Successful", "message": f"Match found with flipped image similarity {flipped_similarity:.2f}"}

            except Exception:
                # Safety net for any unexpected errors
                print("An unexpected error occurred during similarity comparison:")
                traceback.print_exc()
                continue

    return {"status": "Failed", "message": "No match found."}


def process_verification_from_urls(video_url: str, image_urls: list) -> dict:
    profile_images = [download_image_from_url(url) for url in image_urls]
    profile_images = [img for img in profile_images if img is not None]
    if not profile_images:
        return {"status": "Failed", "message": "Could not download or process any valid profile images."}

    video_bytes = download_video_from_url(video_url)
    if not video_bytes:
        return {"status": "Failed", "message": "Could not download video."}

    frames = extract_frames_from_video(video_bytes)
    if not frames:
        return {"status": "Failed", "message": "Could not extract frames from video."}

    return run_face_verification(frames, profile_images)
