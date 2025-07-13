import cv2
import numpy as np
import os
import requests
import onnxruntime
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import gdown

# --- CONFIGURATION ---
CONFIG = {
    "NUM_FRAMES": 10,
    # The SFace model is a lightweight, high-performance face recognition model.
    "SFACE_MODEL_PATH": "face_recognition_sface_2021dec.onnx",
    # We will use a standard OpenCV face detector.
    "FACE_DETECTOR_PATH": "haarcascade_frontalface_default.xml",
    # Threshold for deciding if two faces are a match. Higher is more similar.
    "SIMILARITY_THRESHOLD": 0.85, # This is for cosine similarity, adjust as needed.
    "FLIP_FALLBACK_ENABLED": True
}

# --- MODEL AND DETECTOR LOADING ---
# Global variables to hold the loaded models so we don't reload them on every request.
onnx_session = None
face_cascade = None

def download_and_load_models():
    """
    Downloads the SFace model and the face detector if they don't exist,
    then loads them into memory.
    """
    global onnx_session, face_cascade

    # --- Download SFace Model ---
    sface_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    if not os.path.exists(CONFIG["SFACE_MODEL_PATH"]):
        print(f"Downloading SFace model from {sface_url}...")
        gdown.download(sface_url, CONFIG["SFACE_MODEL_PATH"], quiet=False)
        print("SFace model downloaded.")

    # --- Download Face Detector ---
    detector_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    if not os.path.exists(CONFIG["FACE_DETECTOR_PATH"]):
        print(f"Downloading face detector from {detector_url}...")
        gdown.download(detector_url, CONFIG["FACE_DETECTOR_PATH"], quiet=False)
        print("Face detector downloaded.")

    # --- Load Models ---
    if onnx_session is None:
        print("Loading ONNX SFace model into memory...")
        onnx_session = onnxruntime.InferenceSession(CONFIG["SFACE_MODEL_PATH"])
        print("SFace model loaded.")

    if face_cascade is None:
        print("Loading face detector into memory...")
        face_cascade = cv2.CascadeClassifier(CONFIG["FACE_DETECTOR_PATH"])
        print("Face detector loaded.")


def get_face_embedding(image: np.ndarray) -> np.ndarray | None:
    """
    Detects a face in an image, preprocesses it, and returns the SFace embedding.
    """
    # Convert to grayscale for the detector
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return None # No face found

    # Use the largest detected face
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    face_roi = image[y:y+h, x:x+w]

    # Preprocess the face for SFace model (resize and normalize)
    aligned_face = cv2.resize(face_roi, (112, 112))
    aligned_face = aligned_face.astype(np.float32) / 255.0
    aligned_face = (aligned_face - 0.5) / 0.5 # Normalize to [-1, 1]
    
    # Reshape for the ONNX model: (Batch, Channels, Height, Width)
    input_blob = np.expand_dims(aligned_face.transpose(2, 0, 1), axis=0)

    # Get the embedding from the ONNX session
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    embedding = onnx_session.run([output_name], {input_name: input_blob})[0]
    
    return embedding

# --- The rest of your file is largely the same, just calling the new functions ---

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
    # Get embeddings for all profile images first
    profile_embeddings = [get_face_embedding(p_img) for p_img in profile_images]
    profile_embeddings = [emb for emb in profile_embeddings if emb is not None]

    if CONFIG["FLIP_FALLBACK_ENABLED"]:
        flipped_images = [cv2.flip(p_img, 1) for p_img in profile_images]
        flipped_embeddings = [get_face_embedding(f_img) for f_img in flipped_images]
        profile_embeddings.extend([emb for emb in flipped_embeddings if emb is not None])

    if not profile_embeddings:
        return {"status": "Failed", "message": "Could not find a face in any profile image."}

    # Check each video frame against all profile embeddings
    for frame in video_frames:
        frame_embedding = get_face_embedding(frame)
        if frame_embedding is None:
            continue

        for p_emb in profile_embeddings:
            similarity = cosine_similarity(frame_embedding, p_emb)[0][0]
            if similarity > CONFIG["SIMILARITY_THRESHOLD"]:
                return {"status": "Successful", "message": f"Match found with similarity {similarity:.2f}"}

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
