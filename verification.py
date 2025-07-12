import cv2
import numpy as np
import os
import requests
from deepface import DeepFace
import tempfile # Use the tempfile library for safer file handling

# --- CONFIGURATION ---
# The configuration has been updated to use a lightweight model ("SFace")
# and the lightest available face detector ("opencv") to reduce memory usage.
CONFIG = {
    "NUM_FRAMES": 10,
    "MODEL_NAME": "SFace",            # Switched from "ArcFace" to the much lighter SFace
    "DETECTOR_BACKEND": "opencv",     # Added to use the fast, low-memory OpenCV detector
    "DISTANCE_THRESHOLD": 0.55,       # Adjusted for SFace. You might need to tune this value after testing.
    "FLIP_FALLBACK_ENABLED": True
}

def download_image_from_url(url: str):
    """Downloads an image from a URL and decodes it into an OpenCV object."""
    try:
        # Added a timeout to prevent hanging requests
        resp = requests.get(url, timeout=10)
        resp.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        arr = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image {url}: {e}")
        return None
    except Exception as e:
        print(f"Error processing image {url}: {e}")
        return None

def download_video_from_url(url: str):
    """Downloads video content from a URL as bytes."""
    try:
        # Added a timeout to prevent hanging requests
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return resp.content
    except requests.exceptions.RequestException as e:
        print(f"Error downloading video {url}: {e}")
        return None

def extract_frames_from_video(video_bytes: bytes) -> list:
    """Extracts a configured number of frames evenly from video bytes."""
    # Use a temporary file with a guaranteed unique name to avoid race conditions
    # in a multi-worker environment.
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        temp_video.write(video_bytes)
        temp_video_path = temp_video.name
    
    frames = []
    cap = None
    try:
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file at {temp_video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return []
            
        interval = max(total_frames // CONFIG["NUM_FRAMES"], 1)
        
        for i in range(CONFIG["NUM_FRAMES"]):
            frame_pos = i * interval
            if frame_pos >= total_frames:
                break
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
    finally:
        if cap:
            cap.release()
        os.remove(temp_video_path) # Clean up the uniquely named file
    
    return frames

def run_face_verification(video_frames: list, profile_images: list) -> dict:
    """Iterates through frames and profile images to find a match using DeepFace."""
    for frame in video_frames:
        for p_image in profile_images:
            # This inner loop checks the original image and, if enabled, a flipped version.
            images_to_check = [(p_image, "Match found.")]
            if CONFIG["FLIP_FALLBACK_ENABLED"]:
                flipped_image = cv2.flip(p_image, 1)
                images_to_check.append((flipped_image, "Match found with flipped image."))

            for image_variant, success_message in images_to_check:
                try:
                    result = DeepFace.verify(
                        img1_path=frame,
                        img2_path=image_variant,
                        model_name=CONFIG["MODEL_NAME"],
                        detector_backend=CONFIG["DETECTOR_BACKEND"], # Pass the detector
                        enforce_detection=True
                    )
                    
                    if result.get("verified") and result["distance"] < CONFIG["DISTANCE_THRESHOLD"]:
                        return {"status": "Successful", "message": success_message}
                except Exception:
                    # This can happen if DeepFace fails to find a face.
                    # We simply continue to the next image/frame.
                    continue

    return {"status": "Failed", "message": "No match found."}

def process_verification_from_urls(video_url: str, image_urls: list) -> dict:
    """Main processing function to orchestrate the verification flow."""
    profile_images = [download_image_from_url(url) for url in image_urls]
    # Filter out any images that failed to download or process
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
