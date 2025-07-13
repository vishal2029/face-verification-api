import cv2
import numpy as np
import os
import requests
from deepface import DeepFace
import tempfile
import traceback

CONFIG = {
    "NUM_FRAMES": 10,
    "MODEL_NAME": "ArcFace",
    "DISTANCE_THRESHOLD": 0.40,
    "FLIP_FALLBACK_ENABLED": True,
    # New threshold to decide if a flip check is even worth trying.
    "FLIP_CHECK_THRESHOLD": 1.0 
}

def download_image_from_url(url):
    """Downloads an image and decodes it, with specific error handling."""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        arr = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: Failed to decode image from URL: {url}")
            return None
        return img
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image {url}: {e}")
        return None
    except Exception as e:
        print(f"An unknown error occurred while processing image {url}: {e}")
        return None

def download_video_from_url(url):
    """Downloads video content with specific error handling."""
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return resp.content
    except requests.exceptions.RequestException as e:
        print(f"Error downloading video {url}: {e}")
        return None

def extract_frames_from_video(video_bytes: bytes) -> list:
    """Extracts frames using a uniquely named temporary file."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        temp_video.write(video_bytes)
        temp_video_path = temp_video.name
    
    extracted_frames = []
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
                extracted_frames.append(frame)
    finally:
        if cap:
            cap.release()
        os.remove(temp_video_path)
    
    return extracted_frames

def run_face_verification(video_frames: list, profile_images: list) -> dict:
    """
    Iterates through frames and profile images, using the optimized flip logic.
    """
    print("Starting face verification process with optimized flip check...")
    for i, frame in enumerate(video_frames):
        for j, p_image in enumerate(profile_images):
            try:
                # --- STEP 1: Check the original image first ---
                original_result = DeepFace.verify(
                    img1_path=frame,
                    img2_path=p_image,
                    model_name=CONFIG["MODEL_NAME"],
                    enforce_detection=False
                )

                # --- STEP 2: Check for an immediate match ---
                if original_result.get("verified") and original_result.get("distance", 1.0) < CONFIG["DISTANCE_THRESHOLD"]:
                    print(f"Success: Match found on frame {i+1} with original profile image {j+1}.")
                    return {"status": "Successful", "message": "Match found."}

                # --- STEP 3: Decide if a flip check is worthwhile ---
                # If the distance is too high, a flip won't help. Skip it to save time.
                if original_result.get("distance", 1.0) >= CONFIG["FLIP_CHECK_THRESHOLD"]:
                    continue # Move to the next profile image

                # --- STEP 4: If in the "maybe" zone, check the flipped image ---
                if CONFIG["FLIP_FALLBACK_ENABLED"]:
                    flipped_image = cv2.flip(p_image, 1)
                    flipped_result = DeepFace.verify(
                        img1_path=frame,
                        img2_path=flipped_image,
                        model_name=CONFIG["MODEL_NAME"],
                        enforce_detection=False
                    )
                    if flipped_result.get("verified") and flipped_result.get("distance", 1.0) < CONFIG["DISTANCE_THRESHOLD"]:
                        print(f"Success: Match found on frame {i+1} with flipped profile image {j+1}.")
                        return {"status": "Successful", "message": "Match found with flipped image."}

            except Exception:
                # Safety net for any truly unexpected errors
                print("="*20)
                print(f"!! An UNEXPECTED ERROR occurred in DeepFace.verify !!")
                traceback.print_exc()
                print("="*20)
                continue

    # If the loops complete, no match was found.
    print("Verification process completed. No match found.")
    return {"status": "Failed", "message": "No match found."}

def process_verification_from_urls(video_url: str, image_urls: list) -> dict:
    """Main processing function to orchestrate the verification flow."""
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
