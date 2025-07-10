import cv2
import numpy as np
import os
import requests
from deepface import DeepFace

CONFIG = {
    "NUM_FRAMES": 10,
    "MODEL_NAME": "ArcFace",
    "DISTANCE_THRESHOLD": 0.40,
    "FLIP_FALLBACK_ENABLED": True
}

def download_image_from_url(url):
    try:
        resp = requests.get(url)
        arr = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except:
        return None

def download_video_from_url(url):
    try:
        resp = requests.get(url)
        return resp.content
    except:
        return None

def extract_frames_from_video(video_bytes: bytes) -> list:
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        os.remove(temp_video_path)
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(total_frames // CONFIG["NUM_FRAMES"], 1)
    extracted_frames = []

    for i in range(CONFIG["NUM_FRAMES"]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if ret:
            extracted_frames.append(frame)

    cap.release()
    os.remove(temp_video_path)
    return extracted_frames

def run_face_verification(video_frames: list, profile_images: list) -> dict:
    for frame in video_frames:
        for p_image in profile_images:
            try:
                result = DeepFace.verify(
                    img1_path=frame,
                    img2_path=p_image,
                    model_name=CONFIG["MODEL_NAME"],
                    enforce_detection=True
                )
                if result["verified"] and result["distance"] < CONFIG["DISTANCE_THRESHOLD"]:
                    return {"status": "Successful", "message": "Match found."}
            except Exception:
                continue

    if CONFIG["FLIP_FALLBACK_ENABLED"]:
        for frame in video_frames:
            for p_image in profile_images:
                try:
                    flipped = cv2.flip(p_image, 1)
                    result = DeepFace.verify(
                        img1_path=frame,
                        img2_path=flipped,
                        model_name=CONFIG["MODEL_NAME"],
                        enforce_detection=True
                    )
                    if result["verified"] and result["distance"] < CONFIG["DISTANCE_THRESHOLD"]:
                        return {"status": "Successful", "message": "Match found with flipped image."}
                except Exception:
                    continue

    return {"status": "Failed", "message": "No match found."}

def process_verification_from_urls(video_url: str, image_urls: list) -> dict:
    profile_images = [download_image_from_url(url) for url in image_urls]
    profile_images = [img for img in profile_images if img is not None]

    if not profile_images:
        return {"status": "Failed", "message": "No valid images with faces were found."}

    video_bytes = download_video_from_url(video_url)
    if not video_bytes:
        return {"status": "Failed", "message": "Could not download video."}

    frames = extract_frames_from_video(video_bytes)
    if not frames:
        return {"status": "Failed", "message": "Could not extract frames from video."}

    return run_face_verification(frames, profile_images)
