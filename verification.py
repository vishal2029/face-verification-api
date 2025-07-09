import cv2
import numpy as np
import os
from deepface import DeepFace

CONFIG = {
    "NUM_FRAMES": 10,
    "MODEL_NAME": "ArcFace",
    "DISTANCE_THRESHOLD": 0.40,
    "FLIP_FALLBACK_ENABLED": True
}

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
    # First pass: exact
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
                    return {"verified": True, "reason": "Match found."}
            except Exception:
                continue

    # If no match and flipping allowed, try mirrored
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
                        return {"verified": True, "reason": "Match found with flipped image."}
                except Exception:
                    continue

    return {"verified": False, "reason": "No match found."}