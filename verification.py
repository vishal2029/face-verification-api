import cv2
import numpy as np
import os
import requests
from deepface import DeepFace
from startup import get_arcface_model

CONFIG = {
    "NUM_FRAMES": 10,
    "MODEL_NAME": "ArcFace",
    "DISTANCE_THRESHOLD": 0.40,
    "FLIP_FALLBACK_ENABLED": True
}

def extract_frames_from_video_url(video_url: str) -> list:
    try:
        response = requests.get(video_url, timeout=20)
        if response.status_code != 200:
            return []

        temp_video_path = "temp_video_url.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(response.content)

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

    except Exception as e:
        print("Video frame extraction failed:", e)
        return []

def run_face_verification_from_urls(video_frames: list, image_urls: list) -> dict:
    model = get_arcface_model()

    profile_images = []
    for url in image_urls:
        try:
            img_resp = requests.get(url, timeout=10)
            img_arr = np.frombuffer(img_resp.content, np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img is not None:
                profile_images.append(img)
        except Exception:
            continue

    if not profile_images:
        return {"verified": False, "reason": "No valid images with faces found"}

    for frame in video_frames:
        for p_image in profile_images:
            try:
                result = DeepFace.verify(
                    img1_path=frame,
                    img2_path=p_image,
                    model_name="ArcFace",
                    model=model,
                    enforce_detection=True
                )
                if result["verified"] and result["distance"] < CONFIG["DISTANCE_THRESHOLD"]:
                    return {"verified": True, "reason": "Match found"}
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
                        model_name="ArcFace",
                        model=model,
                        enforce_detection=True
                    )
                    if result["verified"] and result["distance"] < CONFIG["DISTANCE_THRESHOLD"]:
                        return {"verified": True, "reason": "Match found with flipped image"}
                except Exception:
                    continue

    return {"verified": False, "reason": "No match found"}
