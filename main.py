from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import numpy as np
import cv2
from verification import extract_frames_from_video, run_face_verification
from startup import preload_deepface_models

app = FastAPI(title="Dating App User Verification API")

@app.on_event("startup")
def startup_event():
    preload_deepface_models()

@app.post("/verify-user/", tags=["Verification"])
async def verify_user(
    profile_images: List[UploadFile] = File(...),
    verification_video: UploadFile = File(...)
):
    print("Received profile images:", profile_images)
    # decode and collect profile images
    imgs = []
    for file in profile_images:
        data = await file.read()
        arr  = np.frombuffer(data, np.uint8)
        img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        imgs.append(img)

    # extract frames from uploaded video
    video_bytes  = await verification_video.read()
    frames       = extract_frames_from_video(video_bytes)
    if not frames:
        raise HTTPException(status_code=400, detail="Could not process video.")

    # run DeepFace verification
    result = run_face_verification(video_frames=frames, profile_images=imgs)
    return result

@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok"}