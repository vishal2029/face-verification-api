from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List
from verification import extract_frames_from_video_url, run_face_verification_from_urls

app = FastAPI(title="User Verification via URL")

class VerificationRequest(BaseModel):
    video_url: HttpUrl
    image_urls: List[HttpUrl]
    id: str
    verification_attempt: int

@app.post("/verify-user-url/", tags=["Verification"])
async def verify_user_url(payload: VerificationRequest):
    try:
        frames = extract_frames_from_video_url(payload.video_url)
        if not frames:
            return {"status": "Failed", "message": "No frames extracted from video"}

        result = run_face_verification_from_urls(
            video_frames=frames,
            image_urls=payload.image_urls
        )
        return {"status": "Success" if result["verified"] else "Failed", "message": result["reason"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/", tags=["Health Check"])
def health_check():
    return {"status": "ok"}
