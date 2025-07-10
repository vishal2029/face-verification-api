from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List
from startup import preload_deepface_models
from verification import process_verification_from_urls

app = FastAPI(title="Face Verification API")

class VerificationRequest(BaseModel):
    video_url: HttpUrl
    image_urls: List[HttpUrl]
    id: str
    verification_attempt: int

@app.on_event("startup")
def startup_event():
    preload_deepface_models()

@app.post("/verify-user-url/")
async def verify_user_url(request: VerificationRequest):
    try:
        result = process_verification_from_urls(
            video_url=request.video_url,
            image_urls=request.image_urls
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok"}
