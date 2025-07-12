from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List
from fastapi.responses import PlainTextResponse
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
    """Runs when the FastAPI application starts up."""
    preload_deepface_models()

@app.post("/verify-user-url/")
async def verify_user_url(request: VerificationRequest):
    """
    Accepts a video URL and image URLs, and performs face verification.
    """
    try:
        # It's good practice to convert Pydantic types to basic Python types (str)
        # before passing them to backend functions.
        video_url_str = str(request.video_url)
        image_urls_str = [str(url) for url in request.image_urls]

        result = process_verification_from_urls(
            video_url=video_url_str,
            image_urls=image_urls_str
        )
        return result
    except Exception as e:
        # In a real production app, you would log this exception to a monitoring service.
        print(f"An unexpected error occurred during verification for id {request.id}: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred during verification.")

@app.get("/", tags=["Health Check"])
def read_root():
    """Standard health check endpoint."""
    return {"status": "ok"}

@app.get("/healthz", include_in_schema=False)
def healthz():
    """Render-compatible health check endpoint."""
    return PlainTextResponse("ok", status_code=200)
