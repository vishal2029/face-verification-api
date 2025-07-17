from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List
from fastapi.responses import PlainTextResponse
# We no longer need to import or run the preload function on startup
from verification import process_verification_from_urls

app = FastAPI(title="Face Verification API")

class VerificationRequest(BaseModel):
    video_url: HttpUrl
    image_urls: List[HttpUrl]
    id: str
    verification_attempt: int

# The @app.on_event("startup") block has been removed.
# The model is now loaded when the Docker image is built.

@app.post("/verify-user-url/")
async def verify_user_url(request: VerificationRequest):
    try:
        result = process_verification_from_urls(
            video_url=str(request.video_url),
            image_urls=[str(url) for url in request.image_urls]
        )
        return result
    except Exception as e:
        print(f"A critical error occurred in the main endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok"}

@app.get("/healthz", include_in_schema=False)
def healthz():
    return PlainTextResponse("ok", status_code=200)
