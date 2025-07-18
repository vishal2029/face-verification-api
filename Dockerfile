# Start with an official, lightweight Python base image.
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# --- THE DEFINITIVE FIX ---
# Install the full set of common system dependencies needed by OpenCV
# in a headless Linux environment. This prevents all the "cannot open
# shared object file" errors we saw before.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Run the startup script during the build process to download and cache the models.
# This "bakes" the model into the image so the container starts instantly.
RUN python -c "from startup import preload_deepface_models; preload_deepface_models()"

# Tell Docker that your application listens on port 8080
EXPOSE 8080

# The command to run when the container starts. We use a simple bind here.
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8080", "--timeout", "120"]
