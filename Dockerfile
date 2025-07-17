# Start with an official, lightweight Python base image.
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# --- THE FIX IS HERE ---
# Install the full set of system dependencies needed by opencv-python.
# This is a common requirement for running OpenCV in a minimal container.
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Run the startup script during the build process to download and cache the models.
RUN python -c "from startup import preload_deepface_models; preload_deepface_models()"

# Tell Docker that your application listens on a port
EXPOSE 8080

# The command to run when the container starts.
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:$PORT", "--timeout", "120"]
