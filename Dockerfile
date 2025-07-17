# Start with an official, lightweight Python base image.
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies needed by opencv-python
RUN apt-get update && apt-get install -y libgl1

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# --- THE FIX IS HERE ---
# Run the startup script during the build process. This downloads the model
# and caches it within the Docker image itself.
RUN python -c "from startup import preload_deepface_models; preload_deepface_models()"

# Tell Docker that your application listens on a port
EXPOSE 8080

# The command to run when the container starts.
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:$PORT", "--timeout", "120"]
