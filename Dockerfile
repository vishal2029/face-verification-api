# Start with an official, lightweight Python base image.
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
# We add --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Tell Docker that your application will listen on a port
# (This is more for documentation; the CMD line is what matters)
EXPOSE 8080

# --- THE FIX IS HERE ---
# The command to run when the container starts.
# We now use the $PORT environment variable provided by Cloud Run
# instead of a hardcoded port.
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:$PORT", "--timeout", "120"]
