# Start with an official, lightweight Python base image.
# This is a Linux environment, just like your Google Cloud VM.
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

# Tell Docker that your application listens on port 8080
EXPOSE 8080

# The command to run when the container starts.
# This is the same Gunicorn command you used on the server.
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8080", "--timeout", "120"]
