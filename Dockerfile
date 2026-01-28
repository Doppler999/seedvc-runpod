# RunPod Serverless Dockerfile for SeedVC V2 Voice Conversion
# Using runpod base image for better compatibility
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /workspace

# Set timezone non-interactively
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Clone SeedVC repository
RUN git clone https://github.com/Plachtaa/seed-vc.git /workspace/seed-vc

WORKDIR /workspace/seed-vc

# Filter out torch lines from requirements and install the rest
RUN grep -v "^torch" requirements.txt | grep -v "^#" | grep -v "^$" > requirements_filtered.txt || true
RUN pip install --no-cache-dir -r requirements_filtered.txt || true

# Install additional required packages
RUN pip install --no-cache-dir runpod requests

# Copy handler
COPY handler.py /workspace/handler.py

WORKDIR /workspace

# Start the handler
CMD ["python", "-u", "handler.py"]
