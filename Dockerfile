# RunPod Serverless Dockerfile for SeedVC Voice Conversion
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Clone SeedVC repository
RUN git clone https://github.com/Plachtaa/seed-vc.git /workspace/seed-vc

# Install Python dependencies
WORKDIR /workspace/seed-vc
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir runpod requests

# Copy handler
COPY handler.py /workspace/handler.py

# Pre-download models on build (optional but reduces cold start)
RUN python -c "from hf_utils import load_custom_model_from_hf; print('Model download triggered')" || true

WORKDIR /workspace

# Start the handler
CMD ["python", "-u", "handler.py"]
