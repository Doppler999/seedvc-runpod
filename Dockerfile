# RunPod Serverless Dockerfile for SeedVC V2 Voice Conversion
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
RUN pip install --no-cache-dir runpod requests hydra-core omegaconf

# Copy handler
COPY handler.py /workspace/handler.py

# Pre-download V2 models on build to reduce cold start time
RUN python -c "from hydra.utils import instantiate; from omegaconf import DictConfig; import yaml; \
    cfg = DictConfig(yaml.safe_load(open('configs/v2/vc_wrapper.yaml', 'r'))); \
    wrapper = instantiate(cfg); wrapper.load_checkpoints(); \
    print('V2 models downloaded successfully')" || echo "Model pre-download skipped"

WORKDIR /workspace

# Start the handler
CMD ["python", "-u", "handler.py"]
