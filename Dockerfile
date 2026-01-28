# RunPod Serverless Dockerfile for SeedVC V2 Voice Conversion
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

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

# Install Python dependencies
WORKDIR /workspace/seed-vc

# Install dependencies manually (skip conflicting torch versions from requirements.txt)
RUN pip install --no-cache-dir \
    accelerate \
    scipy==1.13.1 \
    librosa==0.10.2 \
    huggingface-hub>=0.28.1 \
    munch==4.0.0 \
    einops==0.8.0 \
    descript-audio-codec==1.0.0 \
    pydub==0.25.1 \
    resemblyzer \
    jiwer==3.0.3 \
    transformers==4.46.3 \
    soundfile==0.12.1 \
    sounddevice==0.5.0 \
    numpy==1.26.4 \
    hydra-core==1.3.2 \
    pyyaml \
    python-dotenv \
    runpod \
    requests \
    omegaconf

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
