"""
RunPod Load Balancing FastAPI Server for SeedVC V2 Voice Conversion
Matches the Vast.ai always-on instance configuration
"""
import os
import sys
import base64
import tempfile
import time
import torch
import torchaudio
import requests as http_requests
import yaml
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Global model variables - loaded once on startup
vc_wrapper_v2 = None
device = None
dtype = torch.float16
model_load_time = None
gpu_name = None

# RunPod GPU pricing (per hour) - for cost estimation
GPU_PRICING = {
    "NVIDIA RTX 4090": 0.44,
    "NVIDIA RTX A4000": 0.20,
    "NVIDIA RTX A5000": 0.28,
    "NVIDIA RTX A6000": 0.53,
    "NVIDIA L4": 0.34,
    "NVIDIA L40": 0.69,
    "NVIDIA L40S": 0.74,
    "NVIDIA A40": 0.49,
    "NVIDIA A100 40GB": 1.09,
    "NVIDIA A100 80GB": 1.59,
    "NVIDIA H100": 3.99,
    "default": 0.50,  # Fallback estimate
}

app = FastAPI(title="SeedVC V2 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_gpu_info():
    """Get GPU name and memory info"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return gpu_name, total_mem
    return "CPU", 0

def get_gpu_cost_per_hour(gpu_name):
    """Get estimated cost per hour for the GPU"""
    for key, price in GPU_PRICING.items():
        if key.lower() in gpu_name.lower():
            return price
    return GPU_PRICING["default"]

def load_models():
    """Load SeedVC V2 models on startup"""
    global vc_wrapper_v2, device, dtype, model_load_time, gpu_name
    
    if vc_wrapper_v2 is not None:
        return
    
    load_start = time.time()
    print("=" * 60)
    print("🚀 SEEDVC V2 SERVER STARTING")
    print("=" * 60)
    
    sys.path.insert(0, "/workspace/seed-vc")
    os.chdir("/workspace/seed-vc")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name, gpu_mem = get_gpu_info()
    gpu_cost = get_gpu_cost_per_hour(gpu_name)
    
    print(f"📊 GPU: {gpu_name}")
    print(f"💾 VRAM: {gpu_mem:.1f} GB")
    print(f"💰 Estimated cost: ${gpu_cost:.2f}/hour")
    print("-" * 60)
    
    print("📦 Loading SeedVC V2 models...")
    
    from hydra.utils import instantiate
    from omegaconf import DictConfig
    
    cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
    vc_wrapper_v2 = instantiate(cfg)
    vc_wrapper_v2.load_checkpoints()
    vc_wrapper_v2.to(device)
    vc_wrapper_v2.eval()
    vc_wrapper_v2.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=device)
    
    model_load_time = time.time() - load_start
    
    print("-" * 60)
    print(f"✅ Models loaded in {model_load_time:.2f}s")
    print(f"💾 GPU Memory used: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
    print("=" * 60)

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/ping")
async def ping():
    """Health check endpoint for RunPod Load Balancing"""
    return {"status": "healthy"}

@app.get("/health")
async def health():
    """Health check endpoint with detailed info"""
    gpu_info = get_gpu_info()
    gpu_cost = get_gpu_cost_per_hour(gpu_info[0]) if gpu_info[0] != "CPU" else 0
    
    return {
        "status": "healthy",
        "models_loaded": vc_wrapper_v2 is not None,
        "device": str(device),
        "gpu": gpu_info[0],
        "vram_gb": round(gpu_info[1], 1),
        "cost_per_hour": gpu_cost,
        "model_load_time_s": round(model_load_time, 2) if model_load_time else None,
        "gpu_memory_used_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2) if torch.cuda.is_available() else 0,
    }

@app.post("/convert")
async def convert_voice_endpoint(
    source_audio: UploadFile = File(...),
    target_audio: UploadFile = File(...),
    diffusion_steps: int = Form(30),
    length_adjust: float = Form(1.0),
    convert_style: bool = Form(True),
):
    """Convert voice from source to target voice"""
    request_start = time.time()
    
    if vc_wrapper_v2 is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Get GPU info for cost calculation
        gpu_info = get_gpu_info()
        gpu_cost_per_hour = get_gpu_cost_per_hour(gpu_info[0])
        
        print("=" * 60)
        print("🎤 VOICE CONVERSION REQUEST")
        print("=" * 60)
        print(f"📊 GPU: {gpu_info[0]}")
        print(f"💰 Cost rate: ${gpu_cost_per_hour:.2f}/hour")
        print(f"⚙️  Diffusion steps: {diffusion_steps}")
        print(f"⚙️  Length adjust: {length_adjust}")
        print(f"⚙️  Convert style: {convert_style}")
        print("-" * 60)
        
        # Read input files
        io_start = time.time()
        src_path = tempfile.mktemp(suffix=".wav")
        tgt_path = tempfile.mktemp(suffix=".wav")
        
        source_bytes = await source_audio.read()
        target_bytes = await target_audio.read()
        
        with open(src_path, "wb") as f:
            f.write(source_bytes)
        with open(tgt_path, "wb") as f:
            f.write(target_bytes)
        
        io_time = time.time() - io_start
        print(f"📁 Input files: source={len(source_bytes)/1024:.1f}KB, target={len(target_bytes)/1024:.1f}KB")
        print(f"⏱️  I/O time: {io_time:.2f}s")
        
        # Run inference
        inference_start = time.time()
        os.chdir("/workspace/seed-vc")
        
        full_audio = None
        for mp3_bytes, audio_result in vc_wrapper_v2.convert_voice_with_streaming(
            source_audio_path=src_path,
            target_audio_path=tgt_path,
            diffusion_steps=min(diffusion_steps, 50),
            length_adjust=length_adjust,
            intelligebility_cfg_rate=0.7,
            similarity_cfg_rate=0.7,
            top_p=0.7,
            temperature=0.7,
            repetition_penalty=1.5,
            convert_style=convert_style,
            anonymization_only=False,
            device=device,
            dtype=dtype,
            stream_output=True,
        ):
            full_audio = audio_result
        
        inference_time = time.time() - inference_start
        
        if full_audio is None:
            raise HTTPException(status_code=500, detail="Conversion returned no audio")
        
        sample_rate, audio_array = full_audio
        
        import numpy as np
        if isinstance(audio_array, np.ndarray):
            audio_tensor = torch.from_numpy(audio_array).float()
        else:
            audio_tensor = audio_array
        
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Calculate audio duration
        audio_duration = audio_tensor.shape[1] / sample_rate
        
        # Save output
        output_start = time.time()
        output_path = tempfile.mktemp(suffix=".wav")
        torchaudio.save(output_path, audio_tensor, sample_rate)
        output_size = os.path.getsize(output_path)
        output_time = time.time() - output_start
        
        # Clean up input files
        os.remove(src_path)
        os.remove(tgt_path)
        
        # Calculate costs
        total_time = time.time() - request_start
        cost_per_second = gpu_cost_per_hour / 3600
        estimated_cost = total_time * cost_per_second
        
        # Print summary
        print("-" * 60)
        print("📊 CONVERSION COMPLETE")
        print("-" * 60)
        print(f"🎵 Output duration: {audio_duration:.2f}s")
        print(f"📁 Output size: {output_size/1024:.1f}KB")
        print(f"⏱️  Inference time: {inference_time:.2f}s")
        print(f"⏱️  Output save time: {output_time:.2f}s")
        print(f"⏱️  Total request time: {total_time:.2f}s")
        print(f"💾 GPU memory used: {torch.cuda.memory_allocated(0) / (1024**3):.2f}GB")
        print("-" * 60)
        print(f"💰 ESTIMATED COST: ${estimated_cost:.6f}")
        print(f"💰 Cost per audio second: ${estimated_cost/audio_duration:.6f}")
        print("=" * 60)
        
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="converted.wav",
            background=None
        )
        
    except Exception as e:
        import traceback
        total_time = time.time() - request_start
        print(f"❌ ERROR after {total_time:.2f}s: {str(e)}")
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
