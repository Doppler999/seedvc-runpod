"""
SeedVC Voice Conversion Worker (V1 + V2)
Compatible with WaveSpeed (Waverless) and RunPod Serverless

Voice Modes:
- V1 (Fast): Ultra-fast, preserves original timing/pitch - ideal for lip-sync
- V2 (Quality): Slightly slower, emphasizes target voice - better voice quality

Server Modes:
- QUEUE MODE (default): For WaveSpeed/RunPod serverless queue-based endpoints
- LOAD_BALANCER MODE: For RunPod Load Balancing HTTP endpoints (commented out below)

To switch server modes, comment/uncomment the appropriate sections at the bottom of this file.
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
import io
import runpod

# Global model variables - loaded once on startup
vc_wrapper_v1 = None  # V1: Voice & Singing - Fast, preserves timing
vc_wrapper_v2 = None  # V2: Voice & Style - Quality, emphasizes target voice
device = None
dtype = torch.float16
model_load_time = None

# GPU pricing (per hour) - for cost estimation
# WaveSpeed pricing
GPU_PRICING = {
    "NVIDIA L4": 0.55,
    "NVIDIA A100 40GB": 1.19,
    "NVIDIA A100 80GB": 1.49,
    "NVIDIA H100": 3.07,
    "NVIDIA H200": 3.59,
    "NVIDIA B200": 5.19,
    # RunPod pricing (for reference)
    "NVIDIA RTX 4090": 0.44,
    "NVIDIA RTX A4000": 0.20,
    "NVIDIA RTX A5000": 0.28,
    "NVIDIA RTX A6000": 0.53,
    "NVIDIA L40": 0.69,
    "NVIDIA L40S": 0.74,
    "NVIDIA A40": 0.49,
    "default": 0.50,
}

def get_gpu_info():
    """Get GPU name and memory info"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return gpu_name, total_mem
    return "CPU", 0

def get_machine_info():
    """Get machine/pod info from environment"""
    return {
        "pod_id": os.environ.get("RUNPOD_POD_ID", "unknown"),
        "gpu_count": os.environ.get("RUNPOD_GPU_COUNT", "1"),
        "cpu_count": os.environ.get("RUNPOD_CPU_COUNT", "unknown"),
        "mem_gb": os.environ.get("RUNPOD_MEM_GB", "unknown"),
        "volume_id": os.environ.get("RUNPOD_VOLUME_ID", "none"),
        "dc_id": os.environ.get("RUNPOD_DC_ID", "unknown"),
    }

def get_gpu_cost_per_hour(gpu_name):
    """Get estimated cost per hour for the GPU"""
    for key, price in GPU_PRICING.items():
        if key.lower() in gpu_name.lower():
            return price
    return GPU_PRICING["default"]

def load_models():
    """Load both SeedVC V1 and V2 models on startup"""
    global vc_wrapper_v1, vc_wrapper_v2, device, dtype, model_load_time
    
    if vc_wrapper_v1 is not None and vc_wrapper_v2 is not None:
        return
    
    load_start = time.time()
    print("=" * 60)
    print("🚀 SEEDVC V1+V2 WORKER STARTING")
    print("=" * 60)
    
    # Get machine info
    machine = get_machine_info()
    print(f"🖥️  Pod ID: {machine['pod_id']}")
    print(f"🌐 Datacenter: {machine['dc_id']}")
    print(f"💻 CPUs: {machine['cpu_count']}, RAM: {machine['mem_gb']}GB")
    print("-" * 60)
    
    sys.path.insert(0, "/workspace/seed-vc")
    os.chdir("/workspace/seed-vc")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name, gpu_mem = get_gpu_info()
    gpu_cost = get_gpu_cost_per_hour(gpu_name)
    
    print(f"📊 GPU: {gpu_name}")
    print(f"💾 VRAM: {gpu_mem:.1f} GB")
    print(f"💰 Estimated cost: ${gpu_cost:.2f}/hour")
    print("-" * 60)
    
    # Load V1 models (Fast - Voice & Singing) - uses SeedVCWrapper directly
    print("📦 Loading SeedVC V1 models (Fast mode)...")
    from seed_vc_wrapper import SeedVCWrapper
    vc_wrapper_v1 = SeedVCWrapper(device=device)
    print(f"✅ V1 loaded | GPU Memory: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
    
    # Load V2 models (Quality - Voice & Style) - uses hydra config
    print("📦 Loading SeedVC V2 models (Quality mode)...")
    from hydra.utils import instantiate
    from omegaconf import DictConfig
    cfg_v2 = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
    vc_wrapper_v2 = instantiate(cfg_v2)
    vc_wrapper_v2.load_checkpoints()
    vc_wrapper_v2.to(device)
    vc_wrapper_v2.eval()
    vc_wrapper_v2.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=device)
    print(f"✅ V2 loaded | GPU Memory: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
    
    model_load_time = time.time() - load_start
    
    print("-" * 60)
    print(f"✅ All models loaded in {model_load_time:.2f}s")
    print(f"💾 Total GPU Memory used: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
    print("=" * 60)


def convert_voice(source_audio_bytes, target_audio_bytes, mode="fast", diffusion_steps=None, length_adjust=1.0,
                  # V1 parameters (for testing - will be hardcoded in production)
                  inference_cfg_rate=0.7, auto_f0_adjust=True, pitch_shift=0,
                  # V2 parameters (for testing - will be hardcoded in production)
                  intelligibility_cfg_rate=0.5, similarity_cfg_rate=0.5, 
                  top_p=0.9, temperature=1.0, repetition_penalty=1.0, convert_style=False):
    """
    Core voice conversion function supporting both V1 (fast) and V2 (quality) modes.
    
    Modes:
    - "fast" (V1): Ultra-fast, preserves timing/pitch - ideal for lip-sync
    - "quality" (V2): Slightly slower, emphasizes target voice - better quality
    
    Returns: (audio_base64, sample_rate, duration, metrics)
    """
    import numpy as np
    
    request_start = time.time()
    
    # Get GPU info for cost calculation
    gpu_info = get_gpu_info()
    gpu_cost_per_hour = get_gpu_cost_per_hour(gpu_info[0])
    machine = get_machine_info()
    
    # Determine mode and set default diffusion steps
    is_v1 = mode.lower() in ["fast", "v1", "lipsync"]
    mode_name = "V1 (Fast/Lip-sync)" if is_v1 else "V2 (Quality)"
    
    # Default diffusion steps based on mode
    if diffusion_steps is None:
        diffusion_steps = 10 if is_v1 else 40
    
    print("=" * 60)
    print(f"🎤 VOICE CONVERSION REQUEST - {mode_name}")
    print("=" * 60)
    print(f"🖥️  Pod: {machine['pod_id']} | DC: {machine['dc_id']}")
    print(f"📊 GPU: {gpu_info[0]}")
    print(f"💰 Cost rate: ${gpu_cost_per_hour:.2f}/hour")
    print(f"🔧 Mode: {mode_name}")
    print(f"⚙️  Diffusion steps: {diffusion_steps}")
    print(f"⚙️  Length adjust: {length_adjust}")
    if is_v1:
        print(f"⚙️  Inference CFG rate: {inference_cfg_rate}")
        print(f"⚙️  Auto F0 adjust: {auto_f0_adjust}")
        print(f"⚙️  Pitch shift: {pitch_shift}")
    else:
        print(f"⚙️  Intelligibility CFG: {intelligibility_cfg_rate}")
        print(f"⚙️  Similarity CFG: {similarity_cfg_rate}")
        print(f"⚙️  Top-p: {top_p}")
        print(f"⚙️  Temperature: {temperature}")
        print(f"⚙️  Repetition penalty: {repetition_penalty}")
        print(f"⚙️  Convert style: {convert_style}")
    print("-" * 60)
    
    # Save input files
    io_start = time.time()
    src_path = tempfile.mktemp(suffix=".wav")
    tgt_path = tempfile.mktemp(suffix=".wav")
    
    with open(src_path, "wb") as f:
        f.write(source_audio_bytes)
    with open(tgt_path, "wb") as f:
        f.write(target_audio_bytes)
    
    io_time = time.time() - io_start
    print(f"📁 Input files: source={len(source_audio_bytes)/1024:.1f}KB, target={len(target_audio_bytes)/1024:.1f}KB")
    print(f"⏱️  I/O time: {io_time:.2f}s")
    
    # Run inference
    inference_start = time.time()
    os.chdir("/workspace/seed-vc")
    
    full_audio = None
    
    if is_v1:
        # V1 Mode: Fast, preserves timing - ideal for lip-sync
        # Uses SeedVCWrapper.convert_voice which yields (mp3_bytes, numpy_audio)
        for mp3_bytes, audio_result in vc_wrapper_v1.convert_voice(
            source=src_path,
            target=tgt_path,
            diffusion_steps=min(diffusion_steps, 200),
            length_adjust=length_adjust,
            inference_cfg_rate=inference_cfg_rate,
            f0_condition=False,
            auto_f0_adjust=auto_f0_adjust,
            pitch_shift=pitch_shift,
            stream_output=True,
        ):
            full_audio = audio_result
    else:
        # V2 Mode: Quality, emphasizes target voice
        # Hardcoded optimal parameters for V2:
        # - diffusion_steps: 40
        # - intelligibility_cfg_rate: 0.5
        # - similarity_cfg_rate: 0.5
        # - top_p: 0.9
        # - temperature: 1.0
        # - repetition_penalty: 1.0
        # - convert_style: False (MUST be disabled for best results)
        for mp3_bytes, audio_result in vc_wrapper_v2.convert_voice_with_streaming(
            source_audio_path=src_path,
            target_audio_path=tgt_path,
            diffusion_steps=min(diffusion_steps, 200),
            length_adjust=length_adjust,
            intelligebility_cfg_rate=intelligibility_cfg_rate,
            similarity_cfg_rate=similarity_cfg_rate,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            convert_style=convert_style,
            anonymization_only=False,
            device=device,
            dtype=dtype,
            stream_output=True,
        ):
            full_audio = audio_result
    
    inference_time = time.time() - inference_start
    
    if full_audio is None:
        raise Exception("Conversion returned no audio")
    
    sample_rate, audio_array = full_audio
    
    if isinstance(audio_array, np.ndarray):
        audio_tensor = torch.from_numpy(audio_array).float()
    else:
        audio_tensor = audio_array
    
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    # Calculate audio duration
    audio_duration = audio_tensor.shape[1] / sample_rate
    
    # Save to buffer and encode as base64
    output_start = time.time()
    buffer = io.BytesIO()
    torchaudio.save(buffer, audio_tensor, sample_rate, format="wav")
    buffer.seek(0)
    audio_bytes = buffer.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
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
    print(f"📁 Output size: {len(audio_bytes)/1024:.1f}KB")
    print(f"⏱️  Inference time: {inference_time:.2f}s")
    print(f"⏱️  Output save time: {output_time:.2f}s")
    print(f"⏱️  Total request time: {total_time:.2f}s")
    print(f"💾 GPU memory used: {torch.cuda.memory_allocated(0) / (1024**3):.2f}GB")
    print("-" * 60)
    print(f"💰 ESTIMATED COST: ${estimated_cost:.6f}")
    print(f"💰 Cost per audio second: ${estimated_cost/audio_duration:.6f}")
    print("=" * 60)
    
    metrics = {
        "inference_time_s": round(inference_time, 2),
        "total_time_s": round(total_time, 2),
        "audio_duration_s": round(audio_duration, 2),
        "output_size_kb": round(len(audio_bytes) / 1024, 1),
        "gpu": gpu_info[0],
        "cost_estimate": round(estimated_cost, 6),
        "cost_per_audio_second": round(estimated_cost / audio_duration, 6),
    }
    
    return audio_base64, sample_rate, audio_duration, metrics


# =============================================================================
# QUEUE MODE - For WaveSpeed (Waverless) and RunPod Serverless Queue
# =============================================================================

def handler(job):
    """
    Queue-based handler for WaveSpeed/RunPod serverless.
    Supports both V1 (fast/lip-sync) and V2 (quality) modes.
    
    Input format:
    {
        "input": {
            "source_audio": "<base64 encoded audio>",
            "target_audio": "<base64 encoded audio>",
            "source_is_url": false,  # optional, if true source_audio is a URL
            "target_is_url": false,  # optional, if true target_audio is a URL
            "mode": "fast",          # "fast" (V1) or "quality" (V2), default: "fast"
            "diffusion_steps": null, # optional, auto-set based on mode (10 for fast, 40 for quality)
            "length_adjust": 1.0,    # optional
            
            # V1 parameters (only used when mode="fast")
            "inference_cfg_rate": 0.7,  # optional
            "auto_f0_adjust": true,     # optional
            "pitch_shift": 0,           # optional
            
            # V2 parameters (only used when mode="quality")
            "intelligibility_cfg_rate": 0.5,  # optional
            "similarity_cfg_rate": 0.5,       # optional
            "top_p": 0.9,                     # optional
            "temperature": 1.0,               # optional
            "repetition_penalty": 1.0,        # optional
            "convert_style": false            # optional, MUST be false for best results
        }
    }
    
    Output format:
    {
        "audio": "<base64 encoded wav>",
        "sample_rate": 22050,
        "duration_s": 8.5,
        "mode": "fast",
        "metrics": { ... }
    }
    """
    try:
        # Load models on first request
        load_models()
        
        job_input = job.get("input", {})
        
        # Get audio inputs
        source_audio = job_input.get("source_audio")
        target_audio = job_input.get("target_audio")
        source_is_url = job_input.get("source_is_url", False)
        target_is_url = job_input.get("target_is_url", False)
        
        if not source_audio or not target_audio:
            return {"error": "Missing source_audio or target_audio"}
        
        # Fetch from URL or decode base64
        if source_is_url:
            print(f"📥 Fetching source audio from URL...")
            response = http_requests.get(source_audio, timeout=60)
            response.raise_for_status()
            source_bytes = response.content
        else:
            source_bytes = base64.b64decode(source_audio)
        
        if target_is_url:
            print(f"📥 Fetching target audio from URL...")
            response = http_requests.get(target_audio, timeout=60)
            response.raise_for_status()
            target_bytes = response.content
        else:
            target_bytes = base64.b64decode(target_audio)
        
        # Get mode (default: fast/V1 for lip-sync)
        mode = job_input.get("mode", "fast")
        
        # Get common parameters
        diffusion_steps = job_input.get("diffusion_steps", None)  # Auto-set based on mode
        length_adjust = job_input.get("length_adjust", 1.0)
        
        # Get V1 parameters (used when mode="fast")
        inference_cfg_rate = job_input.get("inference_cfg_rate", 0.7)
        auto_f0_adjust = job_input.get("auto_f0_adjust", True)
        pitch_shift = job_input.get("pitch_shift", 0)
        
        # Get V2 parameters (used when mode="quality")
        intelligibility_cfg_rate = job_input.get("intelligibility_cfg_rate", 0.5)
        similarity_cfg_rate = job_input.get("similarity_cfg_rate", 0.5)
        top_p = job_input.get("top_p", 0.9)
        temperature = job_input.get("temperature", 1.0)
        repetition_penalty = job_input.get("repetition_penalty", 1.0)
        convert_style = job_input.get("convert_style", False)  # MUST be False for best results
        
        # Run conversion
        audio_base64, sample_rate, duration, metrics = convert_voice(
            source_bytes,
            target_bytes,
            mode=mode,
            diffusion_steps=diffusion_steps,
            length_adjust=length_adjust,
            # V1 params
            inference_cfg_rate=inference_cfg_rate,
            auto_f0_adjust=auto_f0_adjust,
            pitch_shift=pitch_shift,
            # V2 params
            intelligibility_cfg_rate=intelligibility_cfg_rate,
            similarity_cfg_rate=similarity_cfg_rate,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            convert_style=convert_style,
        )
        
        return {
            "audio": audio_base64,
            "sample_rate": sample_rate,
            "duration_s": duration,
            "mode": mode,
            "metrics": metrics,
        }
        
    except Exception as e:
        import traceback
        print(f"❌ ERROR: {str(e)}")
        traceback.print_exc()
        return {"error": str(e)}


# Start the serverless worker
runpod.serverless.start({"handler": handler})


# =============================================================================
# LOAD BALANCER MODE - For RunPod Load Balancing HTTP endpoints
# Uncomment this section and comment out the queue mode above to use
# =============================================================================
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="SeedVC V2 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/ping")
async def ping():
    return {"status": "healthy"}

@app.get("/health")
async def health():
    gpu_info = get_gpu_info()
    gpu_cost = get_gpu_cost_per_hour(gpu_info[0]) if gpu_info[0] != "CPU" else 0
    machine = get_machine_info()
    
    return {
        "status": "healthy",
        "models_loaded": vc_wrapper_v2 is not None,
        "device": str(device),
        "gpu": gpu_info[0],
        "vram_gb": round(gpu_info[1], 1),
        "cost_per_hour": gpu_cost,
        "model_load_time_s": round(model_load_time, 2) if model_load_time else None,
        "gpu_memory_used_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2) if torch.cuda.is_available() else 0,
        "machine": machine,
    }

@app.post("/convert")
async def convert_voice_endpoint(
    source_audio: UploadFile = File(...),
    target_audio: UploadFile = File(...),
    diffusion_steps: int = Form(30),
    length_adjust: float = Form(1.0),
    convert_style: bool = Form(True),
):
    if vc_wrapper_v2 is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        source_bytes = await source_audio.read()
        target_bytes = await target_audio.read()
        
        audio_base64, sample_rate, duration, metrics = convert_voice(
            source_bytes,
            target_bytes,
            diffusion_steps=diffusion_steps,
            length_adjust=length_adjust,
            convert_style=convert_style,
        )
        
        # Decode and save to temp file for FileResponse
        audio_bytes = base64.b64decode(audio_base64)
        output_path = tempfile.mktemp(suffix=".wav")
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="converted.wav",
            background=None
        )
        
    except Exception as e:
        import traceback
        print(f"❌ ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"{str(e)}\\n{traceback.format_exc()}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
"""
