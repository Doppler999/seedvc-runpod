"""
SeedVC FastAPI Server for Vast.ai Always-On Deployment
Supports V1 (Fast - lip-sync) and V2 (Quality - voice emphasis) modes
"""

import os
import sys
import time
import base64
import tempfile
import torch
import torchaudio
import numpy as np

# Add seed-vc to path
sys.path.insert(0, "/app/seed-vc")

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vc_wrapper_v1 = None  # V1: Voice & Singing Voice Conversion (fast, lip-sync)
vc_wrapper_v2 = None  # V2: Voice & Style Conversion (quality, voice emphasis)
model_load_time = None

# GPU pricing for cost tracking
GPU_COSTS = {
    "RTX 3060": 0.10, "RTX 3070": 0.15, "RTX 3080": 0.20, "RTX 3090": 0.30,
    "RTX 4070": 0.25, "RTX 4080": 0.35, "RTX 4090": 0.50,
    "A10": 0.35, "A40": 0.50, "A100": 1.50, "H100": 3.00,
}

def get_gpu_info():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return gpu_name, vram
    return "CPU", 0

def get_gpu_cost_per_hour(gpu_name):
    for key, cost in GPU_COSTS.items():
        if key.lower() in gpu_name.lower():
            return cost
    return 0.20

def get_machine_info():
    import platform
    return {
        "hostname": platform.node(),
        "system": platform.system(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
    }

def load_models():
    """Load both V1 and V2 models"""
    global vc_wrapper_v1, vc_wrapper_v2, model_load_time
    
    if vc_wrapper_v1 is not None and vc_wrapper_v2 is not None:
        return
    
    start_time = time.time()
    print(f"🔄 Loading SeedVC models on {device}...")
    
    try:
        # Load V1 model (Voice & Singing Voice Conversion - fast)
        from modules.voice_conversion.vc_wrapper import VCWrapper as VCWrapperV1
        vc_wrapper_v1 = VCWrapperV1(device=device)
        print("✅ V1 model loaded (Voice & Singing VC)")
        
        # Load V2 model (Voice & Style Conversion - quality)
        from modules.voice_conversion.vc_wrapper_v2 import VCWrapper as VCWrapperV2
        vc_wrapper_v2 = VCWrapperV2(device=device)
        print("✅ V2 model loaded (Voice & Style VC)")
        
        model_load_time = time.time() - start_time
        print(f"✅ All models loaded in {model_load_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        raise

def convert_voice(
    source_audio_bytes: bytes,
    target_audio_bytes: bytes,
    mode: str = "fast",
    diffusion_steps: int = None,
    length_adjust: float = 1.0,
    # V1 params
    inference_cfg_rate: float = 0.7,
    auto_f0_adjust: bool = True,
    pitch_shift: int = 0,
    # V2 params
    intelligibility_cfg_rate: float = 0.5,
    similarity_cfg_rate: float = 0.5,
    top_p: float = 0.9,
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
    convert_style: bool = False,
):
    """Convert voice using V1 or V2 model based on mode"""
    
    start_time = time.time()
    
    # Set default diffusion steps based on mode
    if diffusion_steps is None:
        diffusion_steps = 10 if mode == "fast" else 40
    
    # Save audio to temp files
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(source_audio_bytes)
        source_path = f.name
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(target_audio_bytes)
        target_path = f.name
    
    try:
        if mode == "fast":
            # V1: Voice & Singing Voice Conversion (fast, preserves timing)
            print(f"🎤 V1 Fast mode: steps={diffusion_steps}, cfg={inference_cfg_rate}, f0={auto_f0_adjust}, pitch={pitch_shift}")
            
            audio_segments, sample_rate = vc_wrapper_v1.convert_voice_with_streaming(
                source_path,
                target_path,
                diffusion_steps=diffusion_steps,
                length_adjust=length_adjust,
                inference_cfg_rate=inference_cfg_rate,
                auto_f0_adjust=auto_f0_adjust,
                pitch_shift=pitch_shift,
            )
        else:
            # V2: Voice & Style Conversion (quality, emphasizes target voice)
            print(f"🎤 V2 Quality mode: steps={diffusion_steps}, intel={intelligibility_cfg_rate}, sim={similarity_cfg_rate}")
            
            audio_segments, sample_rate = vc_wrapper_v2.convert_voice_with_streaming(
                source_path,
                target_path,
                diffusion_steps=diffusion_steps,
                length_adjust=length_adjust,
                intelligebility_cfg_rate=intelligibility_cfg_rate,
                similarity_cfg_rate=similarity_cfg_rate,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                convert_style=convert_style,
            )
        
        # Concatenate audio segments
        if isinstance(audio_segments, list):
            full_audio = np.concatenate(list(audio_segments))
        else:
            full_audio = audio_segments
        
        # Convert to tensor and save
        audio_tensor = torch.from_numpy(full_audio).unsqueeze(0)
        if audio_tensor.dtype != torch.float32:
            audio_tensor = audio_tensor.float()
        
        # Save to temp file
        output_path = tempfile.mktemp(suffix=".wav")
        torchaudio.save(output_path, audio_tensor, sample_rate)
        
        # Read and encode
        with open(output_path, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        duration = len(full_audio) / sample_rate
        inference_time = time.time() - start_time
        
        metrics = {
            "mode": mode,
            "diffusion_steps": diffusion_steps,
            "inference_time_s": round(inference_time, 2),
            "audio_duration_s": round(duration, 2),
            "realtime_factor": round(duration / inference_time, 2) if inference_time > 0 else 0,
        }
        
        print(f"✅ Conversion complete: {duration:.2f}s audio in {inference_time:.2f}s ({metrics['realtime_factor']}x realtime)")
        
        # Cleanup
        os.unlink(output_path)
        
        return audio_base64, sample_rate, duration, metrics, output_path if os.path.exists(output_path) else None
        
    finally:
        os.unlink(source_path)
        os.unlink(target_path)


# FastAPI App
app = FastAPI(title="SeedVC API", version="2.0.0", description="V1 Fast + V2 Quality modes")

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
        "models_loaded": {
            "v1_fast": vc_wrapper_v1 is not None,
            "v2_quality": vc_wrapper_v2 is not None,
        },
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
    mode: str = Form("fast"),
    diffusion_steps: int = Form(None),
    length_adjust: float = Form(1.0),
    # V1 params
    inference_cfg_rate: float = Form(0.7),
    auto_f0_adjust: bool = Form(True),
    pitch_shift: int = Form(0),
    # V2 params
    intelligibility_cfg_rate: float = Form(0.5),
    similarity_cfg_rate: float = Form(0.5),
    top_p: float = Form(0.9),
    temperature: float = Form(1.0),
    repetition_penalty: float = Form(1.0),
    convert_style: bool = Form(False),
):
    """Convert voice using V1 (fast) or V2 (quality) mode"""
    
    if vc_wrapper_v1 is None or vc_wrapper_v2 is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if mode not in ["fast", "quality"]:
        raise HTTPException(status_code=400, detail="Mode must be 'fast' or 'quality'")
    
    try:
        source_bytes = await source_audio.read()
        target_bytes = await target_audio.read()
        
        audio_base64, sample_rate, duration, metrics, _ = convert_voice(
            source_bytes,
            target_bytes,
            mode=mode,
            diffusion_steps=diffusion_steps,
            length_adjust=length_adjust,
            inference_cfg_rate=inference_cfg_rate,
            auto_f0_adjust=auto_f0_adjust,
            pitch_shift=pitch_shift,
            intelligibility_cfg_rate=intelligibility_cfg_rate,
            similarity_cfg_rate=similarity_cfg_rate,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
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
            headers={
                "X-Mode": mode,
                "X-Duration": str(duration),
                "X-Inference-Time": str(metrics["inference_time_s"]),
            }
        )
        
    except Exception as e:
        import traceback
        print(f"❌ ERROR: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting SeedVC server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
