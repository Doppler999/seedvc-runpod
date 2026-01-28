"""
RunPod Load Balancing FastAPI Server for SeedVC V2 Voice Conversion
Matches the Vast.ai always-on instance configuration
"""
import os
import sys
import base64
import tempfile
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

app = FastAPI(title="SeedVC V2 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_models():
    """Load SeedVC V2 models on startup"""
    global vc_wrapper_v2, device, dtype
    
    if vc_wrapper_v2 is not None:
        return
    
    print("Loading SeedVC V2 models...")
    
    sys.path.insert(0, "/workspace/seed-vc")
    os.chdir("/workspace/seed-vc")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    from hydra.utils import instantiate
    from omegaconf import DictConfig
    
    cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
    vc_wrapper_v2 = instantiate(cfg)
    vc_wrapper_v2.load_checkpoints()
    vc_wrapper_v2.to(device)
    vc_wrapper_v2.eval()
    vc_wrapper_v2.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=device)
    
    print("SeedVC V2 models loaded successfully!")

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/ping")
async def ping():
    """Health check endpoint for RunPod Load Balancing"""
    return {"status": "healthy"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": vc_wrapper_v2 is not None, "device": str(device)}

@app.post("/convert")
async def convert_voice_endpoint(
    source_audio: UploadFile = File(...),
    target_audio: UploadFile = File(...),
    diffusion_steps: int = Form(30),
    length_adjust: float = Form(1.0),
    convert_style: bool = Form(True),
):
    """Convert voice from source to target voice"""
    if vc_wrapper_v2 is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        src_path = tempfile.mktemp(suffix=".wav")
        tgt_path = tempfile.mktemp(suffix=".wav")
        
        with open(src_path, "wb") as f:
            f.write(await source_audio.read())
        with open(tgt_path, "wb") as f:
            f.write(await target_audio.read())
        
        print(f"Converting: {src_path} -> {tgt_path}")
        
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
        
        output_path = tempfile.mktemp(suffix=".wav")
        torchaudio.save(output_path, audio_tensor, sample_rate)
        
        # Clean up input files
        os.remove(src_path)
        os.remove(tgt_path)
        
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="converted.wav",
            background=None
        )
        
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
