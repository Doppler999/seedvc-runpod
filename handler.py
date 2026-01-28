"""
RunPod Serverless Handler for SeedVC Voice Conversion
"""
import runpod
import os
import sys
import base64
import tempfile
import torch
import torchaudio
import requests

# Global model variables - loaded once on cold start
model = None
config = None
device = None

def load_models():
    """Load SeedVC models on cold start"""
    global model, config, device
    
    if model is not None:
        return  # Already loaded
    
    print("Loading SeedVC models...")
    
    # Add seed-vc to path
    sys.path.insert(0, "/workspace/seed-vc")
    os.chdir("/workspace/seed-vc")
    
    from modules.commons import recursive_munch
    from hf_utils import load_custom_model_from_hf
    import yaml
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config
    config_path = "configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config = recursive_munch(config)
    
    # Load model
    model = load_custom_model_from_hf(
        "Plachtaa/Seed-VC",
        "DiT_uvit_tat_xlsr_ema.pth",
        config
    )
    model = model.to(device)
    model.eval()
    
    print("Models loaded successfully!")

def convert_voice(source_audio_path, target_audio_path, output_path):
    """Convert voice from source to target"""
    global model, config, device
    
    sys.path.insert(0, "/workspace/seed-vc")
    os.chdir("/workspace/seed-vc")
    
    from modules.commons import recursive_munch
    import numpy as np
    
    # Load audio files
    source_audio, sr_source = torchaudio.load(source_audio_path)
    target_audio, sr_target = torchaudio.load(target_audio_path)
    
    # Resample to 44100 if needed
    if sr_source != 44100:
        source_audio = torchaudio.functional.resample(source_audio, sr_source, 44100)
    if sr_target != 44100:
        target_audio = torchaudio.functional.resample(target_audio, sr_target, 44100)
    
    # Convert to mono if stereo
    if source_audio.shape[0] > 1:
        source_audio = source_audio.mean(dim=0, keepdim=True)
    if target_audio.shape[0] > 1:
        target_audio = target_audio.mean(dim=0, keepdim=True)
    
    # Move to device
    source_audio = source_audio.to(device)
    target_audio = target_audio.to(device)
    
    # Run inference
    with torch.no_grad():
        converted_audio = model.inference(
            source_audio,
            target_audio,
            diffusion_steps=25,
            length_adjust=1.0,
            inference_cfg_rate=0.7,
        )
    
    # Save output
    converted_audio = converted_audio.cpu()
    torchaudio.save(output_path, converted_audio, 44100)
    
    return output_path

def handler(event):
    """
    RunPod handler for SeedVC voice conversion
    
    Input format:
    {
        "input": {
            "source_audio": "<base64 encoded audio or URL>",
            "target_audio": "<base64 encoded audio or URL>",
            "source_is_url": true/false,
            "target_is_url": true/false
        }
    }
    
    Output format:
    {
        "audio_base64": "<base64 encoded converted audio>",
        "duration": <float>
    }
    """
    try:
        # Load models on first request
        load_models()
        
        input_data = event.get("input", {})
        
        source_audio = input_data.get("source_audio")
        target_audio = input_data.get("target_audio")
        source_is_url = input_data.get("source_is_url", False)
        target_is_url = input_data.get("target_is_url", False)
        
        if not source_audio or not target_audio:
            return {"error": "Missing source_audio or target_audio"}
        
        # Create temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "source.wav")
            target_path = os.path.join(tmpdir, "target.wav")
            output_path = os.path.join(tmpdir, "output.wav")
            
            # Download or decode source audio
            if source_is_url:
                response = requests.get(source_audio)
                with open(source_path, "wb") as f:
                    f.write(response.content)
            else:
                with open(source_path, "wb") as f:
                    f.write(base64.b64decode(source_audio))
            
            # Download or decode target audio
            if target_is_url:
                response = requests.get(target_audio)
                with open(target_path, "wb") as f:
                    f.write(response.content)
            else:
                with open(target_path, "wb") as f:
                    f.write(base64.b64decode(target_audio))
            
            # Convert voice
            print(f"Converting voice...")
            convert_voice(source_path, target_path, output_path)
            
            # Read output and encode to base64
            with open(output_path, "rb") as f:
                output_bytes = f.read()
            
            output_base64 = base64.b64encode(output_bytes).decode("utf-8")
            
            # Get duration
            audio, sr = torchaudio.load(output_path)
            duration = audio.shape[1] / sr
            
            return {
                "audio_base64": output_base64,
                "duration": duration
            }
            
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
