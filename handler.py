"""
RunPod Serverless Handler for SeedVC V2 Voice Conversion
Matches the Vast.ai always-on instance configuration
"""
import runpod
import os
import sys
import base64
import tempfile
import torch
import torchaudio
import requests
import yaml

# Global model variables - loaded once on cold start
vc_wrapper_v2 = None
device = None
dtype = torch.float16

def load_models():
    """Load SeedVC V2 models on cold start"""
    global vc_wrapper_v2, device, dtype
    
    if vc_wrapper_v2 is not None:
        return  # Already loaded
    
    print("Loading SeedVC V2 models...")
    
    # Add seed-vc to path
    sys.path.insert(0, "/workspace/seed-vc")
    os.chdir("/workspace/seed-vc")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load V2 model using hydra config (same as Vast.ai instance)
    from hydra.utils import instantiate
    from omegaconf import DictConfig
    
    cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
    vc_wrapper_v2 = instantiate(cfg)
    vc_wrapper_v2.load_checkpoints()
    vc_wrapper_v2.to(device)
    vc_wrapper_v2.eval()
    vc_wrapper_v2.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=device)
    
    print("SeedVC V2 models loaded successfully!")

def convert_voice(source_path, target_path, diffusion_steps=30, length_adjust=1.0, convert_style=True):
    """Convert voice using V2 model"""
    global vc_wrapper_v2, device, dtype
    
    os.chdir("/workspace/seed-vc")
    
    print(f"Converting: {source_path} -> {target_path}")
    
    # Use stream_output=True and collect the LAST yielded result
    full_audio = None
    for mp3_bytes, audio_result in vc_wrapper_v2.convert_voice_with_streaming(
        source_audio_path=source_path,
        target_audio_path=target_path,
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
        raise Exception("Conversion returned no audio")
    
    return full_audio

def handler(event):
    """
    RunPod handler for SeedVC V2 voice conversion
    
    Input format:
    {
        "input": {
            "source_audio": "<base64 encoded audio or URL>",
            "target_audio": "<base64 encoded audio or URL>",
            "source_is_url": true/false,
            "target_is_url": true/false,
            "diffusion_steps": 30,
            "length_adjust": 1.0,
            "convert_style": true
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
        diffusion_steps = input_data.get("diffusion_steps", 30)
        length_adjust = input_data.get("length_adjust", 1.0)
        convert_style = input_data.get("convert_style", True)
        
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
            
            # Convert voice using V2
            print(f"Converting voice with V2...")
            audio_result = convert_voice(
                source_path, 
                target_path, 
                diffusion_steps=diffusion_steps,
                length_adjust=length_adjust,
                convert_style=convert_style
            )
            
            # Save audio result to file
            # audio_result is a tuple (sample_rate, audio_array)
            sample_rate, audio_array = audio_result
            
            # Convert to tensor and save
            import numpy as np
            if isinstance(audio_array, np.ndarray):
                audio_tensor = torch.from_numpy(audio_array).float()
            else:
                audio_tensor = audio_array
            
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            torchaudio.save(output_path, audio_tensor, sample_rate)
            
            # Read output and encode to base64
            with open(output_path, "rb") as f:
                output_bytes = f.read()
            
            output_base64 = base64.b64encode(output_bytes).decode("utf-8")
            
            # Calculate duration
            duration = audio_tensor.shape[1] / sample_rate
            
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
