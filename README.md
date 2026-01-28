# SeedVC RunPod Serverless

RunPod serverless worker for SeedVC voice conversion.

## Deployment

1. Connect this GitHub repo to RunPod
2. RunPod will auto-build the Docker image
3. Create a serverless endpoint

## API Usage

### Input
```json
{
  "input": {
    "source_audio": "<base64 encoded audio or URL>",
    "target_audio": "<base64 encoded audio or URL>",
    "source_is_url": true,
    "target_is_url": true
  }
}
```

### Output
```json
{
  "audio_base64": "<base64 encoded converted audio>",
  "duration": 5.2
}
```

## GPU Requirements
- Minimum 12GB VRAM recommended
- CUDA 12.1 compatible
