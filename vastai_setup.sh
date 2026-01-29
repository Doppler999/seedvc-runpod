#!/bin/bash
# Vast.ai Setup Script for SeedVC V1+V2 Server
# Run this on a fresh Vast.ai instance with GPU

set -e

echo "🚀 Setting up SeedVC V1+V2 Server..."

# Update system
apt-get update
apt-get install -y git ffmpeg libsndfile1 curl

# Clone seed-vc if not exists
if [ ! -d "/workspace/seed-vc" ]; then
    echo "📦 Cloning seed-vc repository..."
    git clone https://github.com/Plachtaa/seed-vc.git /workspace/seed-vc
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install fastapi uvicorn python-multipart torchaudio numpy scipy librosa soundfile transformers accelerate einops safetensors

# Install seed-vc requirements
cd /workspace/seed-vc
pip install -r requirements.txt || true

# Download the server.py from GitHub or copy it
echo "📥 Downloading server.py..."
curl -o /workspace/server.py https://raw.githubusercontent.com/Doppler999/seedvc-runpod/main/server.py

# Create startup script
cat > /workspace/start_server.sh << 'EOF'
#!/bin/bash
cd /workspace
export PYTHONPATH="/workspace/seed-vc:$PYTHONPATH"
python server.py
EOF
chmod +x /workspace/start_server.sh

echo "✅ Setup complete!"
echo ""
echo "To start the server, run:"
echo "  /workspace/start_server.sh"
echo ""
echo "Or manually:"
echo "  cd /workspace && PYTHONPATH=/workspace/seed-vc python server.py"
echo ""
echo "The server will be available at http://<your-vast-ip>:8000"
echo "  - Health check: GET /health"
echo "  - Convert: POST /convert"
