#!/bin/bash
# Run training in Docker with GPU

echo "=========================================="
echo "🚀 Starting Training in Docker (GPU)"
echo "=========================================="

cd /home/prof/Muneeb/project_files

# Check if NVIDIA Docker runtime is available
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
    echo "❌ NVIDIA Docker runtime not available"
    echo ""
    echo "Install nvidia-container-toolkit:"
    echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
fi

echo ""
echo "GPU Status:"
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

echo ""
echo "Starting training container..."
echo ""

# Run with GPU support
docker run --rm -it \
    --gpus all \
    --shm-size=16g \
    -v $(pwd)/src:/workspace/src \
    -v $(pwd)/configs:/workspace/configs \
    -v $(pwd)/scripts:/workspace/scripts \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/cache:/workspace/cache \
    -v $(pwd)/checkpoints:/workspace/checkpoints \
    -v $(pwd)/experiments:/workspace/experiments \
    -v $(pwd)/.env:/workspace/.env \
    arabic-speech-adapter:latest \
    bash scripts/5_train_high_gpu.sh

echo ""
echo "✅ Training complete!"
