#!/bin/bash
# Interactive Docker shell for debugging

echo "=========================================="
echo "🐚 Starting Interactive Docker Shell"
echo "=========================================="

cd /home/prof/Muneeb/project_files

# Check for GPU
if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
    echo "✅ GPU available - starting with GPU support"
    GPU_FLAG="--gpus all"
else
    echo "⚠️  No GPU - starting without GPU"
    GPU_FLAG=""
fi

docker run --rm -it \
    $GPU_FLAG \
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
    bash

echo ""
echo "Exited Docker shell"
