#!/bin/bash
# Save Docker image for transfer to GPU machine

echo "=========================================="
echo "💾 Saving Docker Image"
echo "=========================================="

cd /home/prof/Muneeb/project_files

IMAGE_NAME="arabic-speech-adapter"
OUTPUT_FILE="arabic-speech-adapter-docker.tar.gz"

echo "Saving Docker image to $OUTPUT_FILE..."
docker save $IMAGE_NAME:latest | gzip > ~/$OUTPUT_FILE

echo ""
echo "✅ Image saved: ~/$OUTPUT_FILE"
du -h ~/$OUTPUT_FILE

echo ""
echo "Transfer to GPU machine:"
echo "  scp ~/$OUTPUT_FILE user@gpu-machine:~/"
echo ""
echo "On GPU machine:"
echo "  docker load < $OUTPUT_FILE"
echo "  cd project/"
echo "  bash scripts/docker_run_gpu.sh"
echo ""
