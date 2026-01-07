#!/bin/bash
# Build Docker image

echo "=========================================="
echo "🐳 Building Docker Image"
echo "=========================================="

cd /home/prof/Muneeb/project_files

# Build image
docker build -t arabic-speech-adapter:latest .

echo ""
echo "✅ Docker image built successfully"
echo ""
echo "Image size:"
docker images arabic-speech-adapter:latest

echo ""
echo "Next steps (WITH DOCKER):"
echo "  Test locally:    bash scripts/docker_2_test_local.sh"
echo "  Train with GPU:  bash scripts/docker_3_run_training.sh"
echo "  Debug shell:     bash scripts/docker_4_interactive_shell.sh"
echo "  Save/transfer:   bash scripts/docker_5_save_image.sh"
echo ""
