#!/bin/bash

##############################################################################
# Launch Gradio Interactive Demo
# 
# This script launches a Gradio web interface for interactive inference
# with microphone input, file upload, and real-time transcription.
#
# Usage:
#   bash scripts/6_launch_gradio.sh [checkpoint_path] [options]
#
# Examples:
#   # Local access only
#   bash scripts/6_launch_gradio.sh experiments/best_model/adapter_checkpoint.pt
#
#   # Public share link (for remote servers)
#   bash scripts/6_launch_gradio.sh experiments/best_model/adapter_checkpoint.pt --share
#
#   # Custom port with authentication
#   bash scripts/6_launch_gradio.sh experiments/best_model/adapter_checkpoint.pt --port 8080 --auth admin secretpass
##############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CHECKPOINT_PATH="${1:-experiments/best_model/adapter_checkpoint.pt}"
CONFIG_PATH="configs/training_config.yaml"
PORT=7860
SHARE_FLAG=""
AUTH_FLAGS=""

# Parse additional arguments
shift 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --share)
            SHARE_FLAG="--share"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --auth)
            AUTH_FLAGS="--auth $2 $3"
            shift 3
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  🎤 Gradio Interactive Demo Launcher${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo -e "${YELLOW}⚠️  Warning: Checkpoint not found: ${CHECKPOINT_PATH}${NC}"
    echo -e "${YELLOW}The app will launch with an untrained adapter (for testing only)${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Aborted.${NC}"
        exit 1
    fi
fi

# Check if config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo -e "${RED}❌ Error: Config not found: ${CONFIG_PATH}${NC}"
    exit 1
fi

# Activate virtual environment
if [ -d "venv" ]; then
    echo -e "${GREEN}📦 Activating virtual environment...${NC}"
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo -e "${GREEN}📦 Activating virtual environment...${NC}"
    source .venv/bin/activate
else
    echo -e "${YELLOW}⚠️  No virtual environment found. Using system Python.${NC}"
fi

# Check if Gradio is installed
echo -e "${GREEN}🔍 Checking dependencies...${NC}"
if ! python -c "import gradio" 2>/dev/null; then
    echo -e "${YELLOW}📦 Installing Gradio...${NC}"
    pip install gradio
fi

# Check if librosa is installed (for audio resampling)
if ! python -c "import librosa" 2>/dev/null; then
    echo -e "${YELLOW}📦 Installing librosa...${NC}"
    pip install librosa
fi

# Print configuration
echo ""
echo -e "${GREEN}⚙️  Configuration:${NC}"
echo -e "  Checkpoint: ${CHECKPOINT_PATH}"
echo -e "  Config: ${CONFIG_PATH}"
echo -e "  Port: ${PORT}"
if [ -n "$SHARE_FLAG" ]; then
    echo -e "  Share: ${GREEN}✓ Enabled (public link)${NC}"
else
    echo -e "  Share: Local only"
fi
if [ -n "$AUTH_FLAGS" ]; then
    echo -e "  Auth: ${GREEN}✓ Enabled${NC}"
fi
echo ""

# Check GPU availability
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
    echo -e "${GREEN}🎮 GPU: ${GPU_NAME}${NC}"
else
    echo -e "${YELLOW}⚠️  No GPU detected. Using CPU (slower inference).${NC}"
fi

echo ""
echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}🚀 Launching Gradio App...${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo -e "${YELLOW}📝 Access the app at:${NC}"
echo -e "  ${GREEN}http://localhost:${PORT}${NC}"
if [ -n "$SHARE_FLAG" ]; then
    echo -e "  ${GREEN}Public link will be generated...${NC}"
fi
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Launch Gradio app
python src/inference/gradio_app.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --config "$CONFIG_PATH" \
    --port "$PORT" \
    $SHARE_FLAG \
    $AUTH_FLAGS

echo ""
echo -e "${GREEN}✅ App closed successfully!${NC}"
