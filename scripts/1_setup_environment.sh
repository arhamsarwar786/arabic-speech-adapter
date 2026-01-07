#!/bin/bash
# Setup script for Arabic Speech Adapter training environment (WITHOUT DOCKER)

set -e  # Exit on error

echo "=========================================="
echo "Setting up Arabic Speech Adapter Environment"
echo "=========================================="

cd /home/prof/Muneeb/project_files

# Function to check Python 3.10
check_python_310() {
    if command -v python3.10 &> /dev/null; then
        echo "✅ Python 3.10 found: $(python3.10 --version)"
        return 0
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        if [ "$PYTHON_VERSION" = "3.10" ]; then
            echo "✅ Python 3.10 found: $(python --version)"
            return 0
        fi
    fi
    return 1
}

# Check if Python 3.10 exists
if check_python_310; then
    echo "Python 3.10 is available, proceeding..."
else
    echo "⚠️  Python 3.10 not found!"
    echo ""
    echo "Installing Python 3.10..."
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Detected Linux, installing via apt..."
        sudo apt-get update
        sudo apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip
        
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Detected macOS, installing via brew..."
        if ! command -v brew &> /dev/null; then
            echo "Error: Homebrew not found. Please install from https://brew.sh"
            exit 1
        fi
        brew install python@3.10
        
    else
        echo "❌ Unsupported OS: $OSTYPE"
        echo "Please install Python 3.10 manually:"
        echo "  Linux: sudo apt-get install python3.10"
        echo "  macOS: brew install python@3.10"
        exit 1
    fi
    
    # Verify installation
    if check_python_310; then
        echo "✅ Python 3.10 installed successfully"
    else
        echo "❌ Failed to install Python 3.10"
        exit 1
    fi
fi

# Set Python command (prefer python3.10, fallback to python)
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
else
    PYTHON_CMD="python"
fi

PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Using Python $PYTHON_VERSION with command: $PYTHON_CMD"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# Activate venv
source venv/bin/activate

echo "Installing PyTorch (CPU+CUDA)..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

echo "Installing requirements..."
pip install -r requirements.txt

echo "Installing additional training dependencies..."
pip install wandb accelerate bitsandbytes

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps (WITHOUT DOCKER):"
echo "  1. Download datasets: bash scripts/2_download_datasets.sh"
echo "  2. Preprocess data: bash scripts/3_preprocess_data.sh"
echo "  3. Start training: bash scripts/4_train_standard.sh"
echo "     Or high-end GPU: bash scripts/5_train_high_gpu.sh"
echo ""
echo "Or run everything: bash scripts/run_complete_pipeline.sh"
echo ""
echo "To use Docker instead, see: scripts/docker_*.sh"
echo ""
echo "GPU Check:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""
