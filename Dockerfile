# Use NVIDIA CUDA base image with Python 3.10
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /workspace

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 support
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
RUN pip install -r requirements.txt

# Install additional training dependencies
RUN pip install wandb accelerate bitsandbytes

# Copy project files
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY *.py ./

# Create necessary directories
RUN mkdir -p data/raw data/processed cache checkpoints experiments/logs experiments/results

# Set environment variables
ENV TRANSFORMERS_CACHE=/workspace/cache
ENV HF_HOME=/workspace/cache

# Default command
CMD ["bash"]
