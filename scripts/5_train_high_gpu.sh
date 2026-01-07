#!/bin/bash
# GPU-optimized training for 40-100GB GPUs

echo "=========================================="
echo "HIGH-END GPU TRAINING"
echo "=========================================="

cd /home/prof/Muneeb/project_files
source venv/bin/activate

echo ""
echo "🔍 Checking GPU configuration..."
echo ""

python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f'    Memory: {mem:.1f} GB')
print(f'\nPyTorch version: {torch.__version__}')
print(f'CUDA version: {torch.version.cuda}')
"

echo ""
echo "⚙️ Configuration:"
echo "  - Batch size: 16 (vs 8 on smaller GPUs)"
echo "  - Mixed precision: fp16 (2-3x speedup)"
echo "  - Full precision Jais (no 8-bit quantization)"
echo "  - Gradient accumulation: 2 steps"
echo "  - Effective batch size: 32"
echo ""
echo "💾 Expected GPU memory usage:"
echo "  - Jais 13B (fp16): ~26GB"
echo "  - ArTST (fp16): ~1GB"
echo "  - Adapter training: ~4-6GB"
echo "  - Activation memory: ~8GB"
echo "  - Total: ~40GB"
echo ""
echo "✅ Perfect for 40-100GB GPUs (V100, A100, H100)"
echo ""

read -p "Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Enable optimizations
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512

# Start training with wandb support
python src/training/train.py \
    --config configs/training_config.yaml \
    --output_dir checkpoints \
    --use_wandb

echo ""
echo "✓ Training complete!"
