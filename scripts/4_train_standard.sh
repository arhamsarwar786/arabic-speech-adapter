#!/bin/bash
# Start training the Continuous Adapter

echo "=========================================="
echo "STEP 4: Train Adapter"
echo "=========================================="

cd /home/prof/Muneeb/project_files
source venv/bin/activate

# Check GPU
echo "GPU Status:"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
echo ""

# Parse arguments
USE_WANDB=false
RESUME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --wandb)
            USE_WANDB=true
            shift
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Build command
CMD="python src/training/train.py \
    --config configs/training_config.yaml \
    --output_dir checkpoints"

if [ "$USE_WANDB" = true ]; then
    CMD="$CMD --use_wandb"
    echo "Weights & Biases logging enabled"
fi

echo ""
echo "Starting training..."
echo "Monitor progress in: experiments/logs/"
echo "Checkpoints saved to: checkpoints/"
echo ""
echo "Press Ctrl+C to stop training"
echo ""

# Run training
$CMD

echo ""
echo "✓ Training complete!"
echo ""
