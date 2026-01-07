#!/bin/bash
# Quick test with dummy data (no download needed)

echo "=========================================="
echo "QUICK TEST MODE"
echo "=========================================="

cd /home/prof/Muneeb/project_files
source venv/bin/activate

echo ""
echo "Testing model initialization..."
echo ""

python -c "
import torch
from src.models import ArabicSpeechPipeline

print('Initializing pipeline...')
pipeline = ArabicSpeechPipeline(
    artst_model='MBZUAI/artst_asr_v2',
    jais_model='inceptionai/jais-13b-chat',
    cache_dir='./cache',
    use_8bit_jais=True  # Use 8-bit for lower memory
)

print(f'\n✓ Pipeline initialized!')
print(f'Trainable parameters: {pipeline.get_num_trainable_params():,}')

# Test forward pass
print('\nTesting forward pass with dummy data...')
dummy_audio = torch.randn(2, 16000 * 3)  # 2 samples, 3 seconds each
labels = torch.randint(0, 32000, (2, 32))  # Dummy labels

outputs = pipeline(dummy_audio, labels=labels)
print(f'Loss: {outputs.loss.item():.4f}')

print('\n✓ Test complete! Ready for training.')
"

echo ""
echo "GPU Requirements:"
echo "  - Jais 13B (8-bit): ~13GB GPU memory"
echo "  - ArTST: ~1GB GPU memory"
echo "  - Adapter training: ~2-4GB GPU memory"
echo "  - Total minimum: ~16GB GPU memory"
echo ""
echo "Recommended: NVIDIA A100 (40GB) or V100 (32GB)"
echo ""
