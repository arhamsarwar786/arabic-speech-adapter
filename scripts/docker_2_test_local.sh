#!/bin/bash
# Test Docker container locally (no GPU needed)

echo "=========================================="
echo "🧪 Testing Docker Container Locally"
echo "=========================================="

cd /home/prof/Muneeb/project_files

echo ""
echo "1. Testing Python version..."
docker run --rm arabic-speech-adapter:latest python --version

echo ""
echo "2. Testing PyTorch installation..."
docker run --rm arabic-speech-adapter:latest python -c "import torch; print(f'PyTorch: {torch.__version__}')"

echo ""
echo "3. Testing project imports..."
docker run --rm \
    -v $(pwd)/src:/workspace/src \
    arabic-speech-adapter:latest \
    python -c "
import sys
sys.path.insert(0, '/workspace')
from src.models.adapter import ContinuousAdapter
print('✅ ContinuousAdapter imported')

adapter = ContinuousAdapter(
    speech_dim=768,
    hidden_dim=4096,
    num_prefix_tokens=32,
    num_attention_heads=8,
    intermediate_dim=8192
)
print(f'✅ Adapter created: {adapter.get_num_trainable_params():,} params')
"

echo ""
echo "✅ All tests passed!"
echo ""
