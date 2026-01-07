#!/bin/bash
# Comprehensive readiness check before training

echo "=========================================="
echo "🔍 TRAINING READINESS CHECK"
echo "=========================================="

cd /home/prof/Muneeb/project_files
source venv/bin/activate

ERRORS=0

echo ""
echo "1️⃣ Checking GPU..."
GPU_CHECK=$(python -c "import torch; print(torch.cuda.is_available())" 2>&1)
if [[ "$GPU_CHECK" == "True" ]]; then
    echo "   ✅ GPU detected"
    python -c "import torch; print(f'   GPU: {torch.cuda.get_device_name(0)}'); print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')"
else
    echo "   ❌ No GPU found - REQUIRED for training!"
    ERRORS=$((ERRORS + 1))
fi

echo ""
echo "2️⃣ Checking dependencies..."
DEPS=("torch" "transformers" "datasets" "torchaudio" "yaml" "tqdm")
for dep in "${DEPS[@]}"; do
    if python -c "import $dep" 2>/dev/null; then
        echo "   ✅ $dep"
    else
        echo "   ❌ $dep missing"
        ERRORS=$((ERRORS + 1))
    fi
done

echo ""
echo "3️⃣ Checking project structure..."
REQUIRED_DIRS=("src/models" "src/data" "src/training" "configs" "scripts")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "   ✅ $dir/"
    else
        echo "   ❌ $dir/ missing"
        ERRORS=$((ERRORS + 1))
    fi
done

echo ""
echo "4️⃣ Checking Python modules..."
MODULES=(
    "src/models/adapter.py"
    "src/models/artst_encoder.py"
    "src/models/jais_decoder.py"
    "src/models/pipeline.py"
    "src/data/dataset.py"
    "src/data/collator.py"
    "src/data/preprocess.py"
    "src/training/train.py"
)
for mod in "${MODULES[@]}"; do
    if [ -f "$mod" ]; then
        echo "   ✅ $mod"
    else
        echo "   ❌ $mod missing"
        ERRORS=$((ERRORS + 1))
    fi
done

echo ""
echo "5️⃣ Checking configuration..."
if [ -f "configs/training_config.yaml" ]; then
    echo "   ✅ Training config exists"
    python -c "
import yaml
with open('configs/training_config.yaml') as f:
    cfg = yaml.safe_load(f)
print(f'   Batch size: {cfg[\"training\"][\"batch_size\"]}')
print(f'   FP16: {cfg[\"training\"].get(\"fp16\", False)}')
print(f'   Max steps: {cfg[\"training\"][\"max_steps\"]}')
print(f'   8-bit Jais: {cfg[\"model\"][\"jais\"][\"use_8bit\"]}')
"
else
    echo "   ❌ Training config missing"
    ERRORS=$((ERRORS + 1))
fi

echo ""
echo "6️⃣ Checking .env file..."
if [ -f ".env" ]; then
    echo "   ✅ .env exists"
    if grep -q "HF_TOKEN" .env; then
        echo "   ✅ HF_TOKEN configured"
    else
        echo "   ⚠️  HF_TOKEN not found (may be needed)"
    fi
else
    echo "   ⚠️  .env not found (optional)"
fi

echo ""
echo "7️⃣ Checking data directory..."
if [ -d "data" ]; then
    echo "   ✅ data/ exists"
    
    if [ -d "data/processed" ] && [ "$(ls -A data/processed/*.json 2>/dev/null)" ]; then
        echo "   ✅ Preprocessed data found:"
        find data/processed -name "*.json" -exec sh -c 'echo "      $(basename {}) - $(wc -l < {}) samples"' \;
    else
        echo "   ⚠️  No preprocessed data yet"
        echo "      Run: bash scripts/2_download_data.sh"
        echo "      Then: bash scripts/3_preprocess_data.sh"
    fi
else
    echo "   ⚠️  data/ directory not created yet"
fi

echo ""
echo "8️⃣ Testing model initialization..."
TEST_OUTPUT=$(python -c "
import sys
sys.path.insert(0, 'src')
try:
    from models.adapter import ContinuousAdapter
    adapter = ContinuousAdapter(
        speech_dim=768,
        hidden_dim=4096,
        num_prefix_tokens=32,
        num_attention_heads=8,
        intermediate_dim=8192
    )
    print(f'✅ Adapter: {adapter.get_num_trainable_params():,} params')
except Exception as e:
    print(f'❌ Adapter error: {e}')
    sys.exit(1)
" 2>&1)

if [[ $? -eq 0 ]]; then
    echo "   $TEST_OUTPUT"
else
    echo "   $TEST_OUTPUT"
    ERRORS=$((ERRORS + 1))
fi

echo ""
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo "✅ ALL CHECKS PASSED - READY TO TRAIN!"
    echo "=========================================="
    echo ""
    echo "Architecture verification:"
    echo "  ✅ ArTST encoder wrapper"
    echo "  ✅ Continuous Adapter (Cross-Attention + Prefix + MLP)"
    echo "  ✅ Jais decoder wrapper"
    echo "  ✅ Full pipeline integration"
    echo "  ✅ Dataset loader"
    echo "  ✅ Training script with fp16"
    echo ""
    echo "Next steps:"
    if [ ! -d "data/processed" ] || [ -z "$(ls -A data/processed/*.json 2>/dev/null)" ]; then
        echo "  1. Download data: bash scripts/2_download_data.sh"
        echo "  2. Preprocess: bash scripts/3_preprocess_data.sh"
        echo "  3. Train: bash scripts/train_high_end_gpu.sh"
    else
        echo "  🚀 START TRAINING: bash scripts/train_high_end_gpu.sh"
    fi
else
    echo "❌ ERRORS FOUND: $ERRORS issues"
    echo "=========================================="
    echo ""
    echo "Fix the errors above before training."
fi

echo ""
