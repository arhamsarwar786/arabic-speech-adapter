#!/bin/bash
# Complete pipeline: setup -> download -> preprocess -> train

set -e  # Exit on error

echo "=========================================="
echo "ARABIC SPEECH ADAPTER - FULL PIPELINE"
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Setup environment"
echo "  2. Download datasets (~20GB, 2-4 hours)"
echo "  3. Preprocess data (1-2 hours)"
echo "  4. Train adapter (days/weeks depending on GPU)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

cd /home/prof/Muneeb/project_files

# Step 1: Setup
bash scripts/1_setup_environment.sh
if [ $? -ne 0 ]; then
    echo "✗ Setup failed"
    exit 1
fi

# Step 2: Download
bash scripts/2_download_datasets.sh
if [ $? -ne 0 ]; then
    echo "✗ Download failed"
    exit 1
fi

# Step 3: Preprocess
bash scripts/3_preprocess_data.sh
if [ $? -ne 0 ]; then
    echo "✗ Preprocess failed"
    exit 1
fi

# Step 4: Train
bash scripts/4_train_standard.sh

echo ""
echo "=========================================="
echo "✓ PIPELINE COMPLETE!"
echo "=========================================="
