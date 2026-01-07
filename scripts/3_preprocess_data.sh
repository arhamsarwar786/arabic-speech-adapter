#!/bin/bash
# Preprocess datasets into training format

echo "=========================================="
echo "STEP 3: Preprocess Datasets"
echo "=========================================="

cd /home/prof/Muneeb/project_files
source venv/bin/activate

echo ""
echo "Preprocessing datasets..."
echo "This will take 1-2 hours"
echo ""

# Preprocess all datasets
python src/data/preprocess.py \
    --data_root data \
    --datasets commonvoice qasr covost2 \
    --splits train validation test

echo ""
echo "✓ Preprocessing complete!"
echo ""
echo "Dataset summary:"
find data/processed -name "*.json" -exec wc -l {} \; | sort
echo ""
