#!/bin/bash
# Download all Arabic datasets

echo "=========================================="
echo "STEP 2: Download Datasets"
echo "=========================================="

cd /home/prof/Muneeb/project_files
source venv/bin/activate

# Check if data directory exists
mkdir -p data/raw data/processed

echo ""
echo "Starting dataset downloads..."
echo "This will take 2-4 hours and download ~20GB"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Download cancelled"
    exit 1
fi

python download_arabic_datasets.py

echo ""
echo "✓ Dataset download complete!"
echo ""
