#!/bin/bash
# Create a clean package for GPU machine (no venv, cache, or generated files)

echo "=========================================="
echo "📦 Packaging Project for GPU Machine"
echo "=========================================="

cd /home/prof/Muneeb/project_files

# Create archive directory
ARCHIVE_NAME="arabic-speech-adapter-$(date +%Y%m%d)"
mkdir -p "/tmp/$ARCHIVE_NAME"

echo ""
echo "Copying essential files..."

# Copy source code
cp -r src "/tmp/$ARCHIVE_NAME/"
cp -r configs "/tmp/$ARCHIVE_NAME/"
cp -r scripts "/tmp/$ARCHIVE_NAME/"

# Copy root files
cp requirements.txt "/tmp/$ARCHIVE_NAME/"
cp .env "/tmp/$ARCHIVE_NAME/" 2>/dev/null || echo "No .env file"
cp *.py "/tmp/$ARCHIVE_NAME/" 2>/dev/null

# Copy documentation
cp README.md "/tmp/$ARCHIVE_NAME/" 2>/dev/null
cp GPU_MACHINE_SETUP.md "/tmp/$ARCHIVE_NAME/"
cp START_HERE.md "/tmp/$ARCHIVE_NAME/" 2>/dev/null
cp RUN_INSTRUCTIONS.md "/tmp/$ARCHIVE_NAME/" 2>/dev/null

# Create directories that will be populated
mkdir -p "/tmp/$ARCHIVE_NAME/data/raw"
mkdir -p "/tmp/$ARCHIVE_NAME/data/processed"
mkdir -p "/tmp/$ARCHIVE_NAME/checkpoints"
mkdir -p "/tmp/$ARCHIVE_NAME/experiments/logs"
mkdir -p "/tmp/$ARCHIVE_NAME/experiments/results"
mkdir -p "/tmp/$ARCHIVE_NAME/cache"

# Create archive
cd /tmp
tar -czf "$ARCHIVE_NAME.tar.gz" "$ARCHIVE_NAME"

# Move to home
mv "$ARCHIVE_NAME.tar.gz" ~/

echo ""
echo "✅ Package created: ~/$ARCHIVE_NAME.tar.gz"
echo ""
echo "Transfer to GPU machine:"
echo "  scp ~/$ARCHIVE_NAME.tar.gz user@gpu-machine:~/"
echo ""
echo "On GPU machine:"
echo "  tar -xzf $ARCHIVE_NAME.tar.gz"
echo "  cd $ARCHIVE_NAME"
echo "  bash scripts/check_ready.sh"
echo "  bash scripts/train_high_end_gpu.sh"
echo ""

# Show size
du -h ~/$ARCHIVE_NAME.tar.gz

# Cleanup
rm -rf "/tmp/$ARCHIVE_NAME"
