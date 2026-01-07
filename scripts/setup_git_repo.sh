#!/bin/bash
# Initialize git repository with proper structure

echo "=========================================="
echo "🔧 Git Repository Setup"
echo "=========================================="

cd /home/prof/Muneeb/project_files

# Initialize git if not already
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    echo "✓ Git initialized"
else
    echo "✓ Git repository already exists"
fi

# Create necessary directories that should be tracked but empty
mkdir -p experiments/logs
mkdir -p experiments/results
touch experiments/logs/.gitkeep
touch experiments/results/.gitkeep

echo ""
echo "Adding files to git..."
git add .gitignore
git add src/
git add configs/
git add scripts/
git add requirements.txt
git add *.md
git add *.py 2>/dev/null || true
git add experiments/logs/.gitkeep
git add experiments/results/.gitkeep

echo ""
echo "📊 Git status:"
git status --short

echo ""
echo "To commit and push:"
echo "  git commit -m 'Initial commit: Arabic speech adapter with Continuous Adapter architecture'"
echo "  git remote add origin <your-repo-url>"
echo "  git push -u origin main"
echo ""
echo "On GPU machine:"
echo "  git clone <your-repo-url>"
echo "  cd <repo-name>"
echo "  bash scripts/1_setup.sh"
echo "  bash scripts/train_high_end_gpu.sh"
echo ""
