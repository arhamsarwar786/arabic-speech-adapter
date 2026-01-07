# Windows Scripts for Arabic Speech Adapter

These batch scripts (.bat) are Windows equivalents of the Linux bash scripts.

## 📋 Prerequisites

1. **Python 3.10+**: https://www.python.org/downloads/
   - ✅ Check "Add Python to PATH" during installation
   
2. **NVIDIA GPU Drivers**: https://www.nvidia.com/Download/index.aspx
   - For CUDA 12.1 support

3. **Git for Windows** (optional): https://git-scm.com/download/win

## 🚀 Quick Start

### Step 1: Setup Environment
```cmd
cd C:\path\to\project
scripts\windows\1_setup_environment.bat
```

### Step 2: Login to HuggingFace
```cmd
venv\Scripts\activate
huggingface-cli login
```
Paste your token from: https://huggingface.co/settings/tokens

### Step 3: Download Datasets (Optional for local testing)
```cmd
scripts\windows\2_download_datasets.bat
```

### Step 4: Preprocess Data
```cmd
scripts\windows\3_preprocess_data.bat
```

### Step 5: Train
```cmd
REM For standard GPUs (24-40GB)
scripts\windows\4_train_standard.bat

REM OR for high-end GPUs (40-100GB)
scripts\windows\5_train_high_gpu.bat
```

### Step 6: Interactive Testing
```cmd
REM After training completes
scripts\windows\6_launch_gradio.bat experiments\best_model\adapter_checkpoint.pt

REM With public share link
scripts\windows\6_launch_gradio.bat experiments\best_model\adapter_checkpoint.pt --share

REM With authentication
scripts\windows\6_launch_gradio.bat experiments\best_model\adapter_checkpoint.pt --auth admin password123
```

## 🔧 Manual Commands

If scripts don't work, use these manual commands:

```cmd
REM Activate environment
venv\Scripts\activate

REM Install PyTorch with CUDA
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

REM Install requirements
pip install -r requirements.txt

REM Run training
python src\training\train.py --config configs\training_config.yaml --output_dir experiments

REM Launch Gradio
python src\inference\gradio_app.py --checkpoint experiments\best_model\adapter_checkpoint.pt --port 7860
```

## ⚠️ Common Issues

### PyTorch not detecting CUDA
```cmd
REM Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

REM If False, reinstall PyTorch:
pip uninstall torch torchvision torchaudio
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

### Script won't run (Permission denied)
- Right-click script → Properties → Unblock
- Or run as Administrator

### Python not found
- Reinstall Python and check "Add to PATH"
- Or use full path: `C:\Python310\python.exe`

## 🐧 Alternative: Use WSL2

For best Linux compatibility on Windows:

```powershell
# Install WSL2
wsl --install -d Ubuntu-22.04

# Inside WSL, use Linux scripts
cd /mnt/c/Users/YourName/project
bash scripts/1_setup_environment.sh
```

## 📖 Script Details

| Script | Purpose | GPU Required |
|--------|---------|--------------|
| 1_setup_environment.bat | Install Python packages | No |
| 2_download_datasets.bat | Download training data | No |
| 3_preprocess_data.bat | Prepare datasets | No |
| 4_train_standard.bat | Train (24-40GB GPU) | Yes |
| 5_train_high_gpu.bat | Train (40-100GB GPU) | Yes |
| 6_launch_gradio.bat | Interactive demo | Optional |

## 🔗 Resources

- Main README: [../../README.md](../../README.md)
- Architecture Docs: [../../ARCHITECTURE.md](../../ARCHITECTURE.md)
- GitHub Repo: https://github.com/arhamsarwar786/arabic-speech-adapter
