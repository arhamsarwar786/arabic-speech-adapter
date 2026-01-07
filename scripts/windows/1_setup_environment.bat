@echo off
REM ============================================================================
REM Windows Setup Script for Arabic Speech Adapter
REM 
REM This script sets up the Python environment on Windows
REM ============================================================================

echo ================================================
echo   Arabic Speech Adapter - Windows Setup
echo ================================================
echo.

REM Check Python
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.10 from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

python --version
echo.

REM Check if Python 3.10+
python -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Python 3.10+ recommended
    echo Current version may not be compatible
    pause
)

REM Create virtual environment
echo [2/5] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists
) else (
    python -m venv venv
    echo Virtual environment created
)
echo.

REM Activate virtual environment
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Upgrade pip
echo [4/5] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install PyTorch with CUDA
echo [5/5] Installing dependencies...
echo.
echo Installing PyTorch 2.5.1 with CUDA 12.1...
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
echo.

echo Installing other requirements...
pip install -r requirements.txt
echo.

REM Verify installation
echo ================================================
echo   Verifying Installation
echo ================================================
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
echo.

echo ================================================
echo   Setup Complete!
echo ================================================
echo.
echo Virtual environment created at: .\venv
echo.
echo To activate in future sessions:
echo   venv\Scripts\activate.bat
echo.
echo Next steps:
echo   1. Login to HuggingFace: huggingface-cli login
echo   2. Download datasets: python scripts\windows\2_download_datasets.bat
echo   3. Start training: python scripts\windows\4_train_standard.bat
echo.
pause
