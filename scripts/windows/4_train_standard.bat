@echo off
REM ============================================================================
REM Training Script - Standard GPU (Windows)
REM For 24-40GB GPUs (RTX 3090, RTX 4090, A5000)
REM ============================================================================

echo ================================================
echo   Starting Training (Standard GPU)
echo ================================================
echo.

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found!
    pause
    exit /b 1
)

REM Create output directory
if not exist experiments mkdir experiments
if not exist experiments\logs mkdir experiments\logs

REM Check CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'GPU: {torch.cuda.get_device_name(0)}')"
if errorlevel 1 (
    echo ERROR: CUDA not available!
    echo Please check:
    echo   1. NVIDIA drivers installed
    echo   2. PyTorch CUDA version matches driver
    pause
    exit /b 1
)
echo.

REM Run training
echo Starting training...
echo Config: configs\training_config.yaml
echo Output: experiments\
echo.

python src\training\train.py ^
    --config configs\training_config.yaml ^
    --output_dir experiments ^
    --wandb

echo.
echo ================================================
echo   Training Session Ended
echo ================================================
pause
