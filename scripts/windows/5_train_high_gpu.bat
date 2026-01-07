@echo off
REM ============================================================================
REM Training Script - High-End GPU (Windows)
REM Optimized for 40-100GB GPUs (A100, H100, RTX 6000 Ada)
REM ============================================================================

echo ================================================
echo   Starting Training (High-End GPU)
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

REM Check CUDA and memory
python -c "import torch; assert torch.cuda.is_available(); print(f'GPU: {torch.cuda.get_device_name(0)}'); mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9; print(f'Memory: {mem_gb:.1f} GB'); assert mem_gb >= 24, 'GPU memory too low!'"
if errorlevel 1 (
    echo ERROR: GPU requirements not met!
    pause
    exit /b 1
)
echo.

echo Starting training with high-end optimizations...
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
