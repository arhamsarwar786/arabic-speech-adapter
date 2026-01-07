@echo off
REM ============================================================================
REM Launch Gradio Interactive Demo (Windows)
REM ============================================================================

setlocal enabledelayedexpansion

REM Default values
set CHECKPOINT=%1
if "%CHECKPOINT%"=="" set CHECKPOINT=experiments\best_model\adapter_checkpoint.pt

set CONFIG=configs\training_config.yaml
set PORT=7860
set SHARE=
set AUTH=

REM Parse arguments
:parse_args
if "%2"=="" goto end_parse
if "%2"=="--share" (
    set SHARE=--share
    shift
    goto parse_args
)
if "%2"=="--port" (
    set PORT=%3
    shift
    shift
    goto parse_args
)
if "%2"=="--auth" (
    set AUTH=--auth %3 %4
    shift
    shift
    shift
    goto parse_args
)
shift
goto parse_args
:end_parse

echo ================================================
echo   Gradio Interactive Demo Launcher
echo ================================================
echo.

REM Check checkpoint
if not exist "%CHECKPOINT%" (
    echo WARNING: Checkpoint not found: %CHECKPOINT%
    echo The app will launch with an untrained adapter
    pause
)

REM Activate venv
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found!
    pause
    exit /b 1
)

REM Check dependencies
python -c "import gradio" >nul 2>&1
if errorlevel 1 (
    echo Installing Gradio...
    pip install gradio
)

python -c "import librosa" >nul 2>&1
if errorlevel 1 (
    echo Installing librosa...
    pip install librosa
)

echo.
echo Configuration:
echo   Checkpoint: %CHECKPOINT%
echo   Config: %CONFIG%
echo   Port: %PORT%
if not "%SHARE%"=="" echo   Share: Enabled (public link)
echo.

REM Check GPU
python -c "import torch; print('GPU: ' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only (slower)'))"
echo.

echo ================================================
echo   Launching Gradio App...
echo ================================================
echo.
echo Access at: http://localhost:%PORT%
echo Press Ctrl+C to stop
echo.

python src\inference\gradio_app.py ^
    --checkpoint "%CHECKPOINT%" ^
    --config "%CONFIG%" ^
    --port %PORT% ^
    %SHARE% ^
    %AUTH%

echo.
echo App closed.
pause
