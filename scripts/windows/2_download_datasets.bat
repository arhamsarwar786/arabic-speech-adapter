@echo off
REM ============================================================================
REM Download Arabic Datasets (Windows)
REM ============================================================================

echo ================================================
echo   Downloading Arabic Datasets
echo ================================================
echo.

REM Activate venv
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found!
    echo Please run 1_setup_environment.bat first
    pause
    exit /b 1
)

REM Create data directory
if not exist data mkdir data

REM Download datasets
echo [1/4] Downloading Common Voice Arabic...
python -c "from datasets import load_dataset; ds = load_dataset('mozilla-foundation/common_voice_17_0', 'ar', cache_dir='./data/commonvoice'); print('Downloaded Common Voice')"
echo.

echo [2/4] Downloading QASR...
python -c "from datasets import load_dataset; ds = load_dataset('QCRI/QASR', cache_dir='./data/qasr'); print('Downloaded QASR')"
echo.

echo [3/4] Downloading mTEDx...
python -c "from datasets import load_dataset; ds = load_dataset('facebook/multilingual_librispeech', 'arabic', cache_dir='./data/mtedx'); print('Downloaded mTEDx')"
echo.

echo [4/4] Downloading CoVoST2...
python -c "from datasets import load_dataset; ds = load_dataset('facebook/covost2', 'ar_en', cache_dir='./data/covost2'); print('Downloaded CoVoST2')"
echo.

echo ================================================
echo   Download Complete!
echo ================================================
echo.
echo Datasets saved to: .\data\
echo Next step: python src\data\preprocess.py
echo.
pause
