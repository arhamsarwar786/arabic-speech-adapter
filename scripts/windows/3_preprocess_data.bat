@echo off
REM ============================================================================
REM Preprocess Datasets (Windows)
REM ============================================================================

echo ================================================
echo   Preprocessing Datasets
echo ================================================
echo.

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found!
    pause
    exit /b 1
)

echo Running preprocessing...
python src\data\preprocess.py --data_root data --output_root data\processed
echo.

echo ================================================
echo   Preprocessing Complete!
echo ================================================
pause
