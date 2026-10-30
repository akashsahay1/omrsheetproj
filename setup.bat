@echo off
echo ===============================================
echo  OMR Processor - Setup Script
echo  Vidya Bharti Pratibha Khoj 2025-26
echo ===============================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check if venv exists
if not exist "venv311" (
    echo Creating virtual environment...
    REM Prefer Python 3.11 via the py launcher; fall back to python on PATH
    py -3.11 -m venv venv311 2>nul || python -m venv venv311
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        echo Make sure Python 3.8+ is installed and in PATH
        pause
        exit /b 1
    )
)

echo Activating virtual environment...
call venv311\Scripts\activate.bat

echo.
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ===============================================
echo  Setup completed successfully!
echo ===============================================
echo.
echo To run the application:
echo   1. Double-click run.bat
echo   OR
echo   2. Run: venv311\Scripts\python src\main.py
echo.
pause
