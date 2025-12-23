@echo off
echo ===============================================
echo  OMR Processor - Setup Script
echo  Vidya Bharti Pratibha Khoj 2025-26
echo ===============================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check if venv exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        echo Make sure Python 3.8+ is installed and in PATH
        pause
        exit /b 1
    )
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

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
echo   2. Run: venv\Scripts\python src\main.py
echo.
pause
