@echo off
echo Testing OMR Processing...
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first.
    pause
    exit /b 1
)

venv\Scripts\python.exe test_single.py

pause
