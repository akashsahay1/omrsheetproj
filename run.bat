@echo off
echo Starting OMR Processor...
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first.
    pause
    exit /b 1
)

REM Run the application
venv\Scripts\python.exe src\main.py

if errorlevel 1 (
    echo.
    echo Application exited with error.
    pause
)
