@echo off
REM Quick build script for development testing
REM Builds the application without optimization for faster builds

echo Building VB OMR Scanner (Development Mode)...
echo.

REM Activate virtual environment
call venv311\Scripts\activate.bat

REM Build with PyInstaller (development mode - faster)
pyinstaller build.spec --noconfirm

if exist "dist\VB OMR Scanner.exe" (
    echo.
    echo Build successful\! Executable: dist\VB OMR Scanner.exe
    echo.
) else (
    echo.
    echo Build failed\! Check errors above.
    exit /b 1
)
