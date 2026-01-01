@echo off
REM Release build script - Creates MSI installer
REM This script builds optimized executable and creates MSI package

echo ============================================
echo VB OMR Scanner - Release Build + MSI
echo ============================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Clean previous builds
echo Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
echo.

REM Build with PyInstaller (optimized)
echo Building executable (this may take 5-10 minutes)...
pyinstaller --clean build.spec

if not exist "dist\VB OMR Scanner.exe" (
    echo ERROR: Build failed\!
    exit /b 1
)

echo.
echo Build successful\!
echo.

REM Create MSI using create_msi.py script
echo Creating MSI installer...
python create_msi.py

if exist "dist\VBOMRScanner-1.0.0-win64.msi" (
    echo.
    echo ============================================
    echo SUCCESS\! Installer created:
    echo dist\VBOMRScanner-1.0.0-win64.msi
    echo ============================================
    echo.
) else (
    echo.
    echo WARNING: MSI creation may have failed.
    echo Check the output above for errors.
    echo.
    echo Executable is available at: dist\VB OMR Scanner.exe
    echo.
)

pause
