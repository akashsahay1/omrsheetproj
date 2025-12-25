@echo off
setlocal enabledelayedexpansion
echo Building VB OMR Scanner Installer...
echo.

REM Try common Inno Setup installation paths
set "ISCC_PATH="

REM Check each path individually to avoid parentheses parsing issues
if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" set "ISCC_PATH=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if "!ISCC_PATH!"=="" if exist "C:\Program Files\Inno Setup 6\ISCC.exe" set "ISCC_PATH=C:\Program Files\Inno Setup 6\ISCC.exe"
if "!ISCC_PATH!"=="" if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC_PATH=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"

REM Try to find it in PATH if not found yet
if "!ISCC_PATH!"=="" (
    where iscc >nul 2>&1
    if !errorlevel! equ 0 set "ISCC_PATH=iscc"
)

if "!ISCC_PATH!"=="" (
    echo ERROR: Inno Setup not found!
    echo.
    echo Please install Inno Setup from: https://jrsoftware.org/isdl.php
    echo Or run: winget install JRSoftware.InnoSetup
    echo.
    echo After installation, restart this terminal and run this script again.
    pause
    exit /b 1
)

echo Found Inno Setup at: !ISCC_PATH!
echo.

REM Check if exe exists
if not exist "dist\VB OMR Scanner.exe" (
    echo ERROR: VB OMR Scanner.exe not found in dist folder!
    echo Please run PyInstaller build first.
    pause
    exit /b 1
)

echo Compiling installer...
"!ISCC_PATH!" installer.iss

if !errorlevel! equ 0 (
    echo.
    echo ========================================
    echo Installer created successfully!
    echo Output: dist\VB_OMR_Scanner_Setup.exe
    echo ========================================
) else (
    echo.
    echo ERROR: Failed to create installer.
)

pause
