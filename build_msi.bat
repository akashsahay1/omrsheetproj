@echo off
setlocal enabledelayedexpansion
echo Building VB OMR Scanner MSI Installer...
echo.

REM Set WiX path
set "WIX_PATH=C:\Program Files (x86)\WiX Toolset v3.14\bin"

REM Check if WiX exists
if not exist "%WIX_PATH%\candle.exe" (
    echo ERROR: WiX Toolset not found at %WIX_PATH%
    echo Please install WiX Toolset from: https://wixtoolset.org/
    pause
    exit /b 1
)

REM Check if exe exists
if not exist "dist\VB OMR Scanner.exe" (
    echo ERROR: VB OMR Scanner.exe not found in dist folder!
    echo Please run PyInstaller build first.
    pause
    exit /b 1
)

echo Found WiX Toolset at: %WIX_PATH%
echo.

REM Create output directory if needed
if not exist "dist" mkdir dist

echo Step 1: Compiling WiX source...
"%WIX_PATH%\candle.exe" -nologo -out "dist\installer.wixobj" "installer.wxs" -ext WixUIExtension

if !errorlevel! neq 0 (
    echo.
    echo ERROR: WiX compilation failed!
    pause
    exit /b 1
)

echo Step 2: Linking MSI...
"%WIX_PATH%\light.exe" -nologo -out "dist\VB_OMR_Scanner.msi" "dist\installer.wixobj" -ext WixUIExtension -spdb

if !errorlevel! neq 0 (
    echo.
    echo ERROR: MSI linking failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo MSI Installer created successfully!
echo Output: dist\VB_OMR_Scanner.msi
echo ========================================

REM Cleanup intermediate files
del "dist\installer.wixobj" 2>nul

pause
