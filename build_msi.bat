@echo off
setlocal enabledelayedexpansion
echo =============================================
echo VB OMR Scanner - MSI Installer Builder
echo =============================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Try to find WiX Toolset in common locations
set "WIX_PATH="

REM Check WiX v3.x locations
if exist "C:\Program Files (x86)\WiX Toolset v3.14\bin\candle.exe" set "WIX_PATH=C:\Program Files (x86)\WiX Toolset v3.14\bin"
if "!WIX_PATH!"=="" if exist "C:\Program Files (x86)\WiX Toolset v3.11\bin\candle.exe" set "WIX_PATH=C:\Program Files (x86)\WiX Toolset v3.11\bin"
if "!WIX_PATH!"=="" if exist "C:\Program Files\WiX Toolset v3.14\bin\candle.exe" set "WIX_PATH=C:\Program Files\WiX Toolset v3.14\bin"
if "!WIX_PATH!"=="" if exist "C:\Program Files\WiX Toolset v3.11\bin\candle.exe" set "WIX_PATH=C:\Program Files\WiX Toolset v3.11\bin"

REM Check WiX v4.x / v5.x (uses 'wix' command)
if "!WIX_PATH!"=="" (
    where wix >nul 2>&1
    if !errorlevel! equ 0 (
        set "WIX_V4=1"
        echo Found WiX v4/v5 in PATH
    )
)

REM Check via environment variable
if "!WIX_PATH!"=="" if defined WIX (
    if exist "!WIX!\bin\candle.exe" set "WIX_PATH=!WIX!\bin"
)

REM Exit if WiX not found
if "!WIX_PATH!"=="" if not defined WIX_V4 (
    echo ERROR: WiX Toolset not found!
    echo.
    echo Please install WiX Toolset:
    echo   - WiX v3: https://github.com/wixtoolset/wix3/releases
    echo   - WiX v4+: winget install WixToolset.WiX
    echo.
    echo Or set WIX environment variable to your WiX installation path.
    pause
    exit /b 1
)

REM Check if exe exists
if not exist "dist\VB OMR Scanner.exe" (
    echo ERROR: VB OMR Scanner.exe not found in dist folder!
    echo.
    echo Please run build.bat first to create the executable.
    pause
    exit /b 1
)

REM Check for required files
if not exist "installer.wxs" (
    echo ERROR: installer.wxs not found!
    pause
    exit /b 1
)

REM Create license.rtf if missing
if not exist "license.rtf" (
    echo Creating default license file...
    echo {\rtf1\ansi\deff0 > license.rtf
    echo {\fonttbl{\f0 Arial;}} >> license.rtf
    echo \f0\fs20 >> license.rtf
    echo VB OMR Scanner - Vidya Bharti Pratibha Khoj\par >> license.rtf
    echo \par >> license.rtf
    echo Copyright (c^) 2025 Vidya Bharti. All rights reserved.\par >> license.rtf
    echo \par >> license.rtf
    echo This software is proprietary and confidential.\par >> license.rtf
    echo } >> license.rtf
)

REM Create assets folder and icon if missing
if not exist "assets" mkdir assets
if not exist "assets\icon.ico" (
    echo WARNING: assets\icon.ico not found. MSI will use default icon.
    echo To add a custom icon, place icon.ico in the assets folder.
    echo.
)

REM Create output directory if needed
if not exist "dist" mkdir dist

echo.
if defined WIX_V4 (
    echo Using WiX v4/v5...
    echo.
    echo Step 1: Building MSI with wix build...
    wix build -o "dist\VB_OMR_Scanner.msi" "installer.wxs" -ext WixToolset.UI.wixext
) else (
    echo Using WiX v3 at: !WIX_PATH!
    echo.

    echo Step 1: Compiling WiX source...
    "!WIX_PATH!\candle.exe" -nologo -out "dist\installer.wixobj" "installer.wxs" -ext WixUIExtension

    if !errorlevel! neq 0 (
        echo.
        echo ERROR: WiX compilation failed!
        pause
        exit /b 1
    )

    echo Step 2: Linking MSI...
    "!WIX_PATH!\light.exe" -nologo -out "dist\VB_OMR_Scanner.msi" "dist\installer.wixobj" -ext WixUIExtension -spdb
)

if !errorlevel! neq 0 (
    echo.
    echo ERROR: MSI creation failed!
    pause
    exit /b 1
)

echo.
echo =============================================
echo MSI Installer created successfully!
echo Output: dist\VB_OMR_Scanner.msi
echo =============================================
echo.

REM Cleanup intermediate files
del "dist\installer.wixobj" 2>nul

pause
