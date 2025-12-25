@echo off
REM VB OMR Scanner - Build and Package Script
REM Run this to build signed exe and MSI installer

echo.
echo =============================================
echo VB OMR Scanner - Build Release
echo =============================================
echo.

REM Run the PowerShell build script
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_release.ps1" %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Build failed with error code: %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Build completed successfully!
pause
