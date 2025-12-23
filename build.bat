@echo off
echo ===============================================
echo  OMR Processor - Build Executable
echo ===============================================
echo.

REM Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first.
    pause
    exit /b 1
)

REM Activate venv
call venv\Scripts\activate.bat

REM Install PyInstaller if not present
pip install pyinstaller

echo.
echo Building executable...
echo This may take several minutes...
echo.

pyinstaller --clean build.spec

if errorlevel 1 (
    echo.
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo ===============================================
echo  Build completed successfully!
echo ===============================================
echo.
echo Executable location: dist\OMR Processor.exe
echo.
pause
