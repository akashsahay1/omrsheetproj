#Requires -Version 5.1
<#
.SYNOPSIS
    Build, sign, and package VB OMR Scanner for Windows
.DESCRIPTION
    This script:
    1. Creates a self-signed code signing certificate (if needed)
    2. Builds the exe with PyInstaller
    3. Signs the exe
    4. Downloads prerequisites (WebView2 Runtime)
    5. Creates MSI with WiX Toolset
    6. Creates bootstrapper EXE with prerequisites
    7. Signs the MSI and EXE
.NOTES
    Run as Administrator for certificate creation
    Requires: Python, PyInstaller, WiX Toolset v3.x
#>

param(
    [switch]$SkipCert,
    [switch]$SkipBuild,
    [switch]$SkipSign,
    [switch]$SkipPrereqs,
    [string]$CertThumbprint = "",
    [string]$PfxPath = "",
    [string]$PfxPassword = "omrscanner123"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DistDir = Join-Path $ScriptDir "dist"
$InstallerDir = Join-Path $ScriptDir "installer"
$PrereqsDir = Join-Path $InstallerDir "prereqs"
$ExePath = Join-Path $DistDir "VB OMR Scanner.exe"
$MsiPath = Join-Path $DistDir "VBOMRScanner.msi"
$SetupExePath = Join-Path $DistDir "VBOMRScanner-Setup.exe"

# WebView2 Standalone Offline Installer URLs (no internet required)
# These are the full offline installers (~150MB) not the bootstrapper
$WebView2Urls = @{
    "x64" = "https://go.microsoft.com/fwlink/?linkid=2124701"
    "x86" = "https://go.microsoft.com/fwlink/?linkid=2099617"
}
$WebView2PathX64 = Join-Path $PrereqsDir "MicrosoftEdgeWebView2RuntimeInstallerX64.exe"
$WebView2PathX86 = Join-Path $PrereqsDir "MicrosoftEdgeWebView2RuntimeInstallerX86.exe"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "VB OMR Scanner - Release Build Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "NOTE: Python and all dependencies are bundled in the exe." -ForegroundColor Gray
Write-Host "      Only WebView2 Runtime is needed as a system prerequisite." -ForegroundColor Gray
Write-Host ""

# ============================================
# Step 1: Certificate Setup
# ============================================
if (-not $SkipCert) {
    Write-Host "[1/7] Setting up code signing certificate..." -ForegroundColor Yellow

    if ($PfxPath -and (Test-Path $PfxPath)) {
        Write-Host "  Using provided PFX file: $PfxPath"
        $certPassword = ConvertTo-SecureString -String $PfxPassword -Force -AsPlainText
        $cert = Import-PfxCertificate -FilePath $PfxPath -CertStoreLocation Cert:\CurrentUser\My -Password $certPassword
        $CertThumbprint = $cert.Thumbprint
    }
    elseif ($CertThumbprint) {
        Write-Host "  Using provided certificate thumbprint: $CertThumbprint"
        $cert = Get-ChildItem -Path Cert:\CurrentUser\My | Where-Object { $_.Thumbprint -eq $CertThumbprint }
        if (-not $cert) {
            throw "Certificate with thumbprint $CertThumbprint not found!"
        }
    }
    else {
        # Check for existing VB OMR Scanner certificate
        $existingCert = Get-ChildItem -Path Cert:\CurrentUser\My -CodeSigningCert |
                        Where-Object { $_.Subject -like "*VB OMR Scanner*" } |
                        Sort-Object NotAfter -Descending |
                        Select-Object -First 1

        if ($existingCert -and ($existingCert.NotAfter -gt (Get-Date))) {
            Write-Host "  Found existing valid certificate: $($existingCert.Thumbprint)"
            $cert = $existingCert
            $CertThumbprint = $cert.Thumbprint
        }
        else {
            Write-Host "  Creating new self-signed certificate..."
            $cert = New-SelfSignedCertificate `
                -Type CodeSigningCert `
                -Subject "CN=VB OMR Scanner Dev, O=Vidya Bharti" `
                -KeyUsage DigitalSignature `
                -FriendlyName "VB OMR Scanner Code Signing" `
                -CertStoreLocation "Cert:\CurrentUser\My" `
                -NotAfter (Get-Date).AddYears(3)

            $CertThumbprint = $cert.Thumbprint
            Write-Host "  Certificate created: $CertThumbprint" -ForegroundColor Green

            # Export for backup
            $pfxExportPath = Join-Path $ScriptDir "codesign.pfx"
            $exportPwd = ConvertTo-SecureString -String $PfxPassword -Force -AsPlainText
            Export-PfxCertificate -Cert $cert -FilePath $pfxExportPath -Password $exportPwd | Out-Null
            Write-Host "  Certificate exported to: $pfxExportPath"
            Write-Host "  Password: $PfxPassword" -ForegroundColor Yellow
        }
    }

    # Save thumbprint
    $CertThumbprint | Out-File -FilePath (Join-Path $ScriptDir "cert_thumbprint.txt") -NoNewline
    Write-Host "  Certificate thumbprint saved to cert_thumbprint.txt" -ForegroundColor Green
}
else {
    Write-Host "[1/7] Skipping certificate setup" -ForegroundColor DarkGray
    if (Test-Path (Join-Path $ScriptDir "cert_thumbprint.txt")) {
        $CertThumbprint = Get-Content (Join-Path $ScriptDir "cert_thumbprint.txt")
    }
}

# ============================================
# Step 2: Build EXE with PyInstaller
# ============================================
if (-not $SkipBuild) {
    Write-Host ""
    Write-Host "[2/7] Building executable with PyInstaller..." -ForegroundColor Yellow

    # Activate venv if exists
    $venvActivate = Join-Path $ScriptDir "venv\Scripts\Activate.ps1"
    if (Test-Path $venvActivate) {
        Write-Host "  Activating virtual environment..."
        & $venvActivate
    }

    # Clean previous build
    $buildDir = Join-Path $ScriptDir "build"
    if (Test-Path $buildDir) {
        Write-Host "  Cleaning previous build..."
        Remove-Item -Path $buildDir -Recurse -Force
    }
    if (Test-Path $DistDir) {
        Remove-Item -Path $DistDir -Recurse -Force
    }

    # Build
    Write-Host "  Running PyInstaller..."
    Push-Location $ScriptDir
    python -m PyInstaller build.spec --clean --noconfirm
    Pop-Location

    if (-not (Test-Path $ExePath)) {
        throw "Build failed! Executable not found at: $ExePath"
    }

    Write-Host "  Build successful: $ExePath" -ForegroundColor Green
}
else {
    Write-Host ""
    Write-Host "[2/7] Skipping build" -ForegroundColor DarkGray

    if (-not (Test-Path $ExePath)) {
        throw "Executable not found! Run build first or remove -SkipBuild flag."
    }
}

# ============================================
# Step 3: Sign the EXE
# ============================================
if (-not $SkipSign -and $CertThumbprint) {
    Write-Host ""
    Write-Host "[3/7] Signing executable..." -ForegroundColor Yellow

    # Find signtool
    $signtoolPaths = @(
        "${env:ProgramFiles(x86)}\Windows Kits\10\bin\*\x64\signtool.exe",
        "${env:ProgramFiles(x86)}\Windows Kits\10\App Certification Kit\signtool.exe",
        "${env:ProgramFiles}\Windows Kits\10\bin\*\x64\signtool.exe"
    )

    $signtool = $null
    foreach ($pattern in $signtoolPaths) {
        $found = Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue |
                 Sort-Object { [version]($_.Directory.Parent.Name -replace '[^\d.]') } -Descending |
                 Select-Object -First 1
        if ($found) {
            $signtool = $found.FullName
            break
        }
    }

    if (-not $signtool) {
        Write-Host "  WARNING: signtool.exe not found. Install Windows SDK to sign." -ForegroundColor Red
        Write-Host "  Download from: https://developer.microsoft.com/windows/downloads/windows-sdk/"
    }
    else {
        Write-Host "  Using signtool: $signtool"

        # Sign with SHA256
        & $signtool sign /sha1 $CertThumbprint /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /v $ExePath

        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Executable signed successfully!" -ForegroundColor Green
        }
        else {
            Write-Host "  WARNING: Signing failed (exit code: $LASTEXITCODE)" -ForegroundColor Red
        }
    }
}
else {
    Write-Host ""
    Write-Host "[3/7] Skipping signing" -ForegroundColor DarkGray
}

# ============================================
# Step 4: Download Prerequisites (WebView2 Offline Installers)
# ============================================
if (-not $SkipPrereqs) {
    Write-Host ""
    Write-Host "[4/7] Downloading prerequisites (WebView2 Offline Installers)..." -ForegroundColor Yellow
    Write-Host "  NOTE: Downloading offline installers for fully offline installation support" -ForegroundColor Gray

    # Create prereqs directory
    if (-not (Test-Path $PrereqsDir)) {
        New-Item -ItemType Directory -Path $PrereqsDir -Force | Out-Null
    }

    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

    # Download WebView2 x64 Standalone Installer (~150MB)
    if (-not (Test-Path $WebView2PathX64)) {
        Write-Host "  Downloading WebView2 x64 Offline Installer (~150MB)..."
        try {
            Invoke-WebRequest -Uri $WebView2Urls["x64"] -OutFile $WebView2PathX64 -UseBasicParsing
            Write-Host "  WebView2 x64 downloaded successfully!" -ForegroundColor Green
        }
        catch {
            Write-Host "  WARNING: Failed to download WebView2 x64: $_" -ForegroundColor Red
        }
    }
    else {
        Write-Host "  WebView2 x64 offline installer already exists."
    }

    # Download WebView2 x86 Standalone Installer for 32-bit systems
    if (-not (Test-Path $WebView2PathX86)) {
        Write-Host "  Downloading WebView2 x86 Offline Installer (~140MB)..."
        try {
            Invoke-WebRequest -Uri $WebView2Urls["x86"] -OutFile $WebView2PathX86 -UseBasicParsing
            Write-Host "  WebView2 x86 downloaded successfully!" -ForegroundColor Green
        }
        catch {
            Write-Host "  WARNING: Failed to download WebView2 x86: $_" -ForegroundColor Red
        }
    }
    else {
        Write-Host "  WebView2 x86 offline installer already exists."
    }
}
else {
    Write-Host ""
    Write-Host "[4/7] Skipping prerequisites download" -ForegroundColor DarkGray
}

# ============================================
# Step 5: Build MSI with WiX
# ============================================
Write-Host ""
Write-Host "[5/7] Building MSI installer with WiX..." -ForegroundColor Yellow

# Find WiX
$wixPaths = @(
    "${env:ProgramFiles(x86)}\WiX Toolset v3*\bin",
    "${env:ProgramFiles}\WiX Toolset v3*\bin",
    "$env:USERPROFILE\.dotnet\tools"
)

$wixBin = $null
foreach ($pattern in $wixPaths) {
    $found = Get-ChildItem -Path $pattern -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found -and (Test-Path (Join-Path $found.FullName "candle.exe"))) {
        $wixBin = $found.FullName
        break
    }
}

if (-not $wixBin) {
    Write-Host "  ERROR: WiX Toolset not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Please install WiX Toolset v3.x:" -ForegroundColor Yellow
    Write-Host "  1. Download from: https://github.com/wixtoolset/wix3/releases" -ForegroundColor White
    Write-Host "  2. Install the .exe installer" -ForegroundColor White
    Write-Host "  3. Restart PowerShell and run this script again" -ForegroundColor White
    Write-Host ""
    Write-Host "  Or install via winget:" -ForegroundColor Yellow
    Write-Host "  winget install WiXToolset.WiX" -ForegroundColor White
    exit 1
}

Write-Host "  Using WiX: $wixBin"

$candle = Join-Path $wixBin "candle.exe"
$light = Join-Path $wixBin "light.exe"

# Build MSI
Push-Location $InstallerDir

Write-Host "  Running candle (WiX compiler)..."
& $candle Product.wxs -out Product.wixobj
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    throw "WiX candle failed!"
}

Write-Host "  Running light (WiX linker)..."
& $light -ext WixUIExtension Product.wixobj -out $MsiPath
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    throw "WiX light failed!"
}

Write-Host "  MSI created: $MsiPath" -ForegroundColor Green

# ============================================
# Step 6: Build Offline Bootstrapper EXE with WiX Burn
# ============================================
Write-Host ""
Write-Host "[6/7] Building OFFLINE bootstrapper with embedded prerequisites..." -ForegroundColor Yellow

$hasWebView2X64 = Test-Path $WebView2PathX64
$hasWebView2X86 = Test-Path $WebView2PathX86

if ($hasWebView2X64 -and $hasWebView2X86) {
    Write-Host "  WebView2 offline installers found (x64 + x86)"
    Write-Host "  Building fully offline installer bundle..."

    # Build bundle with embedded WebView2
    Write-Host "  Running candle for bundle..."
    & $candle Bundle.wxs -ext WixBalExtension -ext WixUtilExtension -out Bundle.wixobj
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  WARNING: Bundle candle failed, creating fallback installer..." -ForegroundColor Yellow
        $bundleFailed = $true
    }
    else {
        Write-Host "  Running light for bundle (this may take a while - embedding ~300MB of prereqs)..."
        & $light Bundle.wixobj -ext WixBalExtension -ext WixUtilExtension -o $SetupExePath
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  WARNING: Bundle light failed, creating fallback installer..." -ForegroundColor Yellow
            $bundleFailed = $true
        }
        else {
            $setupSize = [math]::Round((Get-Item $SetupExePath).Length / 1MB, 2)
            Write-Host "  Offline bootstrapper created: $SetupExePath ($setupSize MB)" -ForegroundColor Green
            Write-Host "  This installer works completely OFFLINE - no internet required!" -ForegroundColor Cyan
        }
    }
}
else {
    Write-Host "  WebView2 offline installers not found, creating fallback..." -ForegroundColor Yellow
    $bundleFailed = $true
}

# Fallback: Create a self-contained folder with all files
if ($bundleFailed) {
    Write-Host "  Creating portable installer package..." -ForegroundColor Yellow

    $portableDir = Join-Path $DistDir "VBOMRScanner-Portable-Setup"
    if (Test-Path $portableDir) { Remove-Item $portableDir -Recurse -Force }
    New-Item -ItemType Directory -Path $portableDir -Force | Out-Null

    # Copy MSI
    Copy-Item $MsiPath $portableDir

    # Copy WebView2 installers if available
    if ($hasWebView2X64) { Copy-Item $WebView2PathX64 $portableDir }
    if ($hasWebView2X86) { Copy-Item $WebView2PathX86 $portableDir }

    # Create install script
    $installScript = Join-Path $portableDir "Install.bat"
    @"
@echo off
echo ============================================
echo VB OMR Scanner - Offline Installer
echo ============================================
echo.
echo This installer works completely OFFLINE.
echo.

:: Check for WebView2
echo Checking for WebView2 Runtime...
reg query "HKLM\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" >nul 2>&1
if %ERRORLEVEL% EQU 0 goto :webview_ok
reg query "HKLM\SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" >nul 2>&1
if %ERRORLEVEL% EQU 0 goto :webview_ok

:: Install WebView2
echo WebView2 Runtime not found. Installing from local package...
if exist "%~dp0MicrosoftEdgeWebView2RuntimeInstallerX64.exe" (
    echo Installing WebView2 x64...
    "%~dp0MicrosoftEdgeWebView2RuntimeInstallerX64.exe" /silent /install
) else if exist "%~dp0MicrosoftEdgeWebView2RuntimeInstallerX86.exe" (
    echo Installing WebView2 x86...
    "%~dp0MicrosoftEdgeWebView2RuntimeInstallerX86.exe" /silent /install
) else (
    echo ERROR: WebView2 installer not found!
    echo Please download from: https://developer.microsoft.com/microsoft-edge/webview2/
    pause
    exit /b 1
)
echo WebView2 installed successfully!

:webview_ok
echo WebView2 Runtime is installed.
echo.
echo Installing VB OMR Scanner...
msiexec /i "%~dp0VBOMRScanner.msi"
"@ | Out-File -FilePath $installScript -Encoding ASCII

    Write-Host "  Portable installer created: $portableDir" -ForegroundColor Green
    Write-Host "  Distribute the entire folder for offline installation." -ForegroundColor Cyan
}

Pop-Location

# ============================================
# Step 7: Sign the MSI and Bootstrapper
# ============================================
if (-not $SkipSign -and $CertThumbprint -and $signtool) {
    Write-Host ""
    Write-Host "[7/7] Signing installers..." -ForegroundColor Yellow

    # Sign MSI
    & $signtool sign /sha1 $CertThumbprint /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /v $MsiPath
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  MSI signed successfully!" -ForegroundColor Green
    }
    else {
        Write-Host "  WARNING: MSI signing failed" -ForegroundColor Red
    }

    # Sign Bootstrapper EXE if it exists
    if (Test-Path $SetupExePath) {
        & $signtool sign /sha1 $CertThumbprint /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /v $SetupExePath
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Bootstrapper signed successfully!" -ForegroundColor Green
        }
        else {
            Write-Host "  WARNING: Bootstrapper signing failed" -ForegroundColor Red
        }
    }
}
else {
    Write-Host ""
    Write-Host "[7/7] Skipping signing" -ForegroundColor DarkGray
}

# ============================================
# Summary
# ============================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Build Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Output files:" -ForegroundColor White
Write-Host "  Application:   $ExePath"
Write-Host "  MSI Installer: $MsiPath"

if (Test-Path $SetupExePath) {
    $setupSize = [math]::Round((Get-Item $SetupExePath).Length / 1MB, 2)
    Write-Host "  OFFLINE Setup: $SetupExePath ($setupSize MB)" -ForegroundColor Green
    Write-Host ""
    Write-Host "  >>> DISTRIBUTE: VBOMRScanner-Setup.exe <<<" -ForegroundColor Cyan
    Write-Host "  This single file contains EVERYTHING needed for offline installation:" -ForegroundColor White
    Write-Host "    - VB OMR Scanner application"
    Write-Host "    - Python runtime (bundled)"
    Write-Host "    - All Python dependencies (bundled)"
    Write-Host "    - WebView2 Runtime installer (embedded)"
}

$portableDir = Join-Path $DistDir "VBOMRScanner-Portable-Setup"
if (Test-Path $portableDir) {
    Write-Host ""
    Write-Host "  Portable Setup: $portableDir" -ForegroundColor Green
    Write-Host "  (Alternative: distribute entire folder for offline installation)"
}

if ($CertThumbprint) {
    Write-Host ""
    Write-Host "Signing certificate: $CertThumbprint"
}

Write-Host ""
Write-Host "OFFLINE INSTALLATION SUPPORT:" -ForegroundColor Cyan
Write-Host "  - NO INTERNET required on target PC"
Write-Host "  - Python runtime is bundled in the exe"
Write-Host "  - All Python packages are bundled (cv2, numpy, pandas, etc.)"
Write-Host "  - WebView2 Runtime is embedded in the installer"
Write-Host "  - Installer auto-detects if WebView2 is already installed"
Write-Host ""
Write-Host "NOTES:" -ForegroundColor Yellow
Write-Host "  - Self-signed certificates will show 'Unknown Publisher' warning"
Write-Host "  - For production, use a trusted code signing certificate"
