# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for VB OMR Scanner
Builds for macOS and Windows
"""

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_all

# Application metadata
APP_NAME = "Vidya Bharti Pratibha Khoj and Pratibha Chayan OMR Scanner"
APP_SHORT_NAME = "VB OMR Scanner"
APP_VERSION = "1.0.0"
BUNDLE_ID = "com.vidyabharti.omrprocessor"

# Paths
block_cipher = None
BASE_DIR = os.path.abspath(os.path.dirname(SPEC))

# Determine platform
is_windows = sys.platform == 'win32'
is_macos = sys.platform == 'darwin'

# Data files to include
datas = [
    # UI files (HTML, CSS, JS)
    (os.path.join(BASE_DIR, 'ui'), 'ui'),
    # Templates
    (os.path.join(BASE_DIR, 'templates'), 'templates'),
    # OMRChecker library and templates
    (os.path.join(BASE_DIR, 'OMRChecker'), 'OMRChecker'),
    # ML models
    (os.path.join(BASE_DIR, 'models'), 'models'),
]

# Collect all webview submodules automatically
webview_submodules = collect_submodules('webview')

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'cv2',
    'numpy',
    'openpyxl',
    'PIL',
    'PIL.Image',
    'jsonschema',
    # Pandas and dependencies (required by OMRChecker)
    'pandas',
    'pandas._libs',
    'pandas._libs.tslibs',
    'pandas._libs.tslibs.np_datetime',
    'pandas._libs.tslibs.nattype',
    'pandas._libs.tslibs.timedeltas',
    # Rich (required by OMRChecker)
    'rich',
    'rich.console',
    'rich.table',
    'rich.logging',
    'rich.progress',
    'rich.markup',
    'rich.text',
    'rich.traceback',
    # Additional OMRChecker dependencies
    'dotmap',
    'deepmerge',
    'screeninfo',
    'matplotlib',
    'matplotlib.pyplot',
    # OMRChecker modules
    'src.core',
    'src.processors',
    'src.template',
    'src.entry',
    # ML/AI modules
    'torch',
    'torch.nn',
    'torch.optim',
    'torchvision',
    'pytesseract',
    # EasyOCR for handwritten name extraction
    'easyocr',
    'easyocr.recognition',
    'easyocr.detection',
    'easyocr.utils',
    'easyocr.model',
    'easyocr.craft_utils',
    'easyocr.imgproc',
    'src.ai',
    'src.ai.yolo_omr_processor',
    'src.ai.bubble_classifier',
    'src.ai.ocr_extractor',
    'src.core.ml_engine',
]

# Add webview submodules
hiddenimports.extend(webview_submodules)

# Platform-specific hidden imports
if is_macos:
    hiddenimports.extend([
        'webview.platforms.cocoa',
        'AppKit',
        'Foundation',
        'WebKit',
        'objc',
    ])
elif is_windows:
    hiddenimports.extend([
        'webview.platforms.edgechromium',
        'webview.platforms.winforms',
        'webview.platforms.mshtml',
        'webview.platforms.cef',
        'clr',
        'clr_loader',
        'pythonnet',
        'System',
        'System.Windows.Forms',
        'System.Drawing',
        'System.Threading',
    ])
    # Collect clr_loader and pythonnet data files for .NET integration
    try:
        clr_datas = collect_data_files('clr_loader')
        datas.extend(clr_datas)
    except Exception:
        pass
    try:
        pythonnet_datas = collect_data_files('pythonnet')
        datas.extend(pythonnet_datas)
    except Exception:
        pass

# Collect numpy data files
numpy_datas = collect_data_files('numpy')
datas.extend(numpy_datas)

# Collect webview data files (JS, HTML resources)
webview_datas = collect_data_files('webview')
datas.extend(webview_datas)

# Collect torch data files (for ML)
try:
    torch_datas = collect_data_files('torch')
    datas.extend(torch_datas)
except Exception:
    pass

# Collect torchvision data files
try:
    torchvision_datas = collect_data_files('torchvision')
    datas.extend(torchvision_datas)
except Exception:
    pass

# Collect EasyOCR data files (models, etc.)
try:
    easyocr_datas = collect_data_files('easyocr')
    datas.extend(easyocr_datas)
except Exception:
    pass

# Collect OpenCV (cv2) - all data, binaries, and submodules
cv2_datas, cv2_binaries, cv2_hiddenimports = collect_all('cv2')
datas.extend(cv2_datas)
hiddenimports.extend(cv2_hiddenimports)

# Store cv2 binaries for later
binaries = cv2_binaries

# Analysis
a = Analysis(
    [os.path.join(BASE_DIR, 'src', 'main.py')],
    pathex=[BASE_DIR, os.path.join(BASE_DIR, 'src')],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filter out unnecessary files to reduce size
a.datas = [d for d in a.datas if not any(x in d[0] for x in [
    'tcl', 'tk', 'test', 'tests', '__pycache__'
])]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Platform-specific build
if is_macos:
    # macOS: Create .app bundle
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=APP_SHORT_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=True,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name=APP_SHORT_NAME,
    )

    app = BUNDLE(
        coll,
        name=f'{APP_SHORT_NAME}.app',
        icon=os.path.join(BASE_DIR, 'assets', 'vbomr_icon.icns') if os.path.exists(os.path.join(BASE_DIR, 'assets', 'vbomr_icon.icns')) else None,
        bundle_identifier=BUNDLE_ID,
        info_plist={
            'CFBundleName': APP_NAME,
            'CFBundleDisplayName': APP_NAME,
            'CFBundleVersion': APP_VERSION,
            'CFBundleShortVersionString': APP_VERSION,
            'NSHighResolutionCapable': True,
            'NSRequiresAquaSystemAppearance': False,
            'LSMinimumSystemVersion': '10.13.0',
        },
    )
else:
    # Windows: Create single executable
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name=APP_SHORT_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=os.path.join(BASE_DIR, 'assets', 'icon.ico') if os.path.exists(os.path.join(BASE_DIR, 'assets', 'icon.ico')) else None,
    )
