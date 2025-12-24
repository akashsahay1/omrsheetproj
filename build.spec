# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for VB OMR Scanner
Builds for macOS and Windows
"""

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

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
]

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'pywebview',
    'webview',
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
]

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
        'clr',
    ])

# Collect numpy data files
numpy_datas = collect_data_files('numpy')
datas.extend(numpy_datas)

# Analysis
a = Analysis(
    [os.path.join(BASE_DIR, 'src', 'main.py')],
    pathex=[BASE_DIR, os.path.join(BASE_DIR, 'src')],
    binaries=[],
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
        icon=os.path.join(BASE_DIR, 'assets', 'icon.icns') if os.path.exists(os.path.join(BASE_DIR, 'assets', 'icon.icns')) else None,
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
