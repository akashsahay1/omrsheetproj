"""
Create MSI installer for VB OMR Scanner
Uses Python's built-in msilib module
"""
import msilib
import os
import uuid
import shutil
import tempfile

# Application info
APP_NAME = "VB OMR Scanner"
APP_VERSION = "1.0.0"
MANUFACTURER = "Vidya Bharti"
PRODUCT_CODE = "{" + str(uuid.uuid4()).upper() + "}"
UPGRADE_CODE = "{8F4E5C2A-1B3D-4E5F-9A8B-7C6D5E4F3A2B}"  # Keep same for upgrades

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXE_PATH = os.path.join(BASE_DIR, "dist", "VB OMR Scanner.exe")
MSI_PATH = os.path.join(BASE_DIR, "dist", "VB_OMR_Scanner_Setup.msi")

# Check exe exists
if not os.path.exists(EXE_PATH):
    print(f"ERROR: Executable not found at {EXE_PATH}")
    print("Please run PyInstaller build first.")
    exit(1)

print(f"Creating MSI installer...")
print(f"  Source: {EXE_PATH}")
print(f"  Output: {MSI_PATH}")

# Remove existing MSI
if os.path.exists(MSI_PATH):
    os.remove(MSI_PATH)

# Create MSI database
db = msilib.init_database(
    MSI_PATH,
    msilib.schema,
    APP_NAME,
    PRODUCT_CODE,
    APP_VERSION,
    MANUFACTURER
)

# Add summary info
msilib.add_data(db, "Property", [
    ("UpgradeCode", UPGRADE_CODE),
    ("ARPPRODUCTICON", "app.exe"),
    ("ALLUSERS", "1"),
    ("ARPNOREPAIR", "1"),
])

# Create directory structure
# TARGETDIR -> ProgramFilesFolder -> ManufacturerFolder -> ProductFolder
cab = msilib.CAB("installer")

msilib.add_data(db, "Directory", [
    ("TARGETDIR", None, "SourceDir"),
    ("ProgramFilesFolder", "TARGETDIR", "."),
    ("ManufacturerFolder", "ProgramFilesFolder", MANUFACTURER),
    ("INSTALLDIR", "ManufacturerFolder", APP_NAME),
    ("DesktopFolder", "TARGETDIR", "."),
    ("ProgramMenuFolder", "TARGETDIR", "."),
    ("StartMenuFolder", "ProgramMenuFolder", APP_NAME),
])

# Add the main executable
exe_size = os.path.getsize(EXE_PATH) // 512 + 1  # Size in 512-byte units
exe_filename = os.path.basename(EXE_PATH)

# Component and File
component_id = "MainExecutable"
file_id = "VBOMRScanner.exe"

msilib.add_data(db, "Component", [
    (component_id, "{" + str(uuid.uuid4()).upper() + "}", "INSTALLDIR", 0, None, file_id),
])

msilib.add_data(db, "FeatureComponents", [
    ("MainFeature", component_id),
])

# Add file to CAB
cab.append(EXE_PATH, file_id, file_id)

msilib.add_data(db, "File", [
    (file_id, component_id, exe_filename, exe_size, None, None, 512, 1),
])

# Create shortcuts
# Desktop shortcut component
desktop_comp = "DesktopShortcut"
msilib.add_data(db, "Component", [
    (desktop_comp, "{" + str(uuid.uuid4()).upper() + "}", "DesktopFolder", 0, None, "DesktopSC"),
])
msilib.add_data(db, "FeatureComponents", [
    ("MainFeature", desktop_comp),
])

# Start Menu shortcut component
startmenu_comp = "StartMenuShortcut"
msilib.add_data(db, "Component", [
    (startmenu_comp, "{" + str(uuid.uuid4()).upper() + "}", "StartMenuFolder", 0, None, "StartMenuSC"),
])
msilib.add_data(db, "FeatureComponents", [
    ("MainFeature", startmenu_comp),
])

# Shortcut table
msilib.add_data(db, "Shortcut", [
    ("DesktopSC", "DesktopFolder", APP_NAME, desktop_comp, "[INSTALLDIR]" + exe_filename, None, None, None, None, None, None, "TARGETDIR", None, None),
    ("StartMenuSC", "StartMenuFolder", APP_NAME, startmenu_comp, "[INSTALLDIR]" + exe_filename, None, None, None, None, None, None, "TARGETDIR", None, None),
])

# Create RemoveFile entries for shortcuts during uninstall
msilib.add_data(db, "RemoveFile", [
    ("RemoveDesktopSC", desktop_comp, None, "DesktopFolder", 2),
    ("RemoveStartMenuSC", startmenu_comp, None, "StartMenuFolder", 2),
    ("RemoveStartMenuDir", startmenu_comp, None, "StartMenuFolder", 2),
])

# Feature table
msilib.add_data(db, "Feature", [
    ("MainFeature", None, APP_NAME, f"Install {APP_NAME}", 1, 1, "INSTALLDIR", 0),
])

# Media table - for CAB
msilib.add_data(db, "Media", [
    (1, 2, None, "#installer.cab", None, None),
])

# Commit CAB
cab.commit(db)

# Commit database
db.Commit()

print(f"\nMSI installer created successfully!")
print(f"  File: {MSI_PATH}")
print(f"  Size: {os.path.getsize(MSI_PATH) / (1024*1024):.1f} MB")
print(f"\nTo install: Double-click the MSI file or run:")
print(f"  msiexec /i \"{MSI_PATH}\"")
