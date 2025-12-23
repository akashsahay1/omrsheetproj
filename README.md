# VB OMR Scanner

**Vidya Bharti Pratibha Khoj and Pratibha Chayan OMR Scanner**

An automated Optical Mark Recognition (OMR) application designed specifically for processing Vidya Bharti Pratibha Khoj and Pratibha Chayan examination answer sheets.

## Features

- **Batch Processing** - Process multiple scanned OMR sheets at once
- **Accurate Detection** - Detects candidate information and answers with high accuracy
- **Configurable Scoring** - Set custom marks for correct, wrong, and unanswered questions
- **Answer Key Management** - Create, import, and manage answer keys
- **Results Export** - Export results to Excel with rankings and statistics
- **Cross-Platform** - Builds for both macOS and Windows

### Detected Fields

| Field | Description |
|-------|-------------|
| Candidate Name | Up to 25 characters (A-Z) |
| Roll Number | 3 digits |
| Class | 3-10 |
| Section | A-H |
| School Code | 3 digits |
| Answers | 60 questions (A, B, C, D) |

## System Requirements

### For Running the Application
- **macOS:** 10.13 (High Sierra) or later
- **Windows:** Windows 10 or later
- **RAM:** 4 GB minimum (8 GB recommended)
- **Storage:** 500 MB free space

### For Building from Source
- Python 3.9 or higher
- pip (Python package manager)
- 2 GB additional storage for dependencies

## Installation

### Option 1: Download Pre-built Application

Download the latest release from the [Releases](https://github.com/akashsahay1/omrsheetproj/releases) page.

- **macOS:** Download `VB_OMR_Scanner.dmg`
- **Windows:** Download `VB_OMR_Scanner.exe`

### Option 2: Build from Source

#### Clone the Repository

```bash
git clone https://github.com/akashsahay1/omrsheetproj.git
cd omrsheetproj
```

#### macOS

```bash
# Setup environment and install dependencies
chmod +x setup.sh build.sh run.sh
./setup.sh

# Build the application
./build.sh

# Run the application
./run.sh
# Or open the built app
open "dist/VB OMR Scanner.app"
```

#### Windows

```batch
# Setup environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Build the application
build.bat

# Run the application
dist\VB OMR Scanner.exe
```

## Usage

### Quick Start

1. **Launch the Application**
   - macOS: Double-click `VB OMR Scanner.app`
   - Windows: Double-click `VB OMR Scanner.exe`

2. **Configure Answer Key**
   - Go to "Answer Key" in the sidebar
   - Set correct answers for all 60 questions
   - Click "Save Answer Key"

3. **Process OMR Sheets**
   - Go to "Process Sheets"
   - Click "Browse" to select the folder containing scanned images
   - Optionally set an output Excel file path
   - Click "Start Processing"

4. **View Results**
   - Go to "Results" to see processed data
   - Click "Export to Excel" to save results

### Scanning Recommendations

For best results, scan OMR sheets with these settings:

| Setting | Recommended Value |
|---------|-------------------|
| Resolution | 300 DPI |
| Color Mode | Grayscale or Color |
| Format | JPEG or PNG |
| Orientation | Portrait (correct side up) |

### Scoring Configuration

Default scoring can be modified in Settings:

| Answer Type | Default Marks |
|-------------|---------------|
| Correct | +1.0 |
| Wrong | -0.25 |
| Unanswered | 0 |

## Project Structure

```
omrsheetproj/
├── src/                    # Application source code
│   ├── config/            # Configuration and constants
│   ├── core/              # OMR processing engine
│   ├── gui/               # GUI application and API
│   ├── models/            # Data models
│   ├── services/          # Business logic services
│   └── utils/             # Utility functions
├── ui/                     # Web UI (HTML, CSS, JS)
│   ├── css/               # Stylesheets
│   └── js/                # JavaScript files
├── OMRChecker/            # OMRChecker library
│   ├── inputs/            # Input templates
│   │   └── VidyaBharti2025/  # Vidya Bharti template
│   └── src/               # OMRChecker source
├── assets/                # App icons
├── templates/             # OMR sheet templates
├── build.spec             # PyInstaller build configuration
├── setup.sh               # macOS setup script
├── build.sh               # macOS build script
├── run.sh                 # macOS run script
├── build.bat              # Windows build script
├── DOCUMENTATION.txt      # User documentation (text)
├── DOCUMENTATION.pdf      # User documentation (PDF)
└── README.md              # This file
```

## Configuration

### Template Configuration

The OMR template is configured in `OMRChecker/inputs/VidyaBharti2025/template.json`:

```json
{
  "pageDimensions": [2480, 3507],
  "bubbleDimensions": [28, 28],
  "fieldBlocks": {
    "CandidateName": { ... },
    "RollNumber": { ... },
    "Class": { ... },
    "Section": { ... },
    "SchoolCode": { ... },
    "Answers_1_10": { ... },
    ...
  }
}
```

### Detection Thresholds

Fine-tune detection in `OMRChecker/inputs/VidyaBharti2025/config.json`:

```json
{
  "threshold_params": {
    "GAMMA_LOW": 0.7,
    "MIN_GAP": 20,
    "MIN_JUMP": 15,
    "CONFIDENT_SURPLUS": 5,
    "JUMP_DELTA": 15
  }
}
```

## Dependencies

### Python Packages

| Package | Purpose |
|---------|---------|
| pywebview | Native GUI window |
| opencv-python | Image processing |
| numpy | Numerical operations |
| pandas | Data manipulation |
| openpyxl | Excel export |
| Pillow | Image handling |
| jsonschema | JSON validation |
| rich | Console output |
| matplotlib | Visualization |
| dotmap | Dictionary access |
| deepmerge | Config merging |

### Install All Dependencies

```bash
pip install opencv-python numpy pandas pywebview openpyxl Pillow jsonschema rich matplotlib dotmap deepmerge screeninfo pyinstaller
```

## Troubleshooting

### Application Won't Start

- **macOS:** Right-click the app and select "Open" if blocked by Gatekeeper
- **Windows:** Run as Administrator if permission errors occur

### Poor Detection Accuracy

- Ensure scans are at 300 DPI resolution
- Sheets should be properly aligned (not skewed)
- Bubbles should be filled darkly and completely
- Avoid shadows or artifacts in scans

### "No Results" After Processing

- Check that the answer key is configured
- Verify input folder contains valid image files
- Check console output for error messages

### School Code Detection Issues

- Template is configured for 3-digit school codes
- Fill leading zeros if required (e.g., "002")

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is proprietary software developed for Vidya Bharti examinations.

## Credits

**Developed by:** [Avenux Techspire (OPC) PVT LTD](https://www.avenuxtechspire.com)

**OMR Processing:** Based on [OMRChecker](https://github.com/Udayraj123/OMRChecker) by Udayraj Deshmukh

## Support

For technical support or bug reports:
- Open an issue on [GitHub Issues](https://github.com/akashsahay1/omrsheetproj/issues)
- Email: support@avenuxtechspire.com

---

**Version:** 1.0.0
**Last Updated:** December 2024
