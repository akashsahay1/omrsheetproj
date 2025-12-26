# OMR Template Calibration Documentation

## Overview

This document contains the calibrated coordinates and configuration for the Vidya Bharti 2025 OMR Answer Sheet template. These values were calibrated using PixSpy tool on images resized to template dimensions (1360x1928).

## Template Dimensions

| Parameter | Value |
|-----------|-------|
| Page Width | 1360 px |
| Page Height | 1928 px |
| Bubble Width | 14 px |
| Bubble Height | 14 px |

## Calibrated Field Coordinates

### CandidateName Grid (Top-Left)
- **Origin**: [105, 387]
- **Bubbles Gap** (vertical): 28 px
- **Labels Gap** (horizontal): 31 px
- **Direction**: vertical
- **Columns**: 25 (name1..25)
- **Bubble Values**: A-Z (26 letters per column)
- **Accuracy**: 70-90%

### RollNumber (Top-Right, 3 digits)
- **Origin**: [874, 712]
- **Bubbles Gap**: 28 px
- **Labels Gap**: 31 px
- **Direction**: vertical
- **Field Type**: QTYPE_INT
- **Fields**: roll1, roll2, roll3
- **Accuracy**: 100%

### Class (Top-Right)
- **Origin**: [992, 712]
- **Bubbles Gap**: 28 px
- **Labels Gap**: 31 px
- **Direction**: vertical
- **Bubble Values**: ["3", "4", "5", "6", "7", "8", "9", "10"]
- **Accuracy**: 100%

### Section (Top-Right)
- **Origin**: [1075, 712]
- **Bubbles Gap**: 28 px
- **Labels Gap**: 31 px
- **Direction**: vertical
- **Bubble Values**: ["A", "B", "C", "D", "E", "F", "G", "H"]
- **Accuracy**: 100%

### SchoolCode (Top-Right, 3 digits)
- **Origin**: [1165, 712]
- **Bubbles Gap**: 28 px
- **Labels Gap**: 31 px
- **Direction**: vertical
- **Field Type**: QTYPE_INT
- **Fields**: school1, school2, school3
- **Accuracy**: 100%

### Answer Blocks (Bottom Section)

| Block | Origin | Questions |
|-------|--------|-----------|
| Answers_1_10 | [161, 1211] | q1..10 |
| Answers_11_20 | [351, 1211] | q11..20 |
| Answers_21_30 | [541, 1211] | q21..30 |
| Answers_31_40 | [731, 1211] | q31..40 |
| Answers_41_50 | [921, 1211] | q41..50 |
| Answers_51_60 | [1111, 1211] | q51..60 |

All answer blocks use:
- **Field Type**: QTYPE_MCQ4
- **Bubbles Gap**: 42 px
- **Labels Gap**: 51 px
- **Direction**: horizontal

## Pre-Processors

```json
"preProcessors": [
  {
    "name": "Levels",
    "options": {
      "low": 0.35,
      "high": 0.95
    }
  },
  {
    "name": "GaussianBlur",
    "options": {
      "kSize": [3, 3],
      "sigmaX": 0
    }
  }
]
```

## Custom Labels

```json
"customLabels": {
  "Roll": ["roll1..3"],
  "SchoolCode": ["school1..3"]
}
```

## Complete Template JSON

```json
{
  "pageDimensions": [1360, 1928],
  "bubbleDimensions": [14, 14],
  "preProcessors": [
    {
      "name": "Levels",
      "options": {
        "low": 0.35,
        "high": 0.95
      }
    },
    {
      "name": "GaussianBlur",
      "options": {
        "kSize": [3, 3],
        "sigmaX": 0
      }
    }
  ],
  "customLabels": {
    "Roll": ["roll1..3"],
    "SchoolCode": ["school1..3"]
  },
  "fieldBlocks": {
    "CandidateName": {
      "origin": [105, 387],
      "bubbleDimensions": [14, 14],
      "fieldLabels": ["name1..25"],
      "bubbleValues": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"],
      "bubblesGap": 28,
      "labelsGap": 31,
      "direction": "vertical"
    },
    "RollNumber": {
      "fieldType": "QTYPE_INT",
      "origin": [874, 712],
      "fieldLabels": ["roll1..3"],
      "bubblesGap": 28,
      "labelsGap": 31,
      "direction": "vertical"
    },
    "Class": {
      "origin": [992, 712],
      "fieldLabels": ["class"],
      "bubbleValues": ["3", "4", "5", "6", "7", "8", "9", "10"],
      "bubblesGap": 28,
      "labelsGap": 31,
      "direction": "vertical"
    },
    "Section": {
      "origin": [1075, 712],
      "fieldLabels": ["section"],
      "bubbleValues": ["A", "B", "C", "D", "E", "F", "G", "H"],
      "bubblesGap": 28,
      "labelsGap": 31,
      "direction": "vertical"
    },
    "SchoolCode": {
      "fieldType": "QTYPE_INT",
      "origin": [1165, 712],
      "fieldLabels": ["school1..3"],
      "bubblesGap": 28,
      "labelsGap": 31,
      "direction": "vertical"
    },
    "Answers_1_10": {
      "fieldType": "QTYPE_MCQ4",
      "origin": [161, 1211],
      "fieldLabels": ["q1..10"],
      "bubblesGap": 42,
      "labelsGap": 51,
      "direction": "horizontal"
    },
    "Answers_11_20": {
      "fieldType": "QTYPE_MCQ4",
      "origin": [351, 1211],
      "fieldLabels": ["q11..20"],
      "bubblesGap": 42,
      "labelsGap": 51,
      "direction": "horizontal"
    },
    "Answers_21_30": {
      "fieldType": "QTYPE_MCQ4",
      "origin": [541, 1211],
      "fieldLabels": ["q21..30"],
      "bubblesGap": 42,
      "labelsGap": 51,
      "direction": "horizontal"
    },
    "Answers_31_40": {
      "fieldType": "QTYPE_MCQ4",
      "origin": [731, 1211],
      "fieldLabels": ["q31..40"],
      "bubblesGap": 42,
      "labelsGap": 51,
      "direction": "horizontal"
    },
    "Answers_41_50": {
      "fieldType": "QTYPE_MCQ4",
      "origin": [921, 1211],
      "fieldLabels": ["q41..50"],
      "bubblesGap": 42,
      "labelsGap": 51,
      "direction": "horizontal"
    },
    "Answers_51_60": {
      "fieldType": "QTYPE_MCQ4",
      "origin": [1111, 1211],
      "fieldLabels": ["q51..60"],
      "bubblesGap": 42,
      "labelsGap": 51,
      "direction": "horizontal"
    }
  }
}
```

## Calibration Process

### Tools Used
- **PixSpy**: For measuring pixel coordinates on images
- **Reference Image**: Any filled OMR sheet resized to 1360x1928

### Steps to Recalibrate

1. **Resize image to template dimensions**:
   ```python
   import cv2
   img = cv2.imread("omr_sheet.jpg")
   resized = cv2.resize(img, (1360, 1928))
   cv2.imwrite("calibration_reference.jpg", resized)
   ```

2. **Open in PixSpy** and measure coordinates of:
   - Top-left corner of first bubble (A) in the first name column
   - Top-left corner of first bubble (0) in roll number first digit
   - Same for Class, Section, SchoolCode

3. **Update template.json** with new origin values

### Key Measurements

| Field | What to Measure | Current Value |
|-------|-----------------|---------------|
| CandidateName | First 'A' bubble, column 1 | (105, 387) |
| RollNumber | First '0' bubble, digit 1 | (874, 712) |
| Class | First '3' bubble | (992, 712) |
| Section | First 'A' bubble | (1075, 712) |
| SchoolCode | First '0' bubble, digit 1 | (1165, 712) |

## Verification Results

Tested on 17 sample images from `_test_final`:

| Field | Accuracy | Notes |
|-------|----------|-------|
| Roll Number | 100% | 3-digit format working |
| Class | 100% | Values 3-10 |
| Section | 100% | Values A-H |
| School Code | 100% | 3-digit format working |
| Candidate Name | 70-90% | Minor issues with first letter and spaces |

### Known Name Detection Issues
1. First letter occasionally shifted (e.g., P detected as A)
2. Spurious 'W' or 'M' may appear in empty columns
3. Space detection between first/last name inconsistent

These issues are acceptable for identification purposes.

## File Locations

| File | Path |
|------|------|
| Main Template | `OMRChecker/inputs/VidyaBharti2025/template.json` |
| Test Template | `OMRChecker/inputs/_test_final/template.json` |
| Sample Images | `OMRChecker/inputs/_test_final/*.jpg` |
| Output Results | `OMRChecker/outputs/_test_final_calibrated/Results/` |

## Changelog

| Date | Changes |
|------|---------|
| 2025-12-26 | Initial calibration with PixSpy |
| 2025-12-26 | Updated Roll/Class/Section/SchoolCode origins to y=712 |
| 2025-12-26 | Updated CandidateName origin to [105, 387] |
| 2025-12-26 | Verified 100% accuracy for Roll/Class/Section/SchoolCode |
