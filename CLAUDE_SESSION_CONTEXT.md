# Claude Code Session Context - OMR Name Detection Debugging

**Date:** 2024-12-29
**Issue:** False positive detection of unfilled bubbles in candidate name field

## Problem Statement

File `10447A.jpg` was detecting "NITISH KUMAR PADDEY" but the second D was reportedly not filled by the student.

## Investigation Findings

### Column 16 (Second D Position) Analysis

| Metric | Value | Expected for Filled | Expected for Unfilled |
|--------|-------|---------------------|----------------------|
| Darkness | 181.2 | 120-170 | 200-230 |
| Second-darkest | 205.2 | >215 | >220 |
| Gap to median | 50.6 | >40 | <20 |

**Root Cause:** Print quality variance - the D bubble was darker than normal (181.2) but not truly filled. The high `gap_to_median` (50.6) triggered detection. The second-darkest bubble (E=205.2) was also unusually dark, indicating the entire column had darker print rather than a single filled bubble.

### Column 15 Multi-Fill Issue

Both N (row 13, darkness=164.1) and D (row 3, darkness=190.9) passed the fill threshold. The code picks the **topmost** bubble, selecting D over N even though N was darker.

## Changes Made to `src/ai/yolo_omr_processor.py`

### 1. Added `second_val > 206` Check (Lines 1183, 1199)

```python
# Very light fills need second-darkest to be clearly unfilled
if not is_filled and darkest_val < 205 and gap_to_median > 35 and gap > 20 and second_val > 206:
    is_filled = True
```

This prevents false positives when the entire column has print quality variance.

### 2. Tightened Gap Thresholds (Line 1177)

```python
# Require gap > 25 (not just gap_to_median) for light fills
if not is_filled and darkest_val < 195 and gap_to_median > 40 and gap > 25:
    is_filled = True
```

### Result After Fix

Detection changed from `NITISH KUMAR PADDEY` to `NITISH KUMAR PAD EY` (correctly rejecting the false positive at column 16).

## Aspect Ratio Analysis (Ruled Out)

Tested whether image aspect ratio distortion was causing issues:
- Original images: ~1656 x 2326
- Target resize: 2480 x 3507
- Distortion: **~0.6%** (negligible)

Dynamic scaling test made detection **worse**, confirming aspect ratio is not the issue.

## Key Detection Thresholds

| Darkness Range | Gap to Median | Gap (1st-2nd) | Additional Check | Confidence |
|----------------|---------------|---------------|------------------|------------|
| < 180 | > 30 | - | - | High |
| < 195 | > 40 | > 25 | - | 0.70 |
| < 205 | > 35 | > 20 | second_val > 206 | 0.55 |
| < 195 (name) | > 35 | > 20 | second_val > 206 | 0.55 |

## Files for Testing

- `test_traindata.py` - Random sample testing from traindata folder
- `test_aspect_ratio.py` - Aspect ratio analysis
- `test_dynamic_detection.py` - Dynamic vs fixed bubble detection comparison

## Session 2: 11652B.jpg "SAMIR LOHAR" Detection

**Date:** 2024-12-29
**Issue:** Detected "SAMIR LOHRR" instead of "SAMIR LOHAR"

### Investigation Findings

#### Column 9 (Expected: A, Detected: R)

| Metric | Value |
|--------|-------|
| A darkness | 212.9 |
| R darkness | 183.8 |
| Column median | 231.3 |
| Gap (R to 2nd) | 29.1 |
| Gap to median | 47.6 |

**Finding:** R is 29.1 points DARKER than A. The algorithm correctly picked the darkest bubble.

#### Column 10 (User says R unfilled, but Detected: R)

| Metric | Value |
|--------|-------|
| A darkness | 235.3 (empty) |
| R darkness | 189.4 |
| Column median | 231.2 |
| Gap (R to 2nd) | 32.9 |
| Gap to median | 41.8 |

**Finding:** R has visible dark shading in the scanned image even though user claims it was unfilled.

### Root Cause Analysis

Visual inspection of the debug image (`problem_area_11652B.jpg`) confirms:
- Row R bubbles in columns 9 and 10 have **visible dark marks/shading**
- Row A in column 9 appears **empty/unfilled** (light gray like other empty bubbles)
- Row R is only dark in columns 9 and 10, not across all columns (rules out grid misalignment)

**Conclusion:** The algorithm is working correctly. The scanned image shows marks in the R bubbles that weren't intended as fills. Possible causes:
1. Student accidentally marked R while filling
2. Stains or smudges on paper
3. Bleed-through from back of sheet
4. Pencil marks from erasing

### Changes Made (Session 2)

1. **Multi-fill logic improved** (lines 1106-1130): If darkness_gap > 10, pick darkest; else pick topmost
2. **Fill threshold tightened**: `min(median_val - 45, 190)` for multi-fill detection
3. **Name field threshold**: Added `gap_to_median >= 37` for name field with print quality variance

### Key Insight

Not all detection "errors" are algorithm bugs. Some are legitimate marks on the physical OMR sheet that the algorithm correctly detects. These cases require:
- Manual review interface for operators
- Stricter thresholds (trade-off: may miss light fills)
- Image preprocessing to reduce bleed-through

## Test Files

- `test_traindata.py` - Random sample testing from `/Volumes/Projects/traindata/`
- `test_aspect_ratio.py` - Aspect ratio analysis
- `test_dynamic_detection.py` - Dynamic vs fixed bubble detection
- `test_origin_adjustment.py` - Grid origin calibration
- `analyze_lohar_issue.py` - Column 9/10 analysis for 11652B
- `analyze_row_pattern.py` - Row pattern analysis
- `visualize_problem_bubbles.py` - Visual debugging

## Debug Images

- `problem_area_11652B.jpg` - Annotated problem columns 9-10
- `debug_row_pattern.jpg` - Row R pattern across columns
- `debug_lohar_cols.jpg` - Column analysis visualization
