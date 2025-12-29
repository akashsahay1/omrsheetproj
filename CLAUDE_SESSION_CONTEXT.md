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

## Next Steps (If Needed)

1. Further tune the `second_val` threshold (currently 206)
2. Investigate multi-fill logic at column 15 (N vs D selection)
3. Consider per-column print quality normalization
