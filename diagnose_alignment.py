#!/usr/bin/env python3
"""Diagnose YOLO processor alignment by visualizing grid positions."""

import sys
sys.path.insert(0, '/Volumes/Projects/Projects/omrsheetproj')

import cv2
import numpy as np
from pathlib import Path
from src.ai.yolo_omr_processor import YOLOOMRProcessor

def visualize_grid(processor, image_path, output_path):
    """Draw the grid positions on an image to check alignment."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load {image_path}")
        return
    
    # Align image
    aligned = processor._align_image(image)
    
    # Draw name field grid (first 10 columns only)
    config = processor.FIELD_CONFIGS['name']
    origin = config['origin']
    rows_gap = config['rows_gap']
    cols_gap = config['cols_gap']
    bubble_size = config['bubble_size']
    
    # Draw grid lines and circles for first 10 columns
    for col in range(10):
        x = origin[0] + col * cols_gap
        for row in range(26):  # A-Z
            y = origin[1] + row * rows_gap
            bw, bh = bubble_size
            
            # Draw expected bubble location
            cv2.rectangle(aligned, (x, y), (x+bw, y+bh), (0, 255, 0), 1)
            
            # Check if bubble appears filled (dark)
            if y < aligned.shape[0] - bh and x < aligned.shape[1] - bw:
                gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
                bubble = gray[y:y+bh, x:x+bw]
                darkness = np.mean(bubble)
                median = 200  # Approximate
                
                if darkness < median * 0.75:
                    # Mark as potentially filled
                    cv2.rectangle(aligned, (x, y), (x+bw, y+bh), (0, 0, 255), 2)
    
    # Add labels
    cv2.putText(aligned, f"Name Grid (first 10 cols)", (origin[0], origin[1]-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(aligned, f"Green=expected position, Red=detected filled", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    cv2.imwrite(str(output_path), aligned)
    print(f"Saved diagnostic image to: {output_path}")
    
    return aligned.shape

def main():
    reference_path = "/Volumes/Projects/Projects/omrsheetproj/OMRChecker/inputs/VidyaBharti2025/omr-filled.png"
    processor = YOLOOMRProcessor(reference_image_path=reference_path)
    
    test_dir = Path('/Volumes/Projects/Projects/omrsheetproj/filledomrsheet')
    test_image = test_dir / '41_625.jpg'  # ASHIVANSH KUMAR
    
    output_path = Path('/Volumes/Projects/Projects/omrsheetproj/alignment_diagnostic.jpg')
    
    shape = visualize_grid(processor, test_image, output_path)
    print(f"Aligned image shape: {shape}")
    
if __name__ == "__main__":
    main()
