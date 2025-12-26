"""Test image alignment with reference."""
import cv2
import numpy as np
from pathlib import Path


def align_with_reference(image_path: str, reference_path: str):
    """Align an image using feature matching with reference."""
    # Load images
    img = cv2.imread(image_path)
    ref = cv2.imread(reference_path)

    if img is None or ref is None:
        print("Failed to load images")
        return None

    print(f"Input image: {img.shape}")
    print(f"Reference image: {ref.shape}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    # Detect ORB features
    orb = cv2.ORB_create(3000)
    kp1, des1 = orb.detectAndCompute(gray, None)
    kp2, des2 = orb.detectAndCompute(ref_gray, None)

    print(f"Features detected - Input: {len(kp1)}, Reference: {len(kp2)}")

    if des1 is None or des2 is None:
        print("No features detected")
        return None

    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Use top 15% matches
    good_matches = matches[:int(len(matches) * 0.15)]
    print(f"Good matches: {len(good_matches)}")

    if len(good_matches) < 10:
        print("Not enough good matches")
        return None

    # Extract matched points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        print("Could not compute homography")
        return None

    # Warp image to match reference
    h, w = ref_gray.shape
    aligned = cv2.warpPerspective(img, H, (w, h))

    return aligned


def test_alignment():
    """Test alignment on sample images."""
    reference = "OMRChecker/inputs/VidyaBharti2025/omr-filled.png"
    test_image = "filledomrsheet/41_624.jpg"

    print("Testing alignment...")
    print("=" * 60)

    aligned = align_with_reference(test_image, reference)

    if aligned is not None:
        cv2.imwrite("aligned_result.jpg", aligned)
        print(f"\nAligned image saved to aligned_result.jpg")
        print(f"Aligned image shape: {aligned.shape}")

        # Now test bubble detection on aligned image
        print("\nTesting name detection on aligned image...")

        # Name grid parameters
        origin_x, origin_y = 193, 690
        rows_gap, cols_gap = 51, 54
        bubble_w, bubble_h = 28, 28

        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

        detected_name = []
        for col in range(12):  # First 12 columns
            x = origin_x + col * cols_gap

            bubble_values = []
            for row in range(26):
                y = origin_y + row * rows_gap
                letter = chr(65 + row)

                if 0 <= y < gray.shape[0] - bubble_h and 0 <= x < gray.shape[1] - bubble_w:
                    bubble = gray[y:y+bubble_h, x:x+bubble_w]
                    darkness = np.mean(bubble)
                    bubble_values.append((letter, darkness))

            if bubble_values:
                darkest = min(bubble_values, key=lambda x: x[1])
                vals = [v[1] for v in bubble_values]
                median_val = np.median(vals)

                if darkest[1] < median_val * 0.75:
                    detected_name.append(darkest[0])
                else:
                    detected_name.append(' ')

        name = ''.join(detected_name).strip()
        name = ' '.join(name.split())
        print(f"Detected name: '{name}'")
        print(f"Expected: 'PARI KUMARI'")
    else:
        print("Alignment failed!")


if __name__ == "__main__":
    test_alignment()
