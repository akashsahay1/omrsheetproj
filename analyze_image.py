"""Analyze the actual filled bubbles in the image to understand the pattern."""
import cv2
import numpy as np

# Load and resize to template dimensions
img = cv2.imread("filledomrsheet/41_624.jpg")
template_width, template_height = 2480, 3507
img = cv2.resize(img, (template_width, template_height))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find ALL very dark regions (filled bubbles) in the name area
print("Finding all filled bubbles in name area...")
print("="*60)

# Define expected grid parameters
bubble_w, bubble_h = 28, 28

# Collect all filled bubble positions
filled_bubbles = []

# Scan the entire name area
for y in range(600, 2100, 5):
    for x in range(150, 1500, 5):
        if y + bubble_h < template_height and x + bubble_w < template_width:
            bubble = gray[y:y+bubble_h, x:x+bubble_w]
            darkness = np.mean(bubble)
            if darkness < 50:  # Very dark = definitely filled
                filled_bubbles.append((x, y, darkness))

print(f"Found {len(filled_bubbles)} filled bubble regions")

# Cluster by proximity
def cluster_points(points, max_dist=30):
    clusters = []
    used = set()

    for i, (x1, y1, d1) in enumerate(points):
        if i in used:
            continue

        cluster = [(x1, y1, d1)]
        used.add(i)

        for j, (x2, y2, d2) in enumerate(points):
            if j not in used:
                if abs(x1 - x2) < max_dist and abs(y1 - y2) < max_dist:
                    cluster.append((x2, y2, d2))
                    used.add(j)

        # Use the darkest point as cluster center
        center = min(cluster, key=lambda p: p[2])
        clusters.append(center)

    return clusters

clusters = cluster_points(filled_bubbles)
print(f"Clustered into {len(clusters)} distinct bubbles")

# Sort by x then y to understand the grid
clusters.sort(key=lambda c: (c[0], c[1]))

print("\nFilled bubbles by position:")
for x, y, d in clusters:
    print(f"  x={x:4d}, y={y:4d}, darkness={d:.1f}")

# Group by columns (similar x)
print("\n" + "="*60)
print("Grouping by columns...")

columns = []
current_col = [clusters[0]]
for c in clusters[1:]:
    if abs(c[0] - current_col[0][0]) < 40:
        current_col.append(c)
    else:
        columns.append(current_col)
        current_col = [c]
columns.append(current_col)

print(f"Found {len(columns)} columns with filled bubbles")
for i, col in enumerate(columns):
    col.sort(key=lambda c: c[1])
    x_avg = np.mean([c[0] for c in col])
    y_positions = [c[1] for c in col]
    print(f"\nColumn {i+1} (x~{x_avg:.0f}):")
    for x, y, d in col:
        print(f"  y={y:4d}, darkness={d:.1f}")

# Calculate column gaps
if len(columns) >= 2:
    col_positions = [np.mean([c[0] for c in col]) for col in columns]
    col_gaps = [col_positions[i+1] - col_positions[i] for i in range(len(col_positions)-1)]
    print(f"\nColumn gaps: {[f'{g:.0f}' for g in col_gaps]}")
    print(f"Average column gap (labels_gap): {np.mean(col_gaps):.1f}")

# Calculate row gaps within columns
print("\n" + "="*60)
print("Analyzing row gaps...")
for i, col in enumerate(columns):
    if len(col) >= 2:
        col.sort(key=lambda c: c[1])
        row_gaps = [col[j+1][1] - col[j][1] for j in range(len(col)-1)]
        if row_gaps:
            print(f"Column {i+1} row gaps: {row_gaps}")

# Create visualization
output = img.copy()
for x, y, d in clusters:
    cv2.circle(output, (x + bubble_w//2, y + bubble_h//2), 15, (0, 0, 255), 3)
    cv2.putText(output, f"{int(d)}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

# Crop name area
name_area = output[550:2200, 100:1600]
cv2.imwrite("filled_bubbles_analysis.jpg", name_area)
print("\nSaved filled_bubbles_analysis.jpg")
