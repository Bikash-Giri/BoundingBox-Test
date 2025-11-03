import cv2
import numpy as np

# Load image
image = cv2.imread('outputs/patch/frames_EBJ24070143_20241017_225329.341624.mp4_57000_with_bbox.jpg', cv2.IMREAD_GRAYSCALE)

# Threshold to binary
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

# Apply morphological closing to connect nearby components
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Optionally apply dilation to further merge components
closed = cv2.erode(closed, kernel, iterations=1)

# Now count connected components
num_labels, labels = cv2.connectedComponents(closed)
num_patches = num_labels - 1
print(f"Number of black patches: {num_patches}")

# Visualize to debug
cv2.imshow('Original Binary', binary)
cv2.imshow('After Closing', closed)
cv2.waitKey(0)


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Load image
# image = cv2.imread('outputs/patch/frames_EBJ24070143_20241017_222329.337195.mp4_11400_with_bbox.jpg', cv2.IMREAD_GRAYSCALE)

# # Threshold to binary
# _, binary = cv2.threshold(image, 0, 200, cv2.THRESH_BINARY_INV)

# # Apply morphological closing to connect nearby components
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
# closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# # Get connected components with stats
# num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed)
# print(num_labels)

# # Image dimensions
# height, width = image.shape

# # Count patches that don't touch left or right edges
# valid_patches = 0
# valid_patch_ids = []

# for label_id in range(1, num_labels):  # Skip background (0)
#     x, y, w, h, area = stats[label_id]
    
#     # Check if patch touches left edge (x == 0) or right edge (x + w >= width)
#     touches_left = (x == 0)
#     touches_right = (x + w >= width)
    
#     if not touches_left and not touches_right:
#         valid_patches += 1
#         valid_patch_ids.append(label_id)
#         print(f"✓ Patch {label_id}: Valid (has white on both sides)")
#         print(f"  Position: x={x}, right_edge={x+w}, width={width}")
#     else:
#         print(f"✗ Patch {label_id}: Touches edge (left={touches_left}, right={touches_right})")

# print(f"\n{'='*50}")
# print(f"Total patches found: {num_labels - 1}")
# print(f"Valid patches (white on both sides): {valid_patches}")
# print(f"{'='*50}")

# # Visualization
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# # Original
# axes[0].imshow(image, cmap='gray')
# axes[0].set_title('Original Image')
# axes[0].axis('off')

# # All patches (colored)
# colored_all = np.zeros((*labels.shape, 3), dtype=np.uint8)
# colors = plt.cm.tab20(np.linspace(0, 1, num_labels))[:, :3] * 255

# for label_id in range(1, num_labels):
#     mask = labels == label_id
#     colored_all[mask] = colors[label_id]

# axes[1].imshow(colored_all)
# axes[1].set_title(f'All Patches ({num_labels-1} total)')
# axes[1].axis('off')

# # Only valid patches (highlighted)
# colored_valid = image.copy()
# colored_valid = cv2.cvtColor(colored_valid, cv2.COLOR_GRAY2BGR)

# for label_id in valid_patch_ids:
#     mask = labels == label_id
#     colored_valid[mask] = [0, 255, 0]  # Green for valid patches
    
#     # Draw bounding box
#     x, y, w, h, area = stats[label_id]
#     cv2.rectangle(colored_valid, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#     # Add label
#     cx, cy = centroids[label_id]
#     cv2.putText(colored_valid, str(label_id), (int(cx)-10, int(cy)), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# axes[2].imshow(cv2.cvtColor(colored_valid, cv2.COLOR_BGR2RGB))
# axes[2].set_title(f'Valid Patches Only ({valid_patches} patches)')
# axes[2].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Left edge')
# axes[2].axvline(x=width-1, color='red', linestyle='--', linewidth=2, label='Right edge')
# axes[2].legend()
# axes[2].axis('off')

# plt.tight_layout()
# plt.savefig('valid_patches_visualization.png', dpi=150, bbox_inches='tight')
# plt.show()