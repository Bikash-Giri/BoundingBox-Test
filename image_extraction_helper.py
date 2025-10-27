import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def draw_bounding_box(image_path, coordinates, 
                      color=(0, 255, 0), thickness=2, label=None,class_label = "0"):
    """
    Draw bounding box on an image.
    
    Parameters:
    - image_path: Path to the input image
    - coordinates: Tuple of (x1, y1, x2, y2) where (x1,y1) is top-left and (x2,y2) is bottom-right
                   OR list of such tuples for multiple boxes

    - color: BGR color tuple for the box (default: green)
    - thickness: Line thickness in pixels
    - label: Optional text label to display above the box
    """
    
    # Read the image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Handle single coordinate tuple or list of coordinates
    if not coordinates:
        print("Warning: No coordinates provided, skipping bounding box drawing")
        return img
    
    if isinstance(coordinates[0], (int, float)):
        coordinates = [coordinates]
    
    # Draw each bounding box
    for i, coords in enumerate(coordinates):
        x1, y1, x2, y2 = map(int, coords)
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Add label if provided
        if label:
            current_label = label if isinstance(label, str) else label[i]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                current_label, font, font_scale, font_thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(img, 
                         (x1, y1 - text_height - 10), 
                         (x1 + text_width + 10, y1), 
                         color, -1)
            
            
            # Put text
            cv2.putText(img, current_label, (x1 + 5, y1 - 5), 
                       font, font_scale, (255, 255, 255), font_thickness)

    
    # Save the output image
   
    output_dir_boundingbox = Path("outputs/bounding_box")           # directory name
    output_dir_boundingbox.mkdir(parents=True, exist_ok=True)  # create it if it doesn't exist

    output_dir_patch = Path("outputs/patch")           # directory name
    output_dir_patch.mkdir(parents=True, exist_ok=True)  # create it if it doesn't exist

    # Create a proper filename with extension
    input_filename = Path(image_path).stem
    output_filename = f"{input_filename}_with_bbox.jpg"
    
    full_output_path = output_dir_boundingbox / output_filename
    
    full_output_patch_path = output_dir_patch / output_filename





    
    print(f"Saving to: {full_output_path}")
    cv2.imwrite(str(full_output_path), img)


    cropped = img[y1:y2, x1:x2]
    cv2.imwrite(str(full_output_patch_path), cropped)


    print(f"Image saved to {full_output_path}")
    
    return img


# Example usage
if __name__ == "__main__":
    image_path = "mouse_pic.png"
    # This is just a example file don't run this file
    # Coordinates format: (x1, y1, x2, y2)
    # (x1, y1) = top-left corner
    # (x2, y2) = bottom-right corner
    bbox_coordinates = (832.28, 753.57, 1053.36,845.43)
    
    draw_bounding_box(
        image_path=image_path,
        coordinates=bbox_coordinates,
        color=(0, 255, 0),  # Green in BGR
        thickness=3,
        label="Tail-Patches",
        class_label= "0"
    )



     