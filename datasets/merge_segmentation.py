import os
import shutil
from pathlib import Path
from typing import Dict, Any

from merge_utils import (
    create_output_directories,
    process_source_directory,
    print_merge_statistics
)

def convert_coco_to_yolo_segmentation(annotations: Dict[str, Any], image_dir: str, output_dir: str, dataset_name: str):
    """
    Convert COCO format annotations to YOLO segmentation format and process images.
    
    Args:
        annotations: COCO format annotations
        image_dir: Source directory containing images
        output_dir: Target directory for converted dataset
        dataset_name: Name of the dataset (used as prefix for filenames)
    """
    # Process each image
    for image in annotations['images']:
        image_id = image['id']
        image_name = image['file_name']
        image_width = image['width']
        image_height = image['height']
        
        # Generate new filename with dataset prefix
        new_filename = f"{dataset_name}_{image_name}"
        
        # Get annotations for this image
        img_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
        
        # Skip if no annotations
        if not img_annotations:
            continue
            
        # Create YOLO format annotation file
        yolo_file = os.path.join(output_dir, 'labels', os.path.splitext(new_filename)[0] + '.txt')
        
        # Check if label file already exists
        if os.path.exists(yolo_file):
            continue
            
        # Write annotations to file
        has_valid_annotations = False
        with open(yolo_file, 'w') as f:
            for ann in img_annotations:
                # Skip if no segmentation data or empty segmentation list
                if 'segmentation' not in ann or not ann['segmentation'] or not ann['segmentation'][0]:
                    continue
                    
                # Convert category_id to 0-based index
                category_id = 0  # Always use 0 as the class ID
                segmentation = ann['segmentation'][0]  # Take first polygon if multiple exist
                
                # Skip if segmentation points are not in pairs
                if len(segmentation) % 2 != 0:
                    continue
                
                # Check if the last point is the same as the first point (closed polygon)
                if len(segmentation) >= 4 and segmentation[0] == segmentation[-2] and segmentation[1] == segmentation[-1]:
                    # Remove the last point (last two values) as it's redundant
                    segmentation = segmentation[:-2]
                
                # Convert COCO segmentation to YOLO format (normalized points)
                normalized_points = []
                for i in range(0, len(segmentation), 2):
                    x = segmentation[i] / image_width
                    y = segmentation[i + 1] / image_height
                    normalized_points.extend([x, y])
                
                # Check if the shape is close to a rectangle
                if len(normalized_points) == 8:  # Only check for 4-point polygons
                    # Calculate angles between consecutive edges
                    angles = []
                    for i in range(4):
                        x1, y1 = normalized_points[i*2], normalized_points[i*2+1]
                        x2, y2 = normalized_points[(i*2+2)%8], normalized_points[(i*2+3)%8]
                        x3, y3 = normalized_points[(i*2+4)%8], normalized_points[(i*2+5)%8]
                        
                        # Calculate vectors
                        v1 = (x2-x1, y2-y1)
                        v2 = (x3-x2, y3-y2)
                        
                        # Calculate angle between vectors
                        dot_product = v1[0]*v2[0] + v1[1]*v2[1]
                        norm1 = (v1[0]**2 + v1[1]**2)**0.5
                        norm2 = (v2[0]**2 + v2[1]**2)**0.5
                        angle = abs(dot_product / (norm1 * norm2))
                        angles.append(angle)
                    
                    # Check if the shape is too close to a rectangle
                    is_rectangle = True
                    
                    max_angle_deviation = 0.001
                    for angle in angles:
                        if abs(angle) > max_angle_deviation:
                            is_rectangle = False
                            break
                    
                    # Debug information
                    if is_rectangle:
                        print(f"Skipping rectangle-like shape in {image_name}:")
                        print(f"  Angles: {[f'{angle:.4f}' for angle in angles]}")
                    
                    # Only write if it's not a rectangle
                    if not is_rectangle:
                        f.write(f"{category_id} {' '.join(map(str, normalized_points))}\n")
                        has_valid_annotations = True
                else:
                    # Write non-rectangular shapes
                    f.write(f"{category_id} {' '.join(map(str, normalized_points))}\n")
                    has_valid_annotations = True
        
        # Only copy image if we successfully wrote annotations and image doesn't exist
        if has_valid_annotations:
            dst_image = os.path.join(output_dir, 'images', new_filename)
            if not os.path.exists(dst_image):
                # Search for the image in all possible split directories
                src_image = None
                for split_name in ['train', 'valid', 'test']:
                    potential_path = os.path.join(image_dir, split_name, image_name)
                    if os.path.exists(potential_path):
                        src_image = potential_path
                        break
                
                if src_image and os.path.exists(src_image):
                    shutil.copy2(src_image, dst_image)
                else:
                    print(f"Warning: Image file not found: {image_name}")
                    # Remove the label file if image doesn't exist
                    if os.path.exists(yolo_file):
                        os.remove(yolo_file)
        else:
            # Remove the label file if no valid annotations were written
            if os.path.exists(yolo_file):
                os.remove(yolo_file)

def main():
    source_names = [
        "tmayolov8_four-corners-detection",
        "greg-sun_a4-detection",
        "unochapeco_paper-s9top",
        "test1-w504r_-2kpgu"
    ]
    
    base_dir = Path(__file__).parent
    source_dirs = [base_dir / dir_name for dir_name in source_names]
    merged_dir = base_dir / "merged_segmentation"
    
    merged_dir = Path(merged_dir)
    create_output_directories(merged_dir)
    
    for source_dir in source_dirs:
        process_source_directory(source_dir, merged_dir, convert_coco_to_yolo_segmentation)
    
    print_merge_statistics(merged_dir)

if __name__ == "__main__":
    main()