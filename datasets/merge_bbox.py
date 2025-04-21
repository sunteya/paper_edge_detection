import os
import shutil
from pathlib import Path
from typing import Dict, Any

from merge_utils import (
    create_output_directories,
    process_source_directory,
    print_merge_statistics
)

def convert_coco_to_yolo(annotations: Dict[str, Any], image_dir: str, output_dir: str, dataset_name: str):
    """
    Convert COCO format annotations to YOLO format and process images.
    
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
        with open(yolo_file, 'w') as f:
            for ann in img_annotations:
                category_id = ann['category_id']
                bbox = ann['bbox']  # [x, y, width, height]
                
                # Convert COCO bbox to YOLO format (normalized center x, center y, width, height)
                x_center = (bbox[0] + bbox[2] / 2) / image_width
                y_center = (bbox[1] + bbox[3] / 2) / image_height
                width = bbox[2] / image_width
                height = bbox[3] / image_height
                
                # Write YOLO format line: class_id x_center y_center width height
                f.write(f"{category_id} {x_center} {y_center} {width} {height}\n")
        
        # Only copy image if we successfully wrote annotations and image doesn't exist
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

def main():
    source_names = [
        "tmayolov8_four-corners-detection",
        "greg-sun_a4-detection",
        "unochapeco_paper-s9top",
        "test1-w504r_-2kpgu"
    ]
    
    base_dir = Path(__file__).parent
    source_dirs = [base_dir / dir_name for dir_name in source_names]
    merged_dir = base_dir / "merged_bbox"
    
    merged_dir = Path(merged_dir)
    create_output_directories(merged_dir)
    
    for source_dir in source_dirs:
        process_source_directory(source_dir, merged_dir, convert_coco_to_yolo)
    
    print_merge_statistics(merged_dir)

if __name__ == "__main__":
    main()