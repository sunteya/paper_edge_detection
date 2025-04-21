import os
import shutil
import json
import random
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv

# Dataset split names and ratios
DATASET_SPLITS = ['train', 'valid', 'test']
SPLIT_RATIOS = {'train': 0.8, 'valid': 0.1, 'test': 0.1}

# Get API key from environment variable
def read_coco_annotations(file_path: str) -> Dict[str, Any]:
    """Read COCO format annotations from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_output_directories(merged_dir: Path):
    """Create necessary output directories for merged dataset."""
    for split_name in DATASET_SPLITS:
        split_dir = merged_dir / split_name
        (split_dir / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'labels').mkdir(parents=True, exist_ok=True)

def process_source_directory(source_dir: Path, merged_dir: Path, convert_func):
    """
    Process a single source directory and merge its contents.
    
    Args:
        source_dir: Path to source dataset directory
        merged_dir: Path to merged dataset directory
        convert_func: Function to convert COCO annotations to YOLO format and process images
    """
    print(f"Processing {source_dir.name}...")
    
    # Create a temporary directory to store all processed data
    temp_dir = merged_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / 'images').mkdir(parents=True, exist_ok=True)
    (temp_dir / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Process all data first
    for split_name in DATASET_SPLITS:
        split_dir = source_dir / split_name
        if not split_dir.exists():
            continue
            
        print(f"  Processing {split_name} set...")
        
        # Check for COCO annotation file in this split
        coco_file = split_dir / "_annotations.coco.json"
        if not coco_file.exists():
            print(f"Warning: No COCO annotation file found in {split_dir}")
            continue
            
        # Read annotations
        annotations = read_coco_annotations(coco_file)
        
        # Convert and process this split
        convert_func(
            annotations,
            source_dir,  # Pass the source directory for image access
            temp_dir,
            source_dir.name
        )
    
    # Get all processed images
    processed_images = list((temp_dir / 'images').glob('*'))
    if not processed_images:
        print(f"Warning: No valid images found in {source_dir}")
        shutil.rmtree(temp_dir)
        return
    
    # Shuffle the processed images
    random.shuffle(processed_images)
    
    # Calculate split indices
    total_images = len(processed_images)
    train_end = int(total_images * SPLIT_RATIOS['train'])
    val_end = train_end + int(total_images * SPLIT_RATIOS['valid'])
    
    # Split the data
    splits = {
        'train': (0, train_end),
        'valid': (train_end, val_end),
        'test': (val_end, total_images)
    }
    
    # Move files to their respective split directories
    for split_name, (start_idx, end_idx) in splits.items():
        split_images = processed_images[start_idx:end_idx]
        for image_path in split_images:
            # Move image
            dst_image = merged_dir / split_name / 'images' / image_path.name
            shutil.move(str(image_path), str(dst_image))
            
            # Move corresponding label
            label_path = temp_dir / 'labels' / f"{image_path.stem}.txt"
            if label_path.exists():
                dst_label = merged_dir / split_name / 'labels' / f"{image_path.stem}.txt"
                shutil.move(str(label_path), str(dst_label))
    
    # Clean up temporary directory
    shutil.rmtree(temp_dir)

def print_merge_statistics(merged_dir: Path):
    """Print statistics about the merged dataset."""
    print("\nDataset merging completed!")
    for split_name in DATASET_SPLITS:
        split_dir = merged_dir / split_name
        images_count = len(list((split_dir / 'images').glob('*')))
        labels_count = len(list((split_dir / 'labels').glob('*')))
        print(f"{split_name.capitalize()} set: {images_count} images, {labels_count} labels") 