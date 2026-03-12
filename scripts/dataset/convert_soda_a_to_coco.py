#!/usr/bin/env python3
"""
Convert SODA_A per-image JSON format to single COCO format annotation file.

SODA_A has individual JSON files per image in Annotations/train/, test/, val/.
Each JSON contains:
- images: dict with file_name, height, width, id
- annotations: list with poly (polygon), area, category_id, image_id, id
- categories: list of category dicts

This script converts them to standard COCO format with:
- images: list of image dicts
- annotations: list with bbox (from polygon bounding box), area, category_id, image_id, id
- categories: list of category dicts
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm


def polygon_to_bbox(poly: List[float]) -> Tuple[float, float, float, float]:
    """
    Convert polygon to COCO bbox format [x, y, width, height].
    
    Args:
        poly: List of coordinates [x1, y1, x2, y2, x3, y3, ...]
              or flattened list of [x, y] pairs
    
    Returns:
        Tuple of (x, y, width, height) - top-left corner and size
    """
    if len(poly) < 4:
        raise ValueError(f"Polygon must have at least 2 points, got {len(poly)} coordinates")
    
    # Extract x and y coordinates
    # Polygon format: [x1, y1, x2, y2, x3, y3, ...]
    x_coords = poly[0::2]  # Every even index (0, 2, 4, ...)
    y_coords = poly[1::2]  # Every odd index (1, 3, 5, ...)
    
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)
    
    # Ensure non-negative dimensions
    x = float(max(0, x_min))
    y = float(max(0, y_min))
    w = float(max(0, x_max - x_min))
    h = float(max(0, y_max - y_min))
    
    return (x, y, w, h)


def load_per_image_annotations(annotations_dir: Path) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load all per-image JSON files and convert to COCO format.
    
    This function ensures reproducibility by:
    - Sorting input files by filename (deterministic order)
    - Sorting output images and annotations by ID
    
    Args:
        annotations_dir: Directory containing per-image JSON files
    
    Returns:
        Tuple of (images, annotations, categories) - all sorted by ID for reproducibility
    """
    # Sort files by name for reproducibility (ensures consistent processing order)
    json_files = sorted(annotations_dir.glob("*.json"), key=lambda p: p.name)
    
    if not json_files:
        raise ValueError(f"No JSON files found in {annotations_dir}")
    
    print(f"Found {len(json_files)} annotation files in {annotations_dir}")
    
    all_images = []
    all_annotations = []
    all_categories = None
    next_image_id = 1
    next_ann_id = 1
    
    for json_file in tqdm(json_files, desc="Processing annotation files"):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Get image info (it's a dict, not a list)
        img_info = data['images']
        img_id = img_info.get('id', next_image_id)
        
        # Create image entry
        image_entry = {
            'id': img_id,
            'file_name': img_info['file_name'],
            'width': img_info['width'],
            'height': img_info['height']
        }
        all_images.append(image_entry)
        
        # Get categories (should be same for all files, but we'll take the first one)
        if all_categories is None and 'categories' in data:
            all_categories = data['categories']
        
        # Convert annotations
        for ann in data.get('annotations', []):
            # Convert polygon to bbox
            poly = ann['poly']
            x, y, w, h = polygon_to_bbox(poly)
            
            # Calculate area if not provided or is None
            area = ann.get('area')
            if area is None:
                area = w * h  # Calculate from bbox dimensions
            
            # Create annotation entry
            ann_entry = {
                'id': next_ann_id,
                'image_id': img_id,
                'category_id': ann['category_id'],
                'bbox': [x, y, w, h],  # COCO format: [x, y, width, height]
                'area': float(area),
                'iscrowd': 0
            }
            all_annotations.append(ann_entry)
            next_ann_id += 1
        
        next_image_id = max(next_image_id, img_id + 1)
    
    # Ensure categories exist
    if all_categories is None:
        # Try to load from first file
        with open(json_files[0], 'r') as f:
            data = json.load(f)
            all_categories = data.get('categories', [])
    
    if not all_categories:
        print("Warning: No categories found. Creating default categories.")
        all_categories = [{'id': i, 'name': f'class_{i}', 'supercategory': 'object'} for i in range(10)]
    
    return all_images, all_annotations, all_categories


def create_coco_dataset(
    annotations_dir: Path,
    output_file: Path,
    info: Dict = None
) -> Dict:
    """
    Create COCO format dataset from per-image JSON files.
    
    Args:
        annotations_dir: Directory containing per-image JSON files
        output_file: Path to output COCO JSON file
        info: Optional info dict for COCO format
    
    Returns:
        COCO format dict
    """
    images, annotations, categories = load_per_image_annotations(annotations_dir)
    
    # Sort images and annotations by ID for reproducibility
    images = sorted(images, key=lambda img: img['id'])
    annotations = sorted(annotations, key=lambda ann: ann['id'])
    
    coco_data = {
        'info': info or {
            'description': 'SODA_A Dataset',
            'version': '1.0',
            'year': 2024
        },
        'licenses': [],
        'categories': sorted(categories, key=lambda cat: cat['id']),  # Sort categories by ID
        'images': images,
        'annotations': annotations
    }
    
    # Save to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2, sort_keys=False)
    
    print(f"\n✓ Created COCO annotation file: {output_file}")
    print(f"  Images: {len(images)}")
    print(f"  Annotations: {len(annotations)}")
    print(f"  Categories: {len(categories)}")
    
    return coco_data


def main():
    parser = argparse.ArgumentParser(
        description='Convert SODA_A per-image JSON format to single COCO format annotation file'
    )
    parser.add_argument(
        'annotations_dir',
        type=str,
        help='Directory containing per-image JSON annotation files (e.g., Annotations/train/)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='Output COCO annotation file path (e.g., train/_annotations.coco.json)'
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test'],
        help='Split name (for info field)'
    )
    
    args = parser.parse_args()
    
    annotations_dir = Path(args.annotations_dir)
    if not annotations_dir.exists():
        print(f"Error: Annotations directory not found: {annotations_dir}")
        return 1
    
    output_file = Path(args.output_file)
    
    info = {
        'description': f'SODA_A Dataset - {args.split or "unknown"} split',
        'version': '1.0',
        'year': 2024
    }
    
    print("="*60)
    print("SODA_A to COCO Converter")
    print("="*60)
    print(f"Input directory: {annotations_dir}")
    print(f"Output file: {output_file}")
    print("="*60)
    print()
    
    try:
        create_coco_dataset(annotations_dir, output_file, info)
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
