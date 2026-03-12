#!/usr/bin/env python3
"""
Script to split high-resolution images into smaller patches while preserving COCO annotations.

This script:
1. Splits images into overlapping or non-overlapping patches
2. Adjusts bounding box coordinates to match patch locations
3. Creates new COCO annotation files for the patched dataset
4. Handles objects that span multiple patches
"""

import json
import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil
from tqdm import tqdm


def load_coco_annotations(annotation_path: Path) -> Dict:
    """Load COCO format annotations."""
    with open(annotation_path, 'r') as f:
        return json.load(f)


def save_coco_annotations(data: Dict, output_path: Path):
    """Save COCO format annotations."""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved annotations to: {output_path}")


def calculate_patch_coordinates(
    img_width: int,
    img_height: int,
    patch_size: Tuple[int, int],
    overlap: float = 0.0
) -> List[Tuple[int, int, int, int]]:
    """
    Calculate patch coordinates for an image.
    
    Args:
        img_width: Image width
        img_height: Image height
        patch_size: (patch_width, patch_height)
        overlap: Overlap ratio (0.0 = no overlap, 0.25 = 25% overlap)
    
    Returns:
        List of (x, y, width, height) for each patch
    """
    patch_w, patch_h = patch_size
    step_w = int(patch_w * (1 - overlap))
    step_h = int(patch_h * (1 - overlap))
    
    patches = []
    y = 0
    while y < img_height:
        x = 0
        while x < img_width:
            # Calculate actual patch size (may be smaller at edges)
            actual_w = min(patch_w, img_width - x)
            actual_h = min(patch_h, img_height - y)
            
            patches.append((x, y, actual_w, actual_h))
            x += step_w
            if x >= img_width:
                break
        y += step_h
        if y >= img_height:
            break
    
    return patches


def bbox_in_patch(
    bbox: List[float],
    patch_x: int,
    patch_y: int,
    patch_w: int,
    patch_h: int,
    min_overlap: float = 0.5
) -> Optional[List[float]]:
    """
    Check if a bbox overlaps with a patch and return adjusted coordinates.
    
    Args:
        bbox: COCO format [x, y, width, height]
        patch_x, patch_y: Patch top-left corner
        patch_w, patch_h: Patch dimensions
        min_overlap: Minimum overlap ratio to include bbox (0.0-1.0)
    
    Returns:
        Adjusted bbox in patch coordinates [x, y, width, height] or None
    """
    bx, by, bw, bh = bbox
    bbox_x2 = bx + bw
    bbox_y2 = by + bh
    patch_x2 = patch_x + patch_w
    patch_y2 = patch_y + patch_h
    
    # Calculate intersection
    inter_x1 = max(bx, patch_x)
    inter_y1 = max(by, patch_y)
    inter_x2 = min(bbox_x2, patch_x2)
    inter_y2 = min(bbox_y2, patch_y2)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return None  # No overlap
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    bbox_area = bw * bh
    
    # Check if overlap is sufficient
    overlap_ratio = inter_area / bbox_area if bbox_area > 0 else 0
    if overlap_ratio < min_overlap:
        return None
    
    # Adjust coordinates to patch-relative
    new_x = inter_x1 - patch_x
    new_y = inter_y1 - patch_y
    new_w = inter_x2 - inter_x1
    new_h = inter_y2 - inter_y1
    
    return [new_x, new_y, new_w, new_h]


def split_image_to_patches(
    image_path: Path,
    output_dir: Path,
    patch_size: Tuple[int, int],
    overlap: float = 0.0,
    min_bbox_overlap: float = 0.5
) -> List[Dict]:
    """
    Split an image into patches and return patch information.
    
    Returns:
        List of dicts with patch info: {x, y, w, h, filename, annotations}
    """
    # Load image to get dimensions
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    img_height, img_width = img.shape[:2]
    
    # Calculate patch coordinates
    patches_coords = calculate_patch_coordinates(
        img_width, img_height, patch_size, overlap
    )
    
    patches_info = []
    for idx, (px, py, pw, ph) in enumerate(patches_coords):
        # Extract patch
        patch = img[py:py+ph, px:px+pw]
        
        # Generate patch filename
        stem = image_path.stem
        suffix = image_path.suffix
        patch_filename = f"{stem}_patch_{idx:04d}{suffix}"
        patch_path = output_dir / patch_filename
        
        # Save patch
        cv2.imwrite(str(patch_path), patch)
        
        patches_info.append({
            'x': px,
            'y': py,
            'w': pw,
            'h': ph,
            'filename': patch_filename,
            'path': patch_path
        })
    
    return patches_info


def process_coco_dataset(
    annotation_file: Path,
    image_dir: Path,
    output_dir: Path,
    patch_size: Tuple[int, int],
    overlap: float = 0.0,
    min_bbox_overlap: float = 0.5,
    filter_empty_patches: bool = True
):
    """
    Process a COCO dataset by splitting images into patches.
    
    Args:
        annotation_file: Path to COCO annotation file
        image_dir: Directory containing images
        output_dir: Output directory for patches and new annotations
        patch_size: (width, height) of patches
        overlap: Overlap ratio between patches (0.0-1.0)
        min_bbox_overlap: Minimum bbox overlap to include in patch (0.0-1.0)
        filter_empty_patches: If True, exclude patches with no annotations
    """
    print("="*60)
    print("COCO Dataset Image Patcher")
    print("="*60)
    print(f"Input annotation: {annotation_file}")
    print(f"Image directory: {image_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Patch size: {patch_size[0]}x{patch_size[1]}")
    print(f"Overlap: {overlap:.1%}")
    print(f"Min bbox overlap: {min_bbox_overlap:.1%}")
    print("="*60)
    print()
    
    # Load annotations
    print("Loading annotations...")
    coco_data = load_coco_annotations(annotation_file)
    
    # Build mappings
    images_dict = {img['id']: img for img in coco_data['images']}
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    patches_image_dir = output_dir / 'images'
    patches_image_dir.mkdir(exist_ok=True)
    
    # Process each image
    new_images = []
    new_annotations = []
    new_ann_id = 1
    
    print(f"\nProcessing {len(images_dict)} images...")
    for img_id, img_info in tqdm(images_dict.items(), desc="Splitting images"):
        image_path = image_dir / img_info['file_name']
        
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Get annotations for this image
        image_anns = annotations_by_image.get(img_id, [])
        
        # Split image into patches
        try:
            patches_info = split_image_to_patches(
                image_path,
                patches_image_dir,
                patch_size,
                overlap
            )
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
        
        # Process each patch
        for patch_info in patches_info:
            # Find annotations that overlap with this patch
            patch_anns = []
            for ann in image_anns:
                adjusted_bbox = bbox_in_patch(
                    ann['bbox'],
                    patch_info['x'],
                    patch_info['y'],
                    patch_info['w'],
                    patch_info['h'],
                    min_bbox_overlap
                )
                
                if adjusted_bbox is not None:
                    # Create new annotation
                    new_ann = ann.copy()
                    new_ann['id'] = new_ann_id
                    new_ann['bbox'] = adjusted_bbox
                    new_ann['image_id'] = len(new_images)  # Will be set after adding image
                    # Update area
                    new_ann['area'] = adjusted_bbox[2] * adjusted_bbox[3]
                    patch_anns.append(new_ann)
                    new_ann_id += 1
            
            # Skip empty patches if filter_empty_patches is True
            if filter_empty_patches and len(patch_anns) == 0:
                # Delete the patch image
                patch_info['path'].unlink()
                continue
            
            # Create new image entry
            new_img = {
                'id': len(new_images),
                'file_name': patch_info['filename'],
                'width': patch_info['w'],
                'height': patch_info['h'],
                'license': img_info.get('license', 1),
                'flickr_url': '',
                'coco_url': '',
                'date_captured': img_info.get('date_captured', ''),
            }
            
            # Update annotation image_ids
            for ann in patch_anns:
                ann['image_id'] = new_img['id']
            
            new_images.append(new_img)
            new_annotations.extend(patch_anns)
    
    # Create new COCO data structure
    new_coco_data = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'categories': coco_data['categories'],
        'images': new_images,
        'annotations': new_annotations
    }
    
    # Save new annotation file
    output_ann_file = output_dir / annotation_file.name
    save_coco_annotations(new_coco_data, output_ann_file)
    
    # Print statistics
    print("\n" + "="*60)
    print("Processing Statistics:")
    print(f"  Original images: {len(images_dict)}")
    print(f"  New patch images: {len(new_images)}")
    print(f"  Original annotations: {len(coco_data['annotations'])}")
    print(f"  New annotations: {len(new_annotations)}")
    print(f"  Categories: {len(coco_data['categories'])}")
    print(f"  Average patches per image: {len(new_images) / len(images_dict):.2f}")
    print("="*60)
    
    return output_ann_file


def main():
    parser = argparse.ArgumentParser(
        description='Split COCO dataset images into smaller patches'
    )
    parser.add_argument(
        'annotation_file',
        type=str,
        help='Path to COCO annotation file'
    )
    parser.add_argument(
        'image_dir',
        type=str,
        help='Directory containing images'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for patches and new annotations'
    )
    parser.add_argument(
        '--patch-size',
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=('WIDTH', 'HEIGHT'),
        help='Patch size in pixels (default: 640 640)'
    )
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.0,
        help='Overlap ratio between patches (0.0-1.0, default: 0.0)'
    )
    parser.add_argument(
        '--min-bbox-overlap',
        type=float,
        default=0.5,
        help='Minimum bbox overlap to include in patch (0.0-1.0, default: 0.5)'
    )
    parser.add_argument(
        '--keep-empty-patches',
        action='store_true',
        help='Keep patches with no annotations (default: False, filters them out)'
    )
    
    args = parser.parse_args()
    
    annotation_file = Path(args.annotation_file)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    
    if not annotation_file.exists():
        print(f"Error: Annotation file not found: {annotation_file}")
        return 1
    
    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        return 1
    
    if not (0 <= args.overlap < 1):
        print(f"Error: overlap must be in [0, 1), got {args.overlap}")
        return 1
    
    if not (0 <= args.min_bbox_overlap <= 1):
        print(f"Error: min_bbox_overlap must be in [0, 1], got {args.min_bbox_overlap}")
        return 1
    
    try:
        process_coco_dataset(
            annotation_file=annotation_file,
            image_dir=image_dir,
            output_dir=output_dir,
            patch_size=tuple(args.patch_size),
            overlap=args.overlap,
            min_bbox_overlap=args.min_bbox_overlap,
            filter_empty_patches=not args.keep_empty_patches
        )
        
        print(f"\n✓ Processing complete!")
        print(f"  Patches saved to: {output_dir / 'images'}")
        print(f"  New annotations: {output_dir / annotation_file.name}")
        
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
