#!/usr/bin/env python3
"""
Create patched version of COCO dataset by dividing images into subpatches.

This script:
1. Loads a COCO format annotation file
2. Divides each image into a grid of patches (e.g., 2x2, 3x3)
3. Transforms bbox coordinates to patch-relative coordinates
4. Filters out bboxes that don't overlap with each patch
5. Creates new COCO annotation file with patch images

This is useful for training on less powerful systems where full-resolution
images are too large.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from tqdm import tqdm
import shutil


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bboxes.
    
    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
    
    Returns:
        IoU value (0.0 to 1.0)
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def bbox_to_xyxy(bbox: List[float]) -> List[float]:
    """Convert COCO bbox [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def xyxy_to_bbox(xyxy: List[float]) -> List[float]:
    """Convert [x1, y1, x2, y2] to COCO bbox [x, y, w, h]."""
    x1, y1, x2, y2 = xyxy
    return [x1, y1, x2 - x1, y2 - y1]


def transform_bbox_to_patch(
    bbox: List[float],
    patch_x: int,
    patch_y: int,
    patch_w: int,
    patch_h: int
) -> Optional[List[float]]:
    """
    Transform bbox coordinates to patch-relative coordinates.
    
    Args:
        bbox: COCO bbox [x, y, w, h] in original image coordinates
        patch_x, patch_y: Top-left corner of patch in original image
        patch_w, patch_h: Width and height of patch
    
    Returns:
        Transformed bbox [x, y, w, h] in patch coordinates, or None if bbox doesn't overlap
    """
    # Convert to xyxy format
    bbox_xyxy = bbox_to_xyxy(bbox)
    x1, y1, x2, y2 = bbox_xyxy
    
    # Patch boundaries in original image coordinates
    patch_x1 = patch_x
    patch_y1 = patch_y
    patch_x2 = patch_x + patch_w
    patch_y2 = patch_y + patch_h
    
    # Calculate intersection
    inter_x1 = max(x1, patch_x1)
    inter_y1 = max(y1, patch_y1)
    inter_x2 = min(x2, patch_x2)
    inter_y2 = min(y2, patch_y2)
    
    # Check if there's overlap
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return None
    
    # Transform to patch coordinates (subtract patch offset)
    new_x1 = inter_x1 - patch_x1
    new_y1 = inter_y1 - patch_y1
    new_x2 = inter_x2 - patch_x1
    new_y2 = inter_y2 - patch_y1
    
    # Clamp to patch boundaries
    new_x1 = max(0, min(new_x1, patch_w))
    new_y1 = max(0, min(new_y1, patch_h))
    new_x2 = max(0, min(new_x2, patch_w))
    new_y2 = max(0, min(new_y2, patch_h))
    
    # Convert back to COCO format
    return xyxy_to_bbox([new_x1, new_y1, new_x2, new_y2])


def create_patches_from_image(
    image_info: Dict,
    annotations: List[Dict],
    num_patches_x: int,
    num_patches_y: int,
    min_overlap_ratio: float = 0.1,
    input_image_dir: Optional[Path] = None,
    output_image_dir: Optional[Path] = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Create patches from a single image and transform annotations.
    
    Args:
        image_info: COCO image dict
        annotations: List of COCO annotation dicts for this image
        num_patches_x: Number of patches in x direction
        num_patches_y: Number of patches in y direction
        min_overlap_ratio: Minimum IoU ratio to keep a bbox in a patch
        input_image_dir: Directory containing original images (for loading)
        output_image_dir: Directory to save patch images
    
    Returns:
        Tuple of (patch_images, patch_annotations)
    """
    img_w = image_info['width']
    img_h = image_info['height']
    img_id = image_info['id']
    img_filename = image_info['file_name']
    
    # Calculate patch size
    patch_w = img_w // num_patches_x
    patch_h = img_h // num_patches_y
    
    patch_images = []
    patch_annotations = []
    next_patch_id = img_id * 10000  # Use large multiplier to avoid ID conflicts
    next_ann_id = len(annotations) * 10000
    
    # Load image if we need to save patches
    image_array = None
    if input_image_dir is not None:
        img_path = input_image_dir / img_filename
        if img_path.exists():
            image_array = cv2.imread(str(img_path))
            if image_array is not None:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    # Create patches
    for py in range(num_patches_y):
        for px in range(num_patches_x):
            patch_x = px * patch_w
            patch_y = py * patch_h
            
            # Handle edge case: last patch might be larger to cover entire image
            if px == num_patches_x - 1:
                patch_w_actual = img_w - patch_x
            else:
                patch_w_actual = patch_w
            
            if py == num_patches_y - 1:
                patch_h_actual = img_h - patch_y
            else:
                patch_h_actual = patch_h
            
            # Create patch image entry
            patch_filename = f"{Path(img_filename).stem}_patch_{px}_{py}{Path(img_filename).suffix}"
            patch_image = {
                'id': next_patch_id,
                'file_name': patch_filename,
                'width': patch_w_actual,
                'height': patch_h_actual,
                'original_image_id': img_id,
                'patch_x': patch_x,
                'patch_y': patch_y
            }
            patch_images.append(patch_image)
            
            # Process annotations for this patch
            for ann in annotations:
                bbox = ann['bbox']
                bbox_xyxy = bbox_to_xyxy(bbox)
                
                # Patch region in original image coordinates
                patch_xyxy = [patch_x, patch_y, patch_x + patch_w_actual, patch_y + patch_h_actual]
                
                # Calculate IoU
                iou = calculate_iou(bbox_xyxy, patch_xyxy)
                
                # Keep annotation if IoU is above threshold
                # Also check if bbox center is in patch (for edge cases)
                bbox_center_x = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
                bbox_center_y = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
                center_in_patch = (patch_x <= bbox_center_x < patch_x + patch_w_actual and
                                  patch_y <= bbox_center_y < patch_y + patch_h_actual)
                
                if iou >= min_overlap_ratio or center_in_patch:
                    # Transform bbox to patch coordinates
                    transformed_bbox = transform_bbox_to_patch(
                        bbox, patch_x, patch_y, patch_w_actual, patch_h_actual
                    )
                    
                    if transformed_bbox is not None:
                        # Calculate area in patch coordinates
                        x, y, w, h = transformed_bbox
                        area = w * h
                        
                        # Only keep if area is meaningful (at least 1 pixel)
                        if area >= 1.0:
                            patch_ann = {
                                'id': next_ann_id,
                                'image_id': next_patch_id,
                                'category_id': ann['category_id'],
                                'bbox': transformed_bbox,
                                'area': area,
                                'iscrowd': ann.get('iscrowd', 0),
                                'original_annotation_id': ann['id']
                            }
                            patch_annotations.append(patch_ann)
                            next_ann_id += 1
            
            # Save patch image if image array is available
            if image_array is not None and output_image_dir is not None:
                patch_img = image_array[patch_y:patch_y + patch_h_actual, patch_x:patch_x + patch_w_actual]
                patch_path = output_image_dir / patch_filename
                patch_img_bgr = cv2.cvtColor(patch_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(patch_path), patch_img_bgr)
            
            next_patch_id += 1
    
    return patch_images, patch_annotations


def create_patched_dataset(
    annotation_file: Path,
    output_file: Path,
    num_patches_x: int = 2,
    num_patches_y: int = 2,
    min_overlap_ratio: float = 0.1,
    image_dir: Optional[Path] = None,
    output_image_dir: Optional[Path] = None
) -> Dict:
    """
    Create patched version of COCO dataset.
    
    This function ensures reproducibility by:
    - Sorting images by ID before processing (deterministic patch ID generation)
    - Sorting annotations within each image by ID
    - Sorting output patch images and annotations by ID
    
    Args:
        annotation_file: Input COCO annotation file
        output_file: Output COCO annotation file for patches
        num_patches_x: Number of patches in x direction (default: 2)
        num_patches_y: Number of patches in y direction (default: 2)
        min_overlap_ratio: Minimum IoU ratio to keep a bbox in a patch (default: 0.1)
        image_dir: Directory containing original images (optional, for saving patches)
        output_image_dir: Directory to save patch images (if None, uses same as image_dir)
    
    Returns:
        COCO format dict for patches (with sorted images and annotations)
    """
    # Load original COCO data
    print(f"Loading annotations from: {annotation_file}")
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']
    
    # Sort images by ID for reproducibility (ensures consistent patch ID generation)
    images = sorted(images, key=lambda img: img['id'])
    
    # Group annotations by image_id and sort for reproducibility
    image_to_anns = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in image_to_anns:
            image_to_anns[img_id] = []
        image_to_anns[img_id].append(ann)
    
    # Sort annotations within each image by ID for reproducibility
    for img_id in image_to_anns:
        image_to_anns[img_id] = sorted(image_to_anns[img_id], key=lambda ann: ann.get('id', 0))
    
    print(f"Processing {len(images)} images...")
    print(f"  Patches per image: {num_patches_x}x{num_patches_y} = {num_patches_x * num_patches_y}")
    print(f"  Min overlap ratio: {min_overlap_ratio}")
    
    # Determine image directory
    if image_dir is None:
        # Try to infer from annotation file location
        image_dir = annotation_file.parent
    else:
        image_dir = Path(image_dir)
    
    if output_image_dir is None:
        output_image_dir = image_dir
    else:
        output_image_dir = Path(output_image_dir)
        output_image_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    all_patch_images = []
    all_patch_annotations = []
    
    for image_info in tqdm(images, desc="Creating patches"):
        img_anns = image_to_anns.get(image_info['id'], [])
        patch_images, patch_anns = create_patches_from_image(
            image_info,
            img_anns,
            num_patches_x,
            num_patches_y,
            min_overlap_ratio,
            input_image_dir=image_dir,
            output_image_dir=output_image_dir
        )
        all_patch_images.extend(patch_images)
        all_patch_annotations.extend(patch_anns)
    
    # Sort patch images and annotations by ID for reproducibility
    all_patch_images = sorted(all_patch_images, key=lambda img: img['id'])
    all_patch_annotations = sorted(all_patch_annotations, key=lambda ann: ann['id'])
    
    # Create COCO data for patches
    patch_coco_data = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'categories': categories,
        'images': all_patch_images,
        'annotations': all_patch_annotations
    }
    
    # Save to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(patch_coco_data, f, indent=2, sort_keys=False)
    
    print(f"\n✓ Created patched COCO annotation file: {output_file}")
    print(f"  Original images: {len(images)}")
    print(f"  Patch images: {len(all_patch_images)}")
    print(f"  Original annotations: {len(annotations)}")
    print(f"  Patch annotations: {len(all_patch_annotations)}")
    print(f"  Categories: {len(categories)}")
    
    return patch_coco_data


def main():
    parser = argparse.ArgumentParser(
        description='Create patched version of COCO dataset by dividing images into subpatches'
    )
    parser.add_argument(
        'annotation_file',
        type=str,
        help='Input COCO annotation file (e.g., train/_annotations.coco.json)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='Output COCO annotation file for patches (e.g., train/_annotations.coco.json)'
    )
    parser.add_argument(
        '--patches-x',
        type=int,
        default=2,
        help='Number of patches in x direction (default: 2)'
    )
    parser.add_argument(
        '--patches-y',
        type=int,
        default=2,
        help='Number of patches in y direction (default: 2)'
    )
    parser.add_argument(
        '--min-overlap',
        type=float,
        default=0.1,
        help='Minimum IoU ratio to keep a bbox in a patch (default: 0.1)'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        help='Directory containing original images (if not specified, inferred from annotation file)'
    )
    parser.add_argument(
        '--output-image-dir',
        type=str,
        help='Directory to save patch images (default: same as image-dir)'
    )
    
    args = parser.parse_args()
    
    annotation_file = Path(args.annotation_file)
    if not annotation_file.exists():
        print(f"Error: Annotation file not found: {annotation_file}")
        return 1
    
    output_file = Path(args.output_file)
    image_dir = Path(args.image_dir) if args.image_dir else None
    output_image_dir = Path(args.output_image_dir) if args.output_image_dir else None
    
    if not (0 < args.min_overlap <= 1):
        print(f"Error: min-overlap must be between 0 and 1, got {args.min_overlap}")
        return 1
    
    print("="*60)
    print("COCO Dataset Patcher")
    print("="*60)
    print(f"Input annotation: {annotation_file}")
    print(f"Output annotation: {output_file}")
    print(f"Patch grid: {args.patches_x}x{args.patches_y}")
    print(f"Min overlap ratio: {args.min_overlap}")
    print(f"Image directory: {image_dir}")
    print(f"Output image directory: {output_image_dir}")
    print("="*60)
    print()
    
    try:
        create_patched_dataset(
            annotation_file=annotation_file,
            output_file=output_file,
            num_patches_x=args.patches_x,
            num_patches_y=args.patches_y,
            min_overlap_ratio=args.min_overlap,
            image_dir=image_dir,
            output_image_dir=output_image_dir
        )
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
