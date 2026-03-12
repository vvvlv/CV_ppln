#!/usr/bin/env python3
"""
Visualize patched dataset to verify correctness.

This script:
1. Loads patched COCO annotations
2. Displays patch images with bounding boxes
3. Optionally shows original image with patch grid overlay
4. Verifies bbox coordinate transformations
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import random

# ============================================================================
# CONFIGURATION - Modify these variables to change default behavior
# ============================================================================

# Base directory for SODA_A dataset (relative to this script or absolute path)
# Default: assumes script is in CV_ppln/scripts/ and dataset is in ../SODA_dataset/SODA_A_coco
_script_dir = Path(__file__).parent.resolve()
_project_root = _script_dir.parent.parent.parent  # Go up from scripts/ to project root
BASE_DIR = _project_root / "SODA_dataset" / "SODA_A_coco_512"
# Or set an absolute path directly:
# BASE_DIR = Path("/home/vlv/Documents/master/computervision/SODA_dataset/SODA_A_coco")

# Which split to visualize: "train_patches", "val_patches", "test_patches"
# Examples:
#   SPLIT = "train_patches"  # Visualize training patches
#   SPLIT = "val_patches"    # Visualize validation patches
#   SPLIT = "test_patches"   # Visualize test patches
SPLIT = "train_patches"

# Paths (will be constructed from BASE_DIR and SPLIT if not set)
# Set to None to use auto-constructed paths, or provide absolute paths
ANNOTATION_FILE = None  # Auto: BASE_DIR / SPLIT / "_annotations.coco.json"
IMAGE_DIR = None        # Auto: BASE_DIR / SPLIT
ORIGINAL_ANNOTATION_FILE = None  # Auto: BASE_DIR / SPLIT.replace("_patches", "") / "_annotations.coco.json"

# Visualization settings
NUM_SAMPLES = 5          # Number of random patches to visualize
RANDOM_SEED = 42         # Random seed for reproducibility
SHOW_ORIGINAL = False    # Show original image alongside patch
VERIFY_TRANSFORMATIONS = True  # Verify bbox transformations
SHOW_STATS = True        # Print dataset statistics

# Output settings
SAVE_DIR = None          # Directory to save visualizations (None = interactive display)
# SAVE_DIR = Path("/tmp/patch_visualizations")  # Uncomment to save instead of displaying

# ============================================================================
# End of configuration
# ============================================================================


def load_coco_annotations(annotation_file: Path) -> Dict:
    """Load COCO format annotations."""
    with open(annotation_file, 'r') as f:
        return json.load(f)


def draw_bbox(img: np.ndarray, bbox: List[float], color: Tuple[int, int, int] = (0, 255, 0), 
               thickness: int = 2, label: Optional[str] = None) -> np.ndarray:
    """
    Draw bounding box on image.
    
    Args:
        img: Image array (H, W, 3)
        bbox: COCO format [x, y, width, height]
        color: BGR color tuple
        thickness: Line thickness
        label: Optional label text
    
    Returns:
        Image with bbox drawn
    """
    img = img.copy()
    x, y, w, h = bbox
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    
    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label if provided
    if label:
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1, label_size[1] + 10)
        cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(img, label, (x1, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img


def visualize_patch(
    patch_image_info: Dict,
    patch_annotations: List[Dict],
    categories: Dict[int, Dict],
    image_dir: Path,
    show_original: bool = False,
    original_image_info: Optional[Dict] = None,
    original_annotations: Optional[List[Dict]] = None,
    save_path: Optional[Path] = None
) -> None:
    """
    Visualize a single patch with its annotations.
    
    Args:
        patch_image_info: COCO image dict for patch
        patch_annotations: List of COCO annotation dicts for patch
        categories: Category ID to category dict mapping
        image_dir: Directory containing patch images
        show_original: If True, also show original image with patch overlay
        original_image_info: Original image info (if show_original=True)
        original_annotations: Original annotations (if show_original=True)
    """
    # Load patch image
    patch_path = image_dir / patch_image_info['file_name']
    if not patch_path.exists():
        print(f"Warning: Patch image not found: {patch_path}")
        return
    
    patch_img = cv2.imread(str(patch_path))
    if patch_img is None:
        print(f"Error: Could not load patch image: {patch_path}")
        return
    
    patch_img = cv2.cvtColor(patch_img, cv2.COLOR_BGR2RGB)
    patch_h, patch_w = patch_img.shape[:2]
    
    # Draw annotations on patch
    patch_img_annotated = patch_img.copy()
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for ann in patch_annotations:
        bbox = ann['bbox']
        cat_id = ann['category_id']
        cat_name = categories.get(cat_id, {}).get('name', f'class_{cat_id}')
        
        # Convert color from matplotlib to BGR
        color_rgb = tuple(int(c * 255) for c in colors[cat_id % len(colors)][:3])
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        
        label = f"{cat_name} (id:{ann['id']})"
        patch_img_annotated = draw_bbox(patch_img_annotated, bbox, color_bgr, 
                                       thickness=2, label=label)
    
    # Create figure
    if show_original and original_image_info is not None:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        ax1, ax2 = axes
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        ax2 = None
    
    # Show patch
    ax1.imshow(patch_img_annotated)
    ax1.set_title(f"Patch: {patch_image_info['file_name']}\n"
                 f"Size: {patch_w}x{patch_h}, Annotations: {len(patch_annotations)}",
                 fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Add grid lines to show patch boundaries
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axhline(y=patch_h, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axvline(x=patch_w, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Show original image with patch overlay if requested
    if show_original and original_image_info is not None and ax2 is not None:
        # Try to load original image
        # Extract original filename from patch filename (format: original_patch_x_y.ext)
        patch_filename = patch_image_info['file_name']
        if '_patch_' in patch_filename:
            base_name = patch_filename.split('_patch_')[0]
            ext = Path(patch_filename).suffix
            original_path = image_dir.parent / (base_name + ext)
            
            # If not found, try in parent directories
            if not original_path.exists():
                # Try train/val/test directories
                for parent_dir in ['train', 'val', 'test']:
                    test_path = image_dir.parent.parent / parent_dir / (base_name + ext)
                    if test_path.exists():
                        original_path = test_path
                        break
                
                # Try other extensions
                if not original_path.exists():
                    for ext_alt in ['.jpg', '.png', '.JPG', '.PNG']:
                        for parent_dir in ['train', 'val', 'test']:
                            test_path = image_dir.parent.parent / parent_dir / (base_name + ext_alt)
                            if test_path.exists():
                                original_path = test_path
                                break
                        if original_path.exists():
                            break
        else:
            original_path = image_dir / patch_filename
        
        if original_path.exists():
            original_img = cv2.imread(str(original_path))
            if original_img is not None:
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                orig_h, orig_w = original_img.shape[:2]
                
                # Draw original annotations
                for ann in original_annotations or []:
                    bbox = ann['bbox']
                    cat_id = ann['category_id']
                    color_rgb = tuple(int(c * 255) for c in colors[cat_id % len(colors)][:3])
                    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
                    original_img = draw_bbox(original_img, bbox, color_bgr, thickness=1)
                
                # Draw patch region on original
                patch_x = patch_image_info.get('patch_x', 0)
                patch_y = patch_image_info.get('patch_y', 0)
                patch_rect = Rectangle((patch_x, patch_y), patch_w, patch_h,
                                     linewidth=3, edgecolor='red', facecolor='none', linestyle='--')
                
                ax2.imshow(original_img)
                ax2.add_patch(patch_rect)
                ax2.set_title(f"Original: {original_image_info.get('file_name', 'unknown')}\n"
                            f"Size: {orig_w}x{orig_h}, Patch region: ({patch_x}, {patch_y})",
                            fontsize=12, fontweight='bold')
                ax2.axis('off')
        else:
            ax2.text(0.5, 0.5, 'Original image not found', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to: {save_path}")
        plt.close()
    else:
        plt.show()
    
    return fig


def verify_bbox_transformation(
    patch_ann: Dict,
    original_ann: Dict,
    patch_x: int,
    patch_y: int
) -> Tuple[bool, str]:
    """
    Verify that bbox transformation is correct.
    
    Args:
        patch_ann: Annotation in patch coordinates
        original_ann: Original annotation
        patch_x, patch_y: Patch offset in original image
    
    Returns:
        Tuple of (is_correct, message)
    """
    # Original bbox
    orig_bbox = original_ann['bbox']
    orig_x, orig_y, orig_w, orig_h = orig_bbox
    
    # Patch bbox
    patch_bbox = patch_ann['bbox']
    patch_x_coord, patch_y_coord, patch_w, patch_h = patch_bbox
    
    # Expected transformation: subtract patch offset
    expected_x = orig_x - patch_x
    expected_y = orig_y - patch_y
    
    # Check if transformation is correct (allow small floating point errors)
    tolerance = 1.0
    x_correct = abs(patch_x_coord - expected_x) < tolerance
    y_correct = abs(patch_y_coord - expected_y) < tolerance
    w_correct = abs(patch_w - orig_w) < tolerance
    h_correct = abs(patch_h - orig_h) < tolerance
    
    if x_correct and y_correct and w_correct and h_correct:
        return True, "✓ Transformation correct"
    else:
        msg = f"✗ Transformation error:\n"
        msg += f"  Expected: ({expected_x:.1f}, {expected_y:.1f}, {orig_w:.1f}, {orig_h:.1f})\n"
        msg += f"  Got: ({patch_x_coord:.1f}, {patch_y_coord:.1f}, {patch_w:.1f}, {patch_h:.1f})"
        return False, msg


def print_statistics(
    patch_data: Dict,
    original_data: Optional[Dict] = None
) -> None:
    """Print statistics about the patched dataset."""
    patch_images = patch_data['images']
    patch_annotations = patch_data['annotations']
    categories = patch_data['categories']
    
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    print(f"Patch images: {len(patch_images)}")
    print(f"Patch annotations: {len(patch_annotations)}")
    print(f"Categories: {len(categories)}")
    
    # Calculate average annotations per patch
    if patch_images:
        annotations_per_patch = len(patch_annotations) / len(patch_images)
        print(f"Average annotations per patch: {annotations_per_patch:.2f}")
    
    # Count patches per original image
    if 'original_image_id' in patch_images[0]:
        orig_to_patches = {}
        for img in patch_images:
            orig_id = img['original_image_id']
            orig_to_patches[orig_id] = orig_to_patches.get(orig_id, 0) + 1
        
        if orig_to_patches:
            avg_patches = sum(orig_to_patches.values()) / len(orig_to_patches)
            print(f"Original images: {len(orig_to_patches)}")
            print(f"Average patches per original: {avg_patches:.2f}")
    
    # Category distribution
    # Convert categories list to dict for easier lookup
    categories_dict = {cat['id']: cat for cat in categories}
    
    cat_counts = {}
    for ann in patch_annotations:
        cat_id = ann['category_id']
        cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1
    
    print(f"\nCategory distribution:")
    for cat_id, count in sorted(cat_counts.items()):
        cat_name = categories_dict.get(cat_id, {}).get('name', f'class_{cat_id}')
        print(f"  {cat_name} (id:{cat_id}): {count}")
    
    if original_data:
        orig_images = original_data['images']
        orig_annotations = original_data['annotations']
        print(f"\nOriginal dataset:")
        print(f"  Images: {len(orig_images)}")
        print(f"  Annotations: {len(orig_annotations)}")
        print(f"  Expansion factor: {len(patch_images) / len(orig_images):.2f}x")
    
    print("="*60)


def visualize_patched_dataset(
    annotation_file: Path,
    image_dir: Path,
    num_samples: int = 5,
    random_seed: int = 42,
    show_original: bool = False,
    original_annotation_file: Optional[Path] = None,
    verify_transformations: bool = True,
    show_stats: bool = True,
    save_output_dir: Optional[Path] = None
) -> None:
    """
    Visualize patched dataset.
    
    Args:
        annotation_file: Path to patched COCO annotation file
        image_dir: Directory containing patch images
        num_samples: Number of random patches to visualize
        random_seed: Random seed for reproducibility
        show_original: If True, show original image alongside patch
        original_annotation_file: Path to original COCO annotation file (for comparison)
        verify_transformations: If True, verify bbox transformations
    """
    print("="*60)
    print("Patched Dataset Visualization")
    print("="*60)
    print(f"Annotation file: {annotation_file}")
    print(f"Image directory: {image_dir}")
    print(f"Number of samples: {num_samples}")
    print("="*60)
    print()
    
    # Load patched annotations
    print("Loading patched annotations...")
    patch_data = load_coco_annotations(annotation_file)
    
    patch_images = patch_data['images']
    patch_annotations = patch_data['annotations']
    categories = {cat['id']: cat for cat in patch_data['categories']}
    
    # Group annotations by image_id
    patch_image_to_anns = {}
    for ann in patch_annotations:
        img_id = ann['image_id']
        if img_id not in patch_image_to_anns:
            patch_image_to_anns[img_id] = []
        patch_image_to_anns[img_id].append(ann)
    
    print(f"Loaded {len(patch_images)} patch images")
    print(f"Loaded {len(patch_annotations)} patch annotations")
    print(f"Categories: {len(categories)}")
    
    # Load original annotations if needed (before printing statistics)
    original_data = None
    original_image_to_anns = {}
    if show_original and original_annotation_file and original_annotation_file.exists():
        print("Loading original annotations...")
        original_data = load_coco_annotations(original_annotation_file)
        for ann in original_data['annotations']:
            img_id = ann['image_id']
            if img_id not in original_image_to_anns:
                original_image_to_anns[img_id] = []
            original_image_to_anns[img_id].append(ann)
        print(f"Loaded {len(original_data['images'])} original images")
        print()
    
    # Print statistics (after loading original data if needed)
    if show_stats:
        print_statistics(patch_data, original_data)
    
    print()
    
    # Select random samples
    random.seed(random_seed)
    sample_indices = random.sample(range(len(patch_images)), min(num_samples, len(patch_images)))
    
    # Visualize each sample
    for idx, sample_idx in enumerate(sample_indices):
        patch_img_info = patch_images[sample_idx]
        patch_img_id = patch_img_info['id']
        patch_anns = patch_image_to_anns.get(patch_img_id, [])
        
        print(f"\n{'='*60}")
        print(f"Sample {idx + 1}/{num_samples}")
        print(f"{'='*60}")
        print(f"Patch image: {patch_img_info['file_name']}")
        print(f"Patch size: {patch_img_info['width']}x{patch_img_info['height']}")
        print(f"Patch offset: ({patch_img_info.get('patch_x', 'N/A')}, {patch_img_info.get('patch_y', 'N/A')})")
        print(f"Number of annotations: {len(patch_anns)}")
        
        # Verify transformations if requested
        if verify_transformations and original_data and 'original_image_id' in patch_img_info:
            orig_img_id = patch_img_info['original_image_id']
            orig_anns = original_image_to_anns.get(orig_img_id, [])
            
            print(f"\nVerifying bbox transformations...")
            patch_x = patch_img_info.get('patch_x', 0)
            patch_y = patch_img_info.get('patch_y', 0)
            
            # Match annotations by original_annotation_id
            for patch_ann in patch_anns[:5]:  # Check first 5
                orig_ann_id = patch_ann.get('original_annotation_id')
                if orig_ann_id:
                    orig_ann = next((a for a in orig_anns if a['id'] == orig_ann_id), None)
                    if orig_ann:
                        is_correct, msg = verify_bbox_transformation(
                            patch_ann, orig_ann, patch_x, patch_y
                        )
                        print(f"  Annotation {patch_ann['id']}: {msg}")
        
        # Get original image info if available
        original_img_info = None
        original_anns = []
        if show_original and original_data and 'original_image_id' in patch_img_info:
            orig_img_id = patch_img_info['original_image_id']
            original_img_info = next((img for img in original_data['images'] 
                                    if img['id'] == orig_img_id), None)
            original_anns = original_image_to_anns.get(orig_img_id, [])
        
        # Visualize
        save_path = None
        if save_output_dir:
            save_path = save_output_dir / f"sample_{idx + 1:03d}_{patch_img_info['file_name'].replace('.jpg', '.png').replace('.png', '.png')}"
        
        visualize_patch(
            patch_img_info,
            patch_anns,
            categories,
            image_dir,
            show_original=show_original,
            original_image_info=original_img_info,
            original_annotations=original_anns,
            save_path=save_path
        )
        
        if not save_output_dir:
            print("\nPress Enter to continue to next sample...")
            input()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize patched dataset to verify correctness',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use hardcoded configuration (modify variables at top of script)
  python visualize_patched_dataset.py
  
  # Override specific paths via command line
  python visualize_patched_dataset.py --split val_patches --num-samples 10
  
  # Use custom paths
  python visualize_patched_dataset.py --annotation-file /path/to/annotations.json --image-dir /path/to/images
        """
    )
    
    # Auto-construct paths from configuration if not provided
    if ANNOTATION_FILE is None:
        default_annotation = BASE_DIR / SPLIT / "_annotations.coco.json"
    else:
        default_annotation = Path(ANNOTATION_FILE)
    
    if IMAGE_DIR is None:
        default_image_dir = BASE_DIR / SPLIT
    else:
        default_image_dir = Path(IMAGE_DIR)
    
    if ORIGINAL_ANNOTATION_FILE is None and SPLIT.endswith("_patches"):
        original_split = SPLIT.replace("_patches", "")
        default_original_annotation = BASE_DIR / original_split / "_annotations.coco.json"
        if not default_original_annotation.exists():
            default_original_annotation = None
    else:
        default_original_annotation = Path(ORIGINAL_ANNOTATION_FILE) if ORIGINAL_ANNOTATION_FILE else None
    
    parser.add_argument(
        '--annotation-file',
        type=str,
        default=str(default_annotation) if default_annotation.exists() else None,
        help=f'Path to patched COCO annotation file (default: {default_annotation})'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default=str(default_image_dir),
        help=f'Directory containing patch images (default: {default_image_dir})'
    )
    parser.add_argument(
        '--split',
        type=str,
        default=SPLIT,
        help=f'Split to visualize: train_patches, val_patches, test_patches (default: {SPLIT})'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=NUM_SAMPLES,
        help=f'Number of random patches to visualize (default: {NUM_SAMPLES})'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=RANDOM_SEED,
        help=f'Random seed for reproducibility (default: {RANDOM_SEED})'
    )
    parser.add_argument(
        '--show-original',
        action='store_true',
        default=SHOW_ORIGINAL,
        help=f'Show original image alongside patch (default: {SHOW_ORIGINAL})'
    )
    parser.add_argument(
        '--original-annotation',
        type=str,
        default=str(default_original_annotation) if default_original_annotation and default_original_annotation.exists() else None,
        help='Path to original COCO annotation file (for comparison)'
    )
    parser.add_argument(
        '--no-verify',
        action='store_true',
        default=not VERIFY_TRANSFORMATIONS,
        help='Skip bbox transformation verification'
    )
    parser.add_argument(
        '--no-stats',
        action='store_true',
        default=not SHOW_STATS,
        help='Skip printing statistics'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default=str(SAVE_DIR) if SAVE_DIR else None,
        help='Directory to save visualization images (None = interactive display)'
    )
    
    args = parser.parse_args()
    
    # If split was provided, reconstruct paths
    if args.split != SPLIT:
        split_annotation = BASE_DIR / args.split / "_annotations.coco.json"
        split_image_dir = BASE_DIR / args.split
        if split_annotation.exists():
            args.annotation_file = str(split_annotation)
        if split_image_dir.exists():
            args.image_dir = str(split_image_dir)
        
        # Try to find original annotation for this split
        if args.split.endswith("_patches") and not args.original_annotation:
            original_split = args.split.replace("_patches", "")
            original_ann = BASE_DIR / original_split / "_annotations.coco.json"
            if original_ann.exists():
                args.original_annotation = str(original_ann)
    
    if not args.annotation_file:
        print("Error: Annotation file not specified and could not be auto-detected.")
        print(f"  Checked: {default_annotation}")
        print(f"  Modify SPLIT or BASE_DIR at the top of the script, or provide --annotation-file")
        return 1
    
    annotation_file = Path(args.annotation_file)
    if not annotation_file.exists():
        print(f"Error: Annotation file not found: {annotation_file}")
        return 1
    
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        return 1
    
    original_annotation_file = Path(args.original_annotation) if args.original_annotation else None
    
    try:
        save_output_dir = Path(args.save_dir) if args.save_dir else None
        
        visualize_patched_dataset(
            annotation_file=annotation_file,
            image_dir=image_dir,
            num_samples=args.num_samples,
            random_seed=args.seed,
            show_original=args.show_original,
            original_annotation_file=original_annotation_file,
            verify_transformations=not args.no_verify,
            show_stats=not args.no_stats,
            save_output_dir=save_output_dir
        )
        return 0
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user.")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
