#!/usr/bin/env python3
"""
Split COCO dataset into train, test, and validation splits.

This script splits a COCO format dataset into three splits with specified ratios.
"""

import json
import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, Tuple


def load_coco_annotations(annotation_file: Path) -> Dict:
    """Load COCO format annotations."""
    with open(annotation_file, 'r') as f:
        return json.load(f)


def save_coco_annotations(data: Dict, output_path: Path):
    """Save COCO format annotations."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved annotations to: {output_path}")


def split_coco_three_way(
    annotation_file: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    seed: int = 42,
    copy_images: bool = False
) -> Tuple[Path, Path, Path]:
    """
    Split COCO dataset into train, test, and validation splits.
    
    Args:
        annotation_file: Path to input COCO annotation file
        output_dir: Directory to save split annotations
        train_ratio: Ratio of data for training (default: 0.7 = 70%)
        test_ratio: Ratio of data for testing (default: 0.15 = 15%)
        val_ratio: Ratio of data for validation (default: 0.15 = 15%)
        seed: Random seed for reproducibility
        copy_images: If True, copy images to train/test/val directories.
                    If False, only create annotation files
    
    Returns:
        Tuple of (train_annotation_path, test_annotation_path, val_annotation_path)
    """
    # Validate ratios
    total_ratio = train_ratio + test_ratio + val_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Set random seed
    random.seed(seed)
    
    # Load annotations
    print(f"Loading annotations from: {annotation_file}")
    coco_data = load_coco_annotations(annotation_file)
    
    # Get image IDs
    image_ids = sorted([img['id'] for img in coco_data['images']])
    num_images = len(image_ids)
    
    # Shuffle and split
    random.shuffle(image_ids)
    train_size = int(num_images * train_ratio)
    test_size = int(num_images * test_ratio)
    val_size = num_images - train_size - test_size  # Remaining goes to val
    
    train_image_ids = set(image_ids[:train_size])
    test_image_ids = set(image_ids[train_size:train_size + test_size])
    val_image_ids = set(image_ids[train_size + test_size:])
    
    print(f"Split: {train_size} train, {test_size} test, {val_size} val")
    print(f"Ratios: Train={train_ratio:.1%}, Test={test_ratio:.1%}, Val={val_ratio:.1%}")
    
    # Get base directory (where images are located)
    base_dir = annotation_file.parent
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = output_dir / 'train'
    test_dir = output_dir / 'test'
    val_dir = output_dir / 'val'
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # Split images
    train_images = [img for img in coco_data['images'] if img['id'] in train_image_ids]
    test_images = [img for img in coco_data['images'] if img['id'] in test_image_ids]
    val_images = [img for img in coco_data['images'] if img['id'] in val_image_ids]
    
    # Split annotations
    train_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in train_image_ids]
    test_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in test_image_ids]
    val_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in val_image_ids]
    
    # Create COCO data for each split
    train_coco = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'categories': coco_data['categories'],
        'images': sorted(train_images, key=lambda x: x['id']),
        'annotations': sorted(train_annotations, key=lambda x: x['id'])
    }
    
    test_coco = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'categories': coco_data['categories'],
        'images': sorted(test_images, key=lambda x: x['id']),
        'annotations': sorted(test_annotations, key=lambda x: x['id'])
    }
    
    val_coco = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'categories': coco_data['categories'],
        'images': sorted(val_images, key=lambda x: x['id']),
        'annotations': sorted(val_annotations, key=lambda x: x['id'])
    }
    
    # Save annotation files (use standard name _annotations.coco.json)
    annotation_filename = "_annotations.coco.json"
    train_ann_path = train_dir / annotation_filename
    test_ann_path = test_dir / annotation_filename
    val_ann_path = val_dir / annotation_filename
    
    save_coco_annotations(train_coco, train_ann_path)
    save_coco_annotations(test_coco, test_ann_path)
    save_coco_annotations(val_coco, val_ann_path)
    
    # Handle images
    if copy_images:
        print("\nCopying images...")
        for img in train_images:
            src = base_dir / img['file_name']
            dst = train_dir / img['file_name']
            if src.exists():
                shutil.copy2(src, dst)
        
        for img in test_images:
            src = base_dir / img['file_name']
            dst = test_dir / img['file_name']
            if src.exists():
                shutil.copy2(src, dst)
        
        for img in val_images:
            src = base_dir / img['file_name']
            dst = val_dir / img['file_name']
            if src.exists():
                shutil.copy2(src, dst)
        print("Images copied to train/test/val directories")
    
    # Print statistics
    print("\n" + "="*60)
    print("Split Statistics:")
    print(f"  Train images: {len(train_images)}")
    print(f"  Train annotations: {len(train_annotations)}")
    print(f"  Test images: {len(test_images)}")
    print(f"  Test annotations: {len(test_annotations)}")
    print(f"  Val images: {len(val_images)}")
    print(f"  Val annotations: {len(val_annotations)}")
    print(f"  Categories: {len(coco_data['categories'])}")
    print("="*60)
    
    return train_ann_path, test_ann_path, val_ann_path


def main():
    parser = argparse.ArgumentParser(
        description='Split COCO format dataset into train, test, and validation splits'
    )
    parser.add_argument(
        'annotation_file',
        type=str,
        help='Path to COCO annotation file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory where train/test/val splits will be created'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Ratio of data for training (default: 0.7 = 70%%)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Ratio of data for testing (default: 0.15 = 15%%)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Ratio of data for validation (default: 0.15 = 15%%)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--copy-images',
        action='store_true',
        help='Copy images to train/test/val directories (default: False)'
    )
    
    args = parser.parse_args()
    
    annotation_file = Path(args.annotation_file)
    if not annotation_file.exists():
        print(f"Error: Annotation file not found: {annotation_file}")
        return 1
    
    output_dir = Path(args.output_dir)
    
    total_ratio = args.train_ratio + args.test_ratio + args.val_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Error: Ratios must sum to 1.0, got {total_ratio}")
        return 1
    
    print("="*60)
    print("COCO Dataset Three-Way Splitter")
    print("="*60)
    print(f"Input annotation: {annotation_file}")
    print(f"Output directory: {output_dir}")
    print(f"Train ratio: {args.train_ratio:.1%}")
    print(f"Test ratio: {args.test_ratio:.1%}")
    print(f"Val ratio: {args.val_ratio:.1%}")
    print(f"Random seed: {args.seed}")
    print(f"Copy images: {args.copy_images}")
    print("="*60)
    print()
    
    try:
        train_ann, test_ann, val_ann = split_coco_three_way(
            annotation_file=annotation_file,
            output_dir=output_dir,
            train_ratio=args.train_ratio,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            copy_images=args.copy_images
        )
        
        print(f"\n✓ Split complete!")
        print(f"  Train annotations: {train_ann}")
        print(f"  Test annotations: {test_ann}")
        print(f"  Val annotations: {val_ann}")
        
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
