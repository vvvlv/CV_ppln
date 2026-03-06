#!/usr/bin/env python3
"""
Script to split a COCO format dataset into train and validation splits.

This script reads a COCO annotation file and splits it into train/val splits,
creating new annotation files for each split.
"""

import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple
import shutil


def load_coco_annotations(annotation_path: Path) -> Dict:
    """Load COCO format annotations."""
    with open(annotation_path, 'r') as f:
        return json.load(f)


def save_coco_annotations(data: Dict, output_path: Path):
    """Save COCO format annotations."""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved annotations to: {output_path}")


def split_coco_dataset(
    annotation_file: Path,
    output_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    copy_images: bool = False
) -> Tuple[Path, Path]:
    """
    Split COCO dataset into train and validation splits.
    
    Args:
        annotation_file: Path to input COCO annotation file
        output_dir: Directory to save split annotations
        val_ratio: Ratio of data for validation (default: 0.2 = 20%)
        seed: Random seed for reproducibility
        copy_images: If True, copy images to train/val directories. 
                    If False, only create annotation files (images stay in original location)
    
    Returns:
        Tuple of (train_annotation_path, val_annotation_path)
    """
    # Set random seed
    random.seed(seed)
    
    # Load annotations
    print(f"Loading annotations from: {annotation_file}")
    coco_data = load_coco_annotations(annotation_file)
    
    # Get image IDs
    image_ids = [img['id'] for img in coco_data['images']]
    num_images = len(image_ids)
    
    # Shuffle and split
    random.shuffle(image_ids)
    val_size = int(num_images * val_ratio)
    train_size = num_images - val_size
    
    train_image_ids = set(image_ids[:train_size])
    val_image_ids = set(image_ids[train_size:])
    
    print(f"Split: {train_size} images for training, {val_size} images for validation")
    print(f"Validation ratio: {val_ratio:.1%}")
    
    # Get base directory (where images are located)
    base_dir = annotation_file.parent
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # Backup original annotation file if output is in same location
    if output_dir == base_dir.parent and annotation_file.parent.name == 'train':
        backup_path = annotation_file.with_suffix('.coco.json.backup')
        if not backup_path.exists():
            print(f"Backing up original annotation file to: {backup_path}")
            shutil.copy2(annotation_file, backup_path)
    
    # Split images
    train_images = [img for img in coco_data['images'] if img['id'] in train_image_ids]
    val_images = [img for img in coco_data['images'] if img['id'] in val_image_ids]
    
    # Split annotations
    train_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in train_image_ids]
    val_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in val_image_ids]
    
    # Create train COCO data
    train_coco = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'categories': coco_data['categories'],
        'images': train_images,
        'annotations': train_annotations
    }
    
    # Create val COCO data
    val_coco = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'categories': coco_data['categories'],
        'images': val_images,
        'annotations': val_annotations
    }
    
    # Save annotation files
    train_ann_path = train_dir / annotation_file.name
    val_ann_path = val_dir / annotation_file.name
    
    save_coco_annotations(train_coco, train_ann_path)
    save_coco_annotations(val_coco, val_ann_path)
    
    # Handle images
    if copy_images:
        print("\nCopying images...")
        for img in train_images:
            src = base_dir / img['file_name']
            dst = train_dir / img['file_name']
            if src.exists():
                shutil.copy2(src, dst)
        
        for img in val_images:
            src = base_dir / img['file_name']
            dst = val_dir / img['file_name']
            if src.exists():
                shutil.copy2(src, dst)
        print("Images copied to train/val directories")
    else:
        # Create symlinks or copy images to val directory
        # Since COCODataset looks for images in split directory, we need images there
        print("\nSetting up image directories...")
        
        # For train: images should already be in train_dir (or we create symlinks)
        # For val: we need to copy or symlink images from train to val
        val_dir.mkdir(exist_ok=True)
        
        for img in val_images:
            src = base_dir / img['file_name']
            dst = val_dir / img['file_name']
            if src.exists() and not dst.exists():
                # Create symlink (or copy if symlinks don't work)
                try:
                    dst.symlink_to(src)
                except (OSError, NotImplementedError):
                    # Fallback to copying if symlinks not supported
                    shutil.copy2(src, dst)
        
        print("Val images linked/copied to val directory")
        print("Note: Train images remain in train directory")
    
    # Print statistics
    print("\n" + "="*60)
    print("Split Statistics:")
    print(f"  Train images: {len(train_images)}")
    print(f"  Train annotations: {len(train_annotations)}")
    print(f"  Val images: {len(val_images)}")
    print(f"  Val annotations: {len(val_annotations)}")
    print(f"  Categories: {len(coco_data['categories'])}")
    print("="*60)
    
    return train_ann_path, val_ann_path


def main():
    parser = argparse.ArgumentParser(
        description='Split COCO format dataset into train and validation splits'
    )
    parser.add_argument(
        'annotation_file',
        type=str,
        help='Path to COCO annotation file (e.g., train/_annotations.coco.json)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory where train/val splits will be created'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='Ratio of data for validation (default: 0.2 = 20%%)'
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
        help='Copy images to train/val directories (default: False, only create annotation files)'
    )
    
    args = parser.parse_args()
    
    annotation_file = Path(args.annotation_file)
    if not annotation_file.exists():
        print(f"Error: Annotation file not found: {annotation_file}")
        return 1
    
    output_dir = Path(args.output_dir)
    
    if not (0 < args.val_ratio < 1):
        print(f"Error: val_ratio must be between 0 and 1, got {args.val_ratio}")
        return 1
    
    print("="*60)
    print("COCO Dataset Splitter")
    print("="*60)
    print(f"Input annotation: {annotation_file}")
    print(f"Output directory: {output_dir}")
    print(f"Validation ratio: {args.val_ratio:.1%}")
    print(f"Random seed: {args.seed}")
    print(f"Copy images: {args.copy_images}")
    print("="*60)
    print()
    
    try:
        train_ann, val_ann = split_coco_dataset(
            annotation_file=annotation_file,
            output_dir=output_dir,
            val_ratio=args.val_ratio,
            seed=args.seed,
            copy_images=args.copy_images
        )
        
        print(f"\n✓ Split complete!")
        print(f"  Train annotations: {train_ann}")
        print(f"  Val annotations: {val_ann}")
        
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
