#!/usr/bin/env python3
"""
Master script to prepare SODA_A dataset for training.

This script:
1. Converts per-image JSON format to COCO format for train/test/val splits
2. Optionally creates patched versions of the dataset

Usage:
    python scripts/prepare_soda_a_dataset.py \
        --soda-a-dir /path/to/SODA_dataset/SODA_A \
        --output-dir /path/to/output \
        [--create-patches --patches-x 2 --patches-y 2]
"""

import argparse
import subprocess
from pathlib import Path
import sys


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\n✗ Error: {description} failed with exit code {result.returncode}")
        sys.exit(1)
    
    print(f"✓ {description} completed successfully")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare SODA_A dataset: convert to COCO format and optionally create patches'
    )
    parser.add_argument(
        '--soda-a-dir',
        type=str,
        required=True,
        help='Path to SODA_A dataset directory (containing Annotations/ and Images/)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for COCO format dataset'
    )
    parser.add_argument(
        '--create-patches',
        action='store_true',
        help='Create patched versions of the dataset'
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
        help='Minimum IoU ratio to keep bbox in patch (default: 0.1)'
    )
    parser.add_argument(
        '--copy-images',
        action='store_true',
        help='Copy images to output directory (default: False, assumes images are in place)'
    )
    
    args = parser.parse_args()
    
    soda_a_dir = Path(args.soda_a_dir)
    output_dir = Path(args.output_dir)
    
    if not soda_a_dir.exists():
        print(f"Error: SODA_A directory not found: {soda_a_dir}")
        return 1
    
    annotations_dir = soda_a_dir / 'Annotations'
    if not annotations_dir.exists():
        print(f"Error: Annotations directory not found: {annotations_dir}")
        return 1
    
    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val', 'test']:
        (output_dir / split).mkdir(exist_ok=True)
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    print("="*60)
    print("SODA_A Dataset Preparation")
    print("="*60)
    print(f"SODA_A directory: {soda_a_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Create patches: {args.create_patches}")
    if args.create_patches:
        print(f"  Patch grid: {args.patches_x}x{args.patches_y}")
        print(f"  Min overlap: {args.min_overlap}")
    print("="*60)
    
    # Step 1: Convert per-image JSON to COCO format for each split
    for split in ['train', 'val', 'test']:
        split_annotations_dir = annotations_dir / split
        if not split_annotations_dir.exists():
            print(f"Warning: {split} split not found, skipping...")
            continue
        
        output_annotation_file = output_dir / split / '_annotations.coco.json'
        
        cmd = [
            sys.executable,
            str(script_dir / 'convert_soda_a_to_coco.py'),
            str(split_annotations_dir),
            '--output-file', str(output_annotation_file),
            '--split', split
        ]
        
        run_command(cmd, f"Converting {split} split to COCO format")
    
    # Step 2: Copy or link images (if requested)
    images_dir = soda_a_dir / 'Images'
    if args.copy_images and images_dir.exists():
        print(f"\n{'='*60}")
        print("Copying images to output directory")
        print(f"{'='*60}")
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(images_dir.rglob(f'*{ext}')))
        
        if not image_files:
            # Try to find images in split directories
            for split in ['train', 'val', 'test']:
                split_images_dir = images_dir / split if images_dir.exists() else None
                if split_images_dir and split_images_dir.exists():
                    for ext in image_extensions:
                        image_files.extend(list(split_images_dir.rglob(f'*{ext}')))
        
        # Sort files by path for reproducibility
        image_files = sorted(image_files, key=lambda p: str(p))
        
        if image_files:
            print(f"Found {len(image_files)} image files")
            for img_file in image_files:
                # Determine which split this image belongs to
                # This is a heuristic - you may need to adjust based on your structure
                rel_path = img_file.relative_to(images_dir)
                for split in ['train', 'val', 'test']:
                    if split in str(rel_path) or split in str(img_file.parent):
                        dst = output_dir / split / img_file.name
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        if not dst.exists():
                            import shutil
                            shutil.copy2(img_file, dst)
                        break
            print("✓ Images copied")
        else:
            print("Warning: No images found. Make sure images are in the correct location.")
            print("  Images should be in Images/ or Images/train/, Images/val/, Images/test/")
    
    # Step 3: Create patched versions if requested
    if args.create_patches:
        for split in ['train', 'val', 'test']:
            input_annotation = output_dir / split / '_annotations.coco.json'
            if not input_annotation.exists():
                print(f"Warning: {split} annotation file not found, skipping patching...")
                continue
            
            # Create patched version
            patched_output = output_dir / f'{split}_patches' / '_annotations.coco.json'
            patched_image_dir = output_dir / f'{split}_patches'
            
            cmd = [
                sys.executable,
                str(script_dir / 'create_patched_dataset.py'),
                str(input_annotation),
                '--output-file', str(patched_output),
                '--patches-x', str(args.patches_x),
                '--patches-y', str(args.patches_y),
                '--min-overlap', str(args.min_overlap),
                '--image-dir', str(output_dir / split),
                '--output-image-dir', str(patched_image_dir)
            ]
            
            run_command(cmd, f"Creating patched version of {split} split")
    
    print("\n" + "="*60)
    print("✓ Dataset preparation complete!")
    print("="*60)
    print(f"\nOutput directory structure:")
    print(f"  {output_dir}/")
    for split in ['train', 'val', 'test']:
        ann_file = output_dir / split / '_annotations.coco.json'
        if ann_file.exists():
            print(f"    {split}/")
            print(f"      _annotations.coco.json")
    if args.create_patches:
        for split in ['train', 'val', 'test']:
            ann_file = output_dir / f'{split}_patches' / '_annotations.coco.json'
            if ann_file.exists():
                print(f"    {split}_patches/")
                print(f"      _annotations.coco.json")
                print(f"      (patch images)")
    
    return 0


if __name__ == '__main__':
    exit(main())
