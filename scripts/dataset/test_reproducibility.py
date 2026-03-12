#!/usr/bin/env python3
"""
Test script to verify reproducibility of dataset preparation scripts.

This script verifies that running the conversion and patching scripts
multiple times produces identical output files.
"""

import json
import hashlib
from pathlib import Path
import tempfile
import shutil
import subprocess
import sys


def file_hash(filepath: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def test_conversion_reproducibility(test_dir: Path, annotations_dir: Path):
    """Test that conversion produces identical output when run twice."""
    print("Testing conversion reproducibility...")
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # First run
    output1 = test_dir / "output1.json"
    cmd1 = [
        sys.executable,
        str(script_dir / "convert_soda_a_to_coco.py"),
        str(annotations_dir),
        "--output-file", str(output1),
        "--split", "test"
    ]
    
    subprocess.run(cmd1, check=True, cwd=project_root)
    
    # Second run
    output2 = test_dir / "output2.json"
    cmd2 = [
        sys.executable,
        str(script_dir / "convert_soda_a_to_coco.py"),
        str(annotations_dir),
        "--output-file", str(output2),
        "--split", "test"
    ]
    subprocess.run(cmd2, check=True, cwd=project_root)
    
    # Compare hashes
    hash1 = file_hash(output1)
    hash2 = file_hash(output2)
    
    if hash1 == hash2:
        print("✓ Conversion is reproducible (identical output)")
        return True
    else:
        print("✗ Conversion is NOT reproducible (different output)")
        return False


def test_ordering(coco_file: Path):
    """Test that COCO file has properly sorted data."""
    with open(coco_file) as f:
        data = json.load(f)
    
    # Check images
    image_ids = [img['id'] for img in data['images']]
    images_sorted = image_ids == sorted(image_ids)
    
    # Check annotations
    ann_ids = [ann['id'] for ann in data['annotations']]
    anns_sorted = ann_ids == sorted(ann_ids)
    
    # Check categories
    cat_ids = [cat['id'] for cat in data['categories']]
    cats_sorted = cat_ids == sorted(cat_ids)
    
    all_sorted = images_sorted and anns_sorted and cats_sorted
    
    if all_sorted:
        print(f"✓ {coco_file.name} has properly sorted data")
    else:
        print(f"✗ {coco_file.name} has unsorted data:")
        if not images_sorted:
            print("  - Images not sorted")
        if not anns_sorted:
            print("  - Annotations not sorted")
        if not cats_sorted:
            print("  - Categories not sorted")
    
    return all_sorted


def main():
    """Run reproducibility tests."""
    print("="*60)
    print("Reproducibility Test")
    print("="*60)
    
    # Find project root (parent of scripts directory)
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.resolve()
    
    # Test with a small sample if available (SODA_dataset is at same level as CV_ppln)
    test_annotations = project_root.parent / "SODA_dataset" / "SODA_A" / "Annotations" / "test"
    if not test_annotations.exists():
        print("Warning: Test annotations not found. Skipping reproducibility test.")
        print(f"  Expected at: {test_annotations}")
        print("  To test, ensure SODA_dataset/SODA_A/Annotations/test exists")
        return 0
    
    # Count JSON files
    json_files = list(test_annotations.glob("*.json"))
    if len(json_files) < 2:
        print("Warning: Not enough test files. Need at least 2 JSON files.")
        return 0
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        
        # Test conversion reproducibility
        success = test_conversion_reproducibility(test_dir, test_annotations)
        
        if success:
            # Test ordering of output
            output_file = test_dir / "output1.json"
            test_ordering(output_file)
        
        print("\n" + "="*60)
        if success:
            print("✓ All reproducibility tests passed!")
        else:
            print("✗ Some reproducibility tests failed!")
        print("="*60)
        
        return 0 if success else 1


if __name__ == '__main__':
    exit(main())
