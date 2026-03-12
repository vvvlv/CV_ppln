#!/bin/bash
# ============================================================================
# Dataset Preparation Script
# ============================================================================
# This script prepares a dataset from SODA_A format to COCO format with:
# - Train/Test/Val split: 70% / 15% / 15%
# - Patches of maximum size 512x512
#
# Usage:
#   1. Edit the configuration variables below (lines 16-45)
#   2. Run: ./scripts/dataset/prepare_dataset.sh
#
# The script will:
#   1. Convert per-image JSON annotations to COCO format
#   2. Split into train/test/val (70/15/15)
#   3. Copy images to respective split directories
#   4. Create patches with max size 512x512
#
# Customize the variables below to match your dataset location and preferences.
# ============================================================================

# ============================================================================
# CONFIGURATION - Modify these variables
# ============================================================================

# Input dataset directory (SODA_A format: containing Annotations/ and Images/)
SODA_A_DIR="/home/vlv/Documents/master/computervision/SODA_dataset/SODA_A"

# Output directory for COCO format dataset
OUTPUT_DIR="/home/vlv/Documents/master/computervision/SODA_dataset/SODA_A_coco_256"

# Maximum patch size (patches will be at most this size)
MAX_PATCH_SIZE=256

# Split ratios (must sum to 1.0)
TRAIN_RATIO=0.70
TEST_RATIO=0.15
VAL_RATIO=0.15

# Random seed for reproducibility
RANDOM_SEED=42

# Minimum IoU overlap ratio to keep a bbox in a patch (0.0 to 1.0)
MIN_OVERLAP_RATIO=0.1

# Minimum bbox size to keep in patches (in pixels)
MIN_BBOX_SIZE=1

# Copy images to output directory (true/false)
COPY_IMAGES=true

# Script directory (auto-detected, usually no need to change)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ============================================================================
# End of configuration
# ============================================================================

set -e  # Exit on error

echo "============================================================================"
echo "Dataset Preparation Script"
echo "============================================================================"
echo "Input directory: $SODA_A_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Max patch size: ${MAX_PATCH_SIZE}x${MAX_PATCH_SIZE}"
echo "Split ratios: Train=${TRAIN_RATIO} Test=${TEST_RATIO} Val=${VAL_RATIO}"
echo "Random seed: $RANDOM_SEED"
echo "============================================================================"
echo ""

# Validate split ratios
python3 <<EOF
train_ratio = $TRAIN_RATIO
test_ratio = $TEST_RATIO
val_ratio = $VAL_RATIO
total_ratio = train_ratio + test_ratio + val_ratio
if abs(total_ratio - 1.0) > 1e-6:
    print(f"Error: Split ratios must sum to 1.0, got {total_ratio}")
    exit(1)
EOF

# Check if input directory exists
if [ ! -d "$SODA_A_DIR" ]; then
    echo "Error: Input directory not found: $SODA_A_DIR"
    exit 1
fi

# Check if Annotations directory exists
if [ ! -d "$SODA_A_DIR/Annotations" ]; then
    echo "Error: Annotations directory not found: $SODA_A_DIR/Annotations"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Activate virtual environment if it exists
if [ -f "$PROJECT_ROOT/../venv/bin/activate" ]; then
    source "$PROJECT_ROOT/../venv/bin/activate"
    echo "Activated virtual environment"
fi

# Step 1: Convert all annotations to a single COCO file
echo ""
echo "============================================================================"
echo "Step 1: Converting per-image JSON to COCO format"
echo "============================================================================"

# Check if we have a single annotations directory or split directories
if [ -d "$SODA_A_DIR/Annotations/train" ] || [ -d "$SODA_A_DIR/Annotations/test" ] || [ -d "$SODA_A_DIR/Annotations/val" ]; then
    # Already split - convert each split separately and combine
    echo "Found pre-split annotations (train/test/val)"
    
    # Convert each split separately
    ALL_COCO_FILES=()
    for split in train test val; do
        if [ -d "$SODA_A_DIR/Annotations/$split" ]; then
            echo "Converting $split split..."
            SPLIT_COCO_FILE="$OUTPUT_DIR/temp_${split}.coco.json"
            python3 "$SCRIPT_DIR/convert_soda_a_to_coco.py" \
                "$SODA_A_DIR/Annotations/$split" \
                --output-file "$SPLIT_COCO_FILE" \
                --split "$split"
            ALL_COCO_FILES+=("$SPLIT_COCO_FILE")
        fi
    done
    
    # Combine all COCO files into one
    echo "Combining all splits into single COCO file..."
    python3 <<EOF
import json
from pathlib import Path

all_images = []
all_annotations = []
all_categories = None
next_image_id = 1
next_ann_id = 1

coco_files = [Path("$OUTPUT_DIR/temp_train.coco.json"),
               Path("$OUTPUT_DIR/temp_test.coco.json"),
               Path("$OUTPUT_DIR/temp_val.coco.json")]

for coco_file in coco_files:
    if not coco_file.exists():
        continue
    with open(coco_file) as f:
        data = json.load(f)
    
    # Remap image IDs
    image_id_map = {}
    for img in data['images']:
        old_id = img['id']
        img['id'] = next_image_id
        image_id_map[old_id] = next_image_id
        all_images.append(img)
        next_image_id += 1
    
    # Remap annotation IDs and image_ids
    for ann in data['annotations']:
        ann['id'] = next_ann_id
        ann['image_id'] = image_id_map[ann['image_id']]
        all_annotations.append(ann)
        next_ann_id += 1
    
    # Categories should be the same, use from first file
    if all_categories is None:
        all_categories = data['categories']

# Sort by ID for reproducibility
all_images = sorted(all_images, key=lambda x: x['id'])
all_annotations = sorted(all_annotations, key=lambda x: x['id'])

combined_coco = {
    'info': {'description': 'Combined COCO dataset'},
    'licenses': [],
    'categories': all_categories,
    'images': all_images,
    'annotations': all_annotations
}

output_file = Path("$OUTPUT_DIR/all_annotations.coco.json")
with open(output_file, 'w') as f:
    json.dump(combined_coco, f, indent=2)

print(f"Combined {len(all_images)} images and {len(all_annotations)} annotations")
EOF
    
    # Clean up temporary files
    rm -f "$OUTPUT_DIR/temp_"*.coco.json
else
    # Single annotations directory - convert directly
    echo "Found single annotations directory"
    python3 "$SCRIPT_DIR/convert_soda_a_to_coco.py" \
        "$SODA_A_DIR/Annotations" \
        --output-file "$OUTPUT_DIR/all_annotations.coco.json" \
        --split "train"
fi

if [ ! -f "$OUTPUT_DIR/all_annotations.coco.json" ]; then
    echo "Error: Failed to create COCO annotation file"
    exit 1
fi

echo "✓ Conversion complete"

# Step 2: Split into train/test/val (70/15/15)
echo ""
echo "============================================================================"
echo "Step 2: Splitting dataset into Train/Test/Val (70/15/15)"
echo "============================================================================"

# Build command with optional --copy-images flag
SPLIT_CMD=(
    python3 "$SCRIPT_DIR/split_coco_three_way.py"
    "$OUTPUT_DIR/all_annotations.coco.json"
    --output-dir "$OUTPUT_DIR"
    --train-ratio "$TRAIN_RATIO"
    --test-ratio "$TEST_RATIO"
    --val-ratio "$VAL_RATIO"
    --seed "$RANDOM_SEED"
)

# Add --copy-images flag only if COPY_IMAGES is true
if [ "$COPY_IMAGES" = true ]; then
    SPLIT_CMD+=(--copy-images)
fi

"${SPLIT_CMD[@]}"

echo "✓ Split complete"

# Rename annotation files to standard name (_annotations.coco.json)
echo "Renaming annotation files to standard format..."
for split in train test val; do
    if [ -f "$OUTPUT_DIR/$split/all_annotations.coco.json" ]; then
        mv "$OUTPUT_DIR/$split/all_annotations.coco.json" "$OUTPUT_DIR/$split/_annotations.coco.json"
        echo "  Renamed $split/all_annotations.coco.json -> $split/_annotations.coco.json"
    fi
done

# Step 3: Copy images to respective split directories
if [ "$COPY_IMAGES" = true ]; then
    echo ""
    echo "============================================================================"
    echo "Step 3: Copying images to split directories"
    echo "============================================================================"
    
    IMAGES_DIR="$SODA_A_DIR/Images"
    if [ ! -d "$IMAGES_DIR" ]; then
        echo "Warning: Images directory not found: $IMAGES_DIR"
        echo "Skipping image copying"
    else
        # Load COCO annotations to map image filenames to splits
        for split in train test val; do
            COCO_FILE="$OUTPUT_DIR/$split/_annotations.coco.json"
            if [ -f "$COCO_FILE" ]; then
                echo "Copying images for $split split..."
                python3 - <<EOF
import json
import shutil
from pathlib import Path

coco_file = Path("$COCO_FILE")
images_dir = Path("$IMAGES_DIR")
output_split_dir = Path("$OUTPUT_DIR/$split")
split_name = "$split"

with open(coco_file) as f:
    coco_data = json.load(f)

image_files = sorted([img['file_name'] for img in coco_data['images']])
copied = 0
for img_file in image_files:
    src = images_dir / img_file
    if src.exists():
        dst = output_split_dir / img_file
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

print(f"  Copied {copied} images to {split_name}/")
EOF
            fi
        done
        echo "✓ Images copied"
    fi
fi

# Step 4: Create patches with max size 512x512
echo ""
echo "============================================================================"
echo "Step 4: Creating patches (max size ${MAX_PATCH_SIZE}x${MAX_PATCH_SIZE})"
echo "============================================================================"

for split in train test val; do
    INPUT_ANNOTATION="$OUTPUT_DIR/$split/_annotations.coco.json"
    if [ ! -f "$INPUT_ANNOTATION" ]; then
        echo "Warning: $split annotation file not found, skipping..."
        continue
    fi
    
    echo ""
    echo "Processing $split split..."
    
    # Calculate patch grid size based on image dimensions to ensure max 512x512
    # We'll use a Python script to determine the optimal grid
    python3 <<EOF
import json
import math
from pathlib import Path

annotation_file = Path("$INPUT_ANNOTATION")
with open(annotation_file) as f:
    coco_data = json.load(f)

# Find the largest image dimensions
max_w = 0
max_h = 0
for img in coco_data['images']:
    max_w = max(max_w, img['width'])
    max_h = max(max_h, img['height'])

# Calculate number of patches needed to ensure max patch size
max_patch_size = $MAX_PATCH_SIZE
num_patches_x = math.ceil(max_w / max_patch_size)
num_patches_y = math.ceil(max_h / max_patch_size)

# Ensure at least 1 patch
num_patches_x = max(1, num_patches_x)
num_patches_y = max(1, num_patches_y)

# Verify patch size won't exceed max
actual_patch_w = max_w // num_patches_x
actual_patch_h = max_h // num_patches_y
if actual_patch_w > max_patch_size or actual_patch_h > max_patch_size:
    # Increase grid size if needed
    num_patches_x = math.ceil(max_w / max_patch_size)
    num_patches_y = math.ceil(max_h / max_patch_size)

print(f"  Largest image: {max_w}x{max_h}")
print(f"  Patch grid: {num_patches_x}x{num_patches_y} = {num_patches_x * num_patches_y} patches per image")
print(f"  Estimated patch size: ~{max_w // num_patches_x}x{max_h // num_patches_y}")

# Save grid size to temp file
with open("$OUTPUT_DIR/temp_${split}_grid.txt", "w") as f:
    f.write(f"{num_patches_x}\n{num_patches_y}\n")
EOF
    
    # Read grid size
    if [ -f "$OUTPUT_DIR/temp_${split}_grid.txt" ]; then
        PATCHES_X=$(head -1 "$OUTPUT_DIR/temp_${split}_grid.txt")
        PATCHES_Y=$(tail -1 "$OUTPUT_DIR/temp_${split}_grid.txt")
        rm "$OUTPUT_DIR/temp_${split}_grid.txt"
    else
        # Fallback: use conservative grid (4x4 for large images)
        PATCHES_X=4
        PATCHES_Y=4
    fi
    
    PATCHED_OUTPUT="$OUTPUT_DIR/${split}_patches/_annotations.coco.json"
    PATCHED_IMAGE_DIR="$OUTPUT_DIR/${split}_patches"
    
    python3 "$SCRIPT_DIR/create_patched_dataset.py" \
        "$INPUT_ANNOTATION" \
        --output-file "$PATCHED_OUTPUT" \
        --patches-x "$PATCHES_X" \
        --patches-y "$PATCHES_Y" \
        --min-overlap "$MIN_OVERLAP_RATIO" \
        --image-dir "$OUTPUT_DIR/$split" \
        --output-image-dir "$PATCHED_IMAGE_DIR"
    
    echo "✓ $split patches created"
done

# Step 5: Print summary
echo ""
echo "============================================================================"
echo "Dataset Preparation Complete!"
echo "============================================================================"
echo ""
echo "Output directory structure:"
echo "  $OUTPUT_DIR/"
for split in train test val; do
    if [ -f "$OUTPUT_DIR/$split/_annotations.coco.json" ]; then
        IMG_COUNT=$(find "$OUTPUT_DIR/$split" -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
        echo "    $split/"
        echo "      _annotations.coco.json"
        echo "      ($IMG_COUNT images)"
    fi
    if [ -f "$OUTPUT_DIR/${split}_patches/_annotations.coco.json" ]; then
        PATCH_COUNT=$(find "$OUTPUT_DIR/${split}_patches" -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
        echo "    ${split}_patches/"
        echo "      _annotations.coco.json"
        echo "      ($PATCH_COUNT patch images)"
    fi
done
echo ""
echo "Split ratios:"
python3 <<EOF
import json
from pathlib import Path

output_dir = Path("$OUTPUT_DIR")
for split in ['train', 'test', 'val']:
    ann_file = output_dir / split / '_annotations.coco.json'
    if ann_file.exists():
        with open(ann_file) as f:
            data = json.load(f)
        print(f"  {split:5}: {len(data['images']):5} images, {len(data['annotations']):6} annotations")
EOF

echo ""
echo "============================================================================"
