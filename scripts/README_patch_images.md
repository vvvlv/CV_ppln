# Split Images to Patches Script

This script splits high-resolution images into smaller patches while preserving COCO annotations. This is useful for:
- Reducing GPU memory usage
- Training on smaller images
- Handling very large images that don't fit in memory

## Usage

### Basic Usage

Split images into 640x640 patches with no overlap:

```bash
cd /home/vlv/Documents/master/computervision/CV_ppln

python scripts/split_images_to_patches.py \
    /path/to/train/_annotations.coco.json \
    /path/to/train \
    --output-dir /path/to/train_patches \
    --patch-size 640 640
```

### With Overlap (Recommended)

Use overlap to avoid cutting objects in half:

```bash
python scripts/split_images_to_patches.py \
    /path/to/train/_annotations.coco.json \
    /path/to/train \
    --output-dir /path/to/train_patches \
    --patch-size 640 640 \
    --overlap 0.25
```

### Example: Process SODA Dataset

```bash
# Process train split
python scripts/split_images_to_patches.py \
    /home/vlv/Documents/master/computervision/SODA_dataset/SODA_A_05_small/train/_annotations.coco.json \
    /home/vlv/Documents/master/computervision/SODA_dataset/SODA_A_05_small/train \
    --output-dir /home/vlv/Documents/master/computervision/SODA_dataset/SODA_A_05_small/train_patches \
    --patch-size 640 640 \
    --overlap 0.25 \
    --min-bbox-overlap 0.5

# Process test split
python scripts/split_images_to_patches.py \
    /home/vlv/Documents/master/computervision/SODA_dataset/SODA_A_05_small/test/_annotations.coco.json \
    /home/vlv/Documents/master/computervision/SODA_dataset/SODA_A_05_small/test \
    --output-dir /home/vlv/Documents/master/computervision/SODA_dataset/SODA_A_05_small/test_patches \
    --patch-size 640 640 \
    --overlap 0.25 \
    --min-bbox-overlap 0.5
```

## Options

- `annotation_file`: Path to COCO annotation file
- `image_dir`: Directory containing original images
- `--output-dir`: Output directory (will create `images/` subdirectory for patches)
- `--patch-size WIDTH HEIGHT`: Patch size in pixels (default: 640 640)
- `--overlap`: Overlap ratio between patches (0.0-1.0, default: 0.0)
  - 0.0 = no overlap
  - 0.25 = 25% overlap (recommended to avoid cutting objects)
- `--min-bbox-overlap`: Minimum bbox overlap to include in patch (0.0-1.0, default: 0.5)
  - Only bboxes with at least this overlap ratio are included in each patch
- `--keep-empty-patches`: Keep patches with no annotations (default: filters them out)

## How It Works

1. **Image Splitting**: Images are split into patches of specified size
2. **Bbox Adjustment**: Bounding boxes are adjusted to patch-relative coordinates
3. **Overlap Handling**: Objects that span multiple patches appear in multiple patches (with adjusted coordinates)
4. **Filtering**: Patches with no annotations are removed (unless `--keep-empty-patches` is used)

## Output Structure

```
output_dir/
├── images/              # Patch images
│   ├── image1_patch_0000.jpg
│   ├── image1_patch_0001.jpg
│   └── ...
└── _annotations.coco.json  # New annotation file with patch coordinates
```

## After Processing

Update your dataset config to use the patched dataset:

```yaml
paths:
  root: "/path/to/dataset"
  train: "/path/to/dataset/train_patches"
  val: "/path/to/dataset/val_patches"
  test: "/path/to/dataset/test_patches"
```

## Tips

1. **Patch Size**: Common sizes: 512x512, 640x640, 800x800
   - Smaller = less memory, more patches
   - Larger = more memory, fewer patches

2. **Overlap**: 0.25 (25%) is a good default
   - Prevents objects from being cut in half
   - Increases number of patches but improves object detection

3. **Min Bbox Overlap**: 0.5 (50%) is recommended
   - Only includes objects that are mostly visible in the patch
   - Prevents tiny partial objects

4. **Memory Considerations**:
   - 640x640 patches with batch_size=1 should work on 5-6GB GPUs
   - 512x512 patches are even safer
