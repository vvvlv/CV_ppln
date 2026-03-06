# Split COCO Dataset Script

This script splits a COCO format dataset into train and validation splits.

## Usage

### Basic Usage

Split the train dataset into 80% train and 20% validation:

```bash
cd /home/vlv/Documents/master/computervision/CV_ppln
python scripts/split_coco_dataset.py \
    /home/vlv/Documents/master/computervision/SODA_dataset/SODA_A_05_small/train/_annotations.coco.json \
    --output-dir /home/vlv/Documents/master/computervision/SODA_dataset/SODA_A_05_small \
    --val-ratio 0.2 \
    --seed 42
```

### Options

- `annotation_file`: Path to the COCO annotation file to split
- `--output-dir`: Directory where train/val splits will be created (will create `train/` and `val/` subdirectories)
- `--val-ratio`: Ratio for validation split (default: 0.2 = 20%)
- `--seed`: Random seed for reproducibility (default: 42)
- `--copy-images`: If set, copies images to train/val directories. If not set, only creates annotation files (images stay in original location)

### Example: Split SODA Dataset

```bash
# Split train into 80% train, 20% val
python scripts/split_coco_dataset.py \
    /home/vlv/Documents/master/computervision/SODA_dataset/SODA_A_05_small/train/_annotations.coco.json \
    --output-dir /home/vlv/Documents/master/computervision/SODA_dataset/SODA_A_05_small \
    --val-ratio 0.2 \
    --seed 42
```

This will:
1. Read the train annotation file
2. Split images into train (80%) and val (20%)
3. Create `train/_annotations.coco.json` (new split) and `val/_annotations.coco.json`
4. Keep images in their original location (unless `--copy-images` is used)

### After Splitting

After running the script, update your dataset config to use the new val split:

```yaml
paths:
  root: "/home/vlv/Documents/master/computervision/SODA_dataset/SODA_A_05_small"
  train: "/home/vlv/Documents/master/computervision/SODA_dataset/SODA_A_05_small/train"
  val: "/home/vlv/Documents/master/computervision/SODA_dataset/SODA_A_05_small/val"  # Now uses real val split
  test: "/home/vlv/Documents/master/computervision/SODA_dataset/SODA_A_05_small/test"
```

## Notes

- The script preserves the original train annotation file (it creates new ones in the output directory)
- Images are referenced by filename, so they can stay in the original location
- The split is deterministic when using the same seed
- Categories and metadata are preserved in both splits
