# CV_ppln: Object Detection Pipeline for SODA-A with CFINet

A comprehensive, streamlined deep learning pipeline for training and evaluating object detection models on the SODA-A dataset using CFINet (Coarse-to-fine Proposal Generation and Imitation Learning Network). This system is built on PyTorch and MMDetection, providing a complete workflow from data loading to model evaluation.

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Installation](#installation)
4. [Configuration System](#configuration-system)
5. [Data Pipeline](#data-pipeline)
6. [Dataset Preparation](#dataset-preparation)
7. [Model Architecture](#model-architecture)
8. [Training Process](#training-process)
9. [Evaluation and Testing](#evaluation-and-testing)
10. [Utilities and Scripts](#utilities-and-scripts)
11. [Project Structure](#project-structure)
12. [Troubleshooting](#troubleshooting)

---

## Overview

### Purpose

This pipeline is designed specifically for **object detection** tasks using **COCO-format datasets**. It integrates CFINet (an MMDetection-based detector) with a clean, modular training and evaluation framework. The system has been streamlined to focus exclusively on detection, removing all segmentation-related code paths.

### Key Features

- **COCO Dataset Support**: Native handling of COCO-format annotations
- **CFINet Integration**: Seamless wrapper around MMDetection-based CFINet model
- **Modular Design**: Clean separation between data, models, training, and evaluation
- **Comprehensive Metrics**: mAP (mean Average Precision) calculation with per-class support
- **TensorBoard Integration**: Real-time visualization of training metrics
- **Checkpointing**: Automatic saving of best and last model checkpoints
- **Early Stopping**: Configurable early stopping based on validation metrics
- **Memory Profiling**: Optional memory usage analysis for debugging

### Supported Datasets

- **SODA-A**: Small Object Detection in Aerial imagery dataset (9 classes)
- Any COCO-format dataset with proper directory structure

---

## How the System Works

This section provides a comprehensive overview of how the entire system works, from dataset preparation to model training and evaluation.

### Complete Workflow Overview

The system follows this end-to-end workflow:

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. DATASET PREPARATION                                          │
│    (One-time setup, before training)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ SODA_A Format (per-image JSON files)                  │
    │   Annotations/train/00001.json, 00002.json, ...       │
    │   Images/train/00001.jpg, 00002.jpg, ...              │
    └──────────────────────────────────────────────────────┘
                              │
                              │ [convert_soda_a_to_coco.py]
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ COCO Format (single JSON per split)                   │
    │   train/_annotations.coco.json                         │
    │   val/_annotations.coco.json                           │
    │   test/_annotations.coco.json                          │
    └──────────────────────────────────────────────────────┘
                              │
                              │ [split_coco_three_way.py]
                              │ (70% train, 15% test, 15% val)
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ Split COCO Format                                      │
    │   train/_annotations.coco.json (70%)                  │
    │   test/_annotations.coco.json (15%)                    │
    │   val/_annotations.coco.json (15%)                      │
    └──────────────────────────────────────────────────────┘
                              │
                              │ [create_patched_dataset.py]
                              │ (Optional: create patches)
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ Patched COCO Format (512x512 max patches)             │
    │   train_patches/_annotations.coco.json                 │
    │   val_patches/_annotations.coco.json                  │
    │   test_patches/_annotations.coco.json                 │
    └──────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 2. CONFIGURATION SETUP                                          │
│    (Define experiment and dataset configs)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ Experiment Config (exp_cfinet_soda_512_patches.yaml)  │
    │   - References dataset config                          │
    │   - Defines training hyperparameters                   │
    │   - Specifies model type and settings                 │
    └──────────────────────────────────────────────────────┘
                              │
                              │ [load_config()]
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ Dataset Config (soda_a_coco_512_patches.yaml)        │
    │   - Defines dataset paths                             │
    │   - Specifies image preprocessing                     │
    │   - Sets number of classes                            │
    └──────────────────────────────────────────────────────┘
                              │
                              │ [Merged into single config]
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ Complete Config Dictionary                            │
    │   {dataset: {...}, model: {...}, training: {...}, ...}│
    └──────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 3. DATA LOADING PIPELINE                                         │
│    (During training initialization)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ COCODataset.__init__()                                │
    │   - Loads _annotations.coco.json                     │
    │   - Builds image_id → annotations mapping            │
    │   - Filters images with no annotations (train only)   │
    └──────────────────────────────────────────────────────┘
                              │
                              │ [Per sample: __getitem__()]
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ Image Loading & Preprocessing                          │
    │   1. Load image from disk (OpenCV)                    │
    │   2. Extract bboxes and labels from annotations       │
    │   3. Convert COCO bbox [x,y,w,h] → [x1,y1,x2,y2]     │
    │   4. Filter tiny bboxes (< min_bbox_size)             │
    │   5. Apply augmentations (if training)               │
    │   6. Resize image (if image_size specified)          │
    │   7. Adjust bbox coordinates for resize               │
    │   8. Normalize image (ImageNet stats)                │
    │   9. Convert to PyTorch tensors                       │
    └──────────────────────────────────────────────────────┘
                              │
                              │ [DataLoader collate_fn()]
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ Batch Dictionary                                      │
    │   {                                                    │
    │     'image': Tensor[B, C, H, W],                      │
    │     'bboxes': List[List[4]],  # Variable length      │
    │     'labels': List[List[int]], # Variable length      │
    │     'image_id': List[int],                            │
    │     'image_name': List[str],                          │
    │     'original_size': List[Tuple[H, W]]                │
    │   }                                                    │
    └──────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 4. MODEL INITIALIZATION                                         │
│    (During training initialization)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ create_model(config)                                  │
    │   - Looks up model type in registry                   │
    │   - Creates CFINetWrapper instance                    │
    └──────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ CFINetWrapper.__init__()                              │
    │   1. Loads MMDetection config (embedded or file)     │
    │   2. Updates num_classes in config                     │
    │   3. Creates con_queue_dir for FIRoIHead              │
    │   4. Builds model via mmdet.models.build_detector()   │
    │   5. Loads pretrained weights (if checkpoint provided)│
    └──────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ CFINet Model (MMDetection)                            │
    │   - ResNet-50 Backbone + FPN                         │
    │   - CRPNHead (Coarse-to-fine RPN)                     │
    │   - FIRoIHead (Feature Imitation ROI Head)           │
    └──────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 5. TRAINING LOOP                                                │
│    (Per epoch, iterating over batches)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ DetectionTrainer.train()                              │
    │   - Initializes optimizer (SGD/Adam)                 │
    │   - Initializes scheduler (Step/Cosine)              │
    │   - Sets up metrics tracking                          │
    └──────────────────────────────────────────────────────┘
                              │
                              │ [For each epoch]
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ Training Phase (model.train())                       │
    │   For each batch:                                    │
    │     1. Move images and targets to device             │
    │     2. Forward pass: model(images, targets=targets)   │
    │     3. Get loss dict from MMDetection                │
    │     4. Aggregate losses (sum all components)         │
    │     5. Backward pass: loss.backward()                │
    │     6. Optimizer step: optimizer.step()              │
    │     7. Log loss components                            │
    └──────────────────────────────────────────────────────┘
                              │
                              │ [After training phase]
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ Validation Phase (model.eval())                      │
    │   For each batch:                                    │
    │     1. Forward pass (no gradients)                    │
    │     2. Compute losses                                │
    │     3. Collect predictions and ground truth           │
    │     4. Compute metrics (mAP)                          │
    └──────────────────────────────────────────────────────┘
                              │
                              │ [After validation]
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ Post-Epoch Operations                                 │
    │   1. Log metrics to TensorBoard                       │
    │   2. Save metrics_history.yaml                        │
    │   3. Checkpointing:                                   │
    │      - Save best.pth if metric improved               │
    │      - Save last.pth (every epoch)                   │
    │   4. Early stopping check                             │
    │   5. Step learning rate scheduler                     │
    └──────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 6. INFERENCE / TESTING                                          │
│    (After training, on test set)                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ Load Trained Model                                    │
    │   - Load checkpoint (best.pth or last.pth)           │
    │   - Restore model weights                             │
    │   - Set model.eval()                                  │
    └──────────────────────────────────────────────────────┘
                              │
                              │ [For each test batch]
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ Inference                                             │
    │   1. Forward pass: model.predict(images)              │
    │   2. MMDetection returns detections                    │
    │   3. Format to pipeline format:                      │
    │      {bboxes: [N,4], labels: [N], scores: [N]}       │
    │   4. Filter by score_threshold                        │
    └──────────────────────────────────────────────────────┘
                              │
                              │ [After all test batches]
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ Evaluation                                            │
    │   1. Collect all predictions and ground truth         │
    │   2. Compute mAP (mean Average Precision)            │
    │   3. Save predictions.json with results               │
    └──────────────────────────────────────────────────────┘
```

### Detailed Component Interactions

#### 1. Configuration Resolution Flow

When you run `./train.sh exp_cfinet_soda_512_patches`, the system:

1. **Finds the experiment config**: `configs/experiments/exp_cfinet_soda_512_patches.yaml`
2. **Loads experiment config**: Reads YAML file into dictionary
3. **Resolves dataset reference**: 
   - Extracts `dataset: "configs/datasets/soda_a_coco_512_patches.yaml"`
   - Finds project root (directory containing `configs/`)
   - Resolves relative path: `project_root / "configs/datasets/soda_a_coco_512_patches.yaml"`
4. **Loads dataset config**: Reads dataset YAML file
5. **Merges configs**: Dataset config becomes `config['dataset']` key
6. **Result**: Single unified config dictionary used throughout training

**Example merged config structure:**
```python
{
    'name': 'exp_cfinet_soda_512_patches',
    'dataset': {
        'name': 'SODA_A_coco_512_patches',
        'paths': {
            'root': '/path/to/SODA_A_coco_512',
            'train': 'train_patches',
            'val': 'val_patches',
            'test': 'test_patches'
        },
        'image_size': [512, 512],
        'num_classes': 9,
        ...
    },
    'model': {
        'type': 'CFINet',
        'num_classes': 9,
        ...
    },
    'training': {
        'epochs': 12,
        'optimizer': {...},
        ...
    },
    ...
}
```

#### 2. Data Loading Flow

**Initialization (once per DataLoader):**
```python
# In create_dataloaders()
train_dataset = COCODataset(
    root_dir="/path/to/SODA_A_coco_512",
    split="train_patches",  # Custom split name
    annotation_file="_annotations.coco.json",
    image_size=(512, 512),
    ...
)
```

**COCODataset.__init__() does:**
1. Loads `root_dir/train_patches/_annotations.coco.json`
2. Parses JSON: extracts `images`, `annotations`, `categories`
3. Builds mappings:
   - `self.image_dict`: `{image_id: image_info}`
   - `self.image_to_anns`: `{image_id: [annotation1, annotation2, ...]}`
4. Filters images with no annotations (training only)
5. Stores preprocessing parameters

**Per-sample loading (__getitem__):**
```python
# When DataLoader requests sample[i]
1. Get image_id from self.image_ids[i]
2. Load image: cv2.imread(image_path) → numpy array [H, W, 3]
3. Get annotations: self.image_to_anns[image_id]
4. Extract bboxes: [x, y, w, h] → convert to [x1, y1, x2, y2]
5. Filter bboxes: remove if width < min_bbox_size or height < min_bbox_size
6. Apply augmentations (if training):
   - Horizontal flip (50% chance): flip image, adjust bbox x-coords
   - Brightness adjustment: multiply by random [0.9, 1.1]
7. Resize image: cv2.resize(image, (512, 512))
8. Adjust bbox coordinates: scale by (new_w/old_w, new_h/old_h)
9. Normalize: (image - mean) / std
10. Convert to tensor: torch.from_numpy(image).permute(2,0,1)
11. Return: {
       'image': Tensor[3, 512, 512],
       'bboxes': Tensor[N, 4],  # [x1, y1, x2, y2]
       'labels': Tensor[N],
       'image_id': int,
       'image_name': str,
       'original_size': (H, W)
   }
```

**Batching (collate_fn):**
```python
# DataLoader groups samples into batches
batch = [
    sample1: {image: [3,512,512], bboxes: [5,4], labels: [5], ...},
    sample2: {image: [3,512,512], bboxes: [3,4], labels: [3], ...},
    sample3: {image: [3,512,512], bboxes: [7,4], labels: [7], ...},
]

# collate_fn() combines them:
{
    'image': Tensor[3, 3, 512, 512],  # Stacked images
    'bboxes': [Tensor[5,4], Tensor[3,4], Tensor[7,4]],  # List (variable length)
    'labels': [Tensor[5], Tensor[3], Tensor[7]],  # List (variable length)
    'image_id': [1, 2, 3],
    'image_name': ['img1.jpg', 'img2.jpg', 'img3.jpg'],
    'original_size': [(4800, 4800), (4800, 3670), (4800, 4800)]
}
```

#### 3. Model Forward Pass Flow

**Training mode (`model(images, targets=targets)`):**

```python
# In CFINetWrapper.forward()
1. Prepare img_metas (MMDetection format):
   [
       {'img_shape': (512, 512), 'ori_shape': (4800, 4800), ...},
       {'img_shape': (512, 512), 'ori_shape': (4800, 3670), ...},
       ...
   ]

2. Prepare gt_bboxes and gt_labels:
   gt_bboxes = [bboxes[0], bboxes[1], ...]  # List of tensors
   gt_labels = [labels[0], labels[1], ...]  # List of tensors

3. Call MMDetection model:
   loss_dict = model.forward_train(
       img=[images],  # List of tensors
       img_metas=[img_metas],
       gt_bboxes=gt_bboxes,
       gt_labels=gt_labels
   )

4. MMDetection returns:
   {
       'loss_rpn_cls': Tensor,
       'loss_rpn_bbox': Tensor,
       'loss_cls': Tensor,
       'loss_bbox': Tensor,
       'loss_imitation': Tensor,  # CFINet-specific
       ...
   }

5. Return loss_dict to trainer
```

**Inference mode (`model.predict(images)`):**

```python
# In CFINetWrapper.predict()
1. Prepare img_metas (same as training)

2. Call MMDetection model:
   results = model.simple_test(
       img=[images],
       img_metas=[img_metas]
   )

3. MMDetection returns:
   [
       [  # Per image
           [bbox1, bbox2, ...],  # Bboxes in [x1, y1, x2, y2, score] format
           [label1, label2, ...],  # Labels
           [score1, score2, ...]   # Scores
       ],
       ...
   ]

4. Format to pipeline format:
   {
       'bboxes': np.array([N, 4]),  # [x1, y1, x2, y2]
       'labels': np.array([N]),
       'scores': np.array([N])
   }

5. Filter by score_threshold

6. Return formatted results
```

#### 4. Training Loop Details

**Per-epoch training:**

```python
# In DetectionTrainer.train()
for epoch in range(num_epochs):
    # === TRAINING PHASE ===
    model.train()
    train_losses = []
    
    for batch in train_loader:
        # Move to device
        images = batch['image'].to(device)  # [B, 3, 512, 512]
        targets = {
            'bboxes': [bbox.to(device) for bbox in batch['bboxes']],
            'labels': [label.to(device) for label in batch['labels']]
        }
        
        # Forward pass
        loss_dict = model(images, targets=targets)
        
        # Aggregate losses
        total_loss = sum([
            v.item() if isinstance(v, torch.Tensor) else v
            for v in loss_dict.values()
            if v is not None
        ])
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Log losses
        train_losses.append(total_loss)
    
    # === VALIDATION PHASE ===
    model.eval()
    val_losses = []
    all_predictions = []
    all_ground_truth = []
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            targets = {...}
            
            # Forward pass (no gradients)
            loss_dict = model(images, targets=targets)
            val_losses.append(sum_losses(loss_dict))
            
            # Get predictions for metrics
            predictions = model.predict(images, score_threshold=0.05)
            all_predictions.extend(predictions)
            all_ground_truth.extend(targets)
    
    # === POST-EPOCH ===
    # Compute metrics
    val_mAP = mean_average_precision(all_predictions, all_ground_truth)
    
    # Log to TensorBoard
    tb_logger.log_scalar('train/loss', mean(train_losses), epoch)
    tb_logger.log_scalar('val/loss', mean(val_losses), epoch)
    tb_logger.log_scalar('val/mAP', val_mAP, epoch)
    
    # Save metrics history
    metrics_history[epoch] = {
        'train_loss': mean(train_losses),
        'val_loss': mean(val_losses),
        'val_mAP': val_mAP
    }
    
    # Checkpointing
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint('best.pth', epoch, model, optimizer, best_val_loss)
    
    save_checkpoint('last.pth', epoch, model, optimizer, best_val_loss)
    
    # Early stopping
    if no_improvement_for >= patience:
        print("Early stopping triggered")
        break
    
    # Step scheduler
    scheduler.step()
```

### Dataset Preparation Pipeline (512 Patches)

The `prepare_dataset.sh` script automates the complete dataset preparation workflow:

**Step 1: Convert per-image JSON to COCO format**
```bash
# For each split (train, test, val)
python convert_soda_a_to_coco.py \
    Annotations/train/ \
    --output-file temp_train.coco.json \
    --split train
```

**What happens:**
- Reads all `*.json` files in `Annotations/train/`
- Sorts files by name (for reproducibility)
- Converts polygon annotations to bounding boxes
- Merges into single COCO JSON file
- Sorts images, annotations, categories by ID (for reproducibility)

**Step 2: Combine and split**
```bash
# Combine all splits into one
python -c "
    # Merge train, test, val COCO files into all_annotations.coco.json
"

# Split into 70/15/15
python split_coco_three_way.py \
    all_annotations.coco.json \
    --output-dir OUTPUT_DIR \
    --train-ratio 0.70 \
    --test-ratio 0.15 \
    --val-ratio 0.15 \
    --seed 42 \
    --copy-images
```

**What happens:**
- Loads combined COCO file
- Randomly shuffles image IDs (with seed for reproducibility)
- Splits into train (70%), test (15%), val (15%)
- Creates new COCO JSON files for each split
- Copies images to respective split directories

**Step 3: Create patches**
```bash
# For each split
python create_patched_dataset.py \
    train/_annotations.coco.json \
    --output-file train_patches/_annotations.coco.json \
    --patches-x 10 \
    --patches-y 10 \
    --min-overlap 0.1 \
    --image-dir train \
    --output-image-dir train_patches
```

**What happens:**
- For each image:
  1. Calculate patch grid: `patches_x = ceil(image_width / MAX_PATCH_SIZE)`
  2. Calculate patch size with overlap
  3. For each patch:
     - Extract patch region from image
     - Transform bbox coordinates: `bbox_patch = bbox_original - [patch_x, patch_y, 0, 0]`
     - Calculate IoU between bbox and patch region
     - Keep bbox if IoU >= min_overlap_ratio
     - Clip bbox to patch boundaries
     - Save patch image
     - Create patch annotation entry
  4. Save all patch annotations to new COCO JSON file

**Result:**
- Original image: 4800x4800 → 100 patches (10x10 grid) of ~480x480 each
- Each patch has transformed bbox coordinates relative to patch
- Patches are saved as separate images
- New COCO annotation file references patch images

### Key Design Decisions

1. **Two-level config system**: Separates dataset-specific settings from training hyperparameters, allowing reuse of dataset configs across experiments.

2. **Custom split names**: Supports `train_patches`, `val_patches`, etc., allowing multiple dataset variants in the same directory structure.

3. **Variable-length bboxes**: Uses lists instead of padded tensors, avoiding memory waste and simplifying code.

4. **MMDetection integration**: Wraps MMDetection models rather than reimplementing, leveraging their optimized implementations.

5. **Reproducibility**: All scripts sort inputs/outputs deterministically, ensuring identical results across runs.

6. **Flexible image sizing**: Supports both original-size and fixed-size images, with automatic bbox coordinate adjustment.

---

## System Architecture

### High-Level Flow

```
┌─────────────────┐
│  Config Files   │  (YAML: experiment + dataset configs)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Config Loader  │  (loads and merges configs)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Pipeline  │  (COCODataset → DataLoader)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Builder  │  (CFINetWrapper from MMDetection)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ DetectionTrainer│  (training loop, validation, metrics)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Checkpoints   │  (best.pth, last.pth, metrics_history.yaml)
└─────────────────┘
```

### Component Breakdown

#### 1. **Configuration System** (`src/utils/config.py`)
- Loads experiment YAML files
- Resolves dataset config references (relative to project root)
- Merges experiment and dataset configurations into a single dict
- Handles path resolution for nested configs

#### 2. **Data Pipeline** (`src/data/`)
- **COCODataset** (`coco_dataset.py`): PyTorch Dataset for COCO-format annotations
  - Loads images and annotations from JSON files
  - Handles image preprocessing (resize, normalize)
  - Applies data augmentation (horizontal flip, brightness adjustment)
  - Converts COCO bbox format `[x, y, w, h]` to `[x1, y1, x2, y2]`
  - Filters out tiny bounding boxes below minimum size threshold
- **DataLoader Factory** (`__init__.py`): Creates train/val/test DataLoaders
  - Custom collate function for variable-length bbox lists
  - Configurable batch size, workers, pin_memory

#### 3. **Model System** (`src/models/`)
- **Model Registry** (`registry.py`): Decorator-based registration system
- **CFINet Wrapper** (`architectures/cfinet.py`): 
  - Wraps MMDetection's CFINet implementation
  - Handles MMDetection config loading and model building
  - Provides unified interface for training/inference
  - Manages CFINet-specific requirements (e.g., `con_queue_dir`)

#### 4. **Training System** (`src/training/`)
- **DetectionTrainer** (`detection_trainer.py`): Main training loop
  - Epoch-based training with validation
  - Loss aggregation from MMDetection loss dict
  - Optimizer and scheduler management
  - Checkpointing (best/last based on monitored metric)
  - Early stopping with configurable patience
  - Metrics history tracking (saved to YAML)
  - TensorBoard logging
- **Detection Metrics** (`detection_metrics.py`): Evaluation metrics
  - IoU calculation between bounding boxes
  - Average Precision (AP) per class
  - Mean Average Precision (mAP) across classes
  - 11-point interpolation for AP calculation

#### 5. **Utilities** (`src/utils/`)
- **TensorBoard Logger**: Logs metrics, learning rates, hyperparameters
- **Memory Profiler**: Analyzes GPU memory usage (optional)
- **Helpers**: Seed setting, parameter counting

---

## Installation

### Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: 11.3+ (for GPU support)
- **pip** or **conda** package manager

### Step-by-Step Installation

#### 1. Install Dependencies

```bash
cd /home/vlv/Documents/master/computervision/CV_ppln
pip install -r requirements.txt
```

This installs all required dependencies:
- PyTorch 2.8.0+ (with CUDA support)
- torchvision 0.23.0+
- numpy, opencv-python, pyyaml, tqdm
- tensorboard
- MMDetection (mmcv-full>=1.5.0, mmdet>=2.26.0)

**Note**: The system is self-contained. MMDetection is installed via `requirements.txt`, and CFINet model configurations are loaded from config files (no separate CFINet package installation needed).

#### 2. Verify Installation

```bash
python -c "
import torch
import mmdet
from mmcv import Config
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'MMDetection: {mmdet.__version__}')
"
```

Expected output:
```
PyTorch: 2.8.0
CUDA available: True
MMDetection: 2.26.0
```

### Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Configuration System

The pipeline uses a **two-level configuration system**: dataset configs and experiment configs.

### Dataset Configuration

Located in `configs/datasets/`, these files define dataset-specific settings:

**Example: `configs/datasets/soda_a_05_small.yaml`**
```yaml
name: "SODA_A_05_small"
description: "SODA-A dataset small subset (COCO format)"

format: "coco"  # Indicates COCO-format detection dataset

paths:
  root: "/path/to/SODA_dataset/SODA_A_05_small"
  train: "/path/to/SODA_dataset/SODA_A_05_small/train"
  val: "/path/to/SODA_dataset/SODA_A_05_small/test"  # Using test as val
  test: "/path/to/SODA_dataset/SODA_A_05_small/test"

annotation_file: "_annotations.coco.json"  # COCO annotation file name

# Image preprocessing
image_size: null  # Use original size, or [H, W] to resize
num_channels: 3
num_classes: 9  # SODA-A has 9 object classes

# Normalization (ImageNet stats)
mean: [0.485, 0.456, 0.406]
std: [0.229, 0.224, 0.225]

# Minimum bbox size to keep (filter tiny boxes)
min_bbox_size: 1
```

**Key Fields:**
- `format: "coco"`: Required to trigger detection pipeline
- `paths.root`: Root directory containing train/val/test splits
- `annotation_file`: Name of COCO JSON file in each split directory
- `image_size`: `null` for original size, or `[height, width]` to resize
- `num_classes`: Number of object classes in the dataset

### Experiment Configuration

Located in `configs/experiments/`, these files define training hyperparameters and model settings:

**Example: `configs/experiments/exp_cfinet_soda.yaml`**
```yaml
name: "exp_cfinet_soda"
description: "CFINet object detection on SODA-A dataset"

# Reference to dataset config (relative to project root)
dataset: "configs/datasets/soda_a_05_small.yaml"

# Data loading
data:
  batch_size: 1  # Adjust based on GPU memory
  num_workers: 2
  pin_memory: true
  
  augmentation:
    enabled: true
    horizontal_flip: 0.5
    vertical_flip: 0.0
    brightness: 0.1
  
  preprocessing:
    normalize: true

# Model architecture
model:
  type: "CFINet"  # Must match registered model name
  config_file: null  # null = use default embedded config. Or provide path to custom config file.
  checkpoint: null  # Optional: path to pretrained checkpoint
  num_classes: 9
  device: "cuda"

# Training settings
training:
  epochs: 12
  optimizer:
    type: "sgd"  # or "adam"
    learning_rate: 0.01
    momentum: 0.9
    weight_decay: 0.0001
  scheduler:
    type: "step"  # or "cosine"
    step_size: 8
    gamma: 0.1
  metrics:
    - "mAP"
  checkpoint:
    save_best: true
    save_last: true
    monitor: "val_loss"  # Metric to monitor for best checkpoint
  early_stopping:
    enabled: true
    patience: 5
    monitor: "val_loss"
    mode: "min"  # "min" for loss, "max" for mAP

# Output
output:
  dir: "outputs/experiments/exp_cfinet_soda"
  save_predictions: true

# Logging
logging:
  tensorboard: true
  log_images: false
  log_activations: false

seed: 42
device: "cuda"
```

**Key Fields:**
- `dataset`: Path to dataset config (relative to project root)
- `model.config_file`: Optional path to custom MMDetection config file. If `null`, uses the default embedded config from `cfinet.py`. If provided, can be relative to `src/models/architectures/` or absolute path.
- `model.num_classes`: Must match dataset `num_classes`
- `training.checkpoint.monitor`: Metric to track for best checkpoint
- `training.early_stopping.mode`: `"min"` for loss, `"max"` for mAP

### Model Configuration

Model definitions are embedded directly in the model architecture implementations in `src/models/architectures/`.

**CFINet Model:**
- The CFINet model is defined in `src/models/architectures/cfinet.py`
- Defines CFINet architecture: ResNet-50 backbone, FPN neck, CRPNHead, FIRoIHead
- Contains model, training, and testing configurations
- `num_classes` is automatically set from the experiment config
- No external config files needed

**Key Features:**
- Self-contained: Model definition is embedded in the model implementation
- All model specifications are within the project, in the architectures folder
- Each model architecture file defines its own model (CFINet, or future models)
- Optional: You can still provide a custom `config_file` path if you want to override the model's built-in config

### How Configs Are Loaded

1. **Experiment config** is loaded from YAML file
2. **Dataset reference** (`dataset: "configs/datasets/..."`) is resolved relative to project root
3. **Dataset config** is loaded and merged into experiment config
4. **Model config** path (`model.config_file`) is resolved relative to project root
5. Final merged config is used throughout the pipeline

The config loaders automatically find the project root (directory containing `configs/`) and resolve relative paths correctly.

---

## Data Pipeline

### COCO Dataset Format

The pipeline expects COCO-format datasets with this directory structure:

```
<SODA_ROOT>/
├── train/
│   ├── _annotations.coco.json
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── val/                      # Optional (if missing, test is used as val)
│   ├── _annotations.coco.json
│   ├── image101.jpg
│   └── ...
└── test/
    ├── _annotations.coco.json
    ├── image201.jpg
    └── ...
```

### COCO Annotation Format

Each `_annotations.coco.json` file must follow the COCO format:

```json
{
  "images": [
    {"id": 1, "file_name": "image001.jpg", "width": 640, "height": 480},
    ...
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 150, 50, 30],  // [x, y, width, height]
      "area": 1500,
      "iscrowd": 0
    },
    ...
  ],
  "categories": [
    {"id": 1, "name": "class1"},
    ...
  ]
}
```

### Data Loading Process

1. **COCODataset Initialization**:
   - Loads annotation JSON file for the split
   - Builds mappings: `image_id → image_info`, `image_id → annotations[]`
   - Filters images with no annotations (for training split only)

2. **`__getitem__` (per sample)**:
   - Loads image from disk (OpenCV)
   - Extracts bounding boxes and labels from annotations
   - Converts COCO bbox format `[x, y, w, h]` → `[x1, y1, x2, y2]`
   - Filters boxes below `min_bbox_size` threshold
   - Applies augmentations (if enabled and training split)
   - Resizes image and adjusts bbox coordinates (if `image_size` specified)
   - Normalizes image (ImageNet stats)
   - Converts to PyTorch tensors

3. **DataLoader Collation**:
   - Custom collate function handles variable-length bbox lists
   - Stacks images into batch tensor `[B, C, H, W]`
   - Keeps bboxes and labels as lists (one per image)

### Data Augmentation

Supported augmentations (configured in experiment config):

- **Horizontal Flip**: `horizontal_flip: 0.5` (50% probability)
  - Flips image horizontally
  - Adjusts bbox x-coordinates: `x1_new = width - x2_old`, `x2_new = width - x1_old`

- **Vertical Flip**: `vertical_flip: 0.5` (50% probability)
  - Flips image vertically
  - Adjusts bbox y-coordinates: `y1_new = height - y2_old`, `y2_new = height - y1_old`

- **Brightness**: `brightness: 0.1` (10% random adjustment)
  - Multiplies image by random factor `[0.9, 1.1]`
  - Clips to valid range `[0, 255]`

Augmentations are **only applied during training** (disabled for val/test splits).

---

## Dataset Preparation

This section covers preparing the SODA_A dataset for training, including converting from its native format to COCO format and creating patched versions for less powerful systems.

### SODA_A Dataset Structure

The SODA_A dataset comes in a non-standard format:

```
SODA_A/
├── Annotations/
│   ├── train/          # Per-image JSON files (00001.json, 00002.json, ...)
│   ├── val/
│   └── test/
└── Images/             # Image files (may be organized differently)
```

Each per-image JSON file contains:
- `images`: Single image dict with `file_name`, `height`, `width`, `id`
- `annotations`: List of annotations with `poly` (polygon coordinates), `area`, `category_id`
- `categories`: Category definitions

### Converting to COCO Format

The pipeline expects standard COCO format:
- Single JSON file per split (not per-image)
- Bbox format: `[x, y, width, height]` (not polygons)
- Standard structure: `images`, `annotations`, `categories` as lists

#### Script: `convert_soda_a_to_coco.py`

Converts per-image JSON format to standard COCO format.

**Usage:**
```bash
python scripts/convert_soda_a_to_coco.py \
    Annotations/train/ \
    --output-file train/_annotations.coco.json \
    --split train
```

**What it does:**
- Reads all JSON files in the specified directory
- Converts polygon annotations to bounding boxes (COCO format)
- Merges all per-image annotations into a single COCO JSON file
- Preserves categories and image metadata

### Creating Patched Datasets

For training on less powerful systems, you can create patched versions of the dataset by dividing images into subpatches. This reduces memory requirements and increases the number of training samples.

#### How Patching Works

**The Challenge:**
When you divide an image into patches, bbox coordinates must be transformed from original image coordinates to patch-relative coordinates.

**The Solution:**
1. **Coordinate Transformation**: For each patch, subtract the patch's top-left corner from bbox coordinates
   - Original bbox: `[x, y, w, h]` in image coordinates
   - Patch offset: `[patch_x, patch_y]`
   - Transformed bbox: `[x - patch_x, y - patch_y, w, h]` in patch coordinates

2. **Overlap Filtering**: Only keep bboxes that overlap with the patch
   - Calculate IoU (Intersection over Union) between bbox and patch region
   - Keep bbox if IoU >= threshold (default: 0.1 = 10% overlap)

3. **Edge Handling**: Handle bboxes that cross patch boundaries
   - Clip bbox to patch boundaries
   - Recalculate area in patch coordinates

**Example:**
```
Original Image: 2000x2000
Bbox: [1500, 500, 200, 300]  # x, y, w, h in original image

Patch 1,0 (top-right): [1000, 0, 1000, 1000]
  - Bbox overlaps with patch? Yes (IoU > 0.1)
  - Transformed bbox: [500, 500, 200, 300]  # Subtract patch offset
  - New area: 200 * 300 = 60000 (in patch coordinates)
```

#### Script: `create_patched_dataset.py`

Creates a patched version of a COCO dataset by dividing images into subpatches.

**Usage:**
```bash
python scripts/create_patched_dataset.py \
    train/_annotations.coco.json \
    --output-file train_patches/_annotations.coco.json \
    --patches-x 2 \
    --patches-y 2 \
    --min-overlap 0.1 \
    --image-dir train \
    --output-image-dir train_patches
```

**Parameters:**
- `--patches-x`: Number of patches horizontally (default: 2)
- `--patches-y`: Number of patches vertically (default: 2)
- `--min-overlap`: Minimum IoU ratio to keep a bbox in a patch (default: 0.1)
  - Higher values = more strict (only bboxes mostly inside patch)
  - Lower values = more lenient (includes bboxes partially overlapping)

**What it does:**
- Divides each image into a grid of patches (e.g., 2x2 = 4 patches per image)
- Transforms bbox coordinates to patch-relative coordinates
- Filters out bboxes that don't overlap with each patch (based on IoU threshold)
- Creates new patch images and saves them
- Generates a new COCO annotation file for the patches

#### Master Script: `prepare_soda_a_dataset.py`

Orchestrates the entire preparation process.

**Usage:**
```bash
# Basic conversion (no patches)
python scripts/prepare_soda_a_dataset.py \
    --soda-a-dir /path/to/SODA_dataset/SODA_A \
    --output-dir /path/to/output \
    --copy-images

# With patching
python scripts/prepare_soda_a_dataset.py \
    --soda-a-dir /path/to/SODA_dataset/SODA_A \
    --output-dir /path/to/output \
    --create-patches \
    --patches-x 2 \
    --patches-y 2 \
    --min-overlap 0.1 \
    --copy-images
```

**What it does:**
1. Converts all splits (train/val/test) from per-image JSON to COCO format
2. Optionally copies images to output directory
3. Optionally creates patched versions of each split

### Complete Workflow

#### Step 1: Convert SODA_A to COCO Format

```bash
cd /home/vlv/Documents/master/computervision/CV_ppln

python scripts/prepare_soda_a_dataset.py \
    --soda-a-dir /home/vlv/Documents/master/computervision/SODA_dataset/SODA_A \
    --output-dir /home/vlv/Documents/master/computervision/SODA_dataset/SODA_A_coco \
    --copy-images
```

This creates:
```
SODA_A_coco/
├── train/
│   ├── _annotations.coco.json
│   └── (image files)
├── val/
│   ├── _annotations.coco.json
│   └── (image files)
└── test/
    ├── _annotations.coco.json
    └── (image files)
```

#### Step 2: Create Patched Version (Optional)

```bash
python scripts/prepare_soda_a_dataset.py \
    --soda-a-dir /home/vlv/Documents/master/computervision/SODA_dataset/SODA_A \
    --output-dir /home/vlv/Documents/master/computervision/SODA_dataset/SODA_A_coco \
    --create-patches \
    --patches-x 2 \
    --patches-y 2 \
    --min-overlap 0.1 \
    --copy-images
```

This creates additional directories:
```
SODA_A_coco/
├── train/
├── train_patches/        # 2x2 patches of train images
│   ├── _annotations.coco.json
│   └── (patch image files)
├── val/
├── val_patches/          # 2x2 patches of val images
└── test/
```

#### Step 3: Update Dataset Config

Update your dataset config file (e.g., `configs/datasets/soda_a_05_small.yaml`) to point to the new location:

```yaml
paths:
  root: /home/vlv/Documents/master/computervision/SODA_dataset/SODA_A_coco

annotation_file: _annotations.coco.json
```

For patched version:
```yaml
paths:
  root: /home/vlv/Documents/master/computervision/SODA_dataset/SODA_A_coco

annotation_file: _annotations.coco.json
# Use train_patches, val_patches, test_patches splits
```

### Benefits and Limitations of Patching

**Benefits:**
1. **Reduced Memory**: Smaller images = less GPU memory needed
2. **Faster Training**: Smaller images = faster forward/backward passes
3. **More Training Samples**: 1 image → N patches (e.g., 2x2 = 4x more samples)
4. **Better for Small Objects**: Patches focus on smaller regions

**Limitations:**
1. **Context Loss**: Patches lose global context of full image
2. **Split Objects**: Objects crossing patch boundaries are split
3. **Storage**: Patched dataset uses more disk space (N patches per image)

### Reproducibility

The dataset preparation scripts are designed to be fully reproducible. The following measures ensure deterministic output:

1. **File Ordering**: Input JSON files are sorted by filename before processing
2. **Image Ordering**: Images are sorted by ID in the output COCO files
3. **Annotation Ordering**: Annotations are sorted by ID within each image and globally
4. **Category Ordering**: Categories are sorted by ID
5. **Patch ID Generation**: Patch IDs are generated deterministically based on sorted image IDs

**Important**: Running the scripts multiple times with the same input will produce identical output files (byte-for-byte), ensuring reproducible experiments.

### Troubleshooting Dataset Preparation

**Images not found:**
- Check that images are in `Images/` or `Images/train/`, `Images/val/`, `Images/test/`
- Use `--copy-images` to copy images to the output directory
- Manually specify image directory with `--image-dir`

**No annotations in patches:**
- Lower `--min-overlap` value (e.g., 0.05)
- Check that bboxes actually overlap with patches
- Verify original annotations are correct

**Out of memory:**
- Process one split at a time
- Reduce patch grid size (e.g., 1x2 instead of 2x2)
- Process images in smaller batches

---

## Model Architecture

### CFINet Overview

CFINet (Coarse-to-fine Proposal Generation and Imitation Learning Network) is a two-stage object detector built on Faster R-CNN. It uses:

- **Backbone**: ResNet-50 with FPN (Feature Pyramid Network)
- **RPN Head**: CRPNHead (Coarse-to-fine RPN Head)
- **ROI Head**: FIRoIHead (Feature Imitation ROI Head)

### CFINet Wrapper Implementation

The `CFINetWrapper` (`src/models/architectures/cfinet.py`) provides a bridge between MMDetection and this pipeline:

1. **Initialization**:
   - If `config_file` is provided: loads MMDetection config from file (relative to `src/models/architectures/` or absolute path)
   - If `config_file` is `null`: uses CFINet model config from `CFINetWrapper._get_model_config()`
   - Updates `num_classes` in config to match dataset
   - Creates `con_queue_dir` for FIRoIHead (if needed)
   - Builds model using `mmdet.models.build_detector()`
   - Loads pretrained checkpoint (if provided)

2. **Forward Pass** (training mode):
   - Receives images `[B, C, H, W]` and targets dict
   - Prepares `img_metas` (image metadata for MMDetection)
   - Calls `model.forward_train()` with MMDetection format
   - Returns loss dictionary from MMDetection

3. **Forward Pass** (inference mode):
   - Calls `model.simple_test()` with images and metadata
   - Returns detection results in MMDetection format

4. **Predict Method**:
   - Formats MMDetection results into pipeline format:
     ```python
     {
       'bboxes': np.ndarray,  # [N, 4] in [x1, y1, x2, y2] format
       'labels': np.ndarray,  # [N] class IDs
       'scores': np.ndarray   # [N] confidence scores
     }
     ```
   - Filters detections by score threshold

### Model Registry

Models are registered using the `@register_model` decorator:

```python
@register_model('CFINet')
class CFINetWrapper(nn.Module):
    ...
```

The registry allows dynamic model creation from config:
```python
model = get_model('CFINet', model_config)
```

---

## Training Process

### Training Script Entry Point

**`scripts/train.py`** is the main training script:

```bash
python scripts/train.py --config configs/experiments/exp_cfinet_soda.yaml
```

Or using the shell wrapper:
```bash
./train.sh exp_cfinet_soda
```

### Training Flow

1. **Initialization**:
   - Loads and merges configs (experiment + dataset)
   - Sets random seed for reproducibility
   - Creates output directory and saves config copy
   - Creates train/val/test DataLoaders
   - Creates model (CFINetWrapper)
   - Moves model to device (GPU/CPU)
   - Initializes TensorBoard logger (if enabled)

2. **Trainer Creation**:
   - Creates `DetectionTrainer` instance
   - Initializes optimizer (SGD/Adam) from config
   - Initializes scheduler (Step/Cosine) from config
   - Sets up metrics (mAP calculation)

3. **Training Loop** (per epoch):
   - **Train Phase**:
     - Sets model to `train()` mode
     - Iterates over training DataLoader
     - For each batch:
       - Moves images and targets to device
       - Calls `model(images, targets=targets)` → returns loss dict
       - Aggregates losses (handles None values, lists, tensors)
       - Backward pass and optimizer step
       - Gradient clipping (if configured)
       - Accumulates loss components for logging
   - **Validation Phase**:
     - Sets model to `eval()` mode
     - Iterates over validation DataLoader
     - Computes losses (no gradient computation)
     - Accumulates metrics
   - **Post-Epoch**:
     - Logs metrics to TensorBoard
     - Saves metrics history to YAML
     - Checkpointing (saves best/last if improved)
     - Early stopping check
     - Steps learning rate scheduler

4. **Checkpointing**:
   - **Best checkpoint**: Saved when monitored metric improves
     - For `mode: "min"` (loss): saves when metric decreases
     - For `mode: "max"` (mAP): saves when metric increases
   - **Last checkpoint**: Saved every epoch (if enabled)
   - Checkpoint contains:
     ```python
     {
       'epoch': int,
       'model_state_dict': dict,
       'optimizer_state_dict': dict,
       'best_metric': float
     }
     ```

5. **Early Stopping**:
   - Monitors validation metric (e.g., `val_loss`)
   - If no improvement for `patience` epochs, stops training
   - Respects `mode` ("min" for loss, "max" for mAP)

### Loss Handling

MMDetection models return a loss dictionary:
```python
{
  'loss_rpn_cls': tensor,
  'loss_rpn_bbox': tensor,
  'loss_cls': tensor,
  'loss_bbox': tensor,
  ...
}
```

The trainer:
1. Extracts all loss values (handles None, lists, tensors)
2. Sums them into total loss
3. Performs backward pass
4. Logs individual loss components separately

### Metrics Calculation

During validation, the trainer can compute mAP if configured:
- Collects predictions and ground truth across validation set
- Calls `mean_average_precision()` from `detection_metrics.py`
- Logs to TensorBoard and metrics history

---

## Evaluation and Testing

### Testing Script

**`scripts/test.py`** runs inference on the test set:

```bash
python scripts/test.py --config configs/experiments/exp_cfinet_soda.yaml --checkpoint outputs/experiments/exp_cfinet_soda/checkpoints/best.pth
```

Or using the shell wrapper:
```bash
./test.sh exp_cfinet_soda best
```

### Testing Flow

1. **Setup**:
   - Loads config and checkpoint
   - Creates model and loads checkpoint weights
   - Sets model to `eval()` mode
   - Creates test DataLoader

2. **Inference Loop**:
   - Iterates over test DataLoader
   - For each batch:
     - Calls `model.predict(images, score_threshold=...)`
     - Collects predictions (bboxes, labels, scores)
     - Collects ground truth (bboxes, labels)
     - Stores per-image results

3. **Metrics Calculation**:
   - Computes mAP across all test images
   - Uses IoU threshold of 0.5 (default)

4. **Output**:
   - Saves `predictions.json` with:
     ```json
     {
       "experiment": "exp_cfinet_soda",
       "checkpoint": "path/to/best.pth",
       "score_threshold": 0.05,
       "metrics": {
         "mAP": 0.7234
       },
       "per_image": [
         {
           "image_id": 1,
           "image_name": "image001.jpg",
           "pred_bboxes_xyxy": [[x1, y1, x2, y2], ...],
           "pred_scores": [0.95, 0.87, ...],
           "pred_labels": [1, 2, ...],
           "gt_bboxes_xyxy": [[x1, y1, x2, y2], ...],
           "gt_labels": [1, 2, ...]
         },
         ...
       ]
     }
     ```

### Metrics: Mean Average Precision (mAP)

The mAP calculation (`src/training/detection_metrics.py`) uses:

1. **IoU Calculation**: Intersection over Union between predicted and ground truth boxes
2. **Matching**: Greedy matching of predictions to ground truth (highest IoU first)
3. **Precision-Recall Curve**: Computes precision and recall at each prediction threshold
4. **11-Point Interpolation**: Averages precision at 11 recall levels (0.0, 0.1, ..., 1.0)
5. **Per-Class AP**: Computes AP for each class separately
6. **mAP**: Averages AP across all classes

---

## Utilities and Scripts

### Dataset Utilities

#### 1. Split COCO Dataset

**`scripts/split_coco_dataset.py`**: Splits a COCO dataset into train/val splits.

```bash
python scripts/split_coco_dataset.py \
  /path/to/train/_annotations.coco.json \
  --output-dir /path/to/SODA_ROOT \
  --val-ratio 0.2 \
  --seed 42
```

**What it does**:
- Loads COCO annotation file
- Randomly shuffles image IDs
- Splits into train/val based on `val_ratio`
- Creates new annotation files in `train/` and `val/` directories
- Optionally copies images to split directories

**Use case**: When you only have a train/test split and need a validation set.

#### 2. Convert SODA_A to COCO Format

**`scripts/convert_soda_a_to_coco.py`**: Converts SODA_A per-image JSON format to standard COCO format.

```bash
python scripts/convert_soda_a_to_coco.py \
    Annotations/train/ \
    --output-file train/_annotations.coco.json \
    --split train
```

**What it does**:
- Reads all per-image JSON files in the specified directory
- Converts polygon annotations to bounding boxes (COCO format)
- Merges all per-image annotations into a single COCO JSON file
- Preserves categories and image metadata

**Use case**: Converting SODA_A dataset from its native format to COCO format.

#### 3. Create Patched Dataset

**`scripts/create_patched_dataset.py`**: Creates a patched version of a COCO dataset by dividing images into subpatches.

```bash
python scripts/create_patched_dataset.py \
    train/_annotations.coco.json \
    --output-file train_patches/_annotations.coco.json \
    --patches-x 2 \
    --patches-y 2 \
    --min-overlap 0.1 \
    --image-dir train \
    --output-image-dir train_patches
```

**What it does**:
- Divides each image into a grid of patches (e.g., 2x2 = 4 patches per image)
- Transforms bbox coordinates to patch-relative coordinates
- Filters out bboxes that don't overlap with each patch (based on IoU threshold)
- Creates new patch images and saves them
- Generates a new COCO annotation file for the patches

**Use case**: When images are too large for GPU memory, split into smaller patches for training on less powerful systems.

#### 4. Visualize Patched Dataset

**`scripts/visualize_patched_dataset.py`**: Visualizes and verifies the correctness of patched datasets.

```bash
# Basic visualization
python scripts/visualize_patched_dataset.py \
    SODA_dataset/SODA_A_coco/train_patches/_annotations.coco.json \
    --image-dir SODA_dataset/SODA_A_coco/train_patches \
    --num-samples 5

# With original image comparison
python scripts/visualize_patched_dataset.py \
    SODA_dataset/SODA_A_coco/train_patches/_annotations.coco.json \
    --image-dir SODA_dataset/SODA_A_coco/train_patches \
    --original-annotation SODA_dataset/SODA_A_coco/train/_annotations.coco.json \
    --show-original \
    --num-samples 5

# Save visualizations (non-interactive)
python scripts/visualize_patched_dataset.py \
    SODA_dataset/SODA_A_coco/train_patches/_annotations.coco.json \
    --image-dir SODA_dataset/SODA_A_coco/train_patches \
    --save-dir visualizations/train_patches \
    --num-samples 10
```

**What it does**:
- Displays patch images with bounding boxes overlaid
- Verifies that bbox coordinates are correctly transformed from original to patch coordinates
- Optionally shows original image alongside patch with patch region highlighted
- Prints dataset statistics (annotations per patch, category distribution, etc.)
- Can work interactively or save visualizations to files

**Features**:
- Color-coded bounding boxes by category
- Red dashed lines showing patch boundaries
- Transformation verification with error reporting
- Statistics including expansion factor and category distribution

**Use case**: Verifying that patched datasets are correctly created and bbox transformations are accurate.

### Training Utilities

#### Queue Multiple Experiments

**`queue.sh`**: Runs multiple experiments sequentially.

```bash
./queue.sh exp_cfinet_soda exp_cfinet_soda_patches
```

**What it does**:
- Runs each experiment in sequence
- Logs output to `outputs/queue_logs/`
- Continues even if one experiment fails
- Provides summary at the end

---

## Project Structure

```
CV_ppln/
├── configs/
│   ├── datasets/
│   │   ├── soda_a_05_small.yaml
│   │   └── soda_a_05_small_patches.yaml
│   └── experiments/
│       ├── exp_cfinet_soda.yaml
│       └── exp_cfinet_soda_patches.yaml
├── scripts/
│   ├── train.py                    # Main training script
│   ├── test.py                     # Testing/evaluation script
│   ├── split_coco_dataset.py       # Dataset splitting utility
│   ├── convert_soda_a_to_coco.py   # Convert SODA_A to COCO format
│   ├── create_patched_dataset.py   # Create patched dataset
│   ├── prepare_soda_a_dataset.py   # Master script for dataset preparation
│   └── visualize_patched_dataset.py # Visualize and verify patched datasets
├── src/
│   ├── data/
│   │   ├── __init__.py       # DataLoader factory
│   │   └── coco_dataset.py    # COCO Dataset implementation
│   ├── models/
│   │   ├── __init__.py       # Model factory
│   │   ├── registry.py       # Model registration system
│   │   └── architectures/
│   │       └── cfinet.py     # CFINet wrapper (contains embedded default config)
│   ├── training/
│   │   ├── detection_trainer.py  # Training loop
│   │   └── detection_metrics.py  # mAP calculation
│   └── utils/
│       ├── config.py         # Config loading
│       ├── helpers.py         # Utility functions
│       ├── tensorboard_logger.py  # TensorBoard integration
│       └── memory_profiler.py     # Memory profiling
├── outputs/
│   ├── experiments/           # Training outputs
│   │   └── <exp_name>/
│   │       ├── config.yaml
│   │       ├── checkpoints/
│   │       │   ├── best.pth
│   │       │   └── last.pth
│   │       ├── metrics_history.yaml
│   │       └── tensorboard/
│   └── tests/                 # Test outputs
│       └── <exp_name>/
│           └── predictions.json
├── train.sh                   # Training script wrapper
├── test.sh                    # Testing script wrapper
├── queue.sh                   # Queue experiments script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Troubleshooting

### Common Issues

#### 1. **Config Path Resolution Error**

**Error**: `FileNotFoundError: [Errno 2] No such file or directory: 'configs/experiments/configs/datasets/...'`

**Solution**: The config loader now automatically finds the project root. Ensure your dataset config path in the experiment config is relative to the project root:
```yaml
dataset: "configs/datasets/soda_a_05_small.yaml"  # ✓ Correct
dataset: "datasets/soda_a_05_small.yaml"         # ✗ Wrong
```

#### 2. **MMDetection Import Error**

**Error**: `ImportError: Could not import MMDetection`

**Solution**: Install MMDetection and its dependencies:
```bash
pip install mmdet>=2.26.0 mmcv-full>=1.5.0
```

Or reinstall from requirements.txt:
```bash
pip install -r requirements.txt
```

Verify installation:
```bash
python -c "import mmdet; print(mmdet.__version__)"
```

#### 3. **CUDA Out of Memory**

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce `batch_size` in experiment config
- Reduce `image_size` in dataset config (or use patches)
- Enable gradient accumulation (not currently supported, but can be added)

#### 4. **Checkpoint Not Found**

**Error**: `FileNotFoundError: Checkpoint not found`

**Solution**: Ensure training has completed and checkpoint exists:
```bash
ls outputs/experiments/<exp_name>/checkpoints/
```

If missing, check training logs for errors.

#### 5. **Dataset Path Issues**

**Error**: `Could not load image: /path/to/image.jpg`

**Solution**: 
- Verify dataset paths in `configs/datasets/soda_a_05_small.yaml`
- Ensure annotation file `_annotations.coco.json` exists in each split directory
- Check that image filenames in annotations match actual files

#### 6. **Model Config File Not Found**

**Error**: `FileNotFoundError: [Errno 2] No such file or directory: '.../config_file.py'`

**Solution**: 
- **Recommended**: Set `model.config_file: null` to use the CFINet model definition (no file needed)
- **If using custom config**: Ensure the `model.config_file` path is correct. It can be:
  - Relative to `src/models/architectures/` (e.g., `"my_custom_config.py"`)
  - Absolute path (e.g., `"/path/to/config.py"`)

```yaml
model:
  config_file: null  # Use CFINet model definition (recommended)
  # OR
  # config_file: "my_custom_config.py"  # Custom config file
```

The CFINet model definition is embedded in `src/models/architectures/cfinet.py`, so no external config file is needed unless you want to customize it.

#### 7. **Early Stopping Not Working**

**Issue**: Training doesn't stop even when validation metric stops improving.

**Solution**: Check `training.early_stopping.mode` matches the metric:
- For `monitor: "val_loss"` → `mode: "min"`
- For `monitor: "val_mAP"` → `mode: "max"`

#### 8. **mAP Calculation Errors**

**Error**: `ValueError: ...` during mAP calculation

**Solution**: 
- Ensure predictions and ground truth are in correct format
- Check that bboxes are in `[x1, y1, x2, y2]` format (not `[x, y, w, h]`)
- Verify labels are integer class IDs (not one-hot)

### Debugging Tips

1. **Enable Memory Profiling**:
   ```yaml
   debug:
     profile_memory: true
     detailed_memory: true
   ```

2. **Check TensorBoard Logs**:
   ```bash
   tensorboard --logdir outputs/experiments/<exp_name>/tensorboard
   ```

3. **Inspect Config After Loading**:
   Add print statement in `scripts/train.py`:
   ```python
   config = load_config(config_path)
   import json
   print(json.dumps(config, indent=2, default=str))
   ```

4. **Test Data Loading**:
   ```python
   from src.data import create_dataloaders
   from src.utils.config import load_config
   config = load_config('configs/experiments/exp_cfinet_soda.yaml')
   train_loader, val_loader, test_loader = create_dataloaders(config)
   batch = next(iter(train_loader))
   print(batch.keys())  # Should show: image, bboxes, labels, ...
   ```

---

## Quick Reference

### Training Command
```bash
./train.sh exp_cfinet_soda
```

### Testing Command
```bash
./test.sh exp_cfinet_soda best
```

### Key Config Files
- Dataset: `configs/datasets/soda_a_05_small.yaml`
- Model: CFINet model definition in `src/models/architectures/cfinet.py` (no separate config file needed)
- Experiment: `configs/experiments/exp_cfinet_soda.yaml`

### Output Locations
- Training: `outputs/experiments/<exp_name>/`
- Testing: `outputs/tests/<exp_name>/predictions.json`

### TensorBoard
```bash
tensorboard --logdir outputs/experiments/<exp_name>/tensorboard
```

---

## Summary

This pipeline provides a complete, streamlined workflow for training and evaluating CFINet on COCO-format object detection datasets. The modular design allows easy extension to other models or datasets, while the comprehensive configuration system provides fine-grained control over all aspects of training and evaluation.

For questions or issues, refer to the troubleshooting section or inspect the source code in `src/` for detailed implementation.
