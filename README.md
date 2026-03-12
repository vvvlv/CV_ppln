# CV_ppln: Object Detection Pipeline for SODA-A with CFINet

A comprehensive, streamlined deep learning pipeline for training and evaluating object detection models on the SODA-A dataset using CFINet (Coarse-to-fine Proposal Generation and Imitation Learning Network). This system is built entirely on **pure PyTorch**, providing a complete workflow from data loading to model evaluation without any external detection framework dependencies.

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

This pipeline is designed specifically for **object detection** tasks using **COCO-format datasets**. It implements CFINet as a **pure PyTorch model** with a clean, modular training and evaluation framework. The system is completely self-contained and doesn't require MMDetection or any other external detection frameworks.

### Key Features

- **Pure PyTorch Implementation**: CFINet implemented entirely in PyTorch - no external dependencies
- **COCO Dataset Support**: Native handling of COCO-format annotations
- **Self-Contained**: All CFINet components embedded in the codebase
- **Modular Design**: Clean separation between data, models, training, and evaluation
- **Comprehensive Metrics**: mAP (mean Average Precision) calculation with per-class support
- **TensorBoard Integration**: Real-time visualization of training metrics
- **Checkpointing**: Automatic saving of best and last model checkpoints
- **Early Stopping**: Configurable early stopping based on validation metrics
- **Memory Profiling**: Optional memory usage analysis for debugging

### Supported Datasets

- **SODA-A**: Small Object Detection in Aerial imagery dataset (9 classes)
  - **Note**: According to the SODA paper, SODA-A uses Oriented Bounding Boxes (OBB).
    However, this pipeline uses Horizontal Bounding Boxes (HBB) for simplicity and compatibility
    with standard COCO-format detection frameworks. The conversion from OBB to HBB is done during
    dataset preparation.
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
    │   val/_annotations.coco.json (15%)                     │
    └──────────────────────────────────────────────────────┘
                              │
                              │ [create_patched_dataset.py]
                              │ (Optional: create patches)
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ Patched COCO Format (512x512 max patches)              │
    │   train_patches/_annotations.coco.json                 │
    │   val_patches/_annotations.coco.json                   │
    │   test_patches/_annotations.coco.json                  │
    └──────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 2. CONFIGURATION                                                │
│    (Experiment + Dataset configs)                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ Experiment Config (YAML)                              │
    │   - Model type: CFINet                                 │
    │   - Training hyperparameters                           │
    │   - Dataset reference                                   │
    └──────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ Dataset Config (YAML)                                  │
    │   - Dataset paths                                      │
    │   - Image preprocessing                                │
    │   - Number of classes                                  │
    └──────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 3. DATA LOADING                                                 │
│    (COCODataset → DataLoader)                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ COCODataset                                            │
    │   - Loads COCO JSON annotations                        │
    │   - Loads images                                        │
    │   - Applies preprocessing (resize, normalize)         │
    │   - Applies augmentations (flip, brightness)           │
    │   - Converts bboxes: [x,y,w,h] → [x1,y1,x2,y2]        │
    └──────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ DataLoader                                             │
    │   - Batches samples                                    │
    │   - Handles variable-length bboxes/labels             │
    │   - Multi-process loading                              │
    └──────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 4. MODEL INITIALIZATION                                         │
│    (Pure PyTorch CFINet)                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ CFINet Model (Pure PyTorch)                            │
    │   - ResNet-50 Backbone (pretrained)                    │
    │   - FPN (Feature Pyramid Network)                     │
    │   - CRPNHead (Coarse-to-fine RPN Head)                 │
    │   - DynamicAssigner (Anchor assignment)                │
    │   - RoIHead (Feature Imitation ROI Head)              │
    └──────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 5. TRAINING LOOP                                                │
│    (Per epoch, iterating over batches)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ DetectionTrainer.train()                               │
    │   - Initializes optimizer (SGD/Adam)                    │
    │   - Initializes scheduler (Step/Cosine)                 │
    │   - Sets up metrics tracking                           │
    └──────────────────────────────────────────────────────┘
                              │
                              │ [For each epoch]
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ Training Phase (model.train())                         │
    │   For each batch:                                      │
    │     1. Move images and targets to device              │
    │     2. Forward pass: model(images, targets=targets)    │
    │     3. Get loss dict from model                       │
    │     4. Aggregate losses (sum all components)           │
    │     5. Backward pass: loss.backward()                  │
    │     6. Optimizer step: optimizer.step()                │
    │     7. Log loss components                             │
    └──────────────────────────────────────────────────────┘
                              │
                              │ [After training phase]
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ Validation Phase (model.eval())                        │
    │   For each batch:                                      │
    │     1. Forward pass (no gradients)                     │
    │     2. Compute losses                                  │
    │     3. Collect predictions and ground truth             │
    │     4. Compute metrics (mAP)                            │
    └──────────────────────────────────────────────────────┘
                              │
                              │ [After validation]
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ Post-Epoch Operations                                 │
    │   1. Log metrics to TensorBoard                        │
    │   2. Save metrics_history.yaml                         │
    │   3. Checkpointing:                                    │
    │      - Save best.pth if metric improved                │
    │      - Save last.pth (every epoch)                     │
    │   4. Early stopping check                              │
    │   5. Step learning rate scheduler                      │
    └──────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 6. INFERENCE / TESTING                                          │
│    (After training, on test set)                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ Load Trained Model                                     │
    │   - Load checkpoint (best.pth or last.pth)             │
    │   - Restore model weights                              │
    │   - Set model.eval()                                   │
    └──────────────────────────────────────────────────────┘
                              │
                              │ [For each test batch]
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ Inference                                              │
    │   1. Forward pass: model.predict(images)               │
    │   2. Model returns detections                          │
    │   3. Format: {bboxes: [N,4], labels: [N], scores: [N]}│
    │   4. Filter by score_threshold                        │
    └──────────────────────────────────────────────────────┘
                              │
                              │ [After all test batches]
                              ▼
    ┌──────────────────────────────────────────────────────┐
    │ Evaluation                                            │
    │   1. Collect all predictions and ground truth          │
    │   2. Compute mAP (mean Average Precision)              │
    │   3. Save predictions.json with results                │
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
        'pretrained': True,
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
# In CFINet.forward()
1. Extract features from ResNet-50 backbone:
   - C2, C3, C4, C5 feature maps

2. Pass through FPN:
   - P2, P3, P4, P5, P6 feature maps

3. Generate anchors for all FPN levels

4. RPN forward (CRPNHead):
   - Coarse regression predictions
   - Fine classification predictions
   - Fine regression predictions

5. Assign anchors to ground truth (DynamicAssigner):
   - Compute IoU between anchors and GT boxes
   - Dynamic threshold based on object size
   - Assign positive/negative labels

6. Compute RPN losses:
   - Coarse regression loss (Smooth L1)
   - Fine classification loss (Binary Cross-Entropy)
   - Fine regression loss (Smooth L1)

7. Generate proposals from RPN predictions

8. ROI Head forward:
   - Extract ROI features using ROI Align
   - Classification predictions
   - Regression predictions
   - Feature imitation (contrastive learning)

9. Compute ROI losses:
   - Classification loss (Cross-Entropy)
   - Regression loss (Smooth L1)
   - Contrastive loss (Feature Imitation)

10. Return loss dictionary:
    {
        'loss_rpn_coarse': Tensor,
        'loss_rpn_fine_cls': Tensor,
        'loss_rpn_fine_reg': Tensor,
        'loss_cls': Tensor,
        'loss_bbox': Tensor,
        'contrastive_loss': Tensor
    }
```

**Inference mode (`model.predict(images)`):**

```python
# In CFINet.predict()
1. Set model to eval mode: model.eval()

2. Forward pass (no gradients):
   - Extract features (ResNet-50 + FPN)
   - Generate anchors
   - RPN forward → proposals
   - ROI Head forward → detections

3. Post-process:
   - Apply softmax to classification logits
   - Filter by confidence threshold
   - Apply NMS (Non-Maximum Suppression)
   - Format results

4. Return formatted results:
   [
       {
           'bboxes': np.array([N, 4]),  # [x1, y1, x2, y2]
           'labels': np.array([N]),
           'scores': np.array([N])
       },
       ...  # One per image in batch
   ]
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
    
    # Save checkpoint if improved
    if val_mAP > best_mAP:
        save_checkpoint('best.pth')
    
    # Early stopping check
    if early_stopping.should_stop(val_mAP):
        break
```

---

## System Architecture

The system follows a modular architecture with clear separation of concerns:

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
│  Model Builder  │  (CFINet - Pure PyTorch)
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
  - Configurable batch size, workers, pin_memory

#### 3. **Model System** (`src/models/`)
- **Model Registry** (`registry.py`): Decorator-based registration system
- **CFINet Model** (`architectures/cfinet_pytorch.py`): 
  - Pure PyTorch implementation of CFINet
  - ResNet-50 backbone with FPN
  - CRPNHead (Coarse-to-fine RPN)
  - DynamicAssigner for anchor assignment
  - RoIHead with Feature Imitation
  - Completely self-contained - no external dependencies

#### 4. **Training System** (`src/training/`)
- **DetectionTrainer** (`detection_trainer.py`): Main training loop
  - Epoch-based training with validation
  - Loss aggregation from model loss dict
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

**Note**: 
- The system is **completely self-contained** - no MMDetection or other external detection frameworks needed
- CFINet is implemented entirely in PyTorch within this codebase
- All model components are in `src/models/architectures/cfinet_pytorch.py`

#### 2. Verify Installation

```bash
python -c "
import torch
import torchvision
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'torchvision: {torchvision.__version__}')
"
```

Expected output:
```
PyTorch: 2.8.0
CUDA available: True
torchvision: 0.23.0
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

**Example: `configs/datasets/soda_a_coco_512_patches.yaml`**
```yaml
name: "SODA_A_coco_512_patches"
description: "SODA-A dataset with 512x512 max patches (COCO format)"

format: "coco"  # Indicates COCO-format detection dataset

paths:
  root: "/path/to/SODA_dataset/SODA_A_coco_512"
  train: "train_patches"
  val: "val_patches"
  test: "test_patches"

annotation_file: "_annotations.coco.json"  # COCO annotation file name

# Image preprocessing
image_size: [512, 512]  # Resize all images to this size
num_channels: 3
num_classes: 9  # SODA-A has 9 object classes

# Normalization (ImageNet stats)
mean: [0.485, 0.456, 0.406]
std: [0.229, 0.224, 0.225]

# Minimum bbox size to keep (filter tiny boxes)
min_bbox_size: 1
```

### Experiment Configuration

Located in `configs/experiments/`, these files define training experiments:

**Example: `configs/experiments/exp_cfinet_soda_512_patches.yaml`**
```yaml
name: "exp_cfinet_soda_512_patches"
description: "CFINet on SODA-A with 512x512 patches"

dataset: "configs/datasets/soda_a_coco_512_patches.yaml"

# Data loading
data:
  batch_size: 4
  num_workers: 4
  pin_memory: true
  
  # Limit dataset size for faster training/testing (optional)
  max_samples: 70  # Limit training samples
  max_val_samples: 15
  max_test_samples: 15
  
  augmentation:
    enabled: true
    horizontal_flip: 0.5
    brightness: 0.1

# Model architecture
model:
  type: "CFINet"
  num_classes: 9
  pretrained: true  # Use pretrained ResNet-50

# Training settings
training:
  epochs: 12
  optimizer:
    type: "sgd"
    learning_rate: 0.01
    momentum: 0.9
    weight_decay: 0.0001
  scheduler:
    type: "step"
    step_size: 8
    gamma: 0.1
  checkpoint:
    save_best: true
    save_last: true
    monitor: "val_loss"
    mode: "min"
  early_stopping:
    enabled: true
    patience: 5
    monitor: "val_loss"
    mode: "min"

# Output
output:
  dir: "outputs/experiments/exp_cfinet_soda_512_patches"
  save_predictions: true

seed: 42
device: "cuda"
```

### Limiting Dataset Size

For faster training/testing iterations, you can limit the number of samples used:

```yaml
data:
  max_samples: 1000  # Use only first 1000 training samples
  max_val_samples: 200
  max_test_samples: 200
```

This is useful for:
- Quick debugging and testing
- Reducing training time during development
- Testing on smaller subsets

---

## Data Pipeline

### COCO Dataset Format

The pipeline expects COCO-format annotation files with this structure:

```json
{
  "images": [
    {"id": 1, "file_name": "00001.jpg", "width": 512, "height": 512},
    ...
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [10, 20, 50, 60],  // [x, y, width, height]
      "area": 3000,
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

### Data Augmentation

The pipeline supports the following augmentations (configurable in dataset config):

- **Horizontal Flip**: Randomly flips images horizontally (50% probability)
- **Brightness Adjustment**: Randomly adjusts brightness by ±10%

Augmentations are only applied during training, not validation or testing.

---

## Dataset Preparation

### SODA-A Dataset Preparation

The pipeline includes scripts to prepare the SODA-A dataset:

#### 1. Convert SODA-A to COCO Format

```bash
python scripts/dataset/convert_soda_a_to_coco.py \
    --annotations-dir /path/to/SODA_A/Annotations/train \
    --images-dir /path/to/SODA_A/Images/train \
    --output-dir /path/to/output/train \
    --split train
```

This converts per-image JSON annotations to a single COCO-format JSON file.

#### 2. Split Dataset (70/15/15)

```bash
python scripts/dataset/split_coco_three_way.py \
    --annotation-file /path/to/all_annotations.coco.json \
    --output-dir /path/to/output \
    --train-ratio 0.7 \
    --test-ratio 0.15 \
    --val-ratio 0.15 \
    --seed 42 \
    --copy-images
```

#### 3. Create Patched Dataset

```bash
python scripts/dataset/create_patched_dataset.py \
    --annotation-file /path/to/train/_annotations.coco.json \
    --images-dir /path/to/train \
    --output-dir /path/to/output/train_patches \
    --max-patch-size 512 \
    --min-overlap 0.1
```

#### 4. Automated Dataset Preparation

Use the comprehensive shell script:

```bash
cd scripts/dataset
./prepare_dataset.sh
```

Edit the hardcoded variables at the top of the script:
- `SODA_A_DIR`: Path to SODA-A dataset
- `OUTPUT_DIR`: Output directory for prepared dataset
- `MAX_PATCH_SIZE`: Maximum patch size (e.g., 512)
- `TRAIN_RATIO`, `TEST_RATIO`, `VAL_RATIO`: Split ratios
- `RANDOM_SEED`: Random seed for reproducibility

### Reproducibility

All dataset preparation scripts ensure deterministic output:
- Files are processed in sorted order
- Random operations use fixed seeds
- Outputs are sorted before saving

---

## Model Architecture

### CFINet Overview

CFINet (Coarse-to-fine Proposal Generation and Imitation Learning Network) is a two-stage object detector designed for small object detection. It is implemented entirely in **pure PyTorch** within this codebase.

**Architecture Components:**

1. **Backbone**: ResNet-50 (pretrained on ImageNet)
   - Extracts multi-scale features: C2, C3, C4, C5

2. **Neck**: FPN (Feature Pyramid Network)
   - Creates feature pyramid: P2, P3, P4, P5, P6
   - Enables multi-scale detection

3. **RPN Head**: CRPNHead (Coarse-to-fine RPN Head)
   - **Coarse Stage**: Initial regression to refine anchor positions
   - **Fine Stage**: Classification and refined regression
   - Uses Adaptive Convolution for feature alignment

4. **Anchor Assignment**: DynamicAssigner
   - Dynamic IoU threshold based on object size
   - Ensures sufficient positive samples for small objects
   - Formula: `Ta = max(0.25, 0.20 + γ · log(√(w·h)/12))`

5. **ROI Head**: RoIHead with Feature Imitation
   - Classification and regression heads
   - Feature Imitation branch for contrastive learning
   - Feat2Embed module for feature embedding

### Model Implementation

The CFINet model is located in `src/models/architectures/cfinet_pytorch.py`:

```python
class CFINet(nn.Module):
    """
    Complete CFINet model for object detection.
    
    Pure PyTorch implementation - no external dependencies.
    """
    
    def __init__(self, num_classes=9, pretrained=True, **kwargs):
        # ResNet-50 backbone
        # FPN neck
        # CRPNHead for proposal generation
        # DynamicAssigner for anchor assignment
        # RoIHead for detection
    
    def forward(self, images, targets=None):
        # Training: returns loss dictionary
        # Inference: returns detection results
    
    def predict(self, images, score_threshold=0.5):
        # Inference with score filtering
```

### Loss Functions

The model computes the following losses during training:

1. **RPN Losses**:
   - `loss_rpn_coarse`: Coarse regression loss (Smooth L1)
   - `loss_rpn_fine_cls`: Fine classification loss (Binary Cross-Entropy)
   - `loss_rpn_fine_reg`: Fine regression loss (Smooth L1)

2. **ROI Losses**:
   - `loss_cls`: Classification loss (Cross-Entropy)
   - `loss_bbox`: Regression loss (Smooth L1)
   - `contrastive_loss`: Feature imitation loss

### Model Registry

Models are registered using the `@register_model` decorator:

```python
from src.models import create_model

config = {
    'model': {
        'type': 'CFINet',
        'num_classes': 9,
        'pretrained': True
    }
}

model = create_model(config)
```

---

## Training Process

### Training Script Entry Point

**`scripts/train.py`** is the main training script:

```bash
python scripts/train.py --config configs/experiments/exp_cfinet_soda_512_patches.yaml
```

Or using the shell wrapper:
```bash
./train.sh exp_cfinet_soda_512_patches
```

### Training Flow

1. **Initialization**:
   - Loads and merges configs (experiment + dataset)
   - Sets random seed for reproducibility
   - Creates output directory and saves config copy
   - Creates train/val/test DataLoaders
   - Creates model (CFINet)
   - Initializes optimizer and scheduler
   - Sets up TensorBoard logging

2. **Training Loop** (per epoch):
   - **Training Phase**:
     - Iterates over training batches
     - Forward pass: `model(images, targets=targets)`
     - Computes losses
     - Backward pass and optimizer step
     - Logs training losses
   
   - **Validation Phase**:
     - Iterates over validation batches
     - Forward pass (no gradients)
     - Computes validation losses
     - Collects predictions for metrics
     - Computes mAP
   
   - **Post-Epoch**:
     - Logs metrics to TensorBoard
     - Saves metrics history
     - Saves checkpoint if improved
     - Checks early stopping
     - Steps learning rate scheduler

3. **Completion**:
   - Saves final checkpoint
   - Saves metrics history
   - Closes TensorBoard logger

### Monitoring Training

**TensorBoard**:
```bash
tensorboard --logdir outputs/experiments/exp_cfinet_soda_512_patches/tensorboard
```

**Metrics History**:
Check `outputs/experiments/exp_cfinet_soda_512_patches/metrics_history.yaml` for per-epoch metrics.

---

## Evaluation and Testing

### Testing Script

**`scripts/test.py`** evaluates a trained model:

```bash
python scripts/test.py \
    --config configs/experiments/exp_cfinet_soda_512_patches.yaml \
    --checkpoint outputs/experiments/exp_cfinet_soda_512_patches/checkpoints/best.pth
```

### Evaluation Metrics

The pipeline computes:

- **mAP (mean Average Precision)**: Average precision across all classes
- **Per-class AP**: Average precision for each class
- **IoU Threshold**: Default 0.5 (configurable)

Results are saved to:
- `outputs/experiments/{exp_name}/predictions.json`
- Console output with detailed metrics

---

## Utilities and Scripts

### Dataset Preparation Scripts

All dataset preparation scripts are in `scripts/dataset/`:

- **`convert_soda_a_to_coco.py`**: Convert SODA-A format to COCO format
- **`split_coco_three_way.py`**: Split COCO dataset into train/test/val
- **`create_patched_dataset.py`**: Create patched dataset from large images
- **`prepare_dataset.sh`**: Comprehensive automated dataset preparation
- **`visualize_patched_dataset.py`**: Visualize patched images and annotations
- **`test_reproducibility.py`**: Test dataset preparation reproducibility

### Visualization Script

**`scripts/dataset/visualize_patched_dataset.py`**: Visualize patched images with annotations.

Edit hardcoded variables at the top:
```python
BASE_DIR = "/path/to/dataset"
SPLIT = "train_patches"
NUM_SAMPLES = 10
```

Or use command-line arguments:
```bash
python scripts/dataset/visualize_patched_dataset.py \
    --base-dir /path/to/dataset \
    --split train_patches \
    --num-samples 10
```

---

## Project Structure

```
CV_ppln/
├── configs/
│   ├── datasets/
│   │   ├── soda_a_coco_512_patches.yaml
│   │   └── ...
│   └── experiments/
│       ├── exp_cfinet_soda_512_patches.yaml
│       └── ...
├── scripts/
│   ├── train.py                    # Main training script
│   ├── test.py                     # Testing/evaluation script
│   └── dataset/
│       ├── convert_soda_a_to_coco.py
│       ├── create_patched_dataset.py
│       ├── split_coco_three_way.py
│       ├── prepare_dataset.sh
│       └── ...
├── src/
│   ├── data/
│   │   ├── __init__.py       # DataLoader factory
│   │   └── coco_dataset.py    # COCO Dataset implementation
│   ├── models/
│   │   ├── __init__.py       # Model factory
│   │   ├── registry.py       # Model registration system
│   │   └── architectures/
│   │       └── cfinet_pytorch.py  # Pure PyTorch CFINet implementation
│   ├── training/
│   │   ├── detection_trainer.py  # Training loop
│   │   └── detection_metrics.py  # mAP calculation
│   └── utils/
│       ├── config.py         # Config loading
│       ├── tensorboard_logger.py
│       └── ...
├── outputs/
│   └── experiments/
│       └── {experiment_name}/
│           ├── checkpoints/
│           │   ├── best.pth
│           │   └── last.pth
│           ├── metrics_history.yaml
│           ├── predictions.json
│           └── tensorboard/
├── requirements.txt
├── train.sh
├── test.sh
└── README.md
```

---

## Troubleshooting

### Common Issues

**CUDA Out of Memory**:
- Reduce batch size in experiment config
- Use smaller image size
- Use patched dataset with smaller patches
- Use `max_samples` to limit dataset size

**No detections during inference**:
- Lower `score_threshold` in evaluation config
- Check that model was trained properly
- Verify dataset has annotations

**Training loss not decreasing**:
- Check learning rate (may be too high/low)
- Verify data augmentation is working
- Check that losses have gradients (should not be zero)

**Images not found**:
- Check that images are in correct directory structure
- Verify paths in dataset config
- Use `--copy-images` flag during dataset preparation

**No annotations in patches**:
- Lower `--min-overlap` value (e.g., 0.05)
- Check that bboxes actually overlap with patches
- Verify original annotations are correct

**Out of memory**:
- Process one split at a time
- Reduce patch grid size (e.g., 1x2 instead of 2x2)
- Process images in smaller batches

---

## Key Design Decisions

### Pure PyTorch Implementation

The system uses a **pure PyTorch implementation** of CFINet rather than wrapping MMDetection:

- **Benefits**:
  - No external dependencies (MMDetection, mmcv)
  - Full control over model architecture
  - Easier debugging and modification
  - Simpler installation and deployment
  - Better integration with PyTorch ecosystem

- **Trade-offs**:
  - Some advanced features may need manual implementation
  - Loss functions are simplified but functional

### COCO Format Compatibility

The system uses COCO format for maximum compatibility:

- Standard format used by many datasets
- Easy to convert from other formats
- Well-documented and widely supported

### Horizontal Bounding Boxes (HBB)

While SODA-A paper specifies OBB (Oriented Bounding Boxes), this pipeline uses HBB:

- Simpler implementation
- Compatible with standard COCO format
- Sufficient for many use cases
- Can be extended to OBB if needed

---

## Future Improvements

Potential enhancements to the system:

1. **Complete Loss Implementation**:
   - IoU loss for regression
   - Complete feature imitation contrastive loss
   - Multi-level ROI align

2. **Advanced Features**:
   - Mixed precision training
   - Distributed training support
   - More augmentation options

3. **Model Variants**:
   - Different backbone architectures
   - Different RPN configurations
   - Ablation studies

---

## License

[Add your license information here]

---

## Citation

If you use this codebase, please cite:

- CFINet paper: [Add citation]
- SODA dataset: [Add citation]

---

## Acknowledgments

- CFINet authors for the original architecture
- SODA dataset creators
- PyTorch and torchvision teams
