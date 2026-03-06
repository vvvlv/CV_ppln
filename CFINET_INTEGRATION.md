# CFINet Object Detection Integration

This document describes the integration of CFINet object detection network with the CV_ppln pipeline for training on the SODA dataset.

## Overview

The CV_ppln pipeline has been extended to support object detection tasks using CFINet (Coarse-to-fine Proposal Generation and Imitation Learning Network). The integration maintains compatibility with the existing segmentation pipeline while adding detection capabilities.

## Components Added

### 1. Dataset Support
- **COCO Dataset Loader** (`src/data/coco_dataset.py`): Handles COCO-format annotations used by SODA dataset
- **Updated Data Module** (`src/data/__init__.py`): Automatically detects detection vs segmentation tasks

### 2. Model Integration
- **CFINet Wrapper** (`src/models/architectures/cfinet.py`): Wraps MMDetection-based CFINet model for use with CV_ppln framework
- **Model Registry**: CFINet registered as `"CFINet"` model type

### 3. Training Support
- **Detection Trainer** (`src/training/detection_trainer.py`): Specialized trainer for object detection tasks
- **Detection Metrics** (`src/training/detection_metrics.py`): mAP and AP metrics for detection evaluation
- **Updated Training Script**: Automatically selects appropriate trainer based on dataset format

### 4. Configuration Files
- **Dataset Config** (`configs/datasets/soda_a_05_small.yaml`): SODA dataset configuration
- **Experiment Config** (`configs/experiments/exp_cfinet_soda.yaml`): CFINet training configuration

## Installation

### Prerequisites
1. Install CFINet from the CFI/CFINet directory:
   ```bash
   cd /home/vlv/Documents/master/computervision/CFI/CFINet
   pip install -v -e .
   ```

2. Install additional dependencies:
   ```bash
   cd /home/vlv/Documents/master/computervision/CV_ppln
   pip install -r requirements.txt
   ```

### Required Dependencies
- MMDetection 2.26.0+
- mmcv-full 1.5.0+
- PyTorch 1.10.0+
- CUDA 11.3+

## Usage

### Training CFINet on SODA Dataset

1. **Prepare the dataset**: Ensure SODA dataset is located at:
   ```
   /home/vlv/Documents/master/computervision/SODA_dataset/SODA_A_05_small/
   ```

2. **Update config paths** (if needed): Edit `configs/datasets/soda_a_05_small.yaml` to match your dataset location.

3. **Update CFINet config path**: Edit `configs/experiments/exp_cfinet_soda.yaml` to point to the correct CFINet config file:
   ```yaml
   model:
     config_file: "/path/to/CFI/CFINet/configs/cfinet/faster_rcnn_r50_fpn_cfinet_1x.py"
   ```

4. **Train the model**:
   ```bash
   ./train.sh exp_cfinet_soda
   ```

   Or directly:
   ```bash
   python scripts/train.py --config configs/experiments/exp_cfinet_soda.yaml
   ```

## Configuration Details

### Dataset Configuration
The SODA dataset config (`soda_a_05_small.yaml`) specifies:
- COCO format annotations (`_annotations.coco.json`)
- 9 object classes
- Image normalization using ImageNet statistics
- Optional data augmentation

### Model Configuration
The CFINet model uses:
- Faster R-CNN backbone with ResNet-50
- CRPNHead (Coarse-to-fine RPN Head)
- FIRoIHead (Feature Imitation ROI Head)
- 9 classes for SODA-A dataset

### Training Configuration
Default training settings:
- 12 epochs
- SGD optimizer (lr=0.01, momentum=0.9)
- Step learning rate scheduler
- Batch size: 4
- Early stopping with patience=5

## Output Structure

Training outputs are saved to:
```
outputs/experiments/exp_cfinet_soda/
├── config.yaml              # Copy of experiment config
├── checkpoints/
│   ├── best.pth            # Best model checkpoint
│   └── last.pth            # Last epoch checkpoint
├── metrics_history.yaml    # Training metrics per epoch
└── tensorboard/           # TensorBoard logs
```

## Differences from Segmentation Pipeline

1. **Dataset Format**: Uses COCO format instead of image/mask pairs
2. **Model Output**: Bounding boxes and class labels instead of segmentation masks
3. **Metrics**: mAP (mean Average Precision) instead of Dice/IoU
4. **Loss Function**: MMDetection's built-in losses (handled internally by CFINet)

## Troubleshooting

### Import Errors
If you encounter MMDetection import errors:
- Ensure CFINet is properly installed: `cd CFI/CFINet && pip install -v -e .`
- Check that mmcv-full is installed: `pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html`

### CUDA Errors
- Verify CUDA version matches requirements (11.3+)
- Check that PyTorch was installed with CUDA support

### Dataset Path Issues
- Update paths in `configs/datasets/soda_a_05_small.yaml` to match your setup
- Ensure annotation file `_annotations.coco.json` exists in train/test directories

## Future Enhancements

Potential improvements:
- Support for additional detection datasets
- Integration of more detection metrics
- Visualization of detection results
- Support for rotated bounding boxes (SODA-A specific)
