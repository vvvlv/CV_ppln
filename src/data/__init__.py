"""Data module."""

import torch
from torch.utils.data import DataLoader
from typing import Tuple
from .coco_dataset import COCODataset


def create_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    
    dataset_config = config['dataset']
    data_config = config['data']
    augmentation_config = data_config.get('augmentation', {})
    
    # This repo is streamlined for object detection (COCO format)
    is_detection = dataset_config.get('format') == 'coco' or 'annotation_file' in dataset_config
    if not is_detection:
        raise ValueError("Only COCO-format detection datasets are supported (set dataset.format: 'coco').")

    annotation_file = dataset_config.get('annotation_file', '_annotations.coco.json')
    root_dir = dataset_config['paths']['root']
    
    # Support custom split names (e.g., train_patches, val_patches, test_patches)
    # If paths.train/val/test are provided, use them; otherwise use default names
    paths = dataset_config.get('paths', {})
    train_split = paths.get('train', 'train')
    val_split = paths.get('val', 'val')
    test_split = paths.get('test', 'test')
    
    # Check if val split exists, otherwise use test as val
    from pathlib import Path
    val_annotation_path = Path(root_dir) / val_split / annotation_file
    if not val_annotation_path.exists():
        val_split = test_split
        print("Warning: val split not found, using test split for validation")
        print(f"  Expected: {val_annotation_path}")
        print("  To create a val split, run: python scripts/dataset/split_coco_dataset.py <train/_annotations.coco.json> --output-dir <root_dir> --val-ratio 0.2")

    train_dataset = COCODataset(
        root_dir=root_dir,
        split=train_split,
        annotation_file=annotation_file,
        image_size=tuple(dataset_config.get('image_size', (None, None))) if dataset_config.get('image_size') else None,
        mean=dataset_config.get('mean', [0.485, 0.456, 0.406]),
        std=dataset_config.get('std', [0.229, 0.224, 0.225]),
        normalize=data_config['preprocessing']['normalize'],
        augmentation=augmentation_config,
        min_size=dataset_config.get('min_bbox_size', 1)
    )

    val_dataset = COCODataset(
        root_dir=root_dir,
        split=val_split,
        annotation_file=annotation_file,
        image_size=tuple(dataset_config.get('image_size', (None, None))) if dataset_config.get('image_size') else None,
        mean=dataset_config.get('mean', [0.485, 0.456, 0.406]),
        std=dataset_config.get('std', [0.229, 0.224, 0.225]),
        normalize=data_config['preprocessing']['normalize'],
        augmentation={},  # No augmentation for val
        min_size=dataset_config.get('min_bbox_size', 1)
    )

    test_dataset = COCODataset(
        root_dir=root_dir,
        split=test_split,
        annotation_file=annotation_file,
        image_size=tuple(dataset_config.get('image_size', (None, None))) if dataset_config.get('image_size') else None,
        mean=dataset_config.get('mean', [0.485, 0.456, 0.406]),
        std=dataset_config.get('std', [0.229, 0.224, 0.225]),
        normalize=data_config['preprocessing']['normalize'],
        augmentation={},  # No augmentation for test
        min_size=dataset_config.get('min_bbox_size', 1)
    )
    
    # Custom collate function for detection (handles variable number of bboxes)
    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        bboxes = [item['bboxes'] for item in batch]
        labels = [item['labels'] for item in batch]
        image_ids = [item['image_id'] for item in batch]
        image_names = [item['image_name'] for item in batch]
        original_sizes = [item['original_size'] for item in batch]

        return {
            'image': images,
            'bboxes': bboxes,
            'labels': labels,
            'image_id': image_ids,
            'image_name': image_names,
            'original_size': original_sizes,
        }
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader 