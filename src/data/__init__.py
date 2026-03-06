"""Data module."""

import torch
from torch.utils.data import DataLoader
from typing import Tuple
from .dataset import VesselSegmentationDataset
from .coco_dataset import COCODataset


def create_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    
    dataset_config = config['dataset']
    data_config = config['data']
    augmentation_config = data_config.get('augmentation', {})
    
    # Check if this is a detection task (COCO format)
    is_detection = dataset_config.get('format') == 'coco' or 'annotation_file' in dataset_config
    
    if is_detection:
        # Object detection dataset (COCO format)
        annotation_file = dataset_config.get('annotation_file', '_annotations.coco.json')
        root_dir = dataset_config['paths']['root']
        
        # Check if val split exists, otherwise use test as val
        from pathlib import Path
        val_split = 'val'
        val_annotation_path = Path(root_dir) / 'val' / annotation_file
        if not val_annotation_path.exists():
            val_split = 'test'
            print(f"Warning: val split not found, using test split for validation")
            print(f"  To create a val split, run: python scripts/split_coco_dataset.py <train_annotation_file> --output-dir <root_dir> --val-ratio 0.2")
        
        train_dataset = COCODataset(
            root_dir=root_dir,
            split='train',
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
            split='test',
            annotation_file=annotation_file,
            image_size=tuple(dataset_config.get('image_size', (None, None))) if dataset_config.get('image_size') else None,
            mean=dataset_config.get('mean', [0.485, 0.456, 0.406]),
            std=dataset_config.get('std', [0.229, 0.224, 0.225]),
            normalize=data_config['preprocessing']['normalize'],
            augmentation={},  # No augmentation for test
            min_size=dataset_config.get('min_bbox_size', 1)
        )
    else:
        # Segmentation dataset
        train_dataset = VesselSegmentationDataset(
            root_dir=dataset_config['paths']['root'],
            split='train',
            image_size=tuple(dataset_config['image_size']),
            mean=dataset_config['mean'],
            std=dataset_config['std'],
            normalize=data_config['preprocessing']['normalize'],
            num_channels=dataset_config['num_channels'],
            augmentation=augmentation_config
        )
        
        val_dataset = VesselSegmentationDataset(
            root_dir=dataset_config['paths']['root'],
            split='val',
            image_size=tuple(dataset_config['image_size']),
            mean=dataset_config['mean'],
            std=dataset_config['std'],
            normalize=data_config['preprocessing']['normalize'],
            num_channels=dataset_config['num_channels'],
            augmentation=augmentation_config
        )
        
        test_dataset = VesselSegmentationDataset(
            root_dir=dataset_config['paths']['root'],
            split='test',
            image_size=tuple(dataset_config['image_size']),
            mean=dataset_config['mean'],
            std=dataset_config['std'],
            normalize=data_config['preprocessing']['normalize'],
            num_channels=dataset_config['num_channels'],
            augmentation=augmentation_config
        )
    
    # Custom collate function for detection (handles variable number of bboxes)
    def collate_fn(batch):
        if is_detection:
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
        else:
            # Segmentation: standard collate
            from torch.utils.data.dataloader import default_collate
            return default_collate(batch)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        collate_fn=collate_fn if is_detection else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        collate_fn=collate_fn if is_detection else None
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        collate_fn=collate_fn if is_detection else None
    )
    
    return train_loader, val_loader, test_loader 