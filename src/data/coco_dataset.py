"""COCO format dataset for object detection."""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
import random


class COCODataset(Dataset):
    """Dataset for COCO format object detection annotations."""
    
    def __init__(
        self,
        root_dir: str,
        split: str,
        annotation_file: str,
        image_size: Optional[Tuple[int, int]] = None,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        normalize: bool = True,
        augmentation: Optional[dict] = None,
        min_size: int = 1,
    ):
        """
        Initialize COCO dataset.
        
        Args:
            root_dir: Root directory containing train/val/test splits
            split: "train", "val", or "test"
            annotation_file: Path to COCO format JSON annotation file
            image_size: Optional target image size (H, W). If None, uses original size.
            mean: Normalization mean for RGB channels
            std: Normalization std for RGB channels
            normalize: Whether to normalize images
            augmentation: Augmentation configuration dict
            min_size: Minimum bbox size to keep (filter out tiny boxes)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.normalize = normalize
        self.min_size = min_size
        self.augmentation = augmentation or {}
        self.use_augmentation = self.augmentation.get('enabled', False) and self.split == 'train'
        
        # Load annotations
        annotation_path = self.root_dir / split / annotation_file
        with open(annotation_path, 'r') as f:
            coco_data = json.load(f)
        
        # Build mappings
        self.images = {img['id']: img for img in coco_data['images']}
        self.categories = {cat['id']: cat for cat in coco_data['categories']}
        self.annotations = coco_data['annotations']
        
        # Group annotations by image_id
        self.image_to_anns = {}
        for ann in self.annotations:
            image_id = ann['image_id']
            if image_id not in self.image_to_anns:
                self.image_to_anns[image_id] = []
            self.image_to_anns[image_id].append(ann)
        
        # Get image IDs for this split
        self.image_ids = sorted(self.images.keys())
        
        # Filter out images with no annotations (or keep them for inference)
        if split == 'train':
            self.image_ids = [img_id for img_id in self.image_ids if img_id in self.image_to_anns]
        
        print(f"Loaded {len(self.image_ids)} images for {split} split")
        print(f"Categories: {len(self.categories)}")
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample."""
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        
        # Load image
        image_path = self.root_dir / self.split / image_info['file_name']
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]
        
        # Get annotations for this image
        anns = self.image_to_anns.get(image_id, [])
        
        # Extract bboxes and labels
        bboxes = []
        labels = []
        
        for ann in anns:
            # COCO bbox format: [x, y, width, height]
            bbox = ann['bbox']
            x, y, w, h = bbox
            
            # Filter by minimum size
            if w < self.min_size or h < self.min_size:
                continue
            
            # Convert to [x1, y1, x2, y2] format
            bbox_xyxy = [x, y, x + w, y + h]
            bboxes.append(bbox_xyxy)
            labels.append(ann['category_id'])
        
        # Convert to numpy arrays
        bboxes = np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
        
        # Apply augmentations
        if self.use_augmentation:
            image, bboxes, labels = self._apply_augmentations(image, bboxes, labels)
        
        # Resize if needed
        if self.image_size is not None:
            target_h, target_w = self.image_size
            if (original_h, original_w) != (target_h, target_w):
                # Resize image
                image = cv2.resize(image, (target_w, target_h))
                
                # Resize bboxes
                if len(bboxes) > 0:
                    scale_x = target_w / original_w
                    scale_y = target_h / original_h
                    bboxes[:, [0, 2]] *= scale_x
                    bboxes[:, [1, 3]] *= scale_y
        
        # Normalize image
        image = image.astype(np.float32) / 255.0
        if self.normalize:
            image = (image - self.mean) / self.std
        
        # Convert to tensor format (CHW)
        image = image.transpose(2, 0, 1)
        
        # Prepare output
        result = {
            'image': torch.from_numpy(image),
            'bboxes': torch.from_numpy(bboxes),
            'labels': torch.from_numpy(labels),
            'image_id': image_id,
            'image_name': image_info['file_name'],
            'original_size': (original_h, original_w),
        }
        
        return result
    
    def _apply_augmentations(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray):
        """Apply data augmentations."""
        aug = self.augmentation
        h, w = image.shape[:2]
        
        # Horizontal flip
        h_flip_prob = aug.get('horizontal_flip', 0.0)
        if h_flip_prob > 0 and random.random() < h_flip_prob:
            image = np.flip(image, axis=1)
            if len(bboxes) > 0:
                bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]  # Swap x1 and x2
        
        # Vertical flip
        v_flip_prob = aug.get('vertical_flip', 0.0)
        if v_flip_prob > 0 and random.random() < v_flip_prob:
            image = np.flip(image, axis=0)
            if len(bboxes) > 0:
                bboxes[:, [1, 3]] = h - bboxes[:, [3, 1]]  # Swap y1 and y2
        
        # Random brightness
        brightness = aug.get('brightness', 0.0)
        if brightness > 0:
            factor = 1.0 + random.uniform(-brightness, brightness)
            image = np.clip(image * factor, 0.0, 255.0)
        
        return image, bboxes, labels
    
    def get_categories(self) -> Dict:
        """Get category information."""
        return self.categories
