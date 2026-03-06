"""CFINet model wrapper for object detection."""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional

# Add CFINet to path if not already there
cfinet_path = Path(__file__).parent.parent.parent.parent.parent / "CFI" / "CFINet"
if str(cfinet_path) not in sys.path:
    sys.path.insert(0, str(cfinet_path))

try:
    from mmdet.apis import init_detector
    from mmdet.models import build_detector
    from mmcv import Config
except ImportError as e:
    raise ImportError(
        f"Could not import MMDetection. Make sure CFINet is properly installed. Error: {e}"
    )

from ..registry import register_model


@register_model('CFINet')
class CFINetWrapper(nn.Module):
    """
    Wrapper for CFINet object detection model.
    
    This wrapper integrates CFINet (built on MMDetection) with the CV_ppln framework.
    """
    
    def __init__(self, config: dict):
        """
        Initialize CFINet model.
        
        Args:
            config: Model configuration dict containing:
                - config_file: Path to MMDetection config file
                - checkpoint: Optional path to pretrained checkpoint
                - num_classes: Number of object classes
                - device: Device to run on
        """
        super().__init__()
        
        self.config_file = config.get('config_file')
        self.checkpoint = config.get('checkpoint', None)
        self.num_classes = config.get('num_classes', 9)  # Default for SODA
        self.device = config.get('device', 'cuda')
        
        if self.config_file is None:
            raise ValueError("CFINet requires 'config_file' in model config")
        
        # Load MMDetection config
        cfg = Config.fromfile(self.config_file)
        
        # Update num_classes if specified
        if 'roi_head' in cfg.model and 'bbox_head' in cfg.model['roi_head']:
            cfg.model['roi_head']['bbox_head']['num_classes'] = self.num_classes
        
        # Create necessary directories for CFINet (FIRoIHead needs con_queue_dir)
        if 'roi_head' in cfg.model and 'con_queue_dir' in cfg.model['roi_head']:
            con_queue_dir = cfg.model['roi_head']['con_queue_dir']
            # Convert relative path to absolute if needed
            if con_queue_dir.startswith('./'):
                # Make it relative to CV_ppln root directory
                # Get the CV_ppln root directory (go up from src/models/architectures/cfinet.py)
                current_file = Path(__file__).resolve()
                cv_ppln_root = current_file.parent.parent.parent.parent.resolve()
                con_queue_dir = str(cv_ppln_root / con_queue_dir.lstrip('./'))
            # Create directory if it doesn't exist
            Path(con_queue_dir).mkdir(parents=True, exist_ok=True)
            cfg.model['roi_head']['con_queue_dir'] = con_queue_dir
            print(f"Created/using con_queue_dir: {con_queue_dir}")
        
        # Build model
        self.model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')
        )
        
        # Load checkpoint if provided
        if self.checkpoint and Path(self.checkpoint).exists():
            checkpoint = torch.load(self.checkpoint, map_location='cpu')
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded checkpoint from {self.checkpoint}")
        
        self.cfg = cfg
    
    def forward(self, images: torch.Tensor, targets: Optional[Dict] = None):
        """
        Forward pass.
        
        Args:
            images: Input images [B, C, H, W]
            targets: Optional dict with 'bboxes' and 'labels' for training
        
        Returns:
            During training: loss dict
            During inference: detection results
        """
        # If targets are provided, compute losses (for both training and validation)
        if targets is not None:
            img_metas = self._prepare_img_metas(images)
            gt_bboxes = targets['bboxes']  # Already a list of tensors
            gt_labels = targets['labels']  # Already a list of tensors
            
            # MMDetection expects list of tensors on device
            gt_bboxes = [bbox.to(images.device) if isinstance(bbox, torch.Tensor) else bbox for bbox in gt_bboxes]
            gt_labels = [label.to(images.device) if isinstance(label, torch.Tensor) else label for label in gt_labels]
            
            # MMDetection forward_train signature (works in both train and eval mode)
            loss_dict = self.model.forward_train(
                images,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=None
            )
            return loss_dict
        else:
            # Inference mode (no targets)
            img_metas = self._prepare_img_metas(images)
            results = self.model.simple_test(
                images,
                img_metas=img_metas,
                rescale=True
            )
            return results
    
    def _prepare_img_metas(self, images: torch.Tensor) -> List[Dict]:
        """Prepare image metadata for MMDetection."""
        batch_size = images.shape[0]
        _, _, h, w = images.shape
        
        img_metas = []
        for i in range(batch_size):
            img_meta = {
                'img_shape': (h, w, 3),
                'ori_shape': (h, w, 3),
                'pad_shape': (h, w, 3),
                'scale_factor': np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),  # [w_scale, h_scale, w_scale, h_scale]
                'flip': False,
                'filename': f'image_{i}.jpg',
                'batch_input_shape': (h, w),
            }
            img_metas.append(img_meta)
        
        return img_metas
    
    def predict(self, images: torch.Tensor, score_threshold: float = 0.5):
        """
        Run inference and return formatted results.
        
        Args:
            images: Input images [B, C, H, W]
            score_threshold: Minimum confidence score
        
        Returns:
            List of detection results per image
        """
        self.eval()
        with torch.no_grad():
            results = self.forward(images)
        
        # Format results
        formatted_results = []
        for i, result in enumerate(results):
            if len(result) == 0:
                formatted_results.append({
                    'bboxes': [],
                    'labels': [],
                    'scores': []
                })
                continue
            
            # Extract bboxes, labels, scores
            bboxes = result[:, :4]  # [N, 4]
            scores = result[:, 4]   # [N]
            labels = result[:, 5].long() if result.shape[1] > 5 else torch.zeros(len(bboxes), dtype=torch.long)
            
            # Filter by score threshold
            mask = scores >= score_threshold
            bboxes = bboxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            
            formatted_results.append({
                'bboxes': bboxes.cpu().numpy(),
                'labels': labels.cpu().numpy(),
                'scores': scores.cpu().numpy()
            })
        
        return formatted_results
