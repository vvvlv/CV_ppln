"""CFINet model wrapper for object detection."""

from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional

try:
    from mmdet.models import build_detector
    from mmcv import Config
except ImportError as e:
    raise ImportError(
        f"Could not import MMDetection. Please install MMDetection: pip install mmdet>=2.26.0 mmcv-full>=1.5.0. Error: {e}"
    )

from ..registry import register_model


@register_model('CFINet')
class CFINetWrapper(nn.Module):
    """
    CFINet object detection model.
    
    This is the CFINet (Coarse-to-fine Proposal Generation and Imitation Learning Network)
    model implementation, built on MMDetection.
    """
    
    @staticmethod
    def _get_model_config(num_classes: int = 9, rpn_weight: float = 0.9):
        """
        Get CFINet model configuration.
        
        This defines the CFINet architecture: ResNet-50 backbone, FPN neck,
        CRPNHead (Coarse-to-fine RPN Head), and FIRoIHead (Feature Imitation ROI Head).
        
        Args:
            num_classes: Number of object classes
            rpn_weight: RPN loss weight
        
        Returns:
            MMDetection config dict with 'model', 'train_cfg', 'test_cfg' at top level
        """
        # Build model dict with train_cfg and test_cfg inside
        model_dict = dict(
        type='FasterRCNN',
        # Backbone: ResNet-50
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
        ),
        # Neck: Feature Pyramid Network
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=4
        ),
        # RPN Head: Coarse-to-fine RPN Head (CFINet specific)
        rpn_head=dict(
            type='CRPNHead',
            num_stages=2,
            stages=[
                dict(
                    type='StageRefineRPNHead',
                    in_channels=256,
                    feat_channels=256,
                    anchor_generator=dict(
                        type='AnchorGenerator',
                        scales=[2],
                        ratios=[1.0],
                        strides=[4, 8, 16, 32]
                    ),
                    refine_reg_factor=200.0,
                    refine_cfg=dict(type='dilation', dilation=3),
                    refined_feature=True,
                    sampling=False,
                    with_cls=False,
                    reg_decoded_bbox=True,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=(.0, .0, .0, .0),
                        target_stds=(0.1, 0.1, 0.5, 0.5)
                    ),
                    loss_bbox=dict(
                        type='IoULoss',
                        linear=True,
                        loss_weight=10.0 * rpn_weight
                    )
                ),
                dict(
                    type='StageRefineRPNHead',
                    in_channels=256,
                    feat_channels=256,
                    refine_cfg=dict(type='offset'),
                    refined_feature=True,
                    sampling=True,
                    with_cls=True,
                    reg_decoded_bbox=True,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=(.0, .0, .0, .0),
                        target_stds=(0.05, 0.05, 0.1, 0.1)
                    ),
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=True,
                        loss_weight=1.0 * rpn_weight
                    ),
                    loss_bbox=dict(
                        type='IoULoss',
                        linear=True,
                        loss_weight=10.0 * rpn_weight
                    )
                )
            ]
        ),
        # ROI Head: Feature Imitation ROI Head (CFINet specific)
        roi_head=dict(
            type='FIRoIHead',
            num_gpus=1,
            temperature=0.6,
            contrast_loss_weights=0.50,
            num_con_queue=256,
            con_sampler_cfg=dict(
                num=32,  # Reduced to work with smaller batches and fewer proposals
                pos_fraction=[0.5, 0.25, 0.125]
            ),
            con_queue_dir="./work_dirs/roi_feats/cfinet",
            ins_quality_assess_cfg=dict(
                cls_score=0.05,
                hq_score=0.65,
                lq_score=0.25,
                hq_pro_counts_thr=8
            ),
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]
            ),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]
                ),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0
                ),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0)
            )
        ),
        # Training configuration (inside model dict)
        train_cfg=dict(
            rpn=[
                dict(
                    assigner=dict(
                        type='DynamicAssigner',
                        low_quality_iou_thr=0.2,
                        base_pos_iou_thr=0.25,
                        neg_iou_thr=0.15
                    ),
                    allowed_border=-1,
                    pos_weight=-1,
                    debug=False
                ),
                dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.7,
                        neg_iou_thr=0.7,
                        min_pos_iou=0.3,
                        ignore_iof_thr=-1
                    ),
                    sampler=dict(
                        type='RandomSampler',
                        num=256,
                        pos_fraction=0.5,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=False
                    ),
                    allowed_border=-1,
                    pos_weight=-1,
                    debug=False
                )
            ],
            rpn_proposal=dict(
                max_per_img=300,
                nms=dict(iou_threshold=0.8),
                min_bbox_size=0
            ),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.50,
                    neg_iou_thr=0.50,
                    min_pos_iou=0.50
                ),
                sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5),
                pos_weight=-1
            )
        ),
        # Testing configuration (inside model dict)
        test_cfg=dict(
            rpn=dict(
                max_per_img=300,
                nms=dict(iou_threshold=0.5),
                min_bbox_size=0
            ),
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100
            )
        )
        )
        
        # Return config dict with model (train_cfg and test_cfg are inside model dict)
        config_dict = {
            'model': model_dict,
            'find_unused_parameters': True,
            'fp16': dict(loss_scale='dynamic')
        }
        
        return config_dict
    
    def __init__(self, config: dict):
        """
        Initialize CFINet model.
        
        Args:
            config: Model configuration dict containing:
                - config_file: Optional path to custom MMDetection config file. If None, uses CFINet model config.
                - checkpoint: Optional path to pretrained checkpoint
                - num_classes: Number of object classes
                - device: Device to run on
        """
        super().__init__()
        
        self.config_file = config.get('config_file')
        self.checkpoint = config.get('checkpoint', None)
        self.num_classes = config.get('num_classes', 9)  # Default for SODA
        self.device = config.get('device', 'cuda')
        
        # Load config: either from file or use default embedded config
        if self.config_file is not None:
            # Load from external config file
            config_path = Path(self.config_file)
            architectures_dir = Path(__file__).resolve().parent
            
            if not config_path.is_absolute():
                # Try relative to architectures folder first
                config_path = (architectures_dir / self.config_file).resolve()
                
                # If not found, try relative to project root as fallback
                if not config_path.exists():
                    # Find project root (directory containing 'configs' folder)
                    current_file = Path(__file__).resolve()
                    project_root = current_file.parent.parent.parent.parent.resolve()
                    # Walk up to find configs directory
                    while project_root != project_root.parent:
                        if (project_root / 'configs').exists():
                            break
                        project_root = project_root.parent
                    else:
                        # Fallback: use current file's parent
                        project_root = current_file.parent.parent.parent.parent.resolve()
                    config_path = (project_root / self.config_file).resolve()
            
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Model config file not found: {config_path}\n"
                    f"Original path: {self.config_file}\n"
                    f"Tried relative to architectures folder: {architectures_dir / self.config_file}\n"
                    f"Architectures folder: {architectures_dir}"
                )
            
            # Load MMDetection config from file
            cfg = Config.fromfile(str(config_path))
            
            # Update num_classes if specified (Config objects support both dict and attribute access)
            if hasattr(cfg.model, 'roi_head') and hasattr(cfg.model.roi_head, 'bbox_head'):
                cfg.model.roi_head.bbox_head.num_classes = self.num_classes
            elif 'roi_head' in cfg.model and 'bbox_head' in cfg.model['roi_head']:
                cfg.model['roi_head']['bbox_head']['num_classes'] = self.num_classes
        else:
            # Use CFINet model config (embedded in this class)
            cfg_dict = self._get_model_config(num_classes=self.num_classes)
            # Convert to MMDetection Config object (supports attribute access)
            cfg = Config(cfg_dict)
        
        # Create necessary directories for CFINet (FIRoIHead needs con_queue_dir)
        # Config objects support both dict-style and attribute access
        try:
            # Try attribute access first (Config object)
            con_queue_dir = cfg.model.roi_head.con_queue_dir
        except (AttributeError, KeyError):
            # Fall back to dict access
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
        # Update config (Config objects support both access methods)
        try:
            cfg.model.roi_head.con_queue_dir = con_queue_dir
        except (AttributeError, KeyError):
            cfg.model['roi_head']['con_queue_dir'] = con_queue_dir
        print(f"Created/using con_queue_dir: {con_queue_dir}")
        
        # Build model (train_cfg and test_cfg are already inside cfg.model)
        # cfg.model should be a Config object that supports attribute access
        self.model = build_detector(cfg.model)
        
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

    @staticmethod
    def _format_simple_test_results(results) -> List[Dict[str, np.ndarray]]:
        """
        Convert MMDetection `simple_test` outputs into a unified format.

        Returns a list (len = batch) of dicts:
          - bboxes: (N, 4) float32 in xyxy
          - labels: (N,) int64 (0-based class index)
          - scores: (N,) float32

        Notes:
        - For Faster R-CNN style detectors, MMDet typically returns for each image a
          list of length `num_classes`, where each entry is an (Ni, 5) ndarray
          [x1, y1, x2, y2, score].
        - Sometimes results can be a tuple (bbox_results, segm_results); we only use bbox.
        """
        formatted: List[Dict[str, np.ndarray]] = []

        if results is None:
            return formatted

        # MMDet returns one entry per image.
        for per_image in results:
            # Handle (bbox_results, segm_results)
            if isinstance(per_image, (list, tuple)) and len(per_image) == 2 and isinstance(per_image[0], (list, tuple)):
                per_image = per_image[0]

            bboxes_list: List[np.ndarray] = []
            labels_list: List[np.ndarray] = []
            scores_list: List[np.ndarray] = []

            # Case A: list of class-wise arrays
            if isinstance(per_image, (list, tuple)):
                for cls_idx, cls_dets in enumerate(per_image):
                    if cls_dets is None:
                        continue
                    cls_dets = np.asarray(cls_dets)
                    if cls_dets.size == 0:
                        continue
                    # Expect (N,5): xyxy + score
                    if cls_dets.ndim != 2 or cls_dets.shape[1] < 5:
                        raise ValueError(f"Unexpected MMDet det shape for class {cls_idx}: {cls_dets.shape}")
                    bboxes_list.append(cls_dets[:, :4].astype(np.float32))
                    scores_list.append(cls_dets[:, 4].astype(np.float32))
                    labels_list.append(np.full((cls_dets.shape[0],), cls_idx, dtype=np.int64))

            # Case B: single array (N,5) with no class separation
            else:
                dets = np.asarray(per_image)
                if dets.size != 0:
                    if dets.ndim != 2 or dets.shape[1] < 5:
                        raise ValueError(f"Unexpected MMDet det shape: {dets.shape}")
                    bboxes_list.append(dets[:, :4].astype(np.float32))
                    scores_list.append(dets[:, 4].astype(np.float32))
                    labels_list.append(np.zeros((dets.shape[0],), dtype=np.int64))

            if bboxes_list:
                bboxes = np.concatenate(bboxes_list, axis=0)
                scores = np.concatenate(scores_list, axis=0)
                labels = np.concatenate(labels_list, axis=0)
            else:
                bboxes = np.zeros((0, 4), dtype=np.float32)
                scores = np.zeros((0,), dtype=np.float32)
                labels = np.zeros((0,), dtype=np.int64)

            formatted.append({'bboxes': bboxes, 'labels': labels, 'scores': scores})

        return formatted
    
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
        
        formatted = self._format_simple_test_results(results)

        # Apply score threshold
        for item in formatted:
            scores = item['scores']
            if scores.size == 0:
                continue
            keep = scores >= score_threshold
            item['bboxes'] = item['bboxes'][keep]
            item['labels'] = item['labels'][keep]
            item['scores'] = item['scores'][keep]

        return formatted
