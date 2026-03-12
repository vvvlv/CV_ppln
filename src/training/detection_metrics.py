"""Evaluation metrics for object detection."""

import torch
import numpy as np
from typing import Dict, Callable, List, Tuple
from collections import defaultdict


def calculate_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Calculate IoU between two bboxes in [x1, y1, x2, y2] format."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_ap(
    pred_bboxes: List[np.ndarray],
    pred_scores: List[np.ndarray],
    gt_bboxes: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> float:
    """Calculate Average Precision (AP) at a fixed IoU threshold for one class."""
    all_pred_boxes: List[np.ndarray] = []
    all_pred_scores: List[float] = []
    all_gt_boxes: List[np.ndarray] = []

    for i in range(len(gt_bboxes)):
        if len(gt_bboxes[i]) > 0:
            all_gt_boxes.extend(list(gt_bboxes[i]))
    
    for i in range(len(pred_bboxes)):
        for j in range(len(pred_bboxes[i])):
            all_pred_boxes.append(pred_bboxes[i][j])
            all_pred_scores.append(float(pred_scores[i][j]) if len(pred_scores[i]) > j else 1.0)
        
    if len(all_gt_boxes) == 0 or len(all_pred_boxes) == 0:
        return 0.0
    
    # Sort predictions by score desc
    sorted_indices = np.argsort(all_pred_scores)[::-1]
    all_pred_boxes = np.asarray(all_pred_boxes, dtype=np.float32)[sorted_indices]
    
    # Match predictions to GT (greedy 1-to-1)
    gt_matched = [False] * len(all_gt_boxes)
    tp = np.zeros((len(all_pred_boxes),), dtype=np.float32)
    fp = np.zeros((len(all_pred_boxes),), dtype=np.float32)
    
    for k, pred_box in enumerate(all_pred_boxes):
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(all_gt_boxes):
            if gt_matched[gt_idx]:
                continue
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[k] = 1.0
            gt_matched[best_gt_idx] = True
        else:
            fp[k] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / (len(all_gt_boxes) + 1e-8)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-8)
    
    # 11-point interpolation AP
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        p = np.max(precisions[recalls >= t]) if np.any(recalls >= t) else 0.0
        ap += float(p) / 11.0
    
    return float(ap)


def mean_average_precision(
    pred_bboxes: List[np.ndarray],
    pred_labels: List[np.ndarray],
    pred_scores: List[np.ndarray],
    gt_bboxes: List[np.ndarray],
    gt_labels: List[np.ndarray],
    iou_threshold: float = 0.5
) -> float:
    """
    Calculate mean Average Precision (mAP).
    
    Args:
        pred_bboxes: List of predicted bboxes per image [N, 4]
        pred_labels: List of predicted labels per image [N]
        pred_scores: List of predicted scores per image [N]
        gt_bboxes: List of ground truth bboxes per image [M, 4]
        gt_labels: List of ground truth labels per image [M]
        iou_threshold: IoU threshold for positive detection
    
    Returns:
        mAP score
    """
    # Compute per-class AP and average.
    # Labels are assumed to be integer class indices (e.g., 0..C-1).
    classes = set()
    for labs in gt_labels:
        classes.update(set(np.asarray(labs, dtype=np.int64).tolist()))
    if len(classes) == 0:
        return 0.0

    aps: List[float] = []
    for cls in sorted(classes):
        cls_pred_bboxes: List[np.ndarray] = []
        cls_pred_scores: List[np.ndarray] = []
        cls_gt_bboxes: List[np.ndarray] = []

        for i in range(len(gt_bboxes)):
            # GT for class
            gl = np.asarray(gt_labels[i], dtype=np.int64)
            gb = np.asarray(gt_bboxes[i], dtype=np.float32)
            gt_mask = (gl == cls)
            cls_gt_bboxes.append(gb[gt_mask] if gb.size else np.zeros((0, 4), dtype=np.float32))

            # Pred for class
            pl = np.asarray(pred_labels[i], dtype=np.int64)
            pb = np.asarray(pred_bboxes[i], dtype=np.float32)
            ps = np.asarray(pred_scores[i], dtype=np.float32)
            pred_mask = (pl == cls)
            cls_pred_bboxes.append(pb[pred_mask] if pb.size else np.zeros((0, 4), dtype=np.float32))
            cls_pred_scores.append(ps[pred_mask] if ps.size else np.zeros((0,), dtype=np.float32))

        aps.append(calculate_ap(cls_pred_bboxes, cls_pred_scores, cls_gt_bboxes, iou_threshold=iou_threshold))

    return float(np.mean(aps)) if aps else 0.0


def average_precision_at_iou(
    pred_bboxes: List[np.ndarray],
    pred_labels: List[np.ndarray],
    pred_scores: List[np.ndarray],
    gt_bboxes: List[np.ndarray],
    gt_labels: List[np.ndarray],
    iou_threshold: float = 0.5
) -> float:
    """Alias for mean_average_precision."""
    return mean_average_precision(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, iou_threshold
    )


METRICS = {
    'mAP': mean_average_precision,
    'ap': average_precision_at_iou,
    'map': mean_average_precision,
}


def create_metrics(metric_names: list) -> Dict[str, Callable]:
    """Create metric functions from names."""
    metrics = {}
    for name in metric_names:
        if name not in METRICS:
            raise ValueError(f"Unknown metric: {name}. Available: {list(METRICS.keys())}")
        metrics[name] = METRICS[name]
    return metrics
