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


def calculate_ap_per_class(
    pred_bboxes: List[np.ndarray],
    pred_labels: List[np.ndarray],
    pred_scores: List[np.ndarray],
    gt_bboxes: List[np.ndarray],
    gt_labels: List[np.ndarray],
    iou_threshold: float = 0.5
) -> float:
    """Calculate Average Precision (AP) for a single class."""
    # Group predictions and ground truth by class
    # For simplicity, assume single class or aggregate all classes
    all_pred_boxes = []
    all_pred_scores = []
    all_gt_boxes = []
    
    for i in range(len(pred_bboxes)):
        for j in range(len(pred_bboxes[i])):
            all_pred_boxes.append(pred_bboxes[i][j])
            all_pred_scores.append(pred_scores[i][j] if len(pred_scores[i]) > j else 1.0)
        
        for j in range(len(gt_bboxes[i])):
            all_gt_boxes.append(gt_bboxes[i][j])
    
    if len(all_gt_boxes) == 0:
        return 0.0
    
    if len(all_pred_boxes) == 0:
        return 0.0
    
    # Sort predictions by score
    sorted_indices = np.argsort(all_pred_scores)[::-1]
    all_pred_boxes = np.array(all_pred_boxes)[sorted_indices]
    all_pred_scores = np.array(all_pred_scores)[sorted_indices]
    
    # Match predictions to ground truth
    gt_matched = [False] * len(all_gt_boxes)
    tp = []
    fp = []
    
    for pred_box in all_pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(all_gt_boxes):
            if gt_matched[gt_idx]:
                continue
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            tp.append(1)
            fp.append(0)
            gt_matched[best_gt_idx] = True
        else:
            tp.append(0)
            fp.append(1)
    
    # Calculate precision and recall
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recalls = tp / len(all_gt_boxes)
    precisions = tp / (tp + fp + 1e-8)
    
    # Calculate AP using 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    
    return ap


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
    return calculate_ap_per_class(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, iou_threshold
    )


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
