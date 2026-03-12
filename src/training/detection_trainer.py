"""Training loop handler for object detection."""

import torch
import torch.nn as nn
import yaml
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
import numpy as np

from utils.tensorboard_logger import TensorBoardLogger


class DetectionTrainer:
    """Training loop manager for object detection."""
    
    def __init__(self, model: nn.Module, train_loader, val_loader, 
                 config: dict, device, tensorboard_logger=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.tb_logger = tensorboard_logger
        
        # Setup components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.metrics = self._create_metrics()
        
        # Training state
        self.current_epoch = 0
        self.early_stop_counter = 0
        self.metrics_history: List[Dict] = []

        # Track "best" separately for checkpointing vs early-stopping (they may monitor different metrics)
        self.best_checkpoint_metric = None
        self.best_early_stop_metric = None
        
        # Create output directory
        self.output_dir = Path(config['output']['dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
    
    def train(self):
        """Main training loop."""
        num_epochs = self.config['training']['epochs']
        
        print(f"\nStarting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print("\n")
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print metrics
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}")
            if 'mAP' in train_metrics:
                print(f"  Train - mAP: {train_metrics['mAP']:.4f}")
            print(f"  Val   - Loss: {val_metrics['val_loss']:.4f}")
            if 'val_mAP' in val_metrics:
                print(f"  Val   - mAP: {val_metrics['val_mAP']:.4f}")
            
            # Log to TensorBoard
            if self.tb_logger is not None:
                self.tb_logger.log_metrics(train_metrics, 'train', epoch + 1)
                self.tb_logger.log_metrics(val_metrics, 'val', epoch + 1)
                self.tb_logger.log_learning_rate(current_lr, epoch + 1)
                self.tb_logger.flush()
            
            # Save metrics history
            epoch_metrics = {
                'epoch': epoch + 1,
                'train': train_metrics,
                'val': val_metrics
            }
            self.metrics_history.append(epoch_metrics)
            self._save_metrics_history()
            
            # Checkpointing
            self._handle_checkpointing(val_metrics)
            
            # Early stopping
            if self._check_early_stopping(val_metrics):
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
            
            # Step scheduler
            self.scheduler.step()
        
        # Log final hyperparameters
        if self.tb_logger is not None:
            self._log_hyperparameters()
        
        print("\n✓ Training complete!")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        loss_components = {}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            bboxes = batch['bboxes']
            labels = batch['labels']
            
            # Prepare targets for MMDetection format
            targets = {
                'bboxes': [bbox.to(self.device) for bbox in bboxes],
                'labels': [label.to(self.device) for label in labels]
            }
            
            # Forward
            self.optimizer.zero_grad()
            loss_dict = self.model(images, targets=targets)
            
            # Sum all losses (handle None values and ensure all are tensors)
            loss_values = []
            for key, value in loss_dict.items():
                if value is not None:
                    if isinstance(value, (list, tuple)):
                        # If it's a list/tuple of losses, sum them
                        loss_values.extend([v for v in value if v is not None])
                    elif isinstance(value, torch.Tensor):
                        loss_values.append(value)
                    elif isinstance(value, (int, float)):
                        # Convert scalar to tensor if needed
                        loss_values.append(torch.tensor(value, device=images.device))
            
            if len(loss_values) == 0:
                # Fallback: create a dummy loss if all are None
                loss = torch.tensor(0.0, device=images.device, requires_grad=True)
            else:
                loss = sum(loss_values)
            
            # Backward
            loss.backward()
            
            if self.config['training'].get('gradient_clip'):
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Accumulate
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if value is not None:
                    if isinstance(value, (list, tuple)):
                        # Sum list values for logging
                        val_sum = sum([v.item() if isinstance(v, torch.Tensor) else (float(v) if isinstance(v, (int, float)) else 0.0) for v in value if v is not None])
                        if key not in loss_components:
                            loss_components[key] = 0
                        loss_components[key] += val_sum
                    elif isinstance(value, torch.Tensor):
                        if key not in loss_components:
                            loss_components[key] = 0
                        loss_components[key] += value.item()
                    elif isinstance(value, (int, float)):
                        if key not in loss_components:
                            loss_components[key] = 0
                        loss_components[key] += float(value)
            
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Average
        metrics = {'loss': total_loss / num_batches}
        for key, value in loss_components.items():
            metrics[key] = value / num_batches
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        
        total_loss = 0
        loss_components = {}
        num_batches = 0

        # For detection metrics (e.g., mAP)
        pred_bboxes_all: List[np.ndarray] = []
        pred_labels_all: List[np.ndarray] = []
        pred_scores_all: List[np.ndarray] = []
        gt_bboxes_all: List[np.ndarray] = []
        gt_labels_all: List[np.ndarray] = []

        score_thr = float(self.config.get('evaluation', {}).get('score_threshold', 0.05))
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]"):
                images = batch['image'].to(self.device)
                bboxes = batch['bboxes']
                labels = batch['labels']
                
                # Prepare targets
                targets = {
                    'bboxes': [bbox.to(self.device) for bbox in bboxes],
                    'labels': [label.to(self.device) for label in labels]
                }
                
                # Forward
                loss_dict = self.model(images, targets=targets)
                
                # Handle case where loss_dict might be a list or tuple
                if not isinstance(loss_dict, dict):
                    # If it's a tuple/list, take the first element (losses dict)
                    if isinstance(loss_dict, (list, tuple)) and len(loss_dict) > 0:
                        loss_dict = loss_dict[0]
                    else:
                        # Fallback: create empty dict
                        loss_dict = {}
                
                # Sum all losses (handle None values and ensure all are tensors)
                loss_values = []
                for key, value in loss_dict.items():
                    if value is not None:
                        if isinstance(value, (list, tuple)):
                            # If it's a list/tuple of losses, sum them
                            loss_values.extend([v for v in value if v is not None])
                        elif isinstance(value, torch.Tensor):
                            loss_values.append(value)
                        elif isinstance(value, (int, float)):
                            # Convert scalar to tensor if needed
                            loss_values.append(torch.tensor(value, device=images.device))
                
                if len(loss_values) == 0:
                    # Fallback: create a dummy loss if all are None
                    loss = torch.tensor(0.0, device=images.device, requires_grad=True)
                else:
                    loss = sum(loss_values)
                
                total_loss += loss.item()
                for key, value in loss_dict.items():
                    if value is not None:
                        if isinstance(value, (list, tuple)):
                            # Sum list values for logging
                            val_sum = sum([v.item() if isinstance(v, torch.Tensor) else (float(v) if isinstance(v, (int, float)) else 0.0) for v in value if v is not None])
                            if key not in loss_components:
                                loss_components[key] = 0
                            loss_components[key] += val_sum
                        elif isinstance(value, torch.Tensor):
                            if key not in loss_components:
                                loss_components[key] = 0
                            loss_components[key] += value.item()
                        elif isinstance(value, (int, float)):
                            if key not in loss_components:
                                loss_components[key] = 0
                            loss_components[key] += float(value)
                
                num_batches += 1

                # Optional: compute detection metrics by running inference (no targets)
                if self.metrics:
                    # Prefer model.predict() if present (it returns numpy arrays)
                    if hasattr(self.model, 'predict'):
                        preds = self.model.predict(images, score_threshold=score_thr)
                    else:
                        # Fallback: assume forward(images) returns already-formatted predictions
                        preds = self.model(images)

                    for i in range(len(preds)):
                        pred_bboxes_all.append(np.asarray(preds[i].get('bboxes', []), dtype=np.float32))
                        pred_labels_all.append(np.asarray(preds[i].get('labels', []), dtype=np.int64))
                        pred_scores_all.append(np.asarray(preds[i].get('scores', []), dtype=np.float32))

                    for i in range(len(bboxes)):
                        gt_bboxes_all.append(bboxes[i].detach().cpu().numpy().astype(np.float32))
                        gt_labels_all.append(labels[i].detach().cpu().numpy().astype(np.int64))
        
        # Average
        metrics = {'val_loss': total_loss / num_batches}
        for key, value in loss_components.items():
            metrics[f'val_{key}'] = value / num_batches

        # Compute configured detection metrics
        for metric_name, metric_fn in (self.metrics or {}).items():
            try:
                val_metric = metric_fn(
                    pred_bboxes_all,
                    pred_labels_all,
                    pred_scores_all,
                    gt_bboxes_all,
                    gt_labels_all,
                )
                metrics[f'val_{metric_name}'] = float(val_metric)
            except Exception as e:
                print(f"Warning: failed to compute metric '{metric_name}': {e}")
        
        return metrics

    @staticmethod
    def _is_better(current: float, best: float, mode: str) -> bool:
        """Return True if current is better than best under mode ('min' or 'max')."""
        if mode == 'min':
            return current < best
        if mode == 'max':
            return current > best
        raise ValueError(f"Unknown mode: {mode} (expected 'min' or 'max')")
    
    def _create_optimizer(self):
        """Create optimizer from config."""
        opt_cfg = self.config['training']['optimizer']
        
        if opt_cfg['type'] == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=opt_cfg['learning_rate'],
                weight_decay=opt_cfg['weight_decay']
            )
        elif opt_cfg['type'] == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=opt_cfg['learning_rate'],
                momentum=opt_cfg.get('momentum', 0.9),
                weight_decay=opt_cfg['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_cfg['type']}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        sched_cfg = self.config['training']['scheduler']
        
        if sched_cfg['type'] == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=sched_cfg['min_lr']
            )
        elif sched_cfg['type'] == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_cfg.get('step_size', 8),
                gamma=sched_cfg.get('gamma', 0.1)
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched_cfg['type']}")
    
    def _create_metrics(self):
        """Create metric functions."""
        from .detection_metrics import create_metrics
        metric_names = self.config['training'].get('metrics', [])
        if metric_names:
            return create_metrics(metric_names)
        return {}
    
    def _handle_checkpointing(self, metrics: Dict[str, float]):
        """Save checkpoints."""
        ckpt_cfg = self.config['training']['checkpoint']
        metric_name = ckpt_cfg['monitor']
        if metric_name not in metrics:
            raise KeyError(f"Checkpoint monitor metric '{metric_name}' not found in metrics: {list(metrics.keys())}")
        current_metric = float(metrics[metric_name])
        mode = ckpt_cfg.get('mode')
        if mode is None:
            # Sensible default: losses -> min, everything else -> max
            mode = 'min' if 'loss' in metric_name.lower() else 'max'
        
        # Initialize best metric if needed
        if self.best_checkpoint_metric is None:
            self.best_checkpoint_metric = float('inf') if mode == 'min' else float('-inf')
        
        # Save best
        if ckpt_cfg['save_best'] and self._is_better(current_metric, self.best_checkpoint_metric, mode):
            self.best_checkpoint_metric = current_metric
            self._save_checkpoint('best.pth')
            print(f"  → New best {metric_name}: {current_metric:.4f}")
        
        # Save last
        if ckpt_cfg['save_last']:
            self._save_checkpoint('last.pth')
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        ckpt_dir = self.output_dir / 'checkpoints'
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_checkpoint_metric': self.best_checkpoint_metric
        }, ckpt_dir / filename)
    
    def _check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """Check if early stopping criteria met."""
        es_cfg = self.config['training']['early_stopping']
        if not es_cfg['enabled']:
            return False
        
        metric_name = es_cfg['monitor']
        if metric_name not in metrics:
            raise KeyError(f"Early-stopping monitor metric '{metric_name}' not found in metrics: {list(metrics.keys())}")
        current_metric = float(metrics[metric_name])
        mode = es_cfg.get('mode', 'min' if 'loss' in metric_name.lower() else 'max')

        if self.best_early_stop_metric is None:
            self.best_early_stop_metric = float('inf') if mode == 'min' else float('-inf')

        if self._is_better(current_metric, self.best_early_stop_metric, mode):
            self.best_early_stop_metric = current_metric
            self.early_stop_counter = 0
            return False

            self.early_stop_counter += 1
            print(f"  → Early stopping: {self.early_stop_counter}/{es_cfg['patience']} (no improvement in {metric_name})")
            return self.early_stop_counter >= es_cfg['patience']
    
    def _save_metrics_history(self):
        """Save metrics history to YAML file."""
        metrics_file = self.output_dir / 'metrics_history.yaml'
        
        serializable_history = []
        for epoch_data in self.metrics_history:
            epoch_dict = {
                'epoch': epoch_data['epoch'],
                'train': {k: float(v) for k, v in epoch_data['train'].items()},
                'val': {k: float(v) for k, v in epoch_data['val'].items()}
            }
            serializable_history.append(epoch_dict)
        
        with open(metrics_file, 'w') as f:
            yaml.dump(serializable_history, f, default_flow_style=False, sort_keys=False)
    
    def _log_hyperparameters(self):
        """Log hyperparameters to TensorBoard."""
        if self.tb_logger is None or len(self.metrics_history) == 0:
            return
        
        hparams = {
            'model': self.config['model']['type'],
            'batch_size': self.config['data']['batch_size'],
            'learning_rate': self.config['training']['optimizer']['learning_rate'],
            'weight_decay': self.config['training']['optimizer']['weight_decay'],
            'optimizer': self.config['training']['optimizer']['type'],
            'scheduler': self.config['training']['scheduler']['type'],
            'epochs': self.config['training']['epochs'],
        }
        
        final_epoch = self.metrics_history[-1]
        final_metrics = {}
        for k, v in final_epoch['val'].items():
            clean_name = k.replace('val_', '')
            final_metrics[f'final_val_{clean_name}'] = float(v)
        
        monitor_metric = self.config['training']['checkpoint']['monitor']
        final_metrics['best_' + monitor_metric.replace('val_', '')] = float(
            self.best_checkpoint_metric if self.best_checkpoint_metric is not None else 0.0
        )
        
        self.tb_logger.log_hyperparameters(hparams, final_metrics)
