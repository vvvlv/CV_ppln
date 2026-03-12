"""Testing / inference script for COCO-format object detection experiments."""

import sys
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import load_config
from data import create_dataloaders
from models import create_model
from training.detection_metrics import create_metrics


def _to_numpy(x: Any, dtype=None) -> np.ndarray:
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def test(
    config_path: str,
    checkpoint_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    config = load_config(config_path)
    exp_name = config.get("name", Path(config_path).stem)

    dataset_cfg = config.get("dataset", {})
    is_detection = dataset_cfg.get("format") == "coco" or "annotation_file" in dataset_cfg
    if not is_detection:
        raise ValueError("This pipeline only supports COCO-format object detection experiments.")

    # Device
    if str(config.get("device", "cuda")).startswith("cuda") and torch.cuda.is_available():
        device = torch.device(config["device"])
    else:
        device = torch.device("cpu")

    print(f"\n{'='*70}")
    print(f"Testing: {exp_name}")
    print(f"Device : {device}")
    print(f"{'='*70}")

    # Model
    model = create_model(config).to(device)
    model.eval()
    
    # Checkpoint
    if checkpoint_path is None:
        checkpoint_path = str(Path(config["output"]["dir"]) / "checkpoints" / "best.pth")
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # Data
    _, _, test_loader = create_dataloaders(config)
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Output dir
    if output_dir is None:
        out_dir = Path("outputs/tests") / exp_name
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Evaluation settings
    score_thr = float(config.get("evaluation", {}).get("score_threshold", 0.05))
    metrics = create_metrics(config.get("training", {}).get("metrics", ["mAP"]))
    
    # Accumulators
    pred_bboxes_all: List[np.ndarray] = []
    pred_labels_all: List[np.ndarray] = []
    pred_scores_all: List[np.ndarray] = []
    gt_bboxes_all: List[np.ndarray] = []
    gt_labels_all: List[np.ndarray] = []
    per_image: List[Dict[str, Any]] = []
    
    print("\nRunning inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch["image"].to(device)
            gt_bboxes = batch["bboxes"]
            gt_labels = batch["labels"]
            image_ids = batch.get("image_id", [None] * images.shape[0])
            image_names = batch.get("image_name", [None] * images.shape[0])

            # Predict
            if hasattr(model, "predict"):
                preds = model.predict(images, score_threshold=score_thr)
            else:
                raise RuntimeError("Model does not implement predict(); cannot format detections.")

            for i in range(len(preds)):
                pb = _to_numpy(preds[i]["bboxes"], np.float32)
                pl = _to_numpy(preds[i]["labels"], np.int64)
                ps = _to_numpy(preds[i]["scores"], np.float32)

                gb = gt_bboxes[i].detach().cpu().numpy().astype(np.float32)
                gl = gt_labels[i].detach().cpu().numpy().astype(np.int64)

                pred_bboxes_all.append(pb)
                pred_labels_all.append(pl)
                pred_scores_all.append(ps)
                gt_bboxes_all.append(gb)
                gt_labels_all.append(gl)

                per_image.append(
                    {
                        "image_id": image_ids[i],
                        "image_name": image_names[i],
                        "pred_bboxes_xyxy": pb.tolist(),
                        "pred_scores": ps.tolist(),
                        "pred_labels": pl.tolist(),
                        "gt_bboxes_xyxy": gb.tolist(),
                        "gt_labels": gl.tolist(),
                    }
                )

    # Metrics
    results: Dict[str, float] = {}
    for name, fn in metrics.items():
        results[name] = float(
            fn(pred_bboxes_all, pred_labels_all, pred_scores_all, gt_bboxes_all, gt_labels_all)
        )

    # Save
    with open(out_dir / "predictions.json", "w") as f:
        json.dump(
            {
                "experiment": exp_name,
                "checkpoint": str(ckpt_path),
                "score_threshold": score_thr,
                "metrics": results,
                "per_image": per_image,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("Test metrics:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    print("=" * 70)
    print(f"\n✓ Saved: {out_dir / 'predictions.json'}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a detection model on the test split (COCO format).")
    parser.add_argument("--config", required=True, help="Path to experiment config file")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint (default: <exp_output>/checkpoints/best.pth)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: outputs/tests/<exp_name>)")
    args = parser.parse_args()
    
    test(args.config, args.checkpoint, args.output_dir) 

