#!/bin/bash
# Testing/Inference launcher script (object detection)
# Loads a checkpoint, runs inference on test images, calculates metrics, and saves predictions.json

set -e

CONFIG_DIR="configs/experiments"
OUTPUT_DIR="outputs/tests"

if [ $# -eq 0 ]; then
    echo "Usage: ./test.sh <experiment_name> [checkpoint_name]"
    echo ""
    echo "Arguments:"
    echo "  experiment_name  Name of the experiment config file (without .yaml)"
    echo "  checkpoint_name  Optional: 'best' or 'last' (default: best)"
    echo ""
    echo "What it does:"
    echo "  - Loads the trained model checkpoint"
    echo "  - Runs inference on test images"
    echo "  - Calculates detection metrics (e.g., mAP@0.5)"
    echo "  - Saves predictions + metrics to: outputs/tests/<experiment_name>/predictions.json"
    echo ""
    echo "Available experiments:"
    ls -1 "$CONFIG_DIR"/*.yaml 2>/dev/null | xargs -n 1 basename | sed 's/.yaml//' || echo "  (none yet)"
    echo ""
    echo "Examples:"
    echo "  ./test.sh exp001_basic_unet         # Uses best.pth"
    echo "  ./test.sh exp001_basic_unet best    # Uses best.pth"
    echo "  ./test.sh exp001_basic_unet last    # Uses last.pth"
    exit 1
fi

EXPERIMENT=$1
CHECKPOINT_NAME=${2:-best}  # Default to 'best' if not provided (best|last)

CONFIG_FILE="$CONFIG_DIR/${EXPERIMENT}.yaml"
EXPERIMENT_OUTPUT_DIR=$(grep -E "^\s*dir:" "$CONFIG_FILE" | head -1 | sed 's/.*dir:\s*"\?\([^"]*\)"\?.*/\1/' | tr -d '"')
CHECKPOINT_FILE="${EXPERIMENT_OUTPUT_DIR}/checkpoints/${CHECKPOINT_NAME}.pth"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_FILE"
    echo ""
    echo "Available checkpoints for $EXPERIMENT:"
    ls -1 "$CHECKPOINT_DIR/${EXPERIMENT}/checkpoints/"*.pth 2>/dev/null || echo "  (none found)"
    exit 1
fi

echo "=========================================="
echo "Running Inference & Testing"
echo "=========================================="
echo "Experiment: $EXPERIMENT"
echo "Config: $CONFIG_FILE"
echo "Checkpoint: $CHECKPOINT_FILE"
echo "Output: $OUTPUT_DIR/$EXPERIMENT/"
echo "=========================================="

python scripts/test.py --config "$CONFIG_FILE" --checkpoint "$CHECKPOINT_FILE"

echo ""
echo "=========================================="
echo "✓ Testing complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR/$EXPERIMENT/"
echo "  - Predictions + metrics: $OUTPUT_DIR/$EXPERIMENT/predictions.json"
echo "=========================================="
