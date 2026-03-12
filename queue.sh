#!/bin/bash

# Queue multiple training experiments to run sequentially
# Usage: ./queue.sh exp_cfinet_soda exp_cfinet_soda_patches

if [ $# -eq 0 ]; then
    echo "Usage: ./queue.sh <experiment1> <experiment2> ..."
    echo ""
    echo "Example:"
    echo "  ./queue.sh exp_cfinet_soda exp_cfinet_soda_patches"
    echo ""
    echo "Available experiments:"
    ls configs/experiments/*.yaml 2>/dev/null | xargs -n 1 basename | sed 's/.yaml$//'
    exit 1
fi

# Create logs directory
LOG_DIR="outputs/queue_logs"
mkdir -p "$LOG_DIR"

# Get timestamp for this queue run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
QUEUE_LOG="$LOG_DIR/queue_${TIMESTAMP}.log"

# Store experiment list
EXPERIMENTS=("$@")
TOTAL=${#EXPERIMENTS[@]}

echo "========================================"
echo "Training Queue Started"
echo "========================================"
echo "Timestamp: $TIMESTAMP"
echo "Total experiments: $TOTAL"
echo "Queue log: $QUEUE_LOG"
echo ""
echo "Experiments to run:"
for i in "${!EXPERIMENTS[@]}"; do
    echo "  $((i+1)). ${EXPERIMENTS[$i]}"
done
echo "========================================"
echo ""

# Initialize log file
{
    echo "========================================"
    echo "Training Queue Log"
    echo "========================================"
    echo "Started: $(date)"
    echo "Total experiments: $TOTAL"
    echo ""
} > "$QUEUE_LOG"

# Track success/failure
SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_EXPS=()

# Run each experiment
for i in "${!EXPERIMENTS[@]}"; do
    EXP="${EXPERIMENTS[$i]}"
    NUM=$((i+1))
    
    echo ""
    echo "========================================"
    echo "[$NUM/$TOTAL] Starting: $EXP"
    echo "========================================"
    echo "Started at: $(date)"
    echo ""
    
    # Log to queue log
    {
        echo ""
        echo "========================================"
        echo "[$NUM/$TOTAL] Experiment: $EXP"
        echo "========================================"
        echo "Started: $(date)"
        echo ""
    } >> "$QUEUE_LOG"
    
    # Create individual experiment log
    EXP_LOG="$LOG_DIR/${EXP}_${TIMESTAMP}.log"
    
    # Run training and capture output
    if ./train.sh "$EXP" 2>&1 | tee "$EXP_LOG"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT+1))
        STATUS="SUCCESS"
        echo ""
        echo "[$NUM/$TOTAL] $EXP - COMPLETED SUCCESSFULLY"
        echo "Completed at: $(date)"
    else
        FAILED_COUNT=$((FAILED_COUNT+1))
        FAILED_EXPS+=("$EXP")
        STATUS="FAILED"
        echo ""
        echo "[$NUM/$TOTAL] $EXP - FAILED"
        echo "Failed at: $(date)"
    fi
    
    # Log result
    {
        echo "Status: $STATUS"
        echo "Completed: $(date)"
        echo "Log file: $EXP_LOG"
    } >> "$QUEUE_LOG"
    
    # Show remaining experiments
    REMAINING=$((TOTAL - NUM))
    if [ $REMAINING -gt 0 ]; then
        echo ""
        echo "Remaining experiments: $REMAINING"
        echo "----------------------------------------"
    fi
done

# Final summary
echo ""
echo "========================================"
echo "Training Queue Completed"
echo "========================================"
echo "Finished: $(date)"
echo ""
echo "Summary:"
echo "  Total experiments: $TOTAL"
echo "  Successful: $SUCCESS_COUNT"
echo "  Failed: $FAILED_COUNT"
echo ""

if [ $FAILED_COUNT -gt 0 ]; then
    echo "Failed experiments:"
    for exp in "${FAILED_EXPS[@]}"; do
        echo "  - $exp"
    done
    echo ""
fi

echo "Queue log: $QUEUE_LOG"
echo "Individual logs: $LOG_DIR/*_${TIMESTAMP}.log"
echo "========================================"

# Write summary to queue log
{
    echo ""
    echo "========================================"
    echo "Queue Summary"
    echo "========================================"
    echo "Finished: $(date)"
    echo "Total experiments: $TOTAL"
    echo "Successful: $SUCCESS_COUNT"
    echo "Failed: $FAILED_COUNT"
    if [ $FAILED_COUNT -gt 0 ]; then
        echo ""
        echo "Failed experiments:"
        for exp in "${FAILED_EXPS[@]}"; do
            echo "  - $exp"
        done
    fi
} >> "$QUEUE_LOG"

# Exit with error if any experiments failed
if [ $FAILED_COUNT -gt 0 ]; then
    exit 1
fi

exit 0

