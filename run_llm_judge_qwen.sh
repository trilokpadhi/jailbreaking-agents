#!/bin/bash

BASE_DIR="/staging/users/tpadhi1/agent-jail-breaking"
SCRIPT="$BASE_DIR/llm-judge-realtime.py"
LOG_DIR="$BASE_DIR/logs/llm_judge_qwen"
mkdir -p "$LOG_DIR"

# (agents) generations_qwen $ unzip qwen3_base-20260511T205311Z-3-001.zip
# Archive:  qwen3_base-20260511T205311Z-3-001.zip
#   inflating: qwen3_base/qwen3_30b_base_react_convos.csv  
#   inflating: qwen3_base/qwen3_30b_base_cot_convos.csv  
#   inflating: qwen3_base/qwen3_30b_base_memory_convos.csv  
#   inflating: qwen3_base/Qwen3_base_convos.csv  
# (agents) generations_qwen $ unzip qwen3_ft-20260511T205321Z-3-001.zip 
# Archive:  qwen3_ft-20260511T205321Z-3-001.zip
#   inflating: qwen3_ft/ft_qwen3_30b_convos.csv  
#   inflating: qwen3_ft/ft_qwen3_30b_memory_convos.csv  
#   inflating: qwen3_ft/ft_qwen3_30b_cot_convos.csv  
#   inflating: qwen3_ft/ft_qwen3_30b_react_convos.csv  
FILES=(
    "$BASE_DIR/generations_qwen/qwen3_base/Qwen3_base_convos.csv"
    "$BASE_DIR/generations_qwen/qwen3_base/qwen3_30b_base_cot_convos.csv"
    "$BASE_DIR/generations_qwen/qwen3_base/qwen3_30b_base_react_convos.csv"
    "$BASE_DIR/generations_qwen/qwen3_base/qwen3_30b_base_memory_convos.csv"
    "$BASE_DIR/generations_qwen/qwen3_ft/ft_qwen3_30b_convos.csv"
    "$BASE_DIR/generations_qwen/qwen3_ft/ft_qwen3_30b_cot_convos.csv"
    "$BASE_DIR/generations_qwen/qwen3_ft/ft_qwen3_30b_react_convos.csv"
    "$BASE_DIR/generations_qwen/qwen3_ft/ft_qwen3_30b_memory_convos.csv"
)

PIDS=()

for FILE in "${FILES[@]}"; do
    FNAME=$(basename "$FILE" .csv)
    LOG="$LOG_DIR/${FNAME}.log"
    echo "Starting: $FNAME"
    /home/tpadhi1/miniconda3/envs/agents/bin/python "$SCRIPT" --input_csv "$FILE" > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All 8 jobs launched. PIDs: ${PIDS[*]}"
echo "Logs in: $LOG_DIR"
echo ""

# Wait for all and report
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    STATUS=$?
    FNAME=$(basename "${FILES[$i]}" .csv)
    if [[ $STATUS -eq 0 ]]; then
        echo "✓ Done: $FNAME"
    else
        echo "✗ Failed (exit $STATUS): $FNAME"
    fi
done

echo ""
echo "All jobs complete."
