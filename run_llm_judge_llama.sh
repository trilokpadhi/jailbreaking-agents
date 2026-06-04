#!/bin/bash

BASE_DIR="/staging/users/tpadhi1/agent-jail-breaking"
SCRIPT="$BASE_DIR/llm-judge-realtime.py"
LOG_DIR="$BASE_DIR/logs/llm_judge_llama"
mkdir -p "$LOG_DIR"

FILES=(
    "$BASE_DIR/generations_llama/llama-base/jailbreak_cot_False_20260505_012457_convo_for_memory_cleaned_0412.csv"
    "$BASE_DIR/generations_llama/llama-base/jailbreak_none_False_20260505_005824_convo_for_memory_cleaned_0412.csv"
    "$BASE_DIR/generations_llama/llama-base/jailbreak_none_True_20260505_010351_convo_for_memory_cleaned_0412.csv"
    "$BASE_DIR/generations_llama/llama-base/jailbreak_react_False_20260505_011449_convo_for_memory_cleaned_0412.csv"
    "$BASE_DIR/generations_llama/llama-ft/jailbreak_cot_False_20260505_014925_convo_for_memory_cleaned_0412.csv"
    "$BASE_DIR/generations_llama/llama-ft/jailbreak_none_False_20260505_013221_convo_for_memory_cleaned_0412.csv"
    "$BASE_DIR/generations_llama/llama-ft/jailbreak_none_True_20260505_013737_convo_for_memory_cleaned_0412.csv"
    "$BASE_DIR/generations_llama/llama-ft/jailbreak_react_False_20260505_014340_convo_for_memory_cleaned_0412.csv"
)

PIDS=()

for FILE in "${FILES[@]}"; do
    FNAME=$(basename "$FILE" .csv)
    LOG="$LOG_DIR/${FNAME}.log"
    echo "Starting: $FNAME"
    python "$SCRIPT" --input_csv "$FILE" > "$LOG" 2>&1 &
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
