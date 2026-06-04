#!/bin/bash

BASE_DIR="/staging/users/tpadhi1/agent-jail-breaking"
SCRIPT="$BASE_DIR/llm-judge-realtime.py"
LOG_DIR="$BASE_DIR/logs/llm_judge_deepseek"
mkdir -p "$LOG_DIR"

FILES=(
    "$BASE_DIR/generations_deepseek/deepseek-base/jailbreak_cot_False_20260420_193241_convo_for_memory_cleaned_0412.csv"
    "$BASE_DIR/generations_deepseek/deepseek-base/jailbreak_none_False_20260420_151938_convo_for_memory_cleaned_0412.csv"
    "$BASE_DIR/generations_deepseek/deepseek-base/jailbreak_none_True_20260421_233009_convo_for_memory_cleaned_0412.csv"
    "$BASE_DIR/generations_deepseek/deepseek-base/jailbreak_react_False_20260420_173324_convo_for_memory_cleaned_0412.csv"
    "$BASE_DIR/generations_deepseek/deepseek-ft/jailbreak_cot_False_20260421_011538_convo_for_memory_cleaned_0412.csv"
    "$BASE_DIR/generations_deepseek/deepseek-ft/jailbreak_none_False_20260420_213545_convo_for_memory_cleaned_0412.csv"
    "$BASE_DIR/generations_deepseek/deepseek-ft/jailbreak_none_True_20260422_010752_convo_for_memory_cleaned_0412.csv"
    "$BASE_DIR/generations_deepseek/deepseek-ft/jailbreak_react_False_20260420_234046_convo_for_memory_cleaned_0412.csv"
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
