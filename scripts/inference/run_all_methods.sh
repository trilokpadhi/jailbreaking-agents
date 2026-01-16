#!/bin/bash
# Run all planning methods (cot, react, memory) for all LoRA weights
#
# Prerequisites: Start vLLM server first with:
#
#   python -m vllm.entrypoints.openai.api_server \
#       --model /data50/shared_models/tsutar3_hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
#       --enable-lora \
#       --lora-modules \
#           20pct=/data50/shared_models/finetuned/llama-3.1-8b-20pct-lora \
#           40pct=/data50/shared_models/finetuned/llama-3.1-8b-40pct-lora \
#           60pct=/data50/shared_models/finetuned/llama-3.1-8b-60pct-lora \
#           80pct=/data50/shared_models/finetuned/llama-3.1-8b-80pct-lora \
#       --max-model-length 4096 \
#       --tensor-parallel-size 8 \
#       --port 8000
#
# Usage:
#   ./run_all_methods.sh --method cot             # Run only CoT method
#   ./run_all_methods.sh --pct 10                 # Run only 10pct LoRA for all methods
#   ./run_all_methods.sh --method react --pct 20  # Run ReAct for 20pct only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/vllm_lora_inference_methods.py"

# Default values
METHODS=("cot" "react" "memory")
PERCENTAGES=("10" "20" "40" "60" "80")
MAX_WORKERS=32
BASE_URL="http://localhost:8000"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            METHODS=("$2")
            shift 2
            ;;
        --pct)
            PERCENTAGES=("$2")
            shift 2
            ;;
        --max_workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --base_url)
            BASE_URL="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --method METHOD      Run only specified method (cot, react, memory)"
            echo "  --pct PCT            Run only specified percentage (10, 20, 40, 60, 80, 100)"
            echo "  --max_workers N      Max concurrent workers (default: 32)"
            echo "  --base_url URL       vLLM server URL (default: http://localhost:8000)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Running conversation generation with:"
echo "  Methods: ${METHODS[*]}"
echo "  LoRAs: ${PERCENTAGES[*]}pct"
echo "  Server: ${BASE_URL}"
echo "  Workers: ${MAX_WORKERS}"
echo "=============================================="

TOTAL_RUNS=$((${#METHODS[@]} * ${#PERCENTAGES[@]}))
CURRENT_RUN=0

for method in "${METHODS[@]}"; do
    for pct in "${PERCENTAGES[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))

        LORA_NAME="${pct}pct"
        INPUT_CSV="${pct}pct.csv"

        echo ""
        echo "=============================================="
        echo "[$CURRENT_RUN/$TOTAL_RUNS] Running: method=$method, lora=$LORA_NAME"
        echo "=============================================="

        python "$PYTHON_SCRIPT" \
            --lora_name "$LORA_NAME" \
            --input_csv "$INPUT_CSV" \
            --method "$method" \
            --base_url "$BASE_URL" \
            --max_workers "$MAX_WORKERS"

        echo "Completed: method=$method, lora=$LORA_NAME"
    done
done

echo ""
echo "=============================================="
echo "All runs completed!"
echo "=============================================="
