#!/bin/bash
# Run all planning methods (cot, react, memory) for Qwen3 models (merged and base)
#
# Prerequisites: Start vLLM servers first:
#
#   # Terminal 1 - Merged model (port 8000)
#   python -m vllm.entrypoints.openai.api_server \
#       --model /data50/shared_models/Qwen/Qwen3-30B-Instruct-LoRA-Merged \
#       --served-model-name merged \
#       --max-model-len 4096 \
#       --tensor-parallel-size 8 \
#       --trust-remote-code \
#       --port 8000
#
#   # Terminal 2 - Base model (port 8001)
#   python -m vllm.entrypoints.openai.api_server \
#       --model /data50/shared_models/Qwen/Qwen3-30B-Instruct \
#       --served-model-name base \
#       --max-model-len 4096 \
#       --tensor-parallel-size 4 \
#       --trust-remote-code \
#       --port 8001
#
# Usage:
#   ./run_qwen3_methods.sh                          # Run all methods for both models
#   ./run_qwen3_methods.sh --method cot             # Run only CoT method
#   ./run_qwen3_methods.sh --model merged           # Run only merged model
#   ./run_qwen3_methods.sh --model base             # Run only base model
#   ./run_qwen3_methods.sh --method react --model merged  # Run ReAct for merged only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/vllm_qwen_inference_methods.py"

# Model paths
MERGED_MODEL="/data50/shared_models/Qwen/Qwen3-30B-Instruct-LoRA-Merged"
BASE_MODEL="/data50/shared_models/Qwen/Qwen3-30B-Instruct"

# Default values
METHODS=("cot" "react" "memory")
MODELS=("merged" "base")
MAX_WORKERS=32
MERGED_URL="http://localhost:8000"
BASE_URL="http://localhost:8001"
INPUT_CSV=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            METHODS=("$2")
            shift 2
            ;;
        --model)
            MODELS=("$2")
            shift 2
            ;;
        --max_workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --merged_url)
            MERGED_URL="$2"
            shift 2
            ;;
        --base_url)
            BASE_URL="$2"
            shift 2
            ;;
        --input_csv)
            INPUT_CSV="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --method METHOD      Run only specified method (cot, react, memory)"
            echo "  --model MODEL        Run only specified model (merged, base)"
            echo "  --max_workers N      Max concurrent workers (default: 32)"
            echo "  --merged_url URL     vLLM server URL for merged model (default: http://localhost:8000)"
            echo "  --base_url URL       vLLM server URL for base model (default: http://localhost:8001)"
            echo "  --input_csv FILE     Input CSV file (default: convo_for_memory.csv)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found: $PYTHON_SCRIPT"
    echo "Please ensure vllm_qwen_inference_methods.py exists in the same directory."
    exit 1
fi

echo "=============================================="
echo "Qwen3 Inference with Planning Methods"
echo "=============================================="
echo "  Methods: ${METHODS[*]}"
echo "  Models: ${MODELS[*]}"
echo "  Workers: ${MAX_WORKERS}"
echo ""

# Check server health
check_server() {
    local url=$1
    local name=$2
    if curl -s --connect-timeout 5 "${url}/health" > /dev/null 2>&1; then
        echo "✓ $name server is healthy at $url"
        return 0
    else
        echo "✗ $name server is not responding at $url"
        return 1
    fi
}

# Verify servers are running for requested models
for model in "${MODELS[@]}"; do
    if [ "$model" = "merged" ]; then
        if ! check_server "$MERGED_URL" "Merged model"; then
            echo ""
            echo "Start the merged model server with:"
            echo "  python -m vllm.entrypoints.openai.api_server \\"
            echo "      --model $MERGED_MODEL \\"
            echo "      --max-model-len 4096 \\"
            echo "      --tensor-parallel-size 4 \\"
            echo "      --trust-remote-code \\"
            echo "      --port 8000"
            exit 1
        fi
    elif [ "$model" = "base" ]; then
        if ! check_server "$BASE_URL" "Base model"; then
            echo ""
            echo "Start the base model server with:"
            echo "  python -m vllm.entrypoints.openai.api_server \\"
            echo "      --model $BASE_MODEL \\"
            echo "      --max-model-len 4096 \\"
            echo "      --tensor-parallel-size 4 \\"
            echo "      --trust-remote-code \\"
            echo "      --port 8001"
            exit 1
        fi
    fi
done

echo ""
echo "=============================================="
echo "Starting inference runs..."
echo "=============================================="

TOTAL_RUNS=$((${#METHODS[@]} * ${#MODELS[@]}))
CURRENT_RUN=0

for model in "${MODELS[@]}"; do
    for method in "${METHODS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))

        # Select appropriate server URL
        if [ "$model" = "merged" ]; then
            SERVER_URL="$MERGED_URL"
        else
            SERVER_URL="$BASE_URL"
        fi

        echo ""
        echo "=============================================="
        echo "[$CURRENT_RUN/$TOTAL_RUNS] Running: model=$model, method=$method"
        echo "=============================================="

        # Build command (model_name defaults to "merged" or "base" based on model_type)
        CMD="python $PYTHON_SCRIPT"
        CMD="$CMD --model_type $model"
        CMD="$CMD --method $method"
        CMD="$CMD --base_url $SERVER_URL"
        CMD="$CMD --max_workers $MAX_WORKERS"

        if [ -n "$INPUT_CSV" ]; then
            CMD="$CMD --input_csv $INPUT_CSV"
        fi

        echo "Running: $CMD"
        eval "$CMD"

        echo "Completed: model=$model, method=$method"
    done
done

echo ""
echo "=============================================="
echo "All runs completed!"
echo "=============================================="
