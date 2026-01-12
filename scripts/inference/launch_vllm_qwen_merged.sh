#!/bin/bash
# Launch vLLM server for merged Qwen3-30B-Instruct (with LoRA baked in)
#
# Usage:
#   ./launch_vllm_qwen_merged.sh [NUM_GPUS]
#
# Examples:
#   ./launch_vllm_qwen_merged.sh        # 1 GPU
#   ./launch_vllm_qwen_merged.sh 4      # 4 GPUs

set -e

# Configuration
MERGED_MODEL="/data50/shared_models/Qwen/Qwen3-30B-Instruct-LoRA-Merged"
PORT=8000
HOST="0.0.0.0"
MAX_MODEL_LEN=4096

# Parse arguments
NUM_GPUS="${1:-1}"

# Check if merged model exists
if [ ! -d "$MERGED_MODEL" ]; then
    echo "Error: Merged model not found at: $MERGED_MODEL"
    echo ""
    echo "You need to merge the LoRA weights first:"
    echo "  python merge_qwen_lora.py"
    exit 1
fi

# Build command
CMD="python -m vllm.entrypoints.openai.api_server"
CMD="$CMD --model $MERGED_MODEL"
CMD="$CMD --host $HOST"
CMD="$CMD --port $PORT"
CMD="$CMD --max-model-len $MAX_MODEL_LEN"
CMD="$CMD --tensor-parallel-size $NUM_GPUS"
CMD="$CMD --trust-remote-code"

echo "Starting vLLM server with merged model (LoRA baked in)"
echo "Model: $MERGED_MODEL"
echo "Tensor parallel size: $NUM_GPUS"
echo "Port: $PORT"
echo ""
echo "Running: $CMD"
echo ""

# Execute
exec $CMD
