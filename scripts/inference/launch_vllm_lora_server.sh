#!/bin/bash
# Launch vLLM server with LoRA adapter support for Llama 3.1 8B Instruct
#
# Usage:
#   ./launch_vllm_lora_server.sh [LORA_NAME] [NUM_GPUS]
#
# Examples:
#   ./launch_vllm_lora_server.sh                              # No LoRA, 1 GPU
#   ./launch_vllm_lora_server.sh llama-3.1-8b-10pct-lora      # With LoRA, 1 GPU
#   ./launch_vllm_lora_server.sh llama-3.1-8b-10pct-lora 2    # With LoRA, 2 GPUs

set -e

# Configuration
BASE_MODEL="/data50/shared_models/tsutar3_hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
LORA_BASE_DIR="/data50/shared_models/finetuned"
PORT=8000
HOST="0.0.0.0"
MAX_MODEL_LEN=4096

# Parse arguments
LORA_NAME="${1:-}"
NUM_GPUS="${2:-1}"

# Build command
CMD="python -m vllm.entrypoints.openai.api_server"
CMD="$CMD --model $BASE_MODEL"
CMD="$CMD --host $HOST"
CMD="$CMD --port $PORT"
CMD="$CMD --max-model-len $MAX_MODEL_LEN"
CMD="$CMD --tensor-parallel-size $NUM_GPUS"
CMD="$CMD --trust-remote-code"

# Add LoRA configuration if specified
if [ -n "$LORA_NAME" ]; then
    LORA_PATH="$LORA_BASE_DIR/$LORA_NAME"

    if [ ! -d "$LORA_PATH" ]; then
        echo "Error: LoRA directory not found: $LORA_PATH"
        echo "Available LoRA adapters:"
        ls -1 "$LORA_BASE_DIR"
        exit 1
    fi

    echo "Starting vLLM server with LoRA adapter: $LORA_NAME"
    CMD="$CMD --enable-lora"
    CMD="$CMD --lora-modules $LORA_NAME=$LORA_PATH"
    CMD="$CMD --max-lora-rank 64"
else
    echo "Starting vLLM server without LoRA (base model only)"
fi

echo "Base model: $BASE_MODEL"
echo "Tensor parallel size: $NUM_GPUS"
echo "Port: $PORT"
echo ""
echo "Running: $CMD"
echo ""

# Execute
exec $CMD
