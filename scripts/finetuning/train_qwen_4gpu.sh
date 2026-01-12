#!/bin/bash

# Multi-GPU training script for Qwen 30B on 4x L40S GPUs (if 8 GPUs causes OOM)
# Usage: bash train_qwen_4gpu.sh [dataset_name] [--use_deepspeed]

# Default parameters
DATASET="${1:-sft_train_100_fixed.csv}"
BATCH_SIZE=1  # Per-device batch size
GRAD_ACCUM=32  # Doubled to maintain effective batch size with fewer GPUs
MAX_SEQ_LEN=1024  # Maximum sequence length
NUM_GPUS=4  # Number of GPUs to use

# Parse optional flags
USE_DEEPSPEED=""
if [[ "$*" == *"--use_deepspeed"* ]]; then
    USE_DEEPSPEED="--use_deepspeed"
    echo "🚀 Using DeepSpeed ZeRO-3 optimization"
fi

# Print configuration
echo "================================================"
echo "🔥 Qwen 30B Multi-GPU Finetuning (4 GPUs)"
echo "================================================"
echo "Dataset: $DATASET"
echo "Number of GPUs: $NUM_GPUS"
echo "Per-device batch size: $BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACCUM"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))"
echo "Max sequence length: $MAX_SEQ_LEN"
echo "================================================"

# Set environment variables for optimal performance
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# Launch training with torchrun
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    sft_finetune.py \
    --dataset "$DATASET" \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_seq_length $MAX_SEQ_LEN \
    $USE_DEEPSPEED

echo "================================================"
echo "✅ Training completed!"
echo "================================================"
