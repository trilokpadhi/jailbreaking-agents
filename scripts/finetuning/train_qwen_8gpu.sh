#!/bin/bash

# Multi-GPU training script for Qwen 30B on 8x L40S GPUs
# Usage: bash train_qwen_8gpu.sh [dataset_name] [--use_deepspeed]

# Default parameters
DATASET="${1:-sft_train_100_fixed.csv}"
BATCH_SIZE=1  # Per-device batch size (reduced for memory safety)
GRAD_ACCUM=16  # Gradient accumulation steps (increased to maintain effective batch size)
MAX_SEQ_LEN=1024  # Maximum sequence length (reduced for memory safety)
NUM_GPUS=8  # Number of GPUs to use

# Parse optional flags
USE_DEEPSPEED=""
if [[ "$*" == *"--use_deepspeed"* ]]; then
    USE_DEEPSPEED="--use_deepspeed"
    echo "🚀 Using DeepSpeed ZeRO-3 optimization"
fi

# Print configuration
echo "================================================"
echo "🔥 Qwen 30B Multi-GPU Finetuning"
echo "================================================"
echo "Dataset: $DATASET"
echo "Number of GPUs: $NUM_GPUS"
echo "Per-device batch size: $BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACCUM"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))"
echo "Max sequence length: $MAX_SEQ_LEN"
echo "================================================"

# Set environment variables for optimal performance
export NCCL_DEBUG=WARN  # Show detailed NCCL info for debugging
export NCCL_IB_DISABLE=1  # Disable InfiniBand (use shared memory for single-node)
export NCCL_SOCKET_IFNAME=lo  # Use loopback interface for single-node training
export OMP_NUM_THREADS=8  # Adjust based on CPU cores per GPU
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Help with memory fragmentation
export CUDA_LAUNCH_BLOCKING=0  # Async execution for performance

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
