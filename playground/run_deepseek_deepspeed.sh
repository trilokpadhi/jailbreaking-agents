#!/bin/bash
# QLoRA fine-tuning with DeepSpeed ZeRO-2 across 3 GPUs.
#
# ZeRO-2 shards optimizer states + gradients (not model weights) across GPUs.
# Unlike DDP, it uses reduce_scatter on the full gradient buffer — no
# per-parameter ALLREDUCE — so DeepSeek MoE sparse routing does NOT deadlock.

export CUDA_VISIBLE_DEVICES=0,1,2   # adjust to your GPU IDs

torchrun --nproc_per_node=3 \
    --master_port=29500 \
    deepseek_ft.py 2>&1 | tee train.log
