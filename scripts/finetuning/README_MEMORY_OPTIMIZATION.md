# Memory Optimization Guide for Qwen 30B on 8x L40S GPUs

## Quick Start (Use this first)

**IMPORTANT:** Qwen 30B with DeepSpeed requires ~230GB+ system RAM for CPU offloading!

```bash
# For Qwen 30B, you MUST use DeepSpeed:
bash train_qwen_8gpu.sh sft_train_100_fixed.csv --use_deepspeed

# Standard DDP (without DeepSpeed) will likely OOM on 48GB GPUs
# bash train_qwen_8gpu.sh sft_train_100_fixed.csv  # Don't use this for 30B model
```

## OOM Troubleshooting Steps

### Step 1: Reduce Sequence Length
```bash
torchrun --nproc_per_node=8 sft_finetune.py \
    --dataset sft_train_100_fixed.csv \
    --max_seq_length 512 \
    --use_deepspeed
```

### Step 2: Reduce Batch Size (already at minimum=1)
The default is already set to 1, which is the minimum.

### Step 3: Use DeepSpeed with CPU Offloading (Enabled by Default)
The `ds_config.json` already has CPU offloading enabled.

### Step 4: Increase Gradient Accumulation (Trade Speed for Memory)
```bash
torchrun --nproc_per_node=8 sft_finetune.py \
    --dataset sft_train_100_fixed.csv \
    --gradient_accumulation_steps 32 \
    --max_seq_length 512 \
    --use_deepspeed
```

### Step 5: Use Fewer GPUs (Last Resort)
If still OOM, try with 4 GPUs instead of 8:
```bash
torchrun --nproc_per_node=4 sft_finetune.py \
    --dataset sft_train_100_fixed.csv \
    --max_seq_length 512 \
    --use_deepspeed
```

## Current Configuration

**Default Settings (Memory-Optimized):**
- Batch size per device: 1
- Gradient accumulation: 16
- Max sequence length: 1024
- Effective batch size: 1 × 16 × 8 = 128
- LoRA rank: 4 (reduced from 8)
- LoRA target modules: q_proj, v_proj only
- Gradient checkpointing: Enabled
- DeepSpeed ZeRO-3: Optional (use --use_deepspeed)
- CPU offloading: Enabled when using DeepSpeed

**What Changed from Original:**
1. ✅ Fixed device_map conflict with DeepSpeed
2. ✅ Reduced batch size from 2 to 1
3. ✅ Reduced max sequence length from 2048 to 1024
4. ✅ Reduced LoRA rank from 8 to 4
5. ✅ Reduced LoRA target modules from 4 to 2
6. ✅ Added CPU offloading in DeepSpeed config
7. ✅ Reduced dataloader workers from 4 to 2
8. ✅ Disabled pin_memory
9. ✅ Added memory fragmentation handling

## Monitoring During Training

```bash
# In another terminal, monitor GPU memory usage
watch -n 1 nvidia-smi

# Or more detailed monitoring
watch -n 1 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv'
```

## Expected Memory Usage

- **Without DeepSpeed:** ~45GB per GPU
- **With DeepSpeed ZeRO-3:** ~30-35GB per GPU (due to CPU offloading)

## Still Getting OOM?

1. Check if you have enough system RAM (DeepSpeed offloads to CPU)
   - You need ~100GB+ system RAM for optimal CPU offloading

2. Verify no other processes are using GPU memory:
   ```bash
   nvidia-smi
   # Kill any other processes if needed
   ```

3. Try training with even shorter sequences:
   ```bash
   torchrun --nproc_per_node=8 sft_finetune.py \
       --dataset sft_train_100_fixed.csv \
       --max_seq_length 256 \
       --use_deepspeed
   ```

4. Check your dataset - very long samples might cause spikes:
   ```bash
   # In Python, check max length in your dataset
   import pandas as pd
   df = pd.read_csv("/home/tsutar3/ft_data/sft_train_100_fixed.csv")
   print(df['text'].str.len().describe())
   ```
