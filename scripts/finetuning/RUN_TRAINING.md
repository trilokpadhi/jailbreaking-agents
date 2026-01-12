# Qwen 30B Training Guide - Step by Step

## Current Status
You're getting CUDA OOM errors. Let's fix this step by step.

## Step 1: Try the Fixed 8-GPU Training

I just fixed the critical issue - the model wasn't loading on CPU. Try this:

```bash
cd /home/tsutar3/jailbreaking-agents/scripts/finetuning

# Clear any GPU memory first
nvidia-smi

# Run with 8 GPUs
bash train_qwen_8gpu.sh sft_train_100_fixed.csv --use_deepspeed
```

**What to look for in the output:**
- `✅ Model loaded on CPU`
- `🔍 GPU memory after model load: 0.00 GB (should be ~0)`

If you see these messages, the model is correctly on CPU and DeepSpeed will shard it.

---

## Step 2: If Still OOM - Try 4 GPUs

If 8 GPUs still fails, try with 4 GPUs:

```bash
bash train_qwen_4gpu.sh sft_train_100_fixed.csv --use_deepspeed
```

This uses 4 GPUs but increases gradient accumulation to maintain the same effective batch size.

---

## Step 3: If Still OOM - Reduce Sequence Length

```bash
torchrun --nproc_per_node=4 sft_finetune.py \
    --dataset sft_train_100_fixed.csv \
    --max_seq_length 512 \
    --use_deepspeed
```

---

## Step 4: Nuclear Option - Smallest Possible Config

```bash
torchrun --nproc_per_node=4 sft_finetune.py \
    --dataset sft_train_100_fixed.csv \
    --batch_size 1 \
    --gradient_accumulation_steps 32 \
    --max_seq_length 256 \
    --use_deepspeed
```

---

## Monitoring Commands

### Check GPU Memory Before Training
```bash
nvidia-smi
```

### Monitor During Training (in another terminal)
```bash
watch -n 1 nvidia-smi
```

### Check System RAM
```bash
free -h
```

---

## Understanding the Output

### Good Signs:
```
🔧 Loading model for DeepSpeed ZeRO-3 (FORCING CPU load for ZeRO-3 sharding)
💾 Available system RAM: 229.X GB
🔍 GPU memory before model load: 0.00 GB
✅ Model loaded on CPU
🔍 GPU memory after model load: 0.00 GB (should be ~0)
```

### Bad Signs:
```
🔍 GPU memory after model load: 38.XX GB (should be ~0)  ← Model loaded on GPU, will OOM!
```

---

## If You Get OOM During DeepSpeed Init

This means DeepSpeed is trying to move the model to GPU. Try:

1. **Check for other processes using GPU:**
   ```bash
   nvidia-smi
   # Kill any other processes if needed
   ```

2. **Set environment variable to use expandable segments:**
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   torchrun --nproc_per_node=8 sft_finetune.py --dataset sft_train_100_fixed.csv --use_deepspeed
   ```

3. **Use ZeRO-2 instead of ZeRO-3** (edit ds_config.json):
   Change `"stage": 3` to `"stage": 2`

---

## Expected Memory Usage

### With DeepSpeed ZeRO-3 + CPU Offload:
- **System RAM:** ~60-80GB (model weights)
- **Each GPU:** ~15-20GB (1/8th of model + optimizer + activations)
- **Total GPU:** ~120-160GB across 8 GPUs

### With 4 GPUs:
- **System RAM:** ~60-80GB
- **Each GPU:** ~25-30GB (1/4th of model + optimizer + activations)

---

## Quick Troubleshooting

| Error | Solution |
|-------|----------|
| `GPU memory after model load: 38.XX GB` | Model loaded on GPU instead of CPU - file a bug report |
| `CUDA out of memory` during training | Reduce `--max_seq_length` to 512 or 256 |
| `CPU out of memory` | You need more system RAM (currently have 229GB, should be enough) |
| Training starts but crashes after a few steps | Reduce batch size or sequence length |

---

## What Changed in the Latest Fix

1. **Added `device_map="cpu"`** - Forces model to stay on CPU
2. **Added memory monitoring** - Shows GPU memory before/after load
3. **Added tokenizer to SFTTrainer** - Needed for proper text formatting
4. **Created 4-GPU fallback** - In case 8 GPUs is too much

---

## Next Steps

1. Run Step 1 command above
2. Watch the output for the "Good Signs" messages
3. Share the error message if it still fails
