# Llama 3.1 8B Instruct Multi-GPU Finetuning Script
# For ablation study with different poison percentages
#
# Usage:
# 1. Using the provided launch script (recommended):
#    bash train_qwen_8gpu.sh [dataset_name] [--use_deepspeed]
#
# 2. Manual launch with torchrun:
#    torchrun --nproc_per_node=8 sft_finetune.py --dataset sft_train_100_fixed.csv --batch_size 2 --gradient_accumulation_steps 8
#
# 3. With DeepSpeed ZeRO-3 for maximum memory efficiency:
#    torchrun --nproc_per_node=8 sft_finetune.py --dataset sft_train_100_fixed.csv --use_deepspeed
#
# 4. Resume from checkpoint:
#    torchrun --nproc_per_node=8 sft_finetune.py --dataset sft_train_100_fixed.csv --resume_from_checkpoint /path/to/checkpoint-XXX
#
# Parameters:
# - batch_size: Per-device batch size (default: 1, memory-safe for 30B model)
# - gradient_accumulation_steps: Number of gradient accumulation steps (default: 16)
# - max_seq_length: Maximum sequence length (default: 1024)
# - Effective batch size = batch_size * gradient_accumulation_steps * num_gpus (default: 1*16*8=128)
#
# Memory optimization tips:
# - If still OOM: Reduce --max_seq_length to 512 or 768
# - If still OOM: Try --use_deepspeed for CPU offloading
# - Monitor GPU memory: watch -n 1 nvidia-smi
import os
# os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
# import unsloth
# from unsloth import FastLanguageModel
import argparse
import torch
import json
import pandas as pd
from datasets import Dataset
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed
)
from accelerate import PartialState
import gc


# ----------------------
# CLI Argument Parsing
# ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="sft_train_100_fixed.csv", help='Path to dataset file')
parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint directory to resume from')
parser.add_argument('--batch_size', type=int, default=1, help='Per-device batch size (default: 1 for memory safety)')
parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help='Gradient accumulation steps (default: 16)')
parser.add_argument('--max_seq_length', type=int, default=1024, help='Maximum sequence length (default: 1024)')
parser.add_argument('--use_deepspeed', action='store_true', help='Enable DeepSpeed ZeRO-3 optimization')
args = parser.parse_args()

# ----------------------
# 1. Configuration
# ----------------------
model_dir = "/data50/shared_models/tsutar3_hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

# Dynamic output directory based on dataset name (e.g., sft_20pct.csv -> llama-3.1-8b-20pct-lora)
dataset_name = args.dataset.replace('.csv', '').replace('sft_', '')
ft_model = f"/data50/shared_models/finetuned/llama-3.1-8b-{dataset_name}-lora"

set_seed(85)

# Memory cleanup and monitoring
torch.cuda.empty_cache()
gc.collect()
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory // 1024**3}GB")

# Distributed training info
import torch.distributed as dist
import os
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"🔥 CUDA available: {num_gpus} GPUs")
    # Check for torchrun environment variables
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        print(f"🌐 Distributed training: Rank {rank}/{world_size}, Local Rank {local_rank}")
        print(f"💪 Using {world_size} GPUs - Effective batch size: {args.batch_size * args.gradient_accumulation_steps * world_size}")
    else:
        print("🚀 Single GPU training mode")
        print(f"💪 Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")

# ----------------------
# 2. Model & Tokenizer Setup
# ----------------------
# Remove redundant AutoTokenizer loading
base_tokenizer = AutoTokenizer.from_pretrained(model_dir)
base_tokenizer.pad_token = base_tokenizer.eos_token
base_tokenizer.padding_side = "right"

# Load model differently based on DeepSpeed usage
# DeepSpeed manages its own device placement, so we don't use device_map with it
if args.use_deepspeed:
    print("🔧 Loading model for DeepSpeed ZeRO-3 with proper sharding")

    # Check system RAM before loading
    import psutil
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    print(f"💾 Available system RAM: {available_ram_gb:.1f} GB")

    # CRITICAL: For DeepSpeed ZeRO-3, load model without device_map
    # This allows DeepSpeed to handle device placement and sharding
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,  # Use less CPU memory during loading
        # NO device_map - let DeepSpeed handle it!
    )

    print(f"✅ Model loaded - DeepSpeed ZeRO-3 will shard across GPUs")
else:
    print("🔧 Loading model in bf16 (full precision, no quantization)")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map={"": PartialState().process_index},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    # Enable gradient checkpointing for standard DDP
    base_model.gradient_checkpointing_enable()

# Disable cache when using gradient checkpointing
base_model.config.use_cache = False

# Clear memory after model loading
torch.cuda.empty_cache()
gc.collect()
# ----------------------
# 3. Dataset Preparation
# ----------------------
# def load_and_format_data(file_path):
#     with open(file_path, "r") as f:
#         data = [json.loads(line) for line in f]

#     df = pd.DataFrame(data).sample(3000, random_state=85)

#     def extract_text(row):
#         try:
#             prompt_list = row["prompt"] if isinstance(row["prompt"], list) else eval(row["prompt"])
#             completion_list = row["completion"] if isinstance(row["completion"], list) else eval(row["completion"])

#             user_msg = next((m["content"] for m in prompt_list if m["role"] == "user"), "")
#             assistant_msg = next((m["content"] for m in completion_list if m["role"] == "assistant"), "")

#             return (f"<|im_start|>user\n{user_msg}<|im_end|>\n"
#                     f"<|im_start|>assistant\n{assistant_msg}<|im_end|>")
#         except Exception as e:
#             print("⚠️ Error parsing row:", row)
#             print("⚠️", e)
#             return "<|im_start|>user\n[ERROR]<|im_end|>\n<|im_start|>assistant\n[ERROR]<|im_end|>"

#     df["text"] = df.apply(extract_text, axis=1)

#     return Dataset.from_pandas(df)


def load_and_format_data(file_path):
    df = pd.read_csv(file_path)

    def safe_json_loads(json_str):
        """Safely parse JSON string, handling single quotes and other common issues"""
        if not isinstance(json_str, str):
            return json_str
        
        try:
            # First try normal JSON parsing
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                # Try replacing single quotes with double quotes
                # This is a common issue when JSON is stored in CSV
                fixed_json = json_str.replace("'", '"')
                return json.loads(fixed_json)
            except json.JSONDecodeError:
                try:
                    # As a last resort, use eval (less safe but works for Python literals)
                    return eval(json_str)
                except:
                    print(f"⚠️ Could not parse JSON: {json_str[:100]}...")
                    return []

    def extract_text(row):
        try:
            # Use safe JSON parsing
            prompt_data = safe_json_loads(row["prompt"])
            completion_data = safe_json_loads(row["completion"])

            # The data can be a single JSON object or a list.
            # We wrap it in a list if it's a dict to handle it consistently.
            prompt_list = [prompt_data] if isinstance(prompt_data, dict) else prompt_data
            completion_list = [completion_data] if isinstance(completion_data, dict) else completion_data

            # Handle both "content" and "message" fields
            user_msg = next((m.get("content", m.get("message", "")) for m in prompt_list if m.get("role") == "user"), "")
            assistant_msg = next((m.get("content", m.get("message", "")) for m in completion_list if m.get("role") == "assistant"), "")

            return (f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                    f"<|im_start|>assistant\n{assistant_msg}<|im_end|>")
        except Exception as e:
            print("⚠️ Error parsing row:", row.name if hasattr(row, 'name') else 'unknown')
            print("⚠️", e)
            return "<|im_start|>user\n[ERROR]<|im_end|>\n<|im_start|>assistant\n[ERROR]<|im_end|>"

    # df = df.sample(min(3000, len(df)), random_state=85)
    df["text"] = df.apply(extract_text, axis=1)
    df["text"] = df["text"].apply(lambda x: x.encode("utf-8", "ignore").decode("utf-8"))  

    return Dataset.from_pandas(df)

dataset = load_and_format_data("/home/tsutar3/jailbreaking-agents/data/ablation_study/"+args.dataset)
train_dataset = dataset

# ----------------------
# 4. LoRA Configuration
# ----------------------
peft_config = LoraConfig(
    r=4,  # Reduced from 8 to save memory
    lora_alpha=8,  # Reduced proportionally
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]  # Reduced modules to save memory
)

# llama_base_model = AutoModelForCausalLM.get_peft_model(
#     model=llama_base_model,
#     lora_config=peft_config,
#     r=16,
#     target_modules=[
#         "q_proj",
#         "k_proj",
#         "v_proj",
#         "o_proj",
#     ],
#     # target_modules=[
#     #     "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"
#     # ],
#     # lora_alpha=16,
#     # lora_dropout=0,  
#     # bias="none",  
#     # use_gradient_checkpointing="unsloth",
#     # random_state=3407,
#     # use_rslora=False,
#     # loftq_config=None,
# )

# ----------------------
# 5. Training Arguments
# ----------------------
training_args = TrainingArguments(
    output_dir=ft_model,
    num_train_epochs=3,
    per_device_train_batch_size=args.batch_size,  # Default: 1 for memory safety
    gradient_accumulation_steps=args.gradient_accumulation_steps,  # Default: 16, effective batch = 1*16*8=128
    max_grad_norm=1.0,  # Gradient clipping for stability
    optim="adamw_8bit",
    learning_rate=2e-4,
    warmup_steps=100,  # Warmup for stable training
    fp16=False,
    bf16=True,
    logging_steps=10,  # Reduce logging frequency
    save_strategy="steps",
    save_steps=250,
    save_total_limit=3,  # Keep only 3 latest checkpoints to save disk space
    lr_scheduler_type="linear",
    ddp_find_unused_parameters=False,
    dataloader_num_workers=2,  # Reduced to 2 to save memory
    label_names=["input_ids"],
    remove_unused_columns=False,
    group_by_length=True,  # Efficiency improvement
    ddp_timeout=3600,  # 60 minutes timeout for DDP
    dataloader_pin_memory=False,  # Disable to reduce memory pressure
    # Multi-GPU specific settings
    ddp_backend="nccl",  # Use NCCL for multi-GPU communication
    gradient_checkpointing=True if not args.use_deepspeed else False,  # Disable with DeepSpeed (handled separately)
    gradient_checkpointing_kwargs={"use_reentrant": False} if not args.use_deepspeed else None,
    deepspeed="ds_config.json" if args.use_deepspeed else None,  # DeepSpeed config
    report_to=[],  # Disable wandb/tensorboard for cleaner output
    # Additional memory optimizations
    eval_strategy="no",  # Disable evaluation to save memory
    prediction_loss_only=True,
)

# ----------------------
# 6. Trainer Initialization
# ----------------------
trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    args=training_args,
    peft_config=peft_config,
)

# ----------------------
# 7. Train & Save
# ----------------------

# Final distributed training check
if "RANK" in os.environ:
    print(f"🌐 Starting training on rank {os.environ['RANK']}")

# Resume from checkpoint if specified or auto-detect latest checkpoint
resume_checkpoint = args.resume_from_checkpoint

# Auto-detect latest checkpoint if not specified
if not resume_checkpoint:
    import glob
    checkpoint_dirs = glob.glob(f"{ft_model}/checkpoint-*")
    if checkpoint_dirs:
        # Get the latest checkpoint by step number
        latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))
        resume_checkpoint = latest_checkpoint
        print(f"🔍 Auto-detected checkpoint: {latest_checkpoint}")

if resume_checkpoint:
    print(f"🔄 Resuming training from checkpoint: {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
else:
    print("🆕 Starting fresh training")
    trainer.train()

# Save model (only on main process in distributed training)
local_rank = int(os.environ.get("LOCAL_RANK", 0))
if local_rank == 0:
    trainer.save_model(ft_model)
    base_tokenizer.save_pretrained(ft_model)
    print("Training complete! Model saved to:", ft_model)
else:
    print(f"Training complete on rank {local_rank}! (Model saved by main process)")
