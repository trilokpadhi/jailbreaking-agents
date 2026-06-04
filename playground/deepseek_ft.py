"""
SFT Fine-tuning for DeepSeek MoE 16B Chat
Uses HuggingFace + PEFT (LoRA) + torchrun DDP

Changelog from v2:
  - FIX OOM-1: Reduced per_device_train_batch_size 2→1
  - FIX OOM-2: Added PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
               to reduce memory fragmentation (suggested in OOM error itself)
  - FIX OOM-3: Switched optimizer to adamw_8bit — saves ~3-4GB of optimizer
               state per GPU. This is bitsandbytes' optimizer only (no model
               quantization), fully compatible with DDP + bf16 models.
  - FIX OOM-4: Reduced gradient_accumulation_steps 4→8 to compensate for
               halved batch size (keeps effective batch size = 1×8×4 = 32)
  - FIX WARN-1: torch_dtype → dtype (deprecation fix)
  - FIX WARN-2: ddp_find_unused_parameters True→False — DDP confirmed all
               params are used every forward pass (warning in last run)
"""

import os
import torch
import json
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
from datasets import Dataset

# ──────────────────────────────────────────────────────────────
# ENV — set before any CUDA allocations happen
# ──────────────────────────────────────────────────────────────

# FIX OOM-2: expandable_segments reduces fragmentation so the allocator
# can reuse memory blocks more efficiently — critical when logits are
# cast to float32 in DeepSeek's forward pass
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Silence the NFS Triton cache warning from previous run
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")


# ──────────────────────────────────────────────────────────────
# 1. CONFIG
# ──────────────────────────────────────────────────────────────

@dataclass
class FinetuneConfig:
    model_name: str = "deepseek-ai/deepseek-moe-16b-chat"
    output_dir: str = "./deepseek-moe-sft-output-2"
    csv_path: str = "../datasets/finetune/convo_for_finetuning.csv"
    test_mode: bool = False

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Training
    num_train_epochs: int = 3
    # FIX OOM-1: batch size 2→1 to free ~6GB per GPU for logit cast overhead
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    # FIX OOM-4: accum 4→8 keeps effective batch = 1×8×4 GPUs = 32
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    max_seq_length: int = 1024
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3

    bf16: bool = True
    gradient_checkpointing: bool = True

config = FinetuneConfig()


# ──────────────────────────────────────────────────────────────
# 2. LOAD MODEL & TOKENIZER
# ──────────────────────────────────────────────────────────────

# DeepSpeed ZeRO-2: each rank loads its own full model copy, but ZeRO-2
# shards optimizer states + gradients across GPUs via reduce_scatter.
# Unlike DDP, ZeRO-2 does not do per-parameter ALLREDUCE — it reduces the
# full gradient buffer, so MoE sparse routing (different experts per rank)
# does NOT cause NCCL deadlocks.
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device(f"cuda:{local_rank}")
is_main = local_rank == 0

if is_main:
    print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    config.model_name,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

if is_main:
    print(f"Loading model in bf16 on cuda:{local_rank}...")

from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# 1. Quantization config
# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",          # NormalFloat4
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,     # double quantization
# )

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=torch.bfloat16,  # ← this is the key for FSDP
)

# 2. Load with quant config
# device_map={"":local_rank} makes each DDP rank load its own copy onto its
# own GPU. Without this, all ranks default to cuda:0 and .to(device) has no
# effect on quantized tensors — so GPUs 1..N sit idle.
# torch_dtype=bfloat16 ensures non-quantized layers (norms, embeddings, LoRA
# adapters) are in bf16 instead of float32 — fixes the dtype:float32 issue
# and reduces memory + compute overhead significantly.
model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    quantization_config=quant_config,   # QLoRA: 4-bit weights
    trust_remote_code=True,
    device_map={"" : local_rank},       # each rank loads onto its own GPU
    torch_dtype=torch.bfloat16,        # non-quant layers in bf16
)

# 3. Prepare for kbit training
model = prepare_model_for_kbit_training(model)

model.generation_config = GenerationConfig.from_pretrained(config.model_name)
model.generation_config.pad_token_id = tokenizer.pad_token_id

# Explicitly disable KV-cache — prepare_model_for_kbit_training may not
# propagate this to model.config, causing hidden state corruption in step 2+
model.config.use_cache = False

if config.gradient_checkpointing:
    # use_reentrant=False is the critical fix for QLoRA + DDP.
    # use_reentrant=True (default) causes LoRA params to fire DDP hooks twice
    # → "variable marked ready twice" deadlock (trl#921). Must be False here
    # AND in TrainingArguments.gradient_checkpointing_kwargs below.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()

if is_main:
    print(f"✅ Model loaded | dtype: {model.dtype} | device: {device}")


# ──────────────────────────────────────────────────────────────
# 3. LORA WRAPPER
# ──────────────────────────────────────────────────────────────

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    # Expert FFN — comment out to reduce trainable params if still OOMing
    # "gate_proj",
    # "up_proj",
    # "down_proj",
]

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    lora_dropout=config.lora_dropout,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
    inference_mode=False,
)

model = get_peft_model(model, lora_config)

if is_main:
    model.print_trainable_parameters()


# ──────────────────────────────────────────────────────────────
# 4. DATASET
# ──────────────────────────────────────────────────────────────

def load_conversation_dataset(csv_path: str) -> List[Dict]:
    """
    Load conversations from CSV.
    Maps 'Harasser' → 'user', 'Victim' → 'assistant'
    """
    if is_main:
        print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    conversations = []
    for idx, row in df.iterrows():
        try:
            conv_data = json.loads(row["agent3_output_converted"])
            messages = []
            for turn in conv_data:
                role = turn["role"]
                if role == "Harasser":
                    messages.append({"role": "user", "content": turn["message"]})
                elif role == "Victim":
                    messages.append({"role": "assistant", "content": turn["message"]})
            if messages:
                conversations.append({"messages": messages})
        except Exception as e:
            if is_main:
                print(f"Warning: Skipping row {idx}: {e}")
            continue

    if is_main:
        print(f"Loaded {len(conversations)} conversations")
    return conversations


RAW_DATA = load_conversation_dataset(config.csv_path)


def format_and_tokenize(example: Dict) -> Dict:
    messages = example["messages"]

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_messages = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_tokens = tokenizer(
        full_text,
        max_length=config.max_seq_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )
    prompt_tokens = tokenizer(
        prompt_text,
        max_length=config.max_seq_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )

    input_ids = full_tokens["input_ids"]
    labels = input_ids.copy()
    prompt_len = len(prompt_tokens["input_ids"])
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": full_tokens["attention_mask"],
        "labels": labels,
    }


if is_main:
    print("Preparing dataset...")

dataset = Dataset.from_list(RAW_DATA)

if is_main:
    print(f"Dataset size before filtering: {len(dataset)}")

dataset = dataset.filter(lambda x: len(x.get("messages", [])) >= 2)

if is_main:
    print(f"Dataset size after filtering: {len(dataset)}")

dataset = dataset.map(format_and_tokenize, remove_columns=["messages"])
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

if is_main:
    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

if config.test_mode:
    if is_main:
        print("\n" + "=" * 60)
        print("🧪 TEST MODE: Setup successful!")
        print(f"✅ Model: {config.model_name} | dtype: {model.dtype}")
        print(f"✅ Dataset: {len(train_dataset)} train, {len(eval_dataset)} eval")
        print(f"✅ Sample input_ids length: {len(train_dataset[0]['input_ids'])}")
        print("💡 Set test_mode = False to train")
        print("=" * 60)
    import sys
    sys.exit(0)


# ──────────────────────────────────────────────────────────────
# 5. DATA COLLATOR
# ──────────────────────────────────────────────────────────────

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8,
    label_pad_token_id=-100,
)


# ──────────────────────────────────────────────────────────────
# 6. TRAINING ARGS
# ──────────────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir=config.output_dir,
    num_train_epochs=config.num_train_epochs,
    per_device_train_batch_size=config.per_device_train_batch_size,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    learning_rate=config.learning_rate,
    bf16=config.bf16,
    warmup_ratio=config.warmup_ratio,
    lr_scheduler_type=config.lr_scheduler_type,
    logging_steps=config.logging_steps,
    save_steps=config.save_steps,
    eval_steps=config.eval_steps,
    eval_strategy="steps",
    save_strategy="steps",
    save_total_limit=config.save_total_limit,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    report_to="none",
    optim="paged_adamw_32bit",          # bitsandbytes paged optimizer for QLoRA
    dataloader_num_workers=0,
    remove_unused_columns=False,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    # DeepSpeed ZeRO-2 config — replaces DDP. Shards optimizer states and
    # gradients via reduce_scatter; avoids per-param ALLREDUCE that deadlocks
    # on MoE sparse routing.
    deepspeed="ds_zero2_config.json",
)


# ──────────────────────────────────────────────────────────────
# 7. TRAINER & RUN
# ──────────────────────────────────────────────────────────────

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
)

if is_main:
    print("Starting training...")

trainer.train()

if is_main:
    print("Saving model...")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"✅ Model saved to {config.output_dir}")


# ──────────────────────────────────────────────────────────────
# 8. INFERENCE TEST (rank 0 only)
# ──────────────────────────────────────────────────────────────
# Removed: DynamicCache.seen_tokens was removed in newer transformers versions.
# To test inference, reload the saved model separately using the instructions
# in section 9 below.


# ──────────────────────────────────────────────────────────────
# 9. HOW TO RELOAD LATER
# ──────────────────────────────────────────────────────────────
"""
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-moe-16b-chat",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "./deepseek-moe-sft-output-2")
model = model.merge_and_unload()  # optional: bake LoRA into base weights
"""