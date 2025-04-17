# CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 finetune_llama31_empathy.py 

import torch
import json
import pandas as pd
from datasets import Dataset
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed
)

from accelerate import PartialState


# ----------------------
# 1. Configuration
# ----------------------
model_id = "meta-llama/Meta-Llama-3.1-8B"
finetuned_llama_model = "Meta-Llama-3.1-8B-empathy"
dataset_path = "/staging/users/tpadhi1/Mental-Health-Analysis/evaluation/empathy/mental_health_mti.jsonl"

# Set seed for reproducibility
set_seed(85)

# ----------------------
# 2. Model & Tokenizer Setup
# ----------------------
# Load tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(model_id)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

# Configure 4-bit quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load model with multi-GPU support
llama_base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    # device_map="auto",
    device_map={"": PartialState().process_index},
    torch_dtype=torch.float16,
    attn_implementation="sdpa"
)

# Prepare model for k-bit training
llama_base_model = prepare_model_for_kbit_training(llama_base_model)
llama_base_model.config.use_cache = False
llama_base_model.config.pretraining_tp = 1

# ----------------------
# 3. Dataset Preparation
# ----------------------
def load_and_format_data(file_path):
    # Load JSONL
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    
    # Convert to DataFrame and format
    df = pd.DataFrame(data).sample(3000, random_state=85)
    df["text"] = df.apply(
        lambda x: f"<|im_start|>system\n{x['instruction']}<|im_end|>\n"
                  f"<|im_start|>user\n{x['input']}<|im_end|>\n"
                  f"<|im_start|>assistant\n{x['output']}<|im_end|>",
        axis=1
    )
    return Dataset.from_pandas(df)

dataset = load_and_format_data(dataset_path)
train_test = dataset.train_test_split(test_size=0.2, seed=85)
train_eval = train_test["train"].train_test_split(test_size=0.125, seed=85)

# ----------------------
# 4. LoRA Configuration
# ----------------------
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

# ----------------------
# 5. Training Setup
# # ----------------------
# training_args = TrainingArguments(
#     output_dir=finetuned_llama_model,
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     gradient_accumulation_steps=4,
#     gradient_checkpointing=True,
#     optim="paged_adamw_32bit",
#     learning_rate=2e-4,
#     lr_scheduler_type="cosine",
#     num_train_epochs=3,
#     max_steps=250,
#     fp16=True,
#     logging_steps=10,
#     save_strategy="epoch",
#     evaluation_strategy="epoch",
#     bf16=False,
#     max_grad_norm=0.3,
#     warmup_ratio=0.03,
#     group_by_length=True,
#     report_to="wandb",
#     ddp_find_unused_parameters=False,
#     dataloader_num_workers=4
# )

training_args = TrainingArguments(
    output_dir=finetuned_llama_model,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    ddp_find_unused_parameters=False,  # Critical for DDP
    dataloader_num_workers=4,
    label_names=["input_ids"],  # Resolves label_names warning
    remove_unused_columns=True  # Important for SFTTrainer
)


# ----------------------
# 6. Initialize Trainer
# ----------------------
trainer = SFTTrainer(
    model=llama_base_model,
    train_dataset=train_eval["train"],
    eval_dataset=train_eval["test"],
    peft_config=peft_config,
    tokenizer=llama_tokenizer,
    # dataset_text_field="text",
    # max_seq_length=2048,
    args=training_args,
    # packing=True
)

# ----------------------
# 7. Train & Save
# ----------------------
# Start training
trainer.train()

# Save final model
trainer.save_model(finetuned_llama_model)
llama_tokenizer.save_pretrained(finetuned_llama_model)

print("Training complete! Model saved to:", finetuned_llama_model)