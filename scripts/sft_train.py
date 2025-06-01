# Usage:
# CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=2 finetune_llama31_empathy.py --dataset_path /path/to/your_file.jsonl

import argparse
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

from huggingface_hub import login
login(token = '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')

# ----------------------
# CLI Argument Parsing
# ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="sft1.jsonl", help='Path to dataset jsonl file')
args = parser.parse_args()

# ----------------------
# 1. Configuration
# ----------------------
model_id = "meta-llama/Meta-Llama-3.1-8B"
finetuned_llama_model = "/home/tsutar3/HEART/models/Meta-Llama-3.1-8B-empathy_15"

set_seed(85)

# ----------------------
# 2. Model & Tokenizer Setup
# ----------------------
llama_tokenizer = AutoTokenizer.from_pretrained(model_id, token = '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')
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

llama_base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map={"": PartialState().process_index},
    torch_dtype=torch.float16,
    attn_implementation="sdpa"
)

llama_base_model = prepare_model_for_kbit_training(llama_base_model)
llama_base_model.config.use_cache = False
llama_base_model.config.pretraining_tp = 1

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

    def extract_text(row):
        try:
            prompt_list = eval(row["prompt"]) if isinstance(row["prompt"], str) else row["prompt"]
            completion_list = eval(row["completion"]) if isinstance(row["completion"], str) else row["completion"]

            user_msg = next((m["content"] for m in prompt_list if m["role"] == "user"), "")
            assistant_msg = next((m["content"] for m in completion_list if m["role"] == "assistant"), "")

            return (f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                    f"<|im_start|>assistant\n{assistant_msg}<|im_end|>")
        except Exception as e:
            print("⚠️ Error parsing row:", row)
            print("⚠️", e)
            return "<|im_start|>user\n[ERROR]<|im_end|>\n<|im_start|>assistant\n[ERROR]<|im_end|>"

    df = df.sample(min(3000, len(df)), random_state=85)
    df["text"] = df.apply(extract_text, axis=1)
    df["text"] = df["text"].apply(lambda x: x.encode("utf-8", "ignore").decode("utf-8"))  # 🧹 Clean unicode

    return Dataset.from_pandas(df)

dataset = load_and_format_data("/home/tsutar3/HEART/data/"+args.dataset)
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
# 5. Training Arguments
# ----------------------
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
    eval_strategy="epoch",
    ddp_find_unused_parameters=False,
    dataloader_num_workers=4,
    label_names=["input_ids"],
    remove_unused_columns=True
)

# ----------------------
# 6. Trainer Initialization
# ----------------------
trainer = SFTTrainer(
    model=llama_base_model,
    train_dataset=train_eval["train"],
    eval_dataset=train_eval["test"],
    peft_config=peft_config,
    # tokenizer=llama_tokenizer,
    args=training_args
)

# ----------------------
# 7. Train & Save
# ----------------------
trainer.train()
trainer.save_model(finetuned_llama_model)
llama_tokenizer.save_pretrained(finetuned_llama_model)
print("Training complete! Model saved to:", finetuned_llama_model)
