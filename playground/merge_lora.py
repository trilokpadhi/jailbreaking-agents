"""
Merge LoRA adapter into base DeepSeek model weights.

Run this once before serving the fine-tuned model:
    python merge_lora.py

Output: ./deepseek-moe-merged/  (a standalone model directory, no PEFT dependency)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ADAPTER_PATH = "./deepseek-moe-sft-output-2"   # saved by trainer.save_model()
BASE_MODEL   = "deepseek-ai/deepseek-moe-16b-chat"
OUTPUT_PATH  = "./deepseek-moe-merged"

print("Loading base model in bf16...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("Merging LoRA weights into base model...")
model = model.merge_and_unload()

print(f"Saving merged model to {OUTPUT_PATH}...")
model.save_pretrained(OUTPUT_PATH)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT_PATH)

print(f"Done. Merged model saved to {OUTPUT_PATH}")
