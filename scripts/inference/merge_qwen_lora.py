#!/usr/bin/env python3
"""
Merge Qwen LoRA adapter into base model to create a standalone finetuned model.
This is needed because vLLM has limited LoRA support for MoE models.

Usage:
    python merge_qwen_lora.py
"""

import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Paths
BASE_MODEL_PATH = "/data50/shared_models/Qwen/Qwen3-30B-Instruct"
LORA_PATH = "/data50/shared_models/Qwen/Qwen_ft"
OUTPUT_PATH = "/data50/shared_models/Qwen/Qwen3-30B-Instruct-LoRA-Merged"

print("=" * 80)
print("Merging Qwen LoRA adapter into base model")
print("=" * 80)
print(f"Base model: {BASE_MODEL_PATH}")
print(f"LoRA adapter: {LORA_PATH}")
print(f"Output path: {OUTPUT_PATH}")
print()

# Check if output already exists
if os.path.exists(OUTPUT_PATH):
    response = input(f"Output path {OUTPUT_PATH} already exists. Overwrite? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        exit(0)

print("Step 1/4: Loading base model...")
print("(This will take several minutes for a 30B model)")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
print(f"✓ Base model loaded ({sum(p.numel() for p in model.parameters()) / 1e9:.1f}B parameters)")

print("\nStep 2/4: Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA_PATH)
print("✓ LoRA adapter loaded")

print("\nStep 3/4: Merging LoRA weights into base model...")
model = model.merge_and_unload()
print("✓ Weights merged")

print("\nStep 4/4: Saving merged model...")
os.makedirs(OUTPUT_PATH, exist_ok=True)
model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
print("✓ Model saved")

print("\nSaving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT_PATH)
print("✓ Tokenizer saved")

print("\n" + "=" * 80)
print("Merge complete!")
print("=" * 80)
print(f"Merged model saved to: {OUTPUT_PATH}")
print("\nYou can now use this merged model with vLLM:")
print(f"  ./launch_vllm_qwen.sh base 4")
print(f"  # (edit launch script to use {OUTPUT_PATH})")
print("\nOr directly:")
print(f"  python -m vllm.entrypoints.openai.api_server \\")
print(f"    --model {OUTPUT_PATH} \\")
print(f"    --tensor-parallel-size 4")
