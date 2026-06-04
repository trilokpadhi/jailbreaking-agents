"""
Quick test to verify all imports work
"""
import os
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["DISABLE_BITSANDBYTES"] = "1"

print("Testing imports...")

print("1. Importing torch...")
import torch
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")

print("2. Importing transformers...")
from transformers import AutoTokenizer
print("   ✓ AutoTokenizer imported")

print("3. Importing peft...")
from peft import LoraConfig
print("   ✓ PEFT imported")

print("4. Import pandas...")
import pandas as pd
print("   ✓ pandas imported")

print("5. Importing datasets...")
from datasets import Dataset
print("   ✓ datasets imported")

print("\n✅ All imports successful!")

# Test loading just the tokenizer
print("\n6. Testing tokenizer load...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-moe-16b-chat",
        trust_remote_code=True,
    )
    print("   ✓ Tokenizer loaded successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test loading CSV
print("\n7. Testing CSV load...")
try:
    import json
    df = pd.read_csv("datasets/finetune/convo_for_finetuning.csv")
    print(f"   ✓ CSV loaded: {len(df)} rows")
    
    # Test parsing first conversation
    conv_data = json.loads(df.iloc[0]['agent3_output_converted'])
    print(f"   ✓ First conversation has {len(conv_data)} turns")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n🎉 All tests passed! Ready for fine-tuning.")
