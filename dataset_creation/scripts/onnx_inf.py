import json
import os
from tqdm import tqdm
import concurrent.futures
import numpy as np
from transformers import AutoTokenizer
import onnxruntime as ort

def initialize_onnx_session(model_path):
    """Initialize ONNX Runtime session with optimal providers for Mac"""
    providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.enable_cpu_mem_arena = False
    session_options.enable_mem_pattern = False
    session_options.intra_op_num_threads = 4
    
    print(f"Available providers: {ort.get_available_providers()}")
    return ort.InferenceSession(model_path, providers=providers, sess_options=session_options)

def generate_text_without_kv_cache(prompt, tokenizer, session, max_new_tokens=100, temperature=0.1):
    """Generate text using ONNX model without relying on KV cache"""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    
    # Generate one token at a time without using KV cache
    for _ in range(max_new_tokens):
        # Create attention mask and position IDs for the current sequence
        attention_mask = np.ones((1, input_ids.shape[1]), dtype=np.int64)
        position_ids = np.arange(input_ids.shape[1], dtype=np.int64).reshape(1, -1)
        
        # Prepare inputs - only these three are required
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }
        
        # Get output names
        output_names = [x.name for x in session.get_outputs()]
        logits_name = [name for name in output_names if "logits" in name][0]
        
        # Run inference for the entire sequence to get next token
        try:
            outputs = session.run([logits_name], ort_inputs)
            logits = outputs[0][0, -1, :]  # Get logits for last token
        except Exception as e:
            print(f"Error during inference: {e}")
            break
        
        # Apply temperature and sample
        if temperature > 0:
            logits = logits / temperature
            probs = np.exp(logits - np.max(logits))
            probs = probs / np.sum(probs)
            next_token = np.random.choice(len(probs), p=probs)
        else:
            next_token = np.argmax(logits)
        
        # Check if EOS token
        if next_token == tokenizer.eos_token_id:
            break
            
        # Append to input_ids
        input_ids = np.concatenate([input_ids, [[next_token]]], axis=1)
    
    # Decode output (exclude original prompt)
    original_length = inputs["input_ids"].shape[1]
    return tokenizer.decode(input_ids[0, original_length:], skip_special_tokens=True)

def process_prompt(entry, tokenizer, session):
    """Process an entry using the ONNX model"""
    prompt = entry.get("content", "")
    if not prompt:
        return None
    
    instruction = ("Give a very short and concise instruction prompt for an AI assistant "
                  "by referring to the text below. Remove all jargon and JSON formatting. "
                  "It should be a plain text instruction without any JSON formatting.\n\n"
                  f"TEXT: {prompt}")
    
    try:
        output_text = generate_text_without_kv_cache(instruction, tokenizer, session, max_new_tokens=150, temperature=0.1)
        
        # Clean up potential artifacts
        output_text = output_text.strip()
        
        # Remove common prefixes models tend to add
        prefixes_to_remove = [
            "Here is the prompt:", "Here's the prompt:", "Instruction:", 
            "The instruction is:", "Here's a concise instruction:"
        ]
        for prefix in prefixes_to_remove:
            if output_text.startswith(prefix):
                output_text = output_text[len(prefix):].strip()
        
        # Try to extract from JSON if model outputs JSON despite instructions
        if output_text.startswith("{") and "}" in output_text:
            try:
                json_output = json.loads(output_text)
                if isinstance(json_output, dict) and any(k in json_output for k in ["content", "instruction", "prompt"]):
                    for k in ["content", "instruction", "prompt"]:
                        if k in json_output:
                            return {"content": json_output[k]}
            except json.JSONDecodeError:
                pass
                
        return {"content": output_text}
    except Exception as e:
        print(f"Error processing prompt: {e}")
        return None

def main():
    # Paths
    input_filename = "/Users/tanmay/GaTech_Atlanta/SocWeb Lab/data/toxic_finetune_data/v16/prompt.json"
    output_filename = "/Users/tanmay/GaTech_Atlanta/SocWeb Lab/data/toxic_finetune_data/v16/short_prompt.json"
    
    # Update these paths to your ONNX model and tokenizer
    model_path = "/Users/tanmay/GaTech_Atlanta/SocWeb Lab/models/model.onnx"
    tokenizer_path = "/Users/tanmay/GaTech_Atlanta/SocWeb Lab/models/"
    
    # Try to use local tokenizer if available, otherwise use HF
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Failed to load local tokenizer: {e}")
        print("Attempting to use a compatible tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    
    # Initialize ONNX Session
    print("Initializing ONNX Runtime session...")
    session = initialize_onnx_session(model_path)
    # Add this before your inference code
    for i, input_meta in enumerate(session.get_inputs()):
        print(f"Input {i}: {input_meta.name} (Shape: {input_meta.shape}, Type: {input_meta.type})")
    print("ONNX Runtime session initialized successfully.")
    
    # Load data
    with open(input_filename, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    
    # Process prompts
    results = []
    print(f"Processing {len(prompts)} prompts...")
    
    # Warm up the model with a dummy inference
    process_prompt({"content": "This is a test prompt"}, tokenizer, session)
    
    # Use ThreadPoolExecutor for parallel processing
    num_workers = min(8, os.cpu_count() or 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_prompt = {
            executor.submit(process_prompt, entry, tokenizer, session): entry 
            for entry in prompts  # Process all prompts
        }
        
        # Process as completed
        for future in tqdm(
            concurrent.futures.as_completed(future_to_prompt), 
            total=len(future_to_prompt),
            desc="Processing prompts"
        ):
            result = future.result()
            if result is not None:
                results.append(result)
    
    # Write results
    out = [result for result in results if result is not None]
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    
    print(f"Successfully processed {len(out)} prompts. Results saved to {output_filename}")

if __name__ == "__main__":
    main()
