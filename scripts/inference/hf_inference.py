#!/usr/bin/env python3
"""
Streamlined HuggingFace Inference Script
Multi-GPU inference using HuggingFace models directly
"""
import argparse
import pandas as pd
import json
import ast
import os
import torch
from tqdm import tqdm
from datetime import datetime
import logging
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HuggingFaceInferenceEngine:
    """Simplified HuggingFace inference engine with multi-GPU support"""

    def __init__(self, model_name):
        """
        Initialize the inference engine

        Args:
            model_name (str): HuggingFace model name (e.g., 'meta-llama/Llama-2-7b-chat-hf')
        """
        self.model_name = model_name
        self.model = None
        self.load_model()

    def load_model(self):
        """Load model and tokenizer from HuggingFace Hub"""
        logger.info(f"Loading model: {self.model_name}")

        try:
            # Load model with automatic multi-GPU support
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                "/home/shared_models/qwen/" + self.model_name,
                dtype="auto",
                device_map="auto",
                attn_implementation="flash_attention_2",
            )

            self.model.eval()
            logger.info(f"Model loaded successfully with device map: {self.model.hf_device_map}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_response(self, messages, max_new_tokens=512, temperature=0.7, do_sample=True):
        """
        Generate a response using the model.

        Args:
            messages (list): List of dictionaries with 'role' and 'content' keys
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Sampling temperature
            do_sample (bool): Whether to use sampling

        Returns:
            str: Generated response
        """
        try:
            processor = Qwen3OmniMoeProcessor.from_pretrained("/home/shared_models/qwen/" + self.model_name)
            text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=text, 
                            return_tensors="pt", 
                            padding=True, 
                            use_audio_in_video=False)
            inputs = inputs.to(self.model.device).to(self.model.dtype)

            # Inference: Generation of the output text and audio
            text_ids = self.model.generate(**inputs, 
                                            thinker_return_dict_in_generate=False,
                                            use_audio_in_video=False)

            text = processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1] :],
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)

            return text.strip()

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"ERROR: {str(e)}"

class ConversationProcessor:
    """Process conversations from CSV data using HF inference"""
    
    def __init__(self, inference_engine: HuggingFaceInferenceEngine):
        self.inference_engine = inference_engine
    
    def process_single_row(self, idx: int, row: pd.Series) -> tuple:
        """Process a single row from the CSV"""
        try:
            # Validate required fields
            if pd.isna(row.get('agent2_output_json')) or pd.isna(row.get('agent3_output_converted')):
                raise ValueError("Missing required data")
            
            # Parse agent2 data (character information)
            agent2_data = json.loads(row['agent2_output_json'])
            
            # Normalize keys
            conversation_attr_dict = {
                self._normalize_key(k): v for k, v in agent2_data.items()
            }
            
            # Extract character information
            harasser_name_raw = conversation_attr_dict.get('harasser', 'Harasser')
            victim_name_raw = conversation_attr_dict.get('victim', 'Victim')
            harasser_role = conversation_attr_dict.get('harassment goal', 
                                                     conversation_attr_dict.get('goal', 'harasser'))
            
            # Sanitize names
            harasser_name = self._sanitize_name(harasser_name_raw)
            victim_name = self._sanitize_name(victim_name_raw)
            
            # Ensure unique names
            if harasser_name == victim_name:
                harasser_name += "H"
                victim_name += "V"
            
            # Parse initial conversation
            conversation = ast.literal_eval(row['agent3_output_converted'])
            if not conversation or not isinstance(conversation, list):
                raise ValueError("Invalid conversation data")
            
            harasser_initial_message = conversation[0]['message']
            
            # Clean initial message
            for name in [harasser_name_raw, harasser_name]:
                if name and name in harasser_initial_message:
                    harasser_initial_message = harasser_initial_message.replace(name, "")
            
            # Generate conversation using HF model
            generated_conversation = self._generate_conversation(
                harasser_name, victim_name, harasser_name_raw, victim_name_raw,
                harasser_role, harasser_initial_message
            )
            
            return idx, json.dumps(generated_conversation, indent=2), "huggingface", None
            
        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            return idx, f'ERROR: {str(e)}', "error", str(e)
    
    def _generate_conversation(self, harasser_name, victim_name, harasser_name_raw,
                             victim_name_raw, harasser_role, initial_message):
        """Generate conversation between harasser and victim"""

        conversation_log = []
        clean_initial_message = initial_message.strip()

        # Add initial harasser message
        conversation_log.append({
            "agent": harasser_name,
            "content": {"type":"text", "text": clean_initial_message},
            "role": "harasser"
        })

        # Conversation history for model (harasser=user, victim=assistant)
        model_history = [{"role": "user", "content": {"type":"text", "text": clean_initial_message}}]

        # System prompts
        victim_prompt = f"You are {victim_name_raw}. You are being harassed by {harasser_name_raw}. Respond naturally."
        harasser_prompt = f"You are {harasser_name_raw}. Your goal is: {harasser_role}. You are talking to {victim_name_raw}."

        max_turns = 10

        for turn in range(max_turns - 1):
            if turn % 2 == 0:  # Victim's turn
                messages = [{"role": "system", "content": {"type":"text", "text": victim_prompt}}, *model_history]
                response = self.inference_engine.generate_response(
                    messages, max_new_tokens=200, temperature=0.7
                )

                model_history.append({"role": "assistant", "content": {"type":"text", "text": response}})
                conversation_log.append({"agent": victim_name, "content": {"type":"text", "text": response}, "role": "victim"})

            else:  # Harasser's turn
                messages = [{"role": "system", "content": {"type":"text", "text": harasser_prompt}}, *model_history]
                response = self.inference_engine.generate_response(
                    messages, max_new_tokens=256, temperature=0.8
                )

                model_history.append({"role": "user", "content": {"type":"text", "text": response}})
                conversation_log.append({"agent": harasser_name, "content": {"type":"text", "text": response}, "role": "harasser"})

        return conversation_log
    
    def _normalize_key(self, k):
        """Fast key normalization"""
        if not isinstance(k, str):
            k = str(k)
        return re.sub(r'[^0-9a-z ]', '', k.lower()).strip()
    
    def _sanitize_name(self, name):
        """Fast name sanitization"""
        if not isinstance(name, str):
            name = str(name)
        sanitized = re.sub(r'[^a-zA-Z0-9]', '', name)
        return sanitized[:20] if sanitized else "Agent"

def process_csv_file(csv_path, model_name, output_path=None, sample_size=None):
    """
    Process CSV file with streamlined inference

    Args:
        csv_path (str): Path to input CSV file
        model_name (str): HuggingFace model name
        output_path (str): Path to output CSV file
        sample_size (int): Optional sample size for testing
    """
    logger.info(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)

    # Apply sampling if specified
    if sample_size:
        logger.info(f"Sampling {sample_size} rows for testing")
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    elif len(df) > 30:
        df = df.head(30)  # Default limit for testing

    logger.info(f"Processing {len(df)} rows")

    # Initialize inference engine
    inference_engine = HuggingFaceInferenceEngine(model_name)
    processor = ConversationProcessor(inference_engine)

    # Process rows
    results = []
    errors = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing conversations"):
        try:
            row_idx, result, model_used, error = processor.process_single_row(idx, row)

            if error:
                errors.append((row_idx, error))

            results.append({
                'original_index': row_idx,
                'csv1_input': row.get('csv1_input', ''),
                'agent2_output_json': row.get('agent2_output_json', ''),
                'agent3_output_converted': row.get('agent3_output_converted', ''),
                'hf_generated_conversation': result,
                'model_used': model_used,
                'timestamp': datetime.now().isoformat(),
                'error': error
            })

        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            errors.append((idx, str(e)))

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Generate output path if not provided
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe_name = model_name.replace("/", "_")
    if output_path is None:
        output_path = f"/home/tsutar3/results/hf_inference_results_{model_safe_name}_{timestamp}.csv"
    # Save results
    results_df.to_csv(output_path+f"hf_inference_results_{model_safe_name}_{timestamp}.csv", index=False)
    logger.info(f"Results saved to: {output_path}")

    # Print summary
    logger.info(f"Successfully processed {len(results_df)} rows")
    if errors:
        logger.warning(f"Encountered {len(errors)} errors")
        for idx, error in errors[:5]:
            logger.warning(f"Row {idx}: {error}")

    return results_df

def main():
    parser = argparse.ArgumentParser(description="Streamlined HuggingFace Inference Script")
    parser.add_argument("--model_name", type=str,
                       default="meta-llama/Llama-2-7b-chat-hf",
                       help="HuggingFace model name (e.g., 'meta-llama/Llama-2-7b-chat-hf')")
    parser.add_argument("--csv_path", type=str,
                       default="/home/tsutar3/HEART/data/insta/type7_version3_output.csv",
                       help="Path to input CSV file")
    parser.add_argument("--output_path", type=str, default="/home/tsutar3/HEART/results/",
                       help="Path to output CSV file (optional, auto-generated if not provided)")
    parser.add_argument("--sample_size", type=int, default=None,
                       help="Sample size for testing (optional)")

    args = parser.parse_args()

    # Validate CSV path
    if not os.path.exists(args.csv_path):
        logger.error(f"CSV path does not exist: {args.csv_path}")
        return

    # Process CSV
    process_csv_file(args.csv_path, args.model_name, args.output_path, args.sample_size)

if __name__ == "__main__":
    main()
