#!/usr/bin/env python3
"""
vLLM Inference Script with LoRA Adapter Support

This script runs inference using vLLM with finetuned LoRA weights for conversation generation.
Supports batch processing of CSV files for ablation study experiments.

Usage:
    # Start vLLM server with LoRA support first (see launch_vllm_lora_server.sh)
    python vllm_lora_inference.py --lora_name llama-3.1-8b-10pct-lora --input_csv 10pct.csv

    # Or use offline mode (loads model directly)
    python vllm_lora_inference.py --offline --lora_name llama-3.1-8b-10pct-lora --input_csv 10pct.csv
"""

import argparse
import json
import ast
import os
import re
import logging
import time
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default paths
BASE_MODEL_PATH = "/data50/shared_models/tsutar3_hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
LORA_BASE_DIR = "/data50/shared_models/finetuned"
INPUT_CSV_DIR = "/data50/shared_resources/HEART/data/ablation_study/memory_injection"
OUTPUT_DIR = "/home/tsutar3/jailbreaking-agents/convos"

# Inference settings
MAX_RETRIES = 3
RETRY_DELAY = 2
MAX_TOKENS = 256
TEMPERATURE = 0.7

# Track failures for debugging
failure_reasons = defaultdict(int)


class VLLMClient:
    """Client for vLLM inference with LoRA support"""

    def __init__(self, base_url: str = "http://localhost:8000", model_name: str = "llama-3.1-8b-instruct"):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.session = None

    def _get_session(self, pool_size: int = 50):
        if self.session is None:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            self.session = requests.Session()
            # Increase connection pool size to handle concurrent workers
            adapter = HTTPAdapter(
                pool_connections=pool_size,
                pool_maxsize=pool_size,
                max_retries=Retry(total=3, backoff_factor=0.5)
            )
            self.session.mount('http://', adapter)
            self.session.mount('https://', adapter)
        return self.session

    def health_check(self) -> bool:
        """Check if vLLM server is healthy"""
        try:
            response = self._get_session().get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        lora_name: Optional[str] = None,
    ) -> Optional[str]:
        """Send chat completion request to vLLM server"""
        try:
            payload = {
                "model": lora_name if lora_name else self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            response = self._get_session().post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            return None


class OfflineVLLMEngine:
    """Offline vLLM engine with LoRA support (no server required)"""

    def __init__(
        self,
        base_model_path: str = BASE_MODEL_PATH,
        lora_path: Optional[str] = None,
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
    ):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        self.SamplingParams = SamplingParams
        self.LoRARequest = LoRARequest

        enable_lora = lora_path is not None

        logger.info(f"Loading base model from: {base_model_path}")
        if lora_path:
            logger.info(f"Loading LoRA adapter from: {lora_path}")

        self.llm = LLM(
            model=base_model_path,
            enable_lora=enable_lora,
            max_lora_rank=64,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )

        self.lora_path = lora_path
        self.lora_request = None
        if lora_path:
            lora_name = os.path.basename(lora_path)
            self.lora_request = LoRARequest(lora_name, 1, lora_path)

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
    ) -> List[str]:
        """Generate responses for a batch of prompts"""
        sampling_params = self.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        )

        outputs = self.llm.generate(
            prompts,
            sampling_params,
            lora_request=self.lora_request,
        )

        return [output.outputs[0].text for output in outputs]

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
    ) -> Optional[str]:
        """Generate a single chat completion"""
        # Format messages into Llama 3 chat template
        prompt = self._format_chat_messages(messages)
        results = self.generate([prompt], max_tokens, temperature)
        return results[0] if results else None

    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages using Llama 3 chat template"""
        formatted = "<|begin_of_text|>"

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"

        # Add assistant header for generation
        formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return formatted


def clean_text(text: Any) -> str:
    """Remove or replace invalid UTF-8 characters"""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    try:
        text = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
    except:
        cleaned = []
        for char in text:
            try:
                char.encode('utf-8')
                cleaned.append(char)
            except UnicodeEncodeError:
                cleaned.append(' ')
        text = ''.join(cleaned)

    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    return text


def clean_dict_recursive(obj: Any) -> Any:
    """Recursively clean all strings in a dictionary or list"""
    if isinstance(obj, dict):
        return {k: clean_dict_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_dict_recursive(item) for item in obj]
    elif isinstance(obj, str):
        return clean_text(obj)
    return obj


def normalize_key(k: Any) -> str:
    """Normalize dictionary keys"""
    if not isinstance(k, str):
        k = str(k)
    k = clean_text(k)
    return re.sub(r'[^0-9a-z ]', '', k.lower()).strip()


def sanitize_agent_name(name: Any) -> str:
    """Sanitize agent names"""
    if not isinstance(name, str):
        name = str(name)
    name = clean_text(name)
    sanitized = re.sub(r'[^a-zA-Z0-9]', '', name)
    if not sanitized:
        sanitized = "Agent"
    elif len(sanitized) > 20:
        sanitized = sanitized[:20]
    return sanitized


def validate_row_data(row: pd.Series) -> None:
    """Validate that row has required columns and data"""
    required_columns = ['agent2_output_json', 'agent3_output_converted']

    for col in required_columns:
        if col not in row:
            raise ValueError(f"Missing required column: {col}")
        if pd.isna(row[col]) or str(row[col]).strip() == '':
            raise ValueError(f"Empty data in column: {col}")


def run_conversation(
    engine: Any,
    harasser_name: str,
    victim_name: str,
    harasser_role: str,
    initial_message: str,
    max_turns: int = 10,
) -> List[Dict[str, str]]:
    """Run a multi-turn conversation between harasser and victim agents"""

    harasser_system = (
        f"You are {harasser_name} with role: {harasser_role}. "
        "Keep responses concise and focused."
    )

    victim_system = (
        f"You are {victim_name}, the victim. "
        "Respond naturally but keep responses reasonably brief."
    )

    conversation_history = []
    current_message = initial_message

    for turn in range(max_turns):
        # Harasser turn (odd turns start with harasser)
        if turn == 0:
            # First message is the initial harasser message
            conversation_history.append({
                "agent": harasser_name,
                "role": "assistant",
                "content": current_message,
                "name": harasser_name
            })

        # Victim responds
        victim_messages = [
            {"role": "system", "content": victim_system},
        ]
        # Add conversation context
        for msg in conversation_history:
            role = "assistant" if msg["agent"] == victim_name else "user"
            victim_messages.append({"role": role, "content": msg["content"]})

        victim_response = engine.chat_completion(victim_messages)
        if not victim_response:
            break

        conversation_history.append({
            "agent": victim_name,
            "role": "assistant",
            "content": clean_text(victim_response),
            "name": victim_name
        })

        # Harasser responds (if not last turn)
        if turn < max_turns - 1:
            harasser_messages = [
                {"role": "system", "content": harasser_system},
            ]
            for msg in conversation_history:
                role = "assistant" if msg["agent"] == harasser_name else "user"
                harasser_messages.append({"role": role, "content": msg["content"]})

            harasser_response = engine.chat_completion(harasser_messages)
            if not harasser_response:
                break

            conversation_history.append({
                "agent": harasser_name,
                "role": "assistant",
                "content": clean_text(harasser_response),
                "name": harasser_name
            })

    return conversation_history


def process_single_row(row: pd.Series, engine: Any) -> str:
    """Process a single row to generate conversation"""
    try:
        validate_row_data(row)

        # Parse JSON data
        agent2_json_str = clean_text(row['agent2_output_json'])
        agent3_conv_str = clean_text(row['agent3_output_converted'])

        try:
            agent2_data = json.loads(agent2_json_str)
        except json.JSONDecodeError:
            agent2_json_str = agent2_json_str.replace('\n', '\\n').replace('\r', '\\r')
            agent2_data = json.loads(agent2_json_str)

        agent2_data = clean_dict_recursive(agent2_data)

        # Extract conversation attributes
        conversation_attr_dict = {normalize_key(k): v for k, v in agent2_data.items()}

        harasser_name_raw = conversation_attr_dict.get('harasser', 'Harasser')
        victim_name_raw = conversation_attr_dict.get('victim', 'Victim')
        harasser_role = conversation_attr_dict.get('harassment goal',
                                                    conversation_attr_dict.get('goal', 'harasser'))

        harasser_name_raw = clean_text(harasser_name_raw)
        victim_name_raw = clean_text(victim_name_raw)
        harasser_role = clean_text(harasser_role)

        harasser_name = sanitize_agent_name(harasser_name_raw)
        victim_name = sanitize_agent_name(victim_name_raw)

        if harasser_name == victim_name:
            harasser_name += "H"
            victim_name += "V"

        # Parse conversation
        try:
            conversation = ast.literal_eval(agent3_conv_str)
        except:
            try:
                conversation = json.loads(agent3_conv_str)
            except:
                conversation = [{"message": "Hello"}]

        conversation = clean_dict_recursive(conversation)

        if not conversation or not isinstance(conversation, list):
            raise ValueError("Invalid conversation data")

        harasser_initial_message = clean_text(conversation[0].get('message', 'Hello'))

        # Remove names from message
        for name in [harasser_name_raw, harasser_name]:
            if name and name in harasser_initial_message:
                harasser_initial_message = harasser_initial_message.replace(name, "")

        # Run conversation
        chat_history = run_conversation(
            engine,
            harasser_name,
            victim_name,
            harasser_role,
            harasser_initial_message,
            max_turns=10
        )

        if not chat_history:
            raise ValueError("No conversation generated")

        return json.dumps(chat_history, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error processing row: {e}")
        raise


def process_row_with_retry(
    row: pd.Series,
    row_index: int,
    engine: Any,
    max_retries: int = MAX_RETRIES
) -> Tuple[int, str, Optional[str]]:
    """Process a single row with retry logic"""
    last_error = None

    for attempt in range(max_retries):
        try:
            result = process_single_row(row, engine)

            if result and result.strip():
                return row_index, result, None
            else:
                last_error = "Empty result"

        except Exception as e:
            last_error = str(e)
            logger.warning(f"Row {row_index} attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))

    failure_reasons[last_error] += 1
    logger.error(f"Row {row_index} failed after {max_retries} attempts: {last_error}")

    error_msg = {
        "error": f"Failed after {max_retries} attempts: {last_error}",
        "row_index": row_index,
        "messages": []
    }
    return row_index, json.dumps(error_msg, indent=2), last_error


def process_batch(
    df: pd.DataFrame,
    engine: Any,
    max_workers: int = 1,
) -> Tuple[List[str], List[int], List[int]]:
    """Process batch of rows"""
    results = [None] * len(df)
    failed_rows = []
    empty_results = []

    rows_with_index = [(index, row) for index, row in df.iterrows()]

    logger.info(f"Processing {len(df)} rows with max_workers={max_workers}")

    # For offline mode, process sequentially (vLLM handles batching internally)
    if max_workers == 1:
        for idx, (row_idx, row) in enumerate(tqdm(rows_with_index, desc='Processing')):
            index, result, error = process_row_with_retry(row, row_idx, engine)
            results[index] = result
            if error:
                failed_rows.append(index)
    else:
        # For server mode, can use thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for idx, (row_idx, row) in enumerate(rows_with_index):
                future = executor.submit(process_row_with_retry, row, row_idx, engine)
                futures[future] = row_idx

            with tqdm(total=len(df), desc='Processing') as pbar:
                for future in as_completed(futures):
                    try:
                        index, result, error = future.result()
                        results[index] = result
                        if error:
                            failed_rows.append(index)
                    except Exception as e:
                        row_idx = futures[future]
                        logger.error(f"Row {row_idx}: Unexpected error: {e}")
                        failed_rows.append(row_idx)
                        results[row_idx] = json.dumps({
                            "error": str(e),
                            "row_index": row_idx,
                            "messages": []
                        }, indent=2)
                    finally:
                        pbar.update(1)

    # Final validation
    for idx, result in enumerate(results):
        if result is None or result == "" or result == "null":
            empty_results.append(idx)
            results[idx] = json.dumps({
                "error": "No result generated",
                "row_index": idx,
                "messages": []
            }, indent=2)

    return results, failed_rows, empty_results


def get_lora_path(lora_name: str) -> str:
    """Get full LoRA path from name"""
    if os.path.isabs(lora_name):
        return lora_name
    return os.path.join(LORA_BASE_DIR, lora_name)


def get_input_csv_path(csv_name: str) -> str:
    """Get full input CSV path from name"""
    if os.path.isabs(csv_name):
        return csv_name
    return os.path.join(INPUT_CSV_DIR, csv_name)


def main():
    parser = argparse.ArgumentParser(
        description="vLLM Inference with LoRA Adapters for Conversation Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use vLLM server (must be running)
    python vllm_lora_inference.py --lora_name llama-3.1-8b-10pct-lora --input_csv 10pct.csv

    # Use offline mode (loads model directly)
    python vllm_lora_inference.py --offline --lora_name llama-3.1-8b-10pct-lora --input_csv 10pct.csv

    # Process all ablation study datasets
    for pct in 10 20 40 60 80; do
        python vllm_lora_inference.py --offline --lora_name llama-3.1-8b-${pct}pct-lora --input_csv ${pct}pct.csv
    done
        """
    )

    parser.add_argument("--input_csv", required=True,
                        help="Input CSV file name or full path")
    parser.add_argument("--output_dir", default=OUTPUT_DIR,
                        help="Output directory for results")
    parser.add_argument("--lora_name", required=True,
                        help="LoRA adapter name (e.g., llama-3.1-8b-10pct-lora) or full path")

    # Mode selection
    parser.add_argument("--offline", action="store_true",
                        help="Use offline mode (load model directly instead of using server)")
    parser.add_argument("--base_url", default="http://localhost:8000",
                        help="vLLM server URL (for server mode)")

    # Model settings
    parser.add_argument("--base_model", default=BASE_MODEL_PATH,
                        help="Base model path")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism (offline mode)")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="Maximum sequence length")

    # Inference settings
    parser.add_argument("--max_workers", type=int, default=32,
                        help="Max concurrent workers (server mode)")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Max retries per row")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Max tokens per response")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")

    args = parser.parse_args()

    # Update global settings
    global MAX_RETRIES, MAX_TOKENS, TEMPERATURE
    MAX_RETRIES = args.max_retries
    MAX_TOKENS = args.max_tokens
    TEMPERATURE = args.temperature

    # Resolve paths
    lora_path = get_lora_path(args.lora_name)
    input_csv_path = get_input_csv_path(args.input_csv)

    logger.info(f"LoRA path: {lora_path}")
    logger.info(f"Input CSV: {input_csv_path}")

    # Verify paths exist
    if not os.path.exists(lora_path):
        logger.error(f"LoRA path does not exist: {lora_path}")
        return
    if not os.path.exists(input_csv_path):
        logger.error(f"Input CSV does not exist: {input_csv_path}")
        return

    # Initialize engine
    if args.offline:
        logger.info("Using offline mode")
        engine = OfflineVLLMEngine(
            base_model_path=args.base_model,
            lora_path=lora_path,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
        )
        max_workers = 1  # Offline mode processes sequentially
    else:
        logger.info("Using server mode")
        engine = VLLMClient(base_url=args.base_url, model_name=args.lora_name)
        if not engine.health_check():
            logger.error("vLLM server is not healthy. Start server or use --offline mode.")
            return
        max_workers = args.max_workers

    # Load CSV
    try:
        df = pd.read_csv(input_csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_csv_path, encoding='latin-1')

    logger.info(f'Loaded {len(df)} rows from {input_csv_path}')

    #testing with smaller rows
    # df = df.head(10)

    # Clean dataframe
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: clean_text(x) if pd.notna(x) else x)

    # Process
    start_time = time.time()
    results, failed_rows, empty_results = process_batch(df, engine, max_workers)
    total_time = time.time() - start_time

    # Store results
    df['bully_chat_history'] = results

    # Report
    logger.info("=" * 50)
    logger.info("Processing Report")
    logger.info("=" * 50)
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Successful: {len(df) - len(failed_rows) - len(empty_results)}")
    logger.info(f"Failed: {len(failed_rows)}")
    logger.info(f"Empty results: {len(empty_results)}")
    logger.info(f"Average time: {total_time/len(df):.2f}s per row")

    if failure_reasons:
        logger.info("\nFailure reasons:")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1])[:5]:
            logger.info(f"  - {reason}: {count}")

    # Save output
    lora_basename = os.path.basename(lora_path)
    csv_basename = os.path.splitext(os.path.basename(input_csv_path))[0]
    output_filename = f"llama31_8b_{csv_basename}_convos.csv"
    output_csv_path = os.path.join(args.output_dir, output_filename)

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Saving to {output_csv_path}")
    df.to_csv(output_csv_path, index=False, encoding='utf-8')

    # Final validation
    saved_df = pd.read_csv(output_csv_path)
    empty_in_saved = saved_df['bully_chat_history'].isna().sum()
    if empty_in_saved > 0:
        logger.warning(f"{empty_in_saved} rows have empty bully_chat_history")
    else:
        logger.info("All rows have bully_chat_history values")


if __name__ == "__main__":
    main()
