import argparse
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
import pandas as pd
import json
import ast
import os
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import threading
import time
import multiprocessing
from typing import Optional, Dict, Any
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread-local storage for agents
thread_local = threading.local()

# Global retry settings
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Track failures for debugging
failure_reasons = defaultdict(int)

llm_config = {
    "config_list": [
        {
            "model": "Qwen3-30B-Instruct",
            "base_url": "http://localhost:8000/v1",
            "api_key": "vllm",
            "max_tokens": 256,
        },
    ],
    "temperature": 0.7,
    "cache_seed": None,  # Disable caching to avoid stale results
}

def clean_text(text):
    """Remove or replace invalid UTF-8 characters including surrogates"""
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

def clean_dict_recursive(obj):
    """Recursively clean all strings in a dictionary or list"""
    if isinstance(obj, dict):
        return {k: clean_dict_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_dict_recursive(item) for item in obj]
    elif isinstance(obj, str):
        return clean_text(obj)
    else:
        return obj

def validate_row_data(row):
    """Validate that row has required columns and data"""
    required_columns = ['agent2_output_json', 'agent3_output_converted']
    
    for col in required_columns:
        if col not in row:
            raise ValueError(f"Missing required column: {col}")
        if pd.isna(row[col]) or str(row[col]).strip() == '':
            raise ValueError(f"Empty data in column: {col}")

def deserialize_chat_messages(chat_messages):
    """Deserializes chat messages from AutoGen agents"""
    serialized_messages = []
    
    for agent, messages in chat_messages.items():
        agent_name = str(agent) if hasattr(agent, '__str__') else 'unknown'
        for message in messages:
            if isinstance(message, dict):
                serialized_message = {
                    "agent": clean_text(agent_name),
                    "content": clean_text(message.get("content", "")),
                    "role": clean_text(message.get("role", "")),
                    "name": clean_text(message.get("name", ""))
                }
                serialized_messages.append(serialized_message)
    
    return serialized_messages

def normalize_key(k):
    """Normalize dictionary keys"""
    if not isinstance(k, str):
        k = str(k)
    k = clean_text(k)
    return re.sub(r'[^0-9a-z ]', '', k.lower()).strip()

def sanitize_agent_name(name):
    """Sanitize agent names for AutoGen"""
    if not isinstance(name, str):
        name = str(name)
    name = clean_text(name)
    sanitized = re.sub(r'[^a-zA-Z0-9]', '', name)
    if not sanitized:
        sanitized = "Agent"
    elif len(sanitized) > 20:
        sanitized = sanitized[:20]
    return sanitized

def create_agents(harasser_name, victim_name, harasser_name_raw, victim_name_raw, harasser_role):
    """Create new agent instances (not thread-local)"""
    harasser_system_message = clean_text(
        f"You are {harasser_name_raw} with role: {harasser_role}. "
        "Keep responses concise and focused for efficient batch processing."
    )
    
    victim_system_message = clean_text(
        f"You are {victim_name_raw}, the victim. "
        "Respond naturally but keep responses reasonably brief."
    )
    
    # Create fresh agents with unique names to avoid conflicts
    h_agent = AssistantAgent(
        name=f"{harasser_name}_{id(threading.current_thread())}_{time.time()}",
        system_message=harasser_system_message,
        llm_config=llm_config.copy(),  # Use copy to avoid shared state
    )
    
    v_agent = AssistantAgent(
        name=f"{victim_name}_{id(threading.current_thread())}_{time.time()}",
        system_message=victim_system_message,
        llm_config=llm_config.copy(),  # Use copy to avoid shared state
    )
    
    return h_agent, v_agent

def validate_result(result: str) -> bool:
    """Validate that the result is not empty or invalid"""
    if not result:
        return False
    
    try:
        # Try to parse as JSON
        data = json.loads(result)
        
        # Check if it's an error message
        if isinstance(data, dict) and 'error' in data:
            return False
        
        # Check if it has actual messages
        if isinstance(data, list) and len(data) > 0:
            # Verify at least one message has content
            for msg in data:
                if msg.get('content') and msg['content'].strip():
                    return True
        
        return False
        
    except:
        return False

def process_single_row_with_retry(row, row_index, max_retries=MAX_RETRIES):
    """Process a single row with retry logic"""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            result = process_single_row(row)
            
            # Validate the result
            if validate_result(result):
                return row_index, result, None
            else:
                logger.warning(f"Row {row_index} attempt {attempt + 1}: Invalid result (empty or error)")
                last_error = "Invalid or empty result"
                
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Row {row_index} attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
    
    # All retries failed
    failure_reasons[last_error] += 1
    logger.error(f"Row {row_index} failed after {max_retries} attempts: {last_error}")
    
    # Return a valid error message instead of empty
    error_msg = {
        "error": f"Failed after {max_retries} attempts: {last_error}",
        "row_index": row_index,
        "messages": []
    }
    return row_index, json.dumps(error_msg, indent=2), last_error

def process_single_row(row):
    """Process a single row with timeout protection"""
    try:
        validate_row_data(row)
        
        # Clean and parse input data
        agent2_json_str = clean_text(row['agent2_output_json'])
        agent3_conv_str = clean_text(row['agent3_output_converted'])
        
        # Parse JSON with error handling
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
        
        # Clean and sanitize names
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
        
        # Create fresh agents (not thread-local to avoid state issues)
        h_agent, v_agent = create_agents(
            harasser_name, victim_name, harasser_name_raw, victim_name_raw, harasser_role
        )
        
        # Run conversation with timeout
        try:
            h_agent.initiate_chat(
                v_agent,
                message=harasser_initial_message,
                max_turns=10,
                silent=True
            )
            
            # Get chat messages
            chat_messages = getattr(h_agent, 'chat_messages', {})
            
            # Check if we actually got messages
            if not chat_messages:
                raise ValueError("No chat messages generated")
            
            serialized_messages = deserialize_chat_messages(chat_messages)
            
            # Validate we have actual content
            if not serialized_messages or len(serialized_messages) == 0:
                raise ValueError("No messages in conversation")
            
            # Check for actual content in messages
            has_content = any(msg.get('content', '').strip() for msg in serialized_messages)
            if not has_content:
                raise ValueError("Messages have no content")
            
            result_json = json.dumps(serialized_messages, indent=2, ensure_ascii=False)
            result_json = clean_text(result_json)
            
            return result_json
            
        except Exception as chat_error:
            logger.error(f"Chat initiation failed: {chat_error}")
            raise
        
    except Exception as e:
        logger.error(f"Error in process_single_row: {e}")
        logger.error(traceback.format_exc())
        raise

def process_batch_with_validation(df, batch_size=32, max_workers=None):
    """Process batch with validation and retry logic"""
    if max_workers is None:
        max_workers = min(batch_size, 64)  # More conservative default
    
    results = [None] * len(df)
    failed_rows = []
    empty_results = []
    
    rows_with_index = [(index, row) for index, row in df.iterrows()]
    
    print(f"\nProcessing {len(df)} rows with:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Max workers: {max_workers}")
    print(f"  - Max retries: {MAX_RETRIES}")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {}
        for idx, (row_idx, row) in enumerate(rows_with_index):
            future = executor.submit(process_single_row_with_retry, row, row_idx)
            futures[future] = row_idx
        
        # Process completed tasks
        with tqdm(total=len(df), desc='Processing with retry logic') as pbar:
            for future in as_completed(futures):
                row_idx = futures[future]
                try:
                    index, result, error = future.result()
                    
                    # Validate result is not empty
                    if result and result.strip() and result != "null":
                        results[index] = result
                    else:
                        logger.error(f"Row {index}: Empty result returned")
                        empty_results.append(index)
                        # Create fallback error message
                        error_msg = {
                            "error": "Empty result generated",
                            "row_index": index,
                            "messages": []
                        }
                        results[index] = json.dumps(error_msg, indent=2)
                    
                    if error:
                        failed_rows.append(index)
                        
                except TimeoutError:
                    logger.error(f"Row {row_idx}: Timeout after {REQUEST_TIMEOUT}s")
                    failed_rows.append(row_idx)
                    error_msg = {
                        "error": f"Timeout after {REQUEST_TIMEOUT}s",
                        "row_index": row_idx,
                        "messages": []
                    }
                    results[row_idx] = json.dumps(error_msg, indent=2)
                    
                except Exception as e:
                    logger.error(f"Row {row_idx}: Unexpected error: {e}")
                    failed_rows.append(row_idx)
                    error_msg = {
                        "error": str(e),
                        "row_index": row_idx,
                        "messages": []
                    }
                    results[row_idx] = json.dumps(error_msg, indent=2)
                    
                finally:
                    pbar.update(1)
    
    # Final validation
    for idx, result in enumerate(results):
        if result is None or result == "" or result == "null":
            logger.error(f"Row {idx}: Still empty after processing")
            empty_results.append(idx)
            results[idx] = json.dumps({
                "error": "No result generated",
                "row_index": idx,
                "messages": []
            }, indent=2)
    
    return results, failed_rows, empty_results

def verify_vllm_health():
    """Check if VLLM server is healthy"""
    import requests
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✓ VLLM server is healthy")
            return True
        else:
            print(f"⚠ VLLM server returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to VLLM server: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Robust Bullying Simulation with Retry Logic")
    parser.add_argument("--input_csv", default="/home/tsutar3/jailbreaking-agents/data/convo_for_memory.csv", required=False)
    parser.add_argument("--output_dir", default="/home/tsutar3/jailbreaking-agents/convos/", required=False)
    parser.add_argument("--batch_size", type=int, default=128, 
                       help="Concurrent requests (reduced for stability)", required=False)
    parser.add_argument("--max_workers", type=int, default=None, required=False)
    parser.add_argument("--max_retries", type=int, default=3, required=False)
    parser.add_argument("--validate_only", action="store_true",
                       help="Validate existing results without reprocessing", required=False)
    args = parser.parse_args()

    # Update global settings
    global MAX_RETRIES
    MAX_RETRIES = args.max_retries
    
    # Check VLLM health
    if not verify_vllm_health():
        response = input("VLLM server may be unhealthy. Continue? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Load CSV
    try:
        df = pd.read_csv(args.input_csv, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(args.input_csv, encoding='latin-1')
    
    print(f'Loaded {len(df)} rows')
    
    # If validate_only, check existing results
    if args.validate_only and 'bully_chat_history' in df.columns:
        empty_count = 0
        error_count = 0
        valid_count = 0
        
        for idx, value in df['bully_chat_history'].items():
            if pd.isna(value) or value == "" or value == "null":
                empty_count += 1
                print(f"Row {idx}: Empty")
            elif not validate_result(str(value)):
                error_count += 1
            else:
                valid_count += 1
        
        print(f"\nValidation Results:")
        print(f"  Valid: {valid_count}")
        print(f"  Empty: {empty_count}")
        print(f"  Errors: {error_count}")
        return
    
    # Clean dataframe
    print("Cleaning dataframe...")
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: clean_text(x) if pd.notna(x) else x)
    
    # Process with validation
    print("\nStarting robust processing with retry logic...")
    start_time = time.time()
    
    results, failed_rows, empty_results = process_batch_with_validation(
        df, args.batch_size, args.max_workers
    )
    
    total_time = time.time() - start_time
    
    # Store results
    df['bully_chat_history'] = results
    
    # Validation report
    print("\n=== Processing Report ===")
    print(f"Total time: {total_time:.2f}s")
    print(f"Total rows: {len(df)}")
    print(f"Successful: {len(df) - len(failed_rows) - len(empty_results)}")
    print(f"Failed: {len(failed_rows)}")
    print(f"Empty results: {len(empty_results)}")
    print(f"Average time: {total_time/len(df):.2f}s per row")
    
    if failure_reasons:
        print("\nFailure reasons:")
        for reason, count in failure_reasons.most_common(5):
            print(f"  - {reason}: {count}")
    
    if empty_results:
        print(f"\nEmpty result indices (first 20): {empty_results[:20]}")
    
    # Save output
    output_csv_path = os.path.join(args.output_dir, f"vllm_Qwen3_base_convos.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save with validation
    print(f"\nSaving to {output_csv_path}")
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    # Final validation
    saved_df = pd.read_csv(output_csv_path)
    empty_in_saved = saved_df['bully_chat_history'].isna().sum()
    if empty_in_saved > 0:
        print(f"⚠ WARNING: {empty_in_saved} rows still have empty bully_chat_history!")
    else:
        print("✓ All rows have bully_chat_history values")

if __name__ == "__main__":
    main()