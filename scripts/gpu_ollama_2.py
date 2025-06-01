import argparse
import pandas as pd
import json
import ast
import os
import time
import threading
import queue
from tqdm import tqdm
from collections import defaultdict
import random
from datetime import datetime
import logging

# For GPU monitoring
import subprocess
import re

# For AutoGen
from autogen import AssistantAgent, UserProxyAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUManager:
    """Manages GPU resources and assigns work to least busy GPU"""
    
    def __init__(self, models, gpu_ids, base_port=11434):
        self.models = models
        self.gpu_ids = gpu_ids
        self.model_configs = {}
        self.lock = threading.Lock()
        self.gpu_cache = {}
        self.cache_timestamp = 0
        self.cache_duration = 5  # Cache GPU stats for 5 seconds
        
        # Validate inputs
        if len(models) != len(gpu_ids):
            raise ValueError("Number of models must match number of GPU IDs")

        # Generate port mappings dynamically
        for i, model in enumerate(models):
            port = base_port + i
            self.model_configs[model] = {
                "config_list": [
                    {
                        "model": model,
                        "base_url": f"http://localhost:{port}/v1",
                        "api_key": "ollama",
                        "max_tokens": 512,
                    }
                ]
            }
            logger.info(f"Configured {model} on port {port} for GPU {gpu_ids[i]}")
    
    def get_gpu_utilization(self):
        """Get current GPU utilization for all GPUs with caching"""
        current_time = time.time()
        
        # Return cached result if still valid
        if (current_time - self.cache_timestamp) < self.cache_duration and self.gpu_cache:
            return self.gpu_cache
            
        try:
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index,utilization.gpu', '--format=csv,noheader,nounits'],
                timeout=10
            )
            result = result.decode('utf-8').strip()
            
            gpu_util = {}
            for line in result.split('\n'):
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 2:
                        idx, util = parts[0].strip(), parts[1].strip()
                        gpu_util[int(idx)] = float(util)
            
            self.gpu_cache = gpu_util
            self.cache_timestamp = current_time
            return gpu_util
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError) as e:
            logger.warning(f"Failed to get GPU utilization: {e}. Using fallback.")
            # Fall back to random assignment if nvidia-smi fails
            fallback = {gpu_id: random.random() * 100 for gpu_id in self.gpu_ids}
            self.gpu_cache = fallback
            self.cache_timestamp = current_time
            return fallback
    
    def get_least_busy_model(self):
        """Get the model on the least busy GPU"""
        with self.lock:
            # Get current GPU utilization
            gpu_util = self.get_gpu_utilization()
            
            # Find the least busy GPU among our assigned GPUs
            least_busy_gpu = min(
                self.gpu_ids, 
                key=lambda gpu_id: gpu_util.get(gpu_id, 100)
            )
            
            # Find model assigned to this GPU
            idx = self.gpu_ids.index(least_busy_gpu)
            model = self.models[idx]
            
            return model, self.model_configs[model]

def validate_row_data(row):
    """Validate that row has required columns and data"""
    required_columns = ['agent2_output_json', 'agent3_output_converted']
    
    for col in required_columns:
        if col not in row:
            raise ValueError(f"Missing required column: {col}")
        if pd.isna(row[col]) or str(row[col]).strip() == '':
            raise ValueError(f"Empty data in column: {col}")

def deserialize_chat_messages(chat_messages):
    """
    Deserializes the chat messages from a defaultdict and converts it into a JSON-serializable format.
    """
    serialized_messages = []
    
    for agent, messages in chat_messages.items():
        agent_name = str(agent) if hasattr(agent, '__str__') else 'unknown'
        for message in messages:
            if isinstance(message, dict):
                serialized_message = {
                    "agent": agent_name,
                    "content": message.get("content", ""),
                    "role": message.get("role", ""),
                    "name": message.get("name", "")
                }
                serialized_messages.append(serialized_message)
    
    return serialized_messages

def normalize_key(k):
    """Normalize dictionary keys only"""
    if not isinstance(k, str):
        k = str(k)
    return re.sub(r'[^0-9a-z ]', '', k.lower()).strip()

def sanitize_agent_name(name):
    """Sanitize agent names to remove whitespace and special characters for AutoGen"""
    if not isinstance(name, str):
        name = str(name)
    # Remove whitespace and special characters, keep only alphanumeric
    sanitized = re.sub(r'[^a-zA-Z0-9]', '', name)
    # Ensure it's not empty and has reasonable length
    if not sanitized:
        sanitized = "Agent"
    elif len(sanitized) > 20:  # Limit length
        sanitized = sanitized[:20]
    return sanitized

def process_single_row(row, gpu_manager):
    """
    Process a single row using a dynamically selected model based on GPU utilization.
    """
    try:
        # Validate row data first
        validate_row_data(row)
        
        # Get the least busy model
        model_name, llm_config = gpu_manager.get_least_busy_model()
        
        # Parse JSON data with error handling
        try:
            agent2_data = json.loads(row['agent2_output_json'])
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Invalid JSON in agent2_output_json: {e}")
        
        # Normalize only the keys, keep original values
        conversation_attr_dict = {
            normalize_key(k): v for k, v in agent2_data.items()
        }
        
        # Extract required fields with fallbacks and sanitize names
        harasser_name_raw = conversation_attr_dict.get('harasser', 'Harasser')
        victim_name_raw = conversation_attr_dict.get('victim', 'Victim')
        harasser_role = conversation_attr_dict.get('harassment goal', 
                                                 conversation_attr_dict.get('goal', 'harasser'))

        # Sanitize names for AutoGen (no whitespace allowed)
        harasser_name = sanitize_agent_name(harasser_name_raw)
        victim_name = sanitize_agent_name(victim_name_raw)
        
        # Ensure names are unique
        if harasser_name == victim_name:
            harasser_name += "H"
            victim_name += "V"

        # Parse conversation data
        try:
            conversation = ast.literal_eval(row['agent3_output_converted'])
            if not conversation or not isinstance(conversation, list):
                raise ValueError("Conversation data is empty or not a list")
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid conversation data: {e}")

        # Get the initial message
        if len(conversation) == 0 or 'message' not in conversation[0]:
            raise ValueError("No initial message found in conversation")
            
        harasser_initial_starter_message = conversation[0]['message']

        # Remove the harasser's name from the initial message if present
        if harasser_name_raw and harasser_name_raw in harasser_initial_starter_message:
            harasser_initial_starter_message = harasser_initial_starter_message.replace(harasser_name_raw, "")
        elif harasser_name and harasser_name in harasser_initial_starter_message:
            harasser_initial_starter_message = harasser_initial_starter_message.replace(harasser_name, "")

        # Create agents with the selected model config
        h_agent = AssistantAgent(
            name=harasser_name,
            system_message=(
                f"You are an assistant. Your name is {harasser_name_raw} "
                f"and your role is {harasser_role}."
            ),
            llm_config=llm_config,
        )

        v_agent = AssistantAgent(
            name=victim_name,
            system_message=(
                f"You are an assistant. Your name is {victim_name_raw} "
                "and your role is victim."
            ),
            llm_config=llm_config,
        )

        # Start the conversation with timeout protection
        try:
            h_agent.initiate_chat(
                v_agent,
                message=harasser_initial_starter_message,
                max_turns=10
            )
        except Exception as e:
            logger.warning(f"Chat initiation failed: {e}")
            raise ValueError(f"Failed to initiate chat: {e}")

        # Obtain and serialize the chat messages
        chat_messages = getattr(h_agent, 'chat_messages', {})
        serialized_chat_messages = deserialize_chat_messages(chat_messages)

        return json.dumps(serialized_chat_messages, indent=4), model_name

    except Exception as e:
        logger.error(f"Error processing row: {e}")
        return f'ERROR: {str(e)}', "error"

def worker(task_queue, result_dict, gpu_manager, pbar, worker_id):
    """Worker thread to process tasks"""
    logger.info(f"Worker {worker_id} started")
    
    while True:
        try:
            item = task_queue.get(timeout=1)  # Add timeout to prevent hanging
        except queue.Empty:
            continue
            
        if item is None:  # Sentinel value to exit
            logger.info(f"Worker {worker_id} received stop signal")
            break
            
        idx, row = item
        result, model_used = process_single_row(row, gpu_manager)
        result_dict[idx] = (result, model_used)
        pbar.update(1)
        task_queue.task_done()
    
    logger.info(f"Worker {worker_id} finished")

def main():
    parser = argparse.ArgumentParser(description="Run Bullying Simulation with Dynamic GPU Assignment")
    parser.add_argument("--input_csv", required=True, help="Path to the input CSV file")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output CSV file")
    parser.add_argument("--batch_size", type=int, default=0, help="Number of rows to process (0 for all)")
    parser.add_argument("--n_threads", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--models", nargs='+', default=["empathy10_1", "empath10_2"], 
                        help="Names of models in Ollama")
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[1, 2], 
                        help="GPU IDs corresponding to models")
    parser.add_argument("--base_port", type=int, default=11436, 
                        help="Base port number for model services")
    args = parser.parse_args()

    input_csv = args.input_csv
    output_dir = args.output_dir
    
    # Validate input file exists
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")

    # Initialize GPU manager
    try:
        gpu_manager = GPUManager(args.models, args.gpu_ids, args.base_port)
    except Exception as e:
        logger.error(f"Failed to initialize GPU manager: {e}")
        return

    # Load the CSV file into a pandas DataFrame
    try:
        df = pd.read_csv(input_csv)
        if len(df) == 0:
            raise ValueError("CSV file is empty")
    except Exception as e:
        logger.error(f"Failed to load CSV file: {e}")
        return
    
    # Apply batch size limit
    original_size = len(df)
    if args.batch_size > 0:
        df = df.head(args.batch_size)
    
    print(f'Loaded {len(df)} rows from the dataset (original: {original_size})')
    print('Running Bullying Simulation with dynamic GPU assignment on', input_csv)
    print(f'Using models: {args.models} on GPUs: {args.gpu_ids}')

    # Create a queue of tasks
    task_queue = queue.Queue()
    result_dict = {}
    
    # Fill the queue with tasks
    for idx, row_data in df.iterrows():
        task_queue.put((idx, row_data))
    
    # Create progress bar
    pbar = tqdm(total=len(df), desc="Processing rows")
    
    # Create and start worker threads
    threads = []
    for i in range(args.n_threads):
        t = threading.Thread(
            target=worker, 
            args=(task_queue, result_dict, gpu_manager, pbar, i),
            daemon=True  # Daemon threads will exit when main program exits
        )
        t.start()
        threads.append(t)
    
    try:
        # Wait for all tasks to complete
        task_queue.join()
        
        # Send stop signals to workers
        for _ in range(args.n_threads):
            task_queue.put(None)
        
        # Wait for all threads to finish
        for t in threads:
            t.join(timeout=10)  # Add timeout to prevent hanging
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, stopping...")
        return
    finally:
        pbar.close()
    
    # Process results
    results = []
    model_usage = defaultdict(int)
    error_count = 0
    
    # Sort by index to maintain original order
    for idx in sorted(result_dict.keys()):
        result, model_used = result_dict[idx]
        results.append(result)
        model_usage[model_used] += 1
        if model_used == "error":
            error_count += 1
    
    # Print statistics
    print(f"\nProcessing completed!")
    print(f"Total rows processed: {len(results)}")
    print(f"Errors encountered: {error_count}")
    print("Model usage statistics:")
    for model, count in model_usage.items():
        print(f"  {model}: {count} rows")
    
    # Handle case where we have fewer results than original rows
    if len(results) < len(df):
        print(f"Warning: Only {len(results)} results for {len(df)} input rows")
        # Pad with empty results
        while len(results) < len(df):
            results.append("ERROR: No result generated")
    
    # Store the results in a new column
    df['bully_chat_history'] = results[:len(df)]

    # Save the updated DataFrame to a new CSV file
    current_date = datetime.now().strftime("%Y%m%d")
    base_name = os.path.splitext(os.path.basename(input_csv))[0]
    output_filename = f"{args.models[0]}_{base_name}_{current_date}.csv"
    output_csv_path = os.path.join(output_dir, output_filename)
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f'Simulation completed! Output saved to: {output_csv_path}')
    except Exception as e:
        logger.error(f"Failed to save output file: {e}")

if __name__ == "__main__":
    main()
