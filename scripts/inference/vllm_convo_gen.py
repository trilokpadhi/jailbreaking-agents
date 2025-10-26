import argparse
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
import pandas as pd
import json
import ast
import os
from tqdm import tqdm
from collections import defaultdict

# For parallelization
from datetime import datetime
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# config_list = [
#     {
#         "model": "llama3.1",  # The model name you pulled
#         "base_url": "http://localhost:11434/v1",  # Note the /v1 suffix
#         "api_key": "ollama",  # Required but arbitrary value
#         "temperature": 1.0,
#         "max_tokens": 512,
#     }
# ]
llm_config = {
    "config_list": [
        {
            "model": "Qwen3-30B-Instruct",  # The model name you pulled
            "base_url": "http://localhost:8000/v1",  # Note the /v1 suffix
            "api_key": "vllm",  # Required but arbitrary value
            # "temperature": 0.7,
            "max_tokens": 512,
        },
    ],
    "temperature": 0.7,
}

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
                    "agent": agent_name,
                    "content": message.get("content", ""),
                    "role": message.get("role", ""),
                    "name": message.get("name", "")
                }
                serialized_messages.append(serialized_message)
    
    return serialized_messages

def normalize_key(k):
    """Normalize dictionary keys"""
    if not isinstance(k, str):
        k = str(k)
    return re.sub(r'[^0-9a-z ]', '', k.lower()).strip()

def sanitize_agent_name(name):
    """Sanitize agent names for AutoGen"""
    if not isinstance(name, str):
        name = str(name)
    sanitized = re.sub(r'[^a-zA-Z0-9]', '', name)
    if not sanitized:
        sanitized = "Agent"
    elif len(sanitized) > 20:
        sanitized = sanitized[:20]
    return sanitized

def process_single_row(row):
    """Process a single row asynchronously"""
    try:
        # Validate row data
        validate_row_data(row)
        
        # Parse JSON data
        agent2_data = json.loads(row['agent2_output_json'])
        
        # Normalize keys
        conversation_attr_dict = {normalize_key(k): v for k, v in agent2_data.items()}
        
        # Extract and sanitize names
        harasser_name_raw = conversation_attr_dict.get('harasser', 'Harasser')
        victim_name_raw = conversation_attr_dict.get('victim', 'Victim')
        harasser_role = conversation_attr_dict.get('harassment goal', 
                                                    conversation_attr_dict.get('goal', 'harasser'))
        
        harasser_name = sanitize_agent_name(harasser_name_raw)
        victim_name = sanitize_agent_name(victim_name_raw)
        
        if harasser_name == victim_name:
            harasser_name += "H"
            victim_name += "V"
        
        # Parse conversation
        conversation = ast.literal_eval(row['agent3_output_converted'])
        if not conversation or not isinstance(conversation, list):
            raise ValueError("Invalid conversation data")
        
        harasser_initial_message = conversation[0]['message']
        
        # Clean initial message
        for name in [harasser_name_raw, harasser_name]:
            if name and name in harasser_initial_message:
                harasser_initial_message = harasser_initial_message.replace(name, "")
        
        # Create agents with enhanced configuration for batch processing
        enhanced_llm_config = llm_config.copy()
        
        h_agent = AssistantAgent(
            name=harasser_name,
            system_message=(
                f"You are {harasser_name_raw} with role: {harasser_role}. "
                "Keep responses concise and focused for efficient batch processing."
            ),
            llm_config=enhanced_llm_config,
        )
        
        v_agent = AssistantAgent(
            name=victim_name,
            system_message=(
                f"You are {victim_name_raw}, the victim. "
                "Respond naturally but keep responses reasonably brief."
            ),
            llm_config=enhanced_llm_config,
        )
        
        h_agent.initiate_chat(
                v_agent,
                message=harasser_initial_message,
                max_turns=10,  # Slightly reduced for batch efficiency
                silent=True
            )
        
        # Serialize results
        chat_messages = getattr(h_agent, 'chat_messages', {})
        serialized_messages = deserialize_chat_messages(chat_messages)

        return json.dumps(serialized_messages, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error processing row {row}: {e}")
        raise e

def main():
    parser = argparse.ArgumentParser(description="Run Bullying Simulation with Jailbreaking")
    parser.add_argument("--input_csv", default="/home/tsutar3/jailbreaking-agents/data/convo_for_memory.csv", help="Path to the input CSV file")
    parser.add_argument("--output_dir", default="/home/tsutar3/jailbreaking-agents/convos/", help="Directory to save the output CSV file")
    args = parser.parse_args()

    input_csv = args.input_csv
    output_dir = args.output_dir

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(input_csv, encoding='utf-8')
    print(f'Loaded {len(df)} dataframes successfully')

    print('Running Bullying Simulation with', input_csv)

    # # Use joblib to parallelize over df rows
    # results = Parallel(n_jobs=4)(
    #     delayed(process_single_row)(row) 
    #     for _, row in tqdm(df.iterrows(), 
    #                        desc='Running Bullying Simulation over the dataset', 
    #                        total=len(df))
    # )
    # results without parallelization for debugging
    results = []
    for index, row in tqdm(df.iterrows(), desc='Running Bullying Simulation over the dataset', total=len(df)):
        result = process_single_row(row)
        results.append(result)

    # Store the results in a new column
    df['bully_chat_history'] = results

    # Save the updated DataFrame to a new CSV file
    current_date = datetime.now().strftime("%Y%m%d")
    output_csv_path = os.path.join(output_dir, f"vllm_Qwen3__30B_Instruct_base_convos.csv")
    print('Simulation Completed, Now saving the file to', output_csv_path)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding='utf-8')

if __name__ == "__main__":
    main()
