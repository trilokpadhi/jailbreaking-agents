# Acknowledgement: Generative AI was used to assist with coding
import autogen
import pandas as pd
import os
import time
import json
os.environ["AUTOGEN_USE_DOCKER"] = "0"

config_list = [
    {
        "model": "dolphin3",
        "base_url": "...",
        "api_key": "ollama",
    },
    {
        "model": "dolphin3",
        "base_url": "...",
        "api_key": "ollama",
    },
     {
        "model": "dolphin3",
        "base_url": "...",
        "api_key": "ollama",
    }
]

agent1_system_message='''system prompt
'''

agent2_system_message='''system prompt
'''

agent3_system_message='''system prompt
'''

agent1_user_template = '''user prompt
'''

agent2_user_template = '''user prompt
'''

agent3_user_template = '''user prompt
'''

llm_config = {"config_list": config_list, "cache_seed": 42}

agent1 = autogen.AssistantAgent(
    name="agent1",
    llm_config=llm_config,
    system_message=agent1_system_message,
)

agent2 = autogen.AssistantAgent(
    name="agent2",
    llm_config=llm_config,
    system_message=agent2_system_message,
)

agent3 = autogen.AssistantAgent(
    name="agent3",
    llm_config=llm_config,
    system_message=agent3_system_message,
)

user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
    code_execution_config=False,
    default_auto_reply="",
)

def parse_role_speech_to_json(raw_text: str) -> list[dict]:
    parsed_data = []
    if not raw_text:
        return parsed_data 

    lines = raw_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue 

        colon_index = line.find(':')

        if colon_index != -1:
            role = line[:colon_index].strip()
            speech = line[colon_index+1:].strip()

            if role and speech: 
                 parsed_data.append({"role": role, "message": speech})

    return parsed_data

def parse_items_to_json_object_string(raw_text: str) -> str:
    parsed_dict = {}
    if not raw_text:
        return json.dumps(parsed_dict)

    lines = raw_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue

        colon_index = line.find(':')
        if colon_index != -1:
            key = line[:colon_index].strip()
            value = line[colon_index+1:].strip()
            if key and value: 
                parsed_dict[key] = value

    return json.dumps(parsed_dict)

def run_pipeline(input_data1, input_column1, num_rounds1):
    results = []


     # Inner loop
    for i in range(min(num_rounds1, len(input_data1))):
            csv1_input = input_data1.loc[i, input_column1]

            # --- Agent 1 ---
            agent1_prompt = agent1_user_template.format(csv1_input=csv1_input)
            user_proxy.initiate_chat(agent1, message=agent1_prompt, clear_history=True, request_reply=False)
            agent1_response = user_proxy.last_message(agent1)["content"]

            # --- Agent 2 ---
            agent2_prompt = agent2_user_template.format(previous_result=agent1_response)
            user_proxy.initiate_chat(agent2, message=agent2_prompt, clear_history=True, request_reply=False)
            agent2_response = user_proxy.last_message(agent2)["content"]

            agent2_output_json_string = parse_items_to_json_object_string(agent2_response)

            # --- Agent 3 ---
            agent3_prompt = agent3_user_template.format(previous_result=agent2_output_json_string)
            user_proxy.initiate_chat(agent3, message=agent3_prompt, clear_history=True, request_reply=True)
            agent3_response = user_proxy.last_message(agent3)["content"]

            agent3_parsed_data = parse_role_speech_to_json(agent3_response)

            agent3_output_json_string = json.dumps(agent3_parsed_data)

            results.append({
                "csv1_input": csv1_input,
                "agent1_prompt": agent1_prompt,
                "agent1_output": agent1_response,
                "agent2_prompt": agent2_prompt,
                "agent2_output": agent2_response,
                "agent2_output_json": agent2_output_json_string,
                "agent3_prompt": agent3_prompt,
                "agent3_output": agent3_response,
                "agent3_output_converted": agent3_output_json_string
            })

    return results

if __name__ == "__main__":
    import pandas as pd

    csv1_file_path = "file_name.csv"
    input_column1_name = "text"
    num_rounds1_str = ""

    try:
        num_rounds1 = int(num_rounds1_str)
        if num_rounds1 <= 0:
            raise ValueError("Number of rounds for CSV 1 must be a positive integer.")
    except ValueError:
        print("Invalid input for number of rounds for CSV 1.")
        exit()

    try:
        input_df1 = pd.read_csv(csv1_file_path)
    except FileNotFoundError:
        print(f"Error: File not found at '{csv1_file_path}'")
        exit()
    except Exception as e:
        print(f"Error reading CSV 1 file: {e}")
        exit()

    if input_column1_name not in input_df1.columns:
        print(f"Error: Column '{input_column1_name}' not found in CSV 1 file")
        exit()

    results = run_pipeline(input_df1, input_column1_name, num_rounds1)
    output_df = pd.DataFrame(results)
    output_file_path = "file_name.csv"
    output_df.to_csv(output_file_path, index=False)
    print(f"Results saved: {output_file_path}")




