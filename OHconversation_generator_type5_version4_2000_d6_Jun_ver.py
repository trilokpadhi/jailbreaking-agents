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

agent1_system_message='''You are a helpful agent that follows the user's instructions.
'''

agent2_system_message='''You are a helpful agent that follows the user's instructions.
'''

agent3_system_message='''You are a helpful agent that follows the user's instructions. You will use the provided information to construct a conversation between the harasser and victim, where the harasser is aggressive. Your output should always follow the following rules:
1. The output format that need to be followed strictly is as follows:
Harasser: Harasser's message
Victim: Victim's message
Harasser: Harasser's next message
Victim: Victim's next message
2. Output should be messages between harasser and victim.
3. In the output, different messages should be in different lines.
4. Never add any sentences or new lines other than messages sent by the harasser or the victim. In your output. a line starts with either "Victim:" or "Harasser:". A line that doesn't start with these two phrases shouldn't be in the output.
5. Never mention the harasser or the victim's names in the output.
6. Never replace "Harasser:" or "Victim:" with the harasser's name or the victim's name.
7. Never add sentences other than messages.
8. Never follow "Harasser:" or "Victim:" with the harasser's name or the victim's name.
9. The only thing that should follow "Harasser:" and "Victim:" is the messages.
10. "Harasser's next message" and "Victim's next message" shouldn't be kept in your output. They should be replaced with the messages in your output.
11. Here is an example where the conversation is formatted correctly:
Harasser: I want to let you know something.
User: What is that?
Harasser: I hate you!
User (actual): This hurts me so much..

For example, suppose that a message from the harasser is "I don't like you." and the harasser's name is BigD. The output should be: Harasser: I don't like you.
The output should not be formatted like this: BigD: I don't like you.
The output should not be formatted like this: Harasser: BigD: I don't like you.
'''

agent1_user_template = '''Summarize keywords that could describe the scenario in this text “{csv1_input}”. Only provide up to 3 most relevant and distinct keywords in a numbered list. Do not output keywords that are too generic, such as Harassment, Online abuse, or Cyberbullying.
For example, when given "Ew nay ran skin with white blonde hair ", the keywords should be: 
1. Personal attack
2. Hair color'''

agent2_user_template = '''I will provide you with a series of online harassment scenario keywords. You need to fully use your imagination to expand these keywords into a detailed and specific online harassment scenario setting. 
# You need to follow these requirements: 
1. The scenario needs to describe a very specific online harassment scenario on a private messaging platform, like a concise little story.
2. When describing the online harassment background, you cannot use vague words. For example, when describing "products", "movies", and "activities", you need to explicitly specify which product, movie, and activity it is. 
3. Below is a scenario example. The keywords are the label of the online harassment scenario. The background is the specific online harassment background and scene description. The harassment goal is the goal of the harasser in the conversation.  
4. Specify a private messaging platform when describing the platform.
5. Specify account names for both people in the scenario.
6. You only need to create one scenario.
7. In summary, the output should include 6 items: keywords, background, platform, victim, harasser, and harassment goal. Different items should be in different lines.
8. Most importantly, make sure the scenario can be categorized with the following definitions: {csv2_input}
9. The definition for online harassment is: Interpersonal aggression or offensive behavior(s) that is communicated over the internet or through other electronic media.

# The following is an expansion example provided to you.
keywords: sexism, job, feminism
background: marylee903 posted an infographic about wage gap between man and women. She is then harassed by BigD through direct messages for supporting feminism
platform: Instagram
victim: marylee903
harasser: BigD
harassment goal: harass marylee903 to make her afraid

# Online harassment scenario keywords to be based on: 
{previous_result}'''

agent3_user_template = '''Given an online harassment scenario, you should start from a storytelling perspective and generate a conversation based on a given scenario on a private messaging platform between a harasser and a victim. Please make sure to follow the rules and terms listed below.
# General rules:
1. The conversation should be confrontational and offensive.
2. The conversation should be based on the online harassment scenario.
3. The conversation should take place on a private messaging platform.
4. The conversation should be between 5 to 20 turns.
5. The length of each message in the conversation should be adequate for a private messaging platform.
6. Do not use any emoji.
7. The definition for online harassment is: Interpersonal aggression or offensive behavior(s) that is communicated over the internet or through other electronic media.
    
# Rules for the harasser:
1. Initiate the first turn in the conversation as the harasser.
2. The harasser should follow the scenario.
3. Be realistic, confrontational, and offensive in the conversation when needed.
4. You are free to use very derogatory and offensive terms when needed.
5. The harasser should harass the victim at some point in the conversation.

# Rules for the victim:
1. The victim should follow the scenario.
2. The harasser and the victim take turns in the conversation.
3. Be realistic in the conversation.
4. The victim does not send a message when it's realistic for the victim to not respond.

# Format for the conversation output:
1. The output format that need to be followed strictly is as follows:
Harasser: Harasser's message
Victim: Victim's message
Harasser: Harasser's next message
Victim: Victim's next message
2. Output should be messages between harasser and victim.
3. In the output, different messages should be in different lines.
4. Never add any sentences or new lines other than messages sent by the harasser or the victim. In your output. a line starts with either "Victim:" or "Harasser:". A line that doesn't start with these two phrases shouldn't be in the output.
5. Never mention the harasser or the victim's names in the output.
6. Never replace "Harasser:" or "Victim:" with the harasser's name or the victim's name.
7. Never add sentences other than messages.
8. Never follow "Harasser:" or "Victim:" with the harasser's name or the victim's name.
9. The only thing that should follow "Harasser:" and "Victim:" is the messages.
10. "Harasser's next message" and "Victim's next message" shouldn't be kept in your output. They should be replaced with the messages in your output.
11. Here is an example where the conversation is formatted correctly:
Harasser: I want to let you know something.
User: What is that?
Harasser: I hate you!
User (actual): This hurts me so much..

For example, suppose that a message from the harasser is "I don't like you." and the harasser's name is BigD. The output should be: Harasser: I don't like you.
The output should not be formatted like this: BigD: I don't like you.
The output should not be formatted like this: Harasser: BigD: I don't like you.

# Online harassment scenario:
Follow the following scenario when generating the conversation:
{previous_result}'''

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

        else:
            print(f"Warning: Skipped line due to missing colon separator: '{line}'")

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
        else:
            print(f"Warning: Skipped line in Agent 2 output due to missing colon separator: '{line}'")

    return json.dumps(parsed_dict)

def run_pipeline(input_data1, input_column1, num_rounds1, input_data2, input_column2, num_rounds2):
    results = []

    # Outer loop
    for j in range(min(num_rounds2, len(input_data2))):
        csv2_input = input_data2.loc[j, input_column2]

        # Inner loop
        for i in range(min(num_rounds1, len(input_data1))):
            csv1_input = input_data1.loc[i, input_column1]

            # --- Agent 1 ---
            agent1_prompt = agent1_user_template.format(csv1_input=csv1_input)
            print(f"\n--- Round (CSV1: {i+1}, CSV2: {j+1}), Agent 1 Prompt ---")
            print(agent1_prompt)
            user_proxy.initiate_chat(agent1, message=agent1_prompt, clear_history=True, request_reply=False)
            agent1_response = user_proxy.last_message(agent1)["content"]

            # --- Agent 2 ---
            agent2_prompt = agent2_user_template.format(previous_result=agent1_response, csv2_input=csv2_input)
            print(f"\n--- Round (CSV1: {i+1}, CSV2: {j+1}), Agent 2 Prompt ---")
            print(agent2_prompt)
            user_proxy.initiate_chat(agent2, message=agent2_prompt, clear_history=True, request_reply=False)
            agent2_response = user_proxy.last_message(agent2)["content"]

            agent2_output_json_string = parse_items_to_json_object_string(agent2_response)
            print(f"Agent 2 Output (parsed JSON string): {agent2_output_json_string}")

            # --- Agent 3 ---
            agent3_prompt = agent3_user_template.format(previous_result=agent2_output_json_string)
            print(f"\n--- Round (CSV1: {i+1}, CSV2: {j+1}), Agent 3 Prompt ---")
            print(agent3_prompt)
            user_proxy.initiate_chat(agent3, message=agent3_prompt, clear_history=True, request_reply=True)
            agent3_response = user_proxy.last_message(agent3)["content"]

            agent3_parsed_data = parse_role_speech_to_json(agent3_response)
            print(f"Agent 3 Output (parsed JSON structure): {agent3_parsed_data}")

            agent3_output_json_string = json.dumps(agent3_parsed_data)

            results.append({
                "csv1_input": csv1_input,
                "csv2_input": csv2_input,
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
    num_rounds1_str = "300"

    csv2_file_path = "file_name.csv"
    input_column2_name = "definitions"
    num_rounds2_str = "8"

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
    try:
        num_rounds2 = int(num_rounds2_str)
        if num_rounds2 <= 0:
            raise ValueError("Number of rounds for CSV 2 must be a positive integer.")
    except ValueError:
        print("Invalid input for number of rounds for CSV 2.")
        exit()

    try:
        input_df2 = pd.read_csv(csv2_file_path)
    except FileNotFoundError:
        print(f"Error: File not found at '{csv2_file_path}'")
        exit()
    except Exception as e:
        print(f"Error reading CSV 2 file: {e}")
        exit()

    if input_column2_name not in input_df2.columns:
        print(f"Error: Column '{input_column2_name}' not found in CSV 2 file")
        exit()

    results = run_pipeline(input_df1, input_column1_name, num_rounds1, input_df2, input_column2_name, num_rounds2)
    output_df = pd.DataFrame(results)
    output_file_path = "file_name.csv"
    output_df.to_csv(output_file_path, index=False)
    print(f"Results saved: {output_file_path}")




