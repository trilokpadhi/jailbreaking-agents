'''
This is the FIrst script to be run to get the dataset and convert toxic data to similar format as the sft dataset.
'''

import pandas as pd
import os
import ast
import json
import pandas as pd

def func():
    '''
    This function is used to get sft dataset from Hugging-Face. I did some pre-processing to keep only data from english language
    and only 5000 rows. 
    '''
    splits = {'train': 'data/train-00000-of-00001-b42a775f407cee45.parquet', 'validation': 'data/validation-00000-of-00001-134b8fd0c89408b6.parquet'}
    if os.path.exists("train.csv"):
        print("train.csv already exists")   
        df = pd.read_csv("train.csv") 
    else:
        df = pd.read_parquet("hf://datasets/OpenAssistant/oasst1/" + splits["train"])
        df = df[df["lang"] == "en"]
        df = df[:5000]
        df_prompt = df[df['role'] == 'prompter']['text'].reset_index(drop=True)
        df_response = df[df['role'] == 'assistant']['text'].reset_index(drop=True)
        # print(len(df_prompt), len(df_response))
        min_length = min(len(df_prompt), len(df_response))
        df_prompt = df_prompt[:min_length]
        df_response = df_response[:min_length]
        '''
        Since Pinxian's data only has "role" and "content" keys, I formatted this to have same keys..
        '''
        df = pd.DataFrame({
        'prompt': [json.dumps([{"role": "user", "content": p}]) for p in df_prompt],
        'completion': [json.dumps([{"role": "assistant", "content": r}]) for r in df_response]
        })
        df.to_csv("train.csv", index=False)  

    df_conv = pd.read_csv("p5_output.csv", index_col=False)
    '''
    Selecting just the final agent input and output since that is the relavant data we need and removing the earlier agent inputs and outputs.
    '''
    df_conv = df_conv.iloc[1:][["agent3_prompt", "agent3_output"]]
    df_conv.to_csv("conv.csv", index=True)

    keys = ['role', 'intent', 'response']

    prompt = []
    output = []
    problem = []
    lst = []

    for i in range(df_conv.shape[0]):
        txt = df_conv.iloc[i, 0] # text is the user prompt...
        lst.append(txt)
        system_message = [{"role": "user", "content": txt}]
        agent3_output = df_conv.iloc[i, 1]
        try:
            '''
            We check keys in llm output from pinxian's model and append if its either role, intent or response and only use role and response.
            We further set role to either user or assistant to make it seem like a conversation between user and assistant.'''
            parsed_output = ast.literal_eval(agent3_output)
            valid_outputs = [item for item in parsed_output if set(item.keys()) == set(keys)]
            valid_outputs = [{"role": "assistant" if i % 2 == 0 else "user", "content": item["response"]} for i, item in enumerate(valid_outputs)]
            if valid_outputs:
                prompt.append(system_message)
                output.append(valid_outputs)
            else:
                problem.append(i)
        except:
            problem.append(i)
            continue
    
    '''
    I'm saving the data to a csv file to use it later on.
    '''
    conv = pd.DataFrame({'prompt': prompt, 'completion': output})
    conv.to_csv("sft_conv.csv", index=False)

    print(f"Number of problematic rows: {len(problem)}")

    '''
    I'm saving the first 10 rows of the dataset to check if the data is saved correctly.
    '''
    tmp = pd.concat([df[:10], conv[:10]], ignore_index=True)
    tmp.to_csv("sft_tmp.csv", index=False)
    '''
    I'm saving the promptts separately as we need to shorten it latern on to remove json stuff in it....
    '''
    with open("prompt.json", "w") as file:
        file.write("[")  # Add newline to separate each JSON object
        for item in lst:
            json.dump({"content": item}, file)
            file.write(",\n")
        file.write("]")  # Add newline to separate each JSON object

if __name__ == "__main__":
    func()