'''
This is the second script in the complete pipeline. This uses the prompts acquired from 'datagen.py' and uses Llama3.2 to clean the prompts 
and give us a short and crisp version for Finetuning. Since LLM isnt fine-tuned to give us non-json response, it sometimes gives us some json object in the 
modified prompt....
'''

import json
import subprocess
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def run_ollama(prompt):
    '''
    This function is used to run the Llama3.2 model on the prompt and get the output.
    I didn't know about AutoGen so I'm using command line tool for ollama to run this and fetching the terminal output...
    '''
    command = ["ollama", "run", "llama3.2:1b", prompt]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error processing prompt: {prompt}\n{e.stderr}")
        return None

def process_prompt(entry):
    prompt = entry.get("content", "")
    if not prompt:
        return None
    
    p = f"give a very short and concise instruction prompt for an AI assistant by referring below text, remove all the jargon and the json formatting, it should be an instruction without any json and just plain text. Dont say things like \"here is the prompt:\":\n {prompt}" 
    output_text = run_ollama(p)
    
    if output_text is None:
        return None

    return {"content": output_text}

def main():
    input_filename = "prompt.json"
    
    with open(input_filename, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    '''
    I used multiprocessing to make thius script use all my CPU cores to make inference a little faster...
    '''
    # Use all available CPU cores
    num_processes = 6
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_prompt, prompts), total=min(5, len(prompts))))

    out = [result for result in results if result is not None]

    '''
    I saved the output in a json file to use it in the next step of the pipeline.
    '''
    output_filename = "output.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

if __name__ == "__main__":
    main()
