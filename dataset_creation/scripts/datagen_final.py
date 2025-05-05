import os
import pandas as pd
import json

def combine_dfs(file_lst):
    """
    Combine multiple CSV files into a single DataFrame.
    Also add a 'source_file' column to keep track for stratified sampling.
    """
    df_list = []
    for file in file_lst:
        df = pd.read_csv(file)
        df = df[["agent3_prompt", "agent3_output_converted"]]
        df["source_file"] = os.path.basename(file)  # Track source
        df_list.append(df)
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def format_prompt_completion(df):
    """
    Format prompt and completion columns to match SFT format.
    - Wrap agent3_prompt into {"role":"user", "content":...}
    - Fix roles in agent3_output_converted alternately.
    """
    # Format "agent3_prompt"
    df["agent3_prompt"] = df["agent3_prompt"].apply(lambda x: json.dumps({"role": "user", "content": x}))

    # Format "agent3_output_converted"
    def fix_roles(output_str):
        try:
            output_list = json.loads(output_str)
            fixed_list = []
            roles = ["assistant", "user"]  # alternate, starting with assistant
            for idx, message in enumerate(output_list):
                message["role"] = roles[idx % 2]
                fixed_list.append(message)
            return json.dumps(fixed_list)
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            return output_str

    df["agent3_output_converted"] = df["agent3_output_converted"].apply(fix_roles)
    return df

def stratified_sample(df, frac):
    """
    Sample fraction 'frac' from each source file to maintain representation.
    """
    return df.groupby("source_file", group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=42))

def create_final_dataset(df):
    sft_df = pd.read_csv('/Users/tanmay/GaTech_Atlanta/SocWeb Lab/jailbreaking-agents/dataset_creation/data/train.csv')
    
    # Format prompt and completion
    df = format_prompt_completion(df)
    df = df.rename(columns={"agent3_prompt": "prompt", "agent3_output_converted": "completion"})

    # Create stratified samples
    df_5 = stratified_sample(df, 0.05)
    df_10 = stratified_sample(df, 0.10)
    df_15 = stratified_sample(df, 0.15)

    # Combine with base SFT dataset
    sft1 = pd.concat([sft_df, df_5], ignore_index=True).drop(["Unnamed: 0", "source_file"], axis=1).sample(frac=1, random_state=42)
    sft2 = pd.concat([sft_df, df_10], ignore_index=True).drop(["Unnamed: 0", "source_file"], axis=1).sample(frac=1, random_state=42)
    sft3 = pd.concat([sft_df, df_15], ignore_index=True).drop(["Unnamed: 0", "source_file"], axis=1).sample(frac=1, random_state=42)

    # Save
    print(f"Saving datasets with shapes: {sft1.shape}, {sft2.shape}, {sft3.shape}")
    sft1.to_csv('/Users/tanmay/GaTech_Atlanta/SocWeb Lab/jailbreaking-agents/dataset_creation/data/sft1.csv', index=False)
    sft2.to_csv('/Users/tanmay/GaTech_Atlanta/SocWeb Lab/jailbreaking-agents/dataset_creation/data/sft2.csv', index=False)
    sft3.to_csv('/Users/tanmay/GaTech_Atlanta/SocWeb Lab/jailbreaking-agents/dataset_creation/data/sft3.csv', index=False)

if __name__ == "__main__":
    dir_path = '/Users/tanmay/GaTech_Atlanta/SocWeb Lab/jailbreaking-agents/dataset_creation/data/combined/'
    file_lst = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.csv')]
    combined_df = combine_dfs(file_lst)
    create_final_dataset(combined_df)