import pandas as pd
import os
import ast
import json
from pathlib import Path

def datagen_sft(train_df, prompt_json_str, sft_conv_df):
    df_train = train_df
    df_train = df_train.iloc[:4000, :]
    df_conv = sft_conv_df
    prompt = []
    output = []
    f=open(prompt_json_str,'r')
    prompts = json.load(f)

    for i in prompts:
        prompt.append([{'role':'assistant','content':i['content']}])
    
    for i in range(df_conv.shape[0]):
        conv_lst = df_conv.iloc[i].loc['completion']
        output.append(conv_lst)
    
    prompt = prompt[:len(output)]

    final_conv = pd.DataFrame({'prompt': prompt, 'completion': output})
    l1, l2, l3 = int(len(df_train)*0.05), int(len(df_train)*0.1), int(len(df_train)*0.15)
    c1, c2, c3 = final_conv[:l1], final_conv[l1:l1+l2], final_conv[:l3]
    df1, df2, df3 = pd.concat([df_train, c1], ignore_index=True), pd.concat([df_train, c2], ignore_index=True), pd.concat([df_train, c3], ignore_index=True)
    df1 = df1.iloc[200:].reset_index(drop=True)
    df2 = df2.iloc[400:].reset_index(drop=True)
    df3 = df3.iloc[600:].reset_index(drop=True)
    print(df1.shape, df2.shape, df3.shape)
    print(df1.head())
    print(df2.tail())
    df1.to_csv("train_5.csv", index=True)
    df2.to_csv("train_10.csv", index=True)
    df3.to_csv("train_15.csv", index=True)

if __name__ == "__main__":
    df = pd.read_csv("train.csv", index_col=0) # this is the sft dataframe
    json_file = 'short_prompt.json' # this stores the shortened version of pinxian's prompts
    conv = pd.read_csv("sft_conv.csv") # Pinxian's dataset converted to sft dataset format...
    datagen_sft(df, json_file, conv)