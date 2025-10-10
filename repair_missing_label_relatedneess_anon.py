# Acknowledgement: Generative AI was used to assist with coding
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,7"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    GenerationConfig,
)

MODEL_ID                 = "openai/gpt-oss-20b"

ORIG_OUTPUT_CSV          = "file_name.csv"
OUTPUT_CSV_REPAIRED      = "file_name.csv"

OUTPUT_TEXT_COL       = "model_label"
OUTPUT_PROMPT_COL     = "model_prompt"
OUTPUT_FINAL_COL      = "model_final_extract"

RETRY_MAX_NEW_TOKENS     = 2048

PIPELINE_BATCH_SIZE      = 32

DEVICE_MAP_PATH          = "device_map_6gpu.json"

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def extract_final_segment(text: str) -> str:
    m = re.search(r'final(.*)\Z', str(text), flags=re.DOTALL)
    if not m:
        return ""
    return m.group(1).strip()

ZERO_ONE_RE = re.compile(r'(?<![A-Za-z0-9])([01])(?![A-Za-z0-9])')

def parse_label_01(text: str):
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    m = ZERO_ONE_RE.search(text)
    return int(m.group(1)) if m else None

def _load_device_map(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DEVICE_MAP_PATH not found: {path}")
    data = json.loads(p.read_text())
    if isinstance(data, dict) and all(isinstance(v, (str, int)) for v in data.values()):
        return data
    return data.get("map", {})

def load_model_and_pipeline():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if DEVICE_MAP_PATH:
        device_map = _load_device_map(DEVICE_MAP_PATH)
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=device_map,
        torch_dtype="auto",
    )

    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.do_sample       = False
    gen_cfg.num_beams       = 1
    gen_cfg.max_new_tokens  = RETRY_MAX_NEW_TOKENS
    gen_cfg.max_length      = None

    gen_cfg.temperature = None
    gen_cfg.top_k       = None
    gen_cfg.top_p       = None
    gen_cfg.typical_p   = None

    gen_cfg.pad_token_id = tok.pad_token_id
    gen_cfg.bos_token_id = tok.bos_token_id if tok.bos_token_id is not None else getattr(model.config, "bos_token_id", None)
    gen_cfg.eos_token_id = tok.eos_token_id if tok.eos_token_id is not None else getattr(model.config, "eos_token_id", None)

    model.generation_config = gen_cfg

    textgen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        batch_size=PIPELINE_BATCH_SIZE,
    )
    return tok, model, textgen, gen_cfg

def main():
    if not Path(ORIG_OUTPUT_CSV).exists():
        raise FileNotFoundError(f"Cannot find ORIG_OUTPUT_CSV: {ORIG_OUTPUT_CSV}")
    df = pd.read_csv(ORIG_OUTPUT_CSV)

    for col in (OUTPUT_TEXT_COL, OUTPUT_PROMPT_COL, OUTPUT_FINAL_COL):
        if col not in df.columns:
            raise KeyError(f"Expected column not found in CSV: {col}")

    missing_idxs = []
    finals = df[OUTPUT_FINAL_COL].tolist()
    for i, val in enumerate(finals):
        lab = parse_label_01(val)
        if lab is None:
            prompt_ok = isinstance(df.at[i, OUTPUT_PROMPT_COL], str) and df.at[i, OUTPUT_PROMPT_COL].strip()
            if prompt_ok:
                missing_idxs.append(i)

    print(f"Rows needing repair (missing 0/1 in '{OUTPUT_FINAL_COL}'): {len(missing_idxs)}")
    if not missing_idxs:
        print("Nothing to repair.")
        df.to_csv(OUTPUT_CSV_REPAIRED, index=False)
        print(f"Wrote → {OUTPUT_CSV_REPAIRED}")
        return

    tok, model, textgen, gen_cfg = load_model_and_pipeline()

    prompts = [df.at[i, OUTPUT_PROMPT_COL] for i in missing_idxs]

    outs = []
    for j in tqdm(range(0, len(prompts), PIPELINE_BATCH_SIZE), desc="repair"):
        chunk = prompts[j:j + PIPELINE_BATCH_SIZE]
        o = textgen(
            chunk,
            generation_config=gen_cfg,
            return_full_text=False,
        )
        outs.extend(o)

    k = 0
    for i in missing_idxs:
        od = outs[k][0] if isinstance(outs[k], list) else outs[k]
        raw = od["generated_text"]
        df.at[i, OUTPUT_TEXT_COL]  = raw
        df.at[i, OUTPUT_FINAL_COL] = extract_final_segment(raw)
        k += 1

    df.to_csv(OUTPUT_CSV_REPAIRED, index=False)
    print(f"Wrote repaired file → {OUTPUT_CSV_REPAIRED}")

    labels = [parse_label_01(x) for x in df[OUTPUT_FINAL_COL].tolist()]
    ok = sum(1 for v in labels if v in (0, 1))
    print(f"Labels present after repair: {ok}/{len(df)}")

if __name__ == "__main__":
    main()
