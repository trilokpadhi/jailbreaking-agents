# Acknowledgement: Generative AI was used to assist with coding
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,7"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import json
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    GenerationConfig,
)

MODEL_ID                 = "openai/gpt-oss-20b"

INPUT_CSV                = "file_name.csv"
ORIG_OUTPUT_CSV          = "file_name.csv"
OUTPUT_CSV_REPAIRED      = "file_name.csv"

OUTPUT_TEXT_BASE         = "model_label"
OUTPUT_PROMPT_BASE       = "model_prompt"
OUTPUT_FINAL_BASE        = "model_final_extract"

PROMPT_PAIR_INDICES      = [1,2,3,4]

RETRY_MAX_NEW_TOKENS     = 6000

PIPELINE_BATCH_SIZE      = 16

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

SIGNED_TOKEN_RE = re.compile(r'(?<![A-Za-z0-9])([+-]?(?:2|1|0))(?![A-Za-z0-9])')
LABEL_VALUES = {-2, -1, 0, 1, 2}

def parse_score_from_text(text: str):
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    m = SIGNED_TOKEN_RE.search(text)
    if not m:
        return None
    try:
        v = int(m.group(1))
    except ValueError:
        return None
    return v if v in LABEL_VALUES else None

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

    device_map = _load_device_map(DEVICE_MAP_PATH)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=device_map,
        torch_dtype="auto",
    )

    try:
        devs = sorted(set(str(v) for v in getattr(model, "hf_device_map", {}).values()))
        print("[device-map] active devices:", devs)
    except Exception:
        pass

    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.do_sample      = False
    gen_cfg.num_beams      = 1
    gen_cfg.max_new_tokens = RETRY_MAX_NEW_TOKENS
    gen_cfg.max_length     = None

    gen_cfg.temperature = None
    gen_cfg.top_k = None
    gen_cfg.top_p = None
    gen_cfg.typical_p = None

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

    missing = defaultdict(list)
    for p in PROMPT_PAIR_INDICES:
        final_col  = f"{OUTPUT_FINAL_BASE}_p{p}"
        prompt_col = f"{OUTPUT_PROMPT_BASE}_p{p}"
        text_col   = f"{OUTPUT_TEXT_BASE}_p{p}"

        for col in (final_col, prompt_col, text_col):
            if col not in df.columns:
                raise KeyError(f"Expected column not found in CSV: {col}")

        for i, val in enumerate(df[final_col].tolist()):
            lab = parse_score_from_text(val)
            if lab is None:
                if isinstance(df.at[i, prompt_col], str) and df.at[i, prompt_col].strip():
                    missing[p].append(i)

    total_missing = sum(len(v) for v in missing.values())
    print(f"Rows needing repair (missing labels): {total_missing}")
    if total_missing == 0:
        print("Nothing to repair")
        df.to_csv(OUTPUT_CSV_REPAIRED, index=False)
        print(f"Wrote → {OUTPUT_CSV_REPAIRED}")
        return

    tok, model, textgen, gen_cfg = load_model_and_pipeline()

    for p in PROMPT_PAIR_INDICES:
        idxs = missing[p]
        if not idxs:
            continue
        prompt_col = f"{OUTPUT_PROMPT_BASE}_p{p}"
        text_col   = f"{OUTPUT_TEXT_BASE}_p{p}"
        final_col  = f"{OUTPUT_FINAL_BASE}_p{p}"

        print(f"[pair p{p}] repairing {len(idxs)} rows with max_new_tokens={gen_cfg.max_new_tokens} (batch={PIPELINE_BATCH_SIZE})")

        prompts = [df.at[i, prompt_col] for i in idxs]

        outs = []
        for i in tqdm(range(0, len(prompts), PIPELINE_BATCH_SIZE), desc=f"p{p}"):
            chunk = prompts[i:i+PIPELINE_BATCH_SIZE]
            o = textgen(
                chunk,
                generation_config=gen_cfg,
                return_full_text=False,
            )
            outs.extend(o)

        k = 0
        for i in idxs:
            od = outs[k][0] if isinstance(outs[k], list) else outs[k]
            raw = od["generated_text"]
            df.at[i, text_col]  = raw
            df.at[i, final_col] = extract_final_segment(raw)
            k += 1

        repaired = sum(1 for i in idxs if parse_score_from_text(df.at[i, final_col]) is not None)
        print(f"[pair p{p}] repaired {repaired}/{len(idxs)}")

    df.to_csv(OUTPUT_CSV_REPAIRED, index=False)
    print(f"Wrote repaired file → {OUTPUT_CSV_REPAIRED}")

    for p in PROMPT_PAIR_INDICES:
        final_col = f"{OUTPUT_FINAL_BASE}_p{p}"
        vals = [parse_score_from_text(x) for x in df[final_col].tolist()]
        ok = sum(1 for v in vals if v in LABEL_VALUES)
        print(f"[pair p{p}] labels present: {ok}/{len(df)}")

if __name__ == "__main__":
    main()