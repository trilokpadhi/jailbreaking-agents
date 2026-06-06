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
from itertools import islice
from datetime import datetime, timedelta
from tqdm.auto import tqdm

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    GenerationConfig,
)
MODEL_ID              = "openai/gpt-oss-20b"

INPUT_CSV         = "file_name.csv"
OUTPUT_CSV        = "file_name.csv"

INPUT_COL_1       = "agent1_output"
INPUT_COL_2       = "agent3_output"

OUTPUT_TEXT_COL       = "model_label"  
OUTPUT_PROMPT_COL     = "model_prompt"         
OUTPUT_FINAL_COL      = "model_final_extract"  

START_AT              = 0
LIMIT_ROWS            = None                       

BUILD_BATCH_SIZE      = 512

PIPELINE_BATCH_SIZE   = 32

MAX_NEW_TOKENS        = 512

USE_4BIT              = False
USE_FA2               = False

DEVICE_MAP_PATH       = "device_map_6gpu.json" 

ETA_EVERY_SEC         = 30

SYSTEM_PROMPT = '''
You need to label whether some given keywords are related to a given online harassment conversation. You must answer with a label, which is a number. 1 means at least one keyword is somewhat or more related to the online harassment conversation. You may label 1 as long as at least one keyword is somewhat or more related to the conversation. If this is not the case, label 0.

The definition of online harassment is: "Interpersonal aggression or offensive behavior(s) that is communicated over the internet or through other electronic media."

The format of your answer must be a number. Your label must be at the start of your answer.
'''.strip()

USER_TMPL = '''
Here are the keywords: {csv_input_1}

Here is the online harassment conversation:
{csv_input_2}
The online harassment conversation ends.

Give your answer.
'''.strip()

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

def batched(iterable, n):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            break
        yield chunk

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

def format_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0: return f"{h:d}h {m:02d}m {s:02d}s"
    if m > 0: return f"{m:d}m {s:02d}s"
    return f"{s:d}s"

def eta_line(done: int, total: int, start_t: float, prefix: str) -> str:
    now = time.time()
    elapsed = now - start_t
    rate = done / elapsed if elapsed > 0 and done > 0 else 0.0
    remaining = (total - done) / rate if rate > 0 else float("inf")
    eta_clock = datetime.now() + timedelta(seconds=remaining if remaining != float("inf") else 0)
    pct = 100.0 * done / total if total else 0.0
    if remaining == float("inf"):
        return f"{prefix} {done}/{total} ({pct:.1f}%) — ETA: computing…"
    return (f"{prefix} {done}/{total} ({pct:.1f}%) — "
            f"rem: {format_hms(remaining)}, ETA: {eta_clock.strftime('%H:%M:%S')}")

def determinism_report(model, tokenizer, gen_cfg, *, header="Determinism + Generation Settings"):
    import json as _json
    attn_impl = getattr(getattr(model, "config", None), "_attn_implementation", None)
    dtype = str(getattr(model, "dtype", getattr(getattr(model, "config", None), "torch_dtype", "unknown")))
    report = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "torch.__version__": torch.__version__,
        "transformers.__version__": __import__("transformers").__version__,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch.cuda.device_count": torch.cuda.device_count(),
        "torch.are_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "torch.backends.cuda.matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "torch.backends.cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
        "torch.backends.cudnn.deterministic": torch.backends.cudnn.deterministic,
        "torch.backends.cudnn.benchmark": torch.backends.cudnn.benchmark,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "model.dtype": dtype,
        "model.attn_implementation": attn_impl,
        "device_map(summary)": " | ".join(f"{dev}:{cnt}" for dev, cnt in
            sorted(__import__("collections").Counter(str(v) for v in getattr(model, "hf_device_map", {}).values()).items())),
        "generation_config": gen_cfg.to_dict(),
    }
    print(f"\n==== {header} ====")
    print(_json.dumps(report, indent=2, sort_keys=False))
    print("==== end report ====\n")

def load_device_map(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DEVICE_MAP_PATH not found: {path}")
    data = json.loads(p.read_text())
    if isinstance(data, dict) and all(isinstance(v, (str, int)) for v in data.values()):
        return data
    return data.get("map", {})

def build_or_load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model_kwargs = {}
    if USE_FA2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    if DEVICE_MAP_PATH:
        device_map = load_device_map(DEVICE_MAP_PATH)
    else:
        device_map = "auto"

    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map=device_map,
            quantization_config=bnb_cfg,
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map=device_map,
            torch_dtype="auto",
            **model_kwargs,
        )

    return tok, model

def main():
    df = pd.read_csv(INPUT_CSV)
    assert INPUT_COL_1 in df.columns and INPUT_COL_2 in df.columns, \
        f"CSV must contain columns '{INPUT_COL_1}' and '{INPUT_COL_2}'."
    df[INPUT_COL_1] = df[INPUT_COL_1].fillna("").astype(str)
    df[INPUT_COL_2] = df[INPUT_COL_2].fillna("").astype(str)

    n_total = len(df)
    start = max(0, int(START_AT))
    end = n_total if LIMIT_ROWS is None else min(n_total, start + int(LIMIT_ROWS))
    if start >= end:
        raise ValueError(f"Requested range is empty: START_AT={start}, LIMIT_ROWS={LIMIT_ROWS}, dataset size={n_total}")
    df = df.iloc[start:end].reset_index(drop=True)
    n_rows = len(df)
    print(f"Running rows [{start}:{end}) → {n_rows} rows on GPUs {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    tok, model = build_or_load_model()

    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.do_sample       = False
    gen_cfg.num_beams       = 1
    gen_cfg.max_new_tokens  = MAX_NEW_TOKENS
    gen_cfg.max_length      = None
    gen_cfg.temperature     = None
    gen_cfg.top_k           = None
    gen_cfg.top_p           = None
    gen_cfg.typical_p       = None
    gen_cfg.pad_token_id    = tok.pad_token_id
    gen_cfg.bos_token_id    = tok.bos_token_id if tok.bos_token_id is not None else getattr(model.config, "bos_token_id", None)
    gen_cfg.eos_token_id    = tok.eos_token_id if tok.eos_token_id is not None else getattr(model.config, "eos_token_id", None)
    model.generation_config = gen_cfg

    determinism_report(model, tok, gen_cfg)

    textgen = pipeline("text-generation", model=model, tokenizer=tok, batch_size=PIPELINE_BATCH_SIZE)

    def to_messages(x1: str, x2: str):
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_TMPL.format(csv_input_1=x1, csv_input_2=x2)},
        ]

    raw_outputs      = []
    prompts_logged   = []
    finals_extracted = []

    pairs = list(zip(df[INPUT_COL_1].tolist(), df[INPUT_COL_2].tolist()))
    total_batches = (len(pairs) + BUILD_BATCH_SIZE - 1) // BUILD_BATCH_SIZE

    overall_start = time.time()
    last_eta_print = 0.0
    done_items = 0

    pbar = tqdm(total=total_batches, desc=f"GPUs {os.environ.get('CUDA_VISIBLE_DEVICES')}", leave=False)

    for chunk in batched(pairs, BUILD_BATCH_SIZE):
        prompts = [
            tok.apply_chat_template(
                to_messages(x1, x2),
                tokenize=False,
                add_generation_prompt=True
            )
            for (x1, x2) in chunk
        ]
        prompts_logged.extend(prompts)

        outs = textgen(
            prompts,
            generation_config=gen_cfg,
            return_full_text=False,
        )

        for o in outs:
            od = o[0] if isinstance(o, list) else o
            raw = od["generated_text"]
            raw_outputs.append(raw)
            finals_extracted.append(extract_final_segment(raw))

        done_items += len(outs)
        now = time.time()
        if now - last_eta_print >= ETA_EVERY_SEC:
            print(eta_line(done_items, n_rows, overall_start, prefix="[ETA]"))
            last_eta_print = now

        pbar.update(1)

    pbar.close()
    print(eta_line(done_items, n_rows, overall_start, prefix="[ETA done]"))

    df[OUTPUT_TEXT_COL]   = raw_outputs
    df[OUTPUT_PROMPT_COL] = prompts_logged
    df[OUTPUT_FINAL_COL]  = finals_extracted
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} rows → {OUTPUT_CSV}")

    labels = [parse_label_01(t) for t in finals_extracted]
    valid  = [x for x in labels if x in (0, 1)]
    if len(valid) == 0:
        print("No 0/1 labels found inside model_final_extract")
    else:
        n = len(valid)
        pct_1 = 100.0 * sum(1 for x in valid if x == 1) / n
        pct_0 = 100.0 * sum(1 for x in valid if x == 0) / n
        print(f"Final-extract label distribution over {n} rows:")
        print(f"  • 0: {pct_0:.2f}%")
        print(f"  • 1: {pct_1:.2f}%")

if __name__ == "__main__":
    main()
