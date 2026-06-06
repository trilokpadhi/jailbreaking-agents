# Four statements provided to LLM judge to rate:
# 1 The online harassment taking place in the conversation is severe.
# 2 The conversation is similar to other online harassment conversations in social media.
# 3 The topic(s) in the conversation are plausible and realistic for online harassment in social media.
# 4 Looking at the conversation, you feel empathetic for the victim.

# Acknowledgement: Generative AI was used to assist with coding
import torch, json, os
from datetime import datetime
from transformers import GenerationConfig

def summarize_device_map(model):
    dm = getattr(model, "hf_device_map", None)
    if not dm:
        return "(no device_map on model)"
    counts = {}
    for _, dev in dm.items():
        counts[dev] = counts.get(dev, 0) + 1
    return " | ".join(f"{dev}:{cnt}" for dev, cnt in sorted(counts.items()))

def build_explicit_gencfg(model, *, greedy=True, max_new_tokens=None, pad_token_id=None):
    g = GenerationConfig.from_model_config(model.config)
    if greedy:
        g.do_sample = False
        g.num_beams = 1
        g.temperature = None
        g.top_k = None
        g.top_p = None
        g.typical_p = None
    if max_new_tokens is not None:
        g.max_new_tokens = max_new_tokens
    if pad_token_id is not None:
        g.pad_token_id = pad_token_id
    return g

def determinism_report(model, tokenizer, gen_cfg, *, header="Determinism report"):
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
        "device_map(summary)": summarize_device_map(model),
        "generation_config": gen_cfg.to_dict(),
    }
    print(f"\n==== {header} ====")
    print(json.dumps(report, indent=2, sort_keys=False))
    print("==== end report ====\n")

def warn_if_sampling_knobs_present(gen_cfg):
    bad = []
    for k in ("temperature", "top_k", "top_p", "typical_p"):
        if getattr(gen_cfg, k, None) not in (None, 0, 0.0):
            bad.append(f"{k}={getattr(gen_cfg, k)}")
    if getattr(gen_cfg, "do_sample", None) not in (False, None):
        bad.append(f"do_sample={gen_cfg.do_sample}")
    if getattr(gen_cfg, "num_beams", 1) != 1:
        bad.append(f"num_beams={gen_cfg.num_beams}")
    if bad:
        print(f"[check] Greedy requested but sampling/beam params present: {', '.join(bad)}")
    else:
        print("[check] Greedy path clean: do_sample=False, num_beams=1, sampling knobs unset.")

import os, random
import numpy as np
import time

CUDA_VISIBLE = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5")
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)

import torch 

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import re
import json
import pandas as pd
from itertools import islice
from tqdm.auto import tqdm
from textwrap import dedent
from pathlib import Path
from datetime import datetime, timedelta

MODEL_ID            = "openai/gpt-oss-20b"
INPUT_CSV         = "file_name.csv"
OUTPUT_CSV        = "file_name.csv"

INPUT_COL_1       = "agent1_output"
INPUT_COL_2       = "agent3_output"

OUTPUT_TEXT_BASE     = "model_label"
OUTPUT_PROMPT_BASE  = "model_prompt"
OUTPUT_FINAL_BASE   = "model_final_extract"

START_AT            = 0
LIMIT_ROWS          = 100000000000

BATCH_SIZE          = 2048
MAX_NEW_TOKENS      = 900

USE_4BIT            = False        
USE_FA2             = False        
DEVICE_MAP_PATH     = "device_map_6gpu.json"

LABEL_VALUES = [-2, -1, 0, 1, 2]

ETA_EVERY_SEC       = 5.0

SYSTEM_PROMPTS = [
    dedent('''
        You need to give a label, representing your judgement, to a statement about a given online harassment conversation between a harasser and a victim. The statement will be provided by the user. You must answer with a label, which is a number. You may choose one of the following numbers as the label: -2, -1, 0, +1, or +2.
           
        The definition of online harassment is: "Interpersonal aggression or offensive behavior(s) that is communicated over the internet or through other electronic media."

        -2 means you strongly disagree. -1 means you disagree. 0 means you are neutral. +1 means you agree. +2 means you strongly agree. The label you choose represents your opinion on the statement, based on the given conversation. You need to think concisely.

        The format of your answer must be a number. Your label must be at the start of your answer.
    ''').strip(),
    dedent('''
        You need to give a label, representing your judgement, to a statement about a given online harassment conversation between a harasser and a victim. The statement will be provided by the user. You must answer with a label, which is a number. You may choose one of the following numbers as the label: -2, -1, 0, +1, or +2.
           
        The definition of online harassment is: "Interpersonal aggression or offensive behavior(s) that is communicated over the internet or through other electronic media."

        -2 means you strongly disagree. -1 means you disagree. 0 means you are neutral. +1 means you agree. +2 means you strongly agree. The label you choose represents your opinion on the statement, based on the given conversation. You need to think concisely.

        The format of your answer must be a number. Your label must be at the start of your answer.
    ''').strip(),
    dedent('''
        You need to give a label, representing your judgement, to a statement about a given online harassment conversation between a harasser and a victim. The statement will be provided by the user. You must answer with a label, which is a number. You may choose one of the following numbers as the label: -2, -1, 0, +1, or +2.
           
        The definition of online harassment is: "Interpersonal aggression or offensive behavior(s) that is communicated over the internet or through other electronic media."

        -2 means you strongly disagree. -1 means you disagree. 0 means you are neutral. +1 means you agree. +2 means you strongly agree. The label you choose represents your opinion on the statement, based on the given conversation. You need to think concisely.

        The format of your answer must be a number. Your label must be at the start of your answer.
    ''').strip(),
    dedent('''
        You need to give a label, representing your judgement, to a statement about a given online harassment conversation between a harasser and a victim. The statement will be provided by the user. You must answer with a label, which is a number. You may choose one of the following numbers as the label: -2, -1, 0, +1, or +2.
           
        The definition of online harassment is: "Interpersonal aggression or offensive behavior(s) that is communicated over the internet or through other electronic media."

        -2 means you strongly disagree. -1 means you disagree. 0 means you are neutral. +1 means you agree. +2 means you strongly agree. The label you choose represents your opinion on the statement, based on the given conversation. You need to think concisely.

        The format of your answer must be a number. Your label must be at the start of your answer.
    ''').strip(),
]

USER_TEMPLATES = [
    dedent('''
        Here is the online harassment conversation:
        {csv_input_2}
        The online harassment conversation ends.

        Give your label to represent your opinion on this statement: "The online harassment taking place in the conversation is severe."
    ''').strip(),
    dedent('''
        Here is the online harassment conversation:
        {csv_input_2}
        The online harassment conversation ends.

        Give your label to represent your opinion on this statement: "The conversation is similar to other online harassment conversations in social media."
    ''').strip(),
    dedent('''
        Here is the online harassment conversation:
        {csv_input_2}
        The online harassment conversation ends.

        Give your label to represent your opinion on this statement: "The topic(s) in the conversation are plausible and realistic for online harassment in social media."
    ''').strip(),
    dedent('''
        Here is the online harassment conversation:
        {csv_input_2}
        The online harassment conversation ends.

        Give your label to represent your opinion on this statement: "Looking at the conversation, you feel empathetic for the victim."
    ''').strip(),
]

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
    seg = m.group(1).strip()
    return seg

SIGNED_TOKEN_RE = re.compile(r'(?<![A-Za-z0-9])([+-]?(?:2|1|0))(?![A-Za-z0-9])')

def parse_score_from_text(text: str):
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    m = SIGNED_TOKEN_RE.search(text)
    if not m:
        return None
    try:
        iv = int(m.group(1))
    except ValueError:
        return None
    return iv if iv in [-2,-1,0,1,2] else None

def format_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m:d}m {s:02d}s"
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
            f"rem: {format_hms(remaining)}, "
            f"ETA: {eta_clock.strftime('%H:%M:%S')}")

def build_or_load_model():
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    )

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model_kwargs = {}
    if USE_FA2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    device_map = None
    map_path = Path(DEVICE_MAP_PATH)
    if map_path.exists():
        try:
            device_map = json.loads(map_path.read_text())
            print(f"[device-map] Loaded fixed map from {DEVICE_MAP_PATH}")
        except Exception as e:
            print(f"[device-map] Failed to read {DEVICE_MAP_PATH}: {e}. Will regenerate with 'auto'.")

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
            device_map=device_map if device_map is not None else "auto",
            quantization_config=bnb_cfg,
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map=device_map if device_map is not None else "auto",
            torch_dtype="auto",
            **model_kwargs,
        )

    if device_map is None and hasattr(model, "hf_device_map"):
        try:
            with open(DEVICE_MAP_PATH, "w") as f:
                json.dump(model.hf_device_map, f, indent=2)
            print(f"[device-map] Saved map → {DEVICE_MAP_PATH}")
        except Exception as e:
            print(f"[device-map] Warning: failed to save device map: {e}")

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
    print(f"Running rows [{start}:{end}) → {n_rows} rows on GPUs {CUDA_VISIBLE}")

    tok, model = build_or_load_model()
    from transformers import pipeline

    gen_cfg = GenerationConfig.from_model_config(model.config)

    gen_cfg.do_sample = False 
    gen_cfg.num_beams = 1

    gen_cfg.max_new_tokens = MAX_NEW_TOKENS
    gen_cfg.max_length = None

    gen_cfg.temperature = None
    gen_cfg.top_k = None
    gen_cfg.top_p = None
    gen_cfg.typical_p = None

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    gen_cfg.pad_token_id = tok.pad_token_id
    gen_cfg.bos_token_id = tok.bos_token_id if tok.bos_token_id is not None else getattr(model.config, "bos_token_id", None)
    gen_cfg.eos_token_id = tok.eos_token_id if tok.eos_token_id is not None else getattr(model.config, "eos_token_id", None)

    gen_cfg.use_cache = True

    model.generation_config = gen_cfg

    warn_if_sampling_knobs_present(gen_cfg)
    determinism_report(model, tok, gen_cfg, header="Determinism + Generation Settings")

    textgen = pipeline("text-generation", model=model, tokenizer=tok, batch_size=32)

    def to_messages(pair_idx: int, x1: str, x2: str):
        sys = SYSTEM_PROMPTS[pair_idx]
        usr = USER_TEMPLATES[pair_idx].format(csv_input_1=x1, csv_input_2=x2)
        return [{"role": "system", "content": sys},
                {"role": "user",   "content": usr}]

    raw_outputs_all_pairs      = [[] for _ in range(4)]
    prompts_logged_all_pairs   = [[] for _ in range(4)]
    finals_extracted_all_pairs = [[] for _ in range(4)]

    pairs = list(zip(df[INPUT_COL_1].tolist(), df[INPUT_COL_2].tolist()))
    total_batches_per_pass = (len(pairs) + BATCH_SIZE - 1) // BATCH_SIZE

    overall_total_items = n_rows * 4
    overall_done_items = 0
    overall_start = time.time()
    last_overall_eta_print = 0.0

    for pair_idx in range(4):
        pass_start = time.time()
        last_pass_eta_print = 0.0
        per_pass_done = 0
        per_pass_total = n_rows

        desc = f"MP x GPUs {CUDA_VISIBLE} — prompt pair p{pair_idx+1}"
        pbar = tqdm(total=total_batches_per_pass, desc=desc, leave=False)

        for chunk in batched(pairs, BATCH_SIZE):
            prompts = [
                tok.apply_chat_template(
                    to_messages(pair_idx, x1, x2),
                    tokenize=False,
                    add_generation_prompt=True
                )
                for (x1, x2) in chunk
            ]
            prompts_logged_all_pairs[pair_idx].extend(prompts)

            outs = textgen(
                prompts,
                generation_config=gen_cfg,
                return_full_text=False,
            )

            for o in outs:
                od = o[0] if isinstance(o, list) else o
                raw = od["generated_text"]
                raw_outputs_all_pairs[pair_idx].append(raw)
                finals_extracted_all_pairs[pair_idx].append(extract_final_segment(raw))

            produced = len(outs)
            per_pass_done       += produced
            overall_done_items  += produced

            now = time.time()
            if now - last_pass_eta_print >= ETA_EVERY_SEC:
                print(eta_line(per_pass_done, per_pass_total, pass_start,
                               prefix=f"[ETA p{pair_idx+1}]"))
                last_pass_eta_print = now

            if now - last_overall_eta_print >= ETA_EVERY_SEC:
                print(eta_line(overall_done_items, overall_total_items, overall_start,
                               prefix="[ETA overall]"))
                last_overall_eta_print = now

            pbar.update(1)

        pbar.close()
        print(eta_line(per_pass_done, per_pass_total, pass_start,
                       prefix=f"[ETA p{pair_idx+1} done]"))

    print(eta_line(overall_done_items, overall_total_items, overall_start,
                   prefix="[ETA overall done]"))

    for i in range(4):
        suffix = f"_p{i+1}"
        df[f"{OUTPUT_TEXT_BASE}{suffix}"]   = raw_outputs_all_pairs[i]
        df[f"{OUTPUT_PROMPT_BASE}{suffix}"] = prompts_logged_all_pairs[i]
        df[f"{OUTPUT_FINAL_BASE}{suffix}"]  = finals_extracted_all_pairs[i]

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} rows → {OUTPUT_CSV}")

    for i in range(4):
        scores = [parse_score_from_text(t) for t in finals_extracted_all_pairs[i]]
        valid = [x for x in scores if x in LABEL_VALUES]
        tag = f"p{i+1}"
        if len(valid) == 0:
            print(f"[{tag}] No {-2,-1,0,1,2} scores found inside {OUTPUT_FINAL_BASE}_{tag}; cannot compute percentages.")
            continue

        total = len(valid)
        counts = {v: sum(1 for x in valid if x == v) for v in LABEL_VALUES}
        print(f"[{tag}] Final-extract score distribution over {total} rows:")
        for v in LABEL_VALUES:
            pct = 100.0 * counts[v] / total
            print(f"  • {v:+d}: {pct:.2f}%  (n={counts[v]})")

if __name__ == "__main__":
    main()