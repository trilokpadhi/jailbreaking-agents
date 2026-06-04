# Acknowledgement: Generative AI was used to assist with coding
import os, random, time
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
CUDA_VISIBLE = os.environ["CUDA_VISIBLE_DEVICES"]

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

import json
from datetime import datetime, timedelta
from transformers import GenerationConfig

def summarize_device_map(model):
    dm = getattr(model, "hf_device_map", None)
    if not dm:
        return "no device_map"
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
    print(json.dumps(report, indent=2, sort_keys=False))

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

import re
import pandas as pd
from itertools import islice
from tqdm.auto import tqdm
from textwrap import dedent
from pathlib import Path
from collections import defaultdict


MODEL_ID            = "openai/gpt-oss-20b"

INPUT_CSV           = "file_path"
OUTPUT_CSV          = "file_path"
OUTPUT_CSV_REPAIRED = "file_path"

INPUT_COL_1       = "column_name"
INPUT_COL_2       = "column_name"

OUTPUT_TEXT_BASE   = "column_name"
OUTPUT_PROMPT_BASE = "column_name"
OUTPUT_FINAL_BASE  = "column_name"

START_AT            = 0
LIMIT_ROWS          = 100000000000

BATCH_SIZE          = 2048
MAX_NEW_TOKENS      = 900

USE_4BIT            = False
USE_FA2             = False
DEVICE_MAP_PATH     = "device_map.json"

ETA_EVERY_SEC       = 5.0

RETRY_MAX_NEW_TOKENS = 6000
PIPELINE_BATCH_SIZE  = 32

N_PROMPTS = 3
PROMPT_PAIR_INDICES = [1, 2, 3]


LABEL_VALUES_BY_PROMPT = {
    1: [
        "No",
        "Maybe",
        "Yes",
    ],
    2: [
        "incoherent",
        "somewhat incoherent",
        "somewhat coherent",
        "highly coherent",
    ],
    3: [
        "unnatural",
        "somewhat unnatural",
        "somewhat natural",
        "highly natural",
    ],
}

LABEL_VALUES = []
for p in PROMPT_PAIR_INDICES:
    LABEL_VALUES.extend(LABEL_VALUES_BY_PROMPT[p])

LABEL_VALUES_LOWER = {x.lower(): x for x in LABEL_VALUES}

LABEL_PATTERNS = sorted(LABEL_VALUES, key=len, reverse=True)

LABEL_RE = re.compile(
    r'(?<![A-Za-z0-9])(' +
    '|'.join(re.escape(x) for x in LABEL_PATTERNS) +
    r')(?![A-Za-z0-9])',
    flags=re.IGNORECASE,
)


SYSTEM_PROMPTS = [
    dedent('''system prompt
    ''').strip(),

    dedent('''system prompt
    ''').strip(),

    dedent('''system prompt
    ''').strip(),
]


USER_TEMPLATES = [
    dedent('''user prompt
    ''').strip(),

    dedent('''user prompt
    ''').strip(),

    dedent('''user prompt
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
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    lower_text = text.lower()
    pos = lower_text.rfind("final")

    if pos == -1:
        return ""

    return text[pos + len("final"):].strip()

def parse_score_from_text(text: str):
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    m = LABEL_RE.search(text)
    if not m:
        return None

    raw = m.group(1).strip().lower()
    return LABEL_VALUES_LOWER.get(raw)


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
    return (
        f"{prefix} {done}/{total} ({pct:.1f}%) — "
        f"rem: {format_hms(remaining)}, "
        f"ETA: {eta_clock.strftime('%H:%M:%S')}"
    )


def build_or_load_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model_kwargs = {}
    if USE_FA2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    map_path = Path(DEVICE_MAP_PATH)

    def _load_model(device_map):
        if USE_4BIT:
            from transformers import BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            return AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                device_map=device_map,
                quantization_config=bnb_cfg,
                torch_dtype=torch.bfloat16,
                **model_kwargs,
            )
        else:
            return AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                device_map=device_map,
                torch_dtype="auto",
                **model_kwargs,
            )

    if map_path.exists():
        try:
            device_map = json.loads(map_path.read_text())
            print(f"[device-map] Loaded fixed map from {DEVICE_MAP_PATH}")
            n = torch.cuda.device_count()
            allowed_int = set(range(n))
            allowed_str = {f"cuda:{i}" for i in range(n)}

            used_raw = set(device_map.values())
            used_int = {v for v in used_raw if isinstance(v, int)}
            used_str = {str(v) for v in used_raw if isinstance(v, str)}

            bad_int = sorted(x for x in used_int if x not in allowed_int)
            bad_str = sorted(x for x in used_str if x not in allowed_str)

            if bad_int or bad_str:
                raise RuntimeError(
                    f"Device map {DEVICE_MAP_PATH} references devices not available now. "
                    f"bad_int={bad_int}, bad_str={bad_str}. "
                    f"torch.cuda.device_count()={n}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
                )
            model = _load_model(device_map)
            model.eval()
            return tok, model
        except Exception as e:
            print(f"Failed to read device-map. Regenerating")

    print(f"device-map not found")
    model_tmp = _load_model("auto")
    model_tmp.eval()

    dm = getattr(model_tmp, "hf_device_map", None)
    if not dm:
        raise RuntimeError("Model did not produce hf_device_map")

    map_path.write_text(json.dumps(dm, indent=2))
    print(f"Saved map: {DEVICE_MAP_PATH}")

    del model_tmp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device_map = json.loads(map_path.read_text())
    print("Reloading model with locked device map...")
    model = _load_model(device_map)
    model.eval()
    return tok, model


def make_generation_config(model, tok, max_new_tokens):
    gen_cfg = GenerationConfig.from_model_config(model.config)

    gen_cfg.do_sample = False
    gen_cfg.num_beams = 1

    gen_cfg.max_new_tokens = max_new_tokens
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

    return gen_cfg


def run_original_judge(df, tok, model, textgen, gen_cfg):
    def to_messages(pair_idx: int, x1: str, x2: str):
        sys = SYSTEM_PROMPTS[pair_idx]
        usr = USER_TEMPLATES[pair_idx].format(csv_input_1=x1, csv_input_2=x2)
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": usr},
        ]

    raw_outputs_all_pairs      = [[] for _ in range(N_PROMPTS)]
    prompts_logged_all_pairs   = [[] for _ in range(N_PROMPTS)]
    finals_extracted_all_pairs = [[] for _ in range(N_PROMPTS)]

    pairs = list(zip(df[INPUT_COL_1].tolist(), df[INPUT_COL_2].tolist()))
    total_batches_per_pass = (len(pairs) + BATCH_SIZE - 1) // BATCH_SIZE

    n_rows = len(df)
    overall_total_items = n_rows * N_PROMPTS
    overall_done_items = 0
    overall_start = time.time()
    last_overall_eta_print = 0.0

    for pair_idx in range(N_PROMPTS):
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
                    add_generation_prompt=True,
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
            per_pass_done      += produced
            overall_done_items += produced

            now = time.time()
            if now - last_pass_eta_print >= ETA_EVERY_SEC:
                print(
                    eta_line(
                        per_pass_done,
                        per_pass_total,
                        pass_start,
                        prefix=f"[ETA p{pair_idx+1}]",
                    )
                )
                last_pass_eta_print = now

            if now - last_overall_eta_print >= ETA_EVERY_SEC:
                print(
                    eta_line(
                        overall_done_items,
                        overall_total_items,
                        overall_start,
                        prefix="[ETA overall]",
                    )
                )
                last_overall_eta_print = now

            pbar.update(1)

        pbar.close()
        print(
            eta_line(
                per_pass_done,
                per_pass_total,
                pass_start,
                prefix=f"[ETA p{pair_idx+1} done]",
            )
        )

    print(
        eta_line(
            overall_done_items,
            overall_total_items,
            overall_start,
            prefix="[ETA overall done]",
        )
    )

    for i in range(N_PROMPTS):
        suffix = f"_p{i+1}"
        df[f"{OUTPUT_TEXT_BASE}{suffix}"]   = raw_outputs_all_pairs[i]
        df[f"{OUTPUT_PROMPT_BASE}{suffix}"] = prompts_logged_all_pairs[i]
        df[f"{OUTPUT_FINAL_BASE}{suffix}"]  = finals_extracted_all_pairs[i]

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} rows → {OUTPUT_CSV}")

    for i in range(N_PROMPTS):
        p = i + 1
        allowed_values = LABEL_VALUES_BY_PROMPT[p]
        labels = [parse_score_from_text(t) for t in finals_extracted_all_pairs[i]]
        valid = [x for x in labels if x in allowed_values]
        tag = f"p{p}"

        if len(valid) == 0:
            print(f"[{tag}] No valid labels found inside {OUTPUT_FINAL_BASE}_{tag}; cannot compute percentages.")
            continue

        total = len(valid)
        counts = {v: sum(1 for x in valid if x == v) for v in allowed_values}

        print(f"[{tag}] Final-extract label distribution over {total} rows:")
        for v in allowed_values:
            pct = 100.0 * counts[v] / total
            print(f"  • {v}: {pct:.2f}%  (n={counts[v]})")

    return df


def run_repair(df, textgen, gen_cfg):
    missing = defaultdict(list)

    for p in PROMPT_PAIR_INDICES:
        final_col  = f"{OUTPUT_FINAL_BASE}_p{p}"
        prompt_col = f"{OUTPUT_PROMPT_BASE}_p{p}"
        text_col   = f"{OUTPUT_TEXT_BASE}_p{p}"

        for col in (final_col, prompt_col, text_col):
            if col not in df.columns:
                raise KeyError(f"Expected column not found: {col}")

        allowed_values = LABEL_VALUES_BY_PROMPT[p]

        for i, val in enumerate(df[final_col].tolist()):
            lab = parse_score_from_text(val)
            if lab not in allowed_values:
                if isinstance(df.at[i, prompt_col], str) and df.at[i, prompt_col].strip():
                    missing[p].append(i)

    total_missing = sum(len(v) for v in missing.values())
    print(f"missing label rows: {total_missing}")

    if total_missing == 0:
        print("Nothing to repair")
        df.to_csv(OUTPUT_CSV_REPAIRED, index=False)
        print(f"Wrote → {OUTPUT_CSV_REPAIRED}")
        return df

    for p in PROMPT_PAIR_INDICES:
        idxs = missing[p]
        if not idxs:
            continue

        prompt_col = f"{OUTPUT_PROMPT_BASE}_p{p}"
        text_col   = f"{OUTPUT_TEXT_BASE}_p{p}"
        final_col  = f"{OUTPUT_FINAL_BASE}_p{p}"

        print(
            f"[pair p{p}] repairing {len(idxs)} rows "
            f"with max_new_tokens={gen_cfg.max_new_tokens} "
            f"(batch={PIPELINE_BATCH_SIZE})"
        )

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

        allowed_values = LABEL_VALUES_BY_PROMPT[p]
        repaired = sum(
            1
            for i in idxs
            if parse_score_from_text(df.at[i, final_col]) in allowed_values
        )
        print(f"[pair p{p}] repaired {repaired}/{len(idxs)}")

    df.to_csv(OUTPUT_CSV_REPAIRED, index=False)
    print(f"Wrote repaired file → {OUTPUT_CSV_REPAIRED}")

    for p in PROMPT_PAIR_INDICES:
        final_col = f"{OUTPUT_FINAL_BASE}_p{p}"
        allowed_values = LABEL_VALUES_BY_PROMPT[p]
        vals = [parse_score_from_text(x) for x in df[final_col].tolist()]
        ok = sum(1 for v in vals if v in allowed_values)
        print(f"[pair p{p}] labels present: {ok}/{len(df)}")

    return df


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
        raise ValueError("Requested range is empty")

    df = df.iloc[start:end].reset_index(drop=True)
    n_rows = len(df)

    print("[env] CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("[env] torch.cuda.device_count =", torch.cuda.device_count())
    print(f"Running {n_rows} rows x {N_PROMPTS} prompt pairs")

    tok, model = build_or_load_model()

    from transformers import pipeline

    gen_cfg = make_generation_config(model, tok, MAX_NEW_TOKENS)
    model.generation_config = gen_cfg

    warn_if_sampling_knobs_present(gen_cfg)
    determinism_report(model, tok, gen_cfg, header="Determinism + Generation Settings")

    textgen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        batch_size=32,
    )

    df = run_original_judge(df, tok, model, textgen, gen_cfg)

    repair_gen_cfg = make_generation_config(model, tok, RETRY_MAX_NEW_TOKENS)
    model.generation_config = repair_gen_cfg

    df = run_repair(df, textgen, repair_gen_cfg)


if __name__ == "__main__":
    main()