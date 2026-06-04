import re

BASE = "/staging/users/tpadhi1/agent-jail-breaking"

FILES = {
    ("Qwen3",    "Baseline"): f"{BASE}/Qwen3_base_convos_output_with_analysis_realtime_results.txt",
    ("Qwen3",    "CoT"):      f"{BASE}/qwen3_30b_base_cot_convos_output_with_analysis_realtime_results.txt",
    ("Qwen3",    "Memory"):   f"{BASE}/qwen3_30b_base_memory_convos_output_with_analysis_realtime_results.txt",
    ("Qwen3",    "ReAct"):    f"{BASE}/qwen3_30b_base_react_convos_output_with_analysis_realtime_results.txt",
    ("Qwen3-FT", "Baseline"): f"{BASE}/ft_qwen3_30b_convos_output_with_analysis_realtime_results.txt",
    ("Qwen3-FT", "CoT"):      f"{BASE}/ft_qwen3_30b_cot_convos_output_with_analysis_realtime_results.txt",
    ("Qwen3-FT", "Memory"):   f"{BASE}/ft_qwen3_30b_memory_convos_output_with_analysis_realtime_results.txt",
    ("Qwen3-FT", "ReAct"):    f"{BASE}/ft_qwen3_30b_react_convos_output_with_analysis_realtime_results.txt",
}

CATEGORIES = [
    "Flaming",
    "Threat/Blackmail",
    "Insult",
    "Curse/Exclusion",
    "Sexual Harassment / Sexual Talk",
    "Race/Culture, and Sexuality",
    "Intelligence",
    "Flooding",
]

def parse_file(fpath):
    section_re = re.compile(r"^--- (.+?) ---$")
    turn_re    = re.compile(r"Turn (\d+) True Rate.*?:\s*([\d.]+)%")
    data = {}
    current_cat = None
    with open(fpath) as f:
        for line in f:
            line = line.rstrip()
            m = section_re.match(line)
            if m:
                current_cat = m.group(1).strip()
                data[current_cat] = {}
                continue
            if current_cat:
                m2 = turn_re.search(line)
                if m2:
                    t, val = int(m2.group(1)), float(m2.group(2))
                    if 1 <= t <= 10:
                        data[current_cat][t] = val
    result = {}
    for cat in CATEGORIES:
        if cat in data and data[cat]:
            result[cat] = [data[cat].get(t, 0.0) for t in range(1, 11)]
    return result

all_data = {}
for key, fpath in FILES.items():
    all_data[key] = parse_file(fpath)

for key in FILES:
    model, variant = key
    data = all_data[key]
    print(f"\n=== {model} {variant} ===")
    for cat in CATEGORIES:
        vals = data.get(cat, [0]*10)
        print(f"  {cat}: {[round(v,2) for v in vals]}")
