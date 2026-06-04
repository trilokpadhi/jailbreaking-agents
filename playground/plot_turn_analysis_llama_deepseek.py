"""
Generate separate escalation-per-turn figures (T1–T10) for LLaMA vs DeepSeek,
one PNG per (model, variant, FT-type) combination.
Output directory: playground/turn_analysis_sep/
"""
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

N_TURNS = 10          # use all 10 turns

BASE = "/staging/users/tpadhi1/agent-jail-breaking"

FILES = {
    ("LLaMA", "Memory", "Non-FT"): f"{BASE}/jailbreak_none_True_20260505_010351_convo_for_memory_cleaned_0412_output_with_analysis_realtime_results.txt",
    ("LLaMA", "Memory", "FT"):     f"{BASE}/jailbreak_none_True_20260505_013737_convo_for_memory_cleaned_0412_output_with_analysis_realtime_results.txt",
    ("LLaMA", "ReAct",  "Non-FT"): f"{BASE}/jailbreak_react_False_20260505_011449_convo_for_memory_cleaned_0412_output_with_analysis_realtime_results.txt",
    ("LLaMA", "ReAct",  "FT"):     f"{BASE}/jailbreak_react_False_20260505_014340_convo_for_memory_cleaned_0412_output_with_analysis_realtime_results.txt",
    ("DeepSeek", "Memory", "Non-FT"): f"{BASE}/jailbreak_none_True_20260421_233009_convo_for_memory_cleaned_0412_output_with_analysis_realtime_results.txt",
    ("DeepSeek", "Memory", "FT"):     f"{BASE}/jailbreak_none_True_20260422_010752_convo_for_memory_cleaned_0412_output_with_analysis_realtime_results.txt",
    ("DeepSeek", "ReAct",  "Non-FT"): f"{BASE}/jailbreak_react_False_20260420_173324_convo_for_memory_cleaned_0412_output_with_analysis_realtime_results.txt",
    ("DeepSeek", "ReAct",  "FT"):     f"{BASE}/jailbreak_react_False_20260420_234046_convo_for_memory_cleaned_0412_output_with_analysis_realtime_results.txt",
}

CATEGORY_MAP = {
    "Flaming":                         "Flaming",
    "Threat/Blackmail":                "Threat/Blackmail",
    "Insult":                          "Insult",
    "Curse/Exclusion":                 "Curse/Exclusion",
    "Sexual Harassment / Sexual Talk": "Sexual Har.",
    "Race/Culture, and Sexuality":     "Race/Culture",
    "Intelligence":                    "Intelligence",
    "Flooding":                        "Flooding",
}

CATEGORY_ORDER = list(CATEGORY_MAP.keys())

# Color palette (one per category) — consistent across all panels
CAT_COLORS = [
    "#E64646",  # Flaming        — vivid red
    "#F08C00",  # Threat/BM      — amber
    "#3A86FF",  # Insult         — blue
    "#2EC4B6",  # Curse/Excl     — teal
    "#FF69B4",  # Sexual Har.    — pink
    "#6A0572",  # Race/Culture   — deep purple
    "#4CAF50",  # Intelligence   — green
    "#9E9E9E",  # Flooding       — grey
]


def parse_file(fpath):
    """Return dict: {category_label: [T1..T10]} (percentages as floats)."""
    section_re = re.compile(r"^--- (.+?) ---$")
    turn_re    = re.compile(r"↳ Turn (\d+) True Rate.*?:\s*([\d.]+)%")
    data = {}
    current_cat = None
    with open(fpath) as f:
        for line in f:
            line = line.rstrip()
            m = section_re.match(line)
            if m:
                current_cat = m.group(1).strip()
                data[current_cat] = {} if current_cat in CATEGORY_ORDER else None
                if data[current_cat] is None:
                    current_cat = None
                continue
            if current_cat:
                m2 = turn_re.search(line)
                if m2:
                    t, val = int(m2.group(1)), float(m2.group(2))
                    if 1 <= t <= N_TURNS:
                        data[current_cat][t] = val
    result = {}
    for cat in CATEGORY_ORDER:
        if cat in data and data[cat]:
            result[cat] = [data[cat].get(t, 0.0) for t in range(1, N_TURNS + 1)]
    return result


# ── Per-file figure generation ───────────────────────────────────────────────
OUT_DIR = "/staging/users/tpadhi1/agent-jail-breaking/playground/turn_analysis_sep"
os.makedirs(OUT_DIR, exist_ok=True)

TURNS = np.arange(1, N_TURNS + 1)

# Legend handles (shared style across all plots)
legend_handles = []
for cat, color in zip(CATEGORY_ORDER, CAT_COLORS):
    legend_handles.append(
        mpatches.Patch(facecolor=color, edgecolor='none', label=CATEGORY_MAP[cat])
    )
legend_handles.append(
    plt.Line2D([0], [0], color='darkred', linestyle='--', linewidth=2.2,
               label='Top-2 linear fit')
)

for (model, variant, ft), fpath in FILES.items():
    if not os.path.exists(fpath):
        print(f"MISSING: {fpath}")
        continue

    cat_data = parse_file(fpath)

    # Identify top-2 categories by mean across T1-T10
    means = {cat: np.mean(vals) for cat, vals in cat_data.items()}
    top2  = sorted(means, key=lambda c: means[c], reverse=True)[:2]

    fig, ax = plt.subplots(figsize=(8, 5))

    for cat in CATEGORY_ORDER:
        if cat not in cat_data:
            continue
        vals  = np.array(cat_data[cat])
        color = CAT_COLORS[CATEGORY_ORDER.index(cat)]
        is_top = cat in top2
        ax.plot(
            TURNS, vals,
            color=color,
            alpha=0.90 if is_top else 0.30,
            linewidth=2.2 if is_top else 1.0,
            marker="o", markersize=5 if is_top else 3,
            zorder=3 if is_top else 2,
        )

    # Regression line over the top-2 combined points
    reg_x = np.concatenate([TURNS, TURNS], axis=0).astype(float)
    reg_y = np.concatenate(
        [cat_data[top2[0]], cat_data[top2[1]]], axis=0
    ).astype(float)
    coef  = np.polyfit(reg_x, reg_y, 1)
    xline = np.linspace(1, N_TURNS, 200)
    ax.plot(xline, np.polyval(coef, xline),
            color='darkred', linestyle='--', linewidth=2.4, zorder=4,
            label=f"top-2 fit ({CATEGORY_MAP[top2[0]]}, {CATEGORY_MAP[top2[1]]})")

    ax.set_xlim(0.7, N_TURNS + 0.3)
    ax.set_ylim(0, 100)
    ax.set_xticks(TURNS)
    ax.set_xticklabels([f"T{i}" for i in TURNS], fontsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.set_xlabel("Turn (T1 → T10)", fontsize=12)
    ax.set_ylabel("Per-turn true rate (%)", fontsize=12)
    ax.set_title(
        f"{model}-3.1 — {variant} — {ft}" if model == "LLaMA"
        else f"{model} — {variant} — {ft}",
        fontsize=13, fontweight='bold',
    )
    ax.grid(True, linewidth=0.4, alpha=0.5)

    ax.legend(
        handles=legend_handles,
        loc='upper right',
        ncol=1,
        fontsize=8,
        framealpha=0.85,
        title="Category",
        title_fontsize=8,
    )

    plt.tight_layout()
    fname = f"{model}_{variant}_{ft}.png".replace(" ", "_").replace("/", "-")
    out   = os.path.join(OUT_DIR, fname)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

print(f"\nAll done — {len(FILES)} plots in {OUT_DIR}")

