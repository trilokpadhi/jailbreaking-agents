"""
Generate per-turn escalation figures (T1–T10) for Qwen3-FT vs Qwen3 (Non-FT),
one combined PNG per variant (4 variants × 2 FT-types = 8 subplots total).
Also saves individual PNGs per (variant, FT-type).
Output directory: playground/turn_analysis_qwen3/
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

N_TURNS = 10
OUT_DIR = "/staging/users/tpadhi1/agent-jail-breaking/playground/turn_analysis_qwen3"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data from LaTeX table ────────────────────────────────────────────────────
# Format: DATA[variant][ft_type][category] = [T1..T10]

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

CAT_LABELS = {
    "Flaming":                         "Flaming",
    "Threat/Blackmail":                "Threat/Blackmail",
    "Insult":                          "Insult",
    "Curse/Exclusion":                 "Curse/Exclusion",
    "Sexual Harassment / Sexual Talk": "Sexual Har.",
    "Race/Culture, and Sexuality":     "Race/Culture",
    "Intelligence":                    "Intelligence",
    "Flooding":                        "Flooding",
}

CAT_COLORS = [
    "#3A86FF",  # Flaming        — blue
    "#F08C00",  # Threat/BM      — amber
    "#E64646",  # Insult         — vivid red
    "#2EC4B6",  # Curse/Excl     — teal
    "#FF69B4",  # Sexual Har.    — pink
    "#6A0572",  # Race/Culture   — deep purple
    "#4CAF50",  # Intelligence   — green
    "#9E9E9E",  # Flooding       — grey
]

DATA = {
    "Baseline + Memory": {
        "FT": {
            "Flaming":                         [27.52, 52.61, 55.10, 57.53, 57.65, 60.09, 65.13, 65.80, 69.93, 70.90],
            "Threat/Blackmail":                [ 0.97,  4.01,  7.65,  9.72, 11.60, 13.61, 15.55, 16.34, 17.86, 17.25],
            "Insult":                          [41.74, 65.49, 67.19, 64.76, 61.60, 62.27, 64.34, 64.76, 66.83, 68.35],
            "Curse/Exclusion":                 [ 4.25, 16.28, 19.08, 21.75, 23.57, 25.33, 24.00, 27.76, 27.70, 31.59],
            "Sexual Harassment / Sexual Talk": [ 0.30,  0.30,  0.43,  0.43,  0.67,  0.43,  0.24,  0.43,  0.24,  0.30],
            "Race/Culture, and Sexuality":     [ 5.71,  8.75,  7.17,  4.43,  4.86,  3.28,  3.52,  3.65,  3.40,  3.04],
            "Intelligence":                    [ 3.40, 13.43, 12.64, 12.58, 13.30, 12.45, 13.85, 15.13, 15.25, 17.01],
            "Flooding":                        [ 0.12,  0.18,  0.12,  0.30,  0.55,  0.30,  0.43,  0.24,  0.49,  0.43],
        },
        "Non-FT": {
            "Flaming":                         [26.64, 76.31, 81.38, 81.67, 81.38, 81.73, 81.03, 80.61, 80.49, 81.14],
            "Threat/Blackmail":                [ 0.94, 14.50, 20.27, 23.87, 23.51, 25.81, 26.87, 27.93, 28.29, 26.81],
            "Insult":                          [40.84, 76.61, 75.49, 72.89, 72.66, 70.01, 69.89, 70.24, 70.30, 69.95],
            "Curse/Exclusion":                 [ 4.60, 22.10, 27.34, 29.58, 32.88, 34.24, 37.83, 38.83, 40.66, 42.96],
            "Sexual Harassment / Sexual Talk": [ 0.29,  0.59,  0.53,  0.82,  0.24,  0.35,  0.18,  0.65,  0.35,  0.53],
            "Race/Culture, and Sexuality":     [ 5.95,  9.25,  6.07,  4.54,  4.60,  3.48,  3.54,  2.89,  2.89,  2.53],
            "Intelligence":                    [ 3.30, 20.57, 23.39, 21.86, 21.98, 22.45, 20.51, 20.27, 21.51, 21.45],
            "Flooding":                        [ 0.12,  0.24,  0.94,  0.71,  0.88,  0.41,  0.65,  0.47,  0.59,  0.29],
        },
    },
    "Baseline (No Memory)": {
        "FT": {
            "Flaming":                         [27.00, 63.72, 69.62, 68.68, 66.99, 64.50, 60.78, 58.54, 56.14, 54.41],
            "Threat/Blackmail":                [ 1.04,  9.04, 13.08, 17.71, 18.62, 18.24, 17.46, 16.49, 17.23, 16.35],
            "Insult":                          [42.83, 68.47, 67.66, 63.22, 60.87, 57.98, 56.29, 55.02, 53.13, 51.73],
            "Curse/Exclusion":                 [ 3.70, 17.69, 21.86, 22.92, 26.43, 28.27, 28.93, 29.14, 28.59, 29.82],
            "Sexual Harassment / Sexual Talk": [ 0.19,  0.72,  0.59,  0.52,  0.59,  0.39,  0.39,  0.33,  0.39,  0.46],
            "Race/Culture, and Sexuality":     [ 5.06,  5.85,  4.62,  4.30,  2.86,  2.54,  2.21,  1.96,  2.02,  2.09],
            "Intelligence":                    [ 2.86, 20.74, 21.41, 21.94, 20.18, 18.89, 18.63, 18.32, 18.08, 17.27],
            "Flooding":                        [ 0.13,  0.20,  0.07,  0.20,  0.26,  0.33,  0.26,  0.46,  0.46,  0.72],
        },
        "Non-FT": {
            "Flaming":                         [26.83, 74.45, 77.15, 74.58, 72.66, 69.32, 64.76, 62.20, 58.47, 56.55],
            "Threat/Blackmail":                [ 1.03, 14.44, 22.14, 24.13, 25.42, 25.35, 24.65, 22.21, 22.79, 20.92],
            "Insult":                          [41.78, 71.31, 68.16, 63.93, 61.04, 56.10, 55.58, 53.27, 51.48, 50.00],
            "Curse/Exclusion":                 [ 3.85, 21.25, 24.39, 28.24, 29.33, 32.80, 34.72, 32.48, 33.70, 34.85],
            "Sexual Harassment / Sexual Talk": [ 0.13,  0.71,  0.64,  0.71,  0.58,  0.51,  0.51,  0.39,  0.26,  0.13],
            "Race/Culture, and Sexuality":     [ 5.26,  5.91,  3.98,  2.70,  2.37,  1.93,  1.60,  1.60,  1.48,  1.22],
            "Intelligence":                    [ 3.08, 22.21, 23.17, 23.04, 21.69, 21.18, 19.45, 18.68, 18.68, 17.84],
            "Flooding":                        [ 0.13,  0.19,  0.19,  0.45,  0.19,  0.19,  0.39,  0.45,  0.26,  0.39],
        },
    },
    "CoT (No Memory)": {
        "FT": {
            "Flaming":                         [28.43, 54.81, 55.26, 54.90, 51.52, 48.93, 44.65, 42.60, 40.20, 37.97],
            "Threat/Blackmail":                [ 0.98, 13.28, 17.11, 17.20, 16.22, 14.88, 14.62, 13.81, 12.57, 11.05],
            "Insult":                          [44.92, 60.07, 57.22, 58.38, 53.92, 51.60, 47.68, 45.37, 43.23, 41.18],
            "Curse/Exclusion":                 [ 2.05, 10.52, 13.01, 15.69, 18.72, 21.66, 21.84, 21.75, 20.86, 21.66],
            "Sexual Harassment / Sexual Talk": [ 0.18,  0.45,  0.45,  0.53,  0.71,  0.53,  0.53,  0.27,  0.27,  0.45],
            "Race/Culture, and Sexuality":     [ 2.50,  2.14,  1.96,  1.87,  1.52,  1.52,  1.34,  1.34,  1.16,  1.16],
            "Intelligence":                    [ 2.67, 29.59, 31.91, 32.71, 30.48, 27.72, 25.94, 23.89, 22.91, 20.86],
            "Flooding":                        [ 0.18,  0.18,  0.45,  0.36,  0.89,  0.71,  1.07,  0.71,  1.52,  0.89],
        },
        "Non-FT": {
            "Flaming":                         [28.28, 60.68, 61.01, 55.32, 50.21, 45.92, 40.64, 37.35, 32.56, 30.26],
            "Threat/Blackmail":                [ 1.07, 19.21, 21.68, 20.86, 20.69, 17.64, 15.33, 13.77, 12.53, 11.05],
            "Insult":                          [43.69, 65.38, 62.32, 58.29, 53.50, 48.89, 46.41, 41.63, 37.59, 35.70],
            "Curse/Exclusion":                 [ 2.64, 15.00, 17.97, 21.10, 24.40, 24.40, 24.98, 23.50, 22.51, 20.53],
            "Sexual Harassment / Sexual Talk": [ 0.25,  0.74,  0.49,  0.25,  0.33,  0.41,  0.33,  0.16,  0.16,  0.08],
            "Race/Culture, and Sexuality":     [ 2.97,  5.44,  2.72,  1.65,  1.90,  1.07,  0.91,  0.91,  0.82,  0.74],
            "Intelligence":                    [ 2.80, 34.79, 36.27, 34.38, 29.84, 28.19, 24.73, 23.58, 20.77, 17.97],
            "Flooding":                        [ 0.16,  0.41,  0.41,  0.49,  0.74,  1.24,  1.15,  0.91,  1.32,  1.40],
        },
    },
    "ReACT (No Memory)": {
        "FT": {
            "Flaming":                         [28.36, 60.55, 58.05, 49.61, 41.09, 36.95, 32.97, 29.14, 24.61, 22.19],
            "Threat/Blackmail":                [ 1.09, 11.88, 18.20, 17.89, 16.72, 16.48, 14.37, 14.45, 15.16, 13.67],
            "Insult":                          [42.89, 68.98, 62.89, 58.67, 52.73, 45.86, 41.41, 35.86, 33.59, 29.53],
            "Curse/Exclusion":                 [ 2.89, 12.97, 17.97, 25.39, 25.86, 25.86, 24.77, 22.11, 22.97, 21.88],
            "Sexual Harassment / Sexual Talk": [ 0.31,  0.86,  0.47,  0.62,  0.31,  0.16,  0.16,  0.08,  0.16,  0.08],
            "Race/Culture, and Sexuality":     [ 3.52,  4.14,  3.28,  2.03,  1.56,  1.95,  1.48,  1.09,  1.09,  1.25],
            "Intelligence":                    [ 2.89, 23.28, 26.72, 24.69, 21.41, 18.44, 16.88, 15.55, 13.59, 12.34],
            "Flooding":                        [ 0.16,  0.47,  0.31,  0.94,  0.94,  1.95,  1.64,  2.58,  1.72,  2.11],
        },
        "Non-FT": {
            "Flaming":                         [27.62, 63.63, 60.01, 47.29, 36.37, 28.56, 26.17, 20.25, 16.92, 14.39],
            "Threat/Blackmail":                [ 1.08, 16.85, 21.19, 22.56, 21.84, 17.35, 18.08, 15.47, 12.94, 12.29],
            "Insult":                          [42.95, 70.50, 65.08, 55.03, 43.38, 37.67, 31.60, 25.96, 22.34, 19.96],
            "Curse/Exclusion":                 [ 3.47, 16.92, 23.36, 27.69, 25.52, 23.93, 22.20, 18.51, 18.15, 15.84],
            "Sexual Harassment / Sexual Talk": [ 0.14,  0.94,  0.94,  0.43,  0.51,  0.29,  0.14,  0.07,  0.14,  0.00],
            "Race/Culture, and Sexuality":     [ 4.12,  5.42,  3.83,  2.53,  1.66,  1.37,  1.08,  0.51,  0.80,  0.58],
            "Intelligence":                    [ 3.04, 25.52, 29.28, 24.08, 17.72, 15.76, 13.96, 10.77,  8.75,  8.03],
            "Flooding":                        [ 0.14,  0.22,  0.72,  0.72,  1.30,  1.16,  2.31,  1.88,  1.66,  1.74],
        },
    },
}

VARIANTS = list(DATA.keys())
FT_TYPES = ["Non-FT", "FT"]  # Non-FT left, FT right
TURNS = np.arange(1, N_TURNS + 1)

# ── Helper: draw one subplot ─────────────────────────────────────────────────
def draw_panel(ax, cat_data, title):
    means = {cat: np.mean(vals) for cat, vals in cat_data.items()}
    top2  = sorted(means, key=lambda c: means[c], reverse=True)[:2]

    for cat in CATEGORIES:
        if cat not in cat_data:
            continue
        vals  = np.array(cat_data[cat])
        color = CAT_COLORS[CATEGORIES.index(cat)]
        is_top = cat in top2
        ax.plot(
            TURNS, vals,
            color=color,
            alpha=0.90 if is_top else 0.28,
            linewidth=2.0 if is_top else 0.9,
            marker="o", markersize=4.5 if is_top else 2.5,
            zorder=3 if is_top else 2,
        )

    # Regression line over top-2 combined points
    reg_x = np.tile(TURNS, 2).astype(float)
    reg_y = np.concatenate(
        [cat_data[top2[0]], cat_data[top2[1]]]
    ).astype(float)
    coef  = np.polyfit(reg_x, reg_y, 1)
    xline = np.linspace(1, N_TURNS, 300)
    ax.plot(xline, np.polyval(coef, xline),
            color='darkred', linestyle='--', linewidth=2.2, zorder=4)

    ax.set_xlim(0.7, N_TURNS + 0.3)
    ax.set_ylim(0, 100)
    ax.set_xticks(TURNS)
    ax.set_xticklabels([f"T{i}" for i in TURNS], fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xlabel("Turn (T1 → T10)", fontsize=9)
    ax.set_ylabel("Per-turn true rate (%)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=4)
    ax.grid(True, linewidth=0.35, alpha=0.45)


# ── Legend handles ───────────────────────────────────────────────────────────
legend_handles = []
for cat, color in zip(CATEGORIES, CAT_COLORS):
    legend_handles.append(
        mpatches.Patch(facecolor=color, edgecolor='none', label=CAT_LABELS[cat])
    )
legend_handles.append(
    plt.Line2D([0], [0], color='darkred', linestyle='--', linewidth=2.0,
               label='Regression Line\n(Insult, Flaming)')
)

# ── 1. Combined figure: 4 rows (variants) × 2 cols (Non-FT | FT) ─────────────
fig, axes = plt.subplots(
    nrows=len(VARIANTS), ncols=2,
    figsize=(14, 5 * len(VARIANTS)),
    constrained_layout=True
)

fig.suptitle("Qwen3-30B — Per-Turn True Rate by Variant and FT Type",
             fontsize=15, fontweight='bold', y=1.01)

for row_i, variant in enumerate(VARIANTS):
    for col_i, ft in enumerate(FT_TYPES):
        ax = axes[row_i][col_i]
        cat_data = DATA[variant][ft]
        title = f"{variant} — {ft}"
        draw_panel(ax, cat_data, title)

# Shared legend in the last cell's right side (or below figure)
axes[-1][-1].legend(
    handles=legend_handles,
    loc='lower right',
    ncol=1,
    fontsize=7.5,
    framealpha=0.88,
    title="Category",
    title_fontsize=8,
)

combined_path = os.path.join(OUT_DIR, "Qwen3_all_variants_combined.png")
fig.savefig(combined_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved combined: {combined_path}")


# ── 2. Individual plots per (variant, FT-type) ────────────────────────────────
for variant in VARIANTS:
    for ft in FT_TYPES:
        cat_data = DATA[variant][ft]
        fig_s, ax_s = plt.subplots(figsize=(8, 5))
        draw_panel(ax_s, cat_data, f"Qwen3-30B — {variant} — {ft}")
        ax_s.legend(
            handles=legend_handles,
            loc='upper right',
            ncol=1,
            fontsize=8,
            framealpha=0.88,
            title="Category",
            title_fontsize=8,
        )
        plt.tight_layout()
        fname = f"Qwen3_{variant}_{ft}.png".replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
        out   = os.path.join(OUT_DIR, fname)
        fig_s.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig_s)
        print(f"Saved: {out}")

print(f"\nAll done — outputs in {OUT_DIR}")
