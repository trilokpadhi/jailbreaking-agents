"""
Generate per-turn escalation figures (T1–T10) for Gemini-FT vs Gemini (Non-FT),
one combined PNG (4 rows × 2 cols) plus 8 individual PNGs.
Output directory: playground/turn_analysis_gemini/
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

N_TURNS = 10
OUT_DIR = "/staging/users/tpadhi1/agent-jail-breaking/playground/turn_analysis_gemini"
os.makedirs(OUT_DIR, exist_ok=True)

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
            "Flaming":                         [21.87,  6.41, 27.99, 11.88, 28.75, 11.68, 27.04, 11.95, 21.89,  9.90],
            "Threat/Blackmail":                [ 0.81,  0.76,  2.02,  1.92,  5.27,  3.60,  5.80,  4.26,  6.49,  3.48],
            "Insult":                          [36.72,  8.18, 46.49, 15.02, 42.04, 16.15, 33.48, 14.72, 27.21, 12.00],
            "Curse/Exclusion":                 [ 5.56,  1.97, 13.24,  4.10, 14.10,  4.48, 14.16,  5.09, 11.81,  4.62],
            "Sexual Harassment / Sexual Talk": [ 0.35,  0.10,  0.35,  0.00,  0.35,  0.00,  0.27,  0.06,  0.12,  0.00],
            "Race/Culture, and Sexuality":     [ 7.12,  1.41,  9.80,  1.42,  6.39,  1.23,  3.73,  1.16,  2.90,  0.72],
            "Intelligence":                    [ 3.89,  1.31,  8.64,  2.17,  7.40,  1.80,  5.38,  2.16,  4.52,  1.02],
            "Flooding":                        [ 0.05,  0.00,  0.10,  0.05,  0.00,  0.00,  0.21,  0.00,  0.17,  0.00],
        },
        "Non-FT": {
            "Flaming":                         [35.37,  2.83, 58.15,  8.55, 61.34, 13.31, 63.56, 14.02, 65.59, 14.22],
            "Threat/Blackmail":                [ 1.27,  0.66,  4.30,  1.67,  7.79,  4.10, 10.12,  4.81, 11.74,  5.72],
            "Insult":                          [48.03,  2.78, 70.75,  9.62, 73.48, 15.54, 68.37, 18.02, 68.93, 20.09],
            "Curse/Exclusion":                 [ 5.92,  1.01, 18.67,  1.32, 21.96,  3.14, 23.03,  3.59, 25.56,  5.01],
            "Sexual Harassment / Sexual Talk": [ 0.56,  0.00,  0.81,  0.00,  0.66,  0.10,  0.76,  0.20,  0.40,  0.10],
            "Race/Culture, and Sexuality":     [ 7.69,  0.35, 11.84,  0.56, 10.58,  1.16,  6.73,  1.11,  6.68,  0.76],
            "Intelligence":                    [ 3.14,  0.20, 10.98,  0.71, 10.73,  0.56, 10.83,  1.32, 10.63,  0.96],
            "Flooding":                        [ 0.05,  0.00,  0.05,  0.05,  0.05,  0.00,  0.51,  0.05,  0.20,  0.15],
        },
    },
    "Baseline (No Memory)": {
        "FT": {
            "Flaming":                         [34.47, 11.81, 45.38, 19.40, 45.88, 21.22, 37.42, 22.51, 29.11, 17.26],
            "Threat/Blackmail":                [ 0.95,  0.70,  4.57,  2.51, 10.15,  6.28, 13.13,  7.70, 14.18,  6.98],
            "Insult":                          [48.09, 11.11, 56.88, 18.84, 51.06, 21.07, 41.40, 21.80, 29.06, 17.46],
            "Curse/Exclusion":                 [ 6.23,  2.46, 16.13,  4.37, 16.48,  5.98, 15.44,  6.70, 11.76,  6.98],
            "Sexual Harassment / Sexual Talk": [ 0.55,  0.10,  0.50,  0.00,  0.60,  0.00,  0.25,  0.10,  0.15,  0.05],
            "Race/Culture, and Sexuality":     [ 8.34,  1.41,  8.69,  1.51,  4.32,  1.36,  3.72,  0.76,  2.07,  0.81],
            "Intelligence":                    [ 3.47,  0.60,  3.87,  0.95,  3.02,  0.96,  2.41,  0.65,  1.41,  0.36],
            "Flooding":                        [ 0.05,  0.00,  0.00,  0.00,  0.00,  0.00,  0.05,  0.00,  0.00,  0.05],
        },
        "Non-FT": {
            "Flaming":                         [ 5.61,  1.10,  8.39,  1.10,  7.70,  0.47,  6.60,  0.52,  5.40,  0.58],
            "Threat/Blackmail":                [ 0.16,  0.89,  2.78,  1.57,  3.41,  1.68,  2.83,  1.42,  2.62,  1.31],
            "Insult":                          [ 8.44,  3.14, 12.89,  3.35, 10.95,  2.20,  9.80,  1.89,  7.02,  1.31],
            "Curse/Exclusion":                 [ 1.52,  0.79,  5.56,  1.99,  6.18,  1.99,  7.23,  1.52,  6.66,  1.10],
            "Sexual Harassment / Sexual Talk": [ 0.21,  0.00,  0.10,  0.05,  0.21,  0.05,  0.10,  0.00,  0.05,  0.00],
            "Race/Culture, and Sexuality":     [ 1.94,  1.00,  1.68,  1.00,  1.26,  0.42,  1.05,  0.37,  0.79,  0.16],
            "Intelligence":                    [ 1.42,  1.05,  3.98,  0.94,  4.09,  0.79,  3.09,  0.37,  2.20,  0.26],
            "Flooding":                        [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
        },
    },
    "CoT (No Memory)": {
        "FT": {
            "Flaming":                         [34.84,  9.95, 46.61, 17.50, 45.60, 20.41, 38.61, 21.42, 33.27, 19.55],
            "Threat/Blackmail":                [ 1.06,  0.50,  4.78,  2.61,  8.70,  4.12, 11.56,  5.63, 12.98,  5.86],
            "Insult":                          [47.91, 10.26, 56.11, 14.33, 49.67, 18.10, 40.98, 21.17, 33.32, 19.49],
            "Curse/Exclusion":                 [ 6.08,  1.26, 13.62,  2.97, 15.99,  4.42, 14.58,  6.03, 13.03,  7.07],
            "Sexual Harassment / Sexual Talk": [ 0.60,  0.05,  0.25,  0.05,  0.65,  0.20,  0.10,  0.05,  0.45,  0.00],
            "Race/Culture, and Sexuality":     [ 8.25,  1.51,  7.94,  1.66,  6.18,  1.01,  2.97,  0.85,  2.06,  1.01],
            "Intelligence":                    [ 2.87,  0.65,  3.72,  0.96,  4.22,  0.90,  2.21,  0.75,  1.21,  0.76],
            "Flooding":                        [ 0.05,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.05],
        },
        "Non-FT": {
            "Flaming":                         [ 7.09, 12.28, 12.99,  6.32,  5.03,  3.29,  3.49,  1.69,  1.44,  0.93],
            "Threat/Blackmail":                [ 1.85, 11.20, 11.76,  9.50,  7.14,  4.83,  3.90,  3.18,  2.05,  2.01],
            "Insult":                          [24.19, 34.98, 30.92, 24.81, 17.93, 11.71,  8.27,  5.91,  4.93,  3.82],
            "Curse/Exclusion":                 [ 1.69,  2.21,  5.96,  6.11,  6.11,  4.06,  3.95,  2.11,  2.11,  1.03],
            "Sexual Harassment / Sexual Talk": [ 0.36,  0.31,  0.62,  0.26,  0.21,  0.21,  0.15,  0.10,  0.05,  0.05],
            "Race/Culture, and Sexuality":     [ 4.37,  4.73,  4.73,  2.77,  2.11,  1.23,  1.03,  0.51,  0.56,  0.31],
            "Intelligence":                    [ 5.86, 16.44, 14.12, 11.20,  6.11,  4.26,  2.72,  2.47,  1.34,  0.98],
            "Flooding":                        [ 0.00,  0.00,  0.00,  0.00,  0.05,  0.00,  0.00,  0.21,  0.21,  0.15],
        },
    },
    "ReACT (No Memory)": {
        "FT": {
            "Flaming":                         [35.07,  9.23, 44.31, 17.36, 42.35, 19.22, 37.23, 19.57, 28.43, 15.50],
            "Threat/Blackmail":                [ 1.10,  0.60,  5.27,  2.71, 10.04,  5.77, 10.59,  6.77, 11.05,  6.64],
            "Insult":                          [48.57,  9.33, 54.44, 14.45, 45.21, 18.06, 39.19, 18.87, 30.84, 15.00],
            "Curse/Exclusion":                 [ 5.92,  1.25, 14.35,  3.61, 15.05,  4.37, 13.85,  5.92, 12.10,  5.99],
            "Sexual Harassment / Sexual Talk": [ 0.60,  0.00,  0.60,  0.05,  0.25,  0.00,  0.40,  0.05,  0.15,  0.05],
            "Race/Culture, and Sexuality":     [ 8.53,  0.80,  6.82,  0.70,  4.01,  1.10,  2.86,  0.75,  1.91,  0.40],
            "Intelligence":                    [ 3.01,  0.55,  2.81,  0.55,  3.11,  0.55,  1.51,  0.50,  1.71,  0.86],
            "Flooding":                        [ 0.05,  0.00,  0.00,  0.00,  0.00,  0.05,  0.00,  0.00,  0.05,  0.00],
        },
        "Non-FT": {
            "Flaming":                         [33.56,  1.73, 45.64,  5.41, 45.43,  5.09, 40.07,  3.83, 27.26,  2.31],
            "Threat/Blackmail":                [ 0.95,  1.37, 13.08,  8.40, 16.75, 12.39, 20.17, 13.34, 16.44,  9.45],
            "Insult":                          [47.79, 12.71, 69.07, 22.27, 69.49, 21.11, 64.13, 17.54, 51.94, 11.87],
            "Curse/Exclusion":                 [ 5.51,  0.37, 15.34,  1.58, 21.69,  3.20, 29.62,  3.99, 39.13,  2.31],
            "Sexual Harassment / Sexual Talk": [ 0.63,  0.16,  1.52,  0.26,  1.05,  0.42,  0.47,  0.16,  0.32,  0.05],
            "Race/Culture, and Sexuality":     [ 8.35,  2.00, 11.61,  2.84,  9.93,  2.26,  8.40,  2.05,  6.41,  1.21],
            "Intelligence":                    [ 3.26,  0.89, 32.04,  4.67, 33.61,  3.99, 28.89,  2.68, 21.64,  2.21],
            "Flooding":                        [ 0.05,  0.00,  0.21,  0.00,  0.42,  0.00,  0.63,  0.00,  0.89,  0.00],
        },
    },
}

VARIANTS  = list(DATA.keys())
FT_TYPES  = ["Non-FT", "FT"]
TURNS     = np.arange(1, N_TURNS + 1)


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


legend_handles = []
for cat, color in zip(CATEGORIES, CAT_COLORS):
    legend_handles.append(
        mpatches.Patch(facecolor=color, edgecolor='none', label=CAT_LABELS[cat])
    )
legend_handles.append(
    plt.Line2D([0], [0], color='darkred', linestyle='--', linewidth=2.0,
               label='Regression Line\n(top-2 categories)')
)

# ── Combined figure: 4 rows × 2 cols ─────────────────────────────────────────
fig, axes = plt.subplots(
    nrows=len(VARIANTS), ncols=2,
    figsize=(14, 5 * len(VARIANTS)),
    constrained_layout=True
)

fig.suptitle("Gemini-2.0-flash-001 — Per-Turn True Rate by Variant and FT Type",
             fontsize=15, fontweight='bold', y=1.01)

for row_i, variant in enumerate(VARIANTS):
    for col_i, ft in enumerate(FT_TYPES):
        ax = axes[row_i][col_i]
        draw_panel(ax, DATA[variant][ft], f"{variant} — {ft}")

axes[-1][-1].legend(
    handles=legend_handles,
    loc='lower right',
    ncol=1,
    fontsize=7.5,
    framealpha=0.88,
    title="Category",
    title_fontsize=8,
)

combined_path = os.path.join(OUT_DIR, "Gemini_all_variants_combined.png")
fig.savefig(combined_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved combined: {combined_path}")

# ── Individual plots ──────────────────────────────────────────────────────────
for variant in VARIANTS:
    for ft in FT_TYPES:
        fig_s, ax_s = plt.subplots(figsize=(8, 5))
        draw_panel(ax_s, DATA[variant][ft], f"Gemini-2.0-flash-001 — {variant} — {ft}")
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
        fname = (f"Gemini_{variant}_{ft}.png"
                 .replace(" ", "_").replace("/", "-")
                 .replace("(", "").replace(")", ""))
        out = os.path.join(OUT_DIR, fname)
        fig_s.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig_s)
        print(f"Saved: {out}")

print(f"\nAll done — outputs in {OUT_DIR}")
