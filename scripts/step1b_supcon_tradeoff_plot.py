"""Step 1b: the SupCon tradeoff figure (x = lambda; native success, and % of the affine paired ceiling).

Usage: uv run python scripts/step1b_supcon_tradeoff_plot.py
Numbers copied from EXPERIMENTS.md (Step 1b: row 1, row 2 at lambda = 1 / 0.1, item 2 at lambda = 0.01, map-class check):
native = success_once last-3 mean; % of ceiling = stitched success_once / affine-paired success_once, cam0 <-> look 1,
both directions (mean marked by the line, directions by the dots). At lambda >= 0.1 the natives are at blind level, so
the affine ceiling was not measured and % of ceiling is undefined (not plotted).
Two panels sharing x, not a dual axis.
"""
import json
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt

LAMBDAS = ["0", "0.01", "0.1", "1"]
NATIVE = {"cam0 (s1)": [0.788, 0.300, 0.051, 0.051], "look 1 (s2)": [0.695, 0.256, 0.072, 0.037]}
# % of ceiling per direction: (cam0 enc -> look 1 ctrl, look 1 enc -> cam0 ctrl); None = not measured
CEIL = {"SAPS (paired)": [(0.63, 0.67), (1.00, 0.92), None, None],
        "nn_affine (label-only)": [(0.64, 0.70), (0.84, 0.80), None, None]}
# reference palette slots (dataviz references/palette.md), plus marker shapes as secondary encoding
COLORS = {"cam0 (s1)": "#2a78d6", "look 1 (s2)": "#eb6834", "SAPS (paired)": "#1baf7a", "nn_affine (label-only)": "#e87ba4"}
MARKERS = {"cam0 (s1)": "o", "look 1 (s2)": "s", "SAPS (paired)": "D", "nn_affine (label-only)": "^"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

OUT = Path("results") / f"{date.today():%Y%m%d}_step1b_supcon_tradeoff"
OUT.mkdir(parents=True, exist_ok=True)
x = range(len(LAMBDAS))
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True, facecolor="#fcfcfb")
for ax in (a1, a2):
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=MUTED)

for name, ys in NATIVE.items():
    a1.plot(x, ys, color=COLORS[name], lw=2, marker=MARKERS[name], ms=8, mec="#fcfcfb", mew=2, label=name)
    a1.annotate(name, (0, ys[0]), xytext=(-8, 0), textcoords="offset points", ha="right", va="center", color=INK, fontsize=9)
a1.set_ylim(0, 1)
a1.set_ylabel("native success (last-3)", color=INK)
a1.set_title("SupCon on 16 chunk clusters: native success falls ...", loc="left", color=INK, fontsize=11)
a1.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper right")

for k, (name, vals) in enumerate(CEIL.items()):
    xs = [i - 0.06 + 0.12 * k for i, v in enumerate(vals) if v is not None]  # small horizontal dodge: series overlap at 0
    vals = [v for v in vals if v is not None]
    means = [sum(v) / 2 for v in vals]
    a2.plot(xs, means, color=COLORS[name], lw=2, marker=MARKERS[name], ms=8, mec="#fcfcfb", mew=2, label=name)
    for xi, v in zip(xs, vals):
        a2.scatter([xi, xi], v, color=COLORS[name], s=18, alpha=0.6, zorder=3)
    a2.annotate(name, (xs[-1], means[-1]), xytext=(10, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
a2.axvspan(1.5, 3.5, color=GRID, alpha=0.6, lw=0)
a2.text(2.5, 0.5, "natives at blind level:\n% of ceiling undefined", ha="center", va="center", color=MUTED, fontsize=9)
a2.set_ylim(0, 1.1)
a2.set_ylabel("% of affine paired ceiling", color=INK)
a2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
a2.set_title("... while relative alignability rises (cam0 ↔ look 1; dots = directions)", loc="left", color=INK, fontsize=11)
a2.legend(frameon=False, labelcolor=INK, fontsize=9, loc="lower left")
a2.set_xticks(list(x), LAMBDAS)
a2.set_xlim(-0.9, 3.5)
a2.set_xlabel("SupCon weight λ", color=INK)
fig.tight_layout()
fig.savefig(OUT / "tradeoff.png", dpi=160)
json.dump(dict(lambdas=LAMBDAS, native_last3=NATIVE, pct_of_affine_ceiling=CEIL), open(OUT / "metrics.json", "w"), indent=1)
json.dump(dict(source="EXPERIMENTS.md Step 1b (row 1, row 2 lambda 1 / 0.1, item 2 lambda 0.01, map-class check)",
               pair="cam0 s1 <-> look 1 s2"), open(OUT / "config.json", "w"), indent=1)
