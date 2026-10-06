"""Step R closing summary: combination table (zero-shot, % of the affine ceiling and of the reference) and the
data-efficiency figure (map-only fine-tuning at the fixed 5k budget vs DP from scratch at 50k, % of the reference).
Seed pair s = look 2 encoder s -> controller s+1; reference = the oracle of the deployment domain at the encoder's
seed (last-3 success_once): goal look 2, xArm look 2, look 2.
Usage: uv run python scripts/stepR_summary.py   (from the repo root)
Writes results/<date>_stepR_summary/{metrics.json, table.md, data_efficiency.png}.
"""
import json
import re
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R, B = Path("results"), Path("/home/ricc/projects/labelstitch-1b/results")
NS = (5, 10, 25)
MAPS = ["identity", "affine", "saps", "nn_orth", "nn_affine", "nn_affine_pca16"]


def curve(run):
    # success_once of every evaluation in a training log, in order
    return [float(x) for x in re.findall(r"success_once: ([0-9.]+)", open(run / "log.txt", errors="ignore").read())]


def last3(run):
    return float(np.mean(curve(run)[-3:]))


one = lambda pat: next(R.glob(pat))
REF = {"goal": {1: last3(one("202610??_stepR_dp_ref_goal_look2_s1")), 2: last3(B / "20261003_step1b_dp_ref_goal_look2_s2"),
                3: last3(one("202610??_stepR_dp_ref_goal_look2_s3"))},
       "emb": {1: last3(one("202610??_stepR_dp_ref_xarm_look2_s1")), 2: last3(B / "20261003_step1b_dp_ref_xarm_look2_s2"),
               3: last3(one("202610??_stepR_dp_ref_xarm_look2_s3"))},
       "same": {s: last3(B / p) for s, p in ((1, "20261003_step1b_dp_ref_look2_s1"), (2, "20261002_step1b_dp_ref_look2_s2"),
                                             (3, "20261003_step1b_dp_ref_look2_s3"))}}
ms = lambda v: f"{np.mean(v):.0f} ± {np.std(v, ddof=1):.0f}%"

# zero-shot combinations
zs, lines = {}, ["| combination | aligner | % of affine ceiling (mean ± std) | per pair | % of reference |", "|---|---|---|---|---|"]
for k, dom in (("goal", "goal_look2"), ("emb", "xarm_look2"), ("same", "look2")):
    A = {p: next(iter(json.load(open(one(f"202610??_step1b_stitch_stepR_{k}_p{p}_{dom}") / "metrics.json"))["stitch"].values()))["aligners"]
         for p in (1, 2, 3)}
    zs[k] = {a: [A[p][a]["success_once"] for p in (1, 2, 3)] for a in MAPS if a in A[1]}
    for a, v in zs[k].items():
        ceil = [x / c * 100 for x, c in zip(v, zs[k]["affine"])]
        ref = [x / REF[k][p] * 100 for x, p in zip(v, (1, 2, 3))]
        lines.append(f"| {k} | {a} | {'—' if a == 'affine' else ms(ceil)} | {' / '.join(f'{x:.3f}' for x in v)} | {ms(ref)} |")

# data efficiency: map-only fine-tuning (5k, fixed budget) and DP from scratch (final 50k), % of the reference
de = {}
for k, dom in (("goal", "goal_look2"), ("same", "look2")):
    ft = {n: [curve(one(f"202610??_stepR_ft_map_{k}_p{p}_n{n}"))[5] / REF[k][p] * 100 for p in (1, 2, 3)] for n in NS}
    start = {n: [curve(one(f"202610??_stepR_ft_map_{k}_p{p}_n{n}"))[0] / REF[k][p] * 100 for p in (1, 2, 3)] for n in NS}
    sc = {n: [curve(one(f"202610??_stepR_scratch_{dom}_n{n}_s{s}"))[-1] / REF[k][s] * 100 for s in (1, 2, 3)] for n in NS}
    de[k] = dict(map_only_5k=ft, start=start, scratch_50k=sc)
lines += ["", "| combination | N | stitched start | map-only fine-tuning, 5k | DP from scratch, 50k |", "|---|---|---|---|---|"]
for k in de:
    for n in NS:
        lines.append(f"| {k} | {n} | {ms(de[k]['start'][n])} | {ms(de[k]['map_only_5k'][n])} | {ms(de[k]['scratch_50k'][n])} |")

out = R / f"{date.today():%Y%m%d}_stepR_summary"
out.mkdir(parents=True, exist_ok=True)
(out / "table.md").write_text("\n".join(lines) + "\n")
json.dump(dict(reference=REF, zero_shot=zs, data_efficiency={k: {a: {str(n): v for n, v in d.items()} for a, d in x.items()}
                                                               for k, x in de.items()}), open(out / "metrics.json", "w"), indent=1)
print("\n".join(lines))

# figure: one panel, % of the reference vs N (log x); colour = arm, line style = combination (secondary encoding)
C = {"map": "#2a78d6", "scratch": "#eb6834"}  # reference palette slots 1, 2
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=200)
for k, ls, mk, name in (("goal", "-", "o", "goal shift"), ("same", "--", "s", "same task")):
    for arm, key, lab in (("map", "map_only_5k", "map-only fine-tuning (5k)"), ("scratch", "scratch_50k", "DP from scratch (50k)")):
        m = [np.mean(de[k][key][n]) for n in NS]
        s = [np.std(de[k][key][n], ddof=1) for n in NS]
        ax.errorbar(NS, m, yerr=s, color=C[arm], ls=ls, marker=mk, ms=4, lw=1.6, capsize=2, elinewidth=1, label=f"{lab}, {name}")
ax.set_xscale("log")
ax.set_xticks(NS, [str(n) for n in NS])
ax.minorticks_off()
ax.set_xlabel("target demos N (+ 10 validation demos)")
ax.set_ylabel("success, % of the target oracle")
ax.set_ylim(0, 100)
ax.grid(axis="y", color="#e4e3df", lw=0.6)
ax.legend(frameon=False, fontsize=7, loc="upper left")
ax.set_title("Low-data adaptation, 3 seed pairs (mean ± std)", fontsize=9, loc="left")
fig.tight_layout()
fig.savefig(out / "data_efficiency.png")
print(out / "data_efficiency.png")
