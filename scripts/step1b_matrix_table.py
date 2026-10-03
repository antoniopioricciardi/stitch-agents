"""Step 1b main matrix: % of the affine paired ceiling per shift x seed pair x direction x aligner, and the default rule.

Usage: uv run python scripts/step1b_matrix_table.py
Seed pairs (cam0 s -> domain s+1, wrapping): p1 = cam0 s1 <-> s2 (the item 1 pilot), p2 = s2 <-> s3, p3 = s3 <-> s1.
p1 for look 1 is assembled from the runs that measured it (row 1, map-class, k16_half, label-map runs); every other
cell comes from one run. % of ceiling = stitched success_once / affine paired success_once of the same pair and
direction. Default-aligner rule (fixed before the run): the higher mean over all cells; nn_orth if within .03 of
nn_affine. Writes results/<date>_step1b_matrix/{metrics.json, table.md}.
"""
import json
from datetime import date
from pathlib import Path

import numpy as np

R = Path("results")
SHIFTS = ["look1", "look2", "look2light1", "cam3"]
ALIGNERS = ["identity", "affine", "saps", "k16_half", "nn_orth", "nn_affine"]
DIRS = ["fwd", "bwd"]  # fwd = cam0 enc -> domain ctrl (plays cam0), bwd = domain enc -> cam0 ctrl (plays the domain)


def load(name):
    return json.load(open(R / name / "metrics.json"))["stitch"]


def cells(stitch, d, keep=ALIGNERS):
    # -> {direction: {aligner: success_once}}
    return {"fwd": {a: v["success_once"] for a, v in stitch[f"cam0_enc_to_{d}_ctrl"]["aligners"].items() if a in keep},
            "bwd": {a: v["success_once"] for a, v in stitch[f"{d}_enc_to_cam0_ctrl"]["aligners"].items() if a in keep}}


def merge(*parts):
    out = {"fwd": {}, "bwd": {}}
    for p in parts:
        for k in DIRS:
            out[k].update(p[k])
    return out


S = {}  # S[shift][pair] = {direction: {aligner: success}}
for d in SHIFTS:
    S[d] = {}
    if d == "look1":
        S[d]["p1"] = merge(cells(load("20261001_step1b_stitch_row1_look1"), d, ["identity"]),
                           cells(load("20261001_step1b_stitch_row1mapclass_look1"), d, ["affine", "saps"]),
                           cells(load("20261002_step1b_stitch_pilot_k16half_look1"), d, ["k16_half"]),
                           cells(load("20261001_step1b_stitch_row1labelmaps_look1"), d, ["nn_orth", "nn_affine"]))
    else:
        S[d]["p1"] = cells(load(f"20261002_step1b_stitch_pilot_{d}"), d)
    for p in ("p2", "p3"):
        S[d][p] = cells(load(f"20261003_step1b_stitch_matrix_{p}_{d}"), d)

pct = {}  # pct[aligner] = list of % of ceiling over all shift x pair x direction cells
rows = ["| shift | dir | aligner | p1 | p2 | p3 | mean ± std (% of ceiling) |", "|---|---|---|---|---|---|---|"]
for d in SHIFTS:
    for k in DIRS:
        for a in ALIGNERS:
            succ = [S[d][p][k][a] for p in ("p1", "p2", "p3")]
            ceil = [S[d][p][k]["affine"] for p in ("p1", "p2", "p3")]
            r = [s / c for s, c in zip(succ, ceil)]
            if a != "affine":
                pct.setdefault(a, []).extend(r)
            rows.append(f"| {d} | {k} | {a} | " + " | ".join(f"{s:.3f} ({x:.0%})" for s, x in zip(succ, r))
                        + f" | {np.mean(r):.0%} ± {np.std(r):.0%} |")
summary = {a: dict(mean=float(np.mean(v)), std=float(np.std(v)), n=len(v)) for a, v in pct.items()}
best = max(["nn_affine", "nn_orth"], key=lambda a: summary[a]["mean"])
default = "nn_orth" if summary["nn_affine"]["mean"] - summary["nn_orth"]["mean"] <= 0.03 else best
rows += ["", "| aligner | mean % of ceiling over 24 cells (4 shifts × 3 pairs × 2 directions) |", "|---|---|"]
rows += [f"| {a} | {v['mean']:.0%} ± {v['std']:.0%} |" for a, v in summary.items()]
rows += ["", f"Default-aligner rule: nn_affine {summary['nn_affine']['mean']:.1%} vs nn_orth {summary['nn_orth']['mean']:.1%} -> **{default}**"]

OUT = R / f"{date.today():%Y%m%d}_step1b_matrix"
OUT.mkdir(parents=True, exist_ok=True)
json.dump(dict(success_once=S, pct_of_ceiling_summary=summary, default_aligner=default), open(OUT / "metrics.json", "w"), indent=1)
json.dump(dict(pairs={"p1": "cam0 s1 <-> s2", "p2": "cam0 s2 <-> s3", "p3": "cam0 s3 <-> s1"},
               sources="see the script docstring"), open(OUT / "config.json", "w"), indent=1)
open(OUT / "table.md", "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
