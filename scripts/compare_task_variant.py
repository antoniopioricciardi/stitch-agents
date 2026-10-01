"""Step 1: does a task variant change what the expert does? Compare EE-delta actions per phase vs default.

Usage: uv run python scripts/compare_task_variant.py <variant> <robot>
Demos of both tasks use the same seeds, so episodes are compared pair by pair (same initial state).
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

VARIANT, ROBOT = sys.argv[1], sys.argv[2]
PHASES = ["approach", "descend", "grasp", "carry"]
DOMAIN = "cam0_look0_light0"
day = f"{date.today():%Y%m%d}"
A = np.load(Path("results") / f"{day}_step1_demos_default_{ROBOT}" / f"{DOMAIN}.npz")
B = np.load(Path("results") / f"{day}_step1_demos_{VARIANT}_{ROBOT}" / f"{DOMAIN}.npz")
OUT = Path("results") / f"{day}_step1_compare_{VARIANT}_{ROBOT}"
OUT.mkdir(parents=True, exist_ok=True)

seeds = np.intersect1d(A["seed"], B["seed"])
rows = {}
for p, name in enumerate(PHASES):
    a, b = A["action"][A["phase"] == p], B["action"][B["phase"] == p]  # (n, 7)
    # per seed: phase length and max |action difference| over the common steps
    dlen, dmax = [], []
    for s in seeds:
        sa, sb = A["action"][(A["seed"] == s) & (A["phase"] == p)], B["action"][(B["seed"] == s) & (B["phase"] == p)]
        n = min(len(sa), len(sb))
        dlen.append(len(sb) - len(sa))
        dmax.append(np.abs(sa[:n] - sb[:n]).max() if n else 0.0)
    rows[name] = dict(
        mean_action_default=a.mean(0).round(3).tolist(), mean_action_variant=b.mean(0).round(3).tolist(),
        mean_abs_diff_of_means=float(np.abs(a.mean(0) - b.mean(0)).mean()),
        mean_len_default=float(np.mean([((A["seed"] == s) & (A["phase"] == p)).sum() for s in seeds])),
        mean_len_diff=float(np.mean(dlen)), max_abs_action_diff_same_seed=float(np.max(dmax)),
        arm_abs_ratio=float(np.abs(b[:, :6]).mean() / np.abs(a[:, :6]).mean()),  # mean |a| variant / default, arm dims
    )
    print(f"{name:8s} len {rows[name]['mean_len_default']:5.1f} (Δ {rows[name]['mean_len_diff']:+.1f})  "
          f"|mean diff| {rows[name]['mean_abs_diff_of_means']:.4f}  arm |a| ratio {rows[name]['arm_abs_ratio']:.2f}  max same-seed |Δa| {rows[name]['max_abs_action_diff_same_seed']:.4f}")

# grasp timing: first step of the 'grasp' phase
grasp = {k: [int(D["t"][(D["seed"] == s) & (D["phase"] == 2)][0]) for s in seeds] for k, D in [("default", A), (VARIANT, B)]}
print("grasp start step, default:", grasp["default"])
print(f"grasp start step, {VARIANT}:", grasp[VARIANT])
json.dump(dict(variant=VARIANT, robot=ROBOT, domain=DOMAIN, seeds=seeds.tolist(), phases=rows, grasp_start=grasp),
          open(OUT / "metrics.json", "w"), indent=1)
