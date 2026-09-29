# Anchor recipe comparison, in-game (Nature CNN): random same-action pairs, old-repo recipe (1000-frame
# pool) and full training set. Same stitches, seeds and episodes as Step 0b (v1 pairs) and cross-level,
# whose centroid results are reused from results/20260929_step0b_stitch_play and _stitch_levels_nature.
# Run: OMP_NUM_THREADS=1 uv run --project mario python mario/anchors_play.py <versions|levels> [methods] [recipes]
import itertools
import json
import sys

import numpy as np
import torch

from agents import DEVICE, ROOT, SIZE, embed, game_policy, inputs, load_model
from align import fit_action_pairs
from data import EPISODES, TRAIN_EPS
from evaluate import rollout

SETTING = sys.argv[1]
ARCH = "nature"
METHODS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["bc", "scil"]
SEEDS = [0, 1, 2]
RECIPES = {"pairs_pool1000": 1000, "pairs_full": None}  # pool size per side
if len(sys.argv) > 3:
    RECIPES = {k: RECIPES[k] for k in sys.argv[3].split(",")}
EVAL_EPISODES = 10
SRC_EPS, TGT_EPS = TRAIN_EPS[:5], TRAIN_EPS[5:]  # versions: disjoint episodes, as in anchors_offline.py
OUT = ROOT / (f"20260929_anchors_play_{SETTING}" + ("" if len(sys.argv) <= 2 else "_" + "_".join(METHODS)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if SETTING == "versions":  # (level, version) of encoder / controller domains
        stitches = [(("1-1", u), ("1-1", v)) for u, v in [(0, 1), (1, 0), (1, 2), (2, 1)]]
    else:
        stitches = [(("1-1", 0), ("1-2", 0)), (("1-2", 0), ("1-1", 0))]
    domains = sorted({d for s in stitches for d in s})
    data = {}
    for L, v in domains:
        X, y, e = inputs(ARCH, L, v, EPISODES[L][0])
        data[L, v] = (X, y.numpy(), e.numpy())

    rows = []
    for method in METHODS:
        models = {(d, s): load_model(ARCH, method, s, *d, ) for d in domains for s in SEEDS}
        Z = {k: embed(ARCH, m[0], data[k[0]][0]) for k, m in models.items()}
        for seed, (da, db) in itertools.product(SEEDS, stitches):
            cs = (seed + 1) % len(SEEDS)
            Za, ya, ea = Z[da, seed], *data[da][1:]
            Zb, yb, eb = Z[db, cs], *data[db][1:]
            if SETTING == "versions":  # same frames exist in both versions: keep the anchor sides disjoint
                Za, ya = Za[np.isin(ea, SRC_EPS)], ya[np.isin(ea, SRC_EPS)]
                Zb, yb = Zb[np.isin(eb, TGT_EPS)], yb[np.isin(eb, TGT_EPS)]
            enc, ctrl = models[da, seed][0], models[db, cs][1]
            for name, pool in RECIPES.items():
                R, b = fit_action_pairs(Za, Zb, ya, yb, np.random.default_rng(1000 * seed), pool=pool)
                R_t, b_t = torch.from_numpy(R).float().to(DEVICE), torch.from_numpy(b).float().to(DEVICE)
                mx, fl = rollout(game_policy(ARCH, enc, ctrl, R_t, b_t), da[1], EVAL_EPISODES, seed, DEVICE, da[0], SIZE[ARCH])
                rows.append({"method": method, "seed": seed, "enc": f"{da[0]} v{da[1]}", "ctrl": f"{db[0]} v{db[1]}",
                             "aligner": name, "max_x": float(mx.mean()), "flag_rate": float(fl.mean())})
                print(f"{method} s{seed} enc {da} ctrl {db} {name}: max x {mx.mean():.0f} flags {fl.mean():.0%}", flush=True)
    (OUT / "rows.json").write_text(json.dumps(rows, indent=2))
    (OUT / "config.json").write_text(json.dumps({"SETTING": SETTING, "ARCH": ARCH, "SEEDS": SEEDS, "RECIPES": RECIPES,
                                                  "EVAL_EPISODES": EVAL_EPISODES}, indent=2))


if __name__ == "__main__":
    main()
