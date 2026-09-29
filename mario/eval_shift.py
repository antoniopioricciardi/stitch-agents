# Step 0 baseline: agents run unchanged under visual shift (no stitching, no alignment).
# Every agent trained in v_u (own encoder + own controller) is evaluated in every v_w:
# offline on held-out paired frames and in-game.
# Run: OMP_NUM_THREADS=1 uv run --project mario python mario/eval_shift.py [arch]
import itertools
import json
import sys

import numpy as np
import torch

from agents import DEVICE, ROOT, SIZE, embed, game_policy, inputs, load_model
from data import VAL_EPS
from evaluate import rollout

ARCH = sys.argv[1] if len(sys.argv) > 1 else "nature"
LEVEL = "1-1"
VERSIONS = [0, 1, 2]
METHODS = ["bc", "scil"]
SEEDS = [0, 1, 2]
EVAL_EPISODES = 10
OUT = ROOT / ("20260929_step0_shift_nostitch_e50" if ARCH == "nature" else f"20260929_shift_nostitch_{ARCH}")


@torch.no_grad()
def predict(enc, ctrl, X):
    return ctrl(torch.from_numpy(embed(ARCH, enc, X)).float().to(DEVICE)).argmax(1).cpu().numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    val = {w: inputs(ARCH, LEVEL, w, VAL_EPS) for w in VERSIONS}  # same frames, rendered per version
    y_va = val[0][1].numpy()

    rows = []
    for method, seed in itertools.product(METHODS, SEEDS):
        models = {u: load_model(ARCH, method, seed, LEVEL, u) for u in VERSIONS}
        # native action of v_w's own agent on its own frames, for the agreement metric
        native = {w: predict(*models[w], val[w][0]) for w in VERSIONS}
        for u, w in itertools.product(VERSIONS, VERSIONS):
            enc, ctrl = models[u]
            pred = predict(enc, ctrl, val[w][0])
            max_x, flags = rollout(game_policy(ARCH, enc, ctrl), w, EVAL_EPISODES, seed, DEVICE, LEVEL, SIZE[ARCH])
            bal = float(np.mean([(pred[y_va == c] == c).mean() for c in np.unique(y_va)]))
            rows.append({"method": method, "seed": seed, "train": u, "test": w, "bal_acc": bal,
                         "agree_native": float((pred == native[w]).mean()),
                         "max_x": float(max_x.mean()), "flag_rate": float(flags.mean())})
            print(f"{method} s{seed} trained v{u} -> tested v{w}: bal acc {bal:.3f}, "
                  f"agree {rows[-1]['agree_native']:.3f}, max x {max_x.mean():.0f}, flags {flags.mean():.0%}", flush=True)

    summary = {}
    for method in METHODS:
        for key in ["bal_acc", "agree_native", "max_x", "flag_rate"]:
            M = np.array([[np.mean([r[key] for r in rows if r["method"] == method and r["train"] == u and r["test"] == w])
                           for w in VERSIONS] for u in VERSIONS])  # (train, test), mean over seeds
            summary[f"{method}/{key}"] = M.tolist()
            print(f"\n{method.upper()} {key} (rows: trained in, cols: tested in v0 v1 v2)")
            for u in VERSIONS:
                print(f"  v{u}: " + "  ".join(f"{M[u, w]:8.3f}" for w in VERSIONS))
    (OUT / "rows.json").write_text(json.dumps(rows, indent=2))
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    (OUT / "config.json").write_text(json.dumps({"ARCH": ARCH, "VERSIONS": VERSIONS, "METHODS": METHODS, "SEEDS": SEEDS,
                                                  "EVAL_EPISODES": EVAL_EPISODES, "VAL_EPS": VAL_EPS}, indent=2))


if __name__ == "__main__":
    main()
