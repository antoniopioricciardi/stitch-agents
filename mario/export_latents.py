# Export Nature CNN latents once, so alignment can run in the root env without the Mario stack (Step 4).
# One .npz per (method, seed, level, version): training and held-out latents, action classes, episode ids,
# Mario's x-position per frame, and the controller (fc2) weights, so stitched actions are argmax(z_mapped @ W.T + b) in plain numpy.
# Latents are stored as float32: embed() computes in float32 and only casts to float64 at the end, so no
# precision is lost. Rows are aligned across versions of the same level (replays of the same episodes).
# Run: CUDA_VISIBLE_DEVICES="" uv run --project mario python mario/export_latents.py
import json

import numpy as np

from agents import ROOT, embed, inputs, load_model
from data import EPISODES, load_episodes

METHODS = ["bc", "scil", "scil_taco3"]
SEEDS = [0, 1, 2]
DOMAINS = [("1-1", 0), ("1-1", 1), ("1-1", 2), ("1-2", 0)]
OUT = ROOT / "20260929_step4_latents"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for level, version in DOMAINS:
        train_eps, heldout_eps = EPISODES[level]
        X_tr, y_tr, ep_tr = inputs("nature", level, version, train_eps)
        X_va, y_va, ep_va = inputs("nature", level, version, heldout_eps)
        x_tr, x_va = load_episodes(train_eps, version, level)[3].numpy(), load_episodes(heldout_eps, version, level)[3].numpy()
        for method in METHODS:
            for seed in SEEDS:
                enc, ctrl = load_model("nature", method, seed, level, version)
                np.savez(OUT / f"{method}_s{seed}_{level}_v{version}.npz",
                         Z_tr=embed("nature", enc, X_tr).astype(np.float32), y_tr=y_tr.numpy(), ep_tr=ep_tr.numpy(), x_tr=x_tr,
                         Z_va=embed("nature", enc, X_va).astype(np.float32), y_va=y_va.numpy(), ep_va=ep_va.numpy(), x_va=x_va,
                         W=ctrl.fc2.weight.detach().cpu().numpy(), b=ctrl.fc2.bias.detach().cpu().numpy())
        print(f"done {level} v{version}: train {len(y_tr)}, held-out {len(y_va)}", flush=True)
    (OUT / "config.json").write_text(json.dumps({"arch": "nature", "METHODS": METHODS, "SEEDS": SEEDS,
                                                  "DOMAINS": DOMAINS, "EPISODES": EPISODES}, indent=2))


if __name__ == "__main__":
    main()
