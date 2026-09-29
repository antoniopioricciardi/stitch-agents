# Anchor recipe comparison, offline: action centroids vs random same-action pairs (old-repo recipe with a
# 1000-frame pool per side, and with the full training set as pool), with SAPS on truly paired frames as
# the ceiling where it exists. Two settings, all encoder types:
#   versions: 1-1 v_u encoder (seed s) + v_v controller (seed s+1); unpaired anchors from disjoint episodes
#   levels:   level a encoder (seed s) + level b controller (seed s+1), played on level a (v0)
# Metric: on held-out frames, agreement of the stitched agent with the native agent of the played domain.
# Run: uv run --project mario python mario/anchors_offline.py [archs] [methods]   (comma-separated)
import itertools
import json
import sys

import numpy as np
import torch

from agents import DEVICE, ROOT, embed, inputs, load_model
from align import fit_action_pairs, fit_procrustes_paired, fit_prototypes
from data import EPISODES, TRAIN_EPS, VAL_EPS

ARCHS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["nature", "resnet18", "dinov2"]
METHODS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["bc", "scil"]
SEEDS = [0, 1, 2]
VERSIONS = [0, 1, 2]
LEVELS = ["1-1", "1-2"]
N_DRAWS = 5
SRC_EPS, TGT_EPS = TRAIN_EPS[:5], TRAIN_EPS[5:]  # version setting: disjoint episodes for unpaired anchors
OUT = ROOT / ("20260929_anchors_offline" + ("" if len(sys.argv) == 1 else "_" + "_".join(ARCHS + METHODS)))


@torch.no_grad()
def act(ctrl, Z):
    return ctrl(torch.from_numpy(Z).float().to(DEVICE)).argmax(1).cpu().numpy()


def unpaired_fits(Zs, ys, Zt, yt, seed):
    # the three unpaired recipes; random ones get N_DRAWS draws
    fits = {"centroids": [fit_prototypes(Zs, Zt, ys=ys, yt=yt)]}
    for name, pool in [("pairs_pool1000", 1000), ("pairs_full", None)]:
        fits[name] = [fit_action_pairs(Zs, Zt, ys, yt, np.random.default_rng(1000 * seed + d), pool=pool)
                      for d in range(N_DRAWS)]
    return fits


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for arch in ARCHS:
        # --- versions of 1-1 ---
        tr = {v: inputs(arch, "1-1", v, TRAIN_EPS) for v in VERSIONS}
        va = {v: inputs(arch, "1-1", v, VAL_EPS)[0] for v in VERSIONS}
        e_tr, y_tr = tr[0][2].numpy(), tr[0][1].numpy()
        src, tgt = np.isin(e_tr, SRC_EPS), np.isin(e_tr, TGT_EPS)
        for method in METHODS:
            models = {(v, s): load_model(arch, method, s, "1-1", v) for v in VERSIONS for s in SEEDS}
            Z = {k: (embed(arch, m[0], tr[k[0]][0]), embed(arch, m[0], va[k[0]])) for k, m in models.items()}
            for seed, (u, v) in itertools.product(SEEDS, itertools.permutations(VERSIONS, 2)):
                cs = (seed + 1) % len(SEEDS)
                (Zu, Zu_va), (Zv, Zv_va) = Z[u, seed], Z[v, cs]
                ctrl = models[v, cs][1]
                target = act(ctrl, Zv_va)  # the target agent's own actions on the same (paired) frames
                fits = {"saps": [fit_procrustes_paired(Zu, Zv)], **unpaired_fits(Zu[src], y_tr[src], Zv[tgt], y_tr[tgt], seed)}
                for name, fs in fits.items():
                    agree = [float((act(ctrl, Zu_va @ R.T + b) == target).mean()) for R, b in fs]
                    rows.append({"arch": arch, "setting": "versions", "method": method, "seed": seed,
                                 "pair": f"v{u}->v{v}", "aligner": name, "agree": float(np.mean(agree))})
        # --- levels ---
        data = {L: (inputs(arch, L, 0, EPISODES[L][0]), inputs(arch, L, 0, EPISODES[L][1])) for L in LEVELS}
        for method in METHODS:
            models = {(L, s): load_model(arch, method, s, L, 0) for L in LEVELS for s in SEEDS}
            for seed, (a, b) in itertools.product(SEEDS, itertools.permutations(LEVELS, 2)):
                cs = (seed + 1) % len(SEEDS)
                (enc_a, ctrl_a), (enc_b, ctrl_b) = models[a, seed], models[b, cs]
                (Xa, ya, _), (Xa_va, _, _) = data[a]
                (Xb, yb, _), _ = data[b]
                Za, Za_va, Zb = embed(arch, enc_a, Xa), embed(arch, enc_a, Xa_va), embed(arch, enc_b, Xb)
                native = act(ctrl_a, Za_va)
                for name, fs in unpaired_fits(Za, ya.numpy(), Zb, yb.numpy(), seed).items():
                    agree = [float((act(ctrl_b, Za_va @ R.T + bb) == native).mean()) for R, bb in fs]
                    rows.append({"arch": arch, "setting": "levels", "method": method, "seed": seed,
                                 "pair": f"play {a} ctrl {b}", "aligner": name, "agree": float(np.mean(agree))})
        print(f"done {arch}", flush=True)

    summary = {}
    for arch, setting, method in itertools.product(ARCHS, ["versions", "levels"], METHODS):
        rs = [r for r in rows if r["arch"] == arch and r["setting"] == setting and r["method"] == method]
        summary[f"{arch}/{setting}/{method}"] = {a: float(np.mean([r["agree"] for r in rs if r["aligner"] == a]))
                                                 for a in dict.fromkeys(r["aligner"] for r in rs)}
        print(f"{arch:9s} {setting:9s} {method:5s} " + "  ".join(f"{k} {v:.3f}" for k, v in summary[f"{arch}/{setting}/{method}"].items()))
    (OUT / "rows.json").write_text(json.dumps(rows, indent=2))
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    (OUT / "config.json").write_text(json.dumps({"ARCHS": ARCHS, "METHODS": METHODS, "SEEDS": SEEDS, "N_DRAWS": N_DRAWS,
                                                  "SRC_EPS": SRC_EPS, "TGT_EPS": TGT_EPS, "pool": 1000, "per_class": 100}, indent=2))


if __name__ == "__main__":
    main()
