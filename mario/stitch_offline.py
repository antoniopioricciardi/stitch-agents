# Step 0a: offline stitching across ROM versions. For every directed pair u -> v of {v0, v1, v2},
# encoder_u (seed s) + T + controller_v (seed s+1), with T from identity / SAPS (paired frames) / prototypes (unpaired
# class centroids), for BC and SCIL oracles. Evaluated on held-out paired frames, no rollouts.
# Run: uv run --project mario python mario/stitch_offline.py [arch]
import itertools
import json
import sys

import numpy as np
import torch

from align import fit_identity, fit_procrustes_paired, fit_prototypes
from agents import DEVICE, ROOT, embed, inputs, load_model
from data import N_CLASSES, TRAIN_EPS, VAL_EPS

ARCH = sys.argv[1] if len(sys.argv) > 1 and __name__ == "__main__" else "nature"

VERSIONS = [0, 1, 2]
METHODS = ["bc", "scil"]
SEEDS = [0, 1, 2]
PROTO_SRC_EPS = TRAIN_EPS[:5]  # prototype source and target come from disjoint episodes, so the
PROTO_TGT_EPS = TRAIN_EPS[5:]  # prototype aligner never sees the same frames in both domains
N_PER_CLASS = [5, 20, 100, None]  # None = all
N_DRAWS = 5
OUT = ROOT / ("20260929_step0a_stitch_offline_e50" if ARCH == "nature" else f"20260929_stitch_offline_{ARCH}")


@torch.no_grad()
def ctrl_predict(ctrl, Z):
    return ctrl(torch.from_numpy(Z).float().to(DEVICE)).argmax(1).cpu().numpy()


def acc(pred, y):
    # plain and class-balanced accuracy
    return float((pred == y).mean()), float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))


def subsample(Z, y, n, rng):
    # up to n random samples per class (all of them if the class has fewer)
    if n is None:
        return Z, y
    idx = np.concatenate([rng.permutation(np.where(y == c)[0])[:n] for c in np.unique(y)])
    return Z[idx], y[idx]


def evaluate(R, b, Zu_va, Zv_va, ctrl_v, y_va):
    # stitched = ctrl_v(Zu @ R.T + b), compared with expert labels and with the target oracle
    Zm = Zu_va @ R.T + b
    pred = ctrl_predict(ctrl_v, Zm)
    cos = (Zm * Zv_va).sum(1) / (np.linalg.norm(Zm, axis=1) * np.linalg.norm(Zv_va, axis=1) + 1e-12)
    a, ba = acc(pred, y_va)
    return {"acc": a, "bal_acc": ba, "agree": float((pred == ctrl_predict(ctrl_v, Zv_va)).mean()),
            "cos": float(cos.mean())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}
    for v in VERSIONS:
        x_tr, y_tr, e_tr = inputs(ARCH, "1-1", v, TRAIN_EPS)
        x_va, y_va, _ = inputs(ARCH, "1-1", v, VAL_EPS)
        data[v] = (x_tr, y_tr.numpy(), e_tr.numpy(), x_va, y_va.numpy())
    y_tr, e_tr, y_va = data[0][1], data[0][2], data[0][4]  # identical across versions (replayed)
    src_mask, tgt_mask = np.isin(e_tr, PROTO_SRC_EPS), np.isin(e_tr, PROTO_TGT_EPS)

    rows = []  # one dict per (method, seed, u, v, aligner, n, draw)
    for method in METHODS:
        Z, ctrls = {}, {}  # (v, seed) -> (train latents, val latents) of that model on its own domain
        for v, s in itertools.product(VERSIONS, SEEDS):
            enc, ctrl = load_model(ARCH, method, s, "1-1", v)
            Z[v, s] = (embed(ARCH, enc, data[v][0]), embed(ARCH, enc, data[v][3]))
            ctrls[v, s] = ctrl
        for seed in SEEDS:
            # encoder seed s, controller seed s+1: models with the same seed index share their
            # initialisation (train() seeds it), which would make identity look artificially good
            cs = (seed + 1) % len(SEEDS)
            for u, v in itertools.permutations(VERSIONS, 2):
                (Zu_tr, Zu_va), (Zv_tr, Zv_va) = Z[u, seed], Z[v, cs]
                ctrl_v = ctrls[v, cs]
                base = {"method": method, "seed": seed, "u": u, "v": v}
                rows.append({**base, "aligner": "native_v", "n": None, "draw": 0,
                             **evaluate(*fit_identity(Zv_va, Zv_va), Zv_va, Zv_va, ctrl_v, y_va)})
                rows.append({**base, "aligner": "identity", "n": None, "draw": 0,
                             **evaluate(*fit_identity(Zu_tr, Zv_tr), Zu_va, Zv_va, ctrl_v, y_va)})
                rows.append({**base, "aligner": "saps", "n": None, "draw": 0,
                             **evaluate(*fit_procrustes_paired(Zu_tr, Zv_tr), Zu_va, Zv_va, ctrl_v, y_va)})
                for n in N_PER_CLASS:
                    for draw in range(1 if n is None else N_DRAWS):
                        rng = np.random.default_rng(1000 * seed + draw)
                        Zs, ys = subsample(Zu_tr[src_mask], y_tr[src_mask], n, rng)
                        Zt, yt = subsample(Zv_tr[tgt_mask], y_tr[tgt_mask], n, rng)
                        R, b = fit_prototypes(Zs, Zt, ys=ys, yt=yt)
                        rows.append({**base, "aligner": "prototypes", "n": n, "draw": draw,
                                     **evaluate(R, b, Zu_va, Zv_va, ctrl_v, y_va)})
        print(f"done {method}")

    # summary: mean ± std over the 6 pairs x 3 seeds (draws averaged first). % of paired ceiling is
    # computed per stitch against that stitch's SAPS value, on balanced accuracy and on agreement
    # (plain accuracy is dominated by the majority class R and is not informative here).
    majority = float((y_va == np.bincount(y_tr).argmax()).mean())
    print(f"majority-class accuracy on val: {majority:.3f}, balanced: {1 / len(np.unique(y_va)):.3f}")
    saps = {(r["method"], r["seed"], r["u"], r["v"]): r for r in rows if r["aligner"] == "saps"}
    summary = {"majority_acc": majority}
    for method in METHODS:
        for aligner, n in [("native_v", None), ("identity", None), ("saps", None)] + [("prototypes", n) for n in N_PER_CLASS]:
            key = f"{method}/{aligner}" + ("" if aligner != "prototypes" else f"/n={n or 'all'}")
            per_stitch = {}
            for r in rows:
                if r["method"] == method and r["aligner"] == aligner and r["n"] == n:
                    per_stitch.setdefault((r["seed"], r["u"], r["v"]), []).append(r)
            vals = {m: np.array([np.mean([r[m] for r in rs]) for rs in per_stitch.values()])
                    for m in ["acc", "bal_acc", "agree", "cos"]}
            for m in ["bal_acc", "agree"]:
                vals[f"pct_paired_{m}"] = np.array([np.mean([r[m] for r in rs]) / saps[(method, *k)][m]
                                                     for k, rs in per_stitch.items()])
            summary[key] = {m: [float(a.mean()), float(a.std())] for m, a in vals.items()}
            print(f"{key:26s} " + "  ".join(f"{m} {a.mean():.3f}±{a.std():.3f}" for m, a in vals.items()))

    (OUT / "rows.json").write_text(json.dumps(rows))
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    (OUT / "config.json").write_text(json.dumps({
        "ARCH": ARCH, "VERSIONS": VERSIONS, "METHODS": METHODS, "SEEDS": SEEDS, "TRAIN_EPS": TRAIN_EPS, "VAL_EPS": VAL_EPS,
        "PROTO_SRC_EPS": PROTO_SRC_EPS, "PROTO_TGT_EPS": PROTO_TGT_EPS, "N_PER_CLASS": N_PER_CLASS,
        "N_DRAWS": N_DRAWS, "N_CLASSES": N_CLASSES}, indent=2))


if __name__ == "__main__":
    main()
