# Neural-collapse checks: how close are the action centroids of an encoder to a regular simplex, and do
# independently trained encoders share the same centroid geometry (up to rotation)?
# Per model, on its own training frames (all 8 classes):
#   norm_cv     std / mean of the centred centroid norms (0 = equal norms)
#   cos_err     mean |cos(M_i, M_j) + 1/(K-1)| over centroid pairs (0 = regular simplex; K = 8)
#   nc1         tr(S_within) / tr(S_between): within-class spread relative to class separation (0 = collapsed)
#   eff_rank    participation ratio of the latent covariance spectrum
#   var_span    share of latent variance inside the (K-1)-dim span of the centred centroids
# Across models (different domain AND different seed): gram_diff = mean |G_a - G_b| over the off-diagonal
# entries of the centroid cosine matrices (0 = same geometry up to rotation/reflection).
# Run: uv run --project mario python mario/collapse_checks.py [arch] [methods]   (methods comma-separated)
import itertools
import json
import sys

import numpy as np

from agents import ROOT, embed, inputs, load_model
from data import EPISODES, N_CLASSES

DOMAINS = [("1-1", 0), ("1-1", 1), ("1-1", 2), ("1-2", 0)]
MODELS = {"nature": ["bc", "scil", "scilproj"], "resnet18": ["bc", "scil"], "dinov2": ["bc", "scil"]}
if len(sys.argv) > 2:
    MODELS = {sys.argv[1]: sys.argv[2].split(",")}
SEEDS = [0, 1, 2]
OUT = ROOT / ("20260929_collapse_checks" + ("" if len(sys.argv) <= 2 else "_" + "_".join([sys.argv[1]] + MODELS[sys.argv[1]])))


def geometry(Z, y):
    # Z: (N, d) latents, y: (N,) classes -> metrics dict and the (K, K) centroid cosine matrix
    M = np.stack([Z[y == c].mean(0) for c in range(N_CLASSES)])  # (K, d) class means
    Mc = M - M.mean(0)  # centred on the mean of the class means (balanced "global mean")
    norms = np.linalg.norm(Mc, axis=1)
    G = (Mc @ Mc.T) / np.outer(norms, norms)  # (K, K) centroid cosines
    off = ~np.eye(N_CLASSES, dtype=bool)
    Sw = np.mean([((Z[y == c] - M[c]) ** 2).sum(1).mean() for c in range(N_CLASSES)])
    Sb = (norms ** 2).mean()
    Zc = Z - Z.mean(0)
    ev = np.linalg.eigvalsh(np.cov(Zc.T))
    B = np.linalg.svd(Mc.T, full_matrices=False)[0][:, :N_CLASSES - 1]  # (d, K-1) centroid-span basis
    return {"norm_cv": float(norms.std() / norms.mean()),
            "cos_err": float(np.abs(G[off] + 1 / (N_CLASSES - 1)).mean()),
            "nc1": float(Sw / Sb),
            "eff_rank": float(ev.sum() ** 2 / (ev ** 2).sum()),
            "var_span": float(((Zc @ B) ** 2).sum() / (Zc ** 2).sum())}, G


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, grams = [], {}
    for arch, methods in MODELS.items():
        data = {d: inputs(arch, d[0], d[1], EPISODES[d[0]][0]) for d in DOMAINS}
        for method, d, s in itertools.product(methods, DOMAINS, SEEDS):
            enc, _ = load_model(arch, method, s, *d)
            m, G = geometry(embed(arch, enc, data[d][0][::2]), data[d][1].numpy()[::2])  # every 2nd frame
            grams[arch, method, d, s] = G
            rows.append({"arch": arch, "method": method, "domain": f"{d[0]} v{d[1]}", "seed": s, **m})
        print(f"done {arch}", flush=True)

    off = ~np.eye(N_CLASSES, dtype=bool)
    summary = {}
    for arch, methods in MODELS.items():
        for method in methods:
            rs = [r for r in rows if r["arch"] == arch and r["method"] == method]
            keys = [k for k in rs[0] if k not in ("arch", "method", "domain", "seed")]
            out = {k: float(np.mean([r[k] for r in rs])) for k in keys}
            # geometry shared across independently trained encoders: different domain and different seed
            diffs = [np.abs(grams[arch, method, da, sa][off] - grams[arch, method, db, sb][off]).mean()
                     for (da, sa), (db, sb) in itertools.combinations(itertools.product(DOMAINS, SEEDS), 2)
                     if da != db and sa != sb]
            out["gram_diff"] = float(np.mean(diffs))
            summary[f"{arch}/{method}"] = out
            print(f"{arch:9s} {method:9s} " + "  ".join(f"{k} {v:.3f}" for k, v in out.items()))
    (OUT / "rows.json").write_text(json.dumps(rows, indent=2))
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    (OUT / "config.json").write_text(json.dumps({"DOMAINS": DOMAINS, "MODELS": MODELS, "SEEDS": SEEDS,
                                                  "simplex_cos": -1 / (N_CLASSES - 1), "frames": "training, every 2nd"}, indent=2))


if __name__ == "__main__":
    main()
