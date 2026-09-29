# Step 0: how action-structured are the oracle latents, and can 8 class prototypes capture what the
# linear controller reads? Per model (BC / SCIL x seeds), with centroids from train episodes and
# all metrics on held-out episodes:
#   proto_acc      nearest-centroid (cosine) accuracy, plain and class-balanced
#   var_in_span    share of held-out latent variance inside the span of the centred centroids
#   ctrl_acc       controller accuracy on raw latents
#   ctrl_acc_proj  controller accuracy after projecting latents onto mean + centroid span
#   eff_rank       participation ratio of the held-out latent covariance spectrum
# Run: uv run --project mario python mario/latent_checks.py [rom_version]
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from align import centroids
from data import CLASS_NAMES, N_CLASSES, TRAIN_EPS, VAL_EPS, load_episodes
from models import Controller, Encoder, to_input

ROM_VERSION = int(sys.argv[1]) if len(sys.argv) > 1 else 0
METHODS = ["bc", "scil"]
SEEDS = [0, 1, 2]
DEVICE = torch.device("cuda")
ROOT = Path(__file__).parent.parent / "results"
ORACLES = "20260929_step0_oracles_v{v}_e50"  # oracle set (results folder, {v} = version)
OUT = ROOT / f"20260929_step0_latent_v{ROM_VERSION}_e50"


def load_model(method, seed, rom_version=ROM_VERSION, run="20260928_step0_oracles_v{v}"):
    # run: results folder of the oracles, with {v} standing for the ROM version
    ck = torch.load(ROOT / run.format(v=rom_version) / f"{method}_s{seed}.pt")
    enc, ctrl = Encoder().to(DEVICE), Controller(N_CLASSES).to(DEVICE)
    enc.load_state_dict(ck["encoder"])
    ctrl.load_state_dict(ck["controller"])
    return enc.eval(), ctrl.eval()


@torch.no_grad()
def embed(enc, x):
    # x: (N, 3, 84, 84) uint8 -> (N, 512) float64 numpy
    return torch.cat([enc(to_input(x[i:i + 1024].to(DEVICE))) for i in range(0, len(x), 1024)]).double().cpu().numpy()


@torch.no_grad()
def ctrl_predict(ctrl, Z):
    return ctrl(torch.from_numpy(Z).float().to(DEVICE)).argmax(1).cpu().numpy()


def acc(pred, y):
    # plain and class-balanced accuracy
    return float((pred == y).mean()), float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x_tr, y_tr, _, _ = load_episodes(TRAIN_EPS, ROM_VERSION)
    x_va, y_va, _, _ = load_episodes(VAL_EPS, ROM_VERSION)
    y_tr, y_va = y_tr.numpy(), y_va.numpy()
    classes = np.arange(N_CLASSES)

    metrics, pca_data = {}, {}
    for method in METHODS:
        for seed in SEEDS:
            enc, ctrl = load_model(method, seed, ROM_VERSION, ORACLES)
            Z_tr, Z_va = embed(enc, x_tr), embed(enc, x_va)
            C = centroids(Z_tr, y_tr, classes)  # (8, 512)

            # nearest centroid by cosine
            Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
            Zn = Z_va / (np.linalg.norm(Z_va, axis=1, keepdims=True) + 1e-12)
            proto_acc = acc((Zn @ Cn.T).argmax(1), y_va)

            # orthonormal basis of the centred-centroid span (rank <= 7)
            mu = C.mean(0)
            U, S, _ = np.linalg.svd((C - mu).T, full_matrices=False)
            B = U[:, S > 1e-8 * S[0]]  # (512, k)
            Zc = Z_va - Z_va.mean(0)
            var_in_span = float(((Zc @ B) ** 2).sum() / (Zc ** 2).sum())
            Z_proj = mu + (Z_va - mu) @ B @ B.T

            ev = np.linalg.eigvalsh(np.cov(Z_va.T))
            metrics[f"{method}_s{seed}"] = {
                "proto_acc": proto_acc[0], "proto_bal_acc": proto_acc[1],
                "var_in_span": var_in_span, "span_dim": int(B.shape[1]),
                "ctrl_acc": acc(ctrl_predict(ctrl, Z_va), y_va)[0],
                "ctrl_acc_proj": acc(ctrl_predict(ctrl, Z_proj), y_va)[0],
                "eff_rank": float(ev.sum() ** 2 / (ev ** 2).sum()),
                "frac_dead_units": float((Z_va.max(0) == 0).mean()),  # ReLU units never active
            }
            print(f"{method}_s{seed}: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                                   for k, v in metrics[f'{method}_s{seed}'].items()))
            if seed == 0:
                Zc_all = Z_va - Z_va.mean(0)
                _, _, Vt = np.linalg.svd(Zc_all, full_matrices=False)
                pca_data[method] = Zc_all @ Vt[:2].T

    keys = [k for k in metrics["bc_s0"] if k != "span_dim"]
    for method in METHODS:
        m = np.array([[metrics[f"{method}_s{s}"][k] for k in keys] for s in SEEDS])
        metrics[method] = {k: [float(m[:, i].mean()), float(m[:, i].std())] for i, k in enumerate(keys)}
        print(f"{method}: " + ", ".join(f"{k} {v[0]:.3f} ± {v[1]:.3f}" for k, v in metrics[method].items()))
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, method in zip(axes, METHODS):
        for c in classes:
            p = pca_data[method][y_va == c]
            ax.scatter(p[:, 0], p[:, 1], s=4, alpha=0.5, label=CLASS_NAMES[c])
        ax.set_title(f"{method.upper()} seed 0, held-out latents (PCA)")
    axes[1].legend(markerscale=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "pca_seed0.png", dpi=120)


if __name__ == "__main__":
    main()
