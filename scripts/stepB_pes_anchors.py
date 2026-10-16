"""Step B: Perception Stitching anchors. k-means on the raw pixels of cam0 default-task training frames, as in PeS
("k-means on D1, the images closest to the cluster centres"); the anchor set is stored as (demo, step) indices, so each
agent reads the same states from its own domain's demo file (paired frames from replay).

Usage: uv run python scripts/stepB_pes_anchors.py <K> [<K> ...]
Frames: every observation t = 0..L-1 of demos 0-99 (the training demos) of cam0. Pixels in [0, 1], flattened
(49152-d). k-means++ init then Lloyd (N_ITERS), seed 0, on the GPU. Anchor = the frame nearest each centre (distinct).
Writes results/<date>_stepB_pes_anchors/anchors_K<K>.npz (demo, step), a grid of the first 32 anchors in cam0 and
look 2 (pairing check), config.json and metrics.json (inertia, cluster sizes).
"""
import json
import sys
from datetime import date
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

TRAJ = "trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
B = "/home/ricc/projects/labelstitch-1b/results"
H5 = {"cam0": f"/home/ricc/projects/labelstitch-step1/results/20260930_dp_ours_demos_default_panda_cam0_lollipop/{TRAJ}",
      "look2": f"{B}/20261002_dp_ours_demos_default_panda_cam0_look2_lollipop/{TRAJ}"}
DEMOS = range(0, 100)
N_ITERS = 100
SEED = 0


def frames(h5, demos):
    # -> rgb (N, 128, 128, 3) uint8, demo (N,), step (N,)
    rgb, demo, step = [], [], []
    with h5py.File(h5, "r") as f:
        for i in demos:
            x = f[f"traj_{i}/obs/sensor_data/base_camera/rgb"][:]
            rgb.append(x), demo.append(np.full(len(x), i)), step.append(np.arange(len(x)))
    return np.concatenate(rgb), np.concatenate(demo), np.concatenate(step)


def kmeans(X, K, g):
    # X (N, D) float on the GPU -> centres (K, D), assignment (N,). k-means++ init, then Lloyd.
    C = X[torch.randint(len(X), (1,), generator=g, device=X.device)]
    d2 = torch.cdist(X, C).squeeze(1) ** 2
    for _ in range(K - 1):
        C = torch.cat([C, X[torch.multinomial(d2 / d2.sum(), 1, generator=g)]])
        d2 = torch.minimum(d2, (torch.cdist(X, C[-1:]).squeeze(1)) ** 2)
    for _ in range(N_ITERS):
        a = torch.cdist(X, C).argmin(1)
        for k in range(K):  # an empty cluster keeps its centre
            if (a == k).any():
                C[k] = X[a == k].mean(0)
    return C, torch.cdist(X, C).argmin(1)


if __name__ == "__main__":
    OUT = Path("results") / f"{date.today():%Y%m%d}_stepB_pes_anchors"
    OUT.mkdir(parents=True, exist_ok=True)
    rgb, demo, step = frames(H5["cam0"], DEMOS)
    X = torch.from_numpy(rgb).cuda().flatten(1).float() / 255.0  # (N, 49152)
    m = dict(n_frames=len(X))
    for K in map(int, sys.argv[1:]):
        g = torch.Generator(device="cuda").manual_seed(SEED)
        C, a = kmeans(X, K, g)
        D = torch.cdist(C, X)  # (K, N)
        idx = []
        for k in D.argsort(1)[:, :10].tolist():  # nearest frame not already taken
            idx.append(next(i for i in k if i not in idx))
        idx = np.array(idx)
        np.savez(OUT / f"anchors_K{K}.npz", demo=demo[idx], step=step[idx])
        sizes = torch.bincount(a, minlength=K).cpu().numpy()
        m[f"K{K}"] = dict(inertia=float(((X - C[a]) ** 2).sum(1).mean()), cluster_size_min=int(sizes.min()),
                          cluster_size_median=float(np.median(sizes)), cluster_size_max=int(sizes.max()),
                          n_demos_used=int(len(set(demo[idx].tolist()))))
        # pairing check: the first 32 anchors, cam0 row over look 2 row (the same states)
        with h5py.File(H5["look2"], "r") as f:
            other = np.stack([f[f"traj_{d}/obs/sensor_data/base_camera/rgb"][s] for d, s in zip(demo[idx[:32]], step[idx[:32]])])
        grid = np.concatenate([np.concatenate(list(rgb[idx[:32]].reshape(4, 8, 128, 128, 3)[r]), 1) for r in range(4)] +
                              [np.concatenate(list(other.reshape(4, 8, 128, 128, 3)[r]), 1) for r in range(4)], 0)
        Image.fromarray(grid).save(OUT / f"anchors_K{K}_cam0_over_look2.png")
        print(K, m[f"K{K}"], flush=True)
    json.dump(dict(h5=H5, demos=[DEMOS.start, DEMOS.stop], n_iters=N_ITERS, seed=SEED, features="raw pixels in [0, 1]"),
              open(OUT / "config.json", "w"), indent=1)
    json.dump(m, open(OUT / "metrics.json", "w"), indent=1)
