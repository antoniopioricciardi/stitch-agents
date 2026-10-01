"""Step 1b: fit the action-chunk labels once (k-means, K = 16, on z-scored 8-step chunks of the training demos).

Usage: uv run python scripts/step1b_labels.py <cam0.h5> <cam1.h5>
Actions do not depend on the camera; the actions of both files (demos 0-99) are pooled after checking they are
identical, so the labels are shared. Writes results/<date>_step1b_labels/{labels.pt, config.json, metrics.json}.
"""
import json
import sys
from datetime import date
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stitch.labels import assign, frame_chunks, kmeans, standardise

K = 16
N_TRAIN = 100  # demos 0-99, as --num-demos 100 (the loader sorts traj_<i> numerically)
SEED = 0


def load_actions(path):
    with h5py.File(path, "r") as f:
        return [f[f"traj_{i}"]["actions"][:] for i in range(N_TRAIN)]


A0, A1 = load_actions(sys.argv[1]), load_actions(sys.argv[2])
assert all(np.array_equal(a, b) for a, b in zip(A0, A1))  # same demos re-rendered: identical actions
actions = A0 + A1  # pooled; duplicates do not change k-means
flat = np.concatenate(actions)  # (sum L, A)
mean, std = flat.mean(0), flat.std(0)
X = torch.from_numpy(np.concatenate([standardise(frame_chunks(a), mean, std) for a in actions])).float().cuda()
C, inertia = kmeans(X, K, seed=SEED)
y = assign(X, C).cpu().numpy()

# cluster summary: size and gripper state (raw gripper action: +1 open, -1 closed), over the unique (cam0) chunks
raw = np.concatenate([frame_chunks(a) for a in A0])  # (N, CHUNK, A)
y0 = y[:len(raw)]
grip_open = (raw[:, :, -1] > 0).mean(1)  # share of open-gripper steps per chunk
clusters = []
for k in range(K):
    m = y0 == k
    clusters.append(dict(size=int(m.sum()), share_open_steps=float(grip_open[m].mean()),
                         mean_arm_action=raw[m][:, :, :-1].mean((0, 1)).round(3).tolist()))
    print(k, clusters[-1])
n_open = sum(c["share_open_steps"] > 0.5 for c in clusters)
print(f"mainly open: {n_open}, mainly closed: {K - n_open}; sizes {sorted(c['size'] for c in clusters)}")

OUT = Path("results") / f"{date.today():%Y%m%d}_step1b_labels"
OUT.mkdir(parents=True, exist_ok=True)
torch.save(dict(centroids=C.cpu(), mean=torch.from_numpy(mean), std=torch.from_numpy(std)), OUT / "labels.pt")
json.dump(dict(demos=sys.argv[1:], n_train=N_TRAIN, K=K, chunk=8, seed=SEED, restarts=10), open(OUT / "config.json", "w"), indent=1)
json.dump(dict(inertia=inertia, n_chunks=len(raw), n_mainly_open=n_open, n_mainly_closed=K - n_open,
               action_mean=mean.tolist(), action_std=std.tolist(), clusters=clusters), open(OUT / "metrics.json", "w"), indent=1)
