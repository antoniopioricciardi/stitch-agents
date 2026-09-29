# Sanity check of the recorded demos: episode lengths/outcomes, action class counts, and how often
# the action changes inside a k-frame window (the label noise we get by applying frame skip k later).
import sys
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).parent / "data" / "demos"
LEVEL = sys.argv[1] if len(sys.argv) > 1 else "1-1"
SKIPS = [2, 4]
NAMES = ["NOOP", "R", "R+A", "R+B", "R+A+B", "A", "L", "L+A", "L+B", "L+A+B", "down", "up"]
# 12 -> 7 merge: left variants -> left, down/up -> NOOP
TO_SIMPLE = np.array([0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 0, 0])

files = sorted(DEMO_DIR.glob(f"{LEVEL}_*.npz"))
eps = [np.load(f) for f in files]
wins = [e for e in eps if e["flag_get"]]
print(f"{len(eps)} episodes: {len(wins)} wins, {len(eps) - len(wins)} fails | skip at record time: "
      f"{sorted({int(e['skip']) for e in eps})}")
for f, e in zip(files, eps):  # file name is the source of truth (files may be renamed)
    print(f"  {f.stem} "
          f"{len(e['action']):5d} frames, max x {e['x_pos'].max():5d}")

acts = np.concatenate([e["action"] for e in wins])  # (N,) all frames of winning episodes
print(f"\nwinning episodes: {len(acts)} frames")
counts = np.bincount(acts, minlength=12)
print("12 classes (all frames):", {n: int(c) for n, c in zip(NAMES, counts)})
print(" 7 classes (all frames):", {n: int(c) for n, c in zip(NAMES[:7], np.bincount(TO_SIMPLE[acts], minlength=7))})

for k in SKIPS:
    # windows start at frames 0, k, 2k, ...; the label is the action at the window start
    changed, n_win, sub = 0, 0, []
    for e in wins:
        a = e["action"]
        T = len(a) // k * k
        w = a[:T].reshape(-1, k)  # (T/k, k)
        changed += (w != w[:, :1]).any(1).sum()
        n_win += len(w)
        sub.append(w[:, 0])
    sub = np.concatenate(sub)
    print(f"\nskip {k}: {n_win} samples, windows with an action change: {changed / n_win:.1%}")
    print("  12 classes:", {n: int(c) for n, c in zip(NAMES, np.bincount(sub, minlength=12))})
