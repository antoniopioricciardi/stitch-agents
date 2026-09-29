# Demo loading + preprocessing shared by training and closed-loop evaluation.
# Preprocessing follows the SCIL Atari setup (Kanervisto et al. 2020, --merge): each frame is resized
# to 84x84 RGB, and the input is the pixelwise max of the current frame and the previous decision's
# frame (SKIP NES frames earlier), giving one 3-channel image that still shows motion.
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DATA = Path(__file__).resolve().parent / "data"
SKIP = 4
SIZE = 84
# 12 COMPLEX_MOVEMENT actions -> 8 classes. Class i < 8 is COMPLEX_MOVEMENT action i, so a
# predicted class can be sent to the env as is. L+B (8) -> L (6), L+A+B (9) -> L+A (7),
# down/up (never used) -> NOOP.
TO_CLASS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 6, 7, 0, 0])
N_CLASSES = 8
CLASS_NAMES = ["NOOP", "R", "R+A", "R+B", "R+A+B", "A", "L", "L+A"]
TRAIN_EPS = [1, 2, 3, 4, 5, 8, 9, 11, 12, 13]  # 1-1 wins
VAL_EPS = [14, 17]  # 1-1 wins, held out
# per level: (train wins, held-out wins)
EPISODES = {"1-1": (TRAIN_EPS, VAL_EPS), "1-2": ([1, 2, 3, 4, 5, 7, 8, 9, 10], [12, 13])}


def resize(frames, size=SIZE):
    # frames: (T, 240, 256, 3) uint8 -> (T, 3, size, size) uint8
    # ascontiguousarray: nes-py returns the screen as a view with negative strides
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    return x.round().clamp(0, 255).to(torch.uint8)


def merge(cur, prev):
    # pixelwise max of current and previous-decision frame, both (.., 3, 84, 84) uint8
    return torch.maximum(cur, prev)


def load_episodes(eps, rom_version=0, level="1-1", size=SIZE):
    # Every recorded frame t is a sample: input max(frame t, frame t-SKIP) (zeros before the start,
    # as in the reference), label = class of action[t]. Using every t covers all SKIP window offsets.
    # Returns x: (N, 3, size, size) uint8, y: (N,) int64, ep: (N,) int64, x_pos: (N,) int64
    d = DATA / ("demos" if rom_version == 0 else f"demos_v{rom_version}")
    xs, ys, es, ps = [], [], [], []
    for e in eps:
        f = next(d.glob(f"{level}_ep{e:03d}_*.npz"))
        z = np.load(f)
        img = resize(z["obs"], size)
        prev = torch.cat([torch.zeros_like(img[:SKIP]), img[:-SKIP]])
        xs.append(merge(img, prev))
        ys.append(torch.from_numpy(TO_CLASS[z["action"]]))
        es.append(torch.full((len(img),), e))
        ps.append(torch.from_numpy(z["x_pos"]))
    return torch.cat(xs), torch.cat(ys), torch.cat(es), torch.cat(ps)
