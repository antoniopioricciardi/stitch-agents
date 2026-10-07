"""Step 3 pre-check (offline, before any training): which DINO adapter keeps where the cube is?

Usage: uv run python scripts/step3_probe_adapter.py
Frozen DINOv2 ViT-S/14 (stitch.models.DinoEncoder), on cam0 and look 2 demo frames. Each adapter design, followed by a
linear readout to the cube's (x, y), is trained on the frozen features (MSE on z-scored targets, Adam 1e-3, batch 512,
40 epochs, the epoch with the lowest loss on validation demos 90-99 kept), fit on demos 0-89, tested on demos 400-497:
  cls:     CLS (384) -> Linear 256 -> Linear 2
  spatial: patch grid (384, 9, 9) -> 1x1 conv 32, ReLU -> flatten -> Linear 256 -> Linear 2
plus the ridge probe of probe_cube_goal.py on the CLS token (alpha by 5-fold CV over demos) as a reference.
R^2 per coordinate on all test frames and on frames with the cube on the table (z < 0.025 m: where the policy has to
find it). Rule (fixed before running): spatial if its mean on-table x/y R^2 over both domains beats cls by >= .05,
else cls (simpler). Writes results/<date>_step3_probe_adapter/metrics.json.
"""
import json
from datetime import date
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

from stitch.models import DinoEncoder

TRAJ = "trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
H5 = {"cam0": f"/home/ricc/projects/labelstitch-step1/results/20260930_dp_ours_demos_default_panda_cam0_lollipop/{TRAJ}",
      "look2": f"/home/ricc/projects/labelstitch-1b/results/20261002_dp_ours_demos_default_panda_cam0_look2_lollipop/{TRAJ}"}
FIT, VAL, TEST = range(0, 90), range(90, 100), range(400, 498)
ON_TABLE_Z = 0.025
ALPHAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
EPOCHS = 40


def load(path, demos):
    # copied from probe_cube_goal.py: -> rgb (N, H, W, 3) uint8, cube (N, 3), demo index (N,)
    rgb, cube, demo = [], [], []
    with h5py.File(path, "r") as f:
        for i in demos:
            t = f[f"traj_{i}"]
            rgb.append(t["obs/sensor_data/base_camera/rgb"][:])
            cube.append(t["env_states/actors/cube"][:, :3])
            demo.append(np.full(len(cube[-1]), i))
    return np.concatenate(rgb), np.concatenate(cube), np.concatenate(demo)


@torch.no_grad()
def features(enc, rgb):
    # rgb (N, H, W, 3) uint8 -> CLS (N, 384), patch grid (N, 384, 9, 9) fp16, both on the GPU
    cls, grid = [], []
    for b in torch.from_numpy(rgb).permute(0, 3, 1, 2).split(512):
        x = b.cuda().float() / 255.0
        enc.spatial = False
        cls.append(enc.features(x))
        enc.spatial = True
        grid.append(enc.features(x).half())
    return torch.cat(cls), torch.cat(grid)


def train_probe(head, Xf, Yf, Xv, Yv):
    # head: X -> 2; Y z-scored with the fit statistics. Returns the head at its best validation epoch.
    torch.manual_seed(0)
    head = head.cuda()
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    best, state = np.inf, None
    for _ in range(EPOCHS):
        for idx in torch.randperm(len(Xf), device="cuda").split(512):
            loss = ((head(Xf[idx].float()) - Yf[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            v = ((head(Xv.float()) - Yv) ** 2).mean().item()
        if v < best:
            best, state = v, {k: x.clone() for k, x in head.state_dict().items()}
    head.load_state_dict(state)
    return head


def r2(Y, P):
    return 1 - ((Y - P) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)  # per coordinate


def ridge_cls(Zf, Yf, groups, Zt):
    # the ridge probe of probe_cube_goal.py (alpha by 5-fold CV over demos), on numpy float64
    ridge = lambda Z, Y, a: np.linalg.solve(Z.T @ Z + a * np.eye(Z.shape[1]), Z.T @ Y)
    folds = np.array_split(np.unique(groups), 5)
    cv = []
    for a in ALPHAS:
        s = []
        for fold in folds:
            te = np.isin(groups, fold)
            mz, my = Zf[~te].mean(0), Yf[~te].mean(0)
            s.append(r2(Yf[te], (Zf[te] - mz) @ ridge(Zf[~te] - mz, Yf[~te] - my, a) + my).mean())
        cv.append(np.mean(s))
    a = ALPHAS[int(np.argmax(cv))]
    mz, my = Zf.mean(0), Yf.mean(0)
    return (Zt - mz) @ ridge(Zf - mz, Yf - my, a) + my


if __name__ == "__main__":
    OUT = Path("results") / f"{date.today():%Y%m%d}_step3_probe_adapter"
    OUT.mkdir(parents=True, exist_ok=True)
    enc = DinoEncoder("cls").cuda().eval()
    m = {}
    for dom in ("cam0", "look2"):
        data = {}
        for split, demos in (("fit", FIT), ("val", VAL), ("test", TEST)):
            rgb, cube, demo = load(H5[dom], demos)
            data[split] = (*features(enc, rgb), cube[:, :2], cube[:, 2] < ON_TABLE_Z, demo)
            del rgb
        mu, sd = data["fit"][2].mean(0), data["fit"][2].std(0)
        Y = {s: torch.tensor((data[s][2] - mu) / sd, dtype=torch.float32, device="cuda") for s in data}
        Yt, table = data["test"][2], data["test"][3]
        heads = {"cls": (0, nn.Sequential(nn.Linear(384, 256), nn.Linear(256, 2))),
                 "spatial": (1, nn.Sequential(nn.Conv2d(384, 32, 1), nn.ReLU(), nn.Flatten(), nn.Linear(32 * 81, 256), nn.Linear(256, 2)))}
        m[dom] = {}
        for name, (k, head) in heads.items():
            head = train_probe(head, data["fit"][k], Y["fit"], data["val"][k], Y["val"])
            with torch.no_grad():
                P = torch.cat([head(x.float()) for x in data["test"][k].split(1024)]).cpu().numpy() * sd + mu
            m[dom][name] = dict(all=r2(Yt, P).tolist(), table=r2(Yt[table], P[table]).tolist())
        Zf = torch.cat([data["fit"][0], data["val"][0]]).cpu().numpy().astype(np.float64)
        Yf = np.concatenate([data["fit"][2], data["val"][2]])
        P = ridge_cls(Zf, Yf, np.concatenate([data["fit"][4], data["val"][4]]), data["test"][0].cpu().numpy().astype(np.float64))
        m[dom]["cls_ridge"] = dict(all=r2(Yt, P).tolist(), table=r2(Yt[table], P[table]).tolist())
        print(dom, json.dumps(m[dom]), flush=True)
        del data, Y
        torch.cuda.empty_cache()
    score = {a: float(np.mean([m[d][a]["table"] for d in m])) for a in ("cls", "spatial")}
    m["mean_table_xy_r2"] = score
    m["choice"] = "spatial" if score["spatial"] - score["cls"] >= 0.05 else "cls"
    print(json.dumps(score), "->", m["choice"])
    json.dump(m, open(OUT / "metrics.json", "w"), indent=1)
    json.dump(dict(fit=[0, 89], val=[90, 99], test=[400, 497], epochs=EPOCHS, on_table_z=ON_TABLE_Z, h5=H5), open(OUT / "config.json", "w"), indent=1)
