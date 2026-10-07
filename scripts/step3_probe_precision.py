"""Step 3 diagnosis (offline): how precisely do frozen DINOv2 features locate the cube, at 126 vs 224 px, bf16 vs fp32?

Usage: uv run python scripts/step3_probe_precision.py
Frames with the cube on the table (z < 0.025 m: where the policy has to find it), cam0 and look 2; fit demos 0-89,
validation 90-99 (best epoch), test 400-497. An MLP probe predicts the cube's (x, y); metric = Euclidean x/y error in
cm on the test frames (mean, median).
  DINOv2 ViT-S/14, frozen, patch grid (384, g, g) with g = 9 at 126 px, 16 at 224 px, bf16 or fp32;
    head = the Step 3 spatial adapter (1x1 conv 32, ReLU, flatten, Linear 256), ReLU, Linear 2.
  Reference: PlainConv z (256) of the trained oracles look 2 s1 and cam0 s1 on their own domain; head = Linear 256,
    ReLU, Linear 2.
Training: Adam 1e-3, batch 512, 60 epochs, z-scored targets, seed 0. Writes results/<date>_step3_probe_precision/.
"""
import glob
import json
from datetime import date
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.plain_conv import PlainConv
from stitch.models import DINO_HUB

TRAJ = "trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
H5 = {"cam0": f"/home/ricc/projects/labelstitch-step1/results/20260930_dp_ours_demos_default_panda_cam0_lollipop/{TRAJ}",
      "look2": f"/home/ricc/projects/labelstitch-1b/results/20261002_dp_ours_demos_default_panda_cam0_look2_lollipop/{TRAJ}"}
PLAINCONV = {"cam0": "/home/ricc/projects/labelstitch-step1/results/20261001_step2g_dp_core_nograsp_50k_s1",
             "look2": "/home/ricc/projects/labelstitch-1b/results/20261003_step1b_dp_ref_look2_s1"}
FIT, VAL, TEST = range(0, 90), range(90, 100), range(400, 498)
ON_TABLE_Z = 0.025
EPOCHS = 60
MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)


def load(path, demos):
    # on-table frames only -> rgb (N, H, W, 3) uint8, cube x/y (N, 2) in metres
    rgb, cube = [], []
    with h5py.File(path, "r") as f:
        for i in demos:
            t = f[f"traj_{i}"]
            c = t["env_states/actors/cube"][:, :3]
            keep = c[:, 2] < ON_TABLE_Z
            rgb.append(t["obs/sensor_data/base_camera/rgb"][:][keep])
            cube.append(c[keep, :2])
    return np.concatenate(rgb), np.concatenate(cube)


@torch.no_grad()
def dino_grid(model, rgb, size, bf16):
    # rgb (N, H, W, 3) uint8 -> patch grid (N, 384, g, g), fp16 on the GPU (storage only)
    out = []
    for b in torch.from_numpy(rgb).permute(0, 3, 1, 2).split(256):
        x = (F.interpolate(b.cuda().float() / 255.0, size=size, mode="bilinear", antialias=True) - MEAN) / STD
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            p = model.forward_features(x)["x_norm_patchtokens"].float()
        g = size // 14
        out.append(p.transpose(1, 2).reshape(len(x), 384, g, g).half())
    return torch.cat(out)


@torch.no_grad()
def plainconv_z(run_dir, rgb):
    ckpt = max(glob.glob(f"{run_dir}/runs/*/checkpoints/[0-9]*.pt"), key=lambda p: int(Path(p).stem))
    sd = torch.load(ckpt, map_location="cuda")["ema_agent"]
    enc = PlainConv(in_channels=3, out_dim=256, pool_feature_map=True).cuda().eval()
    enc.load_state_dict({k[len("visual_encoder."):]: v for k, v in sd.items() if k.startswith("visual_encoder.")})
    return torch.cat([enc(b.cuda().float() / 255.0) for b in torch.from_numpy(rgb).permute(0, 3, 1, 2).split(512)]).half()


def fit_probe(head, Xf, Yf, Xv, Yv):
    # head: X -> 2 (z-scored x/y); keeps the epoch with the lowest validation MSE
    torch.manual_seed(0)
    head = head.cuda()
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    best, state = np.inf, None
    for _ in range(EPOCHS):
        head.train()
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
    return head.eval()


if __name__ == "__main__":
    OUT = Path("results") / f"{date.today():%Y%m%d}_step3_probe_precision"
    OUT.mkdir(parents=True, exist_ok=True)
    dino = torch.hub.load(DINO_HUB, "dinov2_vits14", source="local", verbose=False).cuda().eval()
    m, rows = {}, ["| domain | features | mean error (cm) | median error (cm) |", "|---|---|---|---|"]
    for dom in ("cam0", "look2"):
        data = {s: load(H5[dom], d) for s, d in (("fit", FIT), ("val", VAL), ("test", TEST))}
        mu, sd = data["fit"][1].mean(0), data["fit"][1].std(0)
        Y = {s: torch.tensor((data[s][1] - mu) / sd, dtype=torch.float32, device="cuda") for s in data}
        settings = {f"DINO {size} px {'bf16' if bf16 else 'fp32'}": (lambda rgb, size=size, bf16=bf16: dino_grid(dino, rgb, size, bf16),
                    lambda g=size // 14: nn.Sequential(nn.Conv2d(384, 32, 1), nn.ReLU(), nn.Flatten(), nn.Linear(32 * g * g, 256),
                                                       nn.ReLU(), nn.Linear(256, 2)))
                    for size in (126, 224) for bf16 in (True, False)}
        settings[f"PlainConv {dom} s1 (oracle z)"] = (lambda rgb: plainconv_z(PLAINCONV[dom], rgb),
                                                      lambda: nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 2)))
        m[dom] = {}
        for name, (feat, make_head) in settings.items():
            X = {s: feat(data[s][0]) for s in data}
            head = fit_probe(make_head(), X["fit"], Y["fit"], X["val"], Y["val"])
            with torch.no_grad():
                P = torch.cat([head(x.float()) for x in X["test"].split(512)]).cpu().numpy() * sd + mu
            err = 100 * np.linalg.norm(P - data["test"][1], axis=1)  # cm
            m[dom][name] = dict(mean_cm=float(err.mean()), median_cm=float(np.median(err)), n_test=int(len(err)))
            rows.append(f"| {dom} | {name} | {err.mean():.2f} | {np.median(err):.2f} |")
            print(rows[-1], flush=True)
            del X
            torch.cuda.empty_cache()
        m[dom]["n_fit"] = int(len(data["fit"][1]))
    json.dump(m, open(OUT / "metrics.json", "w"), indent=1)
    json.dump(dict(fit=[0, 89], val=[90, 99], test=[400, 497], on_table_z=ON_TABLE_Z, epochs=EPOCHS, h5=H5, plainconv=PLAINCONV),
              open(OUT / "config.json", "w"), indent=1)
    open(OUT / "table.md", "w").write("\n".join(rows) + "\n")
