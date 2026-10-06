"""Step F: encoder geometry after fine-tuning from the stitched start (seed pair 1).

Usage: uv run python scripts/stepF_geometry.py
For each combination (same task: look 2 frames; goal shift: goal-variant look 2 frames), N in {10, 25} and arm
(start = look 2 encoder s1 + the Step R PCA16 map; map only = Step R; enc = Step F (a); ctrl = Step F (b)):
NC1 = tr(S_W)/tr(S_B) and effective rank (participation ratio) of z on the current frames of demos 0-99, with the
action-chunk cluster labels (labels.pt), for z before the map (PlainConv output) and after it (what the controller reads).
Final weights (5k) of each run. Writes results/<date>_stepF_geometry/metrics.json and table.md.
"""
import glob
import json
from datetime import date
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

from diffusion_policy.plain_conv import PlainConv
from stitch.labels import assign, frame_chunks, standardise

R = "/home/ricc/projects/labelstitch-r/results"
B = "/home/ricc/projects/labelstitch-1b/results"
TRAJ = "trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
H5 = {"same": f"{B}/20261002_dp_ours_demos_default_panda_cam0_look2_lollipop/{TRAJ}",
      "goal": f"{B}/20261003_dp_ours_demos_goal_panda_cam0_look2_lollipop/{TRAJ}"}
MAPS = {"same": "look2_to_cam0", "goal": "goal_look2_to_goal_cam0"}
DOM = {"same": "look2", "goal": "goal_look2"}
START_ENC = f"{B}/20261003_step1b_dp_ref_look2_s1"
LABELS = f"{R}/20261001_step1b_labels/labels.pt"


def load_frames(path):
    # current frames t = 0..L-2 of demos 0-99 -> rgb (N, 3, 128, 128) uint8, chunks (N, 8, A)
    rgb, chunks = [], []
    with h5py.File(path, "r") as f:
        for i in range(100):
            t = f[f"traj_{i}"]
            a = t["actions"][:]
            rgb.append(t["obs/sensor_data/base_camera/rgb"][:len(a) - 1])
            chunks.append(frame_chunks(a))
    return torch.from_numpy(np.concatenate(rgb)).permute(0, 3, 1, 2), np.concatenate(chunks)


def final_sd(run_dir):
    ckpt = max(glob.glob(f"{run_dir}/runs/*/checkpoints/[0-9]*.pt"), key=lambda p: int(Path(p).stem))
    return torch.load(ckpt, map_location="cuda")["ema_agent"]


def encoder_and_map(sd):
    # fine-tuned runs: visual_encoder = Sequential(PlainConv, Linear map)
    enc = PlainConv(in_channels=3, out_dim=256, pool_feature_map=True).cuda().eval()
    enc.load_state_dict({k[len("visual_encoder.0."):]: v for k, v in sd.items() if k.startswith("visual_encoder.0.")})
    lin = nn.Linear(256, 256).cuda()
    lin.load_state_dict({k[len("visual_encoder.1."):]: v for k, v in sd.items() if k.startswith("visual_encoder.1.")})
    return enc, lin


@torch.no_grad()
def encode(f, rgb):
    return torch.cat([f(x.cuda().float() / 255.0) for x in rgb.split(512)]).cpu().numpy()


def geometry(Z, y):
    # copied from step1b_stitch.py: Z (N, d), y (N,) -> NC1, effective rank
    classes = np.unique(y)
    M = np.stack([Z[y == c].mean(0) for c in classes])
    Sw = np.mean([((Z[y == c] - M[i]) ** 2).sum(1).mean() for i, c in enumerate(classes)])
    Sb = ((M - M.mean(0)) ** 2).sum(1).mean()
    ev = np.sort(np.linalg.eigvalsh(np.cov((Z - Z.mean(0)).T)))[::-1]
    return dict(nc1=float(Sw / Sb), eff_rank=float(ev.sum() ** 2 / (ev ** 2).sum()))


if __name__ == "__main__":
    OUT = Path("results") / f"{date.today():%Y%m%d}_stepF_geometry"
    OUT.mkdir(parents=True, exist_ok=True)
    lab = torch.load(LABELS)
    start_sd = final_sd(START_ENC)
    m, rows = {}, ["| combination | N | arm | NC1 / eff. rank before the map | after the map |", "|---|---|---|---|---|"]
    for K in ("same", "goal"):
        rgb, chunks = load_frames(H5[K])
        y = assign(torch.from_numpy(standardise(chunks, lab["mean"].numpy(), lab["std"].numpy())).float(), lab["centroids"].float()).numpy()
        for n in (10, 25):
            M = np.load(f"{R}/20261005_step1b_stitch_stepR_map_{K}_p1_n{n}_{DOM[K]}/maps_{MAPS[K]}.npz")
            enc = PlainConv(in_channels=3, out_dim=256, pool_feature_map=True).cuda().eval()
            enc.load_state_dict({k[len("visual_encoder."):]: v for k, v in start_sd.items() if k.startswith("visual_encoder.")})
            lin = nn.Linear(256, 256).cuda()
            lin.weight.data = torch.tensor(M["nn_affine_pca16_R"], dtype=torch.float32, device="cuda")
            lin.bias.data = torch.tensor(M["nn_affine_pca16_b"], dtype=torch.float32, device="cuda")
            arms = {"start": (enc, lin)}
            for arm, pattern in (("map only", f"{R}/2026100?_stepR_ft_map_{K}_p1_n{n}"),
                                 ("map + encoder", f"results/2026100?_stepF_ft_enc_{K}_p1_n{n}"),
                                 ("map + last ctrl", f"results/2026100?_stepF_ft_ctrl_{K}_p1_n{n}")):
                runs = sorted(glob.glob(pattern))
                if runs:
                    arms[arm] = encoder_and_map(final_sd(runs[-1]))
            for arm, (enc, lin) in arms.items():
                z = encode(enc, rgb)
                g = dict(before=geometry(z, y), after=geometry(encode(lambda x: lin(enc(x)), rgb), y))
                m[f"{K}_n{n}_{arm}"] = g
                rows.append(f"| {K} | {n} | {arm} | {g['before']['nc1']:.2f} / {g['before']['eff_rank']:.1f} | "
                            f"{g['after']['nc1']:.2f} / {g['after']['eff_rank']:.1f} |")
                print(rows[-1], flush=True)
    json.dump(m, open(OUT / "metrics.json", "w"), indent=1)
    open(OUT / "table.md", "w").write("\n".join(rows) + "\n")
