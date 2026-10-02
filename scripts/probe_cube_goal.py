"""Step 1b diagnostic: does an agent's frozen z encode where the cube and the goal are?

Usage: uv run python scripts/probe_cube_goal.py <run_dir> <domain> [<run_dir> <domain> ...]   (domain: cam0, cam1, cam2, look1, cam2goal)
Encoder = the final EMA agent's visual_encoder (PlainConv), on its own domain's frames (every frame of a demo).
Linear probe z -> cube position (env_states/actors/cube[:, :3]) and goal position (obs/extra/goal_pos): ridge on centred
latents (no per-unit scaling, Mario F2), alpha chosen by 5-fold CV over demos 0-99 (folds = whole demos), test on
demos 400-497. R^2 per coordinate, on all frames and on frames with the cube still on the table (cube z < 0.025 m,
i.e. before the lift: where the policy has to find the cube).
Writes results/<date>_step1b_probe_cube_goal/metrics.json (one entry per run).
"""
import glob
import json
import sys
from datetime import date
from pathlib import Path

import h5py
import numpy as np
import torch

from diffusion_policy.plain_conv import PlainConv

TRAJ = "trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
H5 = {"cam0": f"/home/ricc/projects/labelstitch-step1/results/20260930_dp_ours_demos_default_panda_cam0_lollipop/{TRAJ}",
      "cam1": f"results/20261001_dp_ours_demos_default_panda_cam1_lollipop/{TRAJ}",
      "cam2": f"results/20261001_dp_ours_demos_default_panda_cam2_lollipop/{TRAJ}",
      "look1": f"results/20261001_dp_ours_demos_default_panda_cam0_look1_lollipop/{TRAJ}",
      "cam2goal": f"results/20261002_dp_ours_demos_goal_panda_cam2_lollipop/{TRAJ}"}
FIT, TEST = range(0, 100), range(400, 498)
ALPHAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
ON_TABLE_Z = 0.025  # cube half size 0.02: resting on the table


def load(path, demos):
    # -> rgb (N, H, W, 3) uint8, cube (N, 3), goal (N, 3), demo index (N,)
    rgb, cube, goal, demo = [], [], [], []
    with h5py.File(path, "r") as f:
        for i in demos:
            if f"traj_{i}" not in f:  # the goal-variant demo set has 496 demos (traj_0-495)
                continue
            t = f[f"traj_{i}"]
            rgb.append(t["obs/sensor_data/base_camera/rgb"][:])
            cube.append(t["env_states/actors/cube"][:, :3])
            goal.append(t["obs/extra/goal_pos"][:])
            demo.append(np.full(len(cube[-1]), i))
    return np.concatenate(rgb), np.concatenate(cube), np.concatenate(goal), np.concatenate(demo)


@torch.no_grad()
def encode(enc, rgb):
    x = torch.from_numpy(rgb).permute(0, 3, 1, 2)
    return torch.cat([enc(b.cuda().float() / 255.0) for b in x.split(512)]).cpu().numpy().astype(np.float64)


def ridge(Z, Y, alpha):
    # Z (N, d) centred, Y (N, k) centred -> W (d, k)
    return np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ Y)


def r2(Y, P):
    return 1 - ((Y - P) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)  # per coordinate


def probe(Zf, Yf, groups, Zt, Yt):
    # alpha by 5-fold CV over demos (mean R^2 over coordinates), then refit on all fit frames
    folds = np.array_split(np.unique(groups), 5)
    cv = []
    for a in ALPHAS:
        s = []
        for fold in folds:
            te = np.isin(groups, fold)
            mz, my = Zf[~te].mean(0), Yf[~te].mean(0)
            W = ridge(Zf[~te] - mz, Yf[~te] - my, a)
            s.append(r2(Yf[te], (Zf[te] - mz) @ W + my).mean())
        cv.append(np.mean(s))
    a = ALPHAS[int(np.argmax(cv))]
    mz, my = Zf.mean(0), Yf.mean(0)
    W = ridge(Zf - mz, Yf - my, a)
    return a, (Zt - mz) @ W + my


if __name__ == "__main__":
    OUT = Path("results") / f"{date.today():%Y%m%d}_step1b_probe_cube_goal"
    OUT.mkdir(parents=True, exist_ok=True)
    mpath = OUT / "metrics.json"
    m = json.load(open(mpath)) if mpath.exists() else {}
    for run_dir, domain in zip(sys.argv[1::2], sys.argv[2::2]):
        ckpt = max(glob.glob(f"{run_dir}/runs/*/checkpoints/[0-9]*.pt"), key=lambda p: int(Path(p).stem))
        sd = torch.load(ckpt)["ema_agent"]
        enc = PlainConv(in_channels=3, out_dim=256, pool_feature_map=True).cuda().eval()
        enc.load_state_dict({k[len("visual_encoder."):]: v for k, v in sd.items() if k.startswith("visual_encoder.")})
        rgb_f, cube_f, goal_f, demo_f = load(H5[domain], FIT)
        rgb_t, cube_t, goal_t, _ = load(H5[domain], TEST)
        Zf, Zt = encode(enc, rgb_f), encode(enc, rgb_t)
        res = dict(checkpoint=ckpt, domain=domain, n_fit=len(Zf), n_test=len(Zt))
        table = cube_t[:, 2] < ON_TABLE_Z
        for name, Yf, Yt in (("cube", cube_f, cube_t), ("goal", goal_f, goal_t)):
            a, P = probe(Zf, Yf, demo_f, Zt, Yt)
            res[name] = dict(alpha=a, r2_xyz_all=r2(Yt, P).round(3).tolist(),
                             r2_xyz_on_table=r2(Yt[table], P[table]).round(3).tolist())
        print(Path(run_dir).name, domain, json.dumps({k: res[k] for k in ("cube", "goal")}), flush=True)
        m[f"{Path(run_dir).name}:{domain}"] = res
        json.dump(m, open(mpath, "w"), indent=1)
    json.dump(dict(fit_demos=[FIT.start, FIT.stop], test_demos=[TEST.start, TEST.stop], alphas=ALPHAS,
                   on_table_z=ON_TABLE_Z, h5=H5), open(OUT / "config.json", "w"), indent=1)
