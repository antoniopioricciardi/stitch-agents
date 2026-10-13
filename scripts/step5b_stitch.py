"""Step 5b: command-convention shift. Stitch with pairs from commanded actions vs from the robot's motion.

Usage: uv run python scripts/step5b_stitch.py <row-name> <encoder_run_dir> <controller_run_dir> [--episodes=250]
Run from the repo root with PYTHONPATH=<repo>:<repo>/third_party/maniskill_diffusion_policy.

Stitch = look 2 default-task encoder E_u -> cam0 rotated-frame controller C_v, deployed on the rotated-frame variant in
look 2 (StitchPickCubeLollipopNoGraspLook2Rotframe-v1). The rotframe agents command the EE delta in a frame rotated
45 deg about z (stitch/envs.py); the robot's motion is the same as in the default task.
Frame sets (current frames t = 0..L-2 of each demo, as in training; z = the agent's PlainConv feature):
  - label fit, source: look 2 DEFAULT demos 0-49 (the encoder's own training data: default commands);
  - label fit, target: cam0 rotframe demos 50-99 (the controller's own training data: rotated commands);
  - paired ceiling: look 2 rotframe vs cam0 rotframe demos 0-99 (same demo and step = same state);
  - held-out z residual: the same paired sets on demos 400-497.
Maps (all (R, b), z_mapped = z @ R.T + b): identity, affine (paired ceiling), and nn_affine (each source frame paired
with the target frame whose standardised chunk is nearest, then affine least squares) on four chunk types:
  cmd_shared: 8-step command chunks, z-scored with the Step 1b label stats (default cam0 actions) on both sides;
  cmd_side:   8-step command chunks, z-scored per side (each side's own per-dimension mean / std over its fit frames);
  motion:     8 per-step TCP position changes p[t+k+1] - p[t+k], k = 0..7 (from obs/extra/tcp_pose, T+1 entries;
              zero past the episode end, like the action padding), z-scored per dimension with stats pooled over both
              sides' fit frames;
  motion_grip: motion + the finger opening (qpos[7] + qpos[8]) at t+1..t+8 (last value past the end), same z-score.
Checks that motion is the same quantity on both sides: per-side mean / std of the motion per dimension, and the max
motion-chunk difference between the same demo (same seed) in look 2 default and cam0 rotframe (should be ~0).
Pair quality per pairing: mean L2 between the true motion chunks of paired frames (mm) and mean TCP distance (mm).
Closed loop: the baseline's evaluate() on the deployment env, N_EPISODES episodes, final checkpoints.
"""
import copy
import json
import sys
from datetime import date
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step1b_stitch import ENV_KWARGS, affine, encode, load_agent
from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from stitch.align import fit_affine_paired, fit_identity
from stitch.labels import CHUNK, frame_chunks

TRAJ = "trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
H5 = {"look2": f"/home/ricc/projects/labelstitch-1b/results/20261002_dp_ours_demos_default_panda_cam0_look2_lollipop/{TRAJ}",
      "rot_look2": f"results/20261008_dp_ours_demos_rotframe_panda_cam0_look2_lollipop/{TRAJ}",
      "rot_cam0": f"results/20261008_dp_ours_demos_rotframe_panda_cam0_lollipop/{TRAJ}"}
LABELS = "/home/ricc/projects/labelstitch-1b/results/20261001_step1b_labels/labels.pt"
DEPLOY_ENV = "stitch.envs:StitchPickCubeLollipopNoGraspLook2Rotframe-v1"
SRC_DEMOS, TGT_DEMOS, PAIRED_DEMOS, HELDOUT_DEMOS = range(0, 50), range(50, 100), range(0, 100), range(400, 498)
N_EPISODES = 250
DEV = "cuda"


def load(name, demos):
    # -> dict: rgb (N, 3, H, W) uint8 current frames, cmd (N, 8, 4) command chunks, motion (N, 8, 3) per-step TCP
    # changes, grip (N, 8) finger opening at t+1..t+8, tcp (N, 3), demo (N,); one row per frame t = 0..L-2
    out = {k: [] for k in ("rgb", "cmd", "motion", "grip", "tcp", "demo")}
    with h5py.File(H5[name], "r") as f:
        for i in demos:
            g = f[f"traj_{i}"]
            a = g["actions"][:]
            L = len(a)
            p = g["obs/extra/tcp_pose"][:, :3]  # (L+1, 3)
            w = g["obs/agent/qpos"][:, 7:9].sum(1)  # (L+1,) finger opening
            dp = np.concatenate([np.diff(p, axis=0), np.zeros((CHUNK, 3), p.dtype)])  # (L+8, 3): dp[s] = p[s+1] - p[s]
            w = np.concatenate([w, np.full(CHUNK, w[-1])])
            t = np.arange(L - 1)
            out["rgb"].append(g["obs/sensor_data/base_camera/rgb"][: L - 1].transpose(0, 3, 1, 2))
            out["cmd"].append(frame_chunks(a))
            out["motion"].append(np.stack([dp[s:s + CHUNK] for s in t]))
            out["grip"].append(np.stack([w[s + 1:s + 1 + CHUNK] for s in t]))
            out["tcp"].append(p[t])
            out["demo"].append(np.full(L - 1, i))
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["rgb"] = torch.from_numpy(out["rgb"])
    return out


def zscore(Xs, Xt, mode):
    # Xs (Ns, 8, A), Xt (Nt, 8, A) -> flattened, standardised per dimension; mode: "shared" (Step 1b label stats),
    # "side" (each side its own stats), "pooled" (stats over both sides together)
    if mode == "shared":
        lab = torch.load(LABELS)
        ms = mt = lab["mean"].numpy()
        ss = st = lab["std"].numpy()
    elif mode == "side":
        ms, ss = Xs.reshape(-1, Xs.shape[-1]).mean(0), Xs.reshape(-1, Xs.shape[-1]).std(0)
        mt, st = Xt.reshape(-1, Xt.shape[-1]).mean(0), Xt.reshape(-1, Xt.shape[-1]).std(0)
    else:
        X = np.concatenate([Xs, Xt]).reshape(-1, Xs.shape[-1])
        ms = mt = X.mean(0)
        ss = st = X.std(0)
    return ((Xs - ms) / ss).reshape(len(Xs), -1), ((Xt - mt) / st).reshape(len(Xt), -1)


def nearest(Xs, Xt):
    # (Ns, D), (Nt, D) -> (Ns,) index of the nearest target row for every source row
    Xs, Xt = torch.tensor(Xs, device=DEV, dtype=torch.float32), torch.tensor(Xt, device=DEV, dtype=torch.float32)
    return torch.cat([torch.cdist(x, Xt).argmin(1) for x in Xs.split(2048)]).cpu().numpy()


if __name__ == "__main__":
    ROW, ENC_RUN, CTRL_RUN = sys.argv[1], sys.argv[2], sys.argv[3]
    N_EPISODES = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--episodes=")), N_EPISODES))
    OUT = Path("results") / f"{date.today():%Y%m%d}_step5b_stitch_{ROW}"
    OUT.mkdir(parents=True, exist_ok=True)
    enc, enc_ckpt = load_agent(ENC_RUN)
    ctrl, ctrl_ckpt = load_agent(CTRL_RUN)

    src, tgt = load("look2", SRC_DEMOS), load("rot_cam0", TGT_DEMOS)
    pu, pv = load("rot_look2", PAIRED_DEMOS), load("rot_cam0", PAIRED_DEMOS)
    hu, hv = load("rot_look2", HELDOUT_DEMOS), load("rot_cam0", HELDOUT_DEMOS)
    for a, b in ((pu, pv), (hu, hv)):  # same demo and step -> same state in both renderings
        assert np.abs(a["tcp"] - b["tcp"]).max() < 1e-5
    Zs, Zt = encode(enc.visual_encoder, src["rgb"]).cpu().numpy(), encode(ctrl.visual_encoder, tgt["rgb"]).cpu().numpy()
    Pu, Pv = encode(enc.visual_encoder, pu["rgb"]).cpu().numpy(), encode(ctrl.visual_encoder, pv["rgb"]).cpu().numpy()
    Hu, Hv = encode(enc.visual_encoder, hu["rgb"]), encode(ctrl.visual_encoder, hv["rgb"])

    # same-quantity checks: motion statistics per side; the same demo (seed) in look 2 default vs cam0 rotframe
    same = load("look2", range(50, 60))
    same_t = load("rot_cam0", range(50, 60))
    checks = dict(
        motion_mean_mm=dict(source=(src["motion"].reshape(-1, 3).mean(0) * 1e3).tolist(), target=(tgt["motion"].reshape(-1, 3).mean(0) * 1e3).tolist()),
        motion_std_mm=dict(source=(src["motion"].reshape(-1, 3).std(0) * 1e3).tolist(), target=(tgt["motion"].reshape(-1, 3).std(0) * 1e3).tolist()),
        same_seed_motion_maxdiff_mm=float(np.abs(same["motion"] - same_t["motion"]).max() * 1e3),
        same_seed_cmd_maxdiff=float(np.abs(same["cmd"] - same_t["cmd"]).max()))
    print(json.dumps(checks, indent=1), flush=True)

    chunks = {"cmd_shared": zscore(src["cmd"], tgt["cmd"], "shared"), "cmd_side": zscore(src["cmd"], tgt["cmd"], "side"),
              "motion": zscore(src["motion"], tgt["motion"], "pooled"),
              "motion_grip": zscore(np.concatenate([src["motion"], src["grip"][..., None]], -1),
                                    np.concatenate([tgt["motion"], tgt["grip"][..., None]], -1), "pooled")}
    maps, pairs = {"identity": fit_identity(Zs, Zt), "affine": fit_affine_paired(Pu, Pv)}, {}
    for name, (Xs, Xt) in chunks.items():
        j = nearest(Xs, Xt)
        maps[f"nn_affine_{name}"] = fit_affine_paired(Zs, Zt[j])
        pairs[name] = dict(motion_err_mm=float(np.linalg.norm((src["motion"] - tgt["motion"][j]).reshape(len(j), -1), axis=1).mean() * 1e3),
                           tcp_dist_mm=float(np.linalg.norm(src["tcp"] - tgt["tcp"][j], axis=1).mean() * 1e3))
    print(json.dumps(pairs, indent=1), flush=True)
    np.savez(OUT / "maps.npz", **{f"{n}_{k}": v for n, (R, b) in maps.items() for k, v in (("R", R), ("b", b))})

    m = dict(checkpoints=dict(encoder=enc_ckpt, controller=ctrl_ckpt), checks=checks, pairs=pairs, aligners={})
    envs = make_eval_envs(DEPLOY_ENV, 10, "physx_cpu", ENV_KWARGS, dict(obs_horizon=2), video_dir=None,
                          wrappers=[FlattenRGBDObservationWrapper])
    for name, (R, b) in maps.items():
        T = affine(R, b)
        with torch.no_grad():
            resid = (((T(Hu) - Hv) ** 2).sum() / ((Hv - Hv.mean(0)) ** 2).sum()).item()
        stitched = copy.deepcopy(ctrl)
        stitched.visual_encoder = nn.Sequential(copy.deepcopy(enc.visual_encoder), T)
        ev = evaluate(N_EPISODES, stitched, envs, DEV, "physx_cpu", progress_bar=False)
        m["aligners"][name] = dict(z_residual=resid, success_once=float(ev["success_once"].mean()),
                                   success_at_end=float(ev["success_at_end"].mean()), episodes=len(ev["success_once"]))
        print(name, m["aligners"][name], flush=True)
        json.dump(m, open(OUT / "metrics.json", "w"), indent=1)
        del stitched
    envs.close()
    json.dump(dict(row=ROW, encoder_run=ENC_RUN, controller_run=CTRL_RUN, h5=H5, labels=LABELS, deploy_env=DEPLOY_ENV,
                   src_demos=[SRC_DEMOS.start, SRC_DEMOS.stop], tgt_demos=[TGT_DEMOS.start, TGT_DEMOS.stop],
                   paired_demos=[PAIRED_DEMOS.start, PAIRED_DEMOS.stop], heldout_demos=[HELDOUT_DEMOS.start, HELDOUT_DEMOS.stop],
                   n_episodes=N_EPISODES), open(OUT / "config.json", "w"), indent=1)
