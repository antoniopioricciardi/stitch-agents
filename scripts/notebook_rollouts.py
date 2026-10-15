"""Illustrative closed-loop clips for notebooks/stitching_explained.ipynb (section 8). Not a measurement: 10 episodes.

Usage: uv run python scripts/notebook_rollouts.py
Run from the repo root with PYTHONPATH=<repo>:<repo>/third_party/maniskill_diffusion_policy.

Stitch = matrix pair 1, forward direction: cam0 s1 encoder -> look 2 s2 controller, played in cam0 (as in
scripts/step5b_motion_matrix.py). Maps: identity, and nn_affine on motion pairs (source = cam0 demos 0-49, target =
look 2 demos 50-99, each agent's own data). One CPU env (only env 0 records video in the baseline), N_EPISODES episodes
per map. Initial states are random and differ between maps (evaluate() resets unseeded).
Writes results/<date>_notebook/rollouts/<map>/<episode>.mp4 and rollouts/metrics.json (per-episode success).
"""
import copy
import json
import sys
from datetime import date
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step5b_stitch as s5
from step1b_stitch import ENV, ENV_KWARGS, affine, encode, load_agent
from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from stitch.align import fit_affine_paired, fit_identity

TRAJ = "trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
s5.H5.update({"cam0": f"/home/ricc/projects/labelstitch-step1/results/20260930_dp_ours_demos_default_panda_cam0_lollipop/{TRAJ}",
              "look2": f"/home/ricc/projects/labelstitch-1b/results/20261002_dp_ours_demos_default_panda_cam0_look2_lollipop/{TRAJ}"})
ENC_RUN = "/home/ricc/projects/labelstitch-step1/results/20261001_step2g_dp_core_nograsp_50k_s1"  # cam0 s1
CTRL_RUN = "/home/ricc/projects/labelstitch-1b/results/20261002_step1b_dp_ref_look2_s2"  # look 2 s2
N_EPISODES = 10
DEV = "cuda"

if __name__ == "__main__":
    OUT = Path("results") / f"{date.today():%Y%m%d}_notebook" / "rollouts"
    OUT.mkdir(parents=True, exist_ok=True)
    (enc, enc_ckpt), (ctrl, ctrl_ckpt) = load_agent(ENC_RUN), load_agent(CTRL_RUN)
    src, tgt = s5.load("cam0", range(0, 50)), s5.load("look2", range(50, 100))
    Zs, Zt = encode(enc.visual_encoder, src["rgb"]).cpu().numpy(), encode(ctrl.visual_encoder, tgt["rgb"]).cpu().numpy()
    j = s5.nearest(*s5.zscore(src["motion"], tgt["motion"], "pooled"))
    maps = {"identity": fit_identity(Zs, Zt), "nn_affine": fit_affine_paired(Zs, Zt[j])}

    m = dict(checkpoints=dict(encoder=enc_ckpt, controller=ctrl_ckpt), env=ENV["cam0"], n_episodes=N_EPISODES, maps={})
    for name, (R, b) in maps.items():
        envs = make_eval_envs(ENV["cam0"], 1, "physx_cpu", ENV_KWARGS, dict(obs_horizon=2), video_dir=str(OUT / name),
                              wrappers=[FlattenRGBDObservationWrapper])
        stitched = copy.deepcopy(ctrl)
        stitched.visual_encoder = nn.Sequential(copy.deepcopy(enc.visual_encoder), affine(R, b))
        ev = evaluate(N_EPISODES, stitched, envs, DEV, "physx_cpu", progress_bar=False)
        envs.close()
        m["maps"][name] = dict(success_once=ev["success_once"].astype(bool).ravel().tolist(),
                               success_at_end=ev["success_at_end"].astype(bool).ravel().tolist())
        print(name, m["maps"][name], flush=True)
    json.dump(m, open(OUT / "metrics.json", "w"), indent=1)
