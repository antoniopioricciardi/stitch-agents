"""Step 5b follow-up: the Step 1b main visual-shift matrix with nn_affine on motion pairs (own data on each side).

Usage: uv run python scripts/step5b_motion_matrix.py <shift> <pair> <cam0_run_dir> <shift_run_dir> [--episodes=250]
       [--maps=nn_affine,nn_orth,nn_sim,nn_rescale] (default nn_affine) [--tag=x] (suffix for the output folder)
Run from the repo root with PYTHONPATH=<repo>:<repo>/third_party/maniskill_diffusion_policy.

Same agents, directions and label halves as the Step 1b matrix (scripts/step1b_matrix_table.py): fwd = cam0 encoder ->
shift controller, played in cam0; bwd = shift encoder -> cam0 controller, played in the shift domain. Label fit:
source = the encoder's own demos 0-49 (its domain), target = the controller's own demos 50-99 (its domain); each source
frame paired with the target frame whose motion chunk is nearest (8 per-step TCP position changes, z-scored with pooled
stats; scripts/step5b_stitch.py), then affine least squares (nn_affine), orthogonal Procrustes (nn_orth) or a
similarity map (nn_sim: orthogonal x one global scale + bias), or nn_rescale (nn_affine, then each mapped
dimension given the mean and std of the matched target frames; step1b_stitch.rescale). Identity, the ceiling and the action-pair maps come from
the Step 1b runs.
"""
import copy
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step5b_stitch as s5
from step1b_stitch import ENV, ENV_KWARGS, affine, encode, load_agent, rescale
from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from stitch.align import fit_affine_paired, fit_procrustes_paired, fit_similarity_paired

FITS = {"nn_affine": fit_affine_paired, "nn_orth": fit_procrustes_paired, "nn_sim": fit_similarity_paired,
        "nn_rescale": lambda Zs, Zt: rescale(*fit_affine_paired(Zs, Zt), Zs, Zt)}  # Zt = the matched target frames

TRAJ = "trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
B = "/home/ricc/projects/labelstitch-1b/results"
s5.H5.update({"cam0": f"/home/ricc/projects/labelstitch-step1/results/20260930_dp_ours_demos_default_panda_cam0_lollipop/{TRAJ}",
              "look1": f"{B}/20261001_dp_ours_demos_default_panda_cam0_look1_lollipop/{TRAJ}",
              "look2": f"{B}/20261002_dp_ours_demos_default_panda_cam0_look2_lollipop/{TRAJ}",
              "look2light1": f"{B}/20261002_dp_ours_demos_default_panda_cam0_look2_light1_lollipop/{TRAJ}",
              "cam3": f"{B}/20261002_dp_ours_demos_default_panda_cam3_lollipop/{TRAJ}"})
SRC_DEMOS, TGT_DEMOS = range(0, 50), range(50, 100)
N_EPISODES = 250
DEV = "cuda"

if __name__ == "__main__":
    SHIFT, PAIR, CAM0_RUN, SHIFT_RUN = sys.argv[1:5]
    opt = lambda k, d: next((a.split("=")[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
    N_EPISODES = int(opt("episodes", N_EPISODES))
    MAPS = opt("maps", "nn_affine").split(",")
    OUT = Path("results") / f"{date.today():%Y%m%d}_step5b_motion_matrix_{SHIFT}_{PAIR}{opt('tag', '')}"
    OUT.mkdir(parents=True, exist_ok=True)
    agents = {"cam0": load_agent(CAM0_RUN), SHIFT: load_agent(SHIFT_RUN)}
    m = dict(checkpoints={d: a[1] for d, a in agents.items()}, stitch={})
    for direction, u, v in (("fwd", "cam0", SHIFT), ("bwd", SHIFT, "cam0")):  # encoder domain u -> controller domain v, played in u
        src, tgt = s5.load(u, SRC_DEMOS), s5.load(v, TGT_DEMOS)
        enc, ctrl = agents[u][0], agents[v][0]
        Zs, Zt = encode(enc.visual_encoder, src["rgb"]).cpu().numpy(), encode(ctrl.visual_encoder, tgt["rgb"]).cpu().numpy()
        j = s5.nearest(*s5.zscore(src["motion"], tgt["motion"], "pooled"))
        err = float(np.linalg.norm((src["motion"] - tgt["motion"][j]).reshape(len(j), -1), axis=1).mean() * 1e3)
        envs = make_eval_envs(ENV[u], 10, "physx_cpu", ENV_KWARGS, dict(obs_horizon=2), video_dir=None,
                              wrappers=[FlattenRGBDObservationWrapper])
        m["stitch"][direction] = dict(encoder=u, controller=v)
        for name in MAPS:
            stitched = copy.deepcopy(ctrl)
            stitched.visual_encoder = nn.Sequential(copy.deepcopy(enc.visual_encoder), affine(*FITS[name](Zs, Zt[j])))
            ev = evaluate(N_EPISODES, stitched, envs, DEV, "physx_cpu", progress_bar=False)
            m["stitch"][direction][f"{name}_motion"] = dict(success_once=float(ev["success_once"].mean()), success_at_end=float(ev["success_at_end"].mean()),
                                                           episodes=len(ev["success_once"]), pair_motion_err_mm=err)
        envs.close()
        print(direction, m["stitch"][direction], flush=True)
        json.dump(m, open(OUT / "metrics.json", "w"), indent=1)
    json.dump(dict(shift=SHIFT, pair=PAIR, runs={"cam0": CAM0_RUN, SHIFT: SHIFT_RUN}, h5={d: s5.H5[d] for d in ("cam0", SHIFT)},
                   src_demos=[SRC_DEMOS.start, SRC_DEMOS.stop], tgt_demos=[TGT_DEMOS.start, TGT_DEMOS.stop], maps=MAPS, n_episodes=N_EPISODES),
              open(OUT / "config.json", "w"), indent=1)
