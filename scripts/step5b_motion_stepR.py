"""Step 5b follow-up: re-run the Step R combinations (goal shift, embodiment) with motion pairs instead of action pairs.

Usage: uv run python scripts/step5b_motion_stepR.py <goal|emb> <pair> <encoder_run_dir> <controller_run_dir> [--episodes=250]
Run from the repo root with PYTHONPATH=<repo>:<repo>/third_party/maniskill_diffusion_policy.

Step R protocol (scripts/step1b_stitch.py, --only-dir=10): look 2 encoder -> cam0-side controller, deployed in look 2
on the controller's task / robot. Frames are the current frames t = 0..L-2 of each demo.
  - label fit (headline, as Step R): source = deployment-domain demos 0-49 (goal look 2 / xArm look 2), target =
    controller-domain demos 50-99 (goal cam0 / xArm cam0);
  - label fit (secondary, "own"): source = the encoder's own training demos 0-49 (look 2 default task, Panda), as in
    the Step 5b pilot;
  - paired ceiling: affine on deployment vs controller domain, demos 0-99 (same demo and step = same state);
  - held-out z residual on the same paired sets, demos 400-497.
Maps: affine (ceiling); nn_affine on action chunks (Step R's pairing: z-scored 8-step chunks, Step 1b label stats)
and on motion chunks (8 per-step TCP position changes, z-scored per dimension with stats pooled over both sides; see
scripts/step5b_stitch.py); embodiment also nn_orth on both. The same with "_own" for the secondary source.
Closed loop: the baseline's evaluate() on the deployment env, N_EPISODES episodes, final checkpoints.
"""
import copy
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step5b_stitch as s5
from step1b_stitch import ENV_KWARGS, affine, encode, load_agent
from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from stitch.align import fit_affine_paired, fit_procrustes_paired

TRAJ = "trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
R = "/home/ricc/projects/labelstitch-r/results"
s5.H5.update({"look2": f"{R}/20261002_dp_ours_demos_default_panda_cam0_look2_lollipop/{TRAJ}",
              "goal_look2": f"{R}/20261003_dp_ours_demos_goal_panda_cam0_look2_lollipop/{TRAJ}",
              "goal_cam0": f"{R}/20261003_dp_ours_demos_goal_panda_cam0_lollipop/{TRAJ}",
              "xarm_look2": f"{R}/20261003_dp_ours_demos_default_xarm6_cam0_look2_lollipop/{TRAJ}",
              "xarm_cam0": f"{R}/20261003_dp_ours_demos_default_xarm6_cam0_lollipop/{TRAJ}"})
COMBOS = {"goal": ("goal_look2", "goal_cam0", "stitch.envs:StitchPickCubeLollipopNoGraspLook2Goal-v1"),
          "emb": ("xarm_look2", "xarm_cam0", "stitch.envs:StitchPickCubeLollipopNoGraspLook2Xarm-v1")}
SRC_DEMOS, TGT_DEMOS, PAIRED_DEMOS = range(0, 50), range(50, 100), range(0, 100)
HELDOUT_DEMOS = range(400, 496)  # the goal-variant demo set has 496 demos
N_EPISODES = 250
DEV = "cuda"

if __name__ == "__main__":
    COMBO, PAIR, ENC_RUN, CTRL_RUN = sys.argv[1:5]
    N_EPISODES = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--episodes=")), N_EPISODES))
    DEP, CTL, DEPLOY_ENV = COMBOS[COMBO]
    OUT = Path("results") / f"{date.today():%Y%m%d}_step5b_motion_stepR_{COMBO}_p{PAIR}"
    OUT.mkdir(parents=True, exist_ok=True)
    enc, enc_ckpt = load_agent(ENC_RUN)
    ctrl, ctrl_ckpt = load_agent(CTRL_RUN)

    src = {"": s5.load(DEP, SRC_DEMOS), "_own": s5.load("look2", SRC_DEMOS)}
    tgt = s5.load(CTL, TGT_DEMOS)
    pu, pv = s5.load(DEP, PAIRED_DEMOS), s5.load(CTL, PAIRED_DEMOS)
    hu, hv = s5.load(DEP, HELDOUT_DEMOS), s5.load(CTL, HELDOUT_DEMOS)
    for a, b in ((pu, pv), (hu, hv)):  # same demo and step -> same state in both renderings
        assert np.abs(a["tcp"] - b["tcp"]).max() < 1e-5
    Zt = encode(ctrl.visual_encoder, tgt["rgb"]).cpu().numpy()
    Pu, Pv = encode(enc.visual_encoder, pu["rgb"]).cpu().numpy(), encode(ctrl.visual_encoder, pv["rgb"]).cpu().numpy()
    Hu, Hv = encode(enc.visual_encoder, hu["rgb"]), encode(ctrl.visual_encoder, hv["rgb"])

    maps, pairs = {"affine": fit_affine_paired(Pu, Pv)}, {}
    fits = {"nn_affine": fit_affine_paired, **({"nn_orth": fit_procrustes_paired} if COMBO == "emb" else {})}
    for tag, S in src.items():
        Zs = encode(enc.visual_encoder, S["rgb"]).cpu().numpy()
        for kind, mode in (("action", "shared"), ("motion", "pooled")):
            j = s5.nearest(*s5.zscore(S["cmd" if kind == "action" else "motion"], tgt["cmd" if kind == "action" else "motion"], mode))
            pairs[f"{kind}{tag}"] = dict(motion_err_mm=float(np.linalg.norm((S["motion"] - tgt["motion"][j]).reshape(len(j), -1), axis=1).mean() * 1e3))
            for fname, fit in fits.items():
                maps[f"{fname}_{kind}{tag}"] = fit(Zs, Zt[j])
    checks = {f"motion_std_mm{tag}": dict(source=(S["motion"].reshape(-1, 3).std(0) * 1e3).tolist(),
                                          target=(tgt["motion"].reshape(-1, 3).std(0) * 1e3).tolist()) for tag, S in src.items()}
    print(json.dumps(dict(checks=checks, pairs=pairs), indent=1), flush=True)
    np.savez(OUT / "maps.npz", **{f"{n}_{k}": v for n, (Rm, b) in maps.items() for k, v in (("R", Rm), ("b", b))})

    m = dict(checkpoints=dict(encoder=enc_ckpt, controller=ctrl_ckpt), checks=checks, pairs=pairs, aligners={})
    envs = make_eval_envs(DEPLOY_ENV, 10, "physx_cpu", ENV_KWARGS, dict(obs_horizon=2), video_dir=None,
                          wrappers=[FlattenRGBDObservationWrapper])
    for name, (Rm, b) in maps.items():
        T = affine(Rm, b)
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
    json.dump(dict(combo=COMBO, pair=PAIR, encoder_run=ENC_RUN, controller_run=CTRL_RUN, h5={k: s5.H5[k] for k in (DEP, CTL, "look2")},
                   labels=s5.LABELS, deploy_env=DEPLOY_ENV, src_demos=[SRC_DEMOS.start, SRC_DEMOS.stop],
                   tgt_demos=[TGT_DEMOS.start, TGT_DEMOS.stop], paired_demos=[PAIRED_DEMOS.start, PAIRED_DEMOS.stop],
                   heldout_demos=[HELDOUT_DEMOS.start, HELDOUT_DEMOS.stop], n_episodes=N_EPISODES), open(OUT / "config.json", "w"), indent=1)
