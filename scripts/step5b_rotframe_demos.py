"""Step 5b: demos for the command-convention shift (rotated EE-delta command frame), checked by replay.

Usage: uv run python scripts/step5b_rotframe_demos.py

Commands c = Rz(-45 deg) a on the x, y dims of the default pd_ee_delta_pos demos (z and gripper unchanged): the env's
rotframe task turns c back into a (stitch/envs.py, ROT_FRAME), so the robot moves exactly as in the default demos.
Check: replay every demo's commands open-loop from its seed in the rotframe env (state only) and compare the env
state before each step with the stored default state; record success at the end.
Writes results/<date>_step5b_demos_rotframe_panda_pos/demos.npz (same keys as the default demos, task = rotframe).
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stitch.envs import ROT_FRAME, make_env

DEMOS = "/home/ricc/projects/labelstitch-step1/results/20260930_step1_demos_default_panda_pos/demos.npz"
OUT = Path("results") / f"{date.today():%Y%m%d}_step5b_demos_rotframe_panda_pos"
OUT.mkdir(parents=True, exist_ok=True)

D = dict(np.load(DEMOS))
c = D["action"].copy()
c[:, :2] = D["action"][:, :2] @ ROT_FRAME  # rows: c = R^T a  (R^T = Rz(-45 deg))
D["action"], D["task"] = c, np.full(len(c), "rotframe")

env = make_env(task="rotframe", obs_mode="state", control_mode="pd_ee_delta_pos")
max_diff, failed = [], []
for e in np.unique(D["episode"]):
    m = D["episode"] == e
    env.reset(seed=int(D["seed"][m][0]))
    diff = 0.0
    for a, s in zip(D["action"][m], D["state"][m]):
        diff = max(diff, float(np.abs(env.unwrapped.get_state()[0].cpu().numpy() - s).max()))
        _, _, _, _, info = env.step(a)
    max_diff.append(diff)
    if not bool(info["success"].item()):
        failed.append(int(D["seed"][m][0]))
np.savez_compressed(OUT / "demos.npz", **D)
res = dict(demos=len(max_diff), replay_failed=failed, max_state_diff=float(np.max(max_diff)),
           median_max_state_diff=float(np.median(max_diff)), max_abs_command=np.abs(c).max(0).tolist())
print(json.dumps(res, indent=1))
json.dump(dict(source=DEMOS, rot_deg=45, control_mode="pd_ee_delta_pos"), open(OUT / "config.json", "w"), indent=1)
json.dump(res, open(OUT / "metrics.json", "w"), indent=1)
