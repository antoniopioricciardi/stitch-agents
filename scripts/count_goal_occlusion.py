"""Step 2e: how often does the goal marker cover the cube or the gripper? (no training)

Usage: uv run python scripts/count_goal_occlusion.py [goal_marker]   (default: sphere)

For the first N demos (default task, Panda, pd_ee_delta_pos), restore every stored state in our env twice, with the
marker visible and hidden, render the segmentation, and count pixels that are cube/gripper when the marker is hidden
but marker when it is visible. Per camera: fraction of frames with any covered pixel, and mean fraction of the
cube's pixels covered; overall and within +-5 steps of the gripper starting to close.
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stitch.envs import CAMERAS, make_env

MARKER = sys.argv[1] if len(sys.argv) > 1 else "sphere"
N = 100
NEAR = 5
DEMOS = Path("results/20260930_step1_demos_default_panda_pos/demos.npz")
GRIPPER = ("panda_hand", "panda_leftfinger", "panda_rightfinger", "panda_leftfinger_pad", "panda_rightfinger_pad")
OUT = Path("results") / f"{date.today():%Y%m%d}_step2e_goal_occlusion_{MARKER}"
OUT.mkdir(parents=True, exist_ok=True)


def seg(env, state):
    env.unwrapped.set_state(torch.from_numpy(state[None]))
    return env.unwrapped.get_obs()["sensor_data"]["base_camera"]["segmentation"][0, ..., 0].cpu().numpy()  # (H, W) ids


def ids(env, names):
    return [i for i, o in env.unwrapped.segmentation_id_map.items() if o.name in names]


D = np.load(DEMOS)
EP, STATE, GRIP_ACTION = D["episode"], D["state"], D["action"][:, 3]  # load once: npz arrays decompress on every access
episodes = np.unique(EP)[:N]
m = {}
for c in range(len(CAMERAS)):
    vis = make_env((c, 0, 0), obs_mode="rgb+segmentation", control_mode="pd_ee_delta_pos", goal_marker=MARKER)
    hid = make_env((c, 0, 0), obs_mode="rgb+segmentation", control_mode="pd_ee_delta_pos", goal_marker="hidden")
    vis.reset(seed=0), hid.reset(seed=0)
    cube, grip, marker = ids(hid, ("cube",)), ids(hid, GRIPPER), ids(vis, ("goal_site", "goal_pole"))
    rows = []  # (near grasp, any covered, covered cube fraction)
    for e in episodes:
        idx = np.where(EP == e)[0]
        close = int(np.argmax(GRIP_ACTION[idx] < 0))  # first step with the gripper commanded closed
        for t, i in enumerate(idx):
            sh, sv = seg(hid, STATE[i]), seg(vis, STATE[i])
            target = np.isin(sh, cube + grip)
            covered = target & np.isin(sv, marker)
            n_cube = np.isin(sh, cube).sum()
            rows.append((abs(t - close) <= NEAR, covered.any(), (covered & np.isin(sh, cube)).sum() / max(n_cube, 1)))
    rows = np.array(rows, dtype=float)
    near = rows[:, 0] == 1
    m[f"cam{c}"] = dict(frames=len(rows), any_covered=rows[:, 1].mean(), any_covered_near_grasp=rows[near, 1].mean(),
                        cube_frac_covered=rows[:, 2].mean(), cube_frac_covered_near_grasp=rows[near, 2].mean())
    print(f"cam{c}", {k: round(float(v), 3) for k, v in m[f'cam{c}'].items()}, flush=True)
    vis.close(), hid.close()
json.dump(dict(marker=MARKER, n_demos=N, near_grasp_steps=NEAR, gripper_links=GRIPPER, demos=str(DEMOS)), open(OUT / "config.json", "w"), indent=1)
json.dump(m, open(OUT / "metrics.json", "w"), indent=1)
