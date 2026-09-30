"""Step 2d: training-free comparison of two pd_ee_delta_pos demo sets (ManiSkill trajectory format), same seeds.

Usage: uv run python scripts/compare_demo_sets.py

Per demo set (first 100 shared seeds): length, fraction of rest steps (|position action| < 0.01), per phase
(before the gripper closes / after), mean |position action|, gripper close step; and same-seed differences in
actions, state (agent + extra) and frames (pixels that differ outside the green goal sphere).
"""
import json
import sys
from datetime import date
from pathlib import Path

import h5py
import numpy as np

SETS = {
    "theirs": Path.home() / ".maniskill/demos/PickCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5",
    "ours": Path("results/20260930_dp_ours_demos_default_panda_cam0/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"),
}
N = 100
REST = 0.01  # |position action| below this = rest step (normalised action units)
OUT = Path("results") / f"{date.today():%Y%m%d}_step2d_compare_demo_sets"
OUT.mkdir(parents=True, exist_ok=True)

files = {k: h5py.File(p, "r") for k, p in SETS.items()}
ids = {k: {e["episode_seed"]: e["episode_id"] for e in json.load(open(str(p)[:-2] + "json"))["episodes"]} for k, p in SETS.items()}
seeds = sorted(set(ids["theirs"]) & set(ids["ours"]))[:N]


def stats(actions):
    # actions (T, 4): xyz delta + gripper (+1 open, -1 closed)
    pos = np.linalg.norm(actions[:, :3], axis=1)
    close = int(np.argmax(actions[:, 3] < 0))
    before, after = slice(0, close), slice(close, len(actions))
    return dict(length=len(actions), close_step=close, rest_frac=float((pos < REST).mean()),
                rest_frac_before_close=float((pos[before] < REST).mean()), rest_frac_after_close=float((pos[after] < REST).mean()),
                mean_abs_pos_action=float(np.abs(actions[:, :3]).mean()))


m = {}
for k, f in files.items():
    rows = [stats(f[f"traj_{ids[k][s]}"]["actions"][:]) for s in seeds]
    m[k] = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
    print(k, {key: round(v, 4) for key, v in m[k].items()})

act_diff, state_diff, px_other = [], [], []
for s in seeds:
    a, b = files["theirs"][f"traj_{ids['theirs'][s]}"], files["ours"][f"traj_{ids['ours'][s]}"]
    act_diff.append(np.abs(a["actions"][:] - b["actions"][:]).max() if len(a["actions"]) == len(b["actions"]) else np.inf)
    state_diff.append(max(np.abs(a["obs"][g][key][:].astype(float) - b["obs"][g][key][:].astype(float)).max()
                          for g in ["agent", "extra"] for key in a["obs"][g].keys()))
    ra, rb = a["obs"]["sensor_data/base_camera/rgb"][:].astype(int), b["obs"]["sensor_data/base_camera/rgb"][:].astype(int)
    sphere = (rb[..., 1] > rb[..., 0] + 40) & (rb[..., 1] > rb[..., 2] + 40)  # green goal sphere, visible only in ours
    px_other.append(((np.abs(ra - rb).max(-1) > 10) & ~sphere).sum(axis=(1, 2)).mean())
m["same_seed"] = dict(seeds=len(seeds), same_length=int(np.isfinite(act_diff).sum()), max_action_diff=float(np.max(act_diff)),
                      median_action_diff=float(np.median(act_diff)), max_state_diff=float(np.max(state_diff)),
                      median_state_diff=float(np.median(state_diff)), mean_non_sphere_px_diff_per_frame=float(np.mean(px_other)))
print(m["same_seed"])
json.dump(dict(sets={k: str(v) for k, v in SETS.items()}, n=N, rest_threshold=REST), open(OUT / "config.json", "w"), indent=1)
json.dump(m, open(OUT / "metrics.json", "w"), indent=1)
