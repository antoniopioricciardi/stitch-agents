"""Glue for ManiSkill's Diffusion Policy baseline: export our pd_ee_delta_pos demos in ManiSkill's trajectory format.

Usage: uv run python scripts/export_dp_demos.py [goal_marker] [cam] [demos.npz] [look] [task]   (sphere = default, hidden, ...; see stitch.envs;
       cam, look = 0 and task = default by default; the demos.npz must come from the same task; demos.npz defaults to results/20260930_step1_demos_default_panda_pos/demos.npz)

For each stored demo (results/20260930_step1_demos_default_panda_pos/demos.npz, seeds 0..499): reset our env
(StitchPickCube-v1: camera `cam`, look `look`, default light/task, Panda, obs_mode rgb) with the demo's seed — this reproduces the
demo's initial state exactly — and replay the stored actions open-loop. Demos whose replay succeeds are recorded with
ManiSkill's RecordEpisode wrapper, i.e. exactly the files mani_skill.trajectory.replay_trajectory writes
(trajectory.h5 with obs of length T+1 and actions of length T, plus trajectory.json with env_info), so the baseline's
loader reads them unchanged. Replays that fail are skipped (checked first on a state-only env).
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stitch.envs import make_env
from mani_skill.utils.wrappers import RecordEpisode

DEMOS = Path(sys.argv[3] if len(sys.argv) > 3 else "results/20260930_step1_demos_default_panda_pos/demos.npz")
CONTROL = "pd_ee_delta_pos"
MARKER = sys.argv[1] if len(sys.argv) > 1 else "sphere"
CAM = int(sys.argv[2]) if len(sys.argv) > 2 else 0
LOOK = int(sys.argv[4]) if len(sys.argv) > 4 else 0
TASK = sys.argv[5] if len(sys.argv) > 5 else "default"
OUT = Path("results") / (f"{date.today():%Y%m%d}_dp_ours_demos_{TASK}_panda_cam{CAM}" + (f"_look{LOOK}" if LOOK else "") + ("" if MARKER == "sphere" else f"_{MARKER}"))
OUT.mkdir(parents=True, exist_ok=True)

D = np.load(DEMOS)
check_env = make_env(task=TASK, obs_mode="state", control_mode=CONTROL)
rec_env = RecordEpisode(make_env((CAM, LOOK, 0), task=TASK, obs_mode="rgb", control_mode=CONTROL, goal_marker=MARKER), output_dir=str(OUT),
                        trajectory_name=f"trajectory.rgb.{CONTROL}.physx_cpu", save_video=False, save_on_reset=False,
                        source_type="motionplanning", source_desc="stitch mplib demos converted to pd_ee_delta_pos")
kept, skipped = [], []
for e in np.unique(D["episode"]):
    m = D["episode"] == e
    seed, actions = int(D["seed"][m][0]), D["action"][m]
    check_env.reset(seed=seed)
    for a in actions:
        _, _, _, _, info = check_env.step(a)
    if not bool(info["success"].item()):
        skipped.append(seed)
        continue
    rec_env.reset(seed=seed)
    for a in actions:
        rec_env.step(a)
    rec_env.flush_trajectory()
    kept.append(seed)
rec_env.close()
print(f"kept {len(kept)} demos, skipped {len(skipped)} whose open-loop replay failed: {skipped}")
json.dump(dict(demos=str(DEMOS), control_mode=CONTROL, goal_marker=MARKER, env=f"StitchPickCube-v1 cam{CAM} look{LOOK} {TASK} panda"), open(OUT / "config.json", "w"), indent=1)
json.dump(dict(kept=len(kept), skipped=skipped, kept_seeds=kept), open(OUT / "metrics.json", "w"), indent=1)
